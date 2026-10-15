"""Actual CPU-small generation and GPU-strong full-KV continuation over HTTP."""
from __future__ import annotations
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import threading
import time
import traceback

from meco.transport.frames import MAX_BODY, pack, unpack, digest
from meco.strategy.continuation import continuation_ids, FOLLOWUP
from meco.inference.models import SOURCE, SOURCE_ID, SOURCE_REV, TARGET, TARGET_ID, TARGET_REV, MAPPER_SHA256, validate_pair


def main():
    import torch
    import transformers
    from safetensors.torch import save, load
    from meco.kv.mapper import CacheBridgeMapper, pairs
    from meco.inference.cpu import HFCPUModel
    from meco.inference.common import token_hash
    from meco.inference.gpu import HFGPUModel
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--mapper")
    args = parser.parse_args()
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if digest((SOURCE / name).read_bytes()) != digest((TARGET / name).read_bytes()):
            raise ValueError("tokenizer identity differs: " + name)
    if args.tier == "cpu":
        model = HFCPUModel(source_model_path=str(SOURCE), model_id=SOURCE_ID, precision="float32", threads=4)
    else:
        model = HFGPUModel(source_model_path=str(TARGET), model_id=TARGET_ID, precision="float16", threads=4)
    if args.tier == "gpu" and not args.mapper:
        raise ValueError("GPU continuation requires the selected full-KV mapper")
    mapper = None
    if args.mapper:
        if args.tier != "gpu":
            raise ValueError("mapping belongs to GPU tier")
        if digest((Path(args.mapper) / "manifest.json").read_bytes()) != MAPPER_SHA256:
            raise ValueError("mapper differs from the selected new weights")
        mapper = CacheBridgeMapper(args.mapper, target_model=model.model, source_model_id=SOURCE_ID)
        validate_pair(mapper.manifest)
        if mapper.manifest.get("attention_query_extension") != "native_decision_query_weights_v1":
            raise ValueError("requires the selected new decision-query weights")
    inference_lock = threading.Lock()
    count = 0

    def infer(metadata, payload):
        nonlocal count
        began = time.perf_counter()
        if args.tier == "cpu":
            if payload:
                raise ValueError("CPU input must contain tokens only")
            result = model.generate(metadata["input_ids"], metadata["max_new_tokens"], return_cache=True)
            cache = result.pop("cache")
            followup = "Current observation: " + json.dumps(metadata["observation"], sort_keys=True) + ". " + FOLLOWUP
            prefix, suffix, full = continuation_ids(model.tokenizer, metadata["input_ids"],
                result["output_ids"], result["cache_tokens"], followup=followup)
            serialized = b""
            serialization_start = time.perf_counter()
            if metadata["transfer"]:
                serialized = save({f"{name}_{index}": tensor.detach().to(torch.float16).contiguous()
                    for index, pair in enumerate(pairs(cache)) for name, tensor in zip(("k", "v"), pair)})
            result.update(uid=metadata["uid"], source_model_id=SOURCE_ID, source_revision=SOURCE_REV,
                source_precision="float32", transport_dtype="float16", prefix_ids=prefix,
                suffix_ids=suffix, full_ids=full, prefix_sha256=token_hash(prefix),
                kv_sha256=digest(serialized), kv_bytes=len(serialized),
                serialization_ms=(time.perf_counter() - serialization_start) * 1000)
        else:
            if (metadata["source_model_id"], metadata["source_revision"]) != (SOURCE_ID, SOURCE_REV):
                raise ValueError("source identity mismatch")
            prefix, suffix, full = metadata["prefix_ids"], metadata["suffix_ids"], metadata["full_ids"]
            if full != prefix + suffix or token_hash(prefix) != metadata["prefix_sha256"]:
                raise ValueError("continuation token boundary mismatch")
            if digest(payload) != metadata["kv_sha256"] or len(payload) != metadata["kv_bytes"]:
                raise ValueError("payload digest/size mismatch")
            mapped = None
            torch.cuda.synchronize()
            mapping_start = time.perf_counter()
            if mapper is not None:
                if not payload:
                    raise ValueError("full KV transfer is required; fallback is forbidden")
                tensors = load(payload)
                source = [(tensors[f"k_{i}"], tensors[f"v_{i}"]) for i in range(mapper.manifest["source_layers"])]
                expected_shape = (1, 2, len(prefix), 128)
                if any(tuple(t.shape) != expected_shape or t.dtype != torch.float16
                       for pair in source for t in pair):
                    raise ValueError("source KV geometry mismatch")
                mapped = mapper.apply(source)
            torch.cuda.synchronize()
            mapping_ms = (time.perf_counter() - mapping_start) * 1000
            ids = suffix
            result = model.generate(ids, 96, cache=mapped, prefix_tokens=len(prefix))
            result.update(uid=metadata["uid"], mapping_ms=mapping_ms,
                target_input_tokens=len(ids), upstream_tokens_replayed=0,
                migrated_prefix_tokens=len(prefix),
                prefix_sha256=metadata["prefix_sha256"], kv_sha256=metadata["kv_sha256"],
                resources=model.resource_snapshot())
            serialized = b""
        result["service_ms"] = (time.perf_counter() - began) * 1000
        count += 1
        return pack(result, serialized)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send(self, status, body, content_type):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path != "/health":
                self.send(404, b"{}", "application/json")
                return
            value = {"tier": args.tier, "model_id": model.model_id, "architecture": model.arch,
                     "precision": model.precision, "requests": count,
                     "torch": torch.__version__, "transformers": transformers.__version__,
                     "mapper_sha256": digest((Path(args.mapper) / "manifest.json").read_bytes()) if mapper else None}
            if args.tier == "gpu":
                value["resources"] = model.resource_snapshot()
            self.send(200, json.dumps(value).encode(), "application/json")

        def do_POST(self):
            try:
                if self.path != "/generate":
                    raise ValueError("unknown endpoint")
                length = int(self.headers.get("Content-Length", "0"))
                if not 4 <= length <= MAX_BODY:
                    raise ValueError("request body exceeds bound")
                metadata, payload = unpack(self.rfile.read(length))
                with inference_lock:
                    response = infer(metadata, payload)
                self.send(200, response, "application/octet-stream")
            except Exception as error:
                traceback.print_exc()
                self.send(500, json.dumps({"type": type(error).__name__, "error": str(error)}).encode(), "application/json")

    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
