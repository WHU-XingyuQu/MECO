"""Head-local CacheBridge for pinned Qwen2.5-Coder small -> 7B pairs.

The affine/rotation equations follow the original KVTransfer reference under
CacheBridge reference implementation. No target prefix forward pass occurs during apply().
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import DynamicCache


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def pairs(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    return list(zip(cache.key_cache, cache.value_cache))


def rotate_half(value):
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def content_keys(keys, rotary, positions):
    cos, sin = rotary(keys[0], positions)
    cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
    return [(key.float() * cos - rotate_half(key.float()) * sin) /
            (cos.square() + sin.square()).clamp_min(1e-12) for key in keys]


class CacheBridgeMapper:
    def __init__(self, directory, *, target_model, source_model_id, device="cuda"):
        directory = Path(directory)
        self.manifest = json.loads((directory / "manifest.json").read_text())
        if self.manifest["source_model_id"] != source_model_id:
            raise ValueError("CacheBridge source model identity mismatch")
        config = target_model.config
        if (config.num_hidden_layers != self.manifest["target_layers"] or
                config.num_key_value_heads != self.manifest["target_heads"] or
                Path(config._name_or_path).name != self.manifest["target_revision"]):
            raise ValueError("CacheBridge target model identity mismatch")
        self.model, self.device = target_model, torch.device(device)
        self.layers = []
        for record in self.manifest["layers"]:
            path = directory / record["file"]
            if sha256(path) != record["sha256"]:
                raise ValueError("CacheBridge coefficient checksum mismatch")
            data = load_file(str(path), device=str(device))
            data["selected_list"] = data["selected"].cpu().tolist()
            self.layers.append(data)
        if len(self.layers) != config.num_hidden_layers:
            raise ValueError("CacheBridge layer count mismatch")

    @torch.inference_mode()
    def apply(self, source_pairs, *, source_rotary=None):
        if self.manifest.get("calibration_extension"):
            raise ValueError("calibrated artifact requires its explicit extension mapper")
        if self.manifest.get("source_head_mode", "single") != "single":
            raise ValueError("cross-head control requires its explicit diagnostic mapper")
        # This pair has identical default RoPE theta and head dimension. The
        # loader pins both identities; callers must separately verify token IDs.
        if len(source_pairs) != self.manifest["source_layers"]:
            raise ValueError("source cache depth mismatch")
        length = source_pairs[0][0].shape[2]
        shape = (1, self.manifest["source_heads"], length, self.manifest["head_dim"])
        if any(tuple(t.shape) != shape or not torch.isfinite(t).all()
               for pair in source_pairs for t in pair):
            raise ValueError("invalid source cache geometry or values")
        keys = [key.to(self.device) for key, _ in source_pairs]
        values = [value.to(self.device) for _, value in source_pairs]
        positions = torch.arange(length, device=self.device).unsqueeze(0)
        rotary = source_rotary or self.model.model.rotary_emb
        keys = content_keys(keys, rotary, positions)
        cos, sin = self.model.model.rotary_emb(keys[0], positions)
        cos, sin = cos.float().unsqueeze(1), sin.float().unsqueeze(1)
        cache = DynamicCache()
        dtype = self.model.dtype
        for index, layer in enumerate(self.layers):
            output = []
            for states, component in ((keys, "k"), (values, "v")):
                features = torch.cat([states[i].permute(0, 2, 1, 3)
                                      for i in layer["selected_list"]], dim=-1)
                features = features.index_select(2, layer["source_heads"]).float()
                mapped = torch.einsum("bshf,hfd->bshd", features, layer[component + "_weight"])
                mapped = (mapped + layer[component + "_bias"]).permute(0, 2, 1, 3)
                output.append(mapped)
            rotated = output[0] * cos + rotate_half(output[0]) * sin
            cache.update(rotated.to(dtype).contiguous(), output[1].to(dtype).contiguous(), index)
        return cache
