"""Strong-model CUDA generation with complete mapped prefix state."""
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .common import MAX_PREFIX_TOKENS, validate_ids

def model_architecture(config):
    rope = getattr(config, "rope_parameters", None)
    return {"model_type": config.model_type, "layers": config.num_hidden_layers,
            "kv_heads": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", None) or
            config.hidden_size // config.num_attention_heads,
            "rope_theta": rope["rope_theta"] if rope else config.rope_theta,
            "vocab_size": config.vocab_size}

class HFGPUModel:
    def __init__(self, *, source_model_path, model_id, precision="float16", threads=4):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; the experiment cannot use a CPU substitute")
        if precision not in {"float16", "bfloat16"}:
            raise ValueError("GPU precision must be float16 or bfloat16")
        torch.set_num_threads(threads)
        self.source_model_path = str(Path(source_model_path).resolve(strict=True))
        self.model_id, self.precision = model_id, precision
        self.device = torch.device("cuda:0")
        self.dtype = getattr(torch, precision)
        self.tokenizer = AutoTokenizer.from_pretrained(self.source_model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.source_model_path, torch_dtype=self.dtype, low_cpu_mem_usage=True,
            local_files_only=True, attn_implementation="sdpa").to(self.device).eval()
        self.arch = model_architecture(self.model.config)
        self.artifact_sha256 = None
        torch.cuda.synchronize()

    def resource_snapshot(self):
        return {"device": str(self.device), "allocated_bytes": torch.cuda.memory_allocated(),
                "reserved_bytes": torch.cuda.memory_reserved(),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved()}

    def generate(self, input_ids, max_new_tokens, *, cache=None, prefix_tokens=0, on_token=None):
        validate_ids(input_ids, self.arch["vocab_size"], limit=2048)
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
            raise ValueError("invalid output budget")
        if cache is not None and (not 0 < prefix_tokens <= MAX_PREFIX_TOKENS
                                  or cache.get_seq_length() != prefix_tokens):
            raise ValueError("cache length mismatch")
        torch.cuda.synchronize()
        started = time.perf_counter()
        produced, first_ms = [], None
        with torch.inference_mode():
            result = self.model(
                input_ids=torch.tensor([input_ids], device=self.device), past_key_values=cache,
                cache_position=torch.arange(prefix_tokens, prefix_tokens + len(input_ids),
                                            device=self.device), use_cache=True, return_dict=True,
                logits_to_keep=1)
            live_cache = result.past_key_values
            for step in range(max_new_tokens):
                token = int(result.logits[0, -1].argmax())
                if first_ms is None:
                    first_ms = (time.perf_counter() - started) * 1000
                produced.append(token)
                if on_token:
                    on_token(token, (time.perf_counter() - started) * 1000)
                if token in self.tokenizer.all_special_ids or step + 1 == max_new_tokens:
                    break
                position = prefix_tokens + len(input_ids) + step
                result = self.model(input_ids=torch.tensor([[token]], device=self.device),
                                    past_key_values=live_cache,
                                    cache_position=torch.tensor([position], device=self.device),
                                    use_cache=True, return_dict=True)
                live_cache = result.past_key_values
        torch.cuda.synchronize()
        return {"text": self.tokenizer.decode(produced, skip_special_tokens=True),
                "output_ids": produced, "ttft_ms": first_ms,
                "full_ms": (time.perf_counter() - started) * 1000, "cached_tokens": prefix_tokens}
