"""Small-model FP32 greedy generation and its consumed KV cache."""
from pathlib import Path
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .common import MAX_PREFIX_TOKENS, architecture, validate_ids

class HFCPUModel:
    def __init__(self, *, source_model_path, model_id, precision="float32", threads=4):
        if precision != "float32":
            raise ValueError("This cascade uses the FP32 small CPU model")
        torch.set_num_threads(threads)
        self.model_id, self.precision = model_id, precision
        self.source_model_path = str(Path(source_model_path).resolve(strict=True))
        self.tokenizer = AutoTokenizer.from_pretrained(self.source_model_path, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.source_model_path, dtype=torch.float32, low_cpu_mem_usage=True,
            local_files_only=True, attn_implementation="eager")
        self.model.eval()
        self.arch = architecture(self.model.config)
        if len(self.tokenizer) > self.arch["vocab_size"]:
            raise ValueError("tokenizer exceeds model vocabulary")

    def generate(self, input_ids, max_new_tokens, *, cache=None, prefix_tokens=0, on_token=None,
                 return_cache=False):
        validate_ids(input_ids, self.arch["vocab_size"], limit=2048)
        if type(max_new_tokens) is not int or not 1 <= max_new_tokens <= 128:
            raise ValueError("invalid generation token limit")
        if cache is not None and (not 0 < prefix_tokens <= MAX_PREFIX_TOKENS
                                  or cache.get_seq_length() != prefix_tokens):
            raise ValueError("transferred cache length mismatch")
        started = time.perf_counter()
        produced = []
        first_ms = None
        with torch.inference_mode():
            ids = torch.tensor([input_ids], dtype=torch.long)
            result = self.model(input_ids=ids, past_key_values=cache, use_cache=True,
                                cache_position=torch.arange(prefix_tokens, prefix_tokens + len(input_ids)),
                                return_dict=True, logits_to_keep=1)
            live_cache = result.past_key_values
            for step in range(max_new_tokens):
                next_id = int(result.logits[0, -1].argmax())
                if first_ms is None:
                    first_ms = (time.perf_counter() - started) * 1000
                produced.append(next_id)
                if on_token is not None:
                    on_token(next_id, first_ms if step == 0 else (time.perf_counter() - started) * 1000)
                if next_id in self.tokenizer.all_special_ids or step + 1 == max_new_tokens:
                    break
                position = prefix_tokens + len(input_ids) + step
                result = self.model(input_ids=torch.tensor([[next_id]]), past_key_values=live_cache,
                                    cache_position=torch.tensor([position]), use_cache=True,
                                    return_dict=True)
                live_cache = result.past_key_values
        output = {"text": self.tokenizer.decode(produced, skip_special_tokens=True),
                "output_ids": produced, "ttft_ms": first_ms,
                "full_ms": (time.perf_counter() - started) * 1000,
                "cached_tokens": prefix_tokens}
        if return_cache:
            # The final sampled token has not been consumed by the transformer.
            # Callers must include it in the next suffix, not claim it is cached.
            output["cache"] = live_cache
            output["cache_tokens"] = prefix_tokens + len(input_ids) + len(produced) - 1
            if live_cache.get_seq_length() != output["cache_tokens"]:
                raise RuntimeError("generation cache/token boundary mismatch")
        return output
