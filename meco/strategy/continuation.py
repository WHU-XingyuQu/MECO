"""Exact consumed-token boundary for full KV continuation."""
FOLLOWUP = ("Review the previous maintenance diagnosis against the equipment document "
            "and the observed values. Correct any error. Return only the final JSON "
            "with exactly diagnosis, action, and reason, using the documented uppercase identifiers.")

def continuation_ids(tokenizer, input_ids, output_ids, cache_tokens, *, followup=FOLLOWUP):
    if not input_ids or not output_ids:
        raise ValueError("both actual upstream prompt and generated tokens are required")
    # Greedy generation has sampled, but not consumed, the final output token.
    prefix = list(input_ids) + list(output_ids[:-1])
    if len(prefix) != cache_tokens:
        raise ValueError("upstream cache does not match its consumed token boundary")
    end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    history = list(input_ids) + list(output_ids)
    if history[-1] != end:
        history.append(end)
    followup = "\n<|im_start|>user\n" + followup + "<|im_end|>\n<|im_start|>assistant\n"
    full = history + tokenizer.encode(followup, add_special_tokens=False)
    if full[:cache_tokens] != prefix:
        raise AssertionError("continuation lost the exact upstream token prefix")
    return prefix, full[cache_tokens:], full
