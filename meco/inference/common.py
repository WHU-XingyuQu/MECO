"""Token IDs, model geometry, and exact prefix identity."""
import hashlib
import json
MAX_PREFIX_TOKENS = 1024

def token_hash(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest()

def architecture(config):
    rope = getattr(config, "rope_parameters", None)
    return {"model_type": config.model_type, "layers": config.num_hidden_layers,
            "kv_heads": config.num_key_value_heads,
            "head_dim": getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            "rope_theta": rope["rope_theta"] if rope else config.rope_theta,
            "vocab_size": config.vocab_size}

def validate_ids(ids, vocab_size, *, allow_empty=False, limit=2048):
    if not isinstance(ids, list) or (not ids and not allow_empty) or len(ids) > limit:
        raise ValueError("invalid token ID count")
    if any(type(token) is not int or token < 0 or token >= vocab_size for token in ids):
        raise ValueError("invalid token ID")
