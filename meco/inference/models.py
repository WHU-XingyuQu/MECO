"""Fixed identities for the current small-to-strong model pair."""
from pathlib import Path
SOURCE_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SOURCE_REV = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
TARGET_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
TARGET_REV = "c03e6d358207e414f1eca0bb1891e29f1db0e242"
MAPPER_SHA256 = "7ff495e8be7b2a44e475878f18bfc843b5b3bf501a3f4d9c48019004eee27d66"
CACHE = Path.home()/".cache/huggingface/hub"
SOURCE = CACHE/("models--"+SOURCE_ID.replace("/","--"))/"snapshots"/SOURCE_REV
TARGET = CACHE/("models--"+TARGET_ID.replace("/","--"))/"snapshots"/TARGET_REV

def validate_pair(config):
    for key,value in {"source_model_id":SOURCE_ID,"target_model_id":TARGET_ID,
                      "source_revision":SOURCE_REV,"target_revision":TARGET_REV}.items():
        if config.get(key) != value:
            raise ValueError("artifact differs from the fixed runtime model pair: "+key)
