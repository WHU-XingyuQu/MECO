"""Length-delimited JSON metadata and safe tensor bytes for the cascade service."""
import hashlib
import json
import struct

MAX_BODY = 96 * 1024 * 1024


def pack(metadata, payload=b""):
    header = json.dumps(metadata, ensure_ascii=False, separators=(",", ":")).encode()
    body = struct.pack("!I", len(header)) + header + payload
    if len(body) > MAX_BODY:
        raise ValueError("cascade transport bound exceeded")
    return body


def unpack(body):
    if not 4 <= len(body) <= MAX_BODY:
        raise ValueError("invalid cascade body length")
    length = struct.unpack("!I", body[:4])[0]
    if not 0 < length <= min(len(body) - 4, 1024 * 1024):
        raise ValueError("invalid cascade header length")
    metadata = json.loads(body[4:4 + length])
    if not isinstance(metadata, dict):
        raise ValueError("invalid cascade metadata")
    return metadata, body[4 + length:]


def digest(payload):
    return hashlib.sha256(payload).hexdigest()
