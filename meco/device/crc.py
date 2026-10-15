"""CRC16/CCITT-FALSE and canonical integer validation."""
import re
UINT32 = 0xFFFFFFFF

def crc16(payload: bytes) -> int:
    crc = 0xFFFF
    for byte in payload:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ (0x1021 if crc & 0x8000 else 0)) & 0xFFFF
    return crc

def integer(text: str, lower=0, upper=UINT32) -> int:
    if not re.fullmatch(r"-?(?:0|[1-9][0-9]*)", text):
        raise ValueError("noncanonical integer")
    value = int(text)
    if str(value) != text or not lower <= value <= upper:
        raise ValueError("integer out of range")
    return value
