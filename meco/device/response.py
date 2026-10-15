"""Validate the application response before sending an MCU command."""
import json
import re

def parse_response(text):
    """Accept plain JSON or an exact JSON fence; never repair labels or free text."""
    if not isinstance(text, str) or not text or len(text.encode()) > 4096:
        raise ValueError("invalid response type/size")
    body, format_name = text.strip(), "plain_json"
    if body.startswith("```"):
        match = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", body, re.DOTALL)
        if not match:
            raise ValueError("invalid JSON fence")
        body, format_name = match.group(1), "fenced_json"

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    obj = json.loads(body, object_pairs_hook=unique_object)
    if type(obj) is not dict or set(obj) != {"diagnosis", "action", "reason"}:
        raise ValueError("response must contain exactly diagnosis/action/reason")
    if any(type(value) is not str or not value.strip() for value in obj.values()):
        raise ValueError("response fields must be nonempty strings")
    if len(obj["reason"]) > 256 or any(ord(char) < 32 for char in obj["reason"]):
        raise ValueError("invalid explanation")
    return obj, format_name
