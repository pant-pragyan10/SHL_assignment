from typing import Dict, Any

EXPECTED_SCHEMA = {
    'reply': str,
    'recommendations': list,
    'end_of_conversation': bool,
}


def validate_schema(resp: Dict[str, Any]) -> bool:
    if not isinstance(resp, dict):
        return False
    for k, t in EXPECTED_SCHEMA.items():
        if k not in resp:
            return False
        if not isinstance(resp[k], t):
            return False
    # additional checks: recommendations items must have name and url
    for it in resp.get('recommendations', []):
        if not isinstance(it, dict):
            return False
        if 'name' not in it or 'url' not in it:
            return False
    return True
