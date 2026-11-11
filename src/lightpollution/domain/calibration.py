"""Portable numeric checksum for a JSON calibration exchanged with browsers."""

import hashlib
import json


def calibration_key(value):
    def canonical(item):
        if isinstance(item, dict):
            return {k: canonical(v) for k, v in item.items()}
        if isinstance(item, (list, tuple)):
            return [canonical(v) for v in item]
        if type(item) in (int, float):
            # JSON/JavaScript has one Number type: 22 and 22.0 must agree.
            number = float(item)
            return 0. if number == 0 else number
        return item
    encoded = json.dumps(canonical(value), sort_keys=True, separators=(',', ':'),
                         ensure_ascii=False, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()
