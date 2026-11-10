"""Content keys and atomic files. A partial file is never a published result."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np


def content_key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def atomic_target(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        yield temporary
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path: Path, value):
    serialized = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'
    with atomic_target(path) as temporary:
        temporary.write_text(serialized)


def read_json(path: Path):
    return json.loads(path.read_text())


def write_npz(path: Path, **values):
    with atomic_target(path) as temporary:
        with temporary.open('wb') as stream:
            np.savez_compressed(stream, **values)
