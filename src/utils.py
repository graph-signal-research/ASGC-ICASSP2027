from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    data = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(f'Configuration must be a mapping: {path}')
    return data


def stable_seed(*parts: Any) -> int:
    """Derive a deterministic 32-bit seed from configuration fields."""
    payload = '|'.join(map(str, parts)).encode('utf-8')
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little')
    return value % (2**32 - 1)


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def array_hash(array: np.ndarray) -> str:
    """Hash the dtype, shape, and bytes of a NumPy array."""
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(contiguous.dtype).encode())
    digest.update(json.dumps(list(contiguous.shape)).encode())
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def sensor_id_hash(ids: list[str]) -> str:
    """Hash an ordered sensor-ID list using newline-separated UTF-8 text."""
    payload = ('\n'.join(ids) + '\n').encode()
    return hashlib.sha256(payload).hexdigest()
