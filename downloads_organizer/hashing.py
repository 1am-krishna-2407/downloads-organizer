"""SHA-256 hashing and exact-duplicate grouping (Phase 2)."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Iterable

CHUNK_SIZE = 1024 * 1024


def sha256_file(path: str | Path) -> str:
    """Hash a file in chunks so large files are never loaded into memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        while chunk := fh.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def group_exact_duplicates(items: Iterable[tuple[int, str]]) -> list[list[int]]:
    """Group ids by identical hash. Takes (id, sha256) pairs; returns groups of 2+ ids."""
    by_hash: dict[str, list[int]] = defaultdict(list)
    for doc_id, digest in items:
        if digest:
            by_hash[digest].append(doc_id)
    return sorted((sorted(ids) for ids in by_hash.values() if len(ids) > 1), key=lambda g: g[0])
