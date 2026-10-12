"""Near-duplicate detection: documents whose text is largely the same (FR-09).

Typical finds: resume versions, a re-downloaded PDF with a different watermark,
an assignment draft and its final copy. Uses MinHash + LSH to find candidate
pairs quickly, then confirms each pair with the exact Jaccard similarity of
word 5-grams. Byte-identical files are handled by exact hashing instead.
"""

from __future__ import annotations

import re
import zlib
from collections import defaultdict

import numpy as np

SHINGLE_WORDS = 5
MAX_CHARS = 20_000      # text used per document; enough to recognise versions
MIN_SHINGLES = 20       # shorter texts are too small to compare meaningfully
NUM_PERM = 64
BANDS, ROWS = 16, 4     # LSH: candidate pairs from ~0.5 similarity upwards
_PRIME = (1 << 61) - 1
_rng = np.random.default_rng(20261012)
# a < 2^31 and crc32 values < 2^32 keep a*x + b below 2^64, so the arithmetic is exact.
_A = _rng.integers(1, 1 << 31, NUM_PERM, dtype=np.uint64)
_B = _rng.integers(0, _PRIME, NUM_PERM, dtype=np.uint64)
_WORD = re.compile(r"\w+")


def shingles(text: str) -> set[int]:
    words = _WORD.findall(text[:MAX_CHARS].lower())
    return {
        zlib.crc32(" ".join(words[i : i + SHINGLE_WORDS]).encode())
        for i in range(max(0, len(words) - SHINGLE_WORDS + 1))
    }


def minhash(items: set[int]) -> np.ndarray:
    x = np.fromiter(items, dtype=np.uint64, count=len(items))
    hashed = (np.outer(_A, x) + _B[:, None]) % np.uint64(_PRIME)
    return hashed.min(axis=1)


def jaccard(a: set[int], b: set[int]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def similar_groups(texts: dict[int, str], threshold: float) -> list[tuple[list[int], float]]:
    """Group ids whose texts are at least ``threshold`` similar. Returns (ids, lowest pair score)."""
    sets = {doc_id: s for doc_id, text in texts.items() if len(s := shingles(text)) >= MIN_SHINGLES}
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for doc_id, items in sets.items():
        signature = minhash(items)
        for band in range(BANDS):
            buckets[(band, signature[band * ROWS : (band + 1) * ROWS].tobytes())].append(doc_id)

    parent = {doc_id: doc_id for doc_id in sets}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    checked: set[tuple[int, int]] = set()
    scores: dict[int, float] = {}
    for members in buckets.values():
        for i, a in enumerate(members):
            for b in members[i + 1 :]:
                pair = (min(a, b), max(a, b))
                if pair in checked:
                    continue
                checked.add(pair)
                score = jaccard(sets[a], sets[b])
                if score >= threshold:
                    ra, rb = find(a), find(b)
                    parent[rb] = ra
                    scores[pair[0]] = min(scores.get(pair[0], 1.0), score)
                    scores[pair[1]] = min(scores.get(pair[1], 1.0), score)

    groups: dict[int, list[int]] = defaultdict(list)
    for doc_id in sets:
        groups[find(doc_id)].append(doc_id)
    result = []
    for ids in groups.values():
        if len(ids) > 1:
            result.append((sorted(ids), round(min(scores.get(i, 1.0) for i in ids), 3)))
    return sorted(result, key=lambda g: g[0][0])
