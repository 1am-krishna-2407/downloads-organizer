import hashlib

from downloads_organizer import hashing
from downloads_organizer.hashing import group_exact_duplicates, sha256_file


def test_sha256_matches_hashlib(tmp_path):
    path = tmp_path / "file.bin"
    data = b"abc" * 1000
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_sha256_streams_large_files_in_chunks(tmp_path, monkeypatch):
    monkeypatch.setattr(hashing, "CHUNK_SIZE", 7)
    path = tmp_path / "file.bin"
    data = bytes(range(256)) * 10
    path.write_bytes(data)
    assert sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_group_exact_duplicates():
    items = [(1, "a"), (2, "b"), (3, "a"), (4, None), (5, "c"), (6, "c"), (7, "a")]
    assert group_exact_duplicates(items) == [[1, 3, 7], [5, 6]]


def test_no_duplicates():
    assert group_exact_duplicates([(1, "a"), (2, "b")]) == []
