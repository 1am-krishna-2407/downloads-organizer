import hashlib
import os
from datetime import datetime
from dataclasses import replace
from pathlib import Path

import pytest

from downloads_organizer import scanner
from downloads_organizer.config import load_config
from downloads_organizer.scanner import scan


def _snapshot(root: Path) -> dict[str, tuple]:
    """Every path under root with its size, mtime and content hash."""
    snap = {}
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            path = Path(dirpath) / name
            st = path.stat()
            digest = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
            snap[str(path)] = (st.st_size, st.st_mtime_ns, digest)
    return snap


def test_inventories_supported_and_unsupported_files(config, downloads):
    result = scan(config)
    names = sorted(r.filename for r in result.records)
    assert names == sorted(
        ["resume.pdf", "Notes.DOCX", "setup.exe", "noextension", "invoice.txt", "café menu.xlsx"]
    )
    by_name = {r.filename: r for r in result.records}
    assert by_name["Notes.DOCX"].extension == ".docx"
    assert by_name["Notes.DOCX"].supported
    assert not by_name["setup.exe"].supported
    assert by_name["noextension"].extension == ""
    assert result.errors == []


def test_records_correct_metadata(config, downloads):
    target = downloads / "nested" / "deeper" / "café menu.xlsx"
    os.utime(target, (1_700_000_000, 1_700_000_000))
    record = next(r for r in scan(config).records if r.filename == "café menu.xlsx")

    assert record.original_path == str(target)
    assert record.size_bytes == 2048
    expected = datetime.fromtimestamp(1_700_000_000).strftime("%Y-%m-%dT%H:%M:%S")
    assert record.modified_at.startswith(expected)


def test_skips_temporary_system_and_output_files(config):
    result = scan(config)
    names = {r.filename for r in result.records}
    assert "movie.mp4.crdownload" not in names
    assert "~$Notes.docx" not in names
    assert "desktop.ini" not in names
    assert "old_resume.pdf" not in names
    assert result.skipped == {
        "incomplete_download": 1,
        "temp_or_lock_file": 1,
        "system_file": 1,
        "excluded_folder": 1,
    }


def test_scan_does_not_modify_anything(config, downloads):
    before = _snapshot(downloads)
    scan(config)
    assert _snapshot(downloads) == before


def test_non_recursive_scan_stays_at_top_level(config):
    result = scan(replace(config, recursive=False))
    assert "invoice.txt" not in {r.filename for r in result.records}
    assert "resume.pdf" in {r.filename for r in result.records}


def test_unreadable_file_is_reported_and_scan_continues(config, downloads, monkeypatch):
    real_stat = Path.stat

    def flaky_stat(self, *args, **kwargs):
        if self.name == "resume.pdf":
            raise PermissionError(13, "Access is denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", flaky_stat)
    result = scan(config)

    assert len(result.errors) == 1
    assert result.errors[0].path.endswith("resume.pdf")
    assert "PermissionError" in result.errors[0].error
    assert len(result.records) == 5


def test_unlistable_folder_is_reported(config, monkeypatch):
    real_walk = os.walk

    def walk_with_error(top, onerror=None, **kwargs):
        onerror(PermissionError(13, "Access is denied", str(Path(top) / "locked")))
        yield from real_walk(top, onerror=onerror, **kwargs)

    monkeypatch.setattr(scanner.os, "walk", walk_with_error)
    result = scan(config)
    assert [e.path for e in result.errors] == [str(config.downloads_dir / "locked")]
    assert len(result.records) == 6


def test_missing_root_raises(config, tmp_path):
    with pytest.raises(NotADirectoryError):
        scan(config, tmp_path / "does-not-exist")


def test_load_config_resolves_paths(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        "downloads_dir: dl\n"
        "organized_dir: dl/Organized\n"
        "exclude_dirs: [Installers]\n"
        "supported_extensions: [PDF, .Docx]\n"
        "skip_names: [Desktop.ini]\n",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.downloads_dir == (tmp_path / "dl").resolve()
    assert cfg.data_dir == (Path(os.environ["SDO_HOME"]) / "data").resolve()  # per-user data folder
    assert cfg.exclude_dirs == ((tmp_path / "dl" / "Installers").resolve(),)
    assert cfg.supported_extensions == {".pdf", ".docx"}
    assert cfg.skip_names == {"desktop.ini"}

    override = load_config(cfg_file, downloads_dir=tmp_path / "other")
    assert override.downloads_dir == (tmp_path / "other").resolve()
