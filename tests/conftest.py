from pathlib import Path

import pytest

from downloads_organizer.config import AutomationSettings, Config


@pytest.fixture(autouse=True)
def isolated_user_dirs(tmp_path_factory, monkeypatch):
    """Keep tests away from the real per-user config, data and API keys."""
    home = tmp_path_factory.mktemp("sdo_home")
    monkeypatch.setenv("SDO_HOME", str(home))
    monkeypatch.delenv("SDO_CONFIG", raising=False)
    for name in ("NVIDIA_API_KEY", "OPENROUTER_API_KEY", "SDO_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    from downloads_organizer import classifier
    monkeypatch.setattr(classifier, "_key_sources", lambda: [home / "secrets.env"])
    return home


@pytest.fixture
def downloads(tmp_path: Path) -> Path:
    """A small fake Downloads folder."""
    root = tmp_path / "Downloads"
    (root / "nested" / "deeper").mkdir(parents=True)
    (root / "Organized" / "Career").mkdir(parents=True)

    (root / "resume.pdf").write_bytes(b"%PDF-1.4 fake")
    (root / "Notes.DOCX").write_bytes(b"docx")
    (root / "setup.exe").write_bytes(b"MZ")
    (root / "noextension").write_bytes(b"")
    (root / "nested" / "invoice.txt").write_text("Invoice 42", encoding="utf-8")
    (root / "nested" / "deeper" / "café menu.xlsx").write_bytes(b"x" * 2048)

    # Must be skipped.
    (root / "movie.mp4.crdownload").write_bytes(b"partial")
    (root / "~$Notes.docx").write_bytes(b"lock")
    (root / "desktop.ini").write_text("[.ShellClassInfo]", encoding="utf-8")
    (root / "Organized" / "Career" / "old_resume.pdf").write_bytes(b"already organised")
    return root


@pytest.fixture
def config(downloads: Path, tmp_path: Path) -> Config:
    return Config(
        downloads_dir=downloads,
        organized_dir=downloads / "Organized",
        data_dir=tmp_path / "data",
        reports_dir=tmp_path / "reports",
        logs_dir=tmp_path / "logs",
        supported_extensions=frozenset({".pdf", ".docx", ".txt", ".xlsx"}),
        skip_extensions=frozenset({".crdownload", ".part", ".tmp"}),
        skip_names=frozenset({"desktop.ini", "thumbs.db"}),
        skip_name_prefixes=("~$",),
        automation=AutomationSettings(min_file_age_seconds=0),
    )
