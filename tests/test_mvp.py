"""Tests for setup, configuration, folder detection, locking, OCR and similarity."""

import os
from dataclasses import replace
from pathlib import Path

import pymupdf
import pytest

from downloads_organizer import extractors, similarity
from downloads_organizer.cli import main
from downloads_organizer.config import (
    ExtractionSettings, classification_settings, load_config, user_config_path,
)
from downloads_organizer.locking import AlreadyRunning, run_lock
from downloads_organizer.scanner import folder_kind, scan

needs_tesseract = pytest.mark.skipif(extractors.tessdata_path(None) is None, reason="Tesseract not installed")


# --- configuration ----------------------------------------------------------------------


def test_defaults_work_without_any_config_file(tmp_path):
    cfg = load_config()
    assert cfg.source is None
    assert cfg.data_dir == (Path(os.environ["SDO_HOME"]) / "data").resolve()
    assert cfg.organized_dir == cfg.downloads_dir / "Organized"
    assert ".pdf" in cfg.supported_extensions and cfg.classification.provider == "nvidia"


def test_user_config_only_needs_overrides(tmp_path):
    path = user_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"downloads_dir: '{tmp_path.as_posix()}'\nextraction: {{max_pages: 5}}\n", encoding="utf-8")
    cfg = load_config()
    assert cfg.source == path and cfg.downloads_dir == tmp_path.resolve()
    assert cfg.extraction.max_pages == 5 and cfg.extraction.max_chars == 100_000  # untouched defaults kept


def test_unknown_settings_are_reported(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("dowloads_dir: x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="dowloads_dir"):
        load_config(path)


@pytest.mark.parametrize("provider, model, local", [
    ("nvidia", "nvidia/nemotron-3.5-lightning-30b-a3b", False),
    ("openrouter", "nvidia/nemotron-3.5-lightning", False),
    ("ollama", "nemotron-3-nano:4b", True),
])
def test_provider_presets(provider, model, local):
    s = classification_settings({"provider": provider})
    assert (s.model, s.is_local) == (model, local)


def test_explicit_settings_override_preset_including_null_escalation():
    s = classification_settings({"provider": "nvidia", "model": "nvidia/other", "escalation_model": None})
    assert s.model == "nvidia/other" and s.escalation_model is None


def test_custom_provider_needs_url_and_model():
    with pytest.raises(ValueError, match="custom"):
        classification_settings({"provider": "custom"})
    with pytest.raises(ValueError, match="Unknown provider"):
        classification_settings({"provider": "nope"})


# --- folder detection -------------------------------------------------------------------


def _make(root: Path, files: list[str]) -> Path:
    for f in files:
        (root / f).parent.mkdir(parents=True, exist_ok=True)
        (root / f).write_bytes(b"x")
    return root


@pytest.mark.parametrize("files, expected", [
    (["app/tool.exe", "app/lib.dll"], "software"),
    (["app/tool.exe", "app/bin/x.txt"], "software"),
    (["app/tool.exe", "app/data/core.dll"], "software"),
    (["app/only.exe"], "software"),
    (["app/wrapper/tool.exe", "app/wrapper/x.dll"], "software"),        # zip unpacked into a wrapper folder
    (["app/package.json", "app/index.js"], "project"),
    (["app/manifest.json", "app/popup.js"], "project"),                # browser extension
    (["app/Solution.sln", "app/readme.md"], "project"),
    (["app/node_modules/x/index.js"], "software"),                     # wrapper around node_modules
    (["app/notes.pdf", "app/photo.jpg"], None),
    (["app/setup.exe", "app/manual.pdf"], None),                       # an installer next to a document
])
def test_folder_kind(tmp_path, config, files, expected):
    _make(tmp_path, files)
    assert folder_kind(tmp_path / "app", config) == expected


def test_project_name_pattern(tmp_path, config):
    (tmp_path / "Tetris-master").mkdir()
    assert folder_kind(tmp_path / "Tetris-master", config) == "project"


def test_scan_skips_software_and_reports_it(config, downloads):
    _make(downloads, ["PortableTool/tool.exe", "PortableTool/core.dll", "PortableTool/docs/manual.pdf",
                      "site/node_modules/pkg/readme.md"])
    result = scan(config)
    names = {r.filename for r in result.records}
    assert "manual.pdf" not in names and "readme.md" not in names
    assert (str(downloads / "PortableTool"), "software") in result.skipped_folders

    result = scan(replace(config, skip_software_folders=False, skip_project_folders=False))
    assert "manual.pdf" in {r.filename for r in result.records}


# --- locking ----------------------------------------------------------------------------


def test_only_one_run_at_a_time(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(AlreadyRunning):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):  # released afterwards
        pass


# --- OCR -----------------------------------------------------------------------------------


def _text_image(path: Path, text: str) -> Path:
    doc = pymupdf.open()
    page = doc.new_page(width=900, height=300)
    page.insert_text((40, 120), text, fontsize=28)
    page.get_pixmap(dpi=150).save(path)
    doc.close()
    return path


def _photo(path: Path) -> Path:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 400, 300), False)
    pix.set_rect(pix.irect, (120, 160, 200))
    pix.save(path)
    return path


@needs_tesseract
def test_ocr_reads_text_in_images(tmp_path):
    path = _text_image(tmp_path / "receipt.png", "Payment receipt for the electricity bill, total amount due this month")
    result = extractors.extract(path, ExtractionSettings())
    assert result.status == extractors.OK and "receipt" in result.text.lower()


@needs_tesseract
def test_ocr_treats_pictures_without_words_as_photos(tmp_path):
    result = extractors.extract(_photo(tmp_path / "holiday.png"), ExtractionSettings())
    assert result.status == extractors.EMPTY


@needs_tesseract
def test_ocr_reads_scanned_pdfs(tmp_path):
    image = _text_image(tmp_path / "page.png", "Certificate of completion awarded to the student")
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(page.rect, filename=str(image))  # an image-only page, like a scan
    doc.save(tmp_path / "scan.pdf")
    doc.close()
    result = extractors.extract(tmp_path / "scan.pdf", ExtractionSettings())
    assert result.method == "pymupdf+ocr" and "certificate" in result.text.lower()


def test_tessdata_can_be_configured(tmp_path):
    (tmp_path / "eng.traineddata").write_bytes(b"x")
    extractors.tessdata_path.cache_clear()
    assert extractors.tessdata_path(str(tmp_path)) == str(tmp_path)


# --- similarity ---------------------------------------------------------------------------


def test_similar_groups_find_versions_not_unrelated_documents():
    base = " ".join(f"Section {i} explains the project architecture and its data flow." for i in range(80))
    texts = {
        1: base,
        2: base.replace("Section 5 ", "Part 5 ") + " Appendix added.",
        3: " ".join(f"Recipe {i}: mix flour, sugar and butter, then bake." for i in range(80)),
        4: "too short",
    }
    groups = similarity.similar_groups(texts, 0.8)
    assert [ids for ids, _ in groups] == [[1, 2]]
    assert 0.8 <= groups[0][1] < 1.0


# --- setup commands -------------------------------------------------------------------------


def test_init_non_interactive_writes_config_and_key(tmp_path, monkeypatch, capsys):
    from downloads_organizer import classifier

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)
    assert main(["init", "--downloads", str(tmp_path), "--provider", "nvidia", "--api-key", "nvapi-abc123xyz"]) == 0
    cfg = load_config()
    assert cfg.downloads_dir == tmp_path.resolve()
    assert classifier.load_api_key(cfg.classification) == "nvapi-abc123xyz"
    assert "nvapi-abc123xyz" not in user_config_path().read_text(encoding="utf-8")  # key kept out of the config
    # Running it again keeps the existing config unless --force.
    assert main(["init", "--provider", "ollama"]) == 0
    assert load_config().classification.provider == "nvidia"


def test_doctor_reports_missing_key(capsys, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(f"downloads_dir: '{tmp_path.as_posix()}'\n", encoding="utf-8")
    assert main(["doctor", "--config", str(path)]) == 1
    out = capsys.readouterr().out
    assert "[FAIL] API key" in out and "[ok  ] Downloads folder" in out


def test_doctor_all_good_with_local_model(capsys, tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(f"downloads_dir: '{tmp_path.as_posix()}'\nclassification: {{provider: ollama}}\n",
                    encoding="utf-8")
    assert main(["doctor", "--config", str(path)]) == 0
    assert "not needed (local model)" in capsys.readouterr().out
