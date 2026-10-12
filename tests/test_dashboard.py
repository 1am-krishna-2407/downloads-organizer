"""Smoke tests: every dashboard page renders against a real (temporary) database."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from downloads_organizer import database, decision_engine, pipeline
from downloads_organizer.config import ExtractionSettings
from downloads_organizer.scanner import scan

APP = str(Path(__file__).resolve().parent.parent / "downloads_organizer" / "dashboard.py")
TITLES = {"Overview": "Overview", "Review queue": "Review queue", "All files": "All files",
          "Duplicates": "Duplicate copies", "Move files": "Move files", "History": "History"}


@pytest.fixture
def dashboard_env(config, downloads, tmp_path, monkeypatch):
    (downloads / "book.txt").write_text("Chapter 1", encoding="utf-8")
    (downloads / "book (1).txt").write_text("Chapter 1", encoding="utf-8")
    cfg = replace(config, extraction=ExtractionSettings(ocr_enabled=False))
    conn = database.connect(cfg.database_path)
    database.save_scan(conn, scan(cfg))
    pipeline.process_documents(conn, cfg)
    payload = {"classification": {
        "document_type": "Reference material", "category": "Reference", "subject": "s", "summary": "A book",
        "purpose": "p", "importance": "low", "recommended_action": "ARCHIVE", "confidence": 0.9,
        "reasons": ["book"], "sensitivity": "none", "duplicate_suspected": False}, "model_used": "m"}
    conn.execute("UPDATE documents SET classification_status='done', classification_json=? WHERE filename LIKE 'book%'",
                 (json.dumps(payload),))
    conn.commit()
    decision_engine.decide_all(conn, cfg)
    conn.close()

    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(f"downloads_dir: '{downloads.as_posix()}'\nsupported_extensions: [.txt, .pdf]\n"
                        f"data_dir: '{cfg.data_dir.as_posix()}'\n", encoding="utf-8")
    monkeypatch.setenv("SDO_CONFIG", str(cfg_file))
    return cfg


@pytest.mark.parametrize("page", TITLES)
def test_page_renders(dashboard_env, page):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value(page).run()
    assert not at.exception, at.exception
    assert at.title[0].value == TITLES[page]


def test_move_requires_confirmation(dashboard_env):
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.sidebar.radio[0].set_value("Move files").run()
    move = next(b for b in at.button if b.label == "Move files")
    assert move.disabled
