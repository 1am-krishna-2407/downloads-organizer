"""SQLite storage for scan results, duplicates, file operations and the classification cache."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from downloads_organizer.scanner import ScanResult

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    original_path         TEXT NOT NULL UNIQUE,
    filename              TEXT NOT NULL,
    extension             TEXT NOT NULL,
    size_bytes            INTEGER NOT NULL,
    created_at            TEXT,
    modified_at           TEXT NOT NULL,
    supported             INTEGER NOT NULL DEFAULT 0,
    present               INTEGER NOT NULL DEFAULT 1,
    sha256                TEXT,
    extraction_status     TEXT NOT NULL DEFAULT 'pending',
    document_type         TEXT,
    category              TEXT,
    summary               TEXT,
    importance            TEXT,
    recommended_action    TEXT,
    confidence            REAL,
    classification_status TEXT NOT NULL DEFAULT 'pending',
    first_seen_at         TEXT NOT NULL,
    last_scanned_at       TEXT NOT NULL,
    last_scan_run         INTEGER,
    processing_error      TEXT,
    classification_json   TEXT
);
CREATE INDEX IF NOT EXISTS idx_documents_sha256 ON documents(sha256);

CREATE TABLE IF NOT EXISTS duplicate_groups (
    group_id         INTEGER NOT NULL,
    document_id      INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    duplicate_type   TEXT NOT NULL,
    similarity_score REAL,
    PRIMARY KEY (group_id, document_id)
);

CREATE TABLE IF NOT EXISTS file_operations (
    operation_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id      INTEGER NOT NULL REFERENCES documents(id),
    source_path      TEXT NOT NULL,
    destination_path TEXT NOT NULL,
    operation_type   TEXT NOT NULL,
    approved_at      TEXT,
    completed_at     TEXT,
    status           TEXT NOT NULL,
    batch_id         TEXT,
    error            TEXT,
    rolled_back_at   TEXT
);

-- Final decisions, recomputed by the decision engine on every run.
CREATE TABLE IF NOT EXISTS decisions (
    document_id  INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    action       TEXT NOT NULL,
    destination  TEXT,
    ai_action    TEXT,
    rules        TEXT NOT NULL,
    reasons      TEXT NOT NULL,
    duplicate_of INTEGER,
    decided_at   TEXT NOT NULL
);

-- Decisions you made yourself; they take priority over every rule.
CREATE TABLE IF NOT EXISTS user_overrides (
    document_id INTEGER PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    action      TEXT NOT NULL,
    destination TEXT,
    note        TEXT,
    updated_at  TEXT NOT NULL
);

-- prompt_version should change whenever the prompt or output schema changes,
-- so stale classifications are not reused.
CREATE TABLE IF NOT EXISTS classification_cache (
    sha256              TEXT NOT NULL,
    model_identifier    TEXT NOT NULL,
    prompt_version      TEXT NOT NULL,
    classification_json TEXT NOT NULL,
    created_at          TEXT NOT NULL,
    PRIMARY KEY (sha256, model_identifier, prompt_version)
);

-- Extracted text, keyed by content hash so identical copies are read once.
-- Contains document text: keep the database out of version control.
CREATE TABLE IF NOT EXISTS extracted_text (
    sha256            TEXT PRIMARY KEY,
    extractor_version TEXT NOT NULL,
    status            TEXT NOT NULL,
    method            TEXT,
    page_count        INTEGER,
    char_count        INTEGER NOT NULL,
    truncated         INTEGER NOT NULL,
    text              TEXT NOT NULL,
    error             TEXT,
    extracted_at      TEXT NOT NULL
);

-- One row per Nemotron request, for cost and latency tracking. No content is stored.
CREATE TABLE IF NOT EXISTS api_calls (
    call_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256            TEXT,
    model             TEXT NOT NULL,
    status            TEXT NOT NULL,
    prompt_tokens     INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    created_at        TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scan_runs (
    run_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    root         TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    finished_at  TEXT NOT NULL,
    files_found  INTEGER NOT NULL,
    error_count  INTEGER NOT NULL,
    skipped_json TEXT NOT NULL
);
"""

# When size or modification time changes, the file's content may have changed,
# so the hash and downstream results are reset for reprocessing.
_UPSERT = """
INSERT INTO documents (
    original_path, filename, extension, size_bytes, created_at, modified_at,
    supported, present, first_seen_at, last_scanned_at, last_scan_run
) VALUES (?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
ON CONFLICT(original_path) DO UPDATE SET
    sha256 = CASE WHEN documents.size_bytes = excluded.size_bytes
                   AND documents.modified_at = excluded.modified_at
                  THEN documents.sha256 END,
    processing_error = CASE WHEN documents.size_bytes = excluded.size_bytes
                             AND documents.modified_at = excluded.modified_at
                            THEN documents.processing_error END,
    extraction_status = CASE WHEN documents.size_bytes = excluded.size_bytes
                              AND documents.modified_at = excluded.modified_at
                             THEN documents.extraction_status ELSE 'pending' END,
    classification_status = CASE WHEN documents.size_bytes = excluded.size_bytes
                                  AND documents.modified_at = excluded.modified_at
                                 THEN documents.classification_status ELSE 'pending' END,
    filename = excluded.filename,
    extension = excluded.extension,
    size_bytes = excluded.size_bytes,
    created_at = excluded.created_at,
    modified_at = excluded.modified_at,
    supported = excluded.supported,
    present = 1,
    last_scanned_at = excluded.last_scanned_at,
    last_scan_run = excluded.last_scan_run
"""


def connect(path: str | Path) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # WAL lets the dashboard read while a scheduled run writes; waits instead of failing when busy.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 30000")
    conn.executescript(SCHEMA)
    _migrate(conn)
    return conn


# Columns added after a table was first released: (table, column, definition).
_ADDED_COLUMNS = [
    ("documents", "processing_error", "TEXT"),
    ("documents", "classification_json", "TEXT"),
    ("file_operations", "batch_id", "TEXT"),
    ("file_operations", "error", "TEXT"),
    ("file_operations", "rolled_back_at", "TEXT"),
]


def _migrate(conn: sqlite3.Connection) -> None:
    for table, column, definition in _ADDED_COLUMNS:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
    conn.commit()


def save_scan(conn: sqlite3.Connection, result: ScanResult) -> int:
    """Store a scan and mark previously seen files under the same root that are now gone.

    Returns the scan run id.
    """
    with conn:
        run_id = conn.execute(
            "INSERT INTO scan_runs (root, started_at, finished_at, files_found, error_count, skipped_json)"
            " VALUES (?, ?, ?, ?, ?, ?)",
            (result.root, result.started_at, result.finished_at, len(result.records),
             len(result.errors), json.dumps(result.skipped)),
        ).lastrowid
        conn.executemany(
            _UPSERT,
            [
                (r.original_path, r.filename, r.extension, r.size_bytes, r.created_at,
                 r.modified_at, int(r.supported), result.started_at, result.started_at, run_id)
                for r in result.records
            ],
        )
        prefix = result.root.rstrip("\\/") + ("\\" if "\\" in result.root else "/")
        conn.execute(
            "UPDATE documents SET present = 0"
            " WHERE substr(original_path, 1, length(?)) = ? AND last_scan_run IS NOT ?",
            (prefix, prefix, run_id),
        )
    return run_id


def get_cached_text(conn: sqlite3.Connection, sha256: str) -> sqlite3.Row | None:
    return conn.execute("SELECT * FROM extracted_text WHERE sha256 = ?", (sha256,)).fetchone()


def store_text(conn: sqlite3.Connection, sha256: str, extractor_version: str, extraction, extracted_at: str) -> None:
    conn.execute(
        "INSERT OR REPLACE INTO extracted_text (sha256, extractor_version, status, method, page_count,"
        " char_count, truncated, text, error, extracted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (sha256, extractor_version, extraction.status, extraction.method, extraction.page_count,
         len(extraction.text), int(extraction.truncated), extraction.text, extraction.error, extracted_at),
    )


def replace_exact_duplicates(conn: sqlite3.Connection, groups: list[list[int]]) -> None:
    """Replace all exact-duplicate groups (other duplicate types are left alone)."""
    replace_groups(conn, "exact", [(group, 1.0) for group in groups])


def replace_groups(conn: sqlite3.Connection, kind: str, groups: list[tuple[list[int], float]]) -> None:
    """Replace every group of one kind ('exact' or 'similar')."""
    with conn:
        conn.execute("DELETE FROM duplicate_groups WHERE duplicate_type = ?", (kind,))
        next_id = conn.execute("SELECT COALESCE(MAX(group_id), 0) + 1 FROM duplicate_groups").fetchone()[0]
        conn.executemany(
            "INSERT INTO duplicate_groups (group_id, document_id, duplicate_type, similarity_score)"
            " VALUES (?, ?, ?, ?)",
            [(next_id + i, doc_id, kind, score) for i, (group, score) in enumerate(groups) for doc_id in group],
        )
