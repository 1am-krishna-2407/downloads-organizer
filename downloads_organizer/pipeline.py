"""Phase 2 processing: hash files, extract document text, and group exact duplicates.

Reads files only; never modifies, moves, or deletes them. Work is committed in
batches, so an interrupted run keeps what it finished and resumes from there.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from typing import Callable, Iterator

from downloads_organizer import classifier, database, extractors, similarity
from downloads_organizer.config import Config
from downloads_organizer.hashing import group_exact_duplicates, sha256_file

log = logging.getLogger(__name__)

NOT_APPLICABLE = "not_applicable"  # not a document type we read
COMMIT_EVERY = 25


@dataclass
class ProcessStats:
    hashed: int = 0
    extracted: int = 0
    extraction_cache_hits: int = 0
    errors: int = 0
    statuses: Counter = field(default_factory=Counter)
    duplicate_groups: int = 0
    duplicate_files: int = 0
    deferred: int = 0
    similar_groups: int = 0


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _cache_usable(cached: sqlite3.Row | None, ocr_now_available: bool) -> bool:
    if cached is None or cached["extractor_version"] != extractors.EXTRACTOR_VERSION:
        return False
    # Retry scanned documents once Tesseract has been installed.
    return not (cached["status"] == extractors.NEEDS_OCR and ocr_now_available)


def process_documents(
    conn: sqlite3.Connection,
    config: Config,
    progress: Callable[[int, int], None] | None = None,
) -> ProcessStats:
    stats = ProcessStats()
    settings = config.extraction
    ocr_now_available = extractors.ocr_available(settings)

    rows = conn.execute(
        "SELECT id, original_path, size_bytes, modified_at, supported, sha256, extraction_status"
        " FROM documents WHERE present = 1"
    ).fetchall()
    now = datetime.now().astimezone().timestamp()
    # Only files sharing a size can be identical, so other non-documents need no hash.
    size_counts = Counter(r["size_bytes"] for r in rows if r["size_bytes"] > 0)

    def needs_work(r: sqlite3.Row) -> bool:
        if r["supported"]:
            return (
                r["sha256"] is None
                or r["extraction_status"] == "pending"
                or (r["extraction_status"] == extractors.NEEDS_OCR and ocr_now_available)
            )
        return r["extraction_status"] == "pending" or (
            r["sha256"] is None and size_counts[r["size_bytes"]] > 1
        )

    def settled(r: sqlite3.Row) -> bool:
        # Only recent changes wait; a timestamp in the future (clock skew, unzipped files) never settles.
        age = now - datetime.fromisoformat(r["modified_at"]).timestamp()
        return not 0 <= age < config.automation.min_file_age_seconds

    candidates = [r for r in rows if needs_work(r)]
    todo = [r for r in candidates if settled(r)]
    stats.deferred = len(candidates) - len(todo)

    # Phase 1 (this thread): hash files and reuse cached text where possible.
    waiting: dict[str, list[int]] = {}  # content hash -> documents waiting for its text
    paths: dict[str, str] = {}
    for done, row in enumerate(todo, start=1):
        digest = _process_one(conn, row, size_counts, ocr_now_available, stats)
        if digest:
            waiting.setdefault(digest, []).append(row["id"])
            paths.setdefault(digest, row["original_path"])
        if done % COMMIT_EVERY == 0:
            conn.commit()
    conn.commit()

    # Phase 2 (worker processes): extract each distinct content once. OCR is CPU-heavy.
    for done, (digest, result) in enumerate(_extract_all(paths, settings), start=1):
        database.store_text(conn, digest, extractors.EXTRACTOR_VERSION, result, _now())
        stats.extracted += 1
        stats.errors += result.status == extractors.ERROR
        conn.executemany(
            "UPDATE documents SET sha256 = ?, extraction_status = ?, processing_error = ? WHERE id = ?",
            [(digest, result.status, result.error, doc_id) for doc_id in waiting[digest]],
        )
        if done % COMMIT_EVERY == 0:
            conn.commit()
        if progress:
            progress(done, len(paths))
    conn.commit()

    for status, count in conn.execute(
        "SELECT extraction_status, COUNT(*) FROM documents WHERE present = 1 GROUP BY 1"
    ):
        stats.statuses[status] = count

    # Zero-byte files all share one hash; they are not meaningful duplicates.
    groups = group_exact_duplicates(
        conn.execute(
            "SELECT id, sha256 FROM documents WHERE present = 1 AND size_bytes > 0 AND sha256 IS NOT NULL"
        )
    )
    database.replace_exact_duplicates(conn, groups)
    stats.duplicate_groups = len(groups)
    stats.duplicate_files = sum(len(g) for g in groups)
    stats.similar_groups = _find_similar(conn, config.decisions.similar_threshold)
    log.info(
        "Processed %d files: %d hashed, %d extracted, %d cached, %d errors, %d duplicate groups",
        len(todo), stats.hashed, stats.extracted, stats.extraction_cache_hits, stats.errors, len(groups),
    )
    return stats


def _find_similar(conn: sqlite3.Connection, threshold: float) -> int:
    """Group documents whose text is largely the same but whose bytes differ (versions, re-exports)."""
    rows = conn.execute(
        """SELECT d.id, d.sha256, t.text FROM documents d JOIN extracted_text t ON t.sha256 = d.sha256
           WHERE d.present = 1 AND d.supported = 1 AND d.extraction_status = 'ok'"""
    ).fetchall()
    by_sha: dict[str, list[int]] = {}
    texts: dict[int, str] = {}
    for r in rows:
        if r["sha256"] not in by_sha:
            texts[r["id"]] = r["text"]  # one representative per identical copy set
        by_sha.setdefault(r["sha256"], []).append(r["id"])
    sha_of = {r["id"]: r["sha256"] for r in rows}
    groups = [
        ([doc_id for rep in ids for doc_id in by_sha[sha_of[rep]]], score)
        for ids, score in similarity.similar_groups(texts, threshold)
    ]
    database.replace_groups(conn, "similar", groups)
    return len(groups)


PARALLEL_MIN_JOBS = 8  # below this, starting worker processes costs more than it saves


def extraction_workers(settings) -> int:
    if settings.workers:
        return max(1, settings.workers)
    return max(1, min(4, (os.cpu_count() or 2) - 1))


def _extract_all(paths: dict[str, str], settings) -> Iterator[tuple[str, extractors.Extraction]]:
    """Yield (hash, extraction) for each path, in worker processes when it's worth it."""
    workers = extraction_workers(settings)
    remaining = dict(paths)
    if workers > 1 and len(remaining) >= PARALLEL_MIN_JOBS:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(extractors.extract, path, settings): digest
                           for digest, path in remaining.items()}
                for future in as_completed(futures):
                    digest = futures[future]
                    result = future.result()
                    del remaining[digest]
                    yield digest, result
        except BrokenProcessPool:
            # A worker crashed (e.g. a malformed file broke a native library); finish here instead.
            log.warning("An extraction worker stopped unexpectedly; continuing in the main process")
    for digest, path in list(remaining.items()):
        yield digest, extractors.extract(path, settings)


def _process_one(conn, row, size_counts, ocr_now_available, stats: ProcessStats) -> str | None:
    """Hash a file and settle it from cache. Returns its hash if its text still needs extracting."""
    path = row["original_path"]
    digest = row["sha256"]

    if digest is None and (row["supported"] or size_counts[row["size_bytes"]] > 1):
        try:
            digest = sha256_file(path)
            stats.hashed += 1
        except OSError as exc:
            log.warning("Cannot hash %s: %s", path, type(exc).__name__)
            stats.errors += 1
            conn.execute(
                "UPDATE documents SET extraction_status = ?, processing_error = ? WHERE id = ?",
                (extractors.ERROR, f"{type(exc).__name__}: {exc.strerror or exc}", row["id"]),
            )
            return None

    if not row["supported"]:
        conn.execute(
            "UPDATE documents SET sha256 = ?, extraction_status = ?, processing_error = NULL WHERE id = ?",
            (digest, NOT_APPLICABLE, row["id"]),
        )
        return None

    cached = database.get_cached_text(conn, digest)
    if not _cache_usable(cached, ocr_now_available):
        return digest
    stats.extraction_cache_hits += 1
    conn.execute(
        "UPDATE documents SET sha256 = ?, extraction_status = ?, processing_error = ? WHERE id = ?",
        (digest, cached["status"], cached["error"], row["id"]),
    )
    return None


# --- Phase 3: classification ----------------------------------------------------


@dataclass
class ClassifyStats:
    candidates: int = 0           # unique documents (by content) awaiting classification
    classified: int = 0
    cache_hits: int = 0
    invalid_output: int = 0
    errors: int = 0
    escalated: int = 0
    api_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0
    estimated_prompt_tokens: int = 0
    actions: Counter = field(default_factory=Counter)


MAX_CONSECUTIVE_ERRORS = 5


def _classification_targets(conn: sqlite3.Connection, retry_failed: bool) -> list[sqlite3.Row]:
    """One row per unique content hash that still needs a classification."""
    statuses = ("pending", "error", "invalid_output") if retry_failed else ("pending", "error")
    marks = ",".join("?" * len(statuses))
    return conn.execute(
        f"""
        SELECT d.sha256, MIN(d.filename) AS filename, MIN(d.extension) AS extension,
               MIN(d.size_bytes) AS size_bytes, MAX(d.modified_at) AS modified_at,
               COUNT(*) AS copies, t.text, t.page_count, t.truncated
        FROM documents d JOIN extracted_text t ON t.sha256 = d.sha256
        WHERE d.present = 1 AND d.supported = 1 AND d.extraction_status = 'ok'
          AND d.classification_status IN ({marks})
        GROUP BY d.sha256
        ORDER BY MIN(d.original_path)
        """,
        statuses,
    ).fetchall()


def _document_input(row: sqlite3.Row) -> classifier.DocumentInput:
    return classifier.DocumentInput(
        filename=row["filename"], extension=row["extension"], size_bytes=row["size_bytes"],
        modified_at=row["modified_at"], text=row["text"], page_count=row["page_count"],
        truncated=bool(row["truncated"]), identical_copies=row["copies"] - 1,
    )


def _apply(conn: sqlite3.Connection, sha256: str, status: str, payload: dict | None) -> None:
    if payload is None:
        conn.execute(
            "UPDATE documents SET classification_status = ? WHERE sha256 = ? AND present = 1",
            (status, sha256),
        )
        return
    c = payload["classification"]
    conn.execute(
        "UPDATE documents SET classification_status = ?, document_type = ?, category = ?, summary = ?,"
        " importance = ?, recommended_action = ?, confidence = ?, classification_json = ?"
        " WHERE sha256 = ? AND present = 1",
        (status, c["document_type"], c["category"], c["summary"], c["importance"],
         c["recommended_action"], c["confidence"], json.dumps(payload), sha256),
    )


def classify_documents(
    conn: sqlite3.Connection,
    config: Config,
    doc_classifier: classifier.DocumentClassifier | None,
    limit: int | None = None,
    dry_run: bool = False,
    retry_failed: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> ClassifyStats:
    """Classify each unique document once. With ``dry_run`` nothing is sent; only estimates are made."""
    settings = config.classification
    version = classifier.prompt_version(settings)
    stats = ClassifyStats()
    targets = _classification_targets(conn, retry_failed)
    stats.candidates = len(targets)

    # Reuse cached results first; they cost nothing.
    remaining = []
    for row in targets:
        cached = conn.execute(
            "SELECT classification_json FROM classification_cache"
            " WHERE sha256 = ? AND model_identifier = ? AND prompt_version = ?",
            (row["sha256"], settings.model, version),
        ).fetchone()
        if cached and not dry_run:
            payload = json.loads(cached["classification_json"])
            _apply(conn, row["sha256"], "done", payload)
            stats.cache_hits += 1
            stats.actions[payload["classification"]["recommended_action"]] += 1
        elif not cached:
            remaining.append(row)
    conn.commit()

    if limit is not None:
        remaining = remaining[:limit]
    if dry_run:
        for row in remaining:
            messages = classifier.build_messages(_document_input(row), settings)
            stats.estimated_prompt_tokens += sum(len(m["content"]) for m in messages) // 4
        stats.classified = len(remaining)
        return stats

    # Requests run in worker threads; every database write happens here, on this thread.
    consecutive_errors = 0
    workers = max(1, settings.concurrency)
    pool = ThreadPoolExecutor(max_workers=workers)
    futures = {pool.submit(doc_classifier.classify, _document_input(row)): row["sha256"] for row in remaining}
    try:
        for done, future in enumerate(as_completed(futures), start=1):
            sha = futures[future]
            try:
                outcome = future.result()
            except classifier.FatalApiError:
                conn.commit()
                raise
            except classifier.ApiError as exc:
                log.warning("Classification failed for %s: %s", sha[:12], exc)
                _apply(conn, sha, "error", None)
                conn.commit()
                stats.errors += 1
                consecutive_errors += 1
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    raise classifier.ApiError(
                        f"Stopped after {MAX_CONSECUTIVE_ERRORS} failures in a row; last error: {exc}"
                    ) from exc
                continue
            finally:
                if progress:
                    progress(done, len(remaining))
            consecutive_errors = 0
            _record_outcome(conn, sha, outcome, settings.model, version, stats)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    log.info(
        "Classification: %d classified, %d cached, %d invalid, %d errors, %d calls, %d+%d tokens",
        stats.classified, stats.cache_hits, stats.invalid_output, stats.errors, stats.api_calls,
        stats.prompt_tokens, stats.completion_tokens,
    )
    return stats


def _record_outcome(conn, sha: str, outcome, cache_model: str, version: str, stats: ClassifyStats) -> None:
    now = _now()
    for call in outcome.calls:
        conn.execute(
            "INSERT INTO api_calls (sha256, model, status, prompt_tokens, completion_tokens, latency_ms,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sha, call.model, call.status, call.prompt_tokens, call.completion_tokens, call.latency_ms, now),
        )
        stats.api_calls += 1
        stats.prompt_tokens += call.prompt_tokens
        stats.completion_tokens += call.completion_tokens
        stats.latency_ms += call.latency_ms
    stats.escalated += outcome.escalated

    if outcome.status == "done":
        payload = {
            "classification": outcome.classification.model_dump(),
            "model_used": outcome.model,
            "prompt_version": version,
        }
        conn.execute(
            "INSERT OR REPLACE INTO classification_cache"
            " (sha256, model_identifier, prompt_version, classification_json, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (sha, cache_model, version, json.dumps(payload), now),
        )
        _apply(conn, sha, "done", payload)
        stats.classified += 1
        stats.actions[outcome.classification.recommended_action] += 1
    else:
        log.warning("Invalid model output for %s: %s", sha[:12], outcome.error)
        _apply(conn, sha, "invalid_output", None)
        stats.invalid_output += 1
    conn.commit()
