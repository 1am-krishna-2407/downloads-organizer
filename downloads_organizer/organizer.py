"""Approval-based, collision-safe, reversible file moves (Phase 5).

Nothing is ever deleted. ``plan_moves`` only proposes; ``apply_moves`` runs a
plan the user approved, logging every move before it happens and marking it
complete only after the destination is verified. ``rollback`` reverses a batch.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from downloads_organizer.config import Config
from downloads_organizer.hashing import sha256_file

log = logging.getLogger(__name__)

DEFAULT_ACTIONS = ("KEEP", "ARCHIVE")
MOVABLE_ACTIONS = ("KEEP", "ARCHIVE", "REVIEW", "CLEANUP_CANDIDATE")


@dataclass
class PlannedMove:
    document_id: int
    action: str
    source: Path
    destination: Path
    reason: str


@dataclass
class Skipped:
    document_id: int
    path: str
    reason: str


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _iso_mtime(st: os.stat_result) -> str:
    return datetime.fromtimestamp(st.st_mtime).astimezone().isoformat(timespec="seconds")


def _free_name(folder: Path, name: str, taken: set[str]) -> Path:
    """``name`` in ``folder``, or ``stem (2).ext``, ``stem (3).ext``... so nothing is overwritten."""
    candidate = folder / name
    stem, suffix = Path(name).stem, Path(name).suffix
    n = 2
    while candidate.exists() or os.path.normcase(str(candidate)) in taken:
        candidate = folder / f"{stem} ({n}){suffix}"
        n += 1
    return candidate


def plan_moves(
    conn: sqlite3.Connection,
    config: Config,
    actions: tuple[str, ...] = DEFAULT_ACTIONS,
    document_ids: set[int] | None = None,
) -> tuple[list[PlannedMove], list[Skipped]]:
    """Propose moves for decided documents. Read-only."""
    unknown = set(actions) - set(MOVABLE_ACTIONS)
    if unknown:
        raise ValueError(f"Unknown action(s): {', '.join(sorted(unknown))}")
    marks = ",".join("?" * len(actions))
    rows = conn.execute(
        f"""
        SELECT d.id, d.original_path, d.filename, d.size_bytes, d.modified_at,
               x.action, x.destination, x.reasons
        FROM decisions x JOIN documents d ON d.id = x.document_id
        WHERE d.present = 1 AND x.destination IS NOT NULL AND x.action IN ({marks})
        ORDER BY x.destination, d.original_path
        """,
        actions,
    ).fetchall()

    root = config.organized_dir
    moves, skipped, taken = [], [], set()
    for r in rows:
        if document_ids is not None and r["id"] not in document_ids:
            continue
        source = Path(r["original_path"])
        try:
            st = source.stat()
        except OSError:
            skipped.append(Skipped(r["id"], str(source), "File no longer exists; rescan"))
            continue
        if st.st_size != r["size_bytes"] or _iso_mtime(st) != r["modified_at"]:
            skipped.append(Skipped(r["id"], str(source), "File changed since it was analysed; run again"))
            continue
        destination = _free_name(root / r["destination"], r["filename"], taken)
        taken.add(os.path.normcase(str(destination)))
        reason = (json.loads(r["reasons"]) or [""])[0]
        moves.append(PlannedMove(r["id"], r["action"], source, destination, reason))
    return moves, skipped


def _move_verified(source: Path, destination: Path) -> None:
    """Move without overwriting; on another drive, copy and verify the hash before removing the source."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    size = source.stat().st_size
    try:
        os.rename(source, destination)  # same drive: atomic; on Windows it refuses to overwrite
    except OSError as exc:
        if getattr(exc, "winerror", None) != 17 and exc.errno != 18:  # not "different drive"
            raise
        digest = sha256_file(source)
        shutil.copy2(source, destination)
        if sha256_file(destination) != digest:
            destination.unlink()
            raise OSError("Copy verification failed; original left in place")
        source.unlink()
    if not destination.exists() or destination.stat().st_size != size or source.exists():
        raise OSError("Move could not be verified")


def apply_moves(conn: sqlite3.Connection, moves: list[PlannedMove]) -> tuple[str, int, list[tuple[PlannedMove, str]]]:
    """Run approved moves. Returns (batch_id, moved, failures)."""
    batch_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    moved, failures = 0, []
    for move in moves:
        op_id = conn.execute(
            "INSERT INTO file_operations (document_id, source_path, destination_path, operation_type,"
            " approved_at, status, batch_id) VALUES (?, ?, ?, 'move', ?, 'pending', ?)",
            (move.document_id, str(move.source), str(move.destination), _now(), batch_id),
        ).lastrowid
        conn.commit()  # recorded before the move, so a crash mid-move is still traceable
        try:
            _move_verified(move.source, move.destination)
        except OSError as exc:
            error = f"{type(exc).__name__}: {exc}"
            log.warning("Move failed for %s: %s", move.source, error)
            conn.execute("UPDATE file_operations SET status = 'failed', error = ? WHERE operation_id = ?",
                         (error, op_id))
            failures.append((move, error))
        else:
            conn.execute("UPDATE file_operations SET status = 'completed', completed_at = ? WHERE operation_id = ?",
                         (_now(), op_id))
            conn.execute("UPDATE documents SET present = 0 WHERE id = ?", (move.document_id,))
            moved += 1
            log.info("Moved %s -> %s", move.source, move.destination)
        conn.commit()
    return batch_id, moved, failures


def batches(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return conn.execute(
        """
        SELECT batch_id, MIN(approved_at) AS started,
               SUM(status = 'completed') AS completed, SUM(status = 'failed') AS failed,
               SUM(status = 'rolled_back') AS rolled_back
        FROM file_operations WHERE operation_type = 'move' AND batch_id IS NOT NULL
        GROUP BY batch_id ORDER BY batch_id DESC
        """
    ).fetchall()


def rollback(conn: sqlite3.Connection, config: Config, batch_id: str) -> tuple[int, list[tuple[str, str]]]:
    """Move a batch's files back to where they came from. Never overwrites."""
    ops = conn.execute(
        "SELECT * FROM file_operations WHERE batch_id = ? AND status = 'completed' ORDER BY operation_id DESC",
        (batch_id,),
    ).fetchall()
    restored, problems = 0, []
    for op in ops:
        source, destination = Path(op["source_path"]), Path(op["destination_path"])
        if not destination.exists():
            problems.append((str(destination), "No longer in the organized folder"))
            continue
        if source.exists():
            problems.append((str(source), "Something else now exists at the original location"))
            continue
        try:
            _move_verified(destination, source)
        except OSError as exc:
            problems.append((str(destination), f"{type(exc).__name__}: {exc}"))
            continue
        conn.execute("UPDATE file_operations SET status = 'rolled_back', rolled_back_at = ? WHERE operation_id = ?",
                     (_now(), op["operation_id"]))
        conn.execute("UPDATE documents SET present = 1 WHERE id = ?", (op["document_id"],))
        conn.commit()
        _remove_empty_parents(destination.parent, config.organized_dir)
        restored += 1
    return restored, problems


def _remove_empty_parents(folder: Path, stop_at: Path) -> None:
    """Tidy folders the organizer created and emptied. Only removes empty directories."""
    stop_at = stop_at.resolve()
    folder = folder.resolve()
    while folder != stop_at and stop_at in folder.parents:
        try:
            folder.rmdir()  # fails (harmlessly) if not empty
        except OSError:
            return
        folder = folder.parent
