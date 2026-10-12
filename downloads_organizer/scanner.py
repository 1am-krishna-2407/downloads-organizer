"""Read-only file scanner (Phase 1).

Walks the Downloads folder and records file metadata. It only calls ``stat``;
it never opens, modifies, moves, or deletes a file.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from downloads_organizer.config import Config

log = logging.getLogger(__name__)


@dataclass
class FileRecord:
    original_path: str
    filename: str
    extension: str
    size_bytes: int
    created_at: str | None
    modified_at: str
    supported: bool


@dataclass
class ScanError:
    path: str
    error: str


@dataclass
class ScanResult:
    root: str
    started_at: str
    finished_at: str = ""
    records: list[FileRecord] = field(default_factory=list)
    errors: list[ScanError] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    # Folders left out as a whole: (path, "software" | "project")
    skipped_folders: list[tuple[str, str]] = field(default_factory=list)


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="seconds")


def _key(path: Path | str) -> str:
    """Comparison key for paths (case-insensitive on Windows)."""
    return os.path.normcase(os.path.abspath(path))


def _is_link(path: Path) -> bool:
    # Path.is_junction() only exists on Python 3.12+.
    return path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)())


SOFTWARE_DIR_NAMES = {"node_modules", ".venv", "venv", "site-packages", "__pycache__", ".git"}
EXECUTABLE_EXTENSIONS = (".exe", ".msi", ".com")
LIBRARY_EXTENSIONS = (".dll", ".so", ".dylib", ".pyd")
SOFTWARE_LAYOUT_DIRS = {"bin", "lib", "lib64", "plugins", "platforms", "locales", "usr", "mingw64",
                        "mingw32", "x64", "x86", "runtime", "jre"}


def _has_binaries(folder: str) -> bool:
    try:
        with os.scandir(folder) as entries:
            return any(e.name.lower().endswith(EXECUTABLE_EXTENSIONS + LIBRARY_EXTENSIONS) for e in entries)
    except OSError:
        return False


def folder_kind(folder: Path, config: Config, _depth: int = 0) -> str | None:
    """"project" for a code project, "software" for an unpacked application, else None.

    Files in such folders belong together and are not personal documents, so the
    organizer neither reads nor moves them. Uses directory listings only.
    """
    name = folder.name.lower()
    if config.skip_software_folders and (name in SOFTWARE_DIR_NAMES or name.endswith(".app")):
        return "software"
    try:
        with os.scandir(folder) as it:
            entries = list(it)
    except OSError:
        return None
    files = [e.name.lower() for e in entries if e.is_file(follow_symlinks=False)]
    dirs = [e for e in entries if e.is_dir(follow_symlinks=False)]
    names = {e.name.lower() for e in entries}

    if config.skip_project_folders:
        settings = config.decisions
        markers = [m.lower() for m in settings.project_markers]
        if (
            any(m in names for m in markers)
            or any(f.endswith(m) for f in files for m in markers if m.startswith(".") and m.count(".") == 1)
            or any(re.search(p, folder.name, re.IGNORECASE) for p in settings.project_name_patterns)
        ):
            return "project"

    # A zip often unpacks into a folder holding only one folder; judge by what's inside.
    if not files and len(dirs) == 1 and _depth < 2:
        return folder_kind(Path(dirs[0].path), config, _depth + 1)

    if config.skip_software_folders and files and all(f.endswith(EXECUTABLE_EXTENSIONS + LIBRARY_EXTENSIONS)
                                                      for f in files):
        return "software"
    if config.skip_software_folders and any(f.endswith(EXECUTABLE_EXTENSIONS) for f in files):
        if (
            any(f.endswith(LIBRARY_EXTENSIONS) for f in files)
            or any(d.name.lower() in SOFTWARE_LAYOUT_DIRS for d in dirs)
            or any(_has_binaries(d.path) for d in dirs)
        ):
            return "software"
    return None


def _describe(exc: OSError) -> str:
    return f"{type(exc).__name__}: {exc.strerror or exc}"


def skip_reason(path: Path, config: Config) -> str | None:
    """Why a file should be left out of the inventory, or None to include it."""
    name = path.name.lower()
    if name in config.skip_names:
        return "system_file"
    if name.startswith(config.skip_name_prefixes):
        return "temp_or_lock_file"
    if path.suffix.lower() in config.skip_extensions:
        return "incomplete_download"
    return None


def scan(config: Config, root: str | os.PathLike | None = None) -> ScanResult:
    """Inventory every file under ``root`` (default: the configured Downloads folder).

    Errors on individual files or folders are collected in the result instead
    of stopping the scan.
    """
    root_path = Path(root or config.downloads_dir).resolve()
    if not root_path.is_dir():
        raise NotADirectoryError(f"Scan root does not exist or is not a folder: {root_path}")

    result = ScanResult(root=str(root_path), started_at=_now())
    skipped: Counter[str] = Counter()
    excluded = {_key(p) for p in config.excluded_dirs()}

    def on_walk_error(exc: OSError) -> None:
        path = exc.filename or str(root_path)
        log.warning("Cannot list folder %s: %s", path, _describe(exc))
        result.errors.append(ScanError(path=str(path), error=_describe(exc)))

    for dirpath, dirnames, filenames in os.walk(root_path, onerror=on_walk_error):
        current = Path(dirpath)

        kept = []
        if config.recursive:
            for name in dirnames:
                sub = current / name
                if _key(sub) in excluded:
                    skipped["excluded_folder"] += 1
                elif _is_link(sub):
                    skipped["linked_folder"] += 1
                elif kind := folder_kind(sub, config):
                    skipped[f"{kind}_folder"] += 1
                    result.skipped_folders.append((str(sub), kind))
                else:
                    kept.append(name)
        dirnames[:] = sorted(kept)

        for name in sorted(filenames):
            path = current / name
            reason = skip_reason(path, config)
            if reason:
                skipped[reason] += 1
                continue
            try:
                st = path.stat(follow_symlinks=False)
            except OSError as exc:
                log.warning("Cannot read metadata for %s: %s", path, _describe(exc))
                result.errors.append(ScanError(path=str(path), error=_describe(exc)))
                continue
            if stat.S_ISLNK(st.st_mode):
                skipped["linked_file"] += 1
                continue
            if not stat.S_ISREG(st.st_mode):
                skipped["not_a_regular_file"] += 1
                continue

            extension = path.suffix.lower()
            birthtime = getattr(st, "st_birthtime", None)
            result.records.append(
                FileRecord(
                    original_path=str(path),
                    filename=name,
                    extension=extension,
                    size_bytes=st.st_size,
                    created_at=_iso(birthtime) if birthtime else None,
                    modified_at=_iso(st.st_mtime),
                    supported=extension in config.supported_extensions,
                )
            )

    result.skipped = dict(skipped)
    result.finished_at = _now()
    log.info(
        "Scanned %s: %d files, %d errors, %d skipped",
        root_path, len(result.records), len(result.errors), sum(skipped.values()),
    )
    return result
