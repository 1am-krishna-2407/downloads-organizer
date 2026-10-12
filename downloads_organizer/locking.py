"""One run at a time: a scheduled run, `watch`, the CLI and the dashboard share one database.

Uses an OS-level file lock, which the operating system releases automatically if
the process dies, so a crash can never leave a stale lock behind.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from pathlib import Path


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def run_lock(data_dir: Path, what: str = "run"):
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "organizer.lock"
    fh = open(path, "a+")
    try:
        try:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise AlreadyRunning(
                "Another Downloads Organizer run is in progress (scheduled run, 'watch' or the dashboard). "
                "Try again when it finishes."
            ) from None
        fh.seek(0)
        fh.truncate()
        fh.write(f"{os.getpid()} {what}\n")
        fh.flush()
        try:
            yield
        finally:
            if sys.platform == "win32":
                import msvcrt

                fh.seek(0)
                try:
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except OSError:
                    pass
    finally:
        fh.close()
