# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Directory locks on a shared filesystem, so two jobs never write the same
data or shard folder at once (the second would delete files under the first,
or the first would write into the second's fresh folder).

The lock is a sibling file ``<dir>.lock`` (outside the folder, so an
overwrite cannot delete it), created atomically with O_EXCL, which works on
NFS and Lustre. Its holder refreshes the file's mtime every ``heartbeat``
seconds from a background thread. A lock is taken over when its mtime is
older than ``stale`` seconds (the holder died, on any node) or when its
holder ran on this host and that process no longer exists.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
import uuid
from pathlib import Path

log = logging.getLogger("dfeols")


class LockedError(RuntimeError):
    """Another job holds the lock on this folder."""


def _job_id() -> str:
    for k in ("SLURM_JOB_ID", "PBS_JOBID", "JOB_ID", "LSB_JOBID"):
        if os.environ.get(k):
            return os.environ[k]
    return ""


def _holder() -> dict:
    return {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "job": _job_id(),
        "since": time.strftime("%Y-%m-%d %H:%M:%S"),
        "token": uuid.uuid4().hex,  # identifies this holder, even within one process
    }


class DirLock:
    """Exclusive claim on a folder; use as a context manager or acquire()/release()."""

    def __init__(self, directory, *, heartbeat: float = 30.0, stale: float = 300.0, what: str = "folder"):
        d = Path(directory)
        self.path = d.parent / f"{d.name}.lock"
        self.heartbeat, self.stale, self.what = heartbeat, stale, what
        self._me = _holder()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _try_create(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump(self._me, f)
        return True

    def _is_stale(self, info: dict, age: float) -> bool:
        if age > self.stale:  # holder stopped refreshing: it died, wherever it ran
            return True
        if info.get("host") == socket.gethostname() and info.get("pid"):
            try:
                os.kill(int(info["pid"]), 0)
            except ProcessLookupError:
                return True
            except (PermissionError, ValueError, OverflowError):
                return False
        return False

    def acquire(self) -> DirLock:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            if self._try_create():
                break
            try:
                age = time.time() - self.path.stat().st_mtime
                info = json.loads(self.path.read_text() or "{}")
            except FileNotFoundError:
                continue  # released just now: try again
            except ValueError:
                info, age = {}, time.time() - self.path.stat().st_mtime  # half-written: judge by age
            if not self._is_stale(info, age):
                who = (
                    ", ".join(f"{k} {v}" for k, v in info.items() if v and k != "token") or "another process"
                )
                raise LockedError(
                    f"{self.path.with_suffix('')} is in use by {who} (lock refreshed {age:.0f}s ago). "
                    f"Two jobs must not prepare the same {self.what} at once: wait for that job, or "
                    f"cancel it. If it is gone, the lock goes stale after {self.stale:.0f}s, or delete "
                    f"{self.path}."
                )
            log.warning("removing a stale lock %s (%s)", self.path, info or "unreadable")
            self.path.unlink(missing_ok=True)
        else:
            raise LockedError(f"{self.path}: another job took this lock just now")
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name="dfeols-lock", daemon=True)
        self._thread.start()
        return self

    def _beat(self):
        while not self._stop.wait(self.heartbeat):
            try:
                os.utime(self.path)
            except FileNotFoundError:
                log.warning("lock %s disappeared while held", self.path)
                return

    def release(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        try:  # only remove our own lock (a stale-lock takeover may have replaced it)
            if json.loads(self.path.read_text()).get("token") == self._me["token"]:
                self.path.unlink(missing_ok=True)
        except (FileNotFoundError, ValueError):
            pass

    def __enter__(self) -> DirLock:
        return self.acquire()

    def __exit__(self, *exc):
        self.release()
