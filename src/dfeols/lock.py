# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Ted Matherly
"""Directory locks on a shared filesystem, so two jobs never write the same
data or shard folder at once (the second would delete files under the first,
or the first would write into the second's fresh folder).

The lock is a sibling file ``<dir>.lock`` (outside the folder, so an
overwrite cannot delete it). It is written in full to a temporary file and
hard-linked into place, which fails if the lock exists and works on NFS and
Lustre, so no one ever reads a half-written lock (O_EXCL is the fallback where
hard links are not supported). Its holder refreshes the file's mtime every
``heartbeat`` seconds from a background thread. A lock is taken over when its
mtime is older than ``stale`` seconds (the holder died, on any node) or when
its holder ran on this host and that process no longer exists.

A takeover renames the stale lock aside before deleting it, and deletes it only
if it is the file (inode and mtime) that was judged stale. So two jobs that
both find the same stale lock cannot delete each other's fresh one
(regression: on OSCER a second job, started the same second, unlinked the
first's new lock as stale and crashed when the first rebuilt the folder).
"""

from __future__ import annotations

import errno
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
        self._ino = 0

    def _try_create(self) -> bool:
        tmp = self.path.with_name(f"{self.path.name}.{self._me['token']}.tmp")
        tmp.write_text(json.dumps(self._me))
        try:
            os.link(tmp, self.path)
        except FileExistsError:
            return False
        except OSError as e:
            if e.errno in (errno.EPERM, errno.ENOTSUP, errno.EOPNOTSUPP, errno.EXDEV, errno.EMLINK):
                return self._try_create_excl()  # no hard links on this filesystem
            if os.stat(tmp).st_nlink != 2:  # NFS can report an error for a link it made
                raise
        finally:
            tmp.unlink(missing_ok=True)
        self._ino = os.stat(self.path).st_ino
        return True

    def _try_create_excl(self) -> bool:
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            return False
        with os.fdopen(fd, "w") as f:
            json.dump(self._me, f)
            self._ino = os.fstat(f.fileno()).st_ino
        return True

    def _remove_stale(self, judged: os.stat_result) -> None:
        """Delete the lock that was judged stale, and only that file: if another job
        replaced it in the meantime, put the new one back."""
        aside = self.path.with_name(f"{self.path.name}.stale-{self._me['token']}")
        try:
            os.rename(self.path, aside)
        except FileNotFoundError:
            return  # another job removed it first
        st = os.stat(aside)
        if (st.st_ino, st.st_mtime_ns) != (judged.st_ino, judged.st_mtime_ns):
            try:  # a fresh lock: restore it (fails only if a third job locked meanwhile)
                os.link(aside, self.path)
            except OSError as e:
                log.error("lock %s: could not restore another job's fresh lock (%s)", self.path, e)
        aside.unlink(missing_ok=True)

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
        for _ in range(5):
            if self._try_create():
                break
            try:
                st = self.path.stat()
                info = json.loads(self.path.read_text() or "{}")
            except FileNotFoundError:
                continue  # released just now: try again
            except ValueError:
                info = {}  # an O_EXCL lock being written, or corrupt: judge by age
            age = time.time() - st.st_mtime
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
            log.warning("removing a stale lock %s (%s, %.0fs old)", self.path, info or "unreadable", age)
            self._remove_stale(st)
        else:
            raise LockedError(f"{self.path}: another job took this lock just now")
        self._stop.clear()
        self._thread = threading.Thread(target=self._beat, name="dfeols-lock", daemon=True)
        self._thread.start()
        return self

    def _beat(self):
        while not self._stop.wait(self.heartbeat):
            try:
                if os.stat(self.path).st_ino != self._ino:
                    raise FileNotFoundError
                os.utime(self.path)
            except FileNotFoundError:
                log.error("lock %s was removed or replaced while held", self.path)
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
