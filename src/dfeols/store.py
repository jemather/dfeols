"""The persisted demeaned data: list it, remove it, and check scratch space for it.

Layout (docs/design/sessions.md): <demeaned>/<dataset>/<key>/ with shards/part-*.parquet and
manifest.json (written last, so a manifest means complete), built under the DirLock
<dataset>/<key>.lock. Removal is explicit (planner.md, Ted 2026-10-28): `dfeols demeaned rm`
deletes a key unless a running session uses it or a build holds its lock.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from .lock import DirLock, LockedError, holder

HISTORY = ("timings.jsonl", "passes.jsonl")  # the estimates' history, beside the datasets


def _parse_time(s: str | None) -> _dt.datetime | None:
    if not s:
        return None
    try:
        t = _dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=_dt.timezone.utc)


def _tree_bytes(p: Path) -> int:
    n = 0
    for dirpath, _, files in os.walk(p):
        for f in files:
            try:
                n += os.lstat(os.path.join(dirpath, f)).st_size
            except FileNotFoundError:
                pass
    return n


def live_users(sessions: Path | None) -> dict[str, list[str]]:
    """Demeaned folders (resolved) used by a running session -> the sessions' names. A session
    is running while it holds the lock on its folder."""
    out: dict[str, list[str]] = {}
    if not sessions or not Path(sessions).is_dir():
        return out
    for sdir in sorted(Path(sessions).iterdir()):
        st = sdir / "session.json"
        if not sdir.is_dir() or not st.exists():
            continue
        try:
            dm = json.loads(st.read_text()).get("demeaned")
        except (OSError, ValueError):
            continue
        if dm and holder(sdir) is not None:
            out.setdefault(str(Path(dm).resolve()), []).append(sdir.name)
    return out


def entries(root, sessions=None, purge_days: int | None = None) -> list[dict]:
    """Every <dataset>/<key> folder under `root`: complete or not, its size, when it was
    written, when the site's purge may take it, whether a session uses it or a build holds it."""
    from .session import DEMEANED_FORMAT

    root = Path(root)
    users = live_users(Path(sessions) if sessions else None)
    out = []
    if not root.is_dir():
        return out
    for dsd in sorted(p for p in root.iterdir() if p.is_dir()):
        for kd in sorted(p for p in dsd.iterdir() if p.is_dir()):
            if ".removing-" in kd.name:
                continue
            mf = kd / "manifest.json"
            m = {}
            if mf.exists():
                try:
                    m = json.loads(mf.read_text())
                except (OSError, ValueError):
                    m = {}
            written = _parse_time(m.get("created")) or _dt.datetime.fromtimestamp(
                kd.stat().st_mtime, _dt.timezone.utc
            )
            e = {
                "dataset": dsd.name,
                "key": kd.name,
                "path": str(kd),
                "complete": bool(m),
                "format": m.get("format"),
                "current": m.get("format") == DEMEANED_FORMAT,
                "bytes": _tree_bytes(kd),
                "written": written.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "rows": m.get("rows"),
                "variables": m.get("variables"),
                "fixed_effects": [f.get("name") for f in m.get("fixed_effects", [])],
                "purge_after": (
                    (written + _dt.timedelta(days=purge_days)).strftime("%Y-%m-%d") if purge_days else None
                ),
                "sessions": users.get(str(kd.resolve()), []),
                "building": holder(kd),
            }
            out.append(e)
    return out


def find(root, key: str, sessions=None) -> dict:
    """The one entry whose key starts with `key` (or `<dataset>/<key>`)."""
    ds, _, k = key.rpartition("/")
    hits = [e for e in entries(root, sessions) if e["key"].startswith(k) and (not ds or e["dataset"] == ds)]
    if not hits:
        raise KeyError(f"no demeaned data with key {key!r} under {root}")
    if len(hits) > 1:
        names = ", ".join(f"{e['dataset']}/{e['key']}" for e in hits)
        raise KeyError(f"key {key!r} matches {len(hits)} folders ({names}): give more of it")
    return hits[0]


def remove(root, key: str, sessions=None) -> dict:
    """Delete one key's demeaned data. Refuses while a running session uses it or a build
    holds its lock. The folder is moved aside under the key's lock first, so a session
    starting meanwhile finds no manifest and builds afresh rather than loading half a folder."""
    e = find(root, key, sessions)
    if e["sessions"]:
        raise RuntimeError(
            f"{e['dataset']}/{e['key']} is in use by running session(s) {', '.join(e['sessions'])}: "
            "stop them first (dfeols session stop <name>)"
        )
    kd = Path(e["path"])
    try:
        lock = DirLock(kd, what="demeaned data").acquire()
    except LockedError as err:
        raise RuntimeError(f"{e['dataset']}/{e['key']} is being built or moved: {err}") from err
    try:
        users = live_users(Path(sessions) if sessions else None).get(str(kd.resolve()), [])
        if users:  # a session loaded it since the first look
            raise RuntimeError(
                f"{e['dataset']}/{e['key']} is in use by running session(s) {', '.join(users)}"
            )
        aside = kd.with_name(f"{kd.name}.removing-{uuid.uuid4().hex[:8]}")
        os.rename(kd, aside)
    finally:
        lock.release()
    shutil.rmtree(aside)
    try:
        kd.parent.rmdir()  # the dataset's folder, when this was its last key
    except OSError:
        pass
    return e


def space(path) -> dict | None:
    """Free and total bytes of the file system that holds `path` (or its nearest existing
    parent). A site quota per user is not visible here; this is the file system's."""
    p = Path(path)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        u = shutil.disk_usage(p)
    except OSError:
        return None
    return {"free": u.free, "total": u.total, "path": str(p)}


def fmt_bytes(n: float) -> str:
    for unit, k in (("TB", 1e12), ("GB", 1e9), ("MB", 1e6)):
        if n >= k:
            return f"{n / k:.1f} {unit}"
    return f"{n / 1e3:.1f} kB"


# ---------------------------------------------------------------- CLI


def _roots(args):
    from .session import demeaned_root, sessions_root
    from .site import profile as load_profile

    p = load_profile(args.profile)
    return demeaned_root(args.profile, args.demeaned), sessions_root(args.profile, args.sessions), p


def cmd_list(args) -> None:
    root, sessions, p = _roots(args)
    es = entries(root, sessions, p.scratch_purge_days)
    if args.json:
        print(json.dumps(es, indent=1))
        return
    if not es:
        print(f"no demeaned data under {root}")
        return
    print(f"demeaned data under {root}:")
    for e in es:
        state = []
        if not e["complete"]:
            state.append("being built" if e["building"] else "incomplete (an interrupted build)")
        elif not e["current"]:
            state.append(f"old format {e['format']}: no longer loaded, a session rebuilds it")
        if e["sessions"]:
            state.append(f"in use by {', '.join(e['sessions'])}")
        if e["purge_after"]:
            state.append(f"purged after ~{e['purge_after']}")
        what = ""
        if e["complete"]:
            what = (
                f"{e['rows']:,} rows; {', '.join(e['variables'] or [])}; FE {' + '.join(e['fixed_effects'])}"
            )
        print(f"  {e['dataset']}/{e['key']}  {fmt_bytes(e['bytes'])}  written {e['written']}")
        if what:
            print(f"      {what}")
        if state:
            print(f"      {'; '.join(state)}")
    total = sum(e["bytes"] for e in es)
    sp = space(root)
    free = f"; {fmt_bytes(sp['free'])} free on the file system" if sp else ""
    print(f"{len(es)} folder{'s' if len(es) != 1 else ''}, {fmt_bytes(total)}{free}")


def cmd_rm(args) -> None:
    root, sessions, _ = _roots(args)
    targets = []
    try:
        targets = [find(root, k, sessions) for k in args.keys]
    except KeyError as e:
        sys.exit(str(e.args[0]))
    for e in targets:
        print(f"{e['dataset']}/{e['key']}: {fmt_bytes(e['bytes'])}, written {e['written']}")
    if not args.yes:
        if not sys.stdin.isatty():
            sys.exit("not removing: confirm with --yes")
        if input(
            f"remove {len(targets)} folder{'s' if len(targets) > 1 else ''}? [y/N] "
        ).strip().lower() not in (
            "y",
            "yes",
        ):
            sys.exit("not removed")
    failed = False
    for e in targets:
        try:
            remove(root, f"{e['dataset']}/{e['key']}", sessions)
            print(f"removed {e['dataset']}/{e['key']}")
        except (RuntimeError, KeyError) as err:
            print(f"not removed: {err}", file=sys.stderr)
            failed = True
    if failed:
        sys.exit(1)


def add_parser(sub) -> None:
    sp = sub.add_parser("demeaned", help="persisted demeaned data: list, remove")
    ss = sp.add_subparsers(dest="demeaned_cmd", required=True)

    def common(q):
        q.add_argument("--profile")
        q.add_argument("--demeaned", help="demeaned-data folder (default: <scratch>/demeaned)")
        q.add_argument(
            "--sessions", help="sessions folder, to see which are in use (default: <scratch>/sessions)"
        )

    q = ss.add_parser("list", help="each key's size, age, purge date and users")
    common(q)
    q.add_argument("--json", action="store_true")
    q = ss.add_parser("rm", help="delete demeaned data by key (unless a running session uses it)")
    common(q)
    q.add_argument("keys", nargs="+", help="key, a unique prefix of one, or <dataset>/<key>")
    q.add_argument("--yes", action="store_true", help="do not ask")


def main(args) -> None:
    {"list": cmd_list, "rm": cmd_rm}[args.demeaned_cmd](args)
