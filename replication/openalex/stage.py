# SPDX-License-Identifier: GPL-3.0-or-later
"""D14 stage 1: OpenAlex JSON Lines snapshot -> slim Parquet tables (KMZ 2009 replication).

Reads every file listed in each entity's manifest.json under raw/openalex/<release>/jsonl/,
parses only the fields below (Polars NDJSON with a fixed schema; the rest of each record is
ignored), and writes one Parquet file per input file and table:

  works/          work_id, year, type, source_id, source_type, cited_by_count, authors_count,
                  is_retracted, is_paratext, language, ref_count, fwci, first_page, last_page,
                  topic_id, subfield_id, field_id
  authorships/    work_id, year, author_id, author_order (0-based), position, is_corresponding,
                  n_inst, institution_id (one row per author x institution; null if none),
                  inst_country, countries_n
  refs/           work_id, ref_id (referenced_works)
  authors/        author_id, orcid, display_name, works_count, cited_by_count
  author_affil/   author_id, institution_id, year (the author record's affiliations x years)
  institutions/   institution_id, ror, display_name, country_code, type, lineage, parent_id
  sources/        source_id, issn_l, issn, display_name, type, host_org_id, country_code, is_core

OpenAlex IDs become Int64 (the digits after W/A/I/S/T; subfields/fields their number).
Each input file's tables are written, then a marker `_done/<entity>/<name>.json` with row
counts, which must equal the manifest's record_count; files with a marker are skipped on
rerun. `--check` only compares markers with the manifests.

    cd /srv/dfeols && .venv/bin/python -I repo/replication/openalex/stage.py \
        --src raw/openalex/2026-10-14/jsonl --out build/openalex/stage --workers 8

Uses a spawn process pool (Polars and fork don't mix); POLARS_MAX_THREADS per worker from
--threads. Largest files go first.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

ENTITIES = ("institutions", "sources", "authors", "works")


def _schemas():
    import polars as pl

    S, L, U = pl.Struct, pl.List, pl.Utf8
    return {
        "works": {
            "id": U,
            "publication_year": pl.Int32,
            "type": U,
            "cited_by_count": pl.Int64,
            "authors_count": pl.Int64,
            "is_retracted": pl.Boolean,
            "is_paratext": pl.Boolean,
            "language": U,
            "referenced_works": L(U),
            "referenced_works_count": pl.Int64,
            "fwci": pl.Float64,
            "biblio": S({"first_page": U, "last_page": U}),
            "primary_location": S({"source": S({"id": U, "type": U})}),
            "primary_topic": S({"id": U, "subfield": S({"id": U}), "field": S({"id": U})}),
            "authorships": L(
                S(
                    {
                        "author": S({"id": U}),
                        "author_position": U,
                        "is_corresponding": pl.Boolean,
                        "institutions": L(S({"id": U, "country_code": U})),
                        "countries": L(U),
                    }
                )
            ),
        },
        "authors": {
            "id": U,
            "orcid": U,
            "display_name": U,
            "works_count": pl.Int64,
            "cited_by_count": pl.Int64,
            "affiliations": L(S({"institution": S({"id": U}), "years": L(pl.Int32)})),
        },
        "institutions": {
            "id": U,
            "ror": U,
            "display_name": U,
            "country_code": U,
            "type": U,
            "lineage": L(U),
        },
        "sources": {
            "id": U,
            "issn_l": U,
            "issn": L(U),
            "display_name": U,
            "type": U,
            "host_organization": U,
            "country_code": U,
            "is_core": pl.Boolean,
        },
    }


def _num(e):
    """'https://openalex.org/W123' or '.../fields/14' -> 123 / 14 (Int64)."""
    import polars as pl

    return e.str.extract(r"(\d+)$").cast(pl.Int64)


def _tables(entity: str, d):
    import polars as pl

    c = pl.col
    if entity == "works":
        w = d.select(
            _num(c("id")).alias("work_id"),
            c("publication_year").cast(pl.Int16, strict=False).alias("year"),
            c("type"),
            _num(c("primary_location").struct.field("source").struct.field("id")).alias("source_id"),
            c("primary_location").struct.field("source").struct.field("type").alias("source_type"),
            c("cited_by_count").cast(pl.Int32),
            c("authors_count").cast(pl.Int32),
            c("is_retracted"),
            c("is_paratext"),
            c("language"),
            c("referenced_works_count").cast(pl.Int32).alias("ref_count"),
            c("fwci").cast(pl.Float32),
            c("biblio").struct.field("first_page"),
            c("biblio").struct.field("last_page"),
            _num(c("primary_topic").struct.field("id")).cast(pl.Int32).alias("topic_id"),
            _num(c("primary_topic").struct.field("subfield").struct.field("id")).cast(pl.Int16).alias("subfield_id"),
            _num(c("primary_topic").struct.field("field").struct.field("id")).cast(pl.Int16).alias("field_id"),
        )
        a = (
            d.select(
                _num(c("id")).alias("work_id"),
                c("publication_year").cast(pl.Int16, strict=False).alias("year"),
                c("authorships"),
                pl.int_ranges(0, c("authorships").list.len(), dtype=pl.Int16).alias("author_order"),
            )
            .explode("authorships", "author_order")
            .filter(c("authorships").is_not_null())
            .select(
                "work_id",
                "year",
                _num(c("authorships").struct.field("author").struct.field("id")).alias("author_id"),
                "author_order",
                c("authorships").struct.field("author_position").alias("position"),
                c("authorships").struct.field("is_corresponding"),
                c("authorships").struct.field("institutions").list.len().cast(pl.Int16).alias("n_inst"),
                c("authorships").struct.field("institutions").alias("inst"),
                c("authorships").struct.field("countries").list.len().cast(pl.Int16).alias("countries_n"),
            )
            .explode("inst", empty_as_null=True)  # keep authorships with no institution (null)
            .select(
                "work_id",
                "year",
                "author_id",
                "author_order",
                "position",
                "is_corresponding",
                "n_inst",
                _num(c("inst").struct.field("id")).alias("institution_id"),
                c("inst").struct.field("country_code").alias("inst_country"),
                "countries_n",
            )
        )
        r = (
            d.select(_num(c("id")).alias("work_id"), c("referenced_works"))
            .explode("referenced_works")
            .filter(c("referenced_works").is_not_null())
            .select("work_id", _num(c("referenced_works")).alias("ref_id"))
        )
        return {"works": w, "authorships": a, "refs": r}
    if entity == "authors":
        au = d.select(
            _num(c("id")).alias("author_id"),
            c("orcid").str.extract(r"([0-9X-]{19})$").alias("orcid"),
            "display_name",
            c("works_count").cast(pl.Int32),
            c("cited_by_count").cast(pl.Int32),
        )
        af = (
            d.select(_num(c("id")).alias("author_id"), "affiliations")
            .explode("affiliations")
            .filter(c("affiliations").is_not_null())
            .select(
                "author_id",
                _num(c("affiliations").struct.field("institution").struct.field("id")).alias("institution_id"),
                c("affiliations").struct.field("years").alias("year"),
            )
            .explode("year")
            .filter(c("year").is_not_null())
            .with_columns(c("year").cast(pl.Int16, strict=False))  # a few raw years are malformed (e.g. 2025002): null
        )
        return {"authors": au, "author_affil": af}
    if entity == "institutions":
        return {
            "institutions": d.select(
                _num(c("id")).alias("institution_id"),
                "ror",
                "display_name",
                "country_code",
                "type",
                c("lineage").list.eval(_num(pl.element())).alias("lineage"),
            ).with_columns(c("lineage").list.last().alias("parent_id"))
        }
    if entity == "sources":
        return {
            "sources": d.select(
                _num(c("id")).alias("source_id"),
                "issn_l",
                "issn",
                "display_name",
                "type",
                _num(c("host_organization")).alias("host_org_id"),
                "country_code",
                "is_core",
            )
        }
    raise ValueError(entity)


def stage_file(entity: str, path: str, out: str, name: str) -> dict:
    import polars as pl

    t0 = time.time()
    with gzip.open(path, "rb") as fh:
        b = fh.read()
    d = pl.read_ndjson(b, schema=_schemas()[entity])
    del b
    rows = {"records": d.height}
    for tname, t in _tables(entity, d).items():
        od = Path(out) / tname
        od.mkdir(parents=True, exist_ok=True)
        tmp = od / f".{name}.parquet.part"
        t.write_parquet(tmp, compression="zstd", statistics=True)
        tmp.rename(od / f"{name}.parquet")
        rows[tname] = t.height
    rec = {"entity": entity, "file": path, "name": name, "rows": rows, "seconds": round(time.time() - t0, 1)}
    done = Path(out) / "_done" / entity
    done.mkdir(parents=True, exist_ok=True)
    (done / f"{name}.json").write_text(json.dumps(rec))
    return rec


def files(src: Path, entity: str) -> list[tuple[str, str, int, int]]:
    """(local path, name, record_count, bytes) for every manifest entry."""
    m = json.loads((src / entity / "manifest.json").read_text())
    out = []
    for f in m["files"]:
        rel = f["url"].split(f"/jsonl/{entity}/", 1)[1]  # updated_date=.../part_0000.gz
        p = src / entity / rel
        name = rel.replace("updated_date=", "").replace("/", "_").removesuffix(".gz")
        out.append((str(p), name, int(f["meta"]["record_count"]), int(f["meta"]["content_length"])))
    return out


def check(src: Path, out: Path) -> dict:
    res = {}
    for e in ENTITIES:
        fs = files(src, e)
        done = bad = missing = 0
        for _, name, n, _ in fs:
            mk = out / "_done" / e / f"{name}.json"
            if not mk.exists():
                missing += 1
                continue
            done += 1
            if json.loads(mk.read_text())["rows"]["records"] != n:
                bad += 1
        res[e] = {"files": len(fs), "done": done, "missing": missing, "count_mismatch": bad}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2, help="POLARS_MAX_THREADS per worker")
    ap.add_argument("--entities", nargs="*", default=list(ENTITIES))
    ap.add_argument("--limit", type=int, default=0, help="at most this many files per entity (testing)")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        print(json.dumps(check(a.src, a.out), indent=1))
        return
    os.environ["POLARS_MAX_THREADS"] = str(a.threads)  # inherited by spawned workers
    todo = []
    for e in a.entities:
        fs = files(a.src, e)
        if a.limit:
            fs = fs[: a.limit]
        for p, name, n, size in fs:
            if not (a.out / "_done" / e / f"{name}.json").exists():
                todo.append((size, e, p, name, n))
    todo.sort(reverse=True)  # largest first
    print(f"{time.strftime('%H:%M:%S')} {len(todo)} files to stage ({sum(t[0] for t in todo) / 1e9:.1f} GB)", flush=True)
    t0, done_b, total_b = time.time(), 0, sum(t[0] for t in todo)
    with ProcessPoolExecutor(a.workers, mp_context=get_context("spawn")) as ex:
        futs = {ex.submit(stage_file, e, p, str(a.out), name): (size, e, name, n) for size, e, p, name, n in todo}
        for f in as_completed(futs):
            size, e, name, n = futs[f]
            try:
                rec = f.result()
            except Exception as err:
                print(f"{time.strftime('%H:%M:%S')} FAIL {e} {name}: {type(err).__name__}: {err}", flush=True)
                continue
            done_b += size
            flag = "" if rec["rows"]["records"] == n else f" COUNT MISMATCH manifest {n}"
            el = time.time() - t0
            eta = el / done_b * (total_b - done_b) if done_b else 0
            print(
                f"{time.strftime('%H:%M:%S')} {e} {name} {rec['rows']} {rec['seconds']}s "
                f"[{done_b / total_b:6.1%}, eta {eta / 60:.0f} min]{flag}",
                flush=True,
            )
    print(json.dumps(check(a.src, a.out), indent=1))


if __name__ == "__main__":
    main()
