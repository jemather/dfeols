"""`dfeols config init --from-slurm` on a small cluster shaped like OSCER (docs: planner.md)."""

from datetime import date

import pytest

from dfeols.cli import main
from dfeols.site import load_config, profile
from dfeols.slurm_site import (
    _minutes,
    draft_profile,
    expand_hostlist,
    node_classes,
    parse_scontrol,
    parse_sinfo,
)

# sinfo -h -e -o "%R|%D|%c|%m|%l|%f|%G|%T|%N": partition x node type x state
NODES = {
    "c[001-008]": "20|32000|haswell,fdr10|(null)|mixed",
    "c[009-010]": "20|27800|haswell,fdr10|(null)|drained",  # OSCER: a few report less memory
    "c[101-104]": "24|64200|haswell,fdr10|(null)|allocated",
    "c[201-203]": "64|257600|icelake,hdr100|(null)|idle",
    "c[301-303]": "64|128500|icelake,hdr100|(null)|mixed",
    "c[401-402]": "40|96000|cascadelake,fdr|(null)|down*",
    "g[01-02]": "64|257600|icelake|gpu:a100:2|idle",
}
PARTS = {  # partition: (nodes, time limit, scontrol extras)
    "32gb_20core": (["c[001-008]", "c[009-010]"], "2-00:00:00", "MaxMemPerNode=31916"),
    "64gb_24core": (["c[101-104]"], "2-00:00:00", "MaxMemPerNode=64148"),
    "256gb_64core": (["c[201-203]"], "2-00:00:00", "MaxMemPerNode=257442"),
    "128gb_64core": (["c[301-303]"], "2-00:00:00", "MaxMemPerNode=128426"),
    "96gb_40core": (["c[401-402]"], "2-00:00:00", "MaxMemPerNode=94920"),
    "normal": (list(NODES)[:6], "2-00:00:00", "Default=YES MaxMemPerNode=257442"),
    "normal_avx2": (list(NODES)[:6], "2-00:00:00", "MaxMemPerNode=257442"),
    "largejobs": (list(NODES)[:6], "2-00:00:00", "MinNodes=13 MaxMemPerNode=257442"),
    "longjobs": (["c[001-008]", "c[201-203]"], "7-00:00:00", "MaxMemPerNode=UNLIMITED"),
    "debug": (["c[001-008]"], "00:30:00", ""),
    "gpu": (["g[01-02]"], "2-00:00:00", ""),
}


def _sinfo():
    lines = []
    for p, (nodes, limit, _) in PARTS.items():
        for n in nodes:
            cores, mem, feats, gres, state = NODES[n].split("|")
            k = len(expand_hostlist(n))
            lines.append(f"{p}|{k}|{cores}|{mem}|{limit}|{feats}|{gres}|{state}|{n}")
    return "\n".join(lines)


def _scontrol():
    return "\n".join(
        f"PartitionName={p} AllowGroups=ALL Default=NO MinNodes=0 MaxTime={limit} State=UP {extra}"
        for p, (_, limit, extra) in PARTS.items()
    )


def test_hostlist_and_minutes():
    assert expand_hostlist("c[001-003,010],gpu01") == ["c001", "c002", "c003", "c010", "gpu01"]
    assert expand_hostlist("r[1-2]n[01-02]") == ["r1n01", "r1n02", "r2n01", "r2n02"]
    assert _minutes("2-00:00:00") == 2880 and _minutes("30") == 30 and _minutes("UNLIMITED") == float("inf")


def test_node_classes():
    cls = {c["name"]: c for c in node_classes(parse_sinfo(_sinfo()), parse_scontrol(_scontrol()))}
    h = cls["32gb_20core"]
    assert (h["count"], h["in_service"], h["cores"], h["memory"]) == (10, 8, 20, "27800MB")
    assert h["mixed"] and h["interconnect"] == "fdr10" and h["features"] == ["haswell"] and not h["excluded"]
    assert cls["96gb_40core"]["excluded"] == ["no nodes in service"]
    # normal spans the hardware partitions even though 32gb_20core's nodes differ in memory
    assert cls["normal"]["aliases"] == ["normal_avx2"] and cls["normal"]["excluded"][0].startswith("spans ")
    assert "32gb_20core" in cls["normal"]["excluded"][0] and "debug" not in cls["normal"]["excluded"][0]
    assert cls["largejobs"]["excluded"] == ["jobs need at least 13 nodes"]
    assert cls["debug"]["excluded"] == ["time limit 00:30:00"]
    assert cls["gpu"]["excluded"] == ["GPU nodes"]
    lj = cls["longjobs"]  # mixed, but its 7-day limit is its own
    assert lj["mixed"] and not lj["excluded"] and (lj["cores"], lj["memory"]) == (20, "32000MB")


def test_draft_is_a_loadable_profile(tmp_path, monkeypatch):
    text = draft_profile(_sinfo(), _scontrol(), "oscer", today=date(2026, 10, 28))
    cfg = tmp_path / "config.toml"
    cfg.write_text(text)
    p = profile("oscer", path=cfg)
    assert p.scheduler == "slurm" and p.queue == "normal" and p.max_walltime == "2-00:00:00"
    assert p.billing == "cores"
    assert p.nodes["longjobs"]["max_walltime"] == "7-00:00:00"
    assert "max_walltime" not in p.nodes["32gb_20core"]
    assert p.nodes["gpu"]["use"] is False and "use" not in p.nodes["256gb_64core"]


def test_cli_from_saved_output(tmp_path, monkeypatch, capsys):
    (tmp_path / "s.txt").write_text(_sinfo())
    (tmp_path / "c.txt").write_text(_scontrol())
    cfg = tmp_path / "cfg" / "config.toml"
    monkeypatch.setenv("DFEOLS_CONFIG", str(cfg))
    args = [
        "config",
        "init",
        "--from-slurm",
        "--sinfo",
        str(tmp_path / "s.txt"),
        "--scontrol",
        str(tmp_path / "c.txt"),
    ]
    main(args + ["--profile", "oscer"])
    assert "[profiles.oscer.nodes.32gb_20core]" in capsys.readouterr().out and not cfg.exists()
    main(args + ["--profile", "oscer", "--write"])
    assert "oscer" in load_config()["profiles"] and "cluster" in load_config()["profiles"]
    with pytest.raises(SystemExit):  # never overwrites a profile
        main(args + ["--profile", "oscer", "--write"])
