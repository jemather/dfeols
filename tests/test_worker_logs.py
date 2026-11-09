"""Worker logs go in a folder named by the coordinator's job id."""

import re
from dataclasses import replace

import dfeols
from dfeols.cluster import make_cluster, worker_log_tag


def test_tag_is_coordinator_job_id(monkeypatch):
    for k in ("SLURM_JOB_ID", "PBS_JOBID", "JOB_ID", "LSB_JOBID"):
        monkeypatch.delenv(k, raising=False)
    assert re.fullmatch(r"local-\d{8}-\d{6}-\d+", worker_log_tag())
    monkeypatch.setenv("PBS_JOBID", "123.server/x")
    assert worker_log_tag() == "123.server_x"
    monkeypatch.setenv("SLURM_JOB_ID", "35851711")
    assert worker_log_tag() == "35851711"


def test_slurm_job_script_logs_under_job_id(monkeypatch, tmp_path):
    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    monkeypatch.setenv("SLURM_JOB_ID", "35851711")
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface=None)
    cl = make_cluster(p)
    try:
        script = cl.job_script()
    finally:
        cl.close()
    ld = tmp_path / "logs" / "35851711"
    assert ld.is_dir()
    assert f"{ld}/dfeols-worker-%J.out" in script or f"{ld}/dfeols-worker-%j.out" in script, script


def test_explicit_log_directory_kept(monkeypatch, tmp_path):
    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface=None)
    cl = make_cluster(p, log_directory=str(tmp_path / "mine"))
    try:
        assert str(tmp_path / "mine") in cl.job_script()
    finally:
        cl.close()
    assert not (tmp_path / "logs").exists()
