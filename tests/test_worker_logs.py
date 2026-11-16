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


def test_ib_interface_workers_listen_on_their_own_ib_address(monkeypatch, tmp_path):
    """interface = "ib": each worker job finds its node's IPoIB address at start and listens
    there (OSCER: dfeols' traffic went over eno1 Ethernet, ibp4s0 unused); a node without one
    falls back to its primary address and says so."""
    import subprocess

    from dfeols import cluster as cmod
    from dfeols.cluster import IB_HOST_LINES

    monkeypatch.setenv("DFEOLS_CONFIG", str(tmp_path / "none.toml"))
    p = replace(dfeols.profile("local", scratch=str(tmp_path)), scheduler="slurm", interface="ib")

    def script_with(ib_name):
        monkeypatch.setattr(cmod, "ib_interface", lambda: ib_name)
        cl = make_cluster(p)
        try:
            return cl.job_script()
        finally:
            cl.close()

    # A coordinator without IB (OSCER c943) could not gather from workers on IB addresses:
    # the workers then stay on their default addresses.
    script = script_with(None)
    assert "--host" not in script and IB_HOST_LINES[0] not in script, script
    script = script_with("lo")
    assert "--host $DFEOLS_WORKER_HOST" in script, script

    body = "\n".join(IB_HOST_LINES) + '\necho "HOST=$DFEOLS_WORKER_HOST"\n'
    assert all(line in script for line in IB_HOST_LINES)
    fake = tmp_path / "bin"
    fake.mkdir()
    (fake / "ip").write_text(
        "#!/bin/sh\necho 'lo UNKNOWN 127.0.0.1/8'\necho 'ibp4s0 UP 10.251.87.90/22'\n"
        "echo 'eno1 UP 10.251.83.90/22'\n"
    )
    (fake / "ip").chmod(0o755)
    env = {"PATH": f"{fake}:/usr/bin:/bin"}
    out = subprocess.run(["bash", "-c", body], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "HOST=10.251.87.90" in out.stdout, out
    (fake / "ip").write_text("#!/bin/sh\necho 'eno1 UP 10.251.83.90/22'\n")
    out = subprocess.run(["bash", "-c", body], capture_output=True, text=True, env=env)
    assert out.returncode == 0 and "HOST=" in out.stdout and "no InfiniBand address" in out.stderr, out
