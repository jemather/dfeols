"""A coordinator ended by SIGTERM (scancel, walltime) cancels its worker jobs first."""

import os
import signal
import subprocess
import sys
import textwrap
import time


def test_sigterm_closes_the_cluster(tmp_path):
    flag = tmp_path / "closed"
    code = textwrap.dedent(
        f"""
        import time
        from dfeols.cluster import _close_on_exit

        class Fake:
            def close(self, timeout=None):
                open({str(flag)!r}, "w").write("yes")

        _close_on_exit(Fake())
        print("ready", flush=True)
        time.sleep(60)
        """
    )
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "ready"
    os.kill(p.pid, signal.SIGTERM)
    rc = p.wait(timeout=30)
    assert rc == 128 + signal.SIGTERM
    t = time.time()
    while not flag.exists() and time.time() - t < 5:
        time.sleep(0.05)
    assert flag.read_text() == "yes"
