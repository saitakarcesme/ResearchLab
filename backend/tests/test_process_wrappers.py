from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import PurePosixPath

import pytest

from backend.adapters.autoresearch import (
    RUN_ID_ENV,
    build_managed_posix_script,
    build_posix_group_terminate_script,
    build_setsid_wait_argv,
)
from backend.process_control import MANAGED_RUN_ID_ENV
from backend.supervisor import ResearchSupervisor

WSL_PREFIX = [
    "wsl.exe",
    "-d",
    os.getenv("AUTORESEARCH_WSL_DISTRO", "Ubuntu-24.04"),
    "--",
]


def _wsl_shell(script: str, *, timeout: float = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [*WSL_PREFIX, "sh", "-s"],
        input=script,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )


def test_managed_posix_wrapper_constructs_attached_session() -> None:
    script = build_managed_posix_script("/tmp/research.pid", "sleep 1")
    argv = build_setsid_wait_argv("research-id")
    assert argv[:4] == [
        "setsid",
        "--wait",
        "env",
        f"{RUN_ID_ENV}=research-id",
    ]
    assert argv[4:] == ["sh", "-lc", "exec sh -s"]
    assert "ps -o pgid= -p" in script
    assert "printf '%s\\n' \"$pgid\"" in script
    assert "echo $$" not in script


def test_posix_artifact_delete_refuses_a_symlinked_root() -> None:
    if shutil.which("wsl.exe") is None:
        pytest.skip("WSL is unavailable")
    token = f"delete-safety-{uuid.uuid4().hex}"
    base = f"/tmp/{token}"
    outside = f"/tmp/{token}-outside"
    setup = _wsl_shell(
        f"mkdir -p {outside}/research-id {base}/managed && "
        f"printf keep > {outside}/research-id/keep.txt && "
        f"ln -s {outside} {base}/managed/source"
    )
    if setup.returncode != 0:
        pytest.skip("WSL could not create the symlink safety fixture")
    try:
        script = ResearchSupervisor._posix_remove_child_script(
            PurePosixPath(f"{base}/managed/source"), "research-id"
        )
        result = _wsl_shell(script)
        assert result.returncode != 0
        protected = _wsl_shell(f"cat {outside}/research-id/keep.txt")
        assert protected.returncode == 0
        assert protected.stdout == "keep"
    finally:
        _wsl_shell(f"rm -rf -- {base} {outside}")


def test_wsl_wrapper_stays_attached_and_termination_leaves_no_group() -> None:
    if shutil.which("wsl.exe") is None:
        pytest.skip("WSL is unavailable")
    available = _wsl_shell(
        "command -v setsid >/dev/null && "
        "test \"$(ps -o pgid= -p $$ | tr -d '[:space:]')\" -gt 0"
    )
    if available.returncode != 0:
        pytest.skip("WSL util-linux/procps tools are unavailable")

    run_id = f"wrapper-test-{uuid.uuid4()}"
    pid_file = f"/tmp/{run_id}.pid"
    # This descendant deliberately ignores TERM so the controller must retain the
    # recorded PGID and escalate the whole group after the leader removes its marker.
    body = "sh -c 'trap \"\" TERM; sleep 30'"
    script = build_managed_posix_script(pid_file, body)
    process = subprocess.Popen(
        [*WSL_PREFIX, *build_setsid_wait_argv(run_id)],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdin is not None
    process.stdin.write(script)
    process.stdin.close()
    pgid: str | None = None
    try:
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            marker = _wsl_shell(f"test -f {pid_file} && cat {pid_file}")
            if marker.returncode == 0 and marker.stdout.strip().isdigit():
                pgid = marker.stdout.strip()
                break
            assert process.poll() is None
            time.sleep(0.05)
        assert pgid is not None
        assert process.poll() is None, "setsid wrapper detached before its child exited"

        observed = _wsl_shell(f"ps -o pgid= -p {pgid} | tr -d '[:space:]'")
        assert observed.returncode == 0
        assert observed.stdout.strip() == pgid
        assert _wsl_shell(f"/bin/kill -0 -- -{pgid}").returncode == 0

        terminate = _wsl_shell(
            build_posix_group_terminate_script(pid_file, run_id), timeout=8
        )
        assert terminate.returncode == 0, terminate.stderr
        process.wait(timeout=5)
        assert _wsl_shell(f"/bin/kill -0 -- -{pgid} 2>/dev/null").returncode != 0
        assert _wsl_shell(f"test ! -e {pid_file}").returncode == 0
    finally:
        if pgid is not None:
            _wsl_shell(
                f"/bin/kill -KILL -- -{pgid} 2>/dev/null || true; rm -f {pid_file}"
            )
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows process-marker recovery")
def test_native_cleanup_terminates_only_the_exact_managed_run() -> None:
    run_id = f"native-wrapper-test-{uuid.uuid4()}"
    environment = os.environ.copy()
    environment[MANAGED_RUN_ID_ENV] = run_id
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        env=environment,
    )
    try:
        ResearchSupervisor._terminate_native_process_trees(run_id)
        assert process.wait(timeout=3) != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=3)
