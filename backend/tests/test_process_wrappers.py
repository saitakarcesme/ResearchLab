from __future__ import annotations

import os
import shutil
import subprocess
import time
import uuid

import pytest

from backend.adapters.autoresearch import (
    RUN_ID_ENV,
    build_managed_posix_script,
    build_posix_group_terminate_script,
    build_setsid_wait_argv,
)

WSL_PREFIX = [
    "wsl.exe",
    "-d",
    os.getenv("AUTORESEARCH_WSL_DISTRO", "Ubuntu-24.04"),
    "--",
]


def _wsl_shell(script: str, *, timeout: float = 5) -> subprocess.CompletedProcess[str]:
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
        deadline = time.monotonic() + 5
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
