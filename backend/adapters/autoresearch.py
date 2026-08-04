from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from backend.adapters.base import AdapterContext, ResearchAdapter, ResearchFailed
from backend.codex_usage import parse_codex_jsonl_usage
from backend.local_models import OllamaClient, model_name_from_id
from backend.process_control import (
    MANAGED_RUN_ID_ENV,
    ProcessResult,
    run_managed_process,
)
from backend.researcher_models import researcher_model_command_args
from backend.telemetry import SSHCommandRunner

LIVE_TOKEN_STEP_PATTERN = re.compile(
    r"step\s+\d+.*?\|\s*dt:\s*([\d,]+)ms\s*\|\s*tok/sec:\s*([\d,]+)",
    re.IGNORECASE,
)


def live_tokens_from_output(text: str) -> int:
    """Count tokens completed by the step lines already flushed by train.py."""

    return sum(
        round(int(milliseconds.replace(",", "")) * int(rate.replace(",", "")) / 1000)
        for milliseconds, rate in LIVE_TOKEN_STEP_PATTERN.findall(text)
    )

FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
VAL_BPB_PATTERN = re.compile(rf"^val_bpb:\s*({FLOAT_PATTERN})\s*$", re.MULTILINE)
PEAK_VRAM_PATTERN = re.compile(rf"^peak_vram_mb:\s*({FLOAT_PATTERN})\s*$", re.MULTILINE)
SUMMARY_FIELDS = (
    "val_bpb",
    "training_seconds",
    "total_seconds",
    "peak_vram_mb",
    "mfu_percent",
    "total_tokens_M",
    "num_steps",
    "num_params_M",
    "depth",
)
SUMMARY_PATTERN = re.compile(
    rf"^---\s*\r?\n"
    rf"val_bpb:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"training_seconds:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"total_seconds:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"peak_vram_mb:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"mfu_percent:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"total_tokens_M:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"num_steps:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"num_params_M:\s*({FLOAT_PATTERN})\s*\r?\n"
    rf"depth:\s*({FLOAT_PATTERN})\s*$",
    re.MULTILINE,
)
OOM_PATTERN = re.compile(
    r"(?:CUDA\s+out\s+of\s+memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED)",
    re.IGNORECASE,
)
AGENT_RETRY_BASE_SECONDS = 5
AGENT_RETRY_MAX_SECONDS = 300


class AgentGenerationTimeout(RuntimeError):
    """Raised when a Codex candidate does not finish within the active watchdog."""


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    val_bpb: float
    training_seconds: float
    total_seconds: float
    peak_vram_mb: float
    mfu_percent: float
    total_tokens_m: float
    num_steps: int
    num_params_m: float
    depth: int


def parse_evaluation_summary(
    output: str,
    returncode: int = 0,
    *,
    active_duration_seconds: float | None = None,
) -> EvaluationSummary:
    if returncode != 0:
        raise ValueError(f"evaluation exited with status {returncode}")
    blocks = list(SUMMARY_PATTERN.finditer(output))
    if len(blocks) != 1:
        raise ValueError(
            f"expected exactly one full terminal summary block, found {len(blocks)}"
        )
    block = blocks[0]
    if output[block.end() :].strip():
        raise ValueError("evaluation summary must be the terminal output block")
    for field in SUMMARY_FIELDS:
        occurrences = re.findall(
            rf"^{re.escape(field)}:\s*{FLOAT_PATTERN}\s*$", output, re.MULTILINE
        )
        if len(occurrences) != 1:
            raise ValueError(
                f"expected exactly one {field} field, found {len(occurrences)}"
            )
    values = [float(value) for value in block.groups()]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("all evaluation summary values must be finite")
    summary = EvaluationSummary(
        val_bpb=values[0],
        training_seconds=values[1],
        total_seconds=values[2],
        peak_vram_mb=values[3],
        mfu_percent=values[4],
        total_tokens_m=values[5],
        num_steps=int(values[6]),
        num_params_m=values[7],
        depth=int(values[8]),
    )
    positive = {
        "val_bpb": summary.val_bpb,
        "peak_vram_mb": summary.peak_vram_mb,
        "total_tokens_M": summary.total_tokens_m,
        "num_steps": summary.num_steps,
        "num_params_M": summary.num_params_m,
        "depth": summary.depth,
    }
    invalid = [name for name, value in positive.items() if value <= 0]
    if invalid:
        raise ValueError(
            f"evaluation summary fields must be positive: {', '.join(invalid)}"
        )
    if values[6] != summary.num_steps or values[8] != summary.depth:
        raise ValueError("num_steps and depth must be whole numbers")
    if summary.num_steps < 12:
        raise ValueError("num_steps must be at least 12 for the pinned training loop")
    if summary.training_seconds < 299.0:
        raise ValueError(
            "training_seconds must be at least 299.0 for the pinned budget"
        )
    if summary.total_seconds < summary.training_seconds:
        raise ValueError(
            "total_seconds must be greater than or equal to training_seconds"
        )
    if active_duration_seconds is not None:
        if (
            not math.isfinite(active_duration_seconds)
            or active_duration_seconds < 299.0
        ):
            raise ValueError(
                "managed active execution duration must be at least 299 seconds"
            )
        if summary.training_seconds > active_duration_seconds + 2.0:
            raise ValueError(
                "reported training_seconds exceeds managed active execution duration tolerance"
            )
    return summary


def parse_val_bpb(
    output: str, returncode: int = 0, *, active_duration_seconds: float | None = None
) -> float:
    return parse_evaluation_summary(
        output,
        returncode,
        active_duration_seconds=active_duration_seconds,
    ).val_bpb


def parse_peak_vram_mb(output: str) -> float | None:
    matches = PEAK_VRAM_PATTERN.findall(output)
    if len(matches) != 1:
        return None
    value = float(matches[0])
    return value if math.isfinite(value) else None


def is_better(candidate: float, best: float | None, direction: str) -> bool:
    if not math.isfinite(candidate):
        return False
    if best is None:
        return True
    if direction == "lower_is_better":
        return candidate < best
    if direction == "higher_is_better":
        return candidate > best
    raise ValueError(f"Unsupported metric direction: {direction}")


def agent_retry_delay_seconds(consecutive_failures: int) -> int:
    if consecutive_failures < 1:
        raise ValueError("consecutive_failures must be positive")
    exponent = min(consecutive_failures - 1, 30)
    return min(AGENT_RETRY_MAX_SECONDS, AGENT_RETRY_BASE_SECONDS * (2**exponent))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tail(path: Path, max_chars: int = 3000) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()
    except OSError:
        return ""


def _windows_to_wsl(path: Path) -> str:
    resolved = path.resolve()
    drive = resolved.drive.rstrip(":").lower()
    remainder = resolved.as_posix().split(":", 1)[-1]
    return f"/mnt/{drive}{remainder}"


RUN_ID_ENV = MANAGED_RUN_ID_ENV


def build_managed_posix_script(pid_file: str, body: str) -> str:
    """Build the inner session-leader shell without assuming $$ equals its PGID."""
    marker = shlex.quote(pid_file)
    return (
        "pgid=$(ps -o pgid= -p $$) || exit 70; set -- $pgid; pgid=$1; "
        "case \"$pgid\" in ''|*[!0-9]*) exit 70;; esac; "
        f"printf '%s\\n' \"$pgid\" > {marker} || exit 71; "
        f"cleanup() {{ rm -f -- {marker}; }}; "
        "trap cleanup EXIT; trap 'exit 129' HUP; trap 'exit 130' INT; "
        "trap 'exit 143' TERM; "
        f'{{ {body}; }}; status=$?; exit "$status"'
    )


def build_setsid_wait_argv(run_id: str) -> list[str]:
    return [
        "setsid",
        "--wait",
        "env",
        f"{RUN_ID_ENV}={run_id}",
        "sh",
        "-lc",
        "exec sh -s",
    ]


def build_posix_group_signal_script(
    pid_file: str, run_id: str, signal_name: str
) -> str:
    if signal_name not in {"STOP", "CONT", "TERM", "KILL"}:
        raise ValueError(f"Unsupported process-group signal: {signal_name}")
    marker = shlex.quote(pid_file)
    expected = shlex.quote(f"{RUN_ID_ENV}={run_id}")
    return (
        "i=0; while [ ! -f "
        f'{marker} ] && [ "$i" -lt 20 ]; do sleep 0.05; i=$((i + 1)); done; '
        f'test -f {marker} || exit 72; p="$(cat {marker})"; '
        "case \"$p\" in ''|*[!0-9]*) exit 73;; esac; "
        'test -r "/proc/$p/environ" || exit 74; '
        f"tr '\\0' '\\n' < \"/proc/$p/environ\" | grep -Fqx -- {expected} "
        "|| exit 75; "
        f'/bin/kill -{signal_name} -- -"$p"'
    )


def build_posix_group_terminate_script(pid_file: str, run_id: str) -> str:
    marker = shlex.quote(pid_file)
    validated = build_posix_group_signal_script(pid_file, run_id, "TERM")
    # Keep the validated PGID in this control shell: the managed shell's EXIT trap
    # may remove the marker before descendants that ignore TERM are gone.
    return (
        f'{validated}; status=$?; [ "$status" -eq 0 ] || exit "$status"; '
        'i=0; while /bin/kill -0 -- -"$p" 2>/dev/null && [ "$i" -lt 20 ]; '
        "do sleep 0.05; i=$((i + 1)); done; "
        'if /bin/kill -0 -- -"$p" 2>/dev/null; then /bin/kill -KILL -- -"$p"; fi; '
        f"rm -f -- {marker}"
    )


class KarpathyAutoresearchAdapter(ResearchAdapter):
    """Pinned implementation of Karpathy's single-file autoresearch protocol."""

    def __init__(self, context: AdapterContext):
        super().__init__(context)
        self.settings = context.settings
        self.db = context.database
        self.research_id = str(context.research["id"])
        self.workspace = self.settings.workspace_dir / self.research_id
        source_key = hashlib.sha256(
            str(context.gpu_source["id"]).encode("utf-8")
        ).hexdigest()[:16]
        self.source_key = source_key
        self._wsl_runtime_path: PurePosixPath | None = None
        self.runtime_dir = (
            self.settings.runtime_dir / source_key / self.research_id
        ).resolve()
        try:
            self.runtime_dir.relative_to(self.workspace.resolve())
        except ValueError:
            pass
        else:
            raise RuntimeError("The uv runtime must be outside the editable worktree")
        self.research_log_dir = self.settings.log_dir / self.research_id
        self.research_log_dir.mkdir(parents=True, exist_ok=True)
        self.prepare_hash: str | None = None
        self._agent_failure_streak = 0
        self._evaluation_failure_streak = 0

    def _git(
        self, *args: str, cwd: Path | None = None, check: bool = True
    ) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [self.settings.git_binary, *args],
            cwd=str(cwd or self.workspace),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if check and result.returncode != 0:
            raise RuntimeError(
                (result.stderr or result.stdout or "git command failed").strip()
            )
        return result

    def _commit(self, message: str, *, amend: bool = False) -> str:
        self._git("add", "--", "train.py", "program.md", cwd=self.workspace)
        args = [
            "-c",
            "user.name=Autoresearch Lab",
            "-c",
            "user.email=autoresearch@localhost",
            "commit",
        ]
        if amend:
            args.extend(["--amend", "--no-edit"])
        else:
            args.extend(["-m", message])
        result = self._git(*args, cwd=self.workspace, check=False)
        if (
            result.returncode != 0
            and "nothing to commit" not in (result.stdout + result.stderr).lower()
        ):
            raise RuntimeError((result.stderr or result.stdout).strip())
        return self._git("rev-parse", "HEAD", cwd=self.workspace).stdout.strip()

    @property
    def _best_ref(self) -> str:
        return f"refs/autoresearch/best/{self.research_id}"

    def _set_best_commit(self, commit: str) -> None:
        self._git("cat-file", "-e", f"{commit}^{{commit}}", cwd=self.workspace)
        self._git("update-ref", self._best_ref, commit, cwd=self.workspace)
        self.db.update_research(self.research_id, {"best_git_commit": commit})

    def _restore_persisted_best(
        self, research: Mapping[str, Any], recovered_experiments: Sequence[int]
    ) -> str:
        commit = research.get("best_git_commit")
        if not commit:
            ref_result = self._git("rev-parse", "--verify", self._best_ref, check=False)
            if ref_result.returncode == 0:
                commit = ref_result.stdout.strip()
        if not commit:
            for experiment in reversed(self.db.list_experiments(self.research_id)):
                if experiment.get("accepted") is True and experiment.get("git_commit"):
                    commit = experiment["git_commit"]
                    break
        if commit:
            valid = self._git(
                "cat-file",
                "-e",
                f"{commit}^{{commit}}",
                cwd=self.workspace,
                check=False,
            )
            if valid.returncode != 0:
                raise RuntimeError(
                    f"Persisted best Git commit is unavailable: {commit}"
                )
            self._git("reset", "--hard", str(commit), cwd=self.workspace)
            self._set_best_commit(str(commit))
            return str(commit)

        if recovered_experiments:
            subject = self._git(
                "log", "-1", "--format=%s", cwd=self.workspace
            ).stdout.strip()
            if any(
                subject.startswith(f"experiment {number}:")
                for number in recovered_experiments
            ):
                self._git("reset", "--hard", "HEAD^", cwd=self.workspace)
        head = self._git("rev-parse", "HEAD", cwd=self.workspace).stdout.strip()
        self._set_best_commit(head)
        return head

    def _ensure_upstream(self) -> None:
        cache = self.settings.upstream_cache_dir
        if not (cache / ".git").exists():
            if any(cache.iterdir()):
                raise RuntimeError(
                    f"Upstream cache is non-empty but is not a Git repository: {cache}"
                )
            result = subprocess.run(
                [
                    self.settings.git_binary,
                    "clone",
                    "--no-checkout",
                    self.settings.upstream_url,
                    str(cache),
                ],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout).strip())
        self._git(
            "fetch",
            "--depth",
            "1",
            "origin",
            self.settings.upstream_revision,
            cwd=cache,
        )

    def _ensure_worktree(self) -> None:
        if (self.workspace / ".git").exists():
            return
        if self.workspace.exists() and any(self.workspace.iterdir()):
            raise RuntimeError(
                f"Research workspace is non-empty and unmanaged: {self.workspace}"
            )
        branch = f"autoresearch/{self.research_id[:12]}"
        branch_exists = self._git(
            "show-ref",
            "--verify",
            f"refs/heads/{branch}",
            cwd=self.settings.upstream_cache_dir,
            check=False,
        )
        args = ["worktree", "add"]
        if branch_exists.returncode == 0:
            args.extend([str(self.workspace), branch])
        else:
            args.extend(
                ["-b", branch, str(self.workspace), self.settings.upstream_revision]
            )
        self._git(*args, cwd=self.settings.upstream_cache_dir)

    def _program_text(self) -> str:
        research = self.db.get_research(self.research_id) or self.context.research
        return f"""# Managed autoresearch objective

## Human objective

{research["objective"]}

## Fixed protocol

- Change only `train.py`. Never edit `prepare.py`, `program.md`, dependencies, or the evaluation harness.
- Preserve `GPT.forward(idx, targets=None, reduction='mean')`, including per-token loss for `reduction='none'`.
- Try exactly one measurable hypothesis per iteration and avoid broad refactors.
- The runner alone executes `uv run train.py`, parses one finite `val_bpb`, and accepts only strict improvement.
- `prepare.py` fixes MAX_SEQ_LEN=2048, TIME_BUDGET=300, EVAL_TOKENS, data, and evaluation.
- This GPU target allocation is {research["target_gpu_allocation"]}%; it is scheduling guidance, not a promised utilization cap.
- Continue proposing experiments until the user pauses or stops the research.
"""

    def _apply_3090_bootstrap(self) -> str | None:
        telemetry = self.context.telemetry.sample(self.context.gpu_source)
        total = telemetry.get("memory_total_mb")
        names = " ".join(str(gpu.get("name", "")) for gpu in telemetry.get("gpus", []))
        if total is None or (float(total) > 26_000 and "3090" not in names):
            return None
        train_path = self.workspace / "train.py"
        source = train_path.read_text(encoding="utf-8")
        updated, count = re.subn(
            r"(?m)^(DEVICE_BATCH_SIZE\s*=\s*)\d+", r"\g<1>32", source, count=1
        )
        if count != 1:
            raise RuntimeError("Could not locate DEVICE_BATCH_SIZE in pinned train.py")
        if updated != source:
            train_path.write_text(updated, encoding="utf-8", newline="\n")
            return "RTX 3090/24 GB bootstrap: DEVICE_BATCH_SIZE=32"
        return None

    def _sync_remote_workspace(self, *, initialize: bool) -> PurePosixPath:
        source = self.context.gpu_source
        remote_root = PurePosixPath(str(source["workspace_path"])) / self.research_id
        runner = SSHCommandRunner(source)
        if not initialize:
            exists = runner.run(
                f"test -d {shlex.quote(str(remote_root))}/.git", timeout=15
            )
            initialize = exists.returncode != 0
        if initialize:
            url = shlex.quote(self.settings.upstream_url)
            revision = shlex.quote(self.settings.upstream_revision)
            target = shlex.quote(str(remote_root))
            script = (
                f"if [ ! -d {target}/.git ]; then mkdir -p {shlex.quote(str(remote_root.parent))} && "
                f"git clone --no-checkout {url} {target}; fi && cd {target} && "
                f"git fetch --depth 1 origin {revision} && git checkout -f --detach {revision}"
            )
            result = runner.run(f"sh -lc {shlex.quote(script)}", timeout=180)
            if result.returncode != 0:
                raise RuntimeError((result.stderr or result.stdout).strip())
        self._scp(self.workspace / "train.py", f"{remote_root}/train.py")
        self._scp(self.workspace / "program.md", f"{remote_root}/program.md")
        return remote_root

    def _remote_runtime_dir(self) -> PurePosixPath:
        source = self.context.gpu_source
        return (
            PurePosixPath(str(source["workspace_path"]))
            / ".autoresearch-runtimes"
            / self.source_key
            / self.research_id
        )

    def _wsl_runtime_dir(self) -> PurePosixPath:
        if self._wsl_runtime_path is not None:
            return self._wsl_runtime_path
        configured = os.getenv("AUTORESEARCH_WSL_RUNTIME_ROOT", "").strip()
        if configured:
            root = PurePosixPath(configured)
        else:
            if not self.settings.wsl_distro:
                raise RuntimeError("WSL distribution is not configured")
            result = subprocess.run(
                [
                    "wsl.exe",
                    "-d",
                    self.settings.wsl_distro,
                    "--",
                    "sh",
                    "-s",
                ],
                input="printf '%s' \"$HOME\"",
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            home = result.stdout.strip()
            if result.returncode != 0 or not home.startswith("/"):
                raise RuntimeError("Could not resolve the WSL home directory")
            root = PurePosixPath(home) / ".cache" / "autoresearch-lab" / "runtimes"
        if not root.is_absolute():
            raise RuntimeError(
                "AUTORESEARCH_WSL_RUNTIME_ROOT must be an absolute POSIX path"
            )
        self._wsl_runtime_path = root / self.source_key / self.research_id
        return self._wsl_runtime_path

    def _uv_project_environment(self) -> str:
        if self.context.gpu_source["type"] == "remote":
            return str(self._remote_runtime_dir())
        if self.settings.wsl_distro and os.name == "nt":
            return str(self._wsl_runtime_dir())
        return str(self.runtime_dir)

    def _scp(self, local_path: Path, remote_path: str) -> None:
        source = self.context.gpu_source
        ssh = SSHCommandRunner(source)
        base = ["scp", "-q", "-P", str(source.get("port") or 22)]
        ssh_base = ssh.base_command()
        if "-i" in ssh_base:
            index = ssh_base.index("-i")
            base.extend(["-i", ssh_base[index + 1]])
        base.extend(
            [
                str(local_path),
                f"{source['username']}@{source['host']}:{shlex.quote(remote_path)}",
            ]
        )
        result = subprocess.run(
            base, capture_output=True, text=True, timeout=60, check=False
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout).strip())

    def _run_setup_command(
        self,
        command: Sequence[str],
        name: str,
        timeout: int,
        cwd: Path | None = None,
        *,
        env: Mapping[str, str] | None = None,
        stdin_text: str | None = None,
        pause_callback: Callable[[], None] | None = None,
        resume_callback: Callable[[], None] | None = None,
        terminate_callback: Callable[[], None] | None = None,
    ) -> None:
        output = self.research_log_dir / f"setup-{name}.log"
        result = run_managed_process(
            command,
            cwd=cwd or self.workspace,
            env=env or os.environ.copy(),
            output=output,
            timeout_seconds=timeout,
            runtime=self.context.runtime,
            stdin_text=stdin_text,
            pause_callback=pause_callback,
            resume_callback=resume_callback,
            terminate_callback=terminate_callback,
        )
        if result.control_error:
            self.context.runtime.control_error = result.control_error
            raise ResearchFailed(
                f"{name} process termination could not be confirmed: "
                f"{result.control_error}"
            )
        if result.stopped:
            raise InterruptedError("research stopped")
        if result.timed_out or result.returncode != 0:
            raise RuntimeError(f"{name} failed: {_tail(output)}")

    def _prepare_runtime(self) -> None:
        marker = self.research_log_dir / ".prepared"
        fingerprint_data = {
            "upstream_url": self.settings.upstream_url,
            "upstream_revision": self.settings.upstream_revision,
            "uv_binary": self.settings.uv_binary,
            "wsl_distro": self.settings.wsl_distro,
            "gpu_source_id": self.context.gpu_source["id"],
            "gpu_source_type": self.context.gpu_source["type"],
            "gpu_host": self.context.gpu_source.get("host"),
            "gpu_workspace": self.context.gpu_source.get("workspace_path"),
            "uv_project_environment": self._uv_project_environment(),
        }
        fingerprint = hashlib.sha256(
            json.dumps(fingerprint_data, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        if (
            marker.exists()
            and marker.read_text(encoding="utf-8").strip() == fingerprint
        ):
            return
        self.db.add_log(
            self.research_id,
            "environment_started",
            "Preparing the pinned autoresearch environment.",
        )
        if self.context.gpu_source["type"] == "remote":
            remote = self._sync_remote_workspace(initialize=True)
            runtime = self._remote_runtime_dir()
            runner = SSHCommandRunner(self.context.gpu_source)
            target = shlex.quote(str(remote))
            runtime_parent = shlex.quote(str(runtime.parent))
            runtime_value = shlex.quote(str(runtime))
            pid_file = f"/tmp/autoresearch-setup-{self.research_id}.pid"
            body = (
                f"mkdir -p {runtime_parent} && cd {target} && "
                f"export UV_PROJECT_ENVIRONMENT={runtime_value} && "
                f"{shlex.quote(self.settings.uv_binary)} sync --frozen && "
                f"{shlex.quote(self.settings.uv_binary)} run prepare.py"
            )
            script = build_managed_posix_script(pid_file, body)
            remote_command = shlex.join(build_setsid_wait_argv(self.research_id))
            command = [*runner.base_command(), remote_command]
            self._run_setup_command(
                command,
                "remote",
                1800,
                cwd=None,
                stdin_text=script,
                pause_callback=lambda: self._signal_remote(runner, pid_file, "STOP"),
                resume_callback=lambda: self._signal_remote(runner, pid_file, "CONT"),
                terminate_callback=lambda: self._terminate_remote(runner, pid_file),
            )
        else:
            if self.settings.wsl_distro and os.name == "nt":
                workdir = _windows_to_wsl(self.workspace)
                runtime_path = PurePosixPath(self._uv_project_environment())
                runtime_parent = shlex.quote(str(runtime_path.parent))
                runtime_value = shlex.quote(str(runtime_path))
                pid_file = f"/tmp/autoresearch-setup-{self.research_id}.pid"
                body = (
                    f"mkdir -p {runtime_parent} && "
                    f"export UV_PROJECT_ENVIRONMENT={runtime_value} && "
                    f"{shlex.quote(self.settings.uv_binary)} sync --frozen && "
                    f"{shlex.quote(self.settings.uv_binary)} run prepare.py"
                )
                script = build_managed_posix_script(pid_file, body)
                command = [
                    "wsl.exe",
                    "-d",
                    self.settings.wsl_distro,
                    "--cd",
                    workdir,
                    "--",
                    *build_setsid_wait_argv(self.research_id),
                ]

                self._run_setup_command(
                    command,
                    "prepare",
                    1800,
                    cwd=None,
                    stdin_text=script,
                    pause_callback=lambda: self._signal_wsl(pid_file, "STOP"),
                    resume_callback=lambda: self._signal_wsl(pid_file, "CONT"),
                    terminate_callback=lambda: self._terminate_wsl(pid_file),
                )
            else:
                self.runtime_dir.mkdir(parents=True, exist_ok=True)
                setup_env = os.environ.copy()
                setup_env["UV_PROJECT_ENVIRONMENT"] = self._uv_project_environment()
                self._run_setup_command(
                    [self.settings.uv_binary, "sync", "--frozen"],
                    "sync",
                    900,
                    env=setup_env,
                )
                self._run_setup_command(
                    [self.settings.uv_binary, "run", "prepare.py"],
                    "prepare",
                    1800,
                    env=setup_env,
                )
        marker.write_text(fingerprint + "\n", encoding="utf-8")
        self.db.add_log(
            self.research_id,
            "environment_ready",
            "Autoresearch data and runtime are ready.",
        )

    def prepare(self) -> None:
        recovered = self.db.close_incomplete_experiments(
            self.research_id,
            "Backend stopped before the experiment completed; candidate restored to persisted best.",
        )
        profile: str | None = None
        with self.context.repository_lock:
            self._ensure_upstream()
            self._ensure_worktree()
            research = self.db.get_research(self.research_id) or self.context.research
            self._restore_persisted_best(research, recovered)
            program_path = self.workspace / "program.md"
            desired_program = self._program_text()
            if program_path.read_text(encoding="utf-8") != desired_program:
                program_path.write_text(desired_program, encoding="utf-8", newline="\n")
            if research.get("baseline_value") is None:
                profile = self._apply_3090_bootstrap()
            if self._git("status", "--porcelain", cwd=self.workspace).stdout.strip():
                self._commit("chore: initialize managed research objective")
            head = self._git("rev-parse", "HEAD", cwd=self.workspace).stdout.strip()
            self._set_best_commit(head)
            self.prepare_hash = _sha256(self.workspace / "prepare.py")
        self.db.update_research(
            self.research_id, {"workspace_path": str(self.workspace)}
        )
        if recovered:
            self.db.add_log(
                self.research_id,
                "experiment_recovered",
                "Incomplete experiments were closed and the workspace was restored to the persisted best commit.",
                level="error",
                data={"experiment_numbers": recovered, "best_git_commit": head},
            )
        if profile:
            self.db.add_log(self.research_id, "gpu_profile_applied", profile)
        self._prepare_runtime()

    def _assert_prepare_unchanged(self) -> None:
        if (
            not self.prepare_hash
            or _sha256(self.workspace / "prepare.py") != self.prepare_hash
        ):
            self._git(
                "checkout", "HEAD", "--", "prepare.py", cwd=self.workspace, check=False
            )
            raise RuntimeError(
                "prepare.py changed; the candidate was rejected and the fixed harness restored"
            )

    def _clean_candidate(self, base_sha: str) -> None:
        self._git("reset", "--hard", base_sha, cwd=self.workspace)
        status = self._git(
            "status", "--porcelain", "--untracked-files=all", cwd=self.workspace
        ).stdout.splitlines()
        for line in status:
            relative = line[3:].strip().strip('"')
            path = (self.workspace / relative).resolve()
            try:
                path.relative_to(self.workspace.resolve())
            except ValueError:
                continue
            if path.is_file() or path.is_symlink():
                path.unlink(missing_ok=True)

    def _agent_prompt(self) -> str:
        recent = self.db.list_experiments(self.research_id)[-12:]
        history = [
            {
                "experiment_number": item["experiment_number"],
                "hypothesis": item["hypothesis"],
                "metric_value": item["metric_value"],
                "accepted": item["accepted"],
                "error": item["error"],
            }
            for item in recent
        ]
        return f"""Read program.md, prepare.py, and train.py. Propose and implement exactly one measurable experiment.
You may edit only train.py. Do not run training, Git commands, package installation, or edit any other file.
Preserve the model forward/evaluation contract. Avoid uncontrolled refactors.
The runner will execute the fixed command `uv run train.py` after validating your diff.
Return only the required JSON object matching the supplied schema.
Recent persisted experiment history: {json.dumps(history, ensure_ascii=False)}
"""

    def _agent_command(self, scratch: Path, output_json: Path) -> list[str]:
        schema = Path(__file__).resolve().parents[1] / "agent-output.schema.json"
        context = getattr(self, "context", None)
        research = getattr(context, "research", {})
        researcher_model = str(
            research.get("researcher_model_id")
            or getattr(self.settings, "default_researcher_model", "")
        ).strip()
        command = [self.settings.codex_binary]
        if researcher_model:
            command.extend(researcher_model_command_args(researcher_model))
        command.extend([
            "exec",
            "--json",
            "--ephemeral",
            "--config",
            f'model_reasoning_effort="{self.settings.agent_reasoning_effort}"',
            "--sandbox",
            "workspace-write",
            "--cd",
            str(scratch),
            "--output-schema",
            str(schema),
            "--output-last-message",
            str(output_json),
            "-",
        ])
        return command

    def _release_local_researcher(self) -> None:
        model_id = str(self.context.research.get("researcher_model_id") or "")
        if not model_id.startswith("ollama:"):
            return
        model_name = model_name_from_id(model_id)
        client = OllamaClient()
        client.unload(model_name)
        running_names = {
            str(item.get("name") or item.get("model") or "")
            for item in client.running_models()
        }
        if model_name in running_names:
            raise RuntimeError(
                "The local research agent is still using GPU memory, so the measured training run was not started."
            )
        self.db.add_log(
            self.research_id,
            "local_researcher_released",
            f"Released {model_name} before the measured GPU run.",
            data={"researcher_model_id": model_id},
        )

    def _run_agent(
        self, experiment_number: int, base_sha: str, attempt_number: int
    ) -> dict[str, Any]:
        stem = f"agent-{experiment_number}-attempt-{attempt_number}"
        output_json = self.research_log_dir / f"{stem}.json"
        output_log = self.research_log_dir / f"{stem}.log"
        output_json.unlink(missing_ok=True)
        scratch = (
            self.settings.data_dir
            / "agent-sandboxes"
            / self.research_id
            / f"experiment-{experiment_number}-attempt-{attempt_number}"
        )
        scratch.parent.mkdir(parents=True, exist_ok=True)
        with self.context.repository_lock:
            self._git(
                "worktree",
                "remove",
                "--force",
                str(scratch),
                cwd=self.settings.upstream_cache_dir,
                check=False,
            )
            if scratch.exists():
                shutil.rmtree(scratch)
            self._git(
                "worktree",
                "add",
                "--detach",
                str(scratch),
                base_sha,
                cwd=self.settings.upstream_cache_dir,
            )
        try:
            command = self._agent_command(scratch, output_json)
            result = run_managed_process(
                command,
                cwd=scratch,
                env=os.environ.copy(),
                output=output_log,
                timeout_seconds=self.settings.agent_timeout_seconds,
                runtime=self.context.runtime,
                stdin_text=self._agent_prompt(),
            )
            usage = parse_codex_jsonl_usage(
                output_log.read_text(encoding="utf-8", errors="replace")
                if output_log.exists()
                else ""
            )
            if usage.has_usage:
                self.db.record_codex_token_usage(
                    self.research_id,
                    "candidate",
                    call_id=f"{self.research_id}:{stem}",
                    input_tokens=usage.input_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                    output_tokens=usage.output_tokens,
                    reasoning_output_tokens=usage.reasoning_output_tokens,
                )
            if result.control_error:
                self.context.runtime.control_error = result.control_error
                raise ResearchFailed(
                    "Candidate process termination could not be confirmed: "
                    f"{result.control_error}"
                )
            if result.stopped:
                raise InterruptedError("research stopped")
            if result.timed_out:
                raise AgentGenerationTimeout(
                    "Codex candidate generation timed out after "
                    f"{self.settings.agent_timeout_seconds} active seconds; "
                    f"diagnostics are preserved in {output_log.name}"
                )
            if result.returncode != 0:
                raise RuntimeError(
                    f"Codex candidate generation failed: {_tail(output_log)}"
                )
            try:
                payload = json.loads(output_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Codex did not produce valid structured JSON: {exc}"
                ) from exc
            required = {
                "hypothesis",
                "files_changed",
                "change_summary",
                "evaluation_command",
                "expected_improvement",
            }
            if set(payload) != required:
                raise RuntimeError(
                    "Codex JSON fields did not match the required schema"
                )
            if (
                payload["files_changed"] != ["train.py"]
                or payload["evaluation_command"] != "uv run train.py"
            ):
                raise RuntimeError(
                    "Codex proposed work outside the fixed train.py/evaluation contract"
                )
            self._validate_candidate_diff(scratch)
            candidate = scratch / "train.py"
            if candidate.is_symlink() or candidate.stat().st_size > 2_000_000:
                raise RuntimeError(
                    "Candidate train.py must be a regular file smaller than 2 MB"
                )
            shutil.copyfile(candidate, self.workspace / "train.py")
            return payload
        finally:
            try:
                with self.context.repository_lock:
                    self._git(
                        "worktree",
                        "remove",
                        "--force",
                        str(scratch),
                        cwd=self.settings.upstream_cache_dir,
                        check=False,
                    )
                    self._git(
                        "worktree",
                        "prune",
                        cwd=self.settings.upstream_cache_dir,
                        check=False,
                    )
                if scratch.exists():
                    shutil.rmtree(scratch)
            except Exception:
                # A cleanup failure must never mask an unconfirmed process-group
                # termination. The supervisor keeps the GPU reserved when this
                # control error reaches the worker boundary.
                if not self.context.runtime.control_error:
                    raise

    def _validate_candidate_diff(self, workspace: Path) -> None:
        if (
            not self.prepare_hash
            or _sha256(workspace / "prepare.py") != self.prepare_hash
        ):
            raise RuntimeError(
                "Candidate changed the fixed prepare.py evaluation harness"
            )
        lines = self._git(
            "status", "--porcelain", "--untracked-files=all", cwd=workspace
        ).stdout.splitlines()
        paths = {line[3:].strip().strip('"').replace("\\", "/") for line in lines}
        if paths != {"train.py"}:
            raise RuntimeError(
                f"Candidate must change only train.py; observed: {sorted(paths)}"
            )
        diff = self._git("diff", "--", "train.py", cwd=workspace).stdout
        if not diff.strip():
            raise RuntimeError("Codex returned no train.py change")

    def _ignored_state(self) -> dict[str, str]:
        result = self._git(
            "ls-files",
            "--others",
            "--ignored",
            "--exclude-standard",
            "--",
            ".",
            ":(exclude).venv/**",
            ":(exclude)**/__pycache__/**",
            ":(exclude).pytest_cache/**",
            cwd=self.workspace,
        )
        state: dict[str, str] = {}
        for relative in result.stdout.splitlines():
            normalized = relative.strip().replace("\\", "/")
            if not normalized or normalized.endswith(".pyc"):
                continue
            path = (self.workspace / normalized).resolve()
            try:
                path.relative_to(self.workspace.resolve())
            except ValueError as exc:
                raise RuntimeError(
                    "Ignored path escaped the research workspace"
                ) from exc
            if (
                path.is_symlink()
                or not path.is_file()
                or path.stat().st_size > 10_000_000
            ):
                raise RuntimeError(f"Unsafe ignored candidate artifact: {normalized}")
            state[normalized] = _sha256(path)
        return state

    def _capture_evaluation_state(self) -> tuple[str, str, dict[str, str]]:
        return (
            _sha256(self.workspace / "prepare.py"),
            _sha256(self.workspace / "train.py"),
            self._ignored_state(),
        )

    def _verify_evaluation_state(self, before: tuple[str, str, dict[str, str]]) -> None:
        prepare_hash, train_hash, ignored_state = before
        if _sha256(self.workspace / "prepare.py") != prepare_hash:
            raise RuntimeError("Evaluation modified fixed prepare.py")
        if _sha256(self.workspace / "train.py") != train_hash:
            raise RuntimeError("Evaluation modified train.py while it was running")
        lines = self._git(
            "status", "--porcelain", "--untracked-files=all", cwd=self.workspace
        ).stdout.splitlines()
        paths = {line[3:].strip().strip('"').replace("\\", "/") for line in lines}
        if not paths.issubset({"train.py"}):
            raise RuntimeError(
                f"Evaluation changed out-of-contract paths: {sorted(paths)}"
            )
        if self._ignored_state() != ignored_state:
            raise RuntimeError(
                "Evaluation changed ignored files outside runtime cache directories"
            )
        self._assert_prepare_unchanged()

    def _evaluation_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["UV_PROJECT_ENVIRONMENT"] = self._uv_project_environment()
        env["CUDA_VISIBLE_DEVICES"] = os.getenv(
            "AUTORESEARCH_CUDA_VISIBLE_DEVICES", "0"
        )
        env["AUTORESEARCH_TARGET_ALLOCATION"] = str(
            self.context.research["target_gpu_allocation"]
        )
        if self.settings.use_cuda_mps:
            env["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(
                self.context.research["target_gpu_allocation"]
            )
        return env

    def _signal_remote(
        self, runner: SSHCommandRunner, pid_file: str, signal_name: str
    ) -> None:
        command = build_posix_group_signal_script(
            pid_file, self.research_id, signal_name
        )
        result = runner.run("sh -s", timeout=10, stdin_text=command)
        if result.returncode != 0:
            raise RuntimeError(
                f"Remote process-group signal {signal_name} failed: "
                f"{(result.stderr or result.stdout).strip()[:500]}"
            )

    def _terminate_remote(self, runner: SSHCommandRunner, pid_file: str) -> None:
        command = build_posix_group_terminate_script(pid_file, self.research_id)
        result = runner.run("sh -s", timeout=10, stdin_text=command)
        if result.returncode != 0:
            raise RuntimeError(
                "Remote process-group termination failed: "
                f"{(result.stderr or result.stdout).strip()[:500]}"
            )

    def _wsl_control(self, shell_command: str) -> subprocess.CompletedProcess[bytes]:
        if not self.settings.wsl_distro:
            raise RuntimeError("WSL distribution is not configured")
        return subprocess.run(
            [
                "wsl.exe",
                "-d",
                self.settings.wsl_distro,
                "--",
                "sh",
                "-s",
            ],
            input=shell_command.encode("utf-8"),
            capture_output=True,
            timeout=10,
            check=False,
        )

    def _signal_wsl(self, pid_file: str, signal_name: str) -> None:
        result = self._wsl_control(
            build_posix_group_signal_script(pid_file, self.research_id, signal_name)
        )
        if result.returncode != 0:
            raise RuntimeError(f"WSL process-group signal {signal_name} failed")

    def _terminate_wsl(self, pid_file: str) -> None:
        result = self._wsl_control(
            build_posix_group_terminate_script(pid_file, self.research_id)
        )
        if result.returncode != 0:
            raise RuntimeError("WSL process-group termination failed")

    def _watch_evaluation_tokens(
        self, output: Path, experiment_id: str, stopped: threading.Event
    ) -> None:
        existing = self.db.get_experiment(experiment_id) or {}
        starting_tokens = max(0, int(existing.get("token_count") or 0))
        last_value = starting_tokens
        while True:
            if output.exists():
                try:
                    text = output.read_text(encoding="utf-8", errors="replace")
                    current = starting_tokens + live_tokens_from_output(text)
                    if current > last_value:
                        self.db.update_experiment_token_count(experiment_id, current)
                        last_value = current
                except OSError:
                    pass
            if stopped.wait(0.75):
                return

    def _run_evaluation(
        self, experiment_number: int, experiment_id: str
    ) -> tuple[ProcessResult, Path]:
        output = self.research_log_dir / f"experiment-{experiment_number}.log"
        output.unlink(missing_ok=True)
        env = self._evaluation_env()
        pause_callback = resume_callback = terminate_callback = None
        stdin_text: str | None = None
        cwd: Path | None = self.workspace
        if self.context.gpu_source["type"] == "remote":
            remote = self._sync_remote_workspace(initialize=False)
            remote_path = str(remote)
            pid_file = f"/tmp/autoresearch-{self.research_id}.pid"
            runner = SSHCommandRunner(self.context.gpu_source)
            remote_env = [
                f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])}",
                f"UV_PROJECT_ENVIRONMENT={shlex.quote(env['UV_PROJECT_ENVIRONMENT'])}",
            ]
            if self.settings.use_cuda_mps:
                remote_env.append(
                    f"CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}"
                )
            body = (
                f"cd {shlex.quote(remote_path)} && "
                f"env {' '.join(remote_env)} {shlex.quote(self.settings.uv_binary)} run train.py"
            )
            script = build_managed_posix_script(pid_file, body)
            remote_command = shlex.join(build_setsid_wait_argv(self.research_id))
            command = [*runner.base_command(), remote_command]
            stdin_text = script
            cwd = None
            pause_callback = lambda: self._signal_remote(runner, pid_file, "STOP")
            resume_callback = lambda: self._signal_remote(runner, pid_file, "CONT")
            terminate_callback = lambda: self._terminate_remote(runner, pid_file)
        elif self.settings.wsl_distro and os.name == "nt":
            workdir = _windows_to_wsl(self.workspace)
            pid_file = f"/tmp/autoresearch-{self.research_id}.pid"
            environment = (
                f"CUDA_VISIBLE_DEVICES={shlex.quote(env['CUDA_VISIBLE_DEVICES'])} "
                "UV_PROJECT_ENVIRONMENT="
                f"{shlex.quote(env['UV_PROJECT_ENVIRONMENT'])}"
            )
            if self.settings.use_cuda_mps:
                environment += f" CUDA_MPS_ACTIVE_THREAD_PERCENTAGE={env['CUDA_MPS_ACTIVE_THREAD_PERCENTAGE']}"
            body = (
                f"env {environment} {shlex.quote(self.settings.uv_binary)} run train.py"
            )
            script = build_managed_posix_script(pid_file, body)
            command = [
                "wsl.exe",
                "-d",
                self.settings.wsl_distro,
                "--cd",
                workdir,
                "--",
                *build_setsid_wait_argv(self.research_id),
            ]
            stdin_text = script

            pause_callback = lambda: self._signal_wsl(pid_file, "STOP")
            resume_callback = lambda: self._signal_wsl(pid_file, "CONT")
            terminate_callback = lambda: self._terminate_wsl(pid_file)
            cwd = None
        else:
            command = [self.settings.uv_binary, "run", "train.py"]
        token_watch_stopped = threading.Event()
        token_watcher = threading.Thread(
            target=self._watch_evaluation_tokens,
            args=(output, experiment_id, token_watch_stopped),
            name=f"token-watch-{experiment_number}",
            daemon=True,
        )
        token_watcher.start()
        try:
            result = run_managed_process(
                command,
                cwd=cwd,
                env=env,
                output=output,
                timeout_seconds=self.settings.experiment_timeout_seconds,
                runtime=self.context.runtime,
                stdin_text=stdin_text,
                pause_callback=pause_callback,
                resume_callback=resume_callback,
                terminate_callback=terminate_callback,
            )
        finally:
            token_watch_stopped.set()
            token_watcher.join(timeout=2)
        return result, output

    def _reduce_for_oom(self) -> str | None:
        path = self.workspace / "train.py"
        source = path.read_text(encoding="utf-8")
        batch_match = re.search(r"(?m)^DEVICE_BATCH_SIZE\s*=\s*(\d+)", source)
        if not batch_match:
            return None
        batch = int(batch_match.group(1))
        if batch > 8:
            new_batch = 32 if batch > 32 else 16 if batch > 16 else 8
            updated = re.sub(
                r"(?m)^(DEVICE_BATCH_SIZE\s*=\s*)\d+",
                rf"\g<1>{new_batch}",
                source,
                count=1,
            )
            path.write_text(updated, encoding="utf-8", newline="\n")
            return f"OOM retry reduced DEVICE_BATCH_SIZE from {batch} to {new_batch}"
        sequence_match = re.search(r"(?m)^TRAIN_SEQ_LEN\s*=\s*(\d+)", source)
        if sequence_match:
            sequence_length = int(sequence_match.group(1))
        else:
            sequence_length = 2048
            source, inserted = re.subn(
                r"(?m)^(DEVICE_BATCH_SIZE\s*=\s*\d+[^\n]*\n)",
                r"\1TRAIN_SEQ_LEN = MAX_SEQ_LEN  # training-only OOM control; evaluation remains fixed\n",
                source,
                count=1,
            )
            if inserted != 1:
                return None
            source, token_count = re.subn(
                r"(?m)^(tokens_per_fwdbwd\s*=\s*DEVICE_BATCH_SIZE\s*\*\s*)MAX_SEQ_LEN",
                r"\1TRAIN_SEQ_LEN",
                source,
                count=1,
            )
            source, loader_count = re.subn(
                r'make_dataloader\(tokenizer, DEVICE_BATCH_SIZE, MAX_SEQ_LEN, "train"\)',
                'make_dataloader(tokenizer, DEVICE_BATCH_SIZE, TRAIN_SEQ_LEN, "train")',
                source,
                count=1,
            )
            if token_count != 1 or loader_count != 1:
                return None
        if sequence_length > 512:
            new_sequence_length = sequence_length // 2
            updated = re.sub(
                r"(?m)^(TRAIN_SEQ_LEN\s*=\s*)(?:MAX_SEQ_LEN|\d+)",
                rf"\g<1>{new_sequence_length}",
                source,
                count=1,
            )
            path.write_text(updated, encoding="utf-8", newline="\n")
            return (
                "OOM retry reduced training-only TRAIN_SEQ_LEN from "
                f"{sequence_length} to {new_sequence_length}; validation remains fixed at 2048"
            )
        depth_match = re.search(r"(?m)^DEPTH\s*=\s*(\d+)", source)
        if not depth_match:
            return None
        depth = int(depth_match.group(1))
        new_depth = 6 if depth > 6 else 4 if depth > 4 else None
        if new_depth is None:
            return None
        updated = re.sub(
            r"(?m)^(DEPTH\s*=\s*)\d+", rf"\g<1>{new_depth}", source, count=1
        )
        path.write_text(updated, encoding="utf-8", newline="\n")
        return f"OOM retry reduced DEPTH from {depth} to {new_depth}"

    def _acquire_gpu(self) -> bool:
        while not self.context.runtime.stop_event.is_set():
            if not self.context.runtime.wait_until_running():
                return False
            if self.context.gpu_lock.acquire(timeout=0.5):
                return True
        return False

    def _record_agent_failure(
        self,
        error: Exception,
        *,
        experiment_number: int,
        attempt_number: int,
    ) -> int:
        self._agent_failure_streak += 1
        retry_delay = agent_retry_delay_seconds(self._agent_failure_streak)
        self.context.runtime.set_phase("candidate_retry_backoff")
        next_attempt = attempt_number + 1
        timed_out = isinstance(error, AgentGenerationTimeout)
        kind = "timeout" if timed_out else "candidate_error"
        message = (
            f"{error}. Candidate attempt {next_attempt} is scheduled in "
            f"{retry_delay}s unless the research is paused or stopped."
        )
        self.db.add_log(
            self.research_id,
            "agent_error",
            message,
            level="error",
            data={
                "experiment_number": experiment_number,
                "attempt_number": attempt_number,
                "next_attempt_number": next_attempt,
                "consecutive_failures": self._agent_failure_streak,
                "failure_kind": kind,
                "retry_delay_seconds": retry_delay,
                "agent_timeout_seconds": self.settings.agent_timeout_seconds,
                "reasoning_effort": self.settings.agent_reasoning_effort,
                "diagnostic_log": (
                    f"agent-{experiment_number}-attempt-{attempt_number}.log"
                ),
            },
        )
        stopped = self.context.runtime.stop_event.wait(retry_delay)
        if not stopped:
            self.context.runtime.set_phase("ready")
        return retry_delay

    def run_iteration(self) -> None:
        research = self.db.get_research(self.research_id) or self.context.research
        best = research.get("best_value")
        base_sha = (
            research.get("best_git_commit")
            or self._git("rev-parse", "HEAD", cwd=self.workspace).stdout.strip()
        )
        current_head = self._git("rev-parse", "HEAD", cwd=self.workspace).stdout.strip()
        if current_head != base_sha:
            self._git("reset", "--hard", str(base_sha), cwd=self.workspace)
        experiment_number = self.db.next_experiment_number(self.research_id)
        baseline = best is None
        payload: dict[str, Any]
        if baseline:
            self.context.runtime.set_phase("preparing_experiment")
            payload = {
                "hypothesis": "Establish the pinned autoresearch baseline on the selected GPU profile.",
                "change_summary": "Baseline with the runner-managed RTX 3090 batch-size profile.",
            }
        else:
            attempt_number = self._agent_failure_streak + 1
            self.context.runtime.set_phase("candidate_generation")
            self.db.add_log(
                self.research_id,
                "agent_started",
                (
                    f"Generating candidate attempt {attempt_number} for experiment "
                    f"{experiment_number} with a "
                    f"{self.settings.agent_timeout_seconds}s active watchdog."
                ),
                data={
                    "experiment_number": experiment_number,
                    "attempt_number": attempt_number,
                    "agent_timeout_seconds": self.settings.agent_timeout_seconds,
                    "reasoning_effort": self.settings.agent_reasoning_effort,
                },
            )
            try:
                payload = self._run_agent(
                    experiment_number, str(base_sha), attempt_number
                )
                self._validate_candidate_diff(self.workspace)
                self._release_local_researcher()
            except InterruptedError:
                self._clean_candidate(base_sha)
                self.context.runtime.set_phase("stopping")
                return
            except ResearchFailed:
                try:
                    self._clean_candidate(base_sha)
                except Exception:
                    if not self.context.runtime.control_error:
                        raise
                raise
            except Exception as exc:  # noqa: BLE001 - candidate failures are persisted and the worktree is restored
                self._clean_candidate(base_sha)
                self._record_agent_failure(
                    exc,
                    experiment_number=experiment_number,
                    attempt_number=attempt_number,
                )
                return
            self._agent_failure_streak = 0
            self.context.runtime.set_phase("preparing_experiment")

        experiment = self.db.create_experiment(
            self.research_id, payload["hypothesis"], payload["change_summary"], best
        )
        experiment_id = experiment["id"]
        self.db.add_log(
            self.research_id,
            "experiment_started",
            f"Experiment {experiment['experiment_number']} started: {payload['change_summary']}",
            experiment_id=experiment_id,
            data={
                "experiment_number": experiment["experiment_number"],
                "hypothesis": payload["hypothesis"],
            },
        )
        candidate_sha = base_sha
        if not baseline:
            candidate_sha = self._commit(
                f"experiment {experiment['experiment_number']}: {payload['change_summary']}"
            )
            self.db.add_log(
                self.research_id,
                "change_applied",
                payload["change_summary"],
                experiment_id=experiment_id,
                data={"git_commit": candidate_sha},
            )

        self.context.runtime.set_phase("waiting_for_gpu")
        if not self._acquire_gpu():
            self._clean_candidate(base_sha)
            self.db.finish_experiment(
                experiment_id,
                metric_value=None,
                accepted=False,
                git_commit=candidate_sha,
                error="Stopped before GPU execution began",
            )
            self.context.runtime.set_phase("stopping")
            return
        self.context.runtime.set_phase("gpu_evaluation")
        allocation_mode = (
            "CUDA MPS active-thread percentage"
            if self.settings.use_cuda_mps
            else "exclusive serialized access"
        )
        self.db.add_log(
            self.research_id,
            "gpu_evaluation_started",
            (
                "GPU test started. Live utilization can be low briefly during "
                "setup, compilation, and validation."
            ),
            experiment_id=experiment_id,
            data={
                "phase": "gpu_evaluation",
                "target_gpu_allocation": research["target_gpu_allocation"],
                "allocation_mode": allocation_mode,
                "target_is_hard_utilization_guarantee": False,
                "live_utilization_source": "nvidia-smi_device_sample",
                "mfu_reference": "H100_BF16_PEAK_FLOPS_989.5e12",
            },
        )
        execution_error: str | None = None
        evaluation: ProcessResult | None = None
        output_path: Path | None = None
        safety_changes: list[str] = []
        try:
            while True:
                self._assert_prepare_unchanged()
                evaluation_state = self._capture_evaluation_state()
                evaluation, output_path = self._run_evaluation(
                    experiment["experiment_number"], experiment_id
                )
                if evaluation.control_error:
                    self.context.runtime.control_error = evaluation.control_error
                    message = (
                        "GPU evaluation process termination could not be confirmed: "
                        f"{evaluation.control_error}"
                    )
                    persistence_errors: list[str] = []
                    try:
                        self._clean_candidate(base_sha)
                    except Exception as exc:  # noqa: BLE001 - fail-safe cleanup is best effort
                        persistence_errors.append(f"candidate cleanup failed: {exc}")
                    try:
                        self.db.finish_experiment(
                            experiment_id,
                            metric_value=None,
                            accepted=False,
                            git_commit=candidate_sha,
                            error=message,
                        )
                    except Exception as exc:  # noqa: BLE001 - preserve the terminal control error
                        persistence_errors.append(f"result persistence failed: {exc}")
                    if persistence_errors:
                        message += ". " + "; ".join(persistence_errors)
                    raise ResearchFailed(message)
                self._verify_evaluation_state(evaluation_state)
                output_text = (
                    output_path.read_text(encoding="utf-8", errors="replace")
                    if output_path.exists()
                    else ""
                )
                if evaluation.stopped:
                    self._clean_candidate(base_sha)
                    stop_error = (
                        f"Stop requested but process-group termination was not confirmed: {evaluation.control_error}"
                        if evaluation.control_error
                        else "Stopped by user"
                    )
                    self.db.finish_experiment(
                        experiment_id,
                        metric_value=None,
                        accepted=False,
                        git_commit=candidate_sha,
                        error=stop_error,
                    )
                    return
                if OOM_PATTERN.search(output_text):
                    adjustment = self._reduce_for_oom()
                    if adjustment:
                        safety_changes.append(adjustment)
                        self.db.append_experiment_change_summary(
                            experiment_id, adjustment
                        )
                        self.db.add_log(
                            self.research_id,
                            "oom_retry",
                            adjustment,
                            level="warning",
                            experiment_id=experiment_id,
                        )
                        continue
                break
        except ResearchFailed:
            raise
        except Exception as exc:  # noqa: BLE001 - execution boundary must convert all failures to experiment records
            execution_error = str(exc)
        finally:
            self.context.gpu_lock.release()
            self.context.runtime.set_phase(
                "stopping"
                if self.context.runtime.stop_event.is_set()
                else "processing_results"
            )

        output_text = (
            output_path.read_text(encoding="utf-8", errors="replace")
            if output_path and output_path.exists()
            else ""
        )
        error: str | None = execution_error
        metric: float | None = None
        summary: EvaluationSummary | None = None
        if evaluation is None:
            error = error or "Evaluation did not start"
        elif evaluation.timed_out:
            error = f"Evaluation exceeded the {self.settings.experiment_timeout_seconds}s watchdog"
        else:
            try:
                summary = parse_evaluation_summary(
                    output_text,
                    evaluation.returncode,
                    active_duration_seconds=evaluation.active_duration_seconds,
                )
                metric = summary.val_bpb
            except ValueError as exc:
                suffix = _tail(output_path)
                error = f"{exc}. {suffix}".strip()[:3000]

        if error:
            self._clean_candidate(base_sha)
            self.db.finish_experiment(
                experiment_id,
                metric_value=None,
                accepted=False,
                git_commit=candidate_sha,
                error=error,
            )
            self.db.add_log(
                self.research_id,
                "experiment_error",
                f"Experiment {experiment['experiment_number']} failed: {error}",
                level="error",
                experiment_id=experiment_id,
            )
            self._evaluation_failure_streak += 1
            non_recoverable_environment_error = any(
                marker in error
                for marker in (
                    "Cannot install kernel from repo kernels-community/flash-attn3",
                    "does not have one of build variants",
                )
            )
            if non_recoverable_environment_error:
                raise ResearchFailed(
                    "The pinned Flash Attention GPU kernel is unavailable in this Windows runtime. "
                    "Run this training research through the configured WSL GPU runtime."
                )
            if self._evaluation_failure_streak >= 3:
                raise ResearchFailed(
                    "Three consecutive GPU evaluations failed before producing a metric. "
                    "The research was stopped to prevent an endless retry loop."
                )
            self.context.runtime.set_phase("ready")
            return

        assert metric is not None and summary is not None
        self._evaluation_failure_streak = 0
        accepted = is_better(metric, best, str(research["metric_direction"]))
        if safety_changes:
            candidate_sha = self._commit("", amend=True)
        if accepted:
            updates: dict[str, Any] = {
                "best_value": metric,
                "best_git_commit": candidate_sha,
            }
            if research.get("baseline_value") is None:
                updates["baseline_value"] = metric
            self.db.finish_experiment(
                experiment_id,
                metric_value=metric,
                accepted=True,
                git_commit=candidate_sha,
                error=None,
                token_count=round(summary.total_tokens_m * 1_000_000),
                research_updates=updates,
            )
            try:
                with self.context.repository_lock:
                    self._git(
                        "update-ref", self._best_ref, candidate_sha, cwd=self.workspace
                    )
            except RuntimeError as exc:
                self.db.add_log(
                    self.research_id,
                    "best_ref_warning",
                    f"Best commit is persisted in SQLite, but the convenience Git ref update failed: {exc}",
                    level="warning",
                    experiment_id=experiment_id,
                )
            message = f"Experiment {experiment['experiment_number']} accepted at {metric:.6f}."
            event_type = "experiment_accepted"
        else:
            self._clean_candidate(base_sha)
            self.db.finish_experiment(
                experiment_id,
                metric_value=metric,
                accepted=False,
                git_commit=candidate_sha,
                error=None,
                token_count=round(summary.total_tokens_m * 1_000_000),
            )
            message = f"Experiment {experiment['experiment_number']} rejected at {metric:.6f}; best remains {best:.6f}."
            event_type = "experiment_rejected"
        self.db.add_log(
            self.research_id,
            "experiment_completed",
            f"Experiment {experiment['experiment_number']} completed with val_bpb {metric:.6f}.",
            experiment_id=experiment_id,
            data={
                "metric_name": "val_bpb",
                "metric_value": metric,
                "peak_vram_mb": summary.peak_vram_mb,
                "mfu_percent": summary.mfu_percent,
                "mfu_reference": "H100_BF16_PEAK_FLOPS_989.5e12",
                "training_seconds": summary.training_seconds,
                "total_seconds": summary.total_seconds,
                "num_steps": summary.num_steps,
                "total_tokens_M": summary.total_tokens_m,
                "active_duration_seconds": evaluation.active_duration_seconds,
            },
        )
        self.db.add_log(
            self.research_id,
            event_type,
            message,
            experiment_id=experiment_id,
            data={
                "metric_value": metric,
                "previous_best": best,
                "git_commit": candidate_sha,
            },
        )
        self.context.runtime.set_phase("ready")
