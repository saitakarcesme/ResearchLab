# Autoresearch backend

Python 3.11 FastAPI service for SQLite persistence, real NVIDIA telemetry, research process control, and the pinned Karpathy autoresearch adapter.

## Run locally

From the repository root:

```powershell
py -3.11 -m venv backend/.venv
backend/.venv/Scripts/python -m pip install -r backend/requirements.txt
backend/.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 7331
```

The API initializes `backend/data/autoresearch.sqlite3` and a real local GPU source. It never seeds research, experiment, or article demo records. Swagger documentation is available at `http://127.0.0.1:7331/docs`.

Telemetry works without enabling training. Starting Codex or GPU workloads is deliberately opt-in:

```powershell
$env:AUTORESEARCH_ENABLE_EXECUTION = "1"
backend/.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 7331
```

Creating a research does not start GPU work. When execution is enabled, creation first asks Codex for a read-only, schema-constrained short title and measurable objective; an unavailable Codex falls back deterministically and records a warning. Call `POST /api/researches/{id}/start` explicitly to begin experiments. Each research uses an isolated pinned Git worktree and stores its last validated Git commit for crash-safe resume.

Candidate generation uses a 600-second active-time watchdog and explicitly selects `high` reasoning effort so personal Codex defaults such as `ultra` cannot silently exceed the service budget. Failures retain a separate log per attempt and retry continuously with stop-aware exponential backoff from 5 seconds up to 5 minutes. Override these defaults with `AUTORESEARCH_AGENT_TIMEOUT` and `AUTORESEARCH_AGENT_REASONING_EFFORT`.

By default, experiments on the same physical GPU source are serialized for stable metric comparisons; separate GPU sources may run concurrently. Same-GPU 60/40-style overlap is enabled only with `AUTORESEARCH_USE_CUDA_MPS=1`, and only when NVIDIA CUDA MPS has already been configured and started on that host. In MPS mode each process receives `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE`, but target allocations remain scheduler hints rather than guaranteed utilization caps. The UI/API must treat live `nvidia-smi` telemetry as the truth.

Research detail responses expose `runtime.phase` (`candidate_generation`, `candidate_retry_backoff`, `waiting_for_gpu`, `gpu_evaluation`, or result-processing phases) plus explicit allocation semantics. Without MPS, the selected GPU is serialized exclusively and the target percentage is metadata; with MPS it becomes an active-thread percentage, still not a hard device-utilization guarantee. Live utilization is an instantaneous `nvidia-smi` device sample and normally dips during startup, `torch.compile`, and validation. The pinned runner's `mfu_percent` is throughput normalized to the H100 BF16 peak of 989.5 TFLOP/s, not the RTX GPU-busy percentage.

For CUDA execution through WSL, set `AUTORESEARCH_WSL_DISTRO=Ubuntu-24.04` and install `uv` plus the pinned project dependencies in that distro. Native execution is the default. The evaluation watchdog defaults to 600 seconds.

Each source/research pair gets an absolute `UV_PROJECT_ENVIRONMENT` outside the editable research worktree. Native execution uses `AUTORESEARCH_RUNTIMES` (default: `backend/data/runtimes`). WSL uses the distro's fast ext4-backed `$HOME/.cache/autoresearch-lab/runtimes` by default; override it with an absolute POSIX `AUTORESEARCH_WSL_RUNTIME_ROOT`. SSH sources use a sibling `.autoresearch-runtimes` directory under the configured remote workspace root. The runtime path is part of the preparation fingerprint.

Metric acceptance requires exactly one complete, ordered upstream summary as the terminal output block. The reported training time and the backend-measured active process time must each be at least 299 seconds; 299 is a one-second tolerance around the pinned 300-second upstream budget. This verifies the canonical pinned runner contract and rejects truncated or short synthetic output, but it is not a hostile-code sandbox: a fully adversarial `train.py` would require an evaluator outside the candidate's trust boundary.

## Remote SSH sources

Remote sources use the system `ssh`/`scp` clients in batch mode. `auth_method=agent` uses an already-loaded SSH agent. `auth_method=key_env` reads a key **path**, never key contents, from `AUTORESEARCH_SSH_KEY_<SOURCE_ID>` (uppercase with punctuation replaced by `_`) or `AUTORESEARCH_SSH_KEY_DEFAULT`. No password, token, or private-key content is accepted by or stored in SQLite.

## Useful configuration

- `AUTORESEARCH_DATA_DIR`, `AUTORESEARCH_DATABASE`, `AUTORESEARCH_WORKSPACES`, `AUTORESEARCH_LOGS`, `AUTORESEARCH_RUNTIMES`
- `AUTORESEARCH_CORS_ORIGINS` (comma-separated; localhost/127.0.0.1 ports 3000, 3001, 4173, and 5173 by default)
- `AUTORESEARCH_UPSTREAM` and `AUTORESEARCH_UPSTREAM_REVISION`
- `AUTORESEARCH_EXPERIMENT_TIMEOUT` (minimum 60; default 600)
- `AUTORESEARCH_AGENT_TIMEOUT` (minimum 60; default 600) and `AUTORESEARCH_AGENT_REASONING_EFFORT` (default `high`)
- `AUTORESEARCH_CODEX_BIN`, `AUTORESEARCH_UV_BIN`, `AUTORESEARCH_GIT_BIN`
- `AUTORESEARCH_CUDA_VISIBLE_DEVICES` (default `0`)
- `AUTORESEARCH_WSL_RUNTIME_ROOT` (absolute POSIX path; defaults under the WSL user's `$HOME/.cache`)
- `AUTORESEARCH_USE_CUDA_MPS=1` only when CUDA MPS is deliberately configured and running; this disables the backend's same-GPU serialization and does not itself start MPS

## Tests

```powershell
backend/.venv/Scripts/python -m pytest backend/tests
```
