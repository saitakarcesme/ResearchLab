# Autoresearch Lab

A local-first research console built around Karpathy's pinned
[`autoresearch`](https://github.com/karpathy/autoresearch) protocol. It runs
isolated, measurable experiments on local or SSH-connected NVIDIA GPUs, keeps
only strict metric improvements, streams real telemetry and semantic logs, and
builds Markdown articles solely from persisted experiment records.

## Requirements

- Node.js 22+
- Python 3.11
- Git, `uv`, Codex CLI, and `nvidia-smi`
- An NVIDIA CUDA environment for experiments
- On Windows, WSL2 Ubuntu is recommended for the upstream CUDA workload

The detected workstation path is supported with `Ubuntu-24.04`. Remote hosts
must already have Git, `uv`, CUDA, and SSH authentication configured.

## Setup

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup.ps1
```

## Run

Start the web app, API, live telemetry, and real research execution together:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-local.ps1 -EnableExecution
```

Open `http://localhost:3000/lab`. Omit `-EnableExecution` when you only want to
inspect the UI, SQLite persistence, and real GPU telemetry without allowing
Codex or training subprocesses to start.

### Use the same Lab from a Mac

Install Tailscale on this Windows computer and the Mac, sign both into the same
tailnet, then start ResearchLab with private HTTPS access:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/start-local.ps1 -EnableExecution -EnableTailnetAccess
```

The command prints the private Tailscale URL to open on the Mac. The browser
uses the Windows API through the same origin, so local-GPU work still runs on
this computer. The Windows computer and ResearchLab process must remain on.
Tailscale Serve does not expose the app to the public internet.

### Vast.ai rental and Apple Pay

The GPUs page reads verified single-GPU offers directly from Vast.ai. Connect a
Vast API key with user-read and instance-write permissions to rent a selected
offer. Vast requires prepaid provider credit; ResearchLab checks that balance
before opening checkout.

Apple Pay uses a Stripe-hosted Checkout session and a phone QR code. Configure
the backend process with:

```powershell
$env:STRIPE_SECRET_KEY = "sk_live_..."
$env:STRIPE_WEBHOOK_SECRET = "whsec_..."
$env:AUTORESEARCH_PUBLIC_APP_URL = "https://your-private-researchlab-url"
```

Point the Stripe webhook at
`https://your-private-researchlab-url/api/cloud/stripe-webhook`. The API also
polls the signed Checkout session, so a paid local checkout can recover if a
webhook is briefly delayed. Payment confirmation is idempotent: a rental order
can provision only one instance.

Lab runs measurable GPU experiments and model benchmarks. The selected
research agent is persisted separately from the model being benchmarked.

Research-agent choices can be configured with
`AUTORESEARCH_RESEARCHER_MODELS` (a comma-separated allowlist) and
`AUTORESEARCH_DEFAULT_RESEARCHER_MODEL`. The default catalog contains
`gpt-5.6-sol`, `gpt-5.6-terra`, and `gpt-5.3-codex-spark`.

The first real research start clones upstream commit
`228791fb499afffb54b46200aca536f79142f117`, prepares its environment and data,
then runs the fixed five-minute `val_bpb` baseline. Each research continues
until Pause or Stop is used. Experiments assigned to one GPU source are
serialized for comparable fixed-time metrics; different sources run in
parallel. Same-GPU overlap is available only when CUDA MPS is already running
and `AUTORESEARCH_USE_CUDA_MPS=1` is set. Target allocations are scheduling
hints, never claimed utilization guarantees; actual utilization is always shown
from `nvidia-smi`.

The uv environment is kept outside the Codex-editable worktree. Successful
results require one complete upstream terminal summary plus at least 299 seconds
of runner-measured active execution around the pinned 300-second budget. The
runner rechecks `prepare.py`, `train.py`, tracked paths, and ignored state before
accepting a metric.

## Manual development

API:

```powershell
$env:AUTORESEARCH_ENABLE_EXECUTION = "1"
$env:AUTORESEARCH_WSL_DISTRO = "Ubuntu-24.04"
backend/.venv/Scripts/python -m uvicorn backend.main:app --host 127.0.0.1 --port 7331
```

Web UI in a second terminal:

```powershell
npm run dev
```

Remote SSH keys or passwords are never accepted by the API or stored in
SQLite. `auth_method=agent` uses the active SSH agent. `auth_method=key_env`
reads a key path from `AUTORESEARCH_SSH_KEY_<SOURCE_ID>` or
`AUTORESEARCH_SSH_KEY_DEFAULT`.

## Verification

```powershell
npm run lint
npm test
backend/.venv/Scripts/python -m pytest backend/tests -q
```

The SQLite database, worktrees, and experiment logs live under `backend/data/`
and survive application restarts.
