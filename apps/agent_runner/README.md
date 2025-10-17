## Agent Runner

Minimal local runner for FutureX-style datasets with three modes:

- stub-local: no HTTP, deterministic answers
- stub-http: HTTP to a local stub server
- real: HTTP to a real OpenAI-compatible endpoint

### Requirements

- Python 3.10+
- Install deps:
  - `python -m venv .venv` then activate
  - `pip install -r requirements.txt`

### Quick start

1. Create and activate a virtualenv
2. Install deps: `pip install -r requirements.txt`
3. Verify dataset: `python cli.py verify-data --head 3`

### Modes

- stub-local (default):
  - `python cli.py run --mode stub-local --limit 3`
- stub-http: start stub server then run client
  - Terminal A: `python cli.py stub-serve`
  - Terminal B: `python cli.py run --mode stub-http --limit 3`
- real: requires endpoint and API key
  - Configure via env or flags (see Configuration)
  - Example: `python cli.py run --mode real --limit 3 --endpoint https://your-endpoint/v1/chat/completions --api-key sk-...`

### Configuration (.env or environment)

This CLI discovers the nearest `.env` from the current working directory upward (monorepo-friendly).

Copy `demo.env` to `.env` and fill in values:

```
AGENT_RUNNER_ENDPOINT=
AGENT_RUNNER_API_KEY=
```

Or set environment variables directly (PowerShell):

```
$env:AGENT_RUNNER_ENDPOINT = "https://your-endpoint/v1/chat/completions"
$env:AGENT_RUNNER_API_KEY = "sk-your-key"
```

Flags also work and override env:

```
python cli.py run --mode real --endpoint $env:AGENT_RUNNER_ENDPOINT --api-key $env:AGENT_RUNNER_API_KEY
```

### Outputs and resume

- Writes `out/futurex_<commitsha>/predictions.jsonl` incrementally
- Writes `manifest.json`
- HTTP logs saved under `out/.../logs/` for stub-http/real
- `--resume` (default) appends to an existing predictions file; use `--no-resume` to restart
