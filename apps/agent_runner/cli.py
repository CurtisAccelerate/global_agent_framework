import argparse
import asyncio
import json
import os
import re
import sys
from datetime import datetime, timezone
from typing import Dict, Any, List, Tuple

import httpx
from datasets import load_dataset
from huggingface_hub import HfApi

# Try to load nearest .env to support monorepo usage from any subfolder
try:
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv(usecwd=True))
except Exception:
    pass

# ----------------------------
# Hard-coded config (override via CLI if desired)
# ----------------------------
DATASET_REPO = "futurex-ai/Futurex-Online"
MODEL_NAME = "gpt-5"

STUB_HTTP_DEFAULT = "http://127.0.0.1:8000/v1/chat/completions"

# Regex to extract the TeX-style boxed answer
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")

def get_dataset_commit_sha() -> str:
    api = HfApi()
    info = api.dataset_info(DATASET_REPO)
    return getattr(info, "sha", "unknown")

def ensure_outdir(commit_sha: str, outdir: str) -> str:
    d = os.path.join(outdir, f"futurex_{commit_sha[:8]}")
    os.makedirs(d, exist_ok=True)
    return d

def extract_boxed(content: str) -> str:
    """
    Extract the string inside \boxed{...}. If not found, return the whole content.
    """
    m = BOXED_RE.search(content.strip())
    return m.group(1).strip() if m else content.strip()

async def call_openai_like(
    client: httpx.AsyncClient,
    endpoint: str,
    bearer: str | None,
    prompt: str,
    timeout_s: float,
    trace: bool,
    log_dir: str | None,
    seq_no: int | None,
    row_id: str | None,
) -> str:
    """
    Call an OpenAI-compatible /v1/chat/completions endpoint and return the boxed answer (string).
    """
    payload: Dict[str, Any] = {
        "model": MODEL_NAME,  # cosmetic per your note
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {"Content-Type": "application/json"}
    if bearer:
        headers["Authorization"] = f"Bearer {bearer}"

    if trace:
        redacted_headers = dict(headers)
        if "Authorization" in redacted_headers:
            redacted_headers["Authorization"] = "Bearer ****REDACTED****"
        print("[trace] REQUEST:")
        print(json.dumps({
            "endpoint": endpoint,
            "headers": redacted_headers,
            "payload": payload,
        }, ensure_ascii=False, indent=2))

    start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    r = await client.post(endpoint, json=payload, headers=headers, timeout=timeout_s)
    r.raise_for_status()
    data = r.json()
    if trace:
        try:
            print("[trace] RESPONSE:")
            print(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            print("[trace] RESPONSE (non-JSON)")
            print(r.text[:2000])

    # Persistent per-call log
    if log_dir is not None and seq_no is not None:
        try:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, f"{seq_no:03d}.json")
            redacted_headers = dict(headers)
            if "Authorization" in redacted_headers:
                redacted_headers["Authorization"] = "Bearer ****REDACTED****"
            log_record = {
                "timestamp_utc": start_ts,
                "row_id": row_id,
                "request": {
                    "endpoint": endpoint,
                    "headers": redacted_headers,
                    "payload": payload,
                },
                "response": data,
                "http_status": r.status_code,
            }
            with open(log_path, "w", encoding="utf-8") as lf:
                json.dump(log_record, lf, ensure_ascii=False, indent=2)
        except Exception as _:
            # Best-effort logging; don't fail the run on log error
            pass
    # Typical shape: {"choices":[{"message":{"content":"\\boxed{...}"}}]}
    content = (
        data.get("choices", [{}])[0]
        .get("message", {})
        .get("content", "")
    )
    return extract_boxed(content)

async def predict_one(
    client: httpx.AsyncClient,
    row: Dict[str, Any],
    sem: asyncio.Semaphore,
    mode: str,
    endpoint: str,
    bearer: str | None,
    timeout_s: float,
    trace: bool,
    log_dir: str | None,
    seq_no: int,
) -> Tuple[str, str]:
    """
    Produce a (id, prediction) for one dataset row based on the selected mode.
    Modes:
      - "real"      : HTTP to real endpoint
      - "stub-http" : HTTP to local stub server
      - "stub-local": no HTTP; quick deterministic boxed answer
    """
    _id = row["id"]
    prompt = row["prompt"]

    async with sem:
        if mode == "stub-local":
            # Super-fast local stub: infer a plausible dummy answer from the prompt
            p = prompt.lower()
            if "your_prediction" in p:
                return _id, "DUMMY_PREDICTION"
            if "yes" in p and "no" in p:
                return _id, "Yes"
            # If options A/B/C appear, pick A
            if " a." in p or "a. " in p:
                return _id, "A"
            return _id, "A"
        else:
            # HTTP call (real or stub-http behave the same contract)
            pred = await call_openai_like(
                client, endpoint, bearer, prompt, timeout_s, trace, log_dir, seq_no, _id
            )
            return _id, pred

async def run_cmd(args: argparse.Namespace) -> None:
    # Load dataset (train split)
    print("Loading dataset from Hugging Face... This may take a moment on first run.")
    ds = load_dataset(DATASET_REPO, split="train")
    rows: List[Dict[str, Any]] = list(ds)
    if args.limit:
        rows = rows[: args.limit]

    # Get dataset commit SHA
    commit_sha = get_dataset_commit_sha()

    # Prepare outputs
    out_dir = ensure_outdir(commit_sha, args.outdir)
    preds_path = os.path.join(out_dir, "predictions.jsonl")
    manifest_path = os.path.join(out_dir, "manifest.json")
    logs_dir = os.path.join(out_dir, "logs") if not getattr(args, "logdir", None) else args.logdir

    # Resume support: detect already completed IDs from existing predictions.jsonl
    already_done_ids: set[str] = set()
    if args.resume and os.path.exists(preds_path):
        try:
            with open(preds_path, "r", encoding="utf-8") as rf:
                for line in rf:
                    try:
                        obj = json.loads(line)
                        if isinstance(obj, dict) and "id" in obj:
                            already_done_ids.add(str(obj["id"]))
                    except Exception:
                        continue
        except Exception:
            pass

    # Filter rows if resuming
    all_rows_count = len(rows)
    if already_done_ids:
        rows = [r for r in rows if str(r.get("id")) not in already_done_ids]

    # Resolve mode/endpoints/bearer
    mode = args.mode
    if mode not in ("real", "stub-http", "stub-local"):
        print(f"Invalid --mode: {mode}", file=sys.stderr)
        sys.exit(2)

    endpoint = args.endpoint
    bearer = getattr(args, "api_key", None)
    if mode == "real":
        endpoint = endpoint or os.getenv("AGENT_RUNNER_ENDPOINT")
        bearer = bearer or os.getenv("AGENT_RUNNER_API_KEY")
        if not endpoint or not bearer:
            print(
                "real mode requires --endpoint/AGENT_RUNNER_ENDPOINT and --api-key/AGENT_RUNNER_API_KEY",
                file=sys.stderr,
            )
            sys.exit(2)
    elif mode == "stub-http":
        endpoint = endpoint or STUB_HTTP_DEFAULT
        bearer = None
    else:  # stub-local (no HTTP)
        endpoint = "local-stub"
        bearer = None

    sem = asyncio.Semaphore(max(1, args.concurrency))
    results: List[Tuple[str, str]] = []

    timeout_s = float(args.timeout)

    # Prepare predictions file for incremental durable writes
    write_mode = "a" if args.resume and os.path.exists(preds_path) else "w"
    successful_this_run = 0

    # Determine starting sequence for logs so we don't overwrite previous runs (especially with --resume)
    start_seq = 1
    if mode != "stub-local":
        try:
            os.makedirs(logs_dir, exist_ok=True)
            existing = [fn for fn in os.listdir(logs_dir) if fn.lower().endswith(".json")]
            max_seq = 0
            for fn in existing:
                try:
                    base = os.path.splitext(fn)[0]
                    num = int(base)
                    if num > max_seq:
                        max_seq = num
                except Exception:
                    continue
            start_seq = max_seq + 1
        except Exception:
            # If any error reading logs dir, fall back to starting at 1
            start_seq = 1

    async with httpx.AsyncClient(http2=True) as client:
        with open(preds_path, write_mode, encoding="utf-8") as pred_f:
            tasks = [
                predict_one(
                    client,
                    row,
                    sem,
                    mode,
                    endpoint,
                    bearer,
                    timeout_s,
                    args.trace,
                    logs_dir if mode != "stub-local" else None,
                    start_seq + idx,
                )
                for idx, row in enumerate(rows)
            ]
            for coro in asyncio.as_completed(tasks):
                try:
                    _id, pred = await coro
                    # Incremental write per result (durable)
                    line = json.dumps({"id": _id, "prediction": pred}, ensure_ascii=False) + "\n"
                    pred_f.write(line)
                    pred_f.flush()
                    try:
                        os.fsync(pred_f.fileno())
                    except Exception:
                        # fsync may not be available on some platforms; best-effort
                        pass
                    results.append((_id, pred))
                    successful_this_run += 1
                except Exception as e:
                    # If a row fails, record a dummy to keep the file aligned
                    print(f"[warn] request failed: {e}", file=sys.stderr)
    # predictions.jsonl has already been incrementally written

    # Write manifest.json
    manifest = {
        "dataset_repo": DATASET_REPO,
        "dataset_commit": commit_sha,
        "split": "train",
        "endpoint": endpoint,
        "mode": mode,
        "model": MODEL_NAME,
        "concurrency": args.concurrency,
        "total_rows": all_rows_count,
        "successful": successful_this_run + len(already_done_ids),
        "processed_this_run": successful_this_run,
        "already_present": len(already_done_ids),
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "output_files": {"predictions_jsonl": preds_path},
        "notes": "prediction is the content inside \\boxed{...}; stub-local uses a deterministic dummy",
        "logs_dir": logs_dir if mode != "stub-local" else None,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"- predictions: {preds_path}")
    print(f"- manifest:    {manifest_path}")

def verify_data_cmd(args: argparse.Namespace) -> None:
    # Load and show dataset info quickly
    ds = load_dataset(DATASET_REPO, split="train")
    rows: List[Dict[str, Any]] = list(ds)
    commit_sha = get_dataset_commit_sha()

    print(f"Dataset: {DATASET_REPO}")
    print(f"Split: train")
    print(f"Commit: {commit_sha}")
    print(f"Total rows: {len(rows)}")
    print("-" * 80)
    head = min(args.head, len(rows))
    for i in range(head):
        r = rows[i]
        prompt = r.get("prompt", "")
        snippet = prompt[:200].replace("\n", " ")
        print(f"{i+1}. id={r.get('id')} end_time={r.get('end_time')} level={r.get('level')}")
        print(f"   prompt: {snippet}{'...' if len(prompt) > 200 else ''}")
    print("-" * 80)
    if args.dump_csv:
        out_dir = ensure_outdir(commit_sha, args.outdir)
        csv_path = os.path.join(out_dir, "preview.csv")
        import csv
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "end_time", "level", "prompt"])
            for r in rows[: args.head]:
                w.writerow([r.get("id"), r.get("end_time"), r.get("level"), r.get("prompt")])
        print(f"Preview CSV written: {csv_path}")

def stub_serve_cmd(args: argparse.Namespace) -> None:
    # Tiny FastAPI stub that mimics the OpenAI-like endpoint.
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse
    import uvicorn

    app = FastAPI()

    def choose_dummy(prompt: str) -> str:
        p = prompt.lower()
        if "your_prediction" in p:
            return "\\boxed{DUMMY_PREDICTION}"
        if "yes" in p and "no" in p:
            return "\\boxed{Yes}"
        if " a." in p or "a. " in p:
            return "\\boxed{A}"
        return "\\boxed{A}"

    @app.post("/v1/chat/completions")
    async def completions(req: Request):
        body = await req.json()
        messages = body.get("messages", [])
        content = ""
        if messages and isinstance(messages, list):
            content = messages[0].get("content", "")
        answer = choose_dummy(content)
        return JSONResponse(
            {
                "id": "stub-1",
                "object": "chat.completion",
                "created": int(datetime.now().timestamp()),
                "model": body.get("model", MODEL_NAME),
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": answer}}
                ],
            }
        )

    print(f"Stub server on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="futurex-runner", description="Minimal FutureX local runner")
    sub = p.add_subparsers(dest="cmd", required=True)

    # verify-data
    v = sub.add_parser("verify-data", help="Print dataset commit, size, and a small preview")
    v.add_argument("--head", type=int, default=5, help="How many rows to preview")
    v.add_argument("--outdir", default="out", help="Output dir (for optional CSV)")
    v.add_argument("--dump-csv", action="store_true", help="Also write preview.csv")
    v.set_defaults(func=verify_data_cmd)

    # stub-serve
    s = sub.add_parser("stub-serve", help="Start a tiny stub server that mimics the OpenAI endpoint")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    s.set_defaults(func=stub_serve_cmd)

    # run
    r = sub.add_parser("run", help="Run predictions and write predictions.jsonl + manifest.json")
    r.add_argument(
        "--mode",
        choices=["stub-local", "stub-http", "real"],
        default="stub-local",
        help="stub-local (no HTTP), stub-http (HTTP to local stub), or real (HTTP to real endpoint)",
    )
    r.add_argument(
        "--endpoint",
        default=None,
        help="Endpoint URL for real mode (or set AGENT_RUNNER_ENDPOINT)",
    )
    r.add_argument(
        "--api-key",
        default=None,
        help="API key for real mode (or set AGENT_RUNNER_API_KEY)",
    )
    r.add_argument("--concurrency", type=int, default=2, help="Max concurrent requests (2 or 3)")
    r.add_argument("--timeout", type=float, default=1140.0, help="Per-request timeout seconds")
    r.add_argument("--limit", type=int, default=None, help="Process only the first N rows")
    r.add_argument("--outdir", default="out", help="Output directory")
    r.add_argument("--trace", action="store_true", help="Print HTTP request/response for stub-http/real")
    r.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing predictions.jsonl (default). Use --no-resume to restart from scratch.",
    )
    r.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Process all rows even if predictions.jsonl already has entries.",
    )
    r.set_defaults(func=lambda a: asyncio.run(run_cmd(a)))

    return p

def main():
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()


