import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import Config
from logging_config import setup_logging

try:
    from tools.serper import serper_search
except Exception:
    serper_search = None  # type: ignore

try:
    from tools.serp_dev_stub import serp_dev_search_stub
except Exception:
    serp_dev_search_stub = None  # type: ignore


DEFAULT_QUERIES = ["apple inc", "google inc"]


def _read_query_file(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        raise RuntimeError(f"Failed to read query file {path}: {exc}")
    queries = [line.strip() for line in text.splitlines() if line.strip()]
    if not queries:
        raise RuntimeError(f"No queries found in file {path}")
    return queries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick tester for serper search/scrape tool (stub or live)."
    )
    parser.add_argument(
        "queries",
        nargs="*",
        help="Queries to run (space separated). If omitted, use defaults or --query-file.",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        help="Path to a text file containing one query per line.",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Use live serper API instead of local stub.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=4,
        help="Maximum queries to send in a single request (default: 4).",
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=3,
        help="Maximum organic results (and scrapes) to return per query (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=20.0,
        help="Timeout (seconds) for the underlying coroutine (default: 20).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Dump the full JSON response without previews.",
    )
    return parser.parse_args()


def _resolve_queries(args: argparse.Namespace) -> List[str]:
    pieces: List[str] = []
    if args.queries:
        pieces.extend(args.queries)
    if args.query_file:
        pieces.extend(_read_query_file(args.query_file))
    if not pieces:
        pieces = list(DEFAULT_QUERIES)
    # maintain order but dedupe trivially
    seen = set()
    deduped: List[str] = []
    for q in pieces:
        key = q.strip()
        if not key:
            continue
        if key.lower() in seen:
            continue
        seen.add(key.lower())
        deduped.append(key)
    return deduped


async def run_stub(queries: Iterable[str], max_queries: int, max_results: int) -> Dict[str, Any]:
    if serp_dev_search_stub is None:
        return {"error": "stub_not_available"}
    result: Dict[str, Any] = await serp_dev_search_stub(
        queries=list(queries),
        max_queries=max_queries,
        max_results=max_results,
    )
    return {"mode": "stub", **result}


async def run_live(queries: Iterable[str], max_queries: int, max_results: int) -> Dict[str, Any]:
    if serper_search is None:
        return {"error": "serper_search_not_available"}
    try:
        result = await serper_search(
            queries=list(queries),
            max_queries=max_queries,
            max_results=max_results,
        )
    except Exception as exc:
        return {"error": f"live_call_failed: {type(exc).__name__}: {exc}"}
    return {"mode": "live", **result}


def _build_preview(payload: Dict[str, Any]) -> Dict[str, Any]:
    preview: Dict[str, Any] = {
        "mode": payload.get("mode"),
        "query_count": payload.get("query_count"),
        "results": [],
    }
    for block in payload.get("results", []) or []:
        organic = block.get("organic") or block.get("organic_results") or []
        scraped = block.get("scraped_documents") or []
        preview["results"].append(
            {
                "query": block.get("query"),
                "organic_preview": [
                    {"title": item.get("title"), "link": item.get("link")}
                    for item in organic[:1]
                ],
                "scraped_preview": [
                    {"link": item.get("link"), "status_code": item.get("status_code")}
                    for item in scraped[:1]
                ],
            }
        )
    return preview


async def main() -> None:
    args = parse_args()

    setup_logging(
        debug=True,
        auto_clear=False,
        log_verbosity="verbose",
        errors_only=False,
    )

    queries = _resolve_queries(args)
    if not queries:
        raise SystemExit("No valid queries provided.")

    runner = run_live if args.live else run_stub
    try:
        result = await asyncio.wait_for(
            runner(queries, args.max_queries, args.max_results),
            timeout=args.timeout,
        )
    except asyncio.TimeoutError:
        result = {
            "error": "timeout",
            "mode": "live" if args.live else "stub",
        }

    output = result if args.raw else _build_preview(result) if not result.get("error") else result
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
