import asyncio
import json
import sys

from config import Config
from logging_config import setup_logging

try:
    from tools.serp_dev import serp_dev_search
except Exception as exc:  # pragma: no cover - import guard
    print(f"Failed to import serp_dev tool: {exc}", file=sys.stderr)
    sys.exit(1)


async def main() -> None:
    setup_logging(
        debug=Config.DEBUG,
        auto_clear=False,
        log_verbosity=Config.LOG_VERBOSITY,
        errors_only=Config.ERRORS_ONLY,
    )

    if getattr(Config, "SERP_DEV_USE_STUB", False):
        print("SERP_DEV_USE_STUB is enabled. Disable it to run the live serp.dev integration test.", file=sys.stderr)
        sys.exit(1)

    queries = [
        "serp.dev api updates",
        "serper.dev documentation",
    ]

    print("Running serp_dev_search with queries:")
    for q in queries:
        print(f"  - {q}")
    print("\nFetching…\n")

    result = await serp_dev_search(
        queries=queries,
        max_queries=5,
        scrape_limit=3,
        include_raw_serp=False,
    )

    # Persist a trimmed preview for quick inspection
    preview = {
        "generated_at": result.get("generated_at"),
        "query_count": result.get("query_count"),
        "meta": result.get("meta"),
        "results": [],
    }

    for block in result.get("results", [])[: len(queries)]:
        organic = block.get("organic_results", [])[:3]
        scraped = block.get("scraped_documents", [])[:3]
        preview["results"].append({
            "query": block.get("query"),
            "search_status": block.get("search_status"),
            "organic_preview": [
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "snippet": item.get("snippet"),
                }
                for item in organic
            ],
            "scraped_preview": [
                {
                    "title": item.get("title"),
                    "link": item.get("link"),
                    "status_code": item.get("status_code"),
                    "snippet": item.get("content")[:200] if isinstance(item.get("content"), str) else None,
                }
                for item in scraped
            ],
        })

    print(json.dumps(preview, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
