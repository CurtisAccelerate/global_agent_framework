"""Stub serper.dev search for offline testing (search + web)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union

from urllib.parse import urlparse


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_queries(queries: Union[str, List[str]], max_queries: int = 5) -> List[str]:
    if isinstance(queries, str):
        parts = [p.strip() for p in queries.replace("||", "\n").split("\n")]
    elif isinstance(queries, (list, tuple)):
        parts = [str(p).strip() for p in queries]
    else:
        parts = []
    deduped: List[str] = []
    seen = set()
    for p in parts:
        if not p:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)
    if not deduped:
        deduped = ["stub fallback query"]
    return deduped[: max_queries or 5]


def _fake_organic(query: str, index: int) -> Dict[str, Any]:
    base_url = f"https://example.com/{query.strip().lower().replace(' ', '-')}/{index+1}"
    parsed = urlparse(base_url)
    domain = parsed.hostname or "example.com"
    return {
        "title": f"Stub article {index + 1} for {query}",
        "link": base_url,
        "snippet": f"Automated stub snippet describing {query} (item {index + 1}).",
        "position": index + 1,
        "domain": domain,
    }


def _fake_scrape(query: str, index: int, base_url: str) -> Dict[str, Any]:
    return {
        "link": base_url,
        "status_code": 200,
        "content": _truncate(f"Stubbed scraped content for {query} {index + 1}."),
    }


def _truncate(text: Optional[str], limit: int = 4000) -> Optional[str]:
    if text is None:
        return None
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


async def serp_dev_search_stub(
    queries: Union[str, List[str]],
    max_queries: Optional[int] = 4,
    max_results: Optional[int] = 3,
) -> Dict[str, Any]:
    normalized = _normalize_queries(queries, max_queries or 4)
    results: List[Dict[str, Any]] = []

    for query in normalized:
        organic = []
        scraped = []
        cap = max(1, min(int(max_results or 3), 10))
        for idx in range(cap):
            org = _fake_organic(query, idx)
            organic.append({
                "title": org.get("title"),
                "link": org.get("link"),
                "snippet": org.get("snippet"),
                "position": org.get("position"),
                "domain": org.get("domain"),
                "date": _now_iso_z().split("T", 1)[0],
            })
            scraped.append({
                "link": org["link"],
                "status_code": 200,
                "content": _truncate(f"Stubbed scraped content for {query} {idx + 1}."),
                "domain": org.get("domain"),
            })
        results.append({
            "query": query,
            "organic": organic,
            "scraped_documents": scraped,
            "knowledge_graph": {
                "title": query.title(),
                "description": f"Stub knowledge graph description for {query}.",
                "attributes": {
                    "Source": "stub",
                    "Generated": _now_iso_z(),
                },
            },
            "people_also_ask": [
                {
                    "question": f"What is {query}?",
                    "snippet": f"Stub answer for {query}.",
                    "link": organic[0]["link"] if organic else None,
                }
            ],
        })

    return {
        "generated_at": _now_iso_z(),
        "query_count": len(normalized),
        "results": results,
        "meta": {
            "max_queries": max_queries or 4,
            "max_results": max_results or 3,
            "stub": True,
        },
    }
