"""Serper.dev search + web scraping helpers (google.serper.dev & scrape.serper.dev)."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx
from urllib.parse import urlparse

from config import Config

_logger = logging.getLogger("tool.serper")
_logger.setLevel(logging.DEBUG)


def _truncate(text: Optional[str], limit: int = 4000) -> Optional[str]:
    if not text:
        return text
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _log_payload(label: str, payload: Any, limit: int = 1200) -> None:
    try:
        if isinstance(payload, (dict, list)):
            preview = json.dumps(payload, ensure_ascii=False)[:limit]
        else:
            preview = str(payload)[:limit]
    except Exception:
        preview = str(payload)[:limit]
    _logger.debug("%s: %s", label, preview)


async def _serper_search(client: httpx.AsyncClient, query: str) -> Tuple[int, Any, Dict[str, str]]:
    headers = {
        "X-API-KEY": Config.SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    url = Config.SERPER_SEARCH_URL
    _logger.debug("[HTTP] serper search POST %s", url)
    _log_payload("search.request", {"q": query}, limit=400)
    resp = await client.post(url, json={"q": query}, headers=headers)
    status = resp.status_code
    headers_out = {k.lower(): v for k, v in (resp.headers or {}).items()}
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text[:800] if hasattr(resp, "text") else None}
    _log_payload(
        "search.response",
        {
            "status": status,
            "has_organic": bool(isinstance(data, dict) and data.get("organic")),
            "body_preview": data if isinstance(data, dict) else str(data),
        },
        limit=600,
    )
    return status, data, headers_out


async def _serper_scrape(client: httpx.AsyncClient, url: str) -> Tuple[int, Any, Dict[str, str]]:
    headers = {
        "X-API-KEY": Config.SERPER_API_KEY,
        "Content-Type": "application/json",
    }
    scrape_url = Config.SERPER_WEB_URL
    _logger.debug("[HTTP] serper scrape POST %s", scrape_url)
    _log_payload("scrape.request", {"url": url}, limit=400)
    resp = await client.post(scrape_url, json={"url": url}, headers=headers)
    status = resp.status_code
    headers_out = {k.lower(): v for k, v in (resp.headers or {}).items()}
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text[:800] if hasattr(resp, "text") else None}
    _log_payload(
        "scrape.response",
        {
            "status": status,
            "body_preview": data if isinstance(data, dict) else str(data),
        },
        limit=600,
    )
    return status, data, headers_out


async def serper_search(
    queries: Union[str, List[str]],
    max_queries: Optional[int] = 4,
    max_results: Optional[int] = 3,
) -> Dict[str, Any]:
    """Multi-query search using Serper.dev; returns top organic results with scraped content."""

    if isinstance(queries, str):
        items = [q.strip() for q in queries.replace("||", "\n").split("\n") if q.strip()]
    else:
        items = [str(q).strip() for q in (queries or []) if str(q).strip()]
    deduped: List[str] = []
    seen = set()
    for q in items:
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(q)
    if not deduped:
        raise ValueError("At least one non-empty query is required")
    cap = max(1, min(int(max_queries or 4), 4))
    deduped = deduped[:cap]

    scrape_cap = max(1, min(int(max_results or 3), 10))

    timeout = httpx.Timeout(20.0, connect=5.0)
    verify = bool(Config.SERPER_VERIFY_SSL)
    transport = httpx.AsyncHTTPTransport(http2=False, retries=2)

    results: List[Dict[str, Any]] = []

    _logger.debug(
        "serper_search.start",
        extra={
            "deduped_queries": deduped,
            "max_queries": cap,
            "max_results": scrape_cap,
        },
    )

    async with httpx.AsyncClient(timeout=timeout, verify=verify, transport=transport, trust_env=False) as client:
        for query in deduped:
            st, data, _hdr = await _serper_search(client, query)
            if st != 200 or not isinstance(data, dict):
                results.append({
                    "query": query,
                    "error": {"status_code": st, "body": data},
                    "organic": [],
                    "scraped_documents": [],
                })
                _logger.warning(
                    "serper_search.error",
                    extra={
                        "query": query,
                        "status_code": st,
                    },
                )
                continue

            organic = data.get("organic")
            organic = organic if isinstance(organic, list) else []
            shortlist: List[Dict[str, Any]] = []
            for it in organic:
                link = it.get("link")
                if not link:
                    continue
                parsed = urlparse(link)
                domain = parsed.netloc or parsed.hostname or ""
                shortlist.append({
                    "title": it.get("title"),
                    "link": link,
                    "domain": domain,
                    "snippet": _truncate(it.get("snippet"), 500),
                    "position": it.get("position"),
                    "date": it.get("date"),
                })
                if len(shortlist) >= scrape_cap:
                    break

            scraped_docs: List[Dict[str, Any]] = []
            if shortlist:
                sem = asyncio.Semaphore(3)

                async def _scrape(url: str) -> Dict[str, Any]:
                    async with sem:
                        sc, blob, _ = await _serper_scrape(client, url)
                    content = None
                    if isinstance(blob, dict):
                        content = blob.get("markdown") or blob.get("text") or blob.get("content")
                    parsed = urlparse(url)
                    domain = parsed.netloc or parsed.hostname or ""
                    _logger.debug(
                        "serper_scrape.result",
                        extra={
                            "url": url,
                            "status_code": sc,
                            "has_content": isinstance(content, str) and bool(content.strip()),
                        },
                    )
                    return {
                        "link": url,
                        "status_code": sc,
                        "content": _truncate(content, 4000),
                        "domain": domain,
                    }

                tasks = [asyncio.create_task(_scrape(it["link"])) for it in shortlist]
                scraped_docs = await asyncio.gather(*tasks)

            results.append({
                "query": query,
                "organic": shortlist,
                "scraped_documents": scraped_docs,
                "knowledge_graph": data.get("knowledgeGraph"),
                "people_also_ask": data.get("peopleAlsoAsk"),
            })
            _logger.debug(
                "serper_search.summary",
                extra={
                    "query": query,
                    "organic_count": len(shortlist),
                    "scraped_count": len(scraped_docs),
                },
            )

    final_payload = {"query_count": len(deduped), "results": results}
    _log_payload("serper_search.complete", {"query_count": len(deduped)}, limit=200)
    return final_payload
