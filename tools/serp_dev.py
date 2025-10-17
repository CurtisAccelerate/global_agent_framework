"""serp.dev multi-query search with Serper.dev scraping helpers."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import httpx

from config import Config

_logger = logging.getLogger("tool.serp_dev")


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _truncate(text: Optional[str], limit: int = 4000) -> Optional[str]:
    if not text:
        return text
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def _normalize_queries(queries: Union[str, List[str]], max_queries: int) -> List[str]:
    if isinstance(queries, str):
        raw_parts = []
        for delimiter in ("\n", "||", "\r"):
            if delimiter in queries:
                queries = queries.replace(delimiter, "\n")
        raw_parts = [part.strip() for part in queries.split("\n")]
    elif isinstance(queries, (list, tuple)):
        raw_parts = [str(part).strip() for part in queries]
    else:
        raise TypeError("queries must be a string or list of strings")

    filtered = [q for q in raw_parts if q]
    seen: set = set()
    deduped: List[str] = []
    for item in filtered:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped[:max_queries]


async def _serp_dev_search(
    client: httpx.AsyncClient,
    query: str,
    engine: str,
    serp_key: str,
    num: Optional[int],
    location: Optional[str],
) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    params: Dict[str, Any] = {
        "q": query,
        "engine": engine or "google",
        "api_key": serp_key,
    }
    if num:
        params["num"] = num
    if location:
        params["location"] = location

    url = f"{Config.SERP_DEV_BASE_URL.rstrip('/')}/search"
    _logger.info("[HTTP] serp.dev search q=%r", query)
    resp = await client.get(url, params=params)
    headers = {k.lower(): v for k, v in (resp.headers or {}).items()}
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text[:1000] if hasattr(resp, "text") else None}
    return resp.status_code, data, headers


async def _serper_scrape(
    client: httpx.AsyncClient,
    url: str,
    serper_key: str,
) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    headers = {
        "X-API-KEY": serper_key,
        "Content-Type": "application/json",
    }
    scrape_url = f"{Config.SERPER_BASE_URL.rstrip('/')}/web"
    _logger.info("[HTTP] serper.dev scrape url=%s", url)
    resp = await client.post(scrape_url, json={"url": url}, headers=headers)
    hdrs = {k.lower(): v for k, v in (resp.headers or {}).items()}
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text[:1000] if hasattr(resp, "text") else None}
    return resp.status_code, data, hdrs


async def serp_dev_search(
    queries: Union[str, List[str]],
    max_queries: Optional[int] = 3,
    serp_engine: str = "google",
    serp_num_results: Optional[int] = 10,
    location: Optional[str] = None,
    scrape_limit: Optional[int] = 3,
    scrape_concurrency: Optional[int] = 3,
    include_raw_serp: bool = False,
) -> Dict[str, Any]:
    """Perform serp.dev searches (up to N queries) and scrape top results with Serper.dev."""

    serp_key = (Config.SERP_DEV_API_KEY or "").strip()
    serper_key = (Config.SERPER_API_KEY or "").strip()
    if not serp_key:
        raise ValueError("SERP_DEV_API_KEY is required")
    if not serper_key:
        raise ValueError("SERPER_API_KEY is required")

    try:
        max_q = int(max_queries or 3)
    except Exception:
        max_q = 3
    if max_q <= 0:
        max_q = 3

    try:
        scrape_cap = int(scrape_limit or 3)
    except Exception:
        scrape_cap = 3
    if scrape_cap <= 0:
        scrape_cap = 3

    try:
        concurrency = max(1, int(scrape_concurrency or scrape_cap))
    except Exception:
        concurrency = max(1, scrape_cap)

    normalized_queries = _normalize_queries(queries, max_q)
    if not normalized_queries:
        raise ValueError("At least one non-empty query is required")

    timeout = httpx.Timeout(20.0, connect=5.0)
    verify_ssl = getattr(Config, "SERP_DEV_VERIFY_SSL", True)

    transport_serp = httpx.AsyncHTTPTransport(http2=False, retries=3)
    transport_serper = httpx.AsyncHTTPTransport(http2=False, retries=3)

    results: List[Dict[str, Any]] = []
    global_errors: List[str] = []

    async with httpx.AsyncClient(timeout=timeout, transport=transport_serp, trust_env=False, verify=verify_ssl) as serp_client:
        async with httpx.AsyncClient(timeout=timeout, transport=transport_serper, trust_env=False, verify=verify_ssl) as serper_client:

            for query in normalized_queries:
                search_entry: Dict[str, Any] = {
                    "query": query,
                    "search_status": None,
                    "search_headers": None,
                    "organic_results": [],
                    "scraped_documents": [],
                    "error": None,
                }

                try:
                    status, data, headers = await _serp_dev_search(
                        serp_client,
                        query=query,
                        engine=serp_engine,
                        serp_key=serp_key,
                        num=serp_num_results,
                        location=location,
                    )
                    search_entry["search_status"] = status
                    if headers:
                        search_entry["search_headers"] = {
                            k: headers.get(k) for k in (
                                "x-ratelimit-remaining",
                                "x-ratelimit-reset",
                                "x-ratelimit-limit",
                            ) if k in headers
                        }
                    if status != 200:
                        search_entry["error"] = {
                            "kind": "search_http_error",
                            "status_code": status,
                            "body": data,
                        }
                        results.append(search_entry)
                        continue
                except httpx.RequestError as exc:
                    msg = f"serp.dev request failed for query {query!r}: {exc}"
                    _logger.error(msg)
                    search_entry["error"] = {"kind": "network_error", "message": str(exc)}
                    results.append(search_entry)
                    continue

                organic = []
                if isinstance(data, dict):
                    organic = data.get("organic_results") or data.get("results") or []
                elif isinstance(data, list):
                    organic = data

                organic_items: List[Dict[str, Any]] = []
                for item in organic:
                    if not isinstance(item, dict):
                        continue
                    link = item.get("link") or item.get("url")
                    title = item.get("title") or item.get("name")
                    snippet = item.get("snippet") or item.get("description")
                    if not link:
                        continue
                    organic_items.append({
                        "title": title,
                        "link": link,
                        "snippet": _truncate(snippet, 500),
                        "position": item.get("position") or item.get("rank") or item.get("index"),
                        "source": item.get("source"),
                    })
                    if len(organic_items) >= scrape_cap:
                        break

                search_entry["organic_results"] = organic_items
                if include_raw_serp:
                    search_entry["raw_serp"] = data

                if not organic_items:
                    results.append(search_entry)
                    continue

                sem = asyncio.Semaphore(concurrency)

                async def _scrape_single(item: Dict[str, Any]) -> Dict[str, Any]:
                    link = item.get("link")
                    if not link:
                        return {
                            "link": link,
                            "status_code": None,
                            "error": "missing link",
                        }
                    async with sem:
                        try:
                            status_code, blob, headers = await _serper_scrape(serper_client, link, serper_key)
                        except httpx.RequestError as exc_scrape:
                            _logger.error("Serper scrape failed url=%s err=%s", link, exc_scrape)
                            return {
                                "link": link,
                                "status_code": None,
                                "error": str(exc_scrape),
                            }

                    content = None
                    if isinstance(blob, dict):
                        content = blob.get("markdown") or blob.get("text") or blob.get("content") or blob.get("html")
                    return {
                        "title": item.get("title"),
                        "link": link,
                        "status_code": status_code,
                        "content": _truncate(content, 5000),
                        "word_count": len(content.split()) if isinstance(content, str) else None,
                        "headers": {
                            k: headers.get(k) for k in (
                                "x-search-credits-left",
                                "x-credits-remaining",
                            ) if headers and k in headers
                        },
                        "error": None if status_code == 200 else blob,
                    }

                scrape_tasks = [asyncio.create_task(_scrape_single(item)) for item in organic_items]
                scraped_documents = await asyncio.gather(*scrape_tasks)
                search_entry["scraped_documents"] = scraped_documents
                results.append(search_entry)

    return {
        "generated_at": _now_iso_z(),
        "query_count": len(normalized_queries),
        "results": results,
        "errors": global_errors or None,
        "meta": {
            "max_queries": max_q,
            "scrape_limit": scrape_cap,
            "serp_engine": serp_engine,
            "location": location,
        },
    }


