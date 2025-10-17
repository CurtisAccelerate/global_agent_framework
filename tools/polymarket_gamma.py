import asyncio
from typing import Any, Dict, List, Optional

import httpx
import json as _json
import logging


BASE_URL = "https://gamma-api.polymarket.com"

_logger = logging.getLogger("tool.polymarket")


def _coerce_bool(value: Optional[bool]) -> Optional[str]:
    if value is None:
        return None
    return "true" if bool(value) else "false"


def _simplify_market_record(m: Dict[str, Any]) -> Dict[str, Any]:
    market_id = m.get("id") or m.get("marketId") or m.get("_id")
    question = m.get("question") or m.get("title") or m.get("name")
    slug = m.get("slug") or m.get("urlSlug")
    outcomes = m.get("outcomes")
    outcome_prices = m.get("outcomePrices") or m.get("prices") or []
    # Some Gamma variants serialize arrays as JSON strings
    if isinstance(outcomes, str):
        try:
            outcomes = _json.loads(outcomes)
        except Exception:
            pass
    if isinstance(outcome_prices, str):
        try:
            outcome_prices = _json.loads(outcome_prices)
        except Exception:
            pass

    # Normalize outcomes/prices
    normalized_outcomes: List[str] = []
    if isinstance(outcomes, list) and outcomes and all(isinstance(x, str) for x in outcomes):
        normalized_outcomes = outcomes
    elif isinstance(outcomes, list) and outcomes and all(isinstance(x, dict) for x in outcomes):
        # Try to pull "name" field
        normalized_outcomes = [str(x.get("name") or x.get("title") or x.get("outcome") or idx) for idx, x in enumerate(outcomes)]
    else:
        # Fall back to binary market
        if isinstance(outcome_prices, list) and len(outcome_prices) == 2:
            normalized_outcomes = ["Yes", "No"]
        else:
            # Unknown outcomes; enumerate
            if isinstance(outcome_prices, list) and outcome_prices:
                normalized_outcomes = [f"Outcome {i}" for i in range(len(outcome_prices))]

    # Normalize prices to floats in [0,1]
    probs: List[Optional[float]] = []
    if isinstance(outcome_prices, list):
        for p in outcome_prices:
            try:
                # Some APIs may return cents, ensure we clamp to [0,1]
                fp = float(p)
                if fp > 1.0:
                    fp = fp / 100.0
                if fp < 0.0:
                    fp = 0.0
                if fp > 1.0:
                    fp = 1.0
                probs.append(fp)
            except Exception:
                probs.append(None)

    odds: Dict[str, Optional[float]] = {}
    for idx, name in enumerate(normalized_outcomes):
        value = probs[idx] if idx < len(probs) else None
        odds[str(name)] = value

    # Construct URL best-effort
    url = None
    if slug:
        url = f"https://polymarket.com/event/{slug}"
    elif market_id:
        url = f"https://polymarket.com/market/{market_id}"

    # Additional metadata
    status = m.get("status") or m.get("state")
    end_date = m.get("endDate") or m.get("closeTime") or m.get("end_time")
    volume = m.get("volume") or m.get("volume24hr") or m.get("volume24h")
    liquidity = m.get("liquidity") or m.get("liquidity_in_usd")

    return {
        "id": market_id,
        "question": question,
        "slug": slug,
        "url": url,
        "status": status,
        "endDate": end_date,
        "volume": volume,
        "liquidity": liquidity,
        "odds": odds,
        "raw": {
            # Keep a small subset of raw fields for debugging without being too large
            k: m.get(k) for k in (
                "question",
                "outcomes",
                "outcomePrices",
                "volume",
                "liquidity",
                "endDate",
                "status",
                "slug",
            )
        },
    }


async def polymarket_gamma_get_odds(
    query: Optional[str] = None,
    market_id: Optional[str] = None,
    event_id: Optional[str] = None,
    active: Optional[bool] = True,
    limit: Optional[int] = 10,
    sort: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fetch Polymarket market odds via the public Gamma API.

    At least one of (query, market_id, event_id) is recommended. If none provided,
    returns the top-most active markets up to `limit`.

    Returns a JSON-serializable dict with `markets` and request metadata.
    """
    params: Dict[str, Any] = {}
    if limit is not None:
        try:
            lim = int(limit)
            if lim > 200:
                lim = 200
            if lim <= 0:
                lim = 10
            params["limit"] = lim
        except Exception:
            params["limit"] = 10
    if query:
        params["search"] = query
    if market_id:
        # Many Gamma deployments accept `ids` for filtering by market ids (comma-separated)
        params["ids"] = market_id
    if event_id:
        params["eventId"] = event_id
    if active is not None:
        params["active"] = _coerce_bool(active)
    if sort:
        # Pass through, common options include volume, liquidity, creation, endDate
        params["sort"] = sort

    url = f"{BASE_URL}/markets"
    try:
        _logger.info(
            "[CALL] polymarket_gamma_get_odds query=%r market_id=%r event_id=%r active=%r limit=%r sort=%r",
            query, market_id, event_id, active, limit, sort,
        )
    except Exception:
        pass

    timeout = httpx.Timeout(10.0, connect=5.0)
    # Prefer HTTP/2 when available, but gracefully fall back to HTTP/1.1 if 'h2' is not installed
    prefer_http2 = True
    try:
        import h2  # noqa: F401
    except Exception:
        prefer_http2 = False
    async with httpx.AsyncClient(timeout=timeout, http2=prefer_http2) as client:
        try:
            _logger.info("[HTTP] GET %s params=%s", url, params)
            resp = await client.get(url, params=params)
            status = resp.status_code
            _logger.info("[HTTP] status=%s", status)
            if status != 200:
                return {
                    "ok": False,
                    "error": f"Gamma API error: HTTP {status}",
                    "status_code": status,
                    "url": str(resp.request.url),
                    "body_preview": resp.text[:500] if hasattr(resp, "text") else None,
                }
            data = resp.json()
            markets = data if isinstance(data, list) else data.get("data") or data.get("markets") or []
            simplified = [_simplify_market_record(m) for m in markets]
            try:
                sample = simplified[0] if simplified else None
                if sample:
                    _logger.info(
                        "[RETURN] ok=True count=%d sample_question=%r sample_odds_keys=%s",
                        len(simplified), sample.get("question"), list((sample.get("odds") or {}).keys())[:5]
                    )
                else:
                    _logger.info("[RETURN] ok=True count=0")
            except Exception:
                pass
            return {
                "ok": True,
                "count": len(simplified),
                "markets": simplified,
                "request": {
                    "url": str(resp.request.url),
                },
            }
        except Exception as e:
            try:
                _logger.error("[ERROR] polymarket request failed: %s: %s", type(e).__name__, str(e))
            except Exception:
                pass
            return {
                "ok": False,
                "error": f"Request failed: {type(e).__name__}: {str(e)}",
                "url": url,
                "params": params,
            }


