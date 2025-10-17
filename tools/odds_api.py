import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
import logging

from config import Config
from .odds_devig import devig_and_aggregate


ODDS_BASE_URL = "https://api.the-odds-api.com/v4"

_logger = logging.getLogger("tool.odds_api")


def _now_iso_z() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _iso_z_in_days(days: int) -> str:
    future = datetime.now(timezone.utc) + timedelta(days=days)
    return future.replace(microsecond=0).isoformat().replace("+00:00", "Z")


async def _get_json_with_status(client: httpx.AsyncClient, url: str, params: Dict[str, Any]) -> Tuple[int, Dict[str, Any], Dict[str, str]]:
    resp = await client.get(url, params=params)
    status = resp.status_code
    headers = {k.lower(): v for k, v in (resp.headers or {}).items()}
    try:
        data = resp.json()
    except Exception:
        data = {"text": resp.text[:500] if hasattr(resp, "text") else None}
    return status, data, headers


async def odds_find(
    q: str,
    window_days: Optional[int] = 14,
    limit: Optional[int] = 3,
) -> Dict[str, Any]:
    """
    Free-text finder for games and outrights across all active sports; also fetches odds for the top results.
    Returns discovery plus embedded odds for each item (single tool call convenience).
    """
    if not q or not str(q).strip():
        raise ValueError("q is required")
    ql = str(q).strip().lower()
    window = int(window_days or 14)
    if window <= 0:
        window = 14
    cap = int(limit or 3)
    if cap <= 0:
        cap = 3

    _logger.info("[CALL] odds_find q=%r window_days=%s limit=%s", q, window_days, limit)
    timeout = httpx.Timeout(15.0, connect=5.0)
    prefer_http2 = True
    try:
        import h2  # noqa: F401
    except Exception:
        prefer_http2 = False
    async with httpx.AsyncClient(timeout=timeout, http2=prefer_http2) as client:
        # Fetch sports
        sports_url = f"{ODDS_BASE_URL}/sports"
        _logger.info("[HTTP] GET %s", sports_url)
        status, sports_data, _ = await _get_json_with_status(client, sports_url, {"apiKey": Config.ODDS_API_KEY})
        if status != 200:
            _logger.error("[ERROR] /sports HTTP %s", status)
            return {
                "query": q,
                "matches": [],
                "outrights": [],
                "error": f"/sports HTTP {status}",
                "meta": {"window_days": window, "generated_at": _now_iso_z()},
            }
        sports: List[Dict[str, Any]] = sports_data if isinstance(sports_data, list) else []
        active_sports = [s for s in sports if s.get("active") is True]

        # Outrights
        outrights: List[Dict[str, Any]] = []
        for s in active_sports:
            title = str(s.get("title") or "")
            key = str(s.get("key") or "")
            group = str(s.get("group") or "")
            has_out = bool(s.get("has_outrights") or s.get("hasOutrights") or False)
            if has_out:
                hay = f"{title} {group} {key}".lower()
                if ql in hay:
                    outrights.append({"sport_key": key, "title": title or key})
            if len(outrights) >= cap:
                break

        # Matches: fan-out across sports
        commence_from = _now_iso_z()
        commence_to = _iso_z_in_days(window)
        matches: List[Dict[str, Any]] = []

        sem = asyncio.Semaphore(10)

        async def _fetch_and_filter(sport_key: str) -> List[Dict[str, Any]]:
            url = f"{ODDS_BASE_URL}/sports/{sport_key}/events"
            params = {
                "apiKey": Config.ODDS_API_KEY,
                "commenceTimeFrom": commence_from,
                "commenceTimeTo": commence_to,
            }
            async with sem:
                st, data, _hdrs = await _get_json_with_status(client, url, params)
            _logger.info("[HTTP] GET %s status=%s", url, st)
            if st != 200 or not isinstance(data, list):
                return []
            results: List[Dict[str, Any]] = []
            for ev in data:
                home = str(ev.get("home_team") or "")
                away = str(ev.get("away_team") or "")
                hay = f"{home} {away}".lower()
                if ql in hay:
                    results.append({
                        "sport_key": sport_key,
                        "event_id": str(ev.get("id") or ev.get("event_id") or ""),
                        "home_team": home,
                        "away_team": away,
                        "commence_time": str(ev.get("commence_time") or ""),
                    })
                    if len(results) >= cap:
                        break
            return results

        tasks: List[asyncio.Task] = []
        for s in active_sports:
            key = s.get("key")
            if not key:
                continue
            tasks.append(asyncio.create_task(_fetch_and_filter(str(key))))
        if tasks:
            lists = await asyncio.gather(*tasks)
            for lst in lists:
                matches.extend(lst)

        # Deduplicate matches by (sport_key,event_id)
        seen: set = set()
        uniq_matches: List[Dict[str, Any]] = []
        for m in matches:
            k = (m.get("sport_key"), m.get("event_id"))
            if k in seen:
                continue
            seen.add(k)
            uniq_matches.append(m)

        # Cap to limit
        uniq_matches = uniq_matches[:cap]
        outrights = outrights[:cap]

        # Enrich with odds
        # Matches: group by sport_key to minimize calls (one /odds call per sport_key with eventIds CSV)
        match_groups: Dict[str, List[str]] = {}
        for m in uniq_matches:
            sk = m.get("sport_key")
            eid = m.get("event_id")
            if sk and eid:
                match_groups.setdefault(sk, []).append(str(eid))

        sem = asyncio.Semaphore(5)

        async def _fetch_h2h(sk: str, ids: List[str]) -> Tuple[str, Dict[str, Any], Optional[str]]:
            regions_try = ["us", "us2", "uk", "eu", "au"]
            last_resp: Dict[str, Any] = {}
            for reg in regions_try:
                async with sem:
                    resp = await odds_get(market="h2h", sport_key=sk, event_ids=ids, region=reg)
                _logger.info("[HTTP] odds_get h2h sport_key=%s region=%s status=%s", sk, reg, resp.get("status_code") or 200)
                if isinstance(resp, dict) and ("data" in resp):
                    data = resp.get("data")
                    if isinstance(data, list) and len(data) > 0:
                        return sk, resp, reg
                last_resp = resp
            return sk, last_resp, None

        h2h_tasks: List[asyncio.Task] = []
        for sk, ids in match_groups.items():
            h2h_tasks.append(asyncio.create_task(_fetch_h2h(sk, ids)))
        h2h_results: Dict[str, Dict[str, Any]] = {}
        h2h_region: Dict[str, Optional[str]] = {}
        if h2h_tasks:
            pairs = await asyncio.gather(*h2h_tasks)
            for sk, resp, reg in pairs:
                h2h_results[sk] = resp or {}
                h2h_region[sk] = reg

        # Attach per-match de-vigged probabilities by matching on event id
        for m in uniq_matches:
            sk = m.get("sport_key")
            eid = m.get("event_id")
            probs_blob = None
            try:
                data = (h2h_results.get(sk) or {}).get("data")
                if isinstance(data, list):
                    for ev in data:
                        if str(ev.get("id") or ev.get("event_id")) == str(eid):
                            # Compute consensus de-vigged probabilities for this event's h2h market
                            probs_blob = devig_and_aggregate(bookmakers=ev.get("bookmakers") or [], market_key="h2h", odds_format="american")
                            break
            except Exception:
                pass
            m["probabilities"] = probs_blob
            if sk in h2h_region and h2h_region[sk]:
                m["region"] = h2h_region[sk]

        # Outrights: fetch odds per sport_key (cap already applied) and compute de-vigged probabilities
        async def _fetch_outr(sk: str) -> Tuple[str, Dict[str, Any], Optional[str]]:
            # Try multiple regions to increase likelihood of data
            regions_try = ["us", "us2", "uk", "eu", "au"]
            last_resp: Dict[str, Any] = {}
            for reg in regions_try:
                async with sem:
                    resp = await odds_get(market="outrights", sport_key=sk, region=reg)
                _logger.info("[HTTP] odds_get outrights sport_key=%s region=%s status=%s", sk, reg, resp.get("status_code") or 200)
                # Success path: non-empty data with at least one bookmakers entry containing outcomes
                if isinstance(resp, dict) and ("data" in resp):
                    data = resp.get("data")
                    if isinstance(data, list) and len(data) > 0:
                        first = data[0]
                        books = first.get("bookmakers") or []
                        has_outcomes = False
                        for bk in books:
                            for mkt in (bk.get("markets") or []):
                                if mkt.get("key") == "outrights" and (mkt.get("outcomes") or []):
                                    has_outcomes = True
                                    break
                            if has_outcomes:
                                break
                        if has_outcomes:
                            return sk, resp, reg
                last_resp = resp
            return sk, last_resp, None

        outr_tasks: List[asyncio.Task] = []
        for o in outrights:
            sk = o.get("sport_key")
            if sk:
                outr_tasks.append(asyncio.create_task(_fetch_outr(sk)))
        if outr_tasks:
            pairs2 = await asyncio.gather(*outr_tasks)
            sk_to_resp: Dict[str, Dict[str, Any]] = {}
            sk_to_region: Dict[str, Optional[str]] = {}
            for sk, resp, reg in pairs2:
                sk_to_resp[sk] = resp
                sk_to_region[sk] = reg
            for o in outrights:
                sk = o.get("sport_key")
                blob = (sk_to_resp.get(sk) or {})
                if "data" in blob and blob.get("data") is not None and isinstance(blob.get("data"), list) and blob["data"]:
                    # Odds API outrights returns an array with a single pseudo-event containing bookmakers
                    first = blob["data"][0]
                    probs = devig_and_aggregate(bookmakers=first.get("bookmakers") or [], market_key="outrights", odds_format="american")
                    o["probabilities"] = probs
                else:
                    o["probabilities"] = None
                    if blob.get("status_code"):
                        o["error"] = {"status_code": blob.get("status_code"), "headers": blob.get("headers")}
                if sk in sk_to_region and sk_to_region[sk]:
                    o["region"] = sk_to_region[sk]

        # Build simplified results list
        results: List[Dict[str, Any]] = []
        for m in uniq_matches:
            probs = m.get("probabilities") or {}
            oc = probs.get("outcomes") or []
            if not oc:
                continue
            outcomes_list: List[Dict[str, Any]] = []
            for o in oc:
                outcomes_list.append({
                    "name": o.get("name"),
                    "prob": o.get("prob_mean"),
                    "best_price": o.get("best_price"),
                })
            label = f"{m.get('home_team') or ''} vs {m.get('away_team') or ''}".strip()
            results.append({
                "kind": "match",
                "event_id": m.get("event_id"),
                "commence_time": m.get("commence_time"),
                "label": label,
                "outcomes": outcomes_list,
            })

        for o in outrights:
            probs = o.get("probabilities") or {}
            oc = probs.get("outcomes") or []
            if not oc:
                continue
            outcomes_list: List[Dict[str, Any]] = []
            for it in oc:
                outcomes_list.append({
                    "name": it.get("name"),
                    "prob": it.get("prob_mean"),
                    "best_price": it.get("best_price"),
                })
            results.append({
                "kind": "outright",
                "sport_key": o.get("sport_key"),
                "label": o.get("title") or o.get("sport_key"),
                "outcomes": outcomes_list,
            })

        out = {
            "query": q,
            "results": results,
            "meta": {
                "window_days": window,
                "generated_at": _now_iso_z(),
                "limit": cap,
                "count": len(results),
            },
        }
        try:
            _logger.info("[RETURN] odds_find count=%d sample=%s", len(results), (results[0] if results else None))
        except Exception:
            pass
        return out


async def odds_get(
    market: str,
    sport_key: str,
    event_ids: Optional[List[str]] = None,
    region: Optional[str] = "us",
    odds_format: Optional[str] = "american",
) -> Dict[str, Any]:
    """
    Fetch odds for either a match (h2h by event_id) or an outright (futures) using the provided sport_key.
    Passes through Odds API JSON and includes quota headers.
    """
    mkt = (market or "").strip().lower()
    if mkt not in ("h2h", "outrights"):
        raise ValueError("market must be 'h2h' or 'outrights')")
    if not sport_key:
        raise ValueError("sport_key is required")
    if mkt == "h2h":
        if not event_ids or not isinstance(event_ids, list) or not all(str(e).strip() for e in event_ids):
            raise ValueError("event_ids array is required for market='h2h'")

    timeout = httpx.Timeout(20.0, connect=5.0)
    prefer_http2 = True
    try:
        import h2  # noqa: F401
    except Exception:
        prefer_http2 = False
    _logger.info("[CALL] odds_get market=%s sport_key=%s region=%s odds_format=%s event_ids=%s", market, sport_key, region, odds_format, (event_ids[:5] if event_ids else None))
    async with httpx.AsyncClient(timeout=timeout, http2=prefer_http2) as client:
        url = f"{ODDS_BASE_URL}/sports/{sport_key}/odds"
        params: Dict[str, Any] = {
            "apiKey": Config.ODDS_API_KEY,
            "regions": region or "us",
            "oddsFormat": odds_format or "american",
            "markets": mkt,
        }
        if mkt == "h2h":
            params["eventIds"] = ",".join([str(e).strip() for e in event_ids or []])

        status, data, headers = await _get_json_with_status(client, url, params)
        _logger.info("[HTTP] GET %s status=%s", url, status)
        if status != 200:
            # Return status and headers unchanged for rate-limit handling by caller
            return {
                "status_code": status,
                "headers": {
                    k: headers.get(k) for k in (
                        "x-requests-remaining",
                        "x-requests-used",
                        "x-requests-last",
                        "retry-after",
                    )
                },
                "url": url,
                "params": {k: v for k, v in params.items() if k != "apiKey"},
                "body": data,
            }

        # Success: odds API returns array of events/markets
        out = {
            "data": data,
            "headers": {
                k: headers.get(k) for k in (
                    "x-requests-remaining",
                    "x-requests-used",
                    "x-requests-last",
                )
            }
        }
        try:
            _logger.info("[RETURN] odds_get market=%s events_len=%d", mkt, (len(data) if isinstance(data, list) else -1))
        except Exception:
            pass
        return out


