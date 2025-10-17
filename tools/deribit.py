import asyncio
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime, timezone, timedelta

import httpx
import logging


DERIBIT_BASE_URL = "https://www.deribit.com/api/v2"

_logger = logging.getLogger("tool.deribit")


_INSTRUMENTS_CACHE: Dict[str, Dict[str, Any]] = {}
_INSTRUMENTS_TTL_SECONDS = 60


def _now_ms() -> int:
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def _to_yyyy_mm_dd(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def _parse_yyyy_mm_dd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return f
    except Exception:
        return None


def _extract_ticker_fields(result: Dict[str, Any]) -> Dict[str, Optional[float]]:
    # Accept snake_case or camelCase variants
    mark_iv = result.get("mark_iv")
    if mark_iv is None:
        mark_iv = result.get("markIv")
    mark_price = result.get("mark_price")
    if mark_price is None:
        mark_price = result.get("markPrice")
    last_price = result.get("last_price")
    if last_price is None:
        last_price = result.get("lastPrice")
    return {
        "mark_iv": _coerce_float(mark_iv),
        "mark_price": _coerce_float(mark_price),
        "last_price": _coerce_float(last_price),
    }


async def _deribit_get(client: httpx.AsyncClient, path: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{DERIBIT_BASE_URL}{path}"
    resp = await client.get(url, params=params or {})
    resp.raise_for_status()
    data = resp.json()
    # Deribit wraps result as {"jsonrpc":"2.0","result":...,"usIn":...,"usOut":...,"usDiff":...,"testnet":false}
    if isinstance(data, dict) and "result" in data:
        return data["result"]
    return data


async def _get_instruments(client: httpx.AsyncClient, currency: str) -> List[Dict[str, Any]]:
    now = datetime.now(timezone.utc).timestamp()
    cache = _INSTRUMENTS_CACHE.get(currency)
    if cache and (now - cache.get("fetched_at", 0)) < _INSTRUMENTS_TTL_SECONDS:
        return cache.get("instruments", [])
    result = await _deribit_get(client, "/public/get_instruments", {
        "currency": currency,
        "kind": "option",
        "expired": False,
    })
    instruments = result if isinstance(result, list) else []
    _INSTRUMENTS_CACHE[currency] = {
        "instruments": instruments,
        "fetched_at": now,
    }
    return instruments


async def _get_daily_close(client: httpx.AsyncClient, instrument_name: str, lookback_days: int) -> Tuple[Optional[float], Optional[int]]:
    end_ms = _now_ms()
    start_ms = end_ms - lookback_days * 24 * 60 * 60 * 1000
    try:
        tv = await _deribit_get(client, "/public/get_tradingview_chart_data", {
            "instrument_name": instrument_name,
            "resolution": "1D",
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
        })
    except Exception:
        return None, None
    # Expect arrays: ticks, open, close, high, low, volume
    closes = tv.get("close") or tv.get("closes") or []
    ticks = tv.get("ticks") or tv.get("tick") or []
    if isinstance(closes, list) and closes:
        last_close = _coerce_float(closes[-1])
        last_ts = int(ticks[-1]) if isinstance(ticks, list) and len(ticks) == len(closes) and ticks else None
        return last_close, last_ts
    return None, None


def _select_expiry(instruments: List[Dict[str, Any]], target_date_str: str) -> Optional[Tuple[int, str]]:
    if not instruments:
        return None
    target_dt = _parse_yyyy_mm_dd(target_date_str)
    # Collect unique expiration timestamps
    expiries: List[int] = sorted({int(i.get("expiration_timestamp")) for i in instruments if i.get("expiration_timestamp") is not None})
    if not expiries:
        return None
    def days_diff(ts_ms: int) -> float:
        dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        return (dt.date() - target_dt.date()).days
    # Prefer >= target; else nearest within ±3 days; else nearest
    non_negative = [ts for ts in expiries if days_diff(ts) >= 0]
    if non_negative:
        best = min(non_negative, key=lambda ts: days_diff(ts))
        return best, _to_yyyy_mm_dd(best)
    within3 = [ts for ts in expiries if abs(days_diff(ts)) <= 3]
    if within3:
        best = min(within3, key=lambda ts: abs(days_diff(ts)))
        return best, _to_yyyy_mm_dd(best)
    # Fallback: nearest overall
    best = min(expiries, key=lambda ts: abs(days_diff(ts)))
    return best, _to_yyyy_mm_dd(best)


def _group_by_expiry(instruments: List[Dict[str, Any]]) -> Dict[int, List[Dict[str, Any]]]:
    by_exp: Dict[int, List[Dict[str, Any]]] = {}
    for inst in instruments:
        ts = inst.get("expiration_timestamp")
        if ts is None:
            continue
        ts = int(ts)
        by_exp.setdefault(ts, []).append(inst)
    return by_exp


def _nearest_strike(strikes: List[float], target: float) -> Optional[float]:
    if not strikes:
        return None
    # Choose by absolute distance, then by lower strike on ties
    best = min(strikes, key=lambda s: (abs(s - target), s))
    return best


def _find_instrument(instruments: List[Dict[str, Any]], expiration_timestamp: int, strike: float, option_type: str) -> Optional[Dict[str, Any]]:
    opt = option_type.lower()[0]  # 'c' or 'p'
    def _is_match(x: Dict[str, Any]) -> bool:
        try:
            return (
                int(x.get("expiration_timestamp")) == int(expiration_timestamp)
                and float(x.get("strike")) == float(strike)
                and str(x.get("option_type", "")).lower().startswith(opt)
            )
        except Exception:
            return False
    for inst in instruments:
        if _is_match(inst):
            return inst
    return None


async def deribit_weekly_snapshot(
    currency: str,
    target_date: str,
    lookback_days: Optional[int] = 7,
) -> Dict[str, Any]:
    """
    Return the most recent daily close for the underlying perpetual, the option expiry closest to target_date,
    and ATM option IV/prices for that expiry.
    """
    if currency not in ("BTC", "ETH", "SOL"):
        raise ValueError("currency must be one of: BTC, ETH, SOL")
    lookback = int(lookback_days or 7)
    if lookback <= 0:
        lookback = 7

    timeout = httpx.Timeout(15.0, connect=5.0)
    prefer_http2 = True
    try:
        import h2  # noqa: F401
    except Exception:
        prefer_http2 = False
    _logger.info("[CALL] deribit_weekly_snapshot currency=%s target_date=%s lookback_days=%s", currency, target_date, lookback_days)
    async with httpx.AsyncClient(timeout=timeout, http2=prefer_http2) as client:
        instrument_perp = f"{currency}-PERPETUAL"

        # Fetch latest daily close for underlying perpetual
        _logger.info("[HTTP] get_tradingview_chart_data instrument=%s lookback=%s", instrument_perp, lookback)
        daily_close, last_tick_ts = await _get_daily_close(client, instrument_perp, lookback)
        if daily_close is None:
            raise RuntimeError("Failed to fetch daily close from Deribit")

        # Get instruments and select expiry
        _logger.info("[HTTP] get_instruments currency=%s kind=option", currency)
        instruments = await _get_instruments(client, currency)
        if not instruments:
            raise RuntimeError("No option instruments returned by Deribit for the specified currency")
        expiry_sel = _select_expiry(instruments, target_date)
        if not expiry_sel:
            raise RuntimeError("Failed to select an expiry for the given target_date")
        expiration_ts, expiry_date_str = expiry_sel
        by_exp = _group_by_expiry(instruments)
        on_exp_list = by_exp.get(expiration_ts, [])
        if not on_exp_list:
            raise RuntimeError("No instruments for selected expiry")

        # Determine ATM strike
        strikes = sorted({float(x.get("strike")) for x in on_exp_list if x.get("strike") is not None})
        atm_strike = _nearest_strike(strikes, float(daily_close))
        if atm_strike is None:
            raise RuntimeError("Failed to determine ATM strike for selected expiry")

        # Find corresponding call and put instruments
        call_inst = _find_instrument(on_exp_list, expiration_ts, atm_strike, "call")
        put_inst = _find_instrument(on_exp_list, expiration_ts, atm_strike, "put")
        if not call_inst or not put_inst:
            raise RuntimeError("ATM call/put instruments not found for selected expiry/strike")

        call_name = call_inst.get("instrument_name")
        put_name = put_inst.get("instrument_name")

        # Fetch tickers concurrently
        async def _fetch_ticker(name: str) -> Dict[str, Any]:
            t = await _deribit_get(client, "/public/ticker", {"instrument_name": name})
            fields = _extract_ticker_fields(t)
            return {
                "instrument": name,
                **fields,
            }

        call_task = asyncio.create_task(_fetch_ticker(str(call_name)))
        put_task = asyncio.create_task(_fetch_ticker(str(put_name)))
        call_info, put_info = await asyncio.gather(call_task, put_task)
        _logger.info("[RETURN] deribit_weekly_snapshot expiry=%s atm_strike=%s call_mk=%s put_mk=%s", expiry_date_str, atm_strike, call_info.get("mark_price"), put_info.get("mark_price"))

        asof_dt = datetime.now(timezone.utc)
        # Prefer the candle tick date if available
        if last_tick_ts:
            try:
                asof_dt = datetime.fromtimestamp(int(last_tick_ts) / 1000, tz=timezone.utc)
            except Exception:
                pass

        # Render strike as int when appropriate
        try:
            strike_display: Any = int(round(float(atm_strike)))
        except Exception:
            strike_display = atm_strike

        return {
            "asof_date": asof_dt.strftime("%Y-%m-%d"),
            "underlying": {
                "instrument": instrument_perp,
                "daily_close": _coerce_float(daily_close),
            },
            "expiry": {
                "date": expiry_date_str,
                "expiration_timestamp": int(expiration_ts),
            },
            "atm": {
                "strike": strike_display,
                "call": call_info,
                "put": put_info,
            },
        }


async def deribit_weekly_ladder(
    currency: str,
    expiry_date: str,
    center_strike: float,
    width: Optional[int] = 2,
    both_sides: Optional[bool] = True,
) -> Dict[str, Any]:
    """
    Return a ±width strike ladder around center_strike for the given expiry, fetching mark_iv and mark_price.
    """
    if currency not in ("BTC", "ETH", "SOL"):
        raise ValueError("currency must be one of: BTC, ETH, SOL")
    w = int(width or 1)
    if w <= 0:
        w = 1
    bs = True if both_sides is None else bool(both_sides)

    timeout = httpx.Timeout(15.0, connect=5.0)
    prefer_http2 = True
    try:
        import h2  # noqa: F401
    except Exception:
        prefer_http2 = False
    _logger.info("[CALL] deribit_weekly_ladder currency=%s expiry_date=%s center_strike=%s width=%s both_sides=%s", currency, expiry_date, center_strike, width, both_sides)
    async with httpx.AsyncClient(timeout=timeout, http2=prefer_http2) as client:
        instruments = await _get_instruments(client, currency)
        if not instruments:
            raise RuntimeError("No option instruments returned by Deribit for the specified currency")

        # Group by expiry (format YYYY-MM-DD) for easier selection by date string
        by_exp_ts = _group_by_expiry(instruments)
        exp_map_by_date: Dict[str, List[Dict[str, Any]]] = {}
        for ts, lst in by_exp_ts.items():
            date_str = _to_yyyy_mm_dd(ts)
            exp_map_by_date.setdefault(date_str, []).extend(lst)

        on_exp_list = exp_map_by_date.get(expiry_date)
        if not on_exp_list:
            # try nearest by date if exact not found
            try:
                target_dt = _parse_yyyy_mm_dd(expiry_date)
                candidates: List[Tuple[int, List[Dict[str, Any]]]] = [
                    (ts, lst) for ts, lst in by_exp_ts.items()
                ]
                if not candidates:
                    raise RuntimeError("No instruments for any expiry")
                best_ts, best_list = min(
                    candidates,
                    key=lambda pair: abs((datetime.fromtimestamp(pair[0] / 1000, tz=timezone.utc).date() - target_dt.date()).days),
                )
                on_exp_list = best_list
            except Exception:
                raise RuntimeError("Requested expiry_date not available and no fallback found")

        # Unique sorted strikes
        strikes = sorted({float(x.get("strike")) for x in on_exp_list if x.get("strike") is not None})
        if not strikes:
            raise RuntimeError("No strikes available for the specified expiry")

        # Find nearest index to center_strike
        target = float(center_strike)
        nearest = _nearest_strike(strikes, target)
        if nearest is None:
            raise RuntimeError("Failed to locate nearest strike")
        idx = strikes.index(nearest)
        start = max(0, idx - w)
        end = min(len(strikes) - 1, idx + w)
        selected = strikes[start : end + 1]

        # Build a lookup from (strike, side) -> instrument name
        def _lookup_name(s: float, side: str) -> Optional[str]:
            inst = _find_instrument(on_exp_list, int(on_exp_list[0].get("expiration_timestamp")), s, "call" if side == "C" else "put")
            return inst.get("instrument_name") if inst else None

        async def _fetch(name: str) -> Dict[str, Any]:
            t = await _deribit_get(client, "/public/ticker", {"instrument_name": name})
            fields = _extract_ticker_fields(t)
            return {
                "instrument": name,
                **fields,
            }

        legs: List[Dict[str, Any]] = []
        tasks: List[asyncio.Task] = []
        task_meta: List[Tuple[float, str]] = []  # (strike, side)
        for s in selected:
            name_c = _lookup_name(s, "C")
            if name_c:
                tasks.append(asyncio.create_task(_fetch(name_c)))
                task_meta.append((s, "C"))
            if bs:
                name_p = _lookup_name(s, "P")
                if name_p:
                    tasks.append(asyncio.create_task(_fetch(name_p)))
                    task_meta.append((s, "P"))

        results: List[Dict[str, Any]] = []
        if tasks:
            results = await asyncio.gather(*tasks)

        for (s, side), info in zip(task_meta, results):
            # Render strike as int where possible
            try:
                s_display: Any = int(round(float(s)))
            except Exception:
                s_display = s
            legs.append({
                "strike": s_display,
                "side": side,
                "instrument": info.get("instrument"),
                "mark_iv": info.get("mark_iv"),
                "mark_price": info.get("mark_price"),
            })

        # Sort legs by side then strike
        legs.sort(key=lambda x: (x.get("side"), float(x.get("strike", 0))))
        out = {
            "expiry": expiry_date,
            "legs": legs,
        }
        _logger.info("[RETURN] deribit_weekly_ladder legs=%d sample=%s", len(legs), (legs[0] if legs else None))
        return out


async def deribit_weekly_inputs(
    currency: str,
    target_date: str,
    lookback_days: Optional[int] = 7,
) -> Dict[str, Any]:
    """
    Convenience wrapper: snapshot plus a ±1-strike ladder around ATM for the same expiry.
    """
    _logger.info("[CALL] deribit_weekly_inputs currency=%s target_date=%s lookback_days=%s", currency, target_date, lookback_days)
    snap = await deribit_weekly_snapshot(currency=currency, target_date=target_date, lookback_days=lookback_days)
    expiry_date = snap.get("expiry", {}).get("date")
    atm = snap.get("atm", {})
    strike = atm.get("strike")
    try:
        center = float(strike)
    except Exception:
        # If strike missing, fall back to underlying close
        center = float(snap.get("underlying", {}).get("daily_close") or 0.0)
    ladder = await deribit_weekly_ladder(
        currency=currency,
        expiry_date=str(expiry_date),
        center_strike=center,
        width=1,
        both_sides=True,
    )
    out = dict(snap)
    out["ladder"] = ladder.get("legs", [])
    _logger.info("[RETURN] deribit_weekly_inputs expiry=%s ladder_len=%d", expiry_date, len(out.get("ladder", [])))
    return out


