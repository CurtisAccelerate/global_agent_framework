from __future__ import annotations
import math
import statistics
from typing import Dict, List, Tuple, Iterable, Optional


def american_to_prob(price: float) -> float:
    """Convert American odds to implied probability (without vig)."""
    if price is None:
        return float("nan")
    if price > 0:
        return 100.0 / (price + 100.0)
    else:
        return (-price) / ((-price) + 100.0)


def decimal_to_prob(dec: float) -> float:
    """Convert Decimal odds to implied probability (without vig)."""
    if dec is None or dec <= 1.0:
        return float("nan")
    return 1.0 / dec


def price_to_prob(price: float, odds_format: str) -> float:
    odds_format = (odds_format or "american").lower()
    return american_to_prob(price) if odds_format == "american" else decimal_to_prob(price)


def _power_sum(p_raw: Iterable[float], beta: float) -> float:
    return sum((max(x, 1e-15)) ** beta for x in p_raw)


def devig_power(p_raw: List[float], tol: float = 1e-12, max_iter: int = 60) -> List[float]:
    """
    Remove vig using 'power normalization':
      find beta such that sum(p_i_raw ** beta) = 1, then p_i = (p_i_raw ** beta) / sum(...)
    This handles 2-way, 3-way, and long outright lists gracefully.
    Falls back to simple normalization if the solver does not converge.
    """
    p_raw = [max(float(x), 1e-15) for x in p_raw if math.isfinite(x) and x > 0]
    if not p_raw:
        return []
    s = sum(p_raw)

    if abs(s - 1.0) < 1e-12:
        return [x / s for x in p_raw]

    if s > 1.0:
        lo, hi = 1.0, 10.0
    else:
        lo, hi = 0.01, 1.0

    def f(b):
        return _power_sum(p_raw, b) - 1.0

    while f(hi) > 0 and hi < 100:
        hi *= 2
    while f(lo) < 0 and lo > 1e-6:
        lo /= 2

    it = 0
    while it < max_iter:
        mid = 0.5 * (lo + hi)
        val = f(mid)
        if abs(val) < tol:
            beta = mid
            break
        if (s > 1.0 and val > 0) or (s < 1.0 and val > 0):
            lo = mid
        else:
            hi = mid
        it += 1
    else:
        denom = sum(p_raw)
        return [x / denom for x in p_raw]

    pw = [x ** beta for x in p_raw]
    denom = sum(pw)
    return [x / denom for x in pw]


def better_american(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def better_decimal(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return max(a, b)


def devig_and_aggregate(
    bookmakers: List[dict],
    market_key: str = "h2h",
    odds_format: str = "american",
) -> Dict:
    """
    Given The Odds API 'bookmakers' array for a single market (h2h or outrights),
    return consensus de-vigged probabilities and best prices per outcome name.
    """
    odds_format = (odds_format or "american").lower()
    outcome_probs: Dict[str, List[float]] = {}
    best_price: Dict[str, Tuple[Optional[float], Optional[str]]] = {}
    book_overround: List[float] = []
    book_count = 0

    for bk in bookmakers or []:
        mkts = bk.get("markets") or []
        m = next((m for m in mkts if m.get("key") == market_key), None)
        if not m:
            continue
        oc = m.get("outcomes") or []
        names, p_raws, prices = [], [], []
        for o in oc:
            name = o.get("name")
            price = o.get("price")
            if name is None or price is None:
                continue
            p_raw = price_to_prob(float(price), odds_format)
            if not math.isfinite(p_raw) or p_raw <= 0:
                continue
            names.append(name)
            p_raws.append(p_raw)
            prices.append(price)

        if not names or not p_raws:
            continue

        book_overround.append(sum(p_raws) - 1.0)
        p_fair = devig_power(p_raws)

        for name, p, price in zip(names, p_fair, prices):
            outcome_probs.setdefault(name, []).append(p)
            if odds_format == "american":
                curr, book = best_price.get(name, (None, None))
                bp = better_american(curr, float(price))
                best_price[name] = (bp, bk.get("title"))
            else:
                curr, book = best_price.get(name, (None, None))
                bp = better_decimal(curr, float(price))
                best_price[name] = (bp, bk.get("title"))

        book_count += 1

    outcomes = []
    for name, plist in outcome_probs.items():
        outcomes.append({
            "name": name,
            "prob_mean": float(sum(plist) / len(plist)),
            "prob_median": float(statistics.median(plist)),
            "num_books": len(plist),
            "best_price": best_price.get(name, (None, None))[0],
            "best_book": best_price.get(name, (None, None))[1],
        })

    outcomes.sort(key=lambda x: x["prob_mean"], reverse=True)

    return {
        "market": market_key,
        "outcomes": outcomes,
        "book_count": book_count,
        "book_overround": book_overround,
    }


def devig_from_odds_api_payload(
    event_payload: dict,
    market_key: str = "h2h",
    odds_format: str = "american",
) -> Dict:
    return devig_and_aggregate(
        bookmakers=event_payload.get("bookmakers") or [],
        market_key=market_key,
        odds_format=odds_format,
    )


