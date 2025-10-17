import asyncio
import json
import os
import sys

# Allow running this script directly: add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.odds_api import odds_find, odds_get


async def main():
    # 1) Find a match candidate using defaults inside odds_find
    find = await odds_find(q="Juventus vs Inter")
    matches = find.get("matches") or []
    if not matches:
        # Fallback: try a futures example if discovery yielded nothing (defaults inside odds_find)
        fut_find = await odds_find(q="super bowl winner")
        outrights = fut_find.get("outrights") or []
        if not outrights:
            print(json.dumps({"error": "No matches or outrights found in discovery."}, indent=2))
            return
        sk = outrights[0]["sport_key"]
        res = await odds_get(market="outrights", sport_key=sk)
        print(json.dumps(res, indent=2))
        return
    m0 = matches[0]
    sport_key = m0["sport_key"]
    event_id = m0["event_id"]

    # 2) Fetch h2h odds for the first match
    res = await odds_get(market="h2h", sport_key=sport_key, event_ids=[event_id])
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


