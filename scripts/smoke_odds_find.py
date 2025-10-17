import asyncio
import json
import os
import sys

# Allow running this script directly: add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.odds_api import odds_find


async def main():
    # Simplified smoke: rely on function defaults (window_days=14, limit=3)
    res = await odds_find(q="super bowl winner")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


