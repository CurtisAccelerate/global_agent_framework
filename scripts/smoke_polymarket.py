import asyncio
import json
import os
import sys

# Allow running this script directly: add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.polymarket_gamma import polymarket_gamma_get_odds


async def main():
    res = await polymarket_gamma_get_odds(query="US election", limit=3)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


