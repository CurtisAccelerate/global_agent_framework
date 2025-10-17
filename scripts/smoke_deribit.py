import asyncio
import json
from datetime import datetime, timedelta
import os
import sys

# Allow running this script directly: add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tools.deribit import deribit_weekly_inputs


async def main():
    target_date = (datetime.utcnow() + timedelta(days=7)).strftime("%Y-%m-%d")
    res = await deribit_weekly_inputs(currency="BTC", target_date=target_date, lookback_days=7)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    asyncio.run(main())


