import asyncio
import json

from config import Config
from logging_config import setup_logging
from agent_pipeline_declarations import create_prediction_pipeline_stub


async def main():
    setup_logging(
        debug=Config.DEBUG,
        auto_clear=False,
        log_verbosity=Config.LOG_VERBOSITY,
        errors_only=Config.ERRORS_ONLY,
    )

    pipeline = create_prediction_pipeline_stub()
    result = await pipeline.execute("Stub input question", None)
    data = {
        "success": result.success,
        "data": result.data,
        "stage_results": [
            {
                "stage": idx + 1,
                "success": sr.success,
                "output": sr.data,
                "metadata": sr.metadata,
            }
            for idx, sr in enumerate(result.stage_results or [])
        ],
    }
    print(json.dumps(data, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
