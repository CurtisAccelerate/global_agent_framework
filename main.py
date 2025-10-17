"""
Main entry point for the Agent Framework
"""
import asyncio
import logging
from config import Config
from api import app, pipelines
from agent_pipeline_declarations import (
    create_prediction_pipeline,
    create_prediction_pipeline_stub,
    create_research_pipeline,
)
from logging_config import setup_logging

# Setup logging with real-time flushing and auto-clear, honoring verbosity switches
setup_logging(
    debug=Config.DEBUG,
    auto_clear=True,
    log_verbosity=Config.LOG_VERBOSITY,
    errors_only=Config.ERRORS_ONLY,
)
logger = logging.getLogger(__name__)

def initialize_example_pipelines():
    """Initialize example pipelines"""
    try:
        # Create and register the default pipelines
        futuerex_pipeline = create_prediction_pipeline()
        pipelines[futuerex_pipeline.name] = futuerex_pipeline
        # Provide legacy alias so clients referencing "prediction_pipeline" continue to work
        pipelines.setdefault("prediction_pipeline", futuerex_pipeline)

        research_pipeline = create_research_pipeline()
        pipelines[research_pipeline.name] = research_pipeline
        
        logger.info(f"Initialized {len(pipelines)} example pipelines")
        for name in pipelines.keys():
            logger.info(f"  - {name}")
            
    except Exception as e:
        logger.error(f"Failed to initialize example pipelines: {e}")

async def run_prediction_pipeline_async(input_text: str, use_stub: bool = False) -> None:
    pipeline = create_prediction_pipeline_stub() if use_stub else create_prediction_pipeline()
    logger.info(f"Running pipeline: {pipeline.name}")
    logger.info(f"Input text: {input_text}")
    result = await pipeline.execute(input_text, None)
    if result.success:
        logger.info(f"Pipeline {pipeline.name} finished successfully.")
        logger.info(f"Final output preview: {str(result.data)[:200]}")
    else:
        logger.error(f"Pipeline {pipeline.name} failed: {result.error}")

def main():
    """Main function"""
    try:
        # Validate configuration
        Config.validate()
        logger.info("Configuration validated successfully")
        
        # Initialize example pipelines
        initialize_example_pipelines()
        
        # Start the API server
        import uvicorn
        # Respect environment override for PORT if set; default remains Config.PORT
        logger.info(f"Starting server on {Config.HOST}:{Config.PORT} (DEBUG={Config.DEBUG})")
        uvicorn.run(app, host=Config.HOST, port=Config.PORT)
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the prediction pipeline with example input")
    parser.add_argument("input", nargs="?", default="Predict the impact of AI on job markets over next 5 years", help="Input text for the pipeline")
    parser.add_argument("--stub", action="store_true", help="Use stubbed prediction pipeline (no external API calls)")
    args = parser.parse_args()
    asyncio.run(run_prediction_pipeline_async(args.input, use_stub=args.stub))
