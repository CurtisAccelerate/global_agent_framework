"""
Simple test script for the prediction pipeline
Tests pipeline execution directly with detailed logging
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any, List
from config import Config
from agent_pipeline_declarations import create_prediction_pipeline
from logging_config import setup_logging

# Setup logging with real-time flushing and auto-clear
setup_logging(debug=False, auto_clear=True, log_verbosity="minimal")
logger = logging.getLogger(__name__)

def _load_test_inputs(config_path: str = "test_inputs.json") -> List[str]:
    """Load test inputs from a JSON config file. Supports either 'input' or 'inputs'."""
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
            # Prefer explicit 'input' for single-question runs
            if isinstance(cfg, dict):
                if cfg.get("input"):
                    return [str(cfg["input"])]
                if isinstance(cfg.get("inputs"), list) and cfg["inputs"]:
                    inputs_list = [str(x) for x in cfg["inputs"]]
                    # Optional: limit test count via config
                    try:
                        test_count = int(cfg.get("test_count", 0))
                    except Exception:
                        test_count = 0
                    if test_count and test_count > 0:
                        return inputs_list[:test_count]
                    return inputs_list
        # Fallback to default if config missing/empty
    except Exception as e:
        logger.error(f"Failed to load test inputs from {config_path}: {e}")
    # Default fallback
    return [
        (
            "You are an agent that can predict future events. The event to be predicted: \"Bayer Leverkusen vs. Eintracht Frankfurt (resolved around 2025-09-13 (GMT+8)).\n"
            "A. Bayer Leverkusen win on 2025-09-12\n"
            "B. Bayer Leverkusen vs. Eintracht Frankfurt end in a draw\n"
            "C. Eintracht Frankfurt win on 2025-09-12\"\n"
            "IMPORTANT: Your final answer MUST end with this exact format:\n"
            "listing all plausible options you have identified, separated by commas, within the box. For example: \\boxed{A} for a single option or \\boxed{B, C, D} for multiple options.\n"
            "Do not use any other format. Do not refuse to make a prediction. Do not say \"I cannot predict the future.\" You must make a clear prediction based on the best data currently available, using the box format specified above."
        )
    ]


async def test_pipeline():
    """Test the prediction pipeline directly"""
    logger.info("Starting pipeline test...")
    
    # Force DEBUG mode for detailed logging
    import config
    config.Config.DEBUG = True
    logger.info("DEBUG mode enabled for detailed logging")

    # Validate configuration
    try:
        Config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Create pipeline
    pipeline = create_prediction_pipeline()
    logger.info(f"Created pipeline: {pipeline.name}")
    logger.info(f"Pipeline description: {pipeline.description}")
    logger.info(f"Number of stages: {len(pipeline.stages)}")
    
    # Load test inputs declaratively from JSON config
    test_inputs: List[str] = _load_test_inputs()
    logger.info(f"Loaded {len(test_inputs)} test input(s) from config")
    
    # Execute pipeline
    logger.info("\n" + "="*80)
    logger.info("EXECUTING PIPELINE")
    logger.info("="*80)
    
    start_time = time.time()
    
    all_results: List[Dict[str, Any]] = []
    try:
        for idx, test_input in enumerate(test_inputs, start=1):
            logger.info(f"Test input {idx}/{len(test_inputs)}: {str(test_input)[:200]}")
            result = await pipeline.execute(test_input)
            execution_time = time.time() - start_time
            
            logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
            logger.info(f"Success: {result.success}")
            if result.success:
                logger.info(f"Final result: {result.data}")
            else:
                logger.error(f"Error: {result.error}")
            
            # Log stage results
            if result.stage_results:
                logger.info(f"\nStage Results ({len(result.stage_results)} stages):")
                for i, stage_result in enumerate(result.stage_results):
                    logger.info(f"\n--- STAGE {i+1} ---")
                    logger.info(f"Agent: {stage_result.metadata.get('agent_name', 'Unknown') if stage_result.metadata else 'Unknown'}")
                    logger.info(f"Success: {stage_result.success}")
                    if stage_result.data:
                        logger.info(f"FULL OUTPUT ({len(str(stage_result.data))} chars):\n{str(stage_result.data)}")
                    else:
                        logger.info("Data: None")
                    if stage_result.error:
                        logger.error(f"Error: {stage_result.error}")
                    if stage_result.metadata:
                        logger.debug(f"Metadata: {stage_result.metadata}")
            
            run_result: Dict[str, Any] = {
                "input": test_input,
                "success": result.success,
                "execution_time": execution_time,
                "data": result.data,
                "error": result.error,
                "stage_results": [
                    {
                        "stage_number": i + 1,
                        "agent_name": stage_result.metadata.get('agent_name', 'Unknown') if stage_result.metadata else 'Unknown',
                        "success": stage_result.success,
                        "data_full": str(stage_result.data) if stage_result.data else None,
                        "error": stage_result.error,
                        "metadata": stage_result.metadata
                    }
                    for i, stage_result in enumerate(result.stage_results)
                ],
            }
            all_results.append(run_result)
        
        # Save aggregated results
        output_payload: Dict[str, Any] = {
            "count": len(all_results),
            "results": all_results,
        }
        with open('logs/pipeline_test_results.json', 'w', encoding='utf-8') as f:
            json.dump(output_payload, f, indent=2, ensure_ascii=False, default=str)
        
        # Emit full JSON in logs
        try:
            logger.info("Full JSON results start >>>")
            logger.info(json.dumps(output_payload, indent=2, ensure_ascii=False, default=str))
            logger.info("<<< Full JSON results end")
        except Exception:
            pass
        
        logger.info(f"\nTest results saved to logs/pipeline_test_results.json")
        logger.info(f"Log saved to pipeline_test_results.log")
        return output_payload
    except Exception as e:
        logger.exception(f"Pipeline test failed with exception: {e}")
        return {
            "count": 0,
            "results": [],
            "error": str(e),
            "execution_time": time.time() - start_time,
        }

if __name__ == "__main__":
    asyncio.run(test_pipeline())
