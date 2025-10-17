"""
Real API test with proper logging
"""
import requests
import json
import time
import logging
from config import Config
from logging_config import setup_logging

# Setup logging with auto-clear
setup_logging(debug=True, auto_clear=True)
logger = logging.getLogger(__name__)

def test_api():
    base_url = "http://127.0.0.1:8001"
    
    logger.info("="*80)
    logger.info("TESTING API ENDPOINTS")
    logger.info("="*80)
    
    # Test 1: Root endpoint
    logger.info("1. Testing root endpoint...")
    try:
        response = requests.get(f"{base_url}/")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Response: {response.json()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # Test 2: List pipelines
    logger.info("2. Testing list pipelines...")
    try:
        response = requests.get(f"{base_url}/pipelines")
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Response: {response.json()}")
    except Exception as e:
        logger.error(f"Error: {e}")
    
    # Test 3: Execute pipeline via /responses
    logger.info("3. Testing pipeline execution (/responses)...")
    try:
        payload = {
            "pipeline_name": "prediction_pipeline",
            "input_data": "Predict the impact of AI on job markets over the next 5 years"
        }
        
        logger.info(f"Sending payload: {json.dumps(payload, indent=2)}")
        
        start_time = time.time()
        response = requests.post(f"{base_url}/responses", json=payload, timeout=600)  # 10 minute timeout
        execution_time = time.time() - start_time
        
        logger.info(f"Status: {response.status_code}")
        logger.info(f"Execution time: {execution_time:.2f}s")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"Success: {result.get('success', False)}")
            logger.info(f"Data: {result.get('data', 'No data')}")
            if 'error' in result:
                logger.error(f"Error: {result['error']}")
            if 'stage_results' in result:
                logger.info(f"Stage results: {len(result['stage_results'])} stages")
                for i, stage in enumerate(result['stage_results']):
                    logger.info(f"  Stage {i+1}: {stage.get('agent_name', 'Unknown')} - Success: {stage.get('success', False)}")
                    if 'error' in stage:
                        logger.error(f"    Error: {stage['error']}")
                    if 'metadata' in stage and stage['metadata']:
                        logger.info(f"    Metadata: {stage['metadata']}")
        else:
            logger.error(f"Error response: {response.text}")
            
    except Exception as e:
        logger.error(f"Error: {e}")

if __name__ == "__main__":
    test_api()
