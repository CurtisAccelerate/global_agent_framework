"""
Comprehensive test script for the prediction pipeline and API endpoint
Tests both direct pipeline execution and API calls with detailed logging
"""
import asyncio
import json
import logging
import os
import time
from typing import Dict, Any
import requests
from config import Config
from agent_pipeline_declarations import create_prediction_pipeline
from pipeline import Pipeline

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('test_results.log')
    ]
)
logger = logging.getLogger(__name__)

class PipelineTester:
    """Test the prediction pipeline directly"""
    
    def __init__(self):
        self.pipeline = create_prediction_pipeline()
        # Load test inputs from config file if present
        cfg_path = os.getenv("TEST_INPUTS_PATH", "test_inputs.json")
        self.test_inputs = self._load_inputs(cfg_path)

    @staticmethod
    def _load_inputs(config_path: str) -> list:
        try:
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                if isinstance(cfg, dict):
                    if cfg.get("input"):
                        return [str(cfg["input"])]
                    if isinstance(cfg.get("inputs"), list) and cfg["inputs"]:
                        return [str(x) for x in cfg["inputs"]]
        except Exception as e:
            logging.getLogger(__name__).error(f"Failed to load test inputs from {config_path}: {e}")
        # Fallback defaults
        return [
            "Predict the impact of AI on job markets over the next 5 years",
            "How will climate change affect global food production in the next decade?",
        ]
    
    async def test_pipeline_direct(self, test_input: str) -> Dict[str, Any]:
        """Test pipeline execution directly with detailed logging"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING PIPELINE DIRECTLY")
        logger.info(f"Input: {test_input}")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            result = await self.pipeline.execute(test_input)
            execution_time = time.time() - start_time
            
            test_result = {
                "test_type": "pipeline_direct",
                "input": test_input,
                "success": result.success,
                "execution_time": execution_time,
                "data": result.data,
                "error": result.error,
                "stage_results": []
            }
            
            if result.stage_results:
                for i, stage_result in enumerate(result.stage_results):
                    stage_info = {
                        "stage_number": i + 1,
                        "agent_name": stage_result.metadata.get('agent_name', 'Unknown'),
                        "success": stage_result.success,
                        "data_preview": str(stage_result.data)[:200] + "..." if stage_result.data else None,
                        "error": stage_result.error,
                        "metadata": stage_result.metadata
                    }
                    test_result["stage_results"].append(stage_info)
                    
                    logger.info(f"\n--- STAGE {i+1} RESULT ---")
                    logger.info(f"Agent: {stage_info['agent_name']}")
                    logger.info(f"Success: {stage_info['success']}")
                    logger.info(f"Data Preview: {stage_info['data_preview']}")
                    if stage_info['error']:
                        logger.error(f"Error: {stage_info['error']}")
                    if stage_info['metadata']:
                        logger.debug(f"Metadata: {stage_info['metadata']}")
            
            logger.info(f"\n--- FINAL RESULT ---")
            logger.info(f"Success: {result.success}")
            logger.info(f"Execution Time: {execution_time:.2f}s")
            if result.success:
                logger.info(f"Final Data: {result.data}")
            else:
                logger.error(f"Error: {result.error}")
            
            return test_result
            
        except Exception as e:
            logger.exception(f"Pipeline test failed with exception: {e}")
            return {
                "test_type": "pipeline_direct",
                "input": test_input,
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e),
                "stage_results": []
            }
    
    async def run_all_pipeline_tests(self) -> list:
        """Run all pipeline tests"""
        logger.info("Starting pipeline direct tests...")
        results = []
        
        for i, test_input in enumerate(self.test_inputs):
            logger.info(f"\nRunning pipeline test {i+1}/{len(self.test_inputs)}")
            result = await self.test_pipeline_direct(test_input)
            results.append(result)
            
            # Add delay between tests
            await asyncio.sleep(1)
        
        return results

class APITester:
    """Test the API endpoint"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_inputs = [
            "Predict the impact of AI on job markets over the next 5 years",
            "What will be the adoption rate of autonomous vehicles in major cities by 2030?",
            "How will climate change affect global food production in the next decade?"
        ]
    
    def test_api_health(self) -> Dict[str, Any]:
        """Test API health endpoint"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING API HEALTH")
        logger.info(f"{'='*80}")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            result = {
                "test_type": "api_health",
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            logger.info(f"Health check - Status: {response.status_code}")
            logger.info(f"Response: {result['response']}")
            
            return result
            
        except Exception as e:
            logger.exception(f"API health test failed: {e}")
            return {
                "test_type": "api_health",
                "success": False,
                "error": str(e)
            }
    
    def test_api_pipeline_execution(self, test_input: str) -> Dict[str, Any]:
        """Test API pipeline execution endpoint"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING API PIPELINE EXECUTION")
        logger.info(f"Input: {test_input}")
        logger.info(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            payload = {
                "pipeline_name": "prediction_pipeline",
                "input": test_input
            }
            
            logger.info(f"API Request Payload: {json.dumps(payload, indent=2)}")
            
            response = requests.post(
                f"{self.base_url}/execute",
                json=payload,
                timeout=300  # 5 minute timeout for long-running pipeline
            )
            
            execution_time = time.time() - start_time
            
            result = {
                "test_type": "api_pipeline_execution",
                "input": test_input,
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "execution_time": execution_time,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            logger.info(f"API Response - Status: {response.status_code}")
            logger.info(f"Execution Time: {execution_time:.2f}s")
            
            if response.status_code == 200:
                response_data = response.json()
                logger.info(f"Success: {response_data.get('success', False)}")
                logger.info(f"Data: {response_data.get('data', 'No data')}")
                if 'stage_results' in response_data:
                    logger.info(f"Stage Results Count: {len(response_data['stage_results'])}")
                    for i, stage in enumerate(response_data['stage_results']):
                        logger.info(f"Stage {i+1}: {stage.get('agent_name', 'Unknown')} - Success: {stage.get('success', False)}")
            else:
                logger.error(f"API Error: {result['response']}")
            
            return result
            
        except Exception as e:
            logger.exception(f"API pipeline test failed: {e}")
            return {
                "test_type": "api_pipeline_execution",
                "input": test_input,
                "success": False,
                "execution_time": time.time() - start_time,
                "error": str(e)
            }
    
    def test_api_list_pipelines(self) -> Dict[str, Any]:
        """Test API list pipelines endpoint"""
        logger.info(f"\n{'='*80}")
        logger.info(f"TESTING API LIST PIPELINES")
        logger.info(f"{'='*80}")
        
        try:
            response = requests.get(f"{self.base_url}/pipelines", timeout=10)
            result = {
                "test_type": "api_list_pipelines",
                "success": response.status_code == 200,
                "status_code": response.status_code,
                "response": response.json() if response.status_code == 200 else response.text
            }
            
            logger.info(f"List Pipelines - Status: {response.status_code}")
            logger.info(f"Response: {result['response']}")
            
            return result
            
        except Exception as e:
            logger.exception(f"API list pipelines test failed: {e}")
            return {
                "test_type": "api_list_pipelines",
                "success": False,
                "error": str(e)
            }
    
    def run_all_api_tests(self) -> list:
        """Run all API tests"""
        logger.info("Starting API tests...")
        results = []
        
        # Test health
        results.append(self.test_api_health())
        
        # Test list pipelines
        results.append(self.test_api_list_pipelines())
        
        # Test pipeline execution
        for i, test_input in enumerate(self.test_inputs):
            logger.info(f"\nRunning API test {i+1}/{len(self.test_inputs)}")
            result = self.test_api_pipeline_execution(test_input)
            results.append(result)
            
            # Add delay between tests
            time.sleep(2)
        
        return results

async def main():
    """Main test function"""
    logger.info("Starting comprehensive pipeline and API tests...")
    
    # Validate configuration
    try:
        Config.validate()
        logger.info("Configuration validated successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Test pipeline directly
    logger.info("\n" + "="*100)
    logger.info("PHASE 1: TESTING PIPELINE DIRECTLY")
    logger.info("="*100)
    
    pipeline_tester = PipelineTester()
    pipeline_results = await pipeline_tester.run_all_pipeline_tests()
    
    # Test API
    logger.info("\n" + "="*100)
    logger.info("PHASE 2: TESTING API ENDPOINTS")
    logger.info("="*100)
    
    api_tester = APITester()
    api_results = api_tester.run_all_api_tests()
    
    # Summary
    logger.info("\n" + "="*100)
    logger.info("TEST SUMMARY")
    logger.info("="*100)
    
    pipeline_success = sum(1 for r in pipeline_results if r.get('success', False))
    api_success = sum(1 for r in api_results if r.get('success', False))
    
    logger.info(f"Pipeline Tests: {pipeline_success}/{len(pipeline_results)} passed")
    logger.info(f"API Tests: {api_success}/{len(api_results)} passed")
    
    # Save detailed results
    all_results = {
        "pipeline_results": pipeline_results,
        "api_results": api_results,
        "summary": {
            "pipeline_tests_passed": pipeline_success,
            "pipeline_tests_total": len(pipeline_results),
            "api_tests_passed": api_success,
            "api_tests_total": len(api_results),
            "timestamp": time.time()
        }
    }
    
    with open('test_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"Detailed results saved to test_results.json")
    logger.info(f"Log file saved to test_results.log")

if __name__ == "__main__":
    asyncio.run(main())
