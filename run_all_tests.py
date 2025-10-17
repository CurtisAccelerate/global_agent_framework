"""
Comprehensive test runner for the agent framework
Tests pipeline execution, API endpoints, and ensures proper return values
"""
import asyncio
import json
import logging
import time
import requests
import subprocess
import sys
from typing import Dict, Any
from config import Config
from agent_pipeline_declarations import create_prediction_pipeline
from logging_config import setup_logging

# Setup logging with real-time flushing and auto-clear
setup_logging(debug=True, auto_clear=True)
logger = logging.getLogger(__name__)

class TestRunner:
    def __init__(self):
        self.base_url = "http://127.0.0.1:8001"
        self.server_process = None
        
    def start_server(self):
        """Start the API server"""
        logger.info("Starting API server...")
        try:
            # Start uvicorn on a free port (8001) explicitly
            # Consolidate server logs into pipeline.log to avoid log flooding
            server_log = open("logs/pipeline.log", "a", encoding="utf-8")
            self.server_process = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "api:app", "--host", "127.0.0.1", "--port", "8001"],
                stdout=server_log,
                stderr=server_log,
                text=True
            )
            # Poll readiness up to 20s
            deadline = time.time() + 20
            ready = False
            while time.time() < deadline:
                try:
                    r = requests.get(f"{self.base_url}/", timeout=2)
                    if r.status_code == 200:
                        ready = True
                        break
                except Exception:
                    time.sleep(0.5)
            if not ready:
                logger.error("Server failed to become ready within timeout")
                return False
            logger.info("Server started and ready")
            return True
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop_server(self):
        """Stop the API server"""
        if self.server_process:
            logger.info("Stopping API server...")
            try:
                self.server_process.terminate()
                self.server_process.wait(timeout=5)
            except Exception:
                self.server_process.kill()
            logger.info("Server stopped")
    
    async def test_pipeline_direct(self):
        """Test pipeline execution directly"""
        logger.info("="*80)
        logger.info("TESTING PIPELINE DIRECT EXECUTION")
        logger.info("="*80)
        
        try:
            # Validate configuration
            Config.validate()
            logger.info("Configuration validated successfully")
            
            # Create pipeline
            pipeline = create_prediction_pipeline()
            logger.info(f"Created pipeline: {pipeline.name}")
            
            # Test input
            test_input = "Predict the impact of AI on job markets over the next 5 years"
            logger.info(f"Test input: {test_input}")
            
            # Execute pipeline
            start_time = time.time()
            result = await pipeline.execute(test_input)
            execution_time = time.time() - start_time
            
            logger.info(f"Pipeline execution completed in {execution_time:.2f} seconds")
            logger.info(f"Success: {result.success}")
            
            if result.success:
                logger.info(f"Final result: {result.data}")
                logger.info("✅ PIPELINE DIRECT TEST PASSED")
            else:
                logger.error(f"Pipeline failed: {result.error}")
                logger.error("❌ PIPELINE DIRECT TEST FAILED")
            
            # Log stage results
            if result.stage_results:
                logger.info(f"Stage Results ({len(result.stage_results)} stages):")
                for i, stage_result in enumerate(result.stage_results):
                    logger.info(f"Stage {i+1}: Success={stage_result.success}")
                    if stage_result.error:
                        logger.error(f"  Error: {stage_result.error}")
                    if stage_result.data:
                        logger.info(f"  Data preview: {str(stage_result.data)[:100]}...")
            
            return result
            
        except Exception as e:
            logger.exception(f"Pipeline direct test failed: {e}")
            return None
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        logger.info("="*80)
        logger.info("TESTING API ENDPOINTS")
        logger.info("="*80)
        
        # Test 1: Root endpoint
        logger.info("1. Testing root endpoint...")
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            logger.info(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Response: {data}")
                logger.info("✅ Root endpoint test passed")
            else:
                logger.error(f"❌ Root endpoint test failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ Root endpoint test failed: {e}")
        
        # Test 2: List pipelines
        logger.info("2. Testing list pipelines...")
        try:
            response = requests.get(f"{self.base_url}/pipelines", timeout=10)
            logger.info(f"Status: {response.status_code}")
            if response.status_code == 200:
                data = response.json()
                logger.info(f"Response: {data}")
                logger.info("✅ List pipelines test passed")
            else:
                logger.error(f"❌ List pipelines test failed: {response.text}")
        except Exception as e:
            logger.error(f"❌ List pipelines test failed: {e}")
        
        # Test 3: Execute pipeline via /responses
        logger.info("3. Testing pipeline execution via /responses...")
        try:
            payload = {
                "pipeline_name": "prediction_pipeline",
                "input_data": "Predict the impact of AI on job markets over the next 5 years"
            }
            
            logger.info(f"Sending payload: {json.dumps(payload, indent=2)}")
            
            start_time = time.time()
            response = requests.post(f"{self.base_url}/responses", json=payload, timeout=600)
            execution_time = time.time() - start_time
            
            logger.info(f"Status: {response.status_code}")
            logger.info(f"Execution time: {execution_time:.2f}s")
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Success: {result.get('success', False)}")
                
                if result.get('success', False):
                    logger.info(f"Data: {result.get('data', 'No data')}")
                    logger.info("✅ API pipeline execution test passed")
                else:
                    logger.error(f"Pipeline failed: {result.get('error', 'Unknown error')}")
                    logger.error("❌ API pipeline execution test failed")
                
                if 'stage_results' in result:
                    logger.info(f"Stage results: {len(result['stage_results'])} stages")
                    for i, stage in enumerate(result['stage_results']):
                        logger.info(f"  Stage {i+1}: Success={stage.get('success', False)}")
                        if stage.get('error'):
                            logger.error(f"    Error: {stage['error']}")
            else:
                logger.error(f"❌ API pipeline execution test failed: {response.text}")
                
        except Exception as e:
            logger.error(f"❌ API pipeline execution test failed: {e}")
    
    async def run_all_tests(self):
        """Run all tests"""
        logger.info("🚀 STARTING COMPREHENSIVE TEST SUITE")
        logger.info("="*80)
        
        # Test 1: Pipeline Direct
        logger.info("PHASE 1: Testing Pipeline Direct Execution")
        pipeline_result = await self.test_pipeline_direct()
        
        # Test 2: API Server
        logger.info("\nPHASE 2: Testing API Server")
        if self.start_server():
            try:
                self.test_api_endpoints()
            finally:
                self.stop_server()
        else:
            logger.error("❌ Failed to start API server")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TEST SUMMARY")
        logger.info("="*80)
        
        if pipeline_result and pipeline_result.success:
            logger.info("✅ Pipeline Direct: PASSED")
        else:
            logger.info("❌ Pipeline Direct: FAILED")
        
        logger.info("✅ API Server: Tested (check logs for individual results)")
        
        logger.info("\n🎯 All tests completed! Check logs for detailed results.")

async def main():
    """Main test function"""
    test_runner = TestRunner()
    await test_runner.run_all_tests()

if __name__ == "__main__":
    asyncio.run(main())
