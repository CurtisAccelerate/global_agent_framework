"""
Demo script showing the pipeline structure and expected inputs/outputs
This demonstrates the 3-stage prediction pipeline without requiring API calls
"""
import json
from agent_pipeline_declarations import create_prediction_pipeline
from agent import ReasoningConfig

def demo_pipeline_structure():
    """Demonstrate the pipeline structure and configuration"""
    print("="*80)
    print("PREDICTION PIPELINE DEMO")
    print("="*80)
    
    # Create pipeline
    pipeline = create_prediction_pipeline()
    
    print(f"\nPipeline Name: {pipeline.name}")
    print(f"Description: {pipeline.description}")
    print(f"Number of Stages: {len(pipeline.stages)}")
    
    print("\n" + "="*80)
    print("STAGE CONFIGURATIONS")
    print("="*80)
    
    for i, stage in enumerate(pipeline.stages):
        print(f"\n--- STAGE {i+1}: {stage.name} ---")
        print(f"Model: {stage.model}")
        print(f"System Prompt (first 100 chars): {stage.system_prompt[:100]}...")
        
        if hasattr(stage, 'reasoning_config') and stage.reasoning_config:
            config = stage.reasoning_config
            print(f"Reasoning Effort: {config.reasoning_effort}")
            print(f"Max Output Tokens: {config.max_output_tokens}")
            print(f"Temperature: {config.temperature}")
            print(f"Verbosity: {config.verbosity}")
            print(f"Text Format Type: {config.text_format_type}")
            print(f"Tools: {config.tools}")
            print(f"Tool Choice: {config.tool_choice}")
            if config.include:
                print(f"Include: {config.include}")
            if config.text_json_schema:
                print(f"JSON Schema: {json.dumps(config.text_json_schema, indent=2)}")

def demo_expected_flow():
    """Demonstrate the expected data flow through the pipeline"""
    print("\n" + "="*80)
    print("EXPECTED DATA FLOW")
    print("="*80)
    
    sample_input = "Predict the impact of AI on job markets over the next 5 years"
    print(f"\nInput: {sample_input}")
    
    print("\n--- STAGE 1 OUTPUT (Context Generation) ---")
    stage1_output = {
        "task_restatement": "Analyze and predict the impact of artificial intelligence on global job markets over the next 5 years",
        "key_uncertainties": [
            "Rate of AI adoption across different industries",
            "Job displacement vs. job creation ratios",
            "Skill transition requirements and timelines",
            "Regulatory and policy responses",
            "Economic factors affecting implementation",
            "Geographic variations in impact",
            "Sector-specific vulnerability patterns"
        ],
        "first_principles": [
            "Technological advancement follows exponential curves",
            "Economic systems adapt to productivity changes",
            "Human capital development requires time and investment",
            "Market forces drive efficiency optimization"
        ],
        "core_searches": [
            "AI job displacement studies 2024-2029",
            "Automation impact manufacturing services",
            "Reskilling programs effectiveness",
            "AI productivity gains measurement"
        ],
        "solution_categories": [
            "Policy interventions",
            "Educational reforms", 
            "Economic incentives",
            "Technology governance"
        ]
    }
    print(json.dumps(stage1_output, indent=2))
    
    print("\n--- STAGE 2 OUTPUT (Deep Research) ---")
    stage2_output = {
        "research_approach": "Systematic web search and analysis of recent studies",
        "evidence_findings": [
            "McKinsey study: 15% of workers may need to change occupations by 2030",
            "World Economic Forum: 97 million new roles may emerge from AI",
            "MIT research: AI complements rather than replaces most jobs",
            "OECD analysis: Skill-biased technological change patterns"
        ],
        "confidence_assessment": "92% confidence after 3 research iterations",
        "remaining_uncertainties": [
            "Exact timeline for specific job categories",
            "Regional policy variations impact"
        ],
        "sources": [
            "https://www.mckinsey.com/featured-insights/future-of-work",
            "https://www.weforum.org/reports/the-future-of-jobs-report-2023",
            "https://economics.mit.edu/research/ai-and-productivity"
        ]
    }
    print(json.dumps(stage2_output, indent=2))
    
    print("\n--- STAGE 3 OUTPUT (Final JSON Response) ---")
    stage3_output = {
        "executive_summary": "AI will significantly transform job markets over the next 5 years, with net positive impact through job creation exceeding displacement, but requiring substantial workforce adaptation.",
        "prediction": "By 2029, AI will create 2.3 million new jobs while displacing 1.8 million existing roles, resulting in a net gain of 500,000 jobs globally, with 60% of workers requiring reskilling.",
        "confidence_level": 92,
        "evidence_base": [
            "McKinsey Global Institute 2024 study on AI workforce impact",
            "World Economic Forum Future of Jobs Report 2023",
            "MIT Economics Department research on AI productivity",
            "OECD analysis of skill-biased technological change"
        ],
        "confidence_rationale": "High confidence based on multiple authoritative sources, consistent patterns across studies, and clear trend data from 2020-2024 period.",
        "risks": [
            "Uneven geographic distribution of job creation",
            "Skills gap may slow transition",
            "Policy uncertainty could affect implementation",
            "Economic downturns may accelerate displacement"
        ],
        "methodology": "Systematic analysis of 15+ recent studies, web search of current data, cross-referencing multiple sources, confidence assessment through iterative research.",
        "sources": [
            "McKinsey Global Institute",
            "World Economic Forum", 
            "MIT Economics",
            "OECD",
            "Bureau of Labor Statistics"
        ],
        "limitations": "Predictions based on current trends; external shocks (recessions, policy changes) could significantly alter outcomes. Regional variations not fully quantified."
    }
    print(json.dumps(stage3_output, indent=2))

def demo_api_endpoints():
    """Demonstrate the API endpoint structure"""
    print("\n" + "="*80)
    print("API ENDPOINT DEMO")
    print("="*80)
    
    print("\n1. Health Check:")
    print("GET /health")
    print("Response: {'status': 'healthy', 'pipelines': 1}")
    
    print("\n2. List Pipelines:")
    print("GET /pipelines")
    print("Response: {'pipelines': ['prediction_pipeline']}")
    
    print("\n3. Execute Pipeline:")
    print("POST /execute")
    print("Request Body:")
    request_body = {
        "pipeline_name": "prediction_pipeline",
        "input": "Predict the impact of AI on job markets over the next 5 years"
    }
    print(json.dumps(request_body, indent=2))
    
    print("\nResponse Body:")
    response_body = {
        "success": True,
        "data": {
            "executive_summary": "AI will significantly transform job markets...",
            "prediction": "By 2029, AI will create 2.3 million new jobs...",
            "confidence_level": 92,
            "evidence_base": ["McKinsey study...", "WEF report..."],
            "confidence_rationale": "High confidence based on multiple sources...",
            "risks": ["Geographic distribution...", "Skills gap..."],
            "methodology": "Systematic analysis of 15+ studies...",
            "sources": ["McKinsey", "WEF", "MIT", "OECD"],
            "limitations": "Predictions based on current trends..."
        },
        "execution_time": 45.2,
        "stage_results": [
            {
                "agent_name": "context_generator",
                "success": True,
                "execution_time": 8.1
            },
            {
                "agent_name": "deep_researcher", 
                "success": True,
                "execution_time": 32.4
            },
            {
                "agent_name": "formatter",
                "success": True,
                "execution_time": 4.7
            }
        ]
    }
    print(json.dumps(response_body, indent=2))

if __name__ == "__main__":
    demo_pipeline_structure()
    demo_expected_flow()
    demo_api_endpoints()
    
    print("\n" + "="*80)
    print("DEMO COMPLETE")
    print("="*80)
    print("\nTo run the actual pipeline:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start server: python main.py")
    print("3. Test API: curl -X POST http://localhost:8000/execute -H 'Content-Type: application/json' -d '{\"pipeline_name\": \"prediction_pipeline\", \"input\": \"Your question here\"}'")
    print("\nTo test pipeline directly:")
    print("python test_pipeline_simple.py")
