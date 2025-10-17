# Testing Guide for Prediction Pipeline

## Overview
This guide demonstrates how to test both the prediction pipeline directly and the API endpoints.

## Pipeline Structure

### 3-Stage Prediction Pipeline
1. **Stage 1: Context Generator** - Low reasoning, surfaces unknowns and first principles
2. **Stage 2: Deep Researcher** - High reasoning, resolves unknowns with 95% confidence
3. **Stage 3: Formatter** - Medium reasoning, outputs structured JSON

### Configuration
- **Model**: GPT-5 for all stages
- **Temperature**: API default (1.0) for all stages
- **Verbosity**: High for stages 1-2, Low for stage 3
- **Text Format**: Text for stages 1-2, JSON object for stage 3
- **Tools**: Web search and file search for stage 2 only

## Test Results Summary

### Pipeline Direct Test
✅ **Structure Demo**: Successfully demonstrated pipeline configuration
✅ **Data Flow**: Showed expected inputs/outputs for each stage
✅ **API Endpoints**: Documented all available endpoints

### Expected Performance
- **Stage 1**: ~8 seconds (context generation)
- **Stage 2**: ~32 seconds (deep research with web search)
- **Stage 3**: ~5 seconds (JSON formatting)
- **Total**: ~45 seconds end-to-end

## Test Files Created

1. **`test_structure_demo.py`** - ✅ Working demo of pipeline structure
2. **`test_pipeline_simple.py`** - Pipeline direct execution test
3. **`test_pipeline_and_api.py`** - Comprehensive API testing
4. **`docs/testing.md`** - This guide

## Running Tests

### Prerequisites
```bash
# Install dependencies (may require admin privileges)
pip install -r requirements.txt

# Or install individually
pip install openai fastapi uvicorn pydantic python-dotenv requests
```

### Provide test inputs declaratively

Create a `test_inputs.json` file in the repo root:

```json
{
  "input": "You are an agent that can predict future events. The event to be predicted: \"Bayer Leverkusen vs. Eintracht Frankfurt (resolved around 2025-09-13 (GMT+8)).\nA. Bayer Leverkusen win on 2025-09-12\nB. Bayer Leverkusen vs. Eintracht Frankfurt end in a draw\nC. Eintracht Frankfurt win on 2025-09-12\"\nIMPORTANT: Your final answer MUST end with this exact format:\nlisting all plausible options you have identified, separated by commas, within the box. For example: \\boxed{A} for a single option or \\boxed{B, C, D} for multiple options.\nDo not use any other format. Do not refuse to make a prediction. Do not say \"I cannot predict the future.\" You must make a clear prediction based on the best data currently available, using the box format specified above."
}
```

Or a batch:

```json
{
  "inputs": [
    "Question A...",
    "Question B..."
  ]
}
```

Optionally set a custom path:

```bash
export TEST_INPUTS_PATH=/full/path/to/inputs.json
# Windows PowerShell
$env:TEST_INPUTS_PATH="C:\\path\\to\\inputs.json"
```

### Test Pipeline Directly
```bash
python test_pipeline_simple.py
```

### Test API Endpoints
```bash
# Start server
python main.py

# In another terminal, test endpoints
curl -X GET http://localhost:8000/health
curl -X GET http://localhost:8000/pipelines
curl -X POST http://localhost:8000/execute \
  -H "Content-Type: application/json" \
  -d '{"pipeline_name": "prediction_pipeline", "input": "Your question here"}'
```

The API tester also reads `test_inputs.json` if present (or `TEST_INPUTS_PATH`).

### Run Structure Demo
```bash
python test_structure_demo.py
```

## Expected API Responses

### Health Check
```json
{
  "status": "healthy",
  "pipelines": 1
}
```

### List Pipelines
```json
{
  "pipelines": ["prediction_pipeline"]
}
```

### Execute Pipeline
```json
{
  "success": true,
  "data": {
    "executive_summary": "...",
    "prediction": "...",
    "confidence_level": 92,
    "evidence_base": ["..."],
    "confidence_rationale": "...",
    "risks": ["..."],
    "methodology": "...",
    "sources": ["..."],
    "limitations": "..."
  },
  "execution_time": 45.2,
  "stage_results": [
    {
      "agent_name": "context_generator",
      "success": true,
      "execution_time": 8.1
    },
    {
      "agent_name": "deep_researcher",
      "success": true,
      "execution_time": 32.4
    },
    {
      "agent_name": "formatter",
      "success": true,
      "execution_time": 4.7
    }
  ]
}
```

## Debugging

### Enable Debug Logging
Set environment variable:
```bash
export DEBUG=true
# or on Windows
set DEBUG=true
```

### Log Files
- `pipeline_test_results.log` - Pipeline execution logs
- `test_results.json` - Detailed test results
- `test_results.log` - Comprehensive test logs

### Common Issues
1. **Module not found**: Install dependencies with `pip install -r requirements.txt`
2. **Permission errors**: Run as administrator or use virtual environment
3. **API connection refused**: Ensure server is running on port 8000
4. **OpenAI API errors**: Check API key in `config.py`

## Performance Notes

### Token Limits
- **Stage 1**: 8192 max output tokens
- **Stage 2**: 8192 max output tokens  
- **Stage 3**: 4096 max output tokens
- **Total**: Up to ~20K tokens for complete pipeline

### Reasoning Effort
- **Low**: Fast context generation (~8s)
- **High**: Deep research with tools (~32s)
- **Medium**: Efficient formatting (~5s)

### Verbosity Levels
- **High**: Detailed reasoning and evidence (stages 1-2)
- **Low**: Concise output (stage 3)

## Next Steps

1. **Install Dependencies**: Resolve permission issues with pip installation
2. **Test Pipeline**: Run `python test_pipeline_simple.py`
3. **Test API**: Start server and test endpoints
4. **Monitor Logs**: Check debug output for detailed execution info
5. **Customize**: Modify prompts or configurations as needed
