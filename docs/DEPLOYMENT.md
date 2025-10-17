## Deployment Guide

### Prerequisites
- Python 3.10+
- Install dependencies: `pip install -r requirements.txt`
- Copy `.env.example` to `.env` and populate secrets (`OPENAI_API_KEY`, etc.)
- For Cloud Run: Google Cloud CLI (`gcloud`) installed and authenticated
- Set `SERP_DEV_USE_STUB=true` to disable real serp.dev calls during local testing.

### Run locally (API server)

PowerShell:
```powershell
$env:OPENAI_API_KEY="<your_openai_api_key>"
$env:SERVER_API_KEY="test-key"
python main.py
```

The server starts at `http://localhost:8000`.

### Run locally against test_inputs.json (direct pipeline)

PowerShell (from repo root):
```powershell
python test_pipeline_simple.py
```

Optionally specify a custom inputs path:
```powershell
$env:TEST_INPUTS_PATH="C:\\path\\to\\agent_framework\\test_inputs.json"
python test_pipeline_simple.py
```

### Minimal local API request

```powershell
$env:SERVER_API_KEY="test-key"
# In another terminal after `python main.py` is running:
curl -X POST "http://localhost:8000/v1/chat/completions" ^
  -H "Authorization: Bearer $env:SERVER_API_KEY" ^
  -H "Content-Type: application/json" ^
  -d '{
    "model": "gpt-5",
    "messages": [{"role": "user", "content": "Hello"}],
    "pipeline_name": "prediction_pipeline"
  }'
```

### Deploy to Cloud Run

Ensure you are in the project directory and authenticated:
```powershell
gcloud auth login
gcloud config set project <YOUR_PROJECT_ID>
```

Deploy (provided command):
```powershell
gcloud run deploy agent-endpoint --source . --region us-east1 --allow-unauthenticated --timeout 1200
```

Notes:
- Use `SERP_DEV_USE_STUB=true` in non-production environments to force offline evidence gathering.
- If using CPU during requests only, ensure timeouts are sufficient for longer research stages.



