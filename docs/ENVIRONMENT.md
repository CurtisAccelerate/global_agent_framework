# Environment Configuration

Copy these settings into a local-only `.env` file (or your preferred secrets manager) before running the framework. **Do not commit real secrets.**

```
# Required keys
OPENAI_API_KEY=                    # OpenAI API key for GPT-5 reasoning agents

# Optional service integrations
ODDS_API_KEY=                      # The Odds API for sports betting data (free: 500 req/month)
SERPER_API_KEY=                    # Serper.dev for web search (free: 2,500 searches/month)
SERP_DEV_API_KEY=                  # Alternative to Serper.dev
SERP_DEV_BASE_URL=https://api.serp.dev
SERP_DEV_USE_STUB=false
SERP_DEV_VERIFY_SSL=true

# API server configuration
HOST=0.0.0.0
PORT=8000
DEBUG=false
LOG_VERBOSITY=info
RESPONSES_TIMEOUT_SECONDS=600
FILE_LOGGING_ENABLED=true

# Pipeline defaults
DEFAULT_PIPELINE_NAME=futuerex

# Optional HTTP authentication (leave blank to disable)
SERVER_AUTH_ENABLED=false
SERVER_API_KEY=
```

> Tip: Keep the `.env` file untracked (already covered by `.gitignore`) and share placeholder examples only.

