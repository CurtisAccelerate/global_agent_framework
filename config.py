import os
from dotenv import load_dotenv
from typing import Optional

load_dotenv()


class Config:
    """Configuration management for the agent framework"""

    # Core API keys (must be supplied via environment)
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    ODDS_API_KEY: Optional[str] = os.getenv("ODDS_API_KEY")
    SERPER_API_KEY: Optional[str] = os.getenv("SERPER_API_KEY")
    SERP_DEV_API_KEY: Optional[str] = os.getenv("SERP_DEV_API_KEY")
    SERP_DEV_BASE_URL: str = os.getenv("SERP_DEV_BASE_URL", "https://api.serp.dev")
    SERPER_SEARCH_URL: str = os.getenv("SERPER_SEARCH_URL", "https://google.serper.dev/search")
    SERPER_WEB_URL: str = os.getenv("SERPER_WEB_URL", "https://scrape.serper.dev")
    # Back-compat: allow old env flags to toggle stub
    SERPER_USE_STUB: bool = os.getenv("SERPER_USE_STUB", os.getenv("SERP_DEV_USE_STUB", "false")).lower() in ("1", "true", "yes", "on")
    SERPER_VERIFY_SSL: bool = os.getenv("SERPER_VERIFY_SSL", os.getenv("SERP_DEV_VERIFY_SSL", "true")).lower() in ("1", "true", "yes", "on")
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "false").lower() in ("1", "true", "yes", "on")
    # Logging verbosity: one of {"verbose", "info", "minimal", "error"}
    LOG_VERBOSITY: str = os.getenv("LOG_VERBOSITY", "info").lower()
    # If true, force error-only logging for console and pipeline.log (errors.log always kept)
    ERRORS_ONLY: bool = os.getenv("ERRORS_ONLY", "false").lower() in ("1", "true", "yes", "on")
    # Number of characters to include in Markdown input/output previews
    TRACE_PREVIEW_CHARS: int = int(os.getenv("TRACE_PREVIEW_CHARS", "512"))
    RESPONSES_TIMEOUT_SECONDS: int = int(os.getenv("RESPONSES_TIMEOUT_SECONDS", "600"))
    # Stage 1 drafting controls (upper bound of 3 as per product spec)
    try:
        _stage1_draft_env = int(os.getenv("STAGE1_DRAFT_COUNT", "1"))
    except ValueError:
        _stage1_draft_env = 1
    STAGE1_DRAFT_COUNT: int = max(1, min(_stage1_draft_env, 3))
    # Control file-based detailed logging and pipeline trace writing.
    # Defaults to enabled unless explicitly overridden via env var.
    _fle = os.getenv("FILE_LOGGING_ENABLED")
    if _fle is None:
        FILE_LOGGING_ENABLED: bool = True
    else:
        FILE_LOGGING_ENABLED: bool = _fle.lower() in ("1", "true", "yes", "on")
    # Minimal server API key for OpenAI-compatible Bearer auth
    SERVER_API_KEY: Optional[str] = os.getenv("SERVER_API_KEY")
    SERVER_AUTH_ENABLED: bool = os.getenv("SERVER_AUTH_ENABLED", "false").lower() in ("1", "true", "yes", "on")
    # Default pipeline alias exposed via API endpoints
    DEFAULT_PIPELINE_NAME: str = os.getenv("DEFAULT_PIPELINE_NAME", "futuerex")
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration is present"""
        if not cls.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required")
        if cls.SERVER_AUTH_ENABLED and not cls.SERVER_API_KEY:
            raise ValueError("SERVER_API_KEY is required when SERVER_AUTH_ENABLED is true")
        # Normalize LOG_VERBOSITY
        allowed = {"verbose", "info", "minimal", "error"}
        if cls.LOG_VERBOSITY not in allowed:
            cls.LOG_VERBOSITY = "info"
        return True
