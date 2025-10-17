"""
Logging configuration for real-time debugging
"""
import logging
import os
from datetime import datetime
from typing import Optional, Tuple
try:
    # Optional import; do not hard-require Config during early boot
    from config import Config  # type: ignore
except Exception:
    class _FallbackConfig:  # minimal fallback for type-hints
        DEBUG = False
        LOG_VERBOSITY = "info"
        ERRORS_ONLY = False
    Config = _FallbackConfig()  # type: ignore

def setup_logging(debug=True, auto_clear=True, log_verbosity: Optional[str] = None, errors_only: Optional[bool] = None):
    """Setup logging with real-time flushing and verbosity controls.
    Always clears all files in logs directory on startup.
    log_verbosity: one of {"verbose", "info", "minimal", "error"} (defaults to Config)
    errors_only: if True, only ERROR+ are emitted to console and pipeline.log (errors.log always active)
    """
    
    # Determine if file logging is enabled
    file_logging_enabled = bool(getattr(Config, 'FILE_LOGGING_ENABLED', True))

    # Create/clear logs directory if file logging is enabled
    if file_logging_enabled:
        os.makedirs('logs', exist_ok=True)
        
        # Single log filenames (no timestamp to avoid flooding)
        pipeline_log_path = os.path.join('logs', 'pipeline.log')
        errors_log_path = os.path.join('logs', 'errors.log')
        stage_jsonl_path = os.path.join('logs', 'stage_responses.jsonl')
        md_trace_path = os.path.join('logs', 'pipeline_trace.md')
        serper_log_path = os.path.join('logs', 'serper.log')
        
        # Optionally clear all files in logs directory on startup
        if auto_clear:
            try:
                import glob
                log_files = glob.glob(os.path.join('logs', '*'))
                for log_file in log_files:
                    if os.path.isfile(log_file):
                        os.remove(log_file)
            except Exception:
                pass
    else:
        pipeline_log_path = errors_log_path = stage_jsonl_path = md_trace_path = serper_log_path = None
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Create formatters
    detailed_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    # Console handler with immediate flush
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(simple_formatter)
    console_handler.flush = lambda: None  # Ensure immediate flush
    
    # Flushing file handlers
    class FlushingFileHandler(logging.FileHandler):
        def emit(self, record):
            super().emit(record)
            try:
                self.flush()
            except Exception:
                pass

    # File handlers (optional)
    file_handler = None
    error_handler = None
    serper_handler = None
    if file_logging_enabled and pipeline_log_path and errors_log_path:
        file_handler = FlushingFileHandler(pipeline_log_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(detailed_formatter)
        
        error_handler = FlushingFileHandler(errors_log_path, mode='a', encoding='utf-8')
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(detailed_formatter)

        if serper_log_path:
            serper_handler = FlushingFileHandler(serper_log_path, mode='a', encoding='utf-8')
            serper_handler.setLevel(logging.DEBUG)
            serper_handler.setFormatter(detailed_formatter)
    
    class SpecificNamespaceFilter(logging.Filter):
        def __init__(self, prefix: str):
            super().__init__()
            self.prefix = prefix
        def filter(self, record: logging.LogRecord) -> bool:
            name = record.name or ''
            return name == self.prefix or name.startswith(self.prefix + '.')

    # Add handlers
    root_logger.addHandler(console_handler)
    if file_handler:
        root_logger.addHandler(file_handler)
    if error_handler:
        root_logger.addHandler(error_handler)
    if serper_handler:
        serper_handler.addFilter(SpecificNamespaceFilter('tool.serper'))
        root_logger.addHandler(serper_handler)
    
    # Verbosity configuration
    effective_verbosity = (log_verbosity or getattr(Config, 'LOG_VERBOSITY', 'info')).lower()
    effective_errors_only = errors_only if errors_only is not None else bool(getattr(Config, 'ERRORS_ONLY', False))

    class AllowedNamespaceFilter(logging.Filter):
        def __init__(self, prefixes: Tuple[str, ...]):
            super().__init__()
            self.prefixes = prefixes
        def filter(self, record: logging.LogRecord) -> bool:
            name = record.name or ''
            return any(name == p or name.startswith(p + '.') for p in self.prefixes)

    # Reduce third-party noise by default
    third_party_targets = [
        'openai', 'httpcore', 'httpx', 'urllib3', 'anyio', 'asyncio', 'uvicorn', 'starlette', 'fastapi'
    ]

    def set_lib_levels(level: int) -> None:
        for n in third_party_targets:
            try:
                logging.getLogger(n).setLevel(level)
            except Exception:
                pass

    if effective_errors_only or effective_verbosity == 'error':
        console_level = logging.ERROR
        file_level = logging.ERROR
        root_logger.setLevel(logging.ERROR)
        set_lib_levels(logging.ERROR)
    elif effective_verbosity == 'minimal':
        console_level = logging.INFO
        file_level = logging.INFO
        root_logger.setLevel(logging.INFO)
        set_lib_levels(logging.WARNING)
        # Only allow our key namespaces through on console and file
        allowed = AllowedNamespaceFilter(('pipeline', 'agent', '__main__'))
        console_handler.addFilter(allowed)
        if file_handler:
            file_handler.addFilter(allowed)
    elif effective_verbosity == 'verbose':
        console_level = logging.INFO if not debug else logging.DEBUG
        file_level = logging.DEBUG
        root_logger.setLevel(logging.DEBUG)
        set_lib_levels(logging.INFO)
    else:  # 'info'
        console_level = logging.INFO
        file_level = logging.INFO if not debug else logging.DEBUG
        root_logger.setLevel(logging.DEBUG if debug else logging.INFO)
        set_lib_levels(logging.WARNING)

    console_handler.setLevel(console_level)
    if file_handler:
        file_handler.setLevel(file_level)
    if serper_handler:
        serper_handler.setLevel(logging.DEBUG)
    
    # Create a custom handler that flushes immediately
    class ImmediateFlushHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()

    # Replace console handler with immediate flush version while preserving filters/formatters/levels
    root_logger.removeHandler(console_handler)
    immediate_handler = ImmediateFlushHandler()
    immediate_handler.setLevel(console_level)
    for f in console_handler.filters:
        immediate_handler.addFilter(f)
    immediate_handler.setFormatter(simple_formatter)
    root_logger.addHandler(immediate_handler)
    
    return root_logger
