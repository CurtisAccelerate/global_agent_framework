"""
Local tool implementations for the agent framework.

Each tool should expose a callable function that accepts only JSON-serializable
keyword arguments and returns a JSON-serializable value or a string. Tools are
registered into agents via `ToolDefinition(function=callable, ...)`.
"""

__all__ = [
    # Polymarket Gamma tools
    "polymarket_gamma_get_odds",
    # Deribit tools
    "deribit_weekly_snapshot",
    "deribit_weekly_ladder",
    "deribit_weekly_inputs",
    # Odds API tools
    "odds_find",
    "odds_get",
    # serper.dev tools
    "serper_search",
    "serp_dev_search_stub",
]

try:
    from .polymarket_gamma import polymarket_gamma_get_odds  # noqa: F401
except Exception:
    # Allow environments without optional deps to import the package
    pass


try:
    from .deribit import deribit_weekly_snapshot, deribit_weekly_ladder, deribit_weekly_inputs  # noqa: F401
except Exception:
    pass

try:
    from .odds_api import odds_find, odds_get  # noqa: F401
except Exception:
    pass

try:
    from .serper import serper_search  # noqa: F401
except Exception:
    pass

try:
    from .serp_dev_stub import serp_dev_search_stub  # noqa: F401
except Exception:
    pass

