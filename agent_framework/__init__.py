"""
Lightweight shim package to provide stable imports (minimal-change approach).

New code can import from `agent_framework.*` while existing files remain unchanged.
"""

from .pipeline import *  # re-export
from .agent import *     # re-export
from .config import *    # re-export
from .api import *       # re-export
from .agent_pipeline_declarations import *  # re-export

__all__ = []  # populated implicitly by star imports


