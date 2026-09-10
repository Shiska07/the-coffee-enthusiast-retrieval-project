"""Web search package for external fallback lookups.

This package can handle Tavily or other web search integrations when
local knowledge is insufficient.
"""

from __future__ import annotations

__all__ = ["search_web"]


def search_web(*args, **kwargs):
    """Placeholder web-search entry point."""
    raise NotImplementedError("Web search logic has not been implemented yet.")
