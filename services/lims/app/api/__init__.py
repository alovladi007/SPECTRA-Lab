"""
LIMS API routes.
"""

from .samples import router as samples_router
from .recipes import router as recipes_router
from .sops import router as sops_router
from .eln import router as eln_router

__all__ = ["samples_router", "recipes_router", "sops_router", "eln_router"]
