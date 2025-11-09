"""
LIMS API routes.
"""

# Only import auth router for now - others require database setup
from .auth import router as auth_router

# Database-dependent routers temporarily disabled:
# from .samples import router as samples_router
# from .recipes import router as recipes_router
# from .sops import router as sops_router
# from .eln import router as eln_router

__all__ = ["auth_router"]
