"""Session 6: IO & Schemas for MES/SPC/FDC Ingestion - PRODUCTION READY."""

__version__ = "6.0.0"

# Export main components
from .data import *
from .ingestion import *

__all__ = ["data", "ingestion"]
