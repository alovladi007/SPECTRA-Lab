"""
Data writers for standardized exports.

Will be implemented in Session 6.
"""

from typing import List
from pathlib import Path
from ..data.schemas import DiffusionRun


class StandardWriter:
    """Write data in standardized format."""
    def __init__(self):
        raise NotImplementedError("Session 6: Standard writer")


__all__ = ["StandardWriter"]
