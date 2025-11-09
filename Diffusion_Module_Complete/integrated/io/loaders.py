"""
Data loaders for MES/FDC/SPC formats.

Will be implemented in Session 6.
"""

from typing import List
from pathlib import Path
from ..data.schemas import DiffusionRun, FurnaceFDCRecord, SPCPoint


class MESLoader:
    """Load data from MES exports."""
    def __init__(self):
        raise NotImplementedError("Session 6: MES loader")


class FDCLoader:
    """Load FDC data from furnace systems."""
    def __init__(self):
        raise NotImplementedError("Session 6: FDC loader")


class SPCLoader:
    """Load SPC chart data."""
    def __init__(self):
        raise NotImplementedError("Session 6: SPC loader")


__all__ = ["MESLoader", "FDCLoader", "SPCLoader"]
