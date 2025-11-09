"""
Pytest configuration and fixtures.
"""

import pytest
import sys
from pathlib import Path

# Add module to path
module_dir = Path(__file__).parent.parent
sys.path.insert(0, str(module_dir))


@pytest.fixture
def config():
    """Provide configuration instance."""
    from config import config
    config.initialize()
    return config


@pytest.fixture
def sample_diffusion_recipe():
    """Provide sample diffusion recipe."""
    from data.schemas import DiffusionRecipe, DopantType
    
    return DiffusionRecipe(
        name="Test Boron Diffusion",
        dopant=DopantType.BORON,
        temperature=1000.0,
        time=30.0,
        source_type="constant",
        surface_concentration=1e20,
        background_concentration=1e15
    )


@pytest.fixture
def sample_oxidation_recipe():
    """Provide sample oxidation recipe."""
    from data.schemas import OxidationRecipe, OxidationAmbient
    
    return OxidationRecipe(
        name="Test Dry Oxidation",
        temperature=1000.0,
        time=60.0,
        ambient=OxidationAmbient.DRY,
        pressure=1.0
    )
