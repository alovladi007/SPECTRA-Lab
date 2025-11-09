"""
Tests for data schemas.
"""

import pytest
from pydantic import ValidationError


def test_diffusion_recipe_valid(sample_diffusion_recipe):
    """Test valid diffusion recipe."""
    recipe = sample_diffusion_recipe
    assert recipe.name == "Test Boron Diffusion"
    assert recipe.temperature == 1000.0
    recipe.model_validate()  # Should not raise


def test_diffusion_recipe_invalid_temperature():
    """Test invalid temperature."""
    from data.schemas import DiffusionRecipe, DopantType
    
    with pytest.raises(ValidationError):
        DiffusionRecipe(
            name="Invalid",
            dopant=DopantType.BORON,
            temperature=2000.0,  # > 1400°C
            time=30.0,
            source_type="constant",
            surface_concentration=1e20
        )


def test_oxidation_recipe_valid(sample_oxidation_recipe):
    """Test valid oxidation recipe."""
    recipe = sample_oxidation_recipe
    assert recipe.name == "Test Dry Oxidation"
    assert recipe.temperature == 1000.0
    recipe.model_validate()  # Should not raise


def test_dopant_type_enum():
    """Test dopant type enum."""
    from data.schemas import DopantType
    
    assert "boron" in [d.value for d in DopantType]
    assert "phosphorus" in [d.value for d in DopantType]


def test_source_type_enum():
    """Test source type enum."""
    from data.schemas import SourceType
    
    assert "constant" in [s.value for s in SourceType]
    assert "limited" in [s.value for s in SourceType]
