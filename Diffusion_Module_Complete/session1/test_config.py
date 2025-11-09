"""
Tests for configuration module.
"""

import pytest


def test_config_initialization(config):
    """Test that configuration initializes correctly."""
    assert config is not None
    assert config.env.ENV in ["development", "staging", "production"]


def test_dopant_constants(config):
    """Test dopant constants."""
    d0, ea = config.dopant.get_diffusion_params("boron")
    assert d0 > 0
    assert ea > 0
    
    k = config.dopant.get_segregation_coeff("boron")
    assert 0 < k < 1


def test_invalid_dopant():
    """Test that invalid dopant raises error."""
    from config import config
    
    with pytest.raises(ValueError):
        config.dopant.get_diffusion_params("invalid_dopant")


def test_oxidation_constants(config):
    """Test oxidation constants."""
    assert config.oxidation.DRY_B_DEFAULT > 0
    assert config.oxidation.WET_B_DEFAULT > 0
    assert config.oxidation.WET_B_DEFAULT > config.oxidation.DRY_B_DEFAULT


def test_paths_created(config):
    """Test that paths are created."""
    assert config.paths.DATA_DIR.exists()
    assert config.paths.ARTIFACTS_DIR.exists()
