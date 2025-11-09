"""
Test that all stubs import correctly.
"""

def test_import_main_module():
    """Test main module import."""
    import diffusion_oxidation
    assert diffusion_oxidation.__version__ == "1.0.0"


def test_import_config():
    """Test config import."""
    from config import config
    assert config is not None


def test_import_schemas():
    """Test schemas import."""
    from data.schemas import DiffusionRecipe, OxidationRecipe
    assert DiffusionRecipe is not None
    assert OxidationRecipe is not None


def test_import_core_stubs():
    """Test core module stubs import."""
    from core import erfc, fick_fd, deal_grove, massoud, segregation
    # All should import without error


def test_import_spc_stubs():
    """Test SPC module stubs import."""
    from spc import rules, ewma, cusum, changepoint
    # All should import without error


def test_import_ml_stubs():
    """Test ML module stubs import."""
    from ml import features, vm, forecast, calibrate
    # All should import without error


def test_import_io_stubs():
    """Test IO module stubs import."""
    from io import loaders, writers
    # All should import without error


def test_import_api():
    """Test API router import."""
    from api import router
    assert router is not None
