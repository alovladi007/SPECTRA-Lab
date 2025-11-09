"""
Simulation Configuration Module

This file will contain configuration settings for diffusion and oxidation simulations.
It will be populated with the actual config.py from the Diffusion Module Skeleton Package.

Expected configuration areas:
- Diffusion simulation parameters (temperature, time, dopant properties)
- Oxidation simulation parameters (temperature, pressure, ambient)
- Numerical solver settings (grid spacing, convergence criteria)
- Output file paths
- Default material properties (silicon, dopants, oxides)
"""

from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
import os

# Placeholder configuration structure
class SimulationConfig(BaseModel):
    """Base configuration for simulations"""
    output_dir: str = Field(
        default="data/simulation",
        description="Directory for simulation outputs"
    )

    class Config:
        arbitrary_types_allowed = True


class DiffusionConfig(SimulationConfig):
    """Configuration for diffusion simulations"""
    temperature: float = Field(default=1000.0, description="Temperature in Celsius")
    time: float = Field(default=60.0, description="Diffusion time in minutes")
    grid_points: int = Field(default=100, description="Number of spatial grid points")

    class Config:
        arbitrary_types_allowed = True


class OxidationConfig(SimulationConfig):
    """Configuration for oxidation simulations"""
    temperature: float = Field(default=1000.0, description="Temperature in Celsius")
    time: float = Field(default=60.0, description="Oxidation time in minutes")
    ambient: str = Field(default="dry", description="Ambient type: dry or wet")

    class Config:
        arbitrary_types_allowed = True


# Global configuration instance (will be initialized when actual config is integrated)
_config: Optional[Dict[str, Any]] = None


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load simulation configuration from file

    This function will be implemented with actual configuration loading logic
    from the Diffusion Module Skeleton Package.
    """
    global _config

    if _config is None:
        # Default configuration
        _config = {
            "diffusion": DiffusionConfig().dict(),
            "oxidation": OxidationConfig().dict(),
            "output_dir": "data/simulation"
        }

    return _config


def get_config() -> Dict[str, Any]:
    """Get current configuration"""
    if _config is None:
        return load_config()
    return _config


def set_config(config: Dict[str, Any]) -> None:
    """Set configuration"""
    global _config
    _config = config


# Default export
__all__ = [
    "SimulationConfig",
    "DiffusionConfig",
    "OxidationConfig",
    "load_config",
    "get_config",
    "set_config"
]
