"""
Configuration management for Diffusion & Oxidation module.

Uses Pydantic BaseSettings for environment-based configuration with
validation and type safety.
"""

from typing import Optional, Literal
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
import os


class DopantConstants(BaseSettings):
    """Physical constants for common dopants in silicon."""
    
    model_config = SettingsConfigDict(
        env_prefix="DOPANT_",
        case_sensitive=False,
    )
    
    # Diffusion coefficients (cm²/s) and activation energies (eV)
    # Format: {dopant: (D0, Ea)}
    # Sources: ITRS 2009, Fair & Tsai compilation
    
    BORON_D0: float = 0.76  # cm²/s
    BORON_EA: float = 3.46  # eV
    
    PHOSPHORUS_D0: float = 3.85
    PHOSPHORUS_EA: float = 3.66
    
    ARSENIC_D0: float = 0.066
    ARSENIC_EA: float = 3.44
    
    ANTIMONY_D0: float = 0.214
    ANTIMONY_EA: float = 3.65
    
    # Segregation coefficients (k = C_oxide / C_silicon)
    BORON_K: float = 0.3
    PHOSPHORUS_K: float = 0.1
    ARSENIC_K: float = 0.02
    ANTIMONY_K: float = 0.01
    
    def get_diffusion_params(self, dopant: str) -> tuple[float, float]:
        """Get (D0, Ea) for a dopant."""
        dopant = dopant.upper()
        d0 = getattr(self, f"{dopant}_D0", None)
        ea = getattr(self, f"{dopant}_EA", None)
        
        if d0 is None or ea is None:
            raise ValueError(f"Unknown dopant: {dopant}")
        
        return (d0, ea)
    
    def get_segregation_coeff(self, dopant: str) -> float:
        """Get segregation coefficient k for a dopant."""
        dopant = dopant.upper()
        k = getattr(self, f"{dopant}_K", None)
        
        if k is None:
            raise ValueError(f"Unknown dopant: {dopant}")
        
        return k


class OxidationConstants(BaseSettings):
    """Constants for Deal-Grove thermal oxidation model."""
    
    model_config = SettingsConfigDict(
        env_prefix="OXIDATION_",
        case_sensitive=False,
    )
    
    # Deal-Grove B/A and B parameters
    # These are temperature-dependent; default values at 1000°C
    
    DRY_B_DEFAULT: float = 0.0117  # μm²/hr at 1000°C
    DRY_B_A_DEFAULT: float = 0.0274  # μm/hr at 1000°C
    
    WET_B_DEFAULT: float = 0.287  # μm²/hr at 1000°C
    WET_B_A_DEFAULT: float = 0.202  # μm/hr at 1000°C
    
    # Massoud thin-oxide parameters
    MASSOUD_L: float = 7.0  # nm, characteristic length
    MASSOUD_TAU: float = 10.0  # minutes, characteristic time


class PathConfig(BaseSettings):
    """Path configuration for data, artifacts, and outputs."""
    
    model_config = SettingsConfigDict(
        env_prefix="DIFFUSION_PATH_",
        case_sensitive=False,
    )
    
    # Base directories
    DATA_DIR: Path = Field(default=Path("data/diffusion_oxidation"))
    ARTIFACTS_DIR: Path = Field(default=Path("artifacts/diffusion_oxidation"))
    LOGS_DIR: Path = Field(default=Path("logs/diffusion_oxidation"))
    
    # Subdirectories
    VALIDATION_DATA_DIR: Path = Field(default=Path("data/diffusion_oxidation/validation"))
    TEST_DATA_DIR: Path = Field(default=Path("data/diffusion_oxidation/test"))
    FIXTURES_DIR: Path = Field(default=Path("data/diffusion_oxidation/fixtures"))
    
    # Artifacts subdirectories
    VM_MODELS_DIR: Path = Field(default=Path("artifacts/diffusion_oxidation/vm"))
    CALIBRATION_DIR: Path = Field(default=Path("artifacts/diffusion_oxidation/calibration"))
    SPC_DIR: Path = Field(default=Path("artifacts/diffusion_oxidation/spc"))
    
    @field_validator("*", mode="before")
    @classmethod
    def expand_path(cls, v):
        """Expand paths relative to project root."""
        if isinstance(v, (str, Path)):
            path = Path(v)
            if not path.is_absolute():
                # Get project root (assume we're in services/analysis/app/methods/)
                project_root = Path(__file__).parent.parent.parent.parent.parent
                path = project_root / path
            return path
        return v
    
    def ensure_dirs(self):
        """Create all directories if they don't exist."""
        for field_name in self.model_fields:
            path = getattr(self, field_name)
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)


class ComputeConfig(BaseSettings):
    """Computation and performance settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="DIFFUSION_COMPUTE_",
        case_sensitive=False,
    )
    
    # Numerical solver settings
    USE_NUMBA: bool = Field(default=True, description="Enable numba JIT compilation")
    MAX_GRID_POINTS: int = Field(default=10000, description="Maximum FD grid points")
    DEFAULT_DX: float = Field(default=0.1, description="Default spatial step (nm)")
    DEFAULT_DT: float = Field(default=0.1, description="Default time step (s)")
    
    # Parallelization
    N_JOBS: int = Field(default=-1, description="Number of parallel jobs (-1 = all cores)")
    BATCH_SIZE: int = Field(default=100, description="Batch size for parallel processing")
    
    # Tolerances
    CONVERGENCE_TOL: float = Field(default=1e-6, description="Convergence tolerance")
    MAX_ITERATIONS: int = Field(default=10000, description="Maximum solver iterations")
    
    # Memory limits
    MAX_MEMORY_MB: int = Field(default=8192, description="Maximum memory usage (MB)")


class MLConfig(BaseSettings):
    """Machine Learning and Virtual Metrology settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="DIFFUSION_ML_",
        case_sensitive=False,
    )
    
    # Model selection
    VM_MODEL_TYPE: Literal["ridge", "lasso", "xgboost", "rf"] = "xgboost"
    
    # Training parameters
    CV_FOLDS: int = 5
    RANDOM_STATE: int = 42
    TEST_SIZE: float = 0.2
    
    # Feature engineering
    USE_FDC_FEATURES: bool = True
    USE_RECIPE_FEATURES: bool = True
    USE_TOOL_HISTORY: bool = True
    LOOKBACK_RUNS: int = 10
    
    # Model versioning
    MODEL_VERSION_FORMAT: str = "v{major}.{minor}.{patch}"
    AUTO_VERSION_INCREMENT: bool = True
    
    # ONNX export
    EXPORT_ONNX: bool = True
    ONNX_OPSET_VERSION: int = 15


class SPCConfig(BaseSettings):
    """Statistical Process Control settings."""
    
    model_config = SettingsConfigDict(
        env_prefix="DIFFUSION_SPC_",
        case_sensitive=False,
    )
    
    # Control chart parameters
    EWMA_LAMBDA: float = Field(default=0.2, ge=0, le=1)
    CUSUM_K: float = Field(default=0.5, description="CUSUM reference value")
    CUSUM_H: float = Field(default=5.0, description="CUSUM decision interval")
    
    # Rule checking
    ENABLE_WESTERN_ELECTRIC: bool = True
    ENABLE_NELSON: bool = True
    
    # Change-point detection
    BOCPD_HAZARD: float = Field(default=0.01, description="BOCPD hazard rate")
    BOCPD_MIN_SEPARATION: int = Field(default=5, description="Minimum run separation")
    
    # Alerting
    ALERT_THRESHOLD: Literal["warning", "critical"] = "warning"
    AUTO_TRIAGE: bool = True


class EnvironmentSettings(BaseSettings):
    """Overall environment settings."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )
    
    # Environment
    ENV: Literal["development", "staging", "production"] = "development"
    DEBUG: bool = Field(default=True)
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    
    # Database (inherited from main platform)
    DATABASE_URL: Optional[str] = None
    
    # Object storage (inherited from main platform)
    S3_ENDPOINT: Optional[str] = None
    S3_BUCKET: str = "semiconductorlab-diffusion"
    
    # API settings
    API_V1_PREFIX: str = "/api/v1"
    ENABLE_DOCS: bool = True
    
    @property
    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENV == "production"
    
    @property
    def is_development(self) -> bool:
        """Check if running in development."""
        return self.ENV == "development"


class Config:
    """Main configuration class combining all settings."""
    
    def __init__(self):
        self.dopant = DopantConstants()
        self.oxidation = OxidationConstants()
        self.paths = PathConfig()
        self.compute = ComputeConfig()
        self.ml = MLConfig()
        self.spc = SPCConfig()
        self.env = EnvironmentSettings()
    
    def initialize(self):
        """Initialize configuration (create directories, validate settings)."""
        # Create all necessary directories
        self.paths.ensure_dirs()
        
        # Validate numba availability if requested
        if self.compute.USE_NUMBA:
            try:
                import numba
                print(f"✓ Numba {numba.__version__} available")
            except ImportError:
                print("⚠ Numba requested but not installed, disabling")
                self.compute.USE_NUMBA = False
        
        # Log configuration
        print(f"✓ Diffusion & Oxidation module configured")
        print(f"  Environment: {self.env.ENV}")
        print(f"  Debug mode: {self.env.DEBUG}")
        print(f"  Data directory: {self.paths.DATA_DIR}")
        print(f"  Artifacts directory: {self.paths.ARTIFACTS_DIR}")
    
    def to_dict(self) -> dict:
        """Export configuration as dictionary."""
        return {
            "dopant": self.dopant.model_dump(),
            "oxidation": self.oxidation.model_dump(),
            "paths": {k: str(v) for k, v in self.paths.model_dump().items()},
            "compute": self.compute.model_dump(),
            "ml": self.ml.model_dump(),
            "spc": self.spc.model_dump(),
            "env": self.env.model_dump(),
        }


# Global configuration instance
config = Config()

# Convenience accessors
dopant_constants = config.dopant
oxidation_constants = config.oxidation
paths = config.paths
compute_config = config.compute
ml_config = config.ml
spc_config = config.spc
env_settings = config.env


if __name__ == "__main__":
    # Test configuration
    config.initialize()
    
    # Print sample values
    print("\n=== Dopant Constants ===")
    print(f"Boron D0: {config.dopant.BORON_D0} cm²/s")
    print(f"Boron Ea: {config.dopant.BORON_EA} eV")
    print(f"Boron k: {config.dopant.BORON_K}")
    
    print("\n=== Oxidation Constants ===")
    print(f"Dry B: {config.oxidation.DRY_B_DEFAULT} μm²/hr")
    print(f"Wet B: {config.oxidation.WET_B_DEFAULT} μm²/hr")
    
    print("\n=== Paths ===")
    print(f"Data: {config.paths.DATA_DIR}")
    print(f"Artifacts: {config.paths.ARTIFACTS_DIR}")
    
    print("\n=== Compute ===")
    print(f"Use Numba: {config.compute.USE_NUMBA}")
    print(f"Max grid points: {config.compute.MAX_GRID_POINTS}")
    
    print("\n=== Full Config ===")
    import json
    print(json.dumps(config.to_dict(), indent=2, default=str))
