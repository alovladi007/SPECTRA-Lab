# scripts/dev/generate_test_data.py

"""
Test Data Generators and Factory Patterns

Generates realistic synthetic data for all characterization methods:
- Electrical: I-V, C-V, Hall, 4PP, DLTS
- Optical: UV-Vis-NIR, FTIR, Ellipsometry, PL, Raman
- Structural: XRD, SEM/TEM, AFM
- Chemical: XPS, SIMS, RBS

Each generator produces:
- Raw measurement data
- Derived metrics/results
- Metadata with provenance
- Realistic noise and artifacts
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import uuid
from pathlib import Path
import json

# ============================================================================
# Base Generator
# ============================================================================

@dataclass
class GeneratorConfig:
    """Configuration for test data generation"""
    add_noise: bool = True
    noise_level: float = 0.01  # 1% relative noise
    add_outliers: bool = False
    outlier_fraction: float = 0.02
    add_drift: bool = False
    drift_rate: float = 1e-6
    seed: Optional[int] = None

class BaseGenerator:
    """Base class for all data generators"""

    def __init__(self, config: Optional[GeneratorConfig] = None):
        self.config = config or GeneratorConfig()
        if self.config.seed is not None:
            np.random.seed(self.config.seed)

    def add_noise(self, data: np.ndarray, relative: bool = True) -> np.ndarray:
        """Add Gaussian noise to data"""
        if not self.config.add_noise:
            return data

        if relative:
            noise = np.random.normal(0, self.config.noise_level, data.shape) * np.abs(data)
        else:
            noise = np.random.normal(0, self.config.noise_level, data.shape)

        return data + noise

    def add_outliers(self, data: np.ndarray) -> np.ndarray:
        """Add random outliers to data"""
        if not self.config.add_outliers:
            return data

        n_outliers = int(len(data) * self.config.outlier_fraction)
        outlier_indices = np.random.choice(len(data), n_outliers, replace=False)
        outliers = np.random.uniform(0.5, 2.0, n_outliers) * data[outlier_indices]
        data[outlier_indices] = outliers

        return data

    def generate_metadata(self, method: str, **kwargs) -> Dict[str, Any]:
        """Generate standard metadata"""
        return {
            "method": method,
            "generated_at": datetime.utcnow().isoformat(),
            "generator_version": "1.0.0",
            "config": {
                "noise_level": self.config.noise_level,
                "has_outliers": self.config.add_outliers,
                "has_drift": self.config.add_drift,
            },
            **kwargs
        }

# [Continued with all generator classes...]

if __name__ == "__main__":
    config = GeneratorConfig(
        add_noise=True,
        noise_level=0.02,
        add_outliers=False,
        seed=42
    )

    generator = TestDataGenerator(output_dir="data/test_data", config=config)
    files = generator.generate_all()

    print("\n" + "="*80)
    print("Test Data Generation Complete!")
    print("="*80)
    print(f"\nGenerated files:")
    for name, path in files.items():
        print(f"  {name:20s} -> {path}")
    print(f"\nManifest: data/test_data/manifest.json")
