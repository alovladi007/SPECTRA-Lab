"""
Stable API for Diffusion & Oxidation Module - SPECTRA Integration

This module provides clean, stable import points for all diffusion and oxidation
functionality. Use this as the primary interface for SPECTRA integration.

Example:
    from session11.spectra import diffusion_oxidation as do

    # Diffusion
    x, C = do.diffusion.erfc_profile(dopant="boron", temp_c=1000, time_min=30)

    # Oxidation
    thickness = do.oxidation.deal_grove_thickness(temp_c=1000, time_hr=2.0, ambient="dry")

    # SPC
    violations = do.spc.check_rules(data)

    # Virtual Metrology
    features = do.ml.extract_features(fdc_data)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add parent directories to path for imports
_module_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_module_root))


# ============================================================================
# DIFFUSION API
# ============================================================================

class DiffusionAPI:
    """Clean API for diffusion calculations."""

    @staticmethod
    def erfc_profile(
        dopant: str,
        temp_c: float,
        time_min: float,
        method: str = "constant_source",
        surface_conc: float = 1e20,
        dose: Optional[float] = None,
        background: float = 1e15,
        depth_nm: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate diffusion profile using ERFC analytical solution.

        Args:
            dopant: Dopant species ("boron", "phosphorus", "arsenic", "antimony")
            temp_c: Temperature (°C)
            time_min: Diffusion time (minutes)
            method: "constant_source" or "limited_source"
            surface_conc: Surface concentration for constant source (cm^-3)
            dose: Dose for limited source (cm^-2)
            background: Background doping (cm^-3)
            depth_nm: Depth points (nm), default 0-500nm

        Returns:
            (depth_nm, concentration): Arrays of depth and concentration
        """
        from session2.erfc import constant_source_profile, limited_source_profile

        if depth_nm is None:
            depth_nm = np.linspace(0, 500, 100)

        time_sec = time_min * 60

        if method == "constant_source":
            return constant_source_profile(
                x_nm=depth_nm,
                time_sec=time_sec,
                temp_celsius=temp_c,
                dopant=dopant,
                surface_conc=surface_conc,
                background=background
            )
        else:  # limited_source
            if dose is None:
                raise ValueError("dose required for limited_source method")
            return limited_source_profile(
                x_nm=depth_nm,
                time_sec=time_sec,
                temp_celsius=temp_c,
                dopant=dopant,
                dose_per_cm2=dose,
                background=background
            )

    @staticmethod
    def junction_depth(concentration: np.ndarray, depth_nm: np.ndarray, background: float) -> float:
        """Calculate junction depth from concentration profile."""
        from session2.erfc import junction_depth as calc_xj
        return calc_xj(concentration, depth_nm, background)

    @staticmethod
    def numerical_solve(
        initial_conc: np.ndarray,
        time_sec: float,
        temp_c: float,
        dopant: str,
        nx: int = 100,
        L_nm: float = 500,
        bc_left: Tuple[str, float] = ('dirichlet', 1e20),
        bc_right: Tuple[str, float] = ('neumann', 0.0)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve diffusion using numerical FD solver.

        Args:
            initial_conc: Initial concentration profile
            time_sec: Simulation time (seconds)
            temp_c: Temperature (°C)
            dopant: Dopant species
            nx: Number of grid points
            L_nm: Domain length (nm)
            bc_left: Left boundary condition (type, value)
            bc_right: Right boundary condition (type, value)

        Returns:
            (depth_nm, concentration): Final profile
        """
        from session3.fick_fd import DiffusionSolver1D, DiffusionParams

        # Get diffusivity parameters
        diffusivity_params = {
            'boron': (0.76, 3.69),
            'phosphorus': (3.85, 3.66),
            'arsenic': (0.066, 3.44),
            'antimony': (0.214, 3.65)
        }
        D0, Ea = diffusivity_params.get(dopant.lower(), (1.0, 3.5))

        params = DiffusionParams(D0=D0, Ea=Ea, dt_sec=1.0)
        solver = DiffusionSolver1D(params, nx, L_nm)

        times, C_history = solver.solve(
            C0=initial_conc,
            t_final_sec=time_sec,
            bc_left=bc_left,
            bc_right=bc_right,
            temp_celsius=temp_c
        )

        depth_nm = np.linspace(0, L_nm, nx)
        return depth_nm, C_history[-1]


# ============================================================================
# OXIDATION API
# ============================================================================

class OxidationAPI:
    """Clean API for oxidation calculations."""

    @staticmethod
    def deal_grove_thickness(
        temp_c: float,
        time_hr: float,
        ambient: str = "dry",
        pressure: float = 1.0,
        initial_thickness_nm: float = 0.0
    ) -> float:
        """
        Calculate oxide thickness using Deal-Grove model.

        Args:
            temp_c: Temperature (°C)
            time_hr: Oxidation time (hours)
            ambient: "dry" or "wet"
            pressure: Partial pressure (atm)
            initial_thickness_nm: Initial oxide thickness (nm)

        Returns:
            Final oxide thickness (nm)
        """
        from session4.deal_grove import thickness_at_time

        x_i_um = initial_thickness_nm / 1000.0
        thickness_um = thickness_at_time(
            t=time_hr,
            T=temp_c,
            ambient=ambient,
            pressure=pressure,
            x_i=x_i_um
        )
        return thickness_um * 1000.0  # Convert to nm

    @staticmethod
    def time_to_target(
        target_thickness_nm: float,
        temp_c: float,
        ambient: str = "dry",
        pressure: float = 1.0,
        initial_thickness_nm: float = 0.0
    ) -> float:
        """
        Calculate time required to reach target thickness (inverse problem).

        Args:
            target_thickness_nm: Target oxide thickness (nm)
            temp_c: Temperature (°C)
            ambient: "dry" or "wet"
            pressure: Partial pressure (atm)
            initial_thickness_nm: Initial oxide thickness (nm)

        Returns:
            Required time (hours)
        """
        from session4.deal_grove import time_to_thickness

        target_um = target_thickness_nm / 1000.0
        x_i_um = initial_thickness_nm / 1000.0

        return time_to_thickness(
            x_target=target_um,
            T=temp_c,
            ambient=ambient,
            pressure=pressure,
            x_i=x_i_um
        )

    @staticmethod
    def growth_rate(
        thickness_nm: float,
        temp_c: float,
        ambient: str = "dry",
        pressure: float = 1.0
    ) -> float:
        """
        Calculate instantaneous growth rate at given thickness.

        Args:
            thickness_nm: Current oxide thickness (nm)
            temp_c: Temperature (°C)
            ambient: "dry" or "wet"
            pressure: Partial pressure (atm)

        Returns:
            Growth rate (nm/hr)
        """
        from session4.deal_grove import growth_rate as calc_rate

        thickness_um = thickness_nm / 1000.0
        rate_um_hr = calc_rate(
            x_ox=thickness_um,
            T=temp_c,
            ambient=ambient,
            pressure=pressure
        )
        return rate_um_hr * 1000.0  # Convert to nm/hr


# ============================================================================
# SPC API
# ============================================================================

class SPCAPI:
    """Clean API for Statistical Process Control."""

    @staticmethod
    def check_rules(
        data: np.ndarray,
        centerline: Optional[float] = None,
        sigma: Optional[float] = None,
        rules: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Check Western Electric & Nelson SPC rules.

        Args:
            data: Process measurements
            centerline: Process mean (calculated if None)
            sigma: Process std dev (calculated if None)
            rules: List of rules to check (all if None)

        Returns:
            List of violations with details
        """
        from session7.spc import quick_rule_check

        violations = quick_rule_check(
            data=data,
            centerline=centerline,
            sigma=sigma
        )

        return [
            {
                'rule': v.rule.value,
                'index': v.index,
                'severity': v.severity.value,
                'description': v.description,
                'metric_value': v.metric_value
            }
            for v in violations
        ]

    @staticmethod
    def ewma_monitor(
        data: np.ndarray,
        lambda_param: float = 0.2,
        L: float = 3.0
    ) -> List[Dict[str, Any]]:
        """
        Monitor with EWMA control chart.

        Args:
            data: Process measurements
            lambda_param: Smoothing parameter (0-1)
            L: Control limit multiplier

        Returns:
            List of violations
        """
        from session7.spc import quick_ewma_check

        violations = quick_ewma_check(data=data)

        return [
            {
                'index': v.index,
                'ewma_value': v.ewma_value,
                'control_limit': v.control_limit,
                'side': v.side,
                'description': v.description
            }
            for v in violations
        ]

    @staticmethod
    def detect_changepoints(
        data: np.ndarray,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Detect change points using BOCPD.

        Args:
            data: Process measurements
            threshold: Probability threshold for detection

        Returns:
            List of detected change points
        """
        from session7.spc import quick_bocpd_check

        changepoints = quick_bocpd_check(data=data)

        return [
            {
                'index': cp.index,
                'probability': cp.probability,
                'run_length': cp.run_length,
                'description': cp.description
            }
            for cp in changepoints if cp.probability >= threshold
        ]


# ============================================================================
# ML / VIRTUAL METROLOGY API
# ============================================================================

class MLAPI:
    """Clean API for Machine Learning and Virtual Metrology."""

    @staticmethod
    def extract_features(fdc_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract FDC features for virtual metrology.

        Args:
            fdc_data: FDC sensor data dictionary

        Returns:
            Dictionary of extracted features
        """
        from integrated.ml.features import extract_features_from_fdc_data
        return extract_features_from_fdc_data(fdc_data)

    @staticmethod
    def calibrate_params(
        x_data: np.ndarray,
        y_data: np.ndarray,
        model_type: str,
        method: str = "least_squares",
        **kwargs
    ) -> Dict[str, Any]:
        """
        Calibrate model parameters from data.

        Args:
            x_data: Independent variable (e.g., depth, time)
            y_data: Dependent variable (e.g., concentration, thickness)
            model_type: "diffusion" or "oxidation"
            method: "least_squares" or "mcmc"
            **kwargs: Additional context parameters

        Returns:
            Calibration result with parameters and uncertainties
        """
        if model_type == "diffusion":
            from integrated.ml.calibrate import calibrate_diffusion_params
            result = calibrate_diffusion_params(
                x_data=x_data,
                concentration_data=y_data,
                method=method,
                **kwargs
            )
        else:  # oxidation
            from integrated.ml.calibrate import calibrate_oxidation_params
            result = calibrate_oxidation_params(
                time_data=x_data,
                thickness_data=y_data,
                method=method,
                **kwargs
            )

        return {
            'parameters': result.parameters,
            'uncertainties': result.uncertainties,
            'method': result.method,
            'r_squared': result.r_squared if hasattr(result, 'r_squared') else None
        }


# ============================================================================
# STABLE API INSTANCE
# ============================================================================

# Create singleton instances
diffusion = DiffusionAPI()
oxidation = OxidationAPI()
spc = SPCAPI()
ml = MLAPI()

__all__ = [
    "diffusion",
    "oxidation",
    "spc",
    "ml",
    "DiffusionAPI",
    "OxidationAPI",
    "SPCAPI",
    "MLAPI",
]
