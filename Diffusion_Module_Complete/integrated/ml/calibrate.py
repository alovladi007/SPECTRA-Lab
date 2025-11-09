"""
Calibration & Uncertainty Quantification for Diffusion Parameters - Session 9

This module provides tools for:
1. Parameter estimation using nonlinear least squares
2. Bayesian calibration with MCMC (emcee)
3. Uncertainty quantification with credible intervals
4. Prior definitions for diffusion and oxidation parameters

Supports calibration of:
- Diffusion parameters: D0 (pre-exponential), Ea (activation energy)
- Oxidation parameters: B (parabolic rate constant), A (linear rate constant)
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional, Callable, Dict, List, Any
import numpy as np
from scipy import optimize, stats
import warnings

# Optional: emcee for MCMC (lightweight dependency)
try:
    import emcee
    HAS_EMCEE = True
except ImportError:
    HAS_EMCEE = False
    warnings.warn("emcee not installed. Bayesian calibration will not be available.")


@dataclass
class Prior:
    """
    Prior distribution for a parameter.

    Attributes:
        distribution: scipy.stats distribution object
        name: parameter name
        bounds: (lower, upper) bounds for the parameter
    """
    distribution: Any
    name: str
    bounds: Tuple[float, float]

    def log_prob(self, value: float) -> float:
        """Log probability of value under this prior."""
        if not (self.bounds[0] <= value <= self.bounds[1]):
            return -np.inf
        return self.distribution.logpdf(value)

    def sample(self, size: int = 1) -> np.ndarray:
        """Sample from the prior."""
        samples = self.distribution.rvs(size=size)
        # Clip to bounds
        return np.clip(samples, self.bounds[0], self.bounds[1])


@dataclass
class DiffusionPriors:
    """
    Prior distributions for diffusion parameters.

    Parameters:
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
    """
    D0_prior: Prior
    Ea_prior: Prior

    @classmethod
    def boron_priors(cls) -> 'DiffusionPriors':
        """Default priors for boron diffusion."""
        return cls(
            D0_prior=Prior(
                distribution=stats.lognorm(s=0.5, scale=0.76),  # Mean ~ 0.76
                name="D0_boron",
                bounds=(0.1, 10.0)
            ),
            Ea_prior=Prior(
                distribution=stats.norm(loc=3.69, scale=0.1),
                name="Ea_boron",
                bounds=(3.4, 4.0)
            )
        )

    @classmethod
    def phosphorus_priors(cls) -> 'DiffusionPriors':
        """Default priors for phosphorus diffusion."""
        return cls(
            D0_prior=Prior(
                distribution=stats.lognorm(s=0.5, scale=3.85),  # Mean ~ 3.85
                name="D0_phosphorus",
                bounds=(1.0, 20.0)
            ),
            Ea_prior=Prior(
                distribution=stats.norm(loc=3.66, scale=0.1),
                name="Ea_phosphorus",
                bounds=(3.4, 4.0)
            )
        )

    @classmethod
    def arsenic_priors(cls) -> 'DiffusionPriors':
        """Default priors for arsenic diffusion."""
        return cls(
            D0_prior=Prior(
                distribution=stats.lognorm(s=0.5, scale=0.066),  # Mean ~ 0.066
                name="D0_arsenic",
                bounds=(0.01, 1.0)
            ),
            Ea_prior=Prior(
                distribution=stats.norm(loc=3.44, scale=0.1),
                name="Ea_arsenic",
                bounds=(3.2, 3.8)
            )
        )


@dataclass
class OxidationPriors:
    """
    Prior distributions for Deal-Grove oxidation parameters.

    Parameters:
        B: Parabolic rate constant (µm²/hr)
        A: Linear rate constant (µm/hr)
    """
    B_prior: Prior
    A_prior: Prior

    @classmethod
    def dry_oxidation_priors(cls, temp_celsius: float = 1000.0) -> 'OxidationPriors':
        """
        Default priors for dry oxidation at given temperature.

        Args:
            temp_celsius: Oxidation temperature (°C)
        """
        # Rough temperature-dependent estimates
        # B/A ≈ exp(-Ea/kT), so higher T → higher rates
        T_ratio = (temp_celsius + 273.15) / 1273.15  # Normalized to 1000°C

        return cls(
            B_prior=Prior(
                distribution=stats.lognorm(s=0.3, scale=0.03 * T_ratio**2),
                name="B_dry",
                bounds=(0.001, 1.0)
            ),
            A_prior=Prior(
                distribution=stats.lognorm(s=0.3, scale=0.15 * T_ratio),
                name="A_dry",
                bounds=(0.01, 10.0)
            )
        )

    @classmethod
    def wet_oxidation_priors(cls, temp_celsius: float = 1000.0) -> 'OxidationPriors':
        """Default priors for wet oxidation at given temperature."""
        T_ratio = (temp_celsius + 273.15) / 1273.15

        return cls(
            B_prior=Prior(
                distribution=stats.lognorm(s=0.3, scale=0.38 * T_ratio**2),
                name="B_wet",
                bounds=(0.01, 5.0)
            ),
            A_prior=Prior(
                distribution=stats.lognorm(s=0.3, scale=0.25 * T_ratio),
                name="A_wet",
                bounds=(0.05, 10.0)
            )
        )


@dataclass
class CalibrationResult:
    """
    Result of parameter calibration.

    Attributes:
        parameters: Best-fit parameter values
        uncertainties: Standard errors (for least squares) or credible intervals (for Bayesian)
        method: Calibration method used
        success: Whether calibration succeeded
        metadata: Additional information
    """
    parameters: Dict[str, float]
    uncertainties: Dict[str, Tuple[float, float]]  # (lower, upper) or (std, std)
    method: str
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)

    def credible_interval(self, param_name: str, alpha: float = 0.05) -> Tuple[float, float]:
        """
        Get credible interval for a parameter.

        Args:
            param_name: Name of parameter
            alpha: Significance level (default 0.05 for 95% CI)

        Returns:
            (lower, upper) bounds of credible interval
        """
        if param_name not in self.uncertainties:
            raise ValueError(f"Parameter {param_name} not found")

        return self.uncertainties[param_name]


class LeastSquaresCalibrator:
    """
    Nonlinear least squares calibration for parameters.

    Uses scipy.optimize.least_squares with robust loss functions.
    """

    def __init__(
        self,
        model_func: Callable,
        param_names: List[str],
        bounds: Optional[Tuple[List[float], List[float]]] = None,
        loss: str = 'soft_l1'
    ):
        """
        Initialize calibrator.

        Args:
            model_func: Model function model_func(x, *params) -> y_pred
            param_names: Names of parameters to calibrate
            bounds: ([lower_bounds], [upper_bounds]) for parameters
            loss: Loss function ('linear', 'soft_l1', 'huber', 'cauchy')
        """
        self.model_func = model_func
        self.param_names = param_names
        self.bounds = bounds if bounds is not None else (-np.inf, np.inf)
        self.loss = loss

    def residuals(self, params: np.ndarray, x: np.ndarray, y_obs: np.ndarray) -> np.ndarray:
        """Compute residuals between model and observations."""
        y_pred = self.model_func(x, *params)
        return y_obs - y_pred

    def calibrate(
        self,
        x: np.ndarray,
        y_obs: np.ndarray,
        y_std: Optional[np.ndarray] = None,
        p0: Optional[np.ndarray] = None
    ) -> CalibrationResult:
        """
        Calibrate parameters using least squares.

        Args:
            x: Independent variable data
            y_obs: Observed dependent variable
            y_std: Standard deviation of observations (for weighting)
            p0: Initial parameter guess

        Returns:
            CalibrationResult with best-fit parameters and uncertainties
        """
        n_params = len(self.param_names)
        if p0 is None:
            # Use midpoint of bounds as initial guess
            if isinstance(self.bounds[0], (list, np.ndarray)):
                p0 = np.array([(self.bounds[0][i] + self.bounds[1][i]) / 2
                               for i in range(n_params)])
            else:
                p0 = np.ones(n_params)

        # Weight by measurement uncertainty if provided
        if y_std is not None:
            def weighted_residuals(params, x, y_obs):
                return self.residuals(params, x, y_obs) / y_std
            resid_func = weighted_residuals
        else:
            resid_func = self.residuals

        # Perform least squares optimization
        result = optimize.least_squares(
            resid_func,
            p0,
            args=(x, y_obs),
            bounds=self.bounds,
            loss=self.loss,
            verbose=0
        )

        # Compute parameter uncertainties from Jacobian
        # Covariance = (J^T J)^-1 * residual_variance
        J = result.jac
        try:
            cov = np.linalg.inv(J.T @ J)
            residual_var = np.sum(result.fun**2) / (len(y_obs) - n_params)
            cov *= residual_var
            param_std = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            # Singular matrix, use rough estimate
            param_std = np.abs(result.x) * 0.1

        # Package results
        parameters = {name: result.x[i] for i, name in enumerate(self.param_names)}
        uncertainties = {
            name: (result.x[i] - 2*param_std[i], result.x[i] + 2*param_std[i])
            for i, name in enumerate(self.param_names)
        }

        return CalibrationResult(
            parameters=parameters,
            uncertainties=uncertainties,
            method="least_squares",
            success=result.success,
            metadata={
                'cost': result.cost,
                'optimality': result.optimality,
                'nfev': result.nfev,
                'covariance': cov.tolist() if 'cov' in locals() else None
            }
        )


class BayesianCalibrator:
    """
    Bayesian calibration using MCMC (emcee).

    Estimates posterior distributions of parameters given data and priors.
    """

    def __init__(
        self,
        model_func: Callable,
        param_names: List[str],
        priors: List[Prior],
        likelihood_std: float = 1.0
    ):
        """
        Initialize Bayesian calibrator.

        Args:
            model_func: Model function model_func(x, *params) -> y_pred
            param_names: Names of parameters
            priors: List of Prior objects for each parameter
            likelihood_std: Standard deviation of measurement noise (if known)
        """
        if not HAS_EMCEE:
            raise ImportError("emcee is required for Bayesian calibration. Install with: pip install emcee")

        self.model_func = model_func
        self.param_names = param_names
        self.priors = priors
        self.likelihood_std = likelihood_std

        if len(priors) != len(param_names):
            raise ValueError("Number of priors must match number of parameters")

    def log_prior(self, params: np.ndarray) -> float:
        """Compute log prior probability."""
        log_p = 0.0
        for param_val, prior in zip(params, self.priors):
            log_p += prior.log_prob(param_val)
        return log_p

    def log_likelihood(self, params: np.ndarray, x: np.ndarray, y_obs: np.ndarray) -> float:
        """Compute log likelihood."""
        y_pred = self.model_func(x, *params)
        residuals = y_obs - y_pred

        # Gaussian likelihood: log P(y|params) = -0.5 * sum((y - y_pred)^2 / sigma^2)
        log_like = -0.5 * np.sum((residuals / self.likelihood_std)**2)
        log_like -= len(y_obs) * np.log(self.likelihood_std * np.sqrt(2 * np.pi))

        return log_like

    def log_posterior(self, params: np.ndarray, x: np.ndarray, y_obs: np.ndarray) -> float:
        """Compute log posterior probability (prior + likelihood)."""
        log_p = self.log_prior(params)
        if not np.isfinite(log_p):
            return -np.inf

        return log_p + self.log_likelihood(params, x, y_obs)

    def calibrate(
        self,
        x: np.ndarray,
        y_obs: np.ndarray,
        n_walkers: int = 32,
        n_steps: int = 2000,
        n_burn: int = 500,
        p0: Optional[np.ndarray] = None
    ) -> CalibrationResult:
        """
        Calibrate parameters using MCMC.

        Args:
            x: Independent variable data
            y_obs: Observed dependent variable
            n_walkers: Number of MCMC walkers
            n_steps: Number of MCMC steps
            n_burn: Number of burn-in steps to discard
            p0: Initial parameter guess (optional)

        Returns:
            CalibrationResult with posterior samples and credible intervals
        """
        n_params = len(self.param_names)

        # Initialize walkers
        if p0 is None:
            # Sample from priors
            p0 = np.array([prior.sample(n_walkers) for prior in self.priors]).T
        else:
            # Perturb initial guess
            p0 = p0 + 1e-4 * np.random.randn(n_walkers, n_params)

        # Create sampler
        sampler = emcee.EnsembleSampler(
            n_walkers,
            n_params,
            self.log_posterior,
            args=(x, y_obs)
        )

        # Run MCMC
        sampler.run_mcmc(p0, n_steps, progress=False)

        # Extract samples (discard burn-in)
        samples = sampler.get_chain(discard=n_burn, flat=True)

        # Compute posterior statistics
        param_medians = np.median(samples, axis=0)
        param_lower = np.percentile(samples, 2.5, axis=0)  # 95% CI
        param_upper = np.percentile(samples, 97.5, axis=0)

        parameters = {name: param_medians[i] for i, name in enumerate(self.param_names)}
        uncertainties = {
            name: (param_lower[i], param_upper[i])
            for i, name in enumerate(self.param_names)
        }

        # Acceptance fraction (good range: 0.2-0.5)
        accept_frac = np.mean(sampler.acceptance_fraction)

        return CalibrationResult(
            parameters=parameters,
            uncertainties=uncertainties,
            method="mcmc_emcee",
            success=True,
            metadata={
                'samples': samples,
                'n_walkers': n_walkers,
                'n_steps': n_steps,
                'n_burn': n_burn,
                'acceptance_fraction': accept_frac
            }
        )


def predict_with_uncertainty(
    model_func: Callable,
    x: np.ndarray,
    posterior_samples: np.ndarray,
    credible_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Make predictions with uncertainty bands from posterior samples.

    Args:
        model_func: Model function model_func(x, *params) -> y_pred
        x: Input values for prediction
        posterior_samples: Array of posterior parameter samples (n_samples, n_params)
        credible_level: Credible interval level (default 0.95)

    Returns:
        (y_median, y_lower, y_upper) prediction with credible bands
    """
    n_samples = posterior_samples.shape[0]
    n_points = len(x)

    # Compute predictions for each posterior sample
    predictions = np.zeros((n_samples, n_points))
    for i, params in enumerate(posterior_samples):
        predictions[i] = model_func(x, *params)

    # Compute quantiles
    alpha = 1 - credible_level
    y_median = np.median(predictions, axis=0)
    y_lower = np.percentile(predictions, 100 * alpha / 2, axis=0)
    y_upper = np.percentile(predictions, 100 * (1 - alpha / 2), axis=0)

    return y_median, y_lower, y_upper


# Quick calibration helpers

def calibrate_diffusion_params(
    x_data: np.ndarray,
    concentration_data: np.ndarray,
    time_sec: float,
    temp_celsius: float,
    dopant: str = "boron",
    method: str = "least_squares"
) -> CalibrationResult:
    """
    Quick calibration of diffusion parameters from concentration profile.

    Args:
        x_data: Depth points (nm)
        concentration_data: Concentration at each depth (cm⁻³)
        time_sec: Diffusion time (seconds)
        temp_celsius: Temperature (°C)
        dopant: Dopant type ("boron", "phosphorus", "arsenic")
        method: "least_squares" or "mcmc"

    Returns:
        CalibrationResult with D0 and Ea
    """
    from session2.erfc import constant_source_profile

    # Define model function
    def model(x, D0, Ea):
        # Diffusivity from Arrhenius
        k = 8.617e-5  # eV/K
        T_K = temp_celsius + 273.15
        D = D0 * np.exp(-Ea / (k * T_K))

        # Use erfc solution (assuming constant source for this example)
        _, C = constant_source_profile(
            x_nm=x,
            time_sec=time_sec,
            temp_celsius=temp_celsius,
            D_cm2_per_s=D,
            surface_conc=concentration_data[0],
            background=concentration_data[-1]
        )
        return C

    param_names = ["D0", "Ea"]

    if method == "least_squares":
        # Get bounds from priors
        if dopant.lower() == "boron":
            priors = DiffusionPriors.boron_priors()
        elif dopant.lower() == "phosphorus":
            priors = DiffusionPriors.phosphorus_priors()
        else:  # arsenic
            priors = DiffusionPriors.arsenic_priors()

        bounds = (
            [priors.D0_prior.bounds[0], priors.Ea_prior.bounds[0]],
            [priors.D0_prior.bounds[1], priors.Ea_prior.bounds[1]]
        )

        calibrator = LeastSquaresCalibrator(model, param_names, bounds=bounds)
        return calibrator.calibrate(x_data, concentration_data)

    elif method == "mcmc":
        # Bayesian calibration
        if dopant.lower() == "boron":
            diff_priors = DiffusionPriors.boron_priors()
        elif dopant.lower() == "phosphorus":
            diff_priors = DiffusionPriors.phosphorus_priors()
        else:
            diff_priors = DiffusionPriors.arsenic_priors()

        priors = [diff_priors.D0_prior, diff_priors.Ea_prior]

        # Estimate noise level from data
        noise_std = np.std(np.diff(concentration_data)) * 0.5

        calibrator = BayesianCalibrator(model, param_names, priors, likelihood_std=noise_std)
        return calibrator.calibrate(x_data, concentration_data, n_steps=1500, n_burn=300)

    else:
        raise ValueError(f"Unknown method: {method}")


def calibrate_oxidation_params(
    time_data: np.ndarray,
    thickness_data: np.ndarray,
    temp_celsius: float,
    ambient: str = "dry",
    method: str = "least_squares"
) -> CalibrationResult:
    """
    Quick calibration of oxidation parameters from thickness vs time data.

    Args:
        time_data: Oxidation times (hours)
        thickness_data: Oxide thickness at each time (nm)
        temp_celsius: Temperature (°C)
        ambient: "dry" or "wet"
        method: "least_squares" or "mcmc"

    Returns:
        CalibrationResult with B and A parameters
    """
    from session4.deal_grove import deal_grove_thickness

    # Define model function (Deal-Grove)
    def model(t_hours, B, A):
        thicknesses = []
        for t in t_hours:
            tox, _ = deal_grove_thickness(
                time_hours=t,
                temp_celsius=temp_celsius,
                B_um2_per_hr=B,
                A_um_per_hr=A,
                initial_oxide_nm=0.0
            )
            thicknesses.append(tox)
        return np.array(thicknesses)

    param_names = ["B", "A"]

    if method == "least_squares":
        # Get priors for bounds
        if ambient.lower() == "dry":
            ox_priors = OxidationPriors.dry_oxidation_priors(temp_celsius)
        else:
            ox_priors = OxidationPriors.wet_oxidation_priors(temp_celsius)

        bounds = (
            [ox_priors.B_prior.bounds[0], ox_priors.A_prior.bounds[0]],
            [ox_priors.B_prior.bounds[1], ox_priors.A_prior.bounds[1]]
        )

        calibrator = LeastSquaresCalibrator(model, param_names, bounds=bounds)
        return calibrator.calibrate(time_data, thickness_data)

    elif method == "mcmc":
        if ambient.lower() == "dry":
            ox_priors = OxidationPriors.dry_oxidation_priors(temp_celsius)
        else:
            ox_priors = OxidationPriors.wet_oxidation_priors(temp_celsius)

        priors = [ox_priors.B_prior, ox_priors.A_prior]
        noise_std = np.std(np.diff(thickness_data)) * 0.5

        calibrator = BayesianCalibrator(model, param_names, priors, likelihood_std=noise_std)
        return calibrator.calibrate(time_data, thickness_data, n_steps=1500, n_burn=300)

    else:
        raise ValueError(f"Unknown method: {method}")
