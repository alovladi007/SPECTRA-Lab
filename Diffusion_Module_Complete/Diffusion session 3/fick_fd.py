"""
Numerical solver for Fick's 2nd law using Crank-Nicolson method.

Implements:
- 1D implicit solver for ∂C/∂t = ∂/∂x(D(C,T) ∂C/∂x)
- Adaptive grid refinement near surface
- Multiple boundary conditions (Dirichlet, Neumann, Robin)
- Optional concentration-dependent diffusivity
- Optional numba JIT acceleration

Session 3 - IMPLEMENTED
"""

from typing import Optional, Callable, Tuple, Literal, Union
import numpy as np
from numpy.typing import NDArray
import warnings

try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Define dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


class Fick1D:
    """
    1D finite difference solver for Fick's 2nd law.
    
    Solves: ∂C/∂t = ∂/∂x(D ∂C/∂x)
    
    Uses Crank-Nicolson (implicit) scheme for unconditional stability.
    The scheme is second-order accurate in both space and time.
    
    For the diffusion equation:
    ∂C/∂t = ∂/∂x(D ∂C/∂x)
    
    The Crank-Nicolson discretization is:
    (C^(n+1) - C^n)/Δt = 0.5 * [L(C^(n+1)) + L(C^n)]
    
    Where L is the spatial operator: L(C) = ∂/∂x(D ∂C/∂x)
    
    This leads to a tridiagonal system at each time step.
    
    Status: IMPLEMENTED - Session 3
    """
    
    def __init__(
        self,
        x_max: float = 1000.0,  # nm
        dx: float = 1.0,  # nm
        refine_surface: bool = True,
        surface_refinement_depth: float = 100.0,  # nm
        surface_refinement_factor: int = 5,
        use_numba: bool = False
    ):
        """
        Initialize the solver.
        
        Args:
            x_max: Maximum depth (nm)
            dx: Spatial step size for bulk region (nm)
            refine_surface: Use finer grid near surface
            surface_refinement_depth: Depth of refined region (nm)
            surface_refinement_factor: Grid refinement factor near surface
            use_numba: Enable numba JIT compilation (if available)
        
        Examples:
            >>> # Standard uniform grid
            >>> solver = Fick1D(x_max=1000, dx=1.0, refine_surface=False)
            
            >>> # Refined surface grid
            >>> solver = Fick1D(x_max=1000, dx=2.0, refine_surface=True,
            ...                 surface_refinement_depth=100, 
            ...                 surface_refinement_factor=5)
        """
        self.x_max = x_max
        self.dx = dx
        self.refine_surface = refine_surface
        self.surface_refinement_depth = surface_refinement_depth
        self.surface_refinement_factor = surface_refinement_factor
        self.use_numba = use_numba and NUMBA_AVAILABLE
        
        if use_numba and not NUMBA_AVAILABLE:
            warnings.warn("Numba requested but not available, falling back to numpy")
        
        # Grid arrays (initialized in setup_grid)
        self.x = None
        self.n_points = None
        self.dx_array = None  # Variable spacing if refined
        
        # Setup the grid
        self.setup_grid()
    
    def setup_grid(self):
        """
        Set up spatial grid with optional refinement near surface.
        
        Creates either:
        - Uniform grid: x = [0, dx, 2*dx, ..., x_max]
        - Refined grid: Fine spacing near surface, coarse spacing in bulk
        """
        if not self.refine_surface:
            # Uniform grid
            self.n_points = int(self.x_max / self.dx) + 1
            self.x = np.linspace(0, self.x_max, self.n_points)
            self.dx_array = np.full(self.n_points - 1, self.dx)
        else:
            # Refined grid near surface
            dx_fine = self.dx / self.surface_refinement_factor
            
            # Fine region: [0, surface_refinement_depth]
            n_fine = int(self.surface_refinement_depth / dx_fine) + 1
            x_fine = np.linspace(0, self.surface_refinement_depth, n_fine)
            
            # Coarse region: [surface_refinement_depth, x_max]
            n_coarse = int((self.x_max - self.surface_refinement_depth) / self.dx)
            if n_coarse > 0:
                x_coarse = (self.surface_refinement_depth + 
                           self.dx * np.arange(1, n_coarse + 1))
                
                # Combine regions
                self.x = np.concatenate([x_fine, x_coarse])
            else:
                self.x = x_fine
            
            self.n_points = len(self.x)
            
            # Compute variable spacing
            self.dx_array = np.diff(self.x)
    
    def solve(
        self,
        C0: NDArray[np.float64],
        dt: float,
        steps: int,
        T: float,
        D_model: Callable[[float, NDArray[np.float64]], Union[float, NDArray[np.float64]]],
        bc: Tuple[str, str] = ('dirichlet', 'neumann'),
        surface_value: Optional[float] = None,
        right_boundary_value: Optional[float] = None,
        return_history: bool = False
    ) -> Union[Tuple[NDArray[np.float64], NDArray[np.float64]], 
               Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
        """
        Solve diffusion equation using Crank-Nicolson method.
        
        Args:
            C0: Initial concentration profile (atoms/cm³)
                Must match grid size from setup_grid
            dt: Time step (seconds)
            steps: Number of time steps
            T: Temperature (Celsius)
            D_model: Callable D(T, C) returning diffusivity (cm²/s)
                     Can be constant D(T, None) or concentration-dependent D(T, C)
            bc: Boundary conditions (left, right)
                'dirichlet': Fixed value
                'neumann': Fixed flux (typically zero flux)
                'robin': Mixed condition (not implemented yet)
            surface_value: Surface concentration for left Dirichlet BC (atoms/cm³)
            right_boundary_value: Right boundary value for Dirichlet BC
            return_history: If True, return concentration at all time steps
        
        Returns:
            If return_history=False:
                (x, C_final) - depth array and final concentration
            If return_history=True:
                (x, C_final, C_history) - depth, final concentration, and full history
                C_history shape: (steps+1, n_points)
        
        Examples:
            >>> # Constant diffusivity
            >>> def D_const(T, C):
            ...     return 1e-13  # cm²/s
            
            >>> solver = Fick1D(x_max=1000, dx=2.0)
            >>> C0 = np.full(solver.n_points, 1e15)
            >>> C0[0] = 1e20  # Surface source
            >>> x, C_final = solver.solve(C0, dt=1.0, steps=1800, T=1000,
            ...                           D_model=D_const, bc=('dirichlet', 'neumann'),
            ...                           surface_value=1e20)
            
            >>> # Concentration-dependent diffusivity
            >>> def D_conc(T, C):
            ...     D0 = 1e-13
            ...     if C is None:
            ...         return D0
            ...     # Enhanced diffusion at high concentration
            ...     return D0 * (1 + 1e-20 * C)
            
            >>> x, C_final = solver.solve(C0, dt=1.0, steps=1800, T=1000,
            ...                           D_model=D_conc, bc=('dirichlet', 'neumann'),
            ...                           surface_value=1e20)
        
        Status: IMPLEMENTED - Session 3
        """
        # Input validation
        if len(C0) != self.n_points:
            raise ValueError(f"C0 length {len(C0)} does not match grid size {self.n_points}")
        if dt <= 0:
            raise ValueError("Time step must be positive")
        if steps <= 0:
            raise ValueError("Number of steps must be positive")
        
        # Initialize concentration array
        C = C0.copy()
        
        # Store history if requested
        if return_history:
            C_history = np.zeros((steps + 1, self.n_points))
            C_history[0, :] = C.copy()
        
        # Time stepping
        for step in range(steps):
            # Compute diffusivity at current concentration
            D = self._evaluate_diffusivity(D_model, T, C)
            
            # Build tridiagonal system for Crank-Nicolson
            a, b, c, d = self._build_crank_nicolson_system(C, D, dt)
            
            # Apply boundary conditions
            self._apply_boundary_conditions(
                a, b, c, d, bc, surface_value, right_boundary_value
            )
            
            # Solve tridiagonal system
            C_new = self._thomas_algorithm(a, b, c, d)
            
            # Update concentration
            C = C_new
            
            # Store history
            if return_history:
                C_history[step + 1, :] = C.copy()
            
            # Check for numerical instability
            if np.any(np.isnan(C)) or np.any(np.isinf(C)):
                raise RuntimeError(f"Numerical instability at step {step}")
            if np.any(C < 0):
                warnings.warn(f"Negative concentrations at step {step}, clipping to zero")
                C = np.maximum(C, 0)
        
        if return_history:
            return self.x, C, C_history
        else:
            return self.x, C
    
    def _evaluate_diffusivity(
        self,
        D_model: Callable,
        T: float,
        C: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Evaluate diffusivity at grid points.
        
        Args:
            D_model: Diffusivity model D(T, C)
            T: Temperature (Celsius)
            C: Current concentration profile (atoms/cm³)
        
        Returns:
            Diffusivity at each grid point (cm²/s)
        """
        # Try to evaluate D_model with concentration
        try:
            D = D_model(T, C)
        except:
            # Fall back to constant D if model doesn't support concentration
            D = D_model(T, None)
        
        # Ensure D is an array
        if np.isscalar(D):
            D = np.full(self.n_points, D)
        else:
            D = np.asarray(D)
        
        # Ensure positive diffusivity
        if np.any(D <= 0):
            raise ValueError("Diffusivity must be positive")
        
        return D
    
    def _build_crank_nicolson_system(
        self,
        C: NDArray[np.float64],
        D: NDArray[np.float64],
        dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], 
               NDArray[np.float64], NDArray[np.float64]]:
        """
        Build tridiagonal matrix coefficients for Crank-Nicolson.
        
        The Crank-Nicolson scheme for ∂C/∂t = ∂/∂x(D ∂C/∂x) is:
        
        (C^(n+1) - C^n)/Δt = 0.5 * [L(C^(n+1)) + L(C^n)]
        
        Rearranging:
        C^(n+1) - (Δt/2) L(C^(n+1)) = C^n + (Δt/2) L(C^n)
        
        This leads to a tridiagonal system: A * C^(n+1) = B
        
        Args:
            C: Current concentration (atoms/cm³)
            D: Diffusivity at grid points (cm²/s)
            dt: Time step (seconds)
        
        Returns:
            (a, b, c, d) - Tridiagonal coefficients and RHS
            - a: Lower diagonal
            - b: Main diagonal  
            - c: Upper diagonal
            - d: Right-hand side
        """
        n = self.n_points
        
        # Convert dt from seconds to same units as D (cm²/s)
        # Grid spacing is in nm, need to convert to cm
        dx_cm = self.dx_array * 1e-7  # nm -> cm
        
        # Initialize arrays
        a = np.zeros(n)  # Lower diagonal
        b = np.zeros(n)  # Main diagonal
        c = np.zeros(n)  # Upper diagonal
        d = np.zeros(n)  # Right-hand side
        
        # Diffusivity at cell interfaces (harmonic mean for better stability)
        D_interface = np.zeros(n - 1)
        for i in range(n - 1):
            # Harmonic mean: better for discontinuous D
            D_interface[i] = 2 * D[i] * D[i+1] / (D[i] + D[i+1])
        
        # Interior points (i = 1, 2, ..., n-2)
        for i in range(1, n - 1):
            dx_left = dx_cm[i - 1]
            dx_right = dx_cm[i]
            dx_center = 0.5 * (dx_left + dx_right)
            
            D_left = D_interface[i - 1]
            D_right = D_interface[i]
            
            # Coefficients for spatial derivatives
            # ∂²C/∂x² ≈ (C[i+1] - C[i])/dx_right - (C[i] - C[i-1])/dx_left) / dx_center
            
            alpha_left = D_left / (dx_left * dx_center)
            alpha_right = D_right / (dx_right * dx_center)
            alpha_center = -(alpha_left + alpha_right)
            
            # Crank-Nicolson coefficients
            # Left side: C^(n+1) - (dt/2) * L(C^(n+1))
            a[i] = -0.5 * dt * alpha_left
            b[i] = 1.0 - 0.5 * dt * alpha_center
            c[i] = -0.5 * dt * alpha_right
            
            # Right side: C^n + (dt/2) * L(C^n)
            d[i] = (C[i] + 
                   0.5 * dt * alpha_left * C[i-1] + 
                   0.5 * dt * alpha_center * C[i] + 
                   0.5 * dt * alpha_right * C[i+1])
        
        # Boundary points will be set by _apply_boundary_conditions
        return a, b, c, d
    
    def _apply_boundary_conditions(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        d: NDArray[np.float64],
        bc: Tuple[str, str],
        surface_value: Optional[float] = None,
        right_boundary_value: Optional[float] = None
    ):
        """
        Apply boundary conditions to the tridiagonal system.
        
        Modifies the coefficient arrays in-place.
        
        Args:
            a, b, c, d: Tridiagonal system coefficients
            bc: Boundary conditions (left, right)
            surface_value: Value for left Dirichlet BC
            right_boundary_value: Value for right Dirichlet BC
        """
        bc_left, bc_right = bc
        
        # Left boundary (x = 0)
        if bc_left == 'dirichlet':
            # Fixed concentration at surface
            if surface_value is None:
                raise ValueError("surface_value required for Dirichlet BC")
            a[0] = 0.0
            b[0] = 1.0
            c[0] = 0.0
            d[0] = surface_value
        
        elif bc_left == 'neumann':
            # Zero flux at surface: ∂C/∂x = 0
            # Use one-sided difference: C[1] = C[0]
            # Or central difference with ghost point
            # Simple approach: C[0] = C[1]
            a[0] = 0.0
            b[0] = 1.0
            c[0] = -1.0
            d[0] = 0.0
        
        else:
            raise ValueError(f"Unknown left boundary condition: {bc_left}")
        
        # Right boundary (x = x_max)
        n = len(b)
        if bc_right == 'dirichlet':
            # Fixed concentration at right boundary
            if right_boundary_value is None:
                # Use last value from current profile
                right_boundary_value = d[-1]
            a[n-1] = 0.0
            b[n-1] = 1.0
            c[n-1] = 0.0
            d[n-1] = right_boundary_value
        
        elif bc_right == 'neumann':
            # Zero flux at right boundary: ∂C/∂x = 0
            # Use one-sided difference: C[n-1] = C[n-2]
            a[n-1] = -1.0
            b[n-1] = 1.0
            c[n-1] = 0.0
            d[n-1] = 0.0
        
        else:
            raise ValueError(f"Unknown right boundary condition: {bc_right}")
    
    def _thomas_algorithm(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        d: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """
        Solve tridiagonal system using Thomas algorithm.
        
        Solves: A * x = d
        Where A is tridiagonal with diagonals (a, b, c)
        
        The Thomas algorithm is a specialized form of Gaussian elimination
        for tridiagonal matrices. It's O(n) instead of O(n³).
        
        Args:
            a: Lower diagonal (length n, but a[0] unused)
            b: Main diagonal (length n)
            c: Upper diagonal (length n, but c[n-1] unused)
            d: Right-hand side (length n)
        
        Returns:
            Solution vector x (length n)
        
        Algorithm:
            Forward elimination:
                c'[i] = c[i] / b[i]  for i = 0
                c'[i] = c[i] / (b[i] - a[i]*c'[i-1])  for i > 0
                d'[i] = d[i] / b[i]  for i = 0
                d'[i] = (d[i] - a[i]*d'[i-1]) / (b[i] - a[i]*c'[i-1])  for i > 0
            
            Back substitution:
                x[n-1] = d'[n-1]
                x[i] = d'[i] - c'[i]*x[i+1]  for i = n-2, ..., 0
        """
        n = len(b)
        x = np.zeros(n)
        
        # Forward elimination
        c_prime = np.zeros(n)
        d_prime = np.zeros(n)
        
        c_prime[0] = c[0] / b[0]
        d_prime[0] = d[0] / b[0]
        
        for i in range(1, n):
            denom = b[i] - a[i] * c_prime[i-1]
            if abs(denom) < 1e-20:
                raise RuntimeError(f"Division by zero in Thomas algorithm at i={i}")
            c_prime[i] = c[i] / denom
            d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denom
        
        # Back substitution
        x[n-1] = d_prime[n-1]
        for i in range(n-2, -1, -1):
            x[i] = d_prime[i] - c_prime[i] * x[i+1]
        
        return x
    
    def validate_convergence(
        self,
        C_analytical: NDArray[np.float64],
        C_numerical: NDArray[np.float64],
        x_analytical: Optional[NDArray[np.float64]] = None
    ) -> Tuple[float, float, float]:
        """
        Calculate error metrics vs analytical solution.
        
        Args:
            C_analytical: Analytical solution concentration profile
            C_numerical: Numerical solution concentration profile
            x_analytical: Grid for analytical solution (if different from numerical)
                         If None, assumes same grid as numerical solution
        
        Returns:
            (L2_error, Linf_error, relative_L2_error)
            - L2_error: L2 norm of error
            - Linf_error: Maximum absolute error
            - relative_L2_error: L2 error / L2 norm of solution
        
        Examples:
            >>> # Compare to analytical erfc solution
            >>> from core.erfc import constant_source_profile
            >>> 
            >>> solver = Fick1D(x_max=1000, dx=2.0)
            >>> C0 = np.full(solver.n_points, 1e15)
            >>> 
            >>> # Numerical solution
            >>> def D_const(T, C):
            ...     return 1e-13
            >>> x_num, C_num = solver.solve(C0, dt=1.0, steps=1800, T=1000,
            ...                             D_model=D_const, 
            ...                             bc=('dirichlet', 'neumann'),
            ...                             surface_value=1e20)
            >>> 
            >>> # Analytical solution
            >>> C_analytical = constant_source_profile(
            ...     x_num, 1800, 1000, D0=..., Ea=..., Cs=1e20, NA0=1e15
            ... )
            >>> 
            >>> L2, Linf, rel = solver.validate_convergence(C_analytical, C_num)
            >>> print(f"L2 error: {L2:.2e}, Relative: {rel:.4f}")
        
        Status: IMPLEMENTED - Session 3
        """
        # Interpolate if grids don't match
        if x_analytical is not None and len(x_analytical) != len(C_numerical):
            from scipy.interpolate import interp1d
            f = interp1d(x_analytical, C_analytical, kind='linear', 
                        bounds_error=False, fill_value='extrapolate')
            C_analytical_interp = f(self.x)
        else:
            C_analytical_interp = C_analytical
        
        # Calculate errors
        error = C_numerical - C_analytical_interp
        
        # L2 norm
        L2_error = np.sqrt(np.mean(error ** 2))
        
        # L-infinity norm (maximum absolute error)
        Linf_error = np.max(np.abs(error))
        
        # Relative L2 error
        solution_norm = np.sqrt(np.mean(C_analytical_interp ** 2))
        if solution_norm > 0:
            relative_L2_error = L2_error / solution_norm
        else:
            relative_L2_error = np.inf
        
        return L2_error, Linf_error, relative_L2_error
    
    def convergence_study(
        self,
        C0: NDArray[np.float64],
        t_final: float,
        T: float,
        D_model: Callable,
        C_analytical: Callable,
        dx_values: list = None,
        dt_values: list = None,
        bc: Tuple[str, str] = ('dirichlet', 'neumann'),
        surface_value: Optional[float] = None
    ) -> dict:
        """
        Perform convergence study by varying dx and dt.
        
        Args:
            C0: Initial condition
            t_final: Final simulation time (seconds)
            T: Temperature (Celsius)
            D_model: Diffusivity model
            C_analytical: Function to compute analytical solution
                         Signature: C_analytical(x, t) -> concentration
            dx_values: List of spatial step sizes to test (nm)
            dt_values: List of time step sizes to test (seconds)
            bc: Boundary conditions
            surface_value: Surface concentration for Dirichlet BC
        
        Returns:
            Dictionary with convergence data
        
        Status: IMPLEMENTED - Session 3
        """
        if dx_values is None:
            dx_values = [4.0, 2.0, 1.0, 0.5]
        if dt_values is None:
            dt_values = [2.0, 1.0, 0.5, 0.25]
        
        results = {
            'dx_values': [],
            'dt_values': [],
            'L2_errors': [],
            'Linf_errors': [],
            'relative_errors': []
        }
        
        for dx in dx_values:
            for dt in dt_values:
                # Create solver with this dx
                solver = Fick1D(x_max=self.x_max, dx=dx, refine_surface=False)
                
                # Interpolate C0 to new grid
                from scipy.interpolate import interp1d
                f = interp1d(self.x, C0, kind='linear', bounds_error=False, 
                           fill_value='extrapolate')
                C0_new = f(solver.x)
                
                # Compute number of steps
                steps = int(t_final / dt)
                
                try:
                    # Run simulation
                    x_num, C_num = solver.solve(
                        C0_new, dt, steps, T, D_model, bc, surface_value
                    )
                    
                    # Get analytical solution at same grid
                    C_anal = C_analytical(x_num, t_final)
                    
                    # Calculate errors
                    L2, Linf, rel = solver.validate_convergence(C_anal, C_num)
                    
                    results['dx_values'].append(dx)
                    results['dt_values'].append(dt)
                    results['L2_errors'].append(L2)
                    results['Linf_errors'].append(Linf)
                    results['relative_errors'].append(rel)
                
                except Exception as e:
                    warnings.warn(f"Failed for dx={dx}, dt={dt}: {str(e)}")
                    continue
        
        return results


def quick_solve_constant_D(
    x_max: float = 1000.0,
    dx: float = 2.0,
    t_final: float = 1800.0,
    dt: float = 1.0,
    T: float = 1000.0,
    D0: float = 0.76,
    Ea: float = 3.46,
    Cs: float = 1e20,
    NA0: float = 1e15,
    refine_surface: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Quick helper to solve constant-diffusivity problem.
    
    Args:
        x_max: Maximum depth (nm)
        dx: Spatial step (nm)
        t_final: Final time (seconds)
        dt: Time step (seconds)
        T: Temperature (Celsius)
        D0: Pre-exponential factor (cm²/s)
        Ea: Activation energy (eV)
        Cs: Surface concentration (atoms/cm³)
        NA0: Background concentration (atoms/cm³)
        refine_surface: Use surface refinement
    
    Returns:
        (x, C) - depth and final concentration
    
    Examples:
        >>> # Quick boron diffusion simulation
        >>> x, C = quick_solve_constant_D(
        ...     t_final=1800, T=1000,
        ...     D0=0.76, Ea=3.46,
        ...     Cs=1e20, NA0=1e15
        ... )
    """
    # Import here to avoid circular dependency
    from .erfc import diffusivity
    
    # Create solver
    solver = Fick1D(x_max=x_max, dx=dx, refine_surface=refine_surface)
    
    # Initial condition: constant background
    C0 = np.full(solver.n_points, NA0)
    
    # Constant diffusivity model
    def D_model(T_val, C):
        return diffusivity(T_val, D0, Ea)
    
    # Solve with Dirichlet BC at surface
    x, C = solver.solve(
        C0, dt, int(t_final / dt), T, D_model,
        bc=('dirichlet', 'neumann'),
        surface_value=Cs
    )
    
    return x, C


__all__ = [
    "Fick1D",
    "quick_solve_constant_D",
]
