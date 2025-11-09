"""
Numerical solver for Fick's 2nd law using Crank-Nicolson method.

Implements:
- 1D implicit solver for ∂C/∂t = ∂/∂x(D(C,T) ∂C/∂x)
- Adaptive grid refinement near surface
- Multiple boundary conditions (Dirichlet, Neumann, Robin)
- Optional concentration-dependent diffusivity
- Optional numba JIT acceleration

Will be implemented in Session 3.
"""

from typing import Optional, Callable, Tuple, Literal
import numpy as np
from numpy.typing import NDArray


class Fick1D:
    """
    1D finite difference solver for Fick's 2nd law.
    
    Solves: ∂C/∂t = ∂/∂x(D ∂C/∂x)
    
    Uses Crank-Nicolson (implicit) scheme for unconditional stability.
    
    Status: STUB - To be implemented in Session 3
    """
    
    def __init__(
        self,
        x_max: float = 1000.0,  # nm
        dx: float = 1.0,  # nm
        refine_surface: bool = True,
        use_numba: bool = False
    ):
        """
        Initialize the solver.
        
        Args:
            x_max: Maximum depth (nm)
            dx: Spatial step size (nm)
            refine_surface: Use finer grid near surface
            use_numba: Enable numba JIT compilation
        """
        self.x_max = x_max
        self.dx = dx
        self.refine_surface = refine_surface
        self.use_numba = use_numba
        
        # To be initialized in setup
        self.x = None
        self.C = None
        self.D_model = None
        
        raise NotImplementedError("Session 3: Fick1D solver initialization")
    
    def setup_grid(self):
        """Set up spatial grid with optional refinement."""
        raise NotImplementedError("Session 3: Grid setup")
    
    def solve(
        self,
        C0: NDArray[np.float64],
        dt: float,
        steps: int,
        T: float,
        D_model: Callable[[float, NDArray[np.float64]], float],
        bc: Tuple[str, str] = ('dirichlet', 'neumann'),
        surface_value: Optional[float] = None
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """
        Solve diffusion equation.
        
        Args:
            C0: Initial concentration profile
            dt: Time step (seconds)
            steps: Number of time steps
            T: Temperature (Celsius)
            D_model: Callable D(T, C) returning diffusivity
            bc: Boundary conditions (left, right)
            surface_value: Surface concentration for Dirichlet BC
        
        Returns:
            (x, C_final) - depth array and final concentration
        
        Status: STUB - To be implemented in Session 3
        """
        raise NotImplementedError("Session 3: Solve method")
    
    def _build_tridiagonal_system(
        self,
        C: NDArray[np.float64],
        D: NDArray[np.float64],
        dt: float
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Build tridiagonal matrix coefficients for Crank-Nicolson."""
        raise NotImplementedError("Session 3: Tridiagonal system")
    
    def _apply_boundary_conditions(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        d: NDArray[np.float64],
        bc: Tuple[str, str],
        surface_value: Optional[float] = None
    ):
        """Apply boundary conditions to the system."""
        raise NotImplementedError("Session 3: Boundary conditions")
    
    def _thomas_algorithm(
        self,
        a: NDArray[np.float64],
        b: NDArray[np.float64],
        c: NDArray[np.float64],
        d: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Solve tridiagonal system using Thomas algorithm."""
        raise NotImplementedError("Session 3: Thomas algorithm")
    
    def validate_convergence(
        self,
        C_analytical: NDArray[np.float64],
        C_numerical: NDArray[np.float64]
    ) -> float:
        """Calculate L2 error vs analytical solution."""
        raise NotImplementedError("Session 3: Convergence validation")


__all__ = ["Fick1D"]
