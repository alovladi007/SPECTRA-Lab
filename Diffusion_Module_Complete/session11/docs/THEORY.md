# Diffusion & Oxidation Module - Theory & Physics

**Mathematical Background and Physical Models**

---

## Table of Contents

1. [Diffusion Theory](#diffusion-theory)
2. [Thermal Oxidation](#thermal-oxidation)
3. [Segregation & Moving Boundaries](#segregation--moving-boundaries)
4. [Statistical Process Control](#statistical-process-control)
5. [Numerical Methods](#numerical-methods)
6. [References](#references)

---

## Diffusion Theory

### Fick's Laws

**First Law (Flux):**
```
J = -D ∇C
```
Where:
- J: Diffusion flux (atoms/cm²·s)
- D: Diffusivity (cm²/s)
- C: Concentration (atoms/cm³)

**Second Law (Conservation):**
```
∂C/∂t = ∇·(D∇C)
```

For 1D with constant D:
```
∂C/∂t = D ∂²C/∂x²
```

### Temperature-Dependent Diffusivity

**Arrhenius Relation:**
```
D(T) = D₀ exp(-Eₐ / kT)
```

Where:
- D₀: Pre-exponential factor (cm²/s)
- Eₐ: Activation energy (eV)
- k: Boltzmann constant (8.617×10⁻⁵ eV/K)
- T: Temperature (K)

**Common Dopants in Silicon:**

| Dopant | D₀ (cm²/s) | Eₐ (eV) |
|--------|------------|---------|
| Boron (B) | 0.76 | 3.69 |
| Phosphorus (P) | 3.85 | 3.66 |
| Arsenic (As) | 0.066 | 3.44 |
| Antimony (Sb) | 0.214 | 3.65 |

### Analytical Solutions (ERFC)

####  Constant Source

**Boundary Conditions:**
- C(0,t) = Cs (surface concentration)
- C(∞,t) = Cb (background)
- C(x,0) = Cb

**Solution:**
```
C(x,t) = Cb + (Cs - Cb) erfc(x / 2√(Dt))
```

Where erfc is the complementary error function:
```
erfc(z) = (2/√π) ∫[z,∞] exp(-u²) du
```

#### Limited Source (Gaussian)

**Boundary Conditions:**
- Total dose Q fixed
- C(x,0) = Q δ(x)
- C(∞,t) = Cb

**Solution:**
```
C(x,t) = Cb + (Q / √(πDt)) exp(-x² / 4Dt)
```

### Junction Depth

**Definition:**
Junction depth xⱼ where C(xⱼ) = N_background

**For constant source:**
```
xⱼ ≈ 2√(Dt) erfc⁻¹((Cs - Cb) / Cb)
```

**Scaling:**
Junction depth scales as √(Dt):
- Double temperature → ~4× deeper (exponential D)
- Double time → ~1.4× deeper (√t scaling)

---

## Thermal Oxidation

### Deal-Grove Model

**Governing Equation:**
```
x²ₒₓ + A·xₒₓ = B(t + τ)
```

Where:
- xₒₓ: Oxide thickness
- A: Linear rate constant (proportional to B/k)
- B: Parabolic rate constant
- τ: Time shift for initial oxide

**Rate Constants:**

Temperature-dependent via Arrhenius:
```
B = B₀ exp(-E_B / kT)
B/A = (B/A)₀ exp(-E_{B/A} / kT)
```

**Dry Oxidation (O₂):**
- B₀ = 7.72×10⁵ μm²/hr
- E_B = 2.0 eV
- (B/A)₀ = 3.71×10⁶ μm/hr
- E_{B/A} = 1.96 eV

**Wet Oxidation (H₂O):**
- B₀ = 3.86×10⁸ μm²/hr
- E_B = 0.78 eV
- (B/A)₀ = 6.23×10⁸ μm/hr
- E_{B/A} = 2.05 eV

### Growth Regimes

**Linear Regime (thin oxide, x << A):**
```
dx/dt ≈ B/A = constant
```
Reaction-limited growth

**Parabolic Regime (thick oxide, x >> A):**
```
dx/dt ≈ B / 2x
```
Diffusion-limited growth (oxidant transport)

**Transition:** Occurs at x ≈ A

### Inverse Problem (Time to Target)

From Deal-Grove equation:
```
t = (x²ₜₐᵣgₑₜ + A·xₜₐᵣgₑₜ)/B - τ
```

---

## Segregation & Moving Boundaries

### Segregation Coefficient

**Definition:**
```
m = C_oxide / C_silicon  (at interface)
```

**Common Values:**
- Arsenic (As): m ≈ 0.02 (depletes in oxide)
- Phosphorus (P): m ≈ 0.1
- Boron (B): m ≈ 0.3
- Antimony (Sb): m ≈ 0.01

### Pile-Up/Depletion

As oxide grows, dopants redistribute:
- **m < 1:** Pile-up in silicon (As, P, B, Sb)
- **m > 1:** Depletion in silicon (rare)

**Peak Concentration:**
```
C_peak / C_initial ≈ 1/m  (for m << 1)
```

### Moving Boundary

As SiO₂ grows, Si/SiO₂ interface moves into silicon:
```
x_Si_consumed = 0.44 · x_oxide
```
(Due to volume expansion: 1 Si → 2.2 SiO₂)

**Coupled System:**
- Oxidation (Deal-Grove)
- Diffusion (Fick's law)
- Segregation (boundary condition)
- Interface motion (coordinate transformation)

---

## Statistical Process Control

### Western Electric Rules

**Control Limits:**
- UCL = μ + 3σ
- CL = μ
- LCL = μ - 3σ

**Key Rules:**
1. **Rule 1:** 1 point > 3σ (CRITICAL)
2. **Rule 2:** 2/3 points > 2σ same side (WARNING)
3. **Rule 3:** 4/5 points > 1σ same side (WARNING)
4. **Rule 4:** 8 consecutive same side (WARNING)

### EWMA (Exponentially Weighted Moving Average)

**Recursive Formula:**
```
z_t = λ·x_t + (1-λ)·z_{t-1}
```

Where:
- λ: Smoothing parameter (0 < λ ≤ 1)
- x_t: Current observation
- z_t: EWMA statistic

**Control Limits:**
```
UCL/LCL = μ ± L·σ√(λ/(2-λ)[1-(1-λ)^{2t}])
```

**Advantages:**
- Sensitive to small shifts
- Uses historical data
- Tunable via λ

### CUSUM (Cumulative Sum)

**Tabular Method:**
```
C⁺ᵢ = max(0, xᵢ - (μ + K) + C⁺ᵢ₋₁)
C⁻ᵢ = max(0, (μ - K) - xᵢ + C⁻ᵢ₋₁)
```

Where:
- K: Reference value (typically 0.5σ)
- C⁺: Upper CUSUM
- C⁻: Lower CUSUM

**Alarm:** When C⁺ or C⁻ > h (decision interval)

### BOCPD (Bayesian Online Change Point Detection)

**Posterior:**
```
P(r_t | x_{1:t}) ∝ P(x_t | r_t, x_{1:t-1}) P(r_t | x_{1:t-1})
```

Where:
- r_t: Run length (time since last change point)
- x_t: Current observation

**Hazard Function:**
Constant hazard:
```
H(τ) = 1/λ
```

**Change Point Probability:**
```
P(change at t) = P(r_t = 0 | x_{1:t})
```

---

## Numerical Methods

### Finite Difference (Crank-Nicolson)

**Discretization:**
```
∂C/∂t ≈ (Cⁿ⁺¹ - Cⁿ)/Δt
∂²C/∂x² ≈ (Cⁿ⁺¹ᵢ₊₁ - 2Cⁿ⁺¹ᵢ + Cⁿ⁺¹ᵢ₋₁)/Δx²
```

**Crank-Nicolson (θ-method, θ=0.5):**
```
Cⁿ⁺¹ - Cⁿ = (Δt/2Δx²)D[(∇²C)ⁿ⁺¹ + (∇²C)ⁿ]
```

**Advantages:**
- Unconditionally stable
- Second-order accurate in time
- Implicit (requires linear solve)

**Thomas Algorithm:**
Solves tridiagonal system Ax = b in O(n) time

### Boundary Conditions

**Dirichlet:**
```
C(0,t) = C_s  (fixed concentration)
```

**Neumann:**
```
∂C/∂x|_{x=0} = J_0  (fixed flux)
```

For zero flux:
```
∂C/∂x|_{x=0} = 0
```

---

## References

### Diffusion

1. Fair, R. B. & Tsai, J. C. C. (1977). "A Quantitative Model for the Diffusion of Phosphorus in Silicon and the Emitter Dip Effect." *J. Electrochem. Soc.*, 124(7), 1107-1118.

2. Plummer, J. D., Deal, M. D., & Griffin, P. B. (2000). *Silicon VLSI Technology: Fundamentals, Practice and Modeling*. Prentice Hall.

### Oxidation

3. Deal, B. E. & Grove, A. S. (1965). "General Relationship for the Thermal Oxidation of Silicon." *J. Appl. Phys.*, 36(12), 3770-3778.

4. Massoud, H. Z. (1988). "Thermal Oxidation of Silicon in Dry Oxygen: Accurate Determination of the Kinetic Rate Constants." *J. Electrochem. Soc.*, 135(8), 1939-1944.

### SPC

5. Montgomery, D. C. (2012). *Statistical Quality Control: A Modern Introduction* (7th ed.). Wiley.

6. Adams, B. M. & Tseng, I. (1998). "Robustness of Forecast-based Monitoring Schemes." *J. Quality Technology*, 30(4), 328-339.

7. Ryan, A. G. & Woodall, W. H. (2011). "Bayesian Methods for Online Change Point Detection." *Sequential Analysis*, 30(1), 1-16.

### Numerical Methods

8. Press, W. H., et al. (2007). *Numerical Recipes: The Art of Scientific Computing* (3rd ed.). Cambridge University Press.

9. LeVeque, R. J. (2007). *Finite Difference Methods for Ordinary and Partial Differential Equations*. SIAM.

---

**Next:** See [WORKFLOW.md](WORKFLOW.md) for practical semiconductor manufacturing workflows.
