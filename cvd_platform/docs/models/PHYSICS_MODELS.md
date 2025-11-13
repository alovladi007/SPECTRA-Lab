# CVD Reactor Physics Models - Mathematical Derivations

## Table of Contents
1. [Gas Flow Dynamics](#gas-flow-dynamics)
2. [Mass Transport](#mass-transport)
3. [Reaction Kinetics](#reaction-kinetics)
4. [Heat Transfer](#heat-transfer)
5. [Deposition Rate Models](#deposition-rate-models)
6. [Model Validation](#model-validation)

## 1. Gas Flow Dynamics

### Navier-Stokes Equations

The gas flow in a CVD reactor is governed by the Navier-Stokes equations for compressible flow:

**Continuity Equation:**
```
∂ρ/∂t + ∇·(ρv) = 0
```

**Momentum Equation:**
```
ρ(∂v/∂t + v·∇v) = -∇P + μ∇²v + (ξ + μ/3)∇(∇·v) + ρg
```

where:
- ρ = gas density (kg/m³)
- v = velocity vector (m/s)
- P = pressure (Pa)
- μ = dynamic viscosity (Pa·s)
- ξ = bulk viscosity (Pa·s)
- g = gravitational acceleration (m/s²)

### Reynolds Number

The flow regime is characterized by the Reynolds number:

```
Re = ρVL/μ
```

For CVD processes:
- Re < 2300: Laminar flow (typical for most CVD)
- 2300 < Re < 4000: Transitional
- Re > 4000: Turbulent flow

**Typical CVD Reynolds Numbers:**
- LPCVD (Low Pressure): Re ≈ 1-100 (laminar)
- APCVD (Atmospheric Pressure): Re ≈ 100-1000 (laminar to transitional)
- Plasma-Enhanced CVD: Re varies widely

### Simplified Cylindrical Coordinates

For axisymmetric reactors (r, θ, z):

**Continuity:**
```
∂ρ/∂t + 1/r ∂(rρu)/∂r + ∂(ρw)/∂z = 0
```

**Radial Momentum:**
```
ρ(∂u/∂t + u∂u/∂r + w∂u/∂z) = -∂P/∂r + μ[∂²u/∂r² + 1/r ∂u/∂r - u/r² + ∂²u/∂z²]
```

**Axial Momentum:**
```
ρ(∂w/∂t + u∂w/∂r + w∂w/∂z) = -∂P/∂z + μ[∂²w/∂r² + 1/r ∂w/∂r + ∂²w/∂z²] + ρg
```

### Boundary Conditions

- **Inlet:** Specified velocity or mass flow rate
  ```
  v_inlet = Q/(πR²) where Q = volumetric flow rate
  ```

- **Outlet:** Zero pressure gradient
  ```
  ∂P/∂n = 0
  ```

- **Walls:** No-slip condition
  ```
  v_wall = 0
  ```

- **Centerline:** Symmetry
  ```
  ∂v/∂r = 0 at r = 0
  ```

## 2. Mass Transport

### Species Conservation Equation

For each chemical species i:

```
∂C_i/∂t + v·∇C_i = D_i∇²C_i + R_i
```

where:
- C_i = molar concentration of species i (mol/m³)
- D_i = diffusion coefficient (m²/s)
- R_i = reaction rate (mol/(m³·s))

### Binary Diffusion Coefficient

Chapman-Enskog theory for binary diffusion:

```
D_AB = 0.001858 T^(3/2) √(1/M_A + 1/M_B) / (P σ_AB² Ω_D)
```

where:
- T = temperature (K)
- M_A, M_B = molecular weights (g/mol)
- P = pressure (atm)
- σ_AB = collision diameter (Å)
- Ω_D = collision integral (dimensionless)

**Typical Values for CVD Gases:**

| Gas | D (cm²/s at STP) |
|-----|------------------|
| H₂ in N₂ | 0.674 |
| SiH₄ in N₂ | 0.18 |
| NH₃ in N₂ | 0.23 |

### Multicomponent Diffusion

For systems with >2 species, use Maxwell-Stefan equations:

```
∇x_i = Σ(j≠i) [(x_i N_j - x_j N_i)/(C_T D_ij)]
```

where:
- x_i = mole fraction of species i
- N_i = molar flux of species i
- C_T = total molar concentration

### Knudsen Diffusion

At low pressure, molecular mean free path becomes comparable to pore size:

```
D_K = (d_pore/3)√(8RT/πM)
```

Effective diffusion coefficient:

```
1/D_eff = 1/D_bulk + 1/D_K
```

## 3. Reaction Kinetics

### Arrhenius Equation

Temperature dependence of reaction rate constant:

```
k(T) = k₀ exp(-E_a/RT)
```

where:
- k₀ = pre-exponential factor (units vary)
- E_a = activation energy (J/mol)
- R = gas constant = 8.314 J/(mol·K)
- T = temperature (K)

### Surface Reaction Rate

For a heterogeneous (surface) reaction:

```
R_surf = k_s C_surface^n
```

where:
- k_s = surface reaction rate constant (units depend on n)
- C_surface = concentration at surface
- n = reaction order

### Silicon Deposition from Silane

**Reaction:**
```
SiH₄(g) → Si(s) + 2H₂(g)
```

**Rate Expression:**
```
r_Si = k₀ exp(-E_a/RT) [SiH₄]^n

Typical values:
k₀ = 1.0 × 10⁸ s⁻¹
E_a = 170 kJ/mol
n = 1.0 (first order)
```

**Temperature Dependence:**

| T (°C) | k (s⁻¹) | Growth Rate (nm/min) |
|--------|---------|---------------------|
| 600 | 0.05 | 5 |
| 700 | 0.5 | 50 |
| 800 | 3.0 | 300 |
| 900 | 12.0 | 1200 |

### Silicon Nitride Deposition

**Reaction:**
```
3SiH₄ + 4NH₃ → Si₃N₄ + 12H₂
```

**Rate Expression:**
```
r_SiN = k₀ exp(-E_a/RT) [SiH₄]^a [NH₃]^b

Typical values:
k₀ = 5.0 × 10⁷ (sccm)^(-(a+b)) s⁻¹
E_a = 150 kJ/mol
a = 1.0, b = 0.5
```

### Langmuir-Hinshelwood Mechanism

For complex surface reactions with adsorption/desorption:

```
r = (k_ads K_ads C_gas) / (1 + K_ads C_gas)

where:
k_ads = adsorption rate constant
K_ads = adsorption equilibrium constant = exp(ΔH_ads/RT)
```

## 4. Heat Transfer

### Energy Equation

```
ρC_p(∂T/∂t + v·∇T) = k∇²T + Q_rxn + Q_rad
```

where:
- C_p = specific heat capacity (J/(kg·K))
- k = thermal conductivity (W/(m·K))
- Q_rxn = heat of reaction (W/m³)
- Q_rad = radiative heat transfer (W/m³)

### Radiative Heat Transfer

Stefan-Boltzmann law for radiation:

```
q_rad = ε σ (T_hot⁴ - T_cold⁴)
```

where:
- ε = emissivity (0-1)
- σ = Stefan-Boltzmann constant = 5.67 × 10⁻⁸ W/(m²·K⁴)

**Typical Emissivities:**
- Silicon wafer: ε ≈ 0.6-0.7
- Quartz: ε ≈ 0.9
- Graphite susceptor: ε ≈ 0.8

### Convective Heat Transfer

Newton's law of cooling:

```
q_conv = h(T_surface - T_∞)
```

Nusselt number correlation for forced convection:

```
Nu = hL/k = C Re^m Pr^n
```

where:
- Nu = Nusselt number
- Re = Reynolds number
- Pr = Prandtl number = μC_p/k

**For laminar flow over flat plate:**
```
Nu = 0.664 Re^0.5 Pr^0.33
```

### Multi-Zone Heater Control

State-space representation:

```
dT/dt = A·T + B·u

where:
T = [T₁, T₂, ..., T_n]ᵀ (zone temperatures)
u = [P₁, P₂, ..., P_n]ᵀ (heater powers)

A_ij = -α_i (self-cooling) for i=j
     = β_ij (inter-zone coupling) for i≠j

B_ii = γ_i (power-to-temperature gain)
```

## 5. Deposition Rate Models

### General Growth Rate Equation

```
Growth Rate = (MW/ρ_film) × Flux

where:
MW = molecular weight of film material (kg/mol)
ρ_film = film density (kg/m³)
Flux = deposition flux (mol/(m²·s))
```

### Flux Components

Total flux to surface:

```
J_total = J_diffusion + J_reaction

J_diffusion = D ∂C/∂z|_surface (mass transport limited)
J_reaction = k_s C_surface^n (reaction limited)
```

### Limiting Regimes

**1. Mass Transport Limited (Low Temperature):**
```
Rate ∝ D·C_gas (independent of T)
```

**2. Reaction Limited (High Temperature):**
```
Rate ∝ k(T)·C_surface ∝ exp(-E_a/RT)
```

**3. Mixed Regime (Intermediate Temperature):**
```
1/Rate_total = 1/Rate_diffusion + 1/Rate_reaction
```

### Film Thickness Uniformity

Radial non-uniformity due to depletion:

```
t(r) = t_center × exp(-r²/L²)

where:
L = characteristic length = √(D·h_chamber/k_s)

Uniformity = (t_max - t_min)/t_avg × 100%
```

### Rotation Effect

Wafer rotation improves uniformity:

```
U = U₀/(1 + ω·τ_mix)

where:
ω = rotation speed (rad/s)
τ_mix = mixing time constant
```

## 6. Model Validation

### Experimental Validation

**1. Thickness Measurements:**
- Ellipsometry: ±0.5 nm accuracy
- Profilometry: ±1 nm
- XRF: ±2% for composition

**2. Temperature Measurements:**
- Thermocouples: ±2°C
- Pyrometry: ±5°C
- Infrared imaging: ±10°C

**3. Pressure & Flow:**
- Capacitance manometer: ±0.25% reading
- MFC: ±1% full scale

### Model-to-Hardware Correlation

**Typical Agreement:**
- Thickness: Within 5% of measured
- Uniformity: Within 10% of measured
- Temperature: Within 5°C
- Deposition rate: Within 10%

### Sensitivity Analysis

**Key Parameters:**

| Parameter | Thickness Sensitivity | Uniformity Sensitivity |
|-----------|----------------------|------------------------|
| Temperature | +10°C → +30% rate | +10°C → -5% uniformity |
| Pressure | +10% → -8% rate | +10% → +3% uniformity |
| Flow rate | +10% → +5% rate | +10% → +8% uniformity |
| Rotation | +50% → no rate change | +50% → +20% uniformity |

### Uncertainty Quantification

Monte Carlo analysis with parameter variations:

```
σ_thickness² = Σ(∂t/∂p_i)² σ_p_i²

where:
p_i = input parameters (T, P, flows, etc.)
σ_p_i = uncertainty in parameter i
```

**Typical Uncertainty Budget:**
- Temperature control: ±2°C → ±6% thickness
- Pressure control: ±0.1 Torr → ±2% thickness
- Flow control: ±2% → ±2% thickness
- Time control: ±1 s → ±0.8% thickness

**Total RSS Uncertainty:** ±6.7% thickness

---

## References

1. Kleijn, C.R. (1991). "Transport Phenomena in Chemical Vapor Deposition Reactors." *PhD Thesis, TU Delft*.

2. Roenigk, K.F. & Jensen, K.F. (1987). "Low Pressure CVD of Silicon Nitride." *J. Electrochem. Soc.*, 134(7), 1777-1785.

3. Coltrin, M.E., Kee, R.J., & Evans, G.H. (1989). "A Mathematical Model of Silicon CVD." *J. Electrochem. Soc.*, 136(3), 819-829.

4. Bird, R.B., Stewart, W.E., & Lightfoot, E.N. (2007). *Transport Phenomena*, 2nd Ed. Wiley.

5. Mahajan, R.L. (1996). "Transport Phenomena in CVD Systems." *Advances in Heat Transfer*, Vol. 28.
