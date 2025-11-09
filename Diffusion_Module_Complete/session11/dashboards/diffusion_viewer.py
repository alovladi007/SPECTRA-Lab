"""
Diffusion Profile Viewer Dashboard

Interactive Streamlit dashboard for visualizing diffusion profiles.
Allows comparison between ERFC analytical and numerical FD solutions.

Usage:
    streamlit run diffusion_viewer.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from session11.spectra import diffusion_oxidation as do

# Page configuration
st.set_page_config(
    page_title="Diffusion Profile Viewer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Diffusion Profile Viewer")
st.markdown("Interactive visualization of dopant diffusion profiles")

# Sidebar for inputs
st.sidebar.header("Parameters")

# Dopant selection
dopant = st.sidebar.selectbox(
    "Dopant",
    ["boron", "phosphorus", "arsenic", "antimony"],
    help="Select dopant species"
)

# Temperature slider
temp_c = st.sidebar.slider(
    "Temperature (°C)",
    min_value=800,
    max_value=1200,
    value=1000,
    step=10,
    help="Diffusion temperature"
)

# Time slider
time_min = st.sidebar.slider(
    "Time (minutes)",
    min_value=1,
    max_value=180,
    value=30,
    step=1,
    help="Diffusion time"
)

# Method selection
method = st.sidebar.radio(
    "Diffusion Method",
    ["constant_source", "limited_source"],
    help="Select diffusion profile type"
)

# Method-specific parameters
if method == "constant_source":
    surface_conc_exp = st.sidebar.slider(
        "Surface Concentration (log₁₀ cm⁻³)",
        min_value=15.0,
        max_value=21.0,
        value=20.0,
        step=0.1,
        help="Surface concentration (logarithmic scale)"
    )
    surface_conc = 10**surface_conc_exp
    dose = None
else:  # limited_source
    dose_exp = st.sidebar.slider(
        "Dose (log₁₀ cm⁻²)",
        min_value=12.0,
        max_value=16.0,
        value=14.0,
        step=0.1,
        help="Implanted dose (logarithmic scale)"
    )
    dose = 10**dose_exp
    surface_conc = 1e20  # Dummy value

# Background doping
background_exp = st.sidebar.slider(
    "Background (log₁₀ cm⁻³)",
    min_value=13.0,
    max_value=18.0,
    value=15.0,
    step=0.1,
    help="Background doping concentration"
)
background = 10**background_exp

# Solver comparison checkbox
compare_solvers = st.sidebar.checkbox(
    "Compare ERFC vs Numerical",
    value=False,
    help="Show both analytical and numerical solutions"
)

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Concentration Profile")

    # Calculate profiles
    depth_nm = np.linspace(0, 500, 200)

    try:
        # ERFC analytical solution
        x_erfc, C_erfc = do.diffusion.erfc_profile(
            dopant=dopant,
            temp_c=temp_c,
            time_min=time_min,
            method=method,
            surface_conc=surface_conc,
            dose=dose,
            background=background,
            depth_nm=depth_nm
        )

        # Calculate junction depth
        xj_erfc = do.diffusion.junction_depth(C_erfc, x_erfc, background)

        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot ERFC
        ax.semilogy(x_erfc, C_erfc, 'b-', linewidth=2, label='ERFC Analytical')
        ax.axvline(xj_erfc, color='b', linestyle='--', alpha=0.5,
                   label=f'xⱼ (ERFC) = {xj_erfc:.1f} nm')

        # Numerical solution if requested
        if compare_solvers:
            # Prepare initial condition
            C0 = np.full(len(depth_nm), background)

            if method == "constant_source":
                bc_left = ('dirichlet', surface_conc)
            else:
                bc_left = ('neumann', 0.0)

            x_num, C_num = do.diffusion.numerical_solve(
                initial_conc=C0,
                time_sec=time_min * 60,
                temp_c=temp_c,
                dopant=dopant,
                nx=len(depth_nm),
                L_nm=500,
                bc_left=bc_left,
                bc_right=('neumann', 0.0)
            )

            xj_num = do.diffusion.junction_depth(C_num, x_num, background)

            ax.semilogy(x_num, C_num, 'r--', linewidth=2, label='Numerical FD')
            ax.axvline(xj_num, color='r', linestyle='--', alpha=0.5,
                       label=f'xⱼ (Numerical) = {xj_num:.1f} nm')

        # Background line
        ax.axhline(background, color='gray', linestyle=':', alpha=0.5, label='Background')

        ax.set_xlabel('Depth (nm)', fontsize=12)
        ax.set_ylabel('Concentration (cm⁻³)', fontsize=12)
        ax.set_title(f'{dopant.capitalize()} Diffusion @ {temp_c}°C, {time_min} min',
                     fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error calculating profile: {e}")

with col2:
    st.subheader("Results")

    if 'xj_erfc' in locals():
        st.metric("Junction Depth (ERFC)", f"{xj_erfc:.1f} nm")

        if compare_solvers and 'xj_num' in locals():
            error_pct = abs(xj_erfc - xj_num) / xj_erfc * 100
            st.metric("Junction Depth (Numerical)", f"{xj_num:.1f} nm",
                      delta=f"{error_pct:.1f}% error")

        st.metric("Surface Concentration", f"{surface_conc:.2e} cm⁻³")
        st.metric("Background", f"{background:.2e} cm⁻³")

        if method == "limited_source" and dose:
            st.metric("Dose", f"{dose:.2e} cm⁻²")

    st.subheader("Physics")

    st.markdown(f"""
    **Dopant:** {dopant.capitalize()}
    **Temperature:** {temp_c}°C
    **Time:** {time_min} minutes

    **Model:** {"Constant Source" if method == "constant_source" else "Limited Source"}

    **Key Equations:**
    - Diffusivity: D = D₀ exp(-Eₐ/kT)
    - Junction: C(xⱼ) = N_background

    **Physics Insights:**
    - Higher T → deeper junction
    - Longer t → ∝ √(Dt) scaling
    - Heavier dopants diffuse slower
    """)

# Footer
st.markdown("---")
st.caption("SPECTRA Diffusion Module • Session 11 Dashboard • Powered by Streamlit")
