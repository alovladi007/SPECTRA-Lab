"""
Oxide Thickness Planner Dashboard

Interactive Streamlit dashboard for planning thermal oxidation processes.
Supports forward (thickness vs time) and inverse (time to target) calculations.

Usage:
    streamlit run oxide_planner.py
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
    page_title="Oxide Thickness Planner",
    page_icon="🔬",
    layout="wide"
)

st.title("🔬 Oxide Thickness Planner")
st.markdown("Plan thermal oxidation processes using Deal-Grove model")

# Sidebar for inputs
st.sidebar.header("Parameters")

# Calculation mode
calc_mode = st.sidebar.radio(
    "Calculation Mode",
    ["Forward (thickness vs time)", "Inverse (time to target)"],
    help="Choose forward or inverse problem"
)

# Ambient selection
ambient = st.sidebar.selectbox(
    "Ambient",
    ["dry", "wet"],
    help="Select oxidation ambient (dry O₂ or wet H₂O)"
)

# Temperature slider
temp_c = st.sidebar.slider(
    "Temperature (°C)",
    min_value=800,
    max_value=1200,
    value=1000,
    step=10,
    help="Oxidation temperature"
)

# Pressure
pressure = st.sidebar.slider(
    "Pressure (atm)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1,
    help="Partial pressure of oxidant"
)

# Initial oxide
initial_thickness_nm = st.sidebar.number_input(
    "Initial Oxide (nm)",
    min_value=0.0,
    max_value=100.0,
    value=5.0,
    step=1.0,
    help="Initial oxide thickness"
)

# Mode-specific inputs
if calc_mode == "Forward (thickness vs time)":
    max_time_hr = st.sidebar.slider(
        "Maximum Time (hours)",
        min_value=0.1,
        max_value=24.0,
        value=10.0,
        step=0.1,
        help="Maximum time for plot"
    )
    target_thickness_nm = None
else:  # Inverse mode
    target_thickness_nm = st.sidebar.number_input(
        "Target Thickness (nm)",
        min_value=initial_thickness_nm + 1.0,
        max_value=1000.0,
        value=100.0,
        step=1.0,
        help="Target oxide thickness"
    )
    max_time_hr = None

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    if calc_mode == "Forward (thickness vs time)":
        st.subheader("Thickness vs Time")

        try:
            # Calculate thickness curve
            time_points = np.linspace(0.1, max_time_hr, 100)
            thickness_points = np.array([
                do.oxidation.deal_grove_thickness(
                    temp_c=temp_c,
                    time_hr=t,
                    ambient=ambient,
                    pressure=pressure,
                    initial_thickness_nm=initial_thickness_nm
                )
                for t in time_points
            ])

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(time_points, thickness_points, 'b-', linewidth=2)
            ax.axhline(initial_thickness_nm, color='gray', linestyle='--',
                       alpha=0.5, label=f'Initial: {initial_thickness_nm:.1f} nm')

            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Oxide Thickness (nm)', fontsize=12)
            ax.set_title(f'{ambient.capitalize()} Oxidation @ {temp_c}°C',
                         fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            st.pyplot(fig)

            # Calculate growth rate at final point
            final_rate = do.oxidation.growth_rate(
                thickness_nm=thickness_points[-1],
                temp_c=temp_c,
                ambient=ambient,
                pressure=pressure
            )

        except Exception as e:
            st.error(f"Error calculating thickness: {e}")

    else:  # Inverse mode
        st.subheader("Time to Target")

        try:
            # Calculate required time
            required_time_hr = do.oxidation.time_to_target(
                target_thickness_nm=target_thickness_nm,
                temp_c=temp_c,
                ambient=ambient,
                pressure=pressure,
                initial_thickness_nm=initial_thickness_nm
            )

            # Create growth curve up to target
            time_points = np.linspace(0.1, required_time_hr * 1.2, 100)
            thickness_points = np.array([
                do.oxidation.deal_grove_thickness(
                    temp_c=temp_c,
                    time_hr=t,
                    ambient=ambient,
                    pressure=pressure,
                    initial_thickness_nm=initial_thickness_nm
                )
                for t in time_points
            ])

            # Create plot
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(time_points, thickness_points, 'b-', linewidth=2)
            ax.axhline(target_thickness_nm, color='r', linestyle='--',
                       alpha=0.7, label=f'Target: {target_thickness_nm:.1f} nm')
            ax.axvline(required_time_hr, color='r', linestyle='--',
                       alpha=0.7, label=f'Required Time: {required_time_hr:.2f} hr')
            ax.plot(required_time_hr, target_thickness_nm, 'ro', markersize=10)

            ax.set_xlabel('Time (hours)', fontsize=12)
            ax.set_ylabel('Oxide Thickness (nm)', fontsize=12)
            ax.set_title(f'{ambient.capitalize()} Oxidation @ {temp_c}°C',
                         fontsize=14)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=10)

            st.pyplot(fig)

            # Calculate growth rate at target
            target_rate = do.oxidation.growth_rate(
                thickness_nm=target_thickness_nm,
                temp_c=temp_c,
                ambient=ambient,
                pressure=pressure
            )

        except Exception as e:
            st.error(f"Error calculating time: {e}")

with col2:
    st.subheader("Results")

    if calc_mode == "Forward (thickness vs time)":
        if 'thickness_points' in locals():
            final_thickness = thickness_points[-1]
            growth = final_thickness - initial_thickness_nm

            st.metric("Final Thickness", f"{final_thickness:.1f} nm")
            st.metric("Growth", f"{growth:.1f} nm")
            st.metric("Growth Rate", f"{final_rate:.2f} nm/hr")

    else:  # Inverse mode
        if 'required_time_hr' in locals():
            growth = target_thickness_nm - initial_thickness_nm

            st.metric("Required Time", f"{required_time_hr:.2f} hours")
            st.metric("Growth", f"{growth:.1f} nm")
            st.metric("Growth Rate @ Target", f"{target_rate:.2f} nm/hr")

    st.subheader("Deal-Grove Model")

    st.markdown(f"""
    **Ambient:** {ambient.capitalize()}
    **Temperature:** {temp_c}°C
    **Pressure:** {pressure} atm
    **Initial Oxide:** {initial_thickness_nm:.1f} nm

    **Key Equation:**
    ```
    x² + Ax = B(t + τ)
    ```

    **Growth Regimes:**
    - **Linear:** Thin oxide (x << A)
      - Rate ∝ B/A
    - **Parabolic:** Thick oxide (x >> A)
      - Rate ∝ √(B/t)

    **Wet vs Dry:**
    - Wet: Faster growth (H₂O)
    - Dry: Slower, higher quality (O₂)
    """)

    # Show temperature comparison
    if st.checkbox("Show Temperature Comparison", value=False):
        st.subheader("Temperature Effect")

        temps = [900, 1000, 1100]
        times_at_temps = []

        if calc_mode == "Inverse (time to target)":
            for T in temps:
                t_req = do.oxidation.time_to_target(
                    target_thickness_nm=target_thickness_nm,
                    temp_c=T,
                    ambient=ambient,
                    pressure=pressure,
                    initial_thickness_nm=initial_thickness_nm
                )
                times_at_temps.append(t_req)

            for T, t in zip(temps, times_at_temps):
                st.write(f"**{T}°C:** {t:.2f} hours")

# Footer
st.markdown("---")
st.caption("SPECTRA Diffusion Module • Session 11 Dashboard • Powered by Streamlit")
