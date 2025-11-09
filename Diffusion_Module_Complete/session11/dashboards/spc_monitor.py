"""
SPC Monitor Dashboard

Interactive Streamlit dashboard for Statistical Process Control monitoring.
Visualizes Western Electric/Nelson rules, EWMA, and change point detection.

Usage:
    streamlit run spc_monitor.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add module to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from session11.spectra import diffusion_oxidation as do

# Page configuration
st.set_page_config(
    page_title="SPC Monitor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 SPC Monitor")
st.markdown("Statistical Process Control monitoring with rule detection")

# Sidebar for inputs
st.sidebar.header("Data Source")

data_source = st.sidebar.radio(
    "Data Source",
    ["Upload CSV", "Generate Synthetic"]
)

# Initialize data
data = None
timestamps = None

if data_source == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader(
        "Upload CSV (timestamp, value)",
        type=['csv']
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            data = df['value'].values
            timestamps = df['timestamp'].tolist()
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")

else:  # Generate synthetic
    st.sidebar.subheader("Synthetic Data Parameters")

    n_points = st.sidebar.slider(
        "Number of Points",
        min_value=50,
        max_value=500,
        value=200,
        step=10
    )

    base_mean = st.sidebar.number_input(
        "Base Mean",
        value=100.0,
        step=1.0
    )

    base_std = st.sidebar.number_input(
        "Base Std Dev",
        value=5.0,
        step=0.1,
        min_value=0.1
    )

    # Add process shift
    add_shift = st.sidebar.checkbox("Add Process Shift", value=False)

    if add_shift:
        shift_point = st.sidebar.slider(
            "Shift Point",
            min_value=10,
            max_value=n_points - 10,
            value=n_points // 2
        )

        shift_magnitude = st.sidebar.number_input(
            "Shift Magnitude (σ units)",
            value=2.0,
            step=0.1
        )
    else:
        shift_point = None
        shift_magnitude = 0.0

    # Add outlier
    add_outlier = st.sidebar.checkbox("Add Outlier", value=False)

    if add_outlier:
        outlier_point = st.sidebar.slider(
            "Outlier Point",
            min_value=10,
            max_value=n_points - 10,
            value=n_points // 4
        )

        outlier_magnitude = st.sidebar.number_input(
            "Outlier Magnitude (σ units)",
            value=4.0,
            step=0.1
        )
    else:
        outlier_point = None
        outlier_magnitude = 0.0

    # Generate data
    np.random.seed(42)
    data = np.random.normal(base_mean, base_std, n_points)

    if add_shift and shift_point:
        data[shift_point:] += shift_magnitude * base_std

    if add_outlier and outlier_point:
        data[outlier_point] += outlier_magnitude * base_std

    # Generate timestamps
    start_time = datetime.now() - timedelta(hours=n_points)
    timestamps = [start_time + timedelta(hours=i) for i in range(n_points)]

# Main content
if data is not None:
    # SPC Analysis Options
    st.sidebar.header("SPC Methods")

    enable_rules = st.sidebar.checkbox("Western Electric Rules", value=True)
    enable_ewma = st.sidebar.checkbox("EWMA", value=False)
    enable_changepoints = st.sidebar.checkbox("Change Points (BOCPD)", value=False)

    # Run analysis
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Control Chart")

        # Calculate control limits
        centerline = np.mean(data)
        sigma = np.std(data, ddof=1)
        ucl = centerline + 3 * sigma
        lcl = centerline - 3 * sigma

        # Create plot
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot data
        indices = np.arange(len(data))
        ax.plot(indices, data, 'bo-', markersize=4, linewidth=1, label='Data')

        # Control limits
        ax.axhline(centerline, color='green', linestyle='-', linewidth=2, label='CL')
        ax.axhline(ucl, color='red', linestyle='--', linewidth=1, label='UCL')
        ax.axhline(lcl, color='red', linestyle='--', linewidth=1, label='LCL')

        # ±1σ and ±2σ zones
        ax.axhline(centerline + sigma, color='orange', linestyle=':', alpha=0.5)
        ax.axhline(centerline - sigma, color='orange', linestyle=':', alpha=0.5)
        ax.axhline(centerline + 2*sigma, color='orange', linestyle=':', alpha=0.5)
        ax.axhline(centerline - 2*sigma, color='orange', linestyle=':', alpha=0.5)

        # Check rules if enabled
        if enable_rules:
            violations = do.spc.check_rules(data)

            if violations:
                # Plot violations
                violation_indices = [v['index'] for v in violations]
                violation_values = [data[v['index']] for v in violations]

                ax.plot(violation_indices, violation_values, 'rx',
                        markersize=12, markeredgewidth=2, label='Violations')

        # EWMA if enabled
        if enable_ewma:
            ewma_violations = do.spc.ewma_monitor(data)

            if ewma_violations:
                ewma_indices = [v['index'] for v in ewma_violations]
                ewma_values = [data[v['index']] for v in ewma_violations]

                ax.plot(ewma_indices, ewma_values, 'ms',
                        markersize=10, markerfacecolor='none',
                        markeredgewidth=2, label='EWMA Violations')

        # Change points if enabled
        if enable_changepoints:
            changepoints = do.spc.detect_changepoints(data, threshold=0.5)

            if changepoints:
                cp_indices = [cp['index'] for cp in changepoints]

                for idx in cp_indices:
                    ax.axvline(idx, color='purple', linestyle='--',
                               alpha=0.5, linewidth=2)

        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Value', fontsize=12)
        ax.set_title('Statistical Process Control Chart', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)

        st.pyplot(fig)

    with col2:
        st.subheader("Statistics")

        st.metric("Mean (CL)", f"{centerline:.2f}")
        st.metric("Std Dev (σ)", f"{sigma:.2f}")
        st.metric("UCL", f"{ucl:.2f}")
        st.metric("LCL", f"{lcl:.2f}")

        st.subheader("Violations")

        if enable_rules and 'violations' in locals() and violations:
            st.write(f"**Total: {len(violations)}**")

            # Group by rule
            rule_counts = {}
            for v in violations:
                rule = v['rule']
                rule_counts[rule] = rule_counts.get(rule, 0) + 1

            for rule, count in rule_counts.items():
                severity = violations[0]['severity']  # Get severity
                color = "🔴" if severity == "CRITICAL" else "🟡"
                st.write(f"{color} **{rule}:** {count}")

            # Show details in expander
            with st.expander("Violation Details"):
                for v in violations:
                    st.write(f"**Index {v['index']}:** {v['description']}")

        else:
            st.success("No rule violations detected")

        if enable_ewma and 'ewma_violations' in locals() and ewma_violations:
            st.write(f"**EWMA Violations: {len(ewma_violations)}**")

        if enable_changepoints and 'changepoints' in locals() and changepoints:
            st.write(f"**Change Points: {len(changepoints)}**")

            with st.expander("Change Point Details"):
                for cp in changepoints:
                    st.write(f"**Index {cp['index']}:** Prob = {cp['probability']:.2%}")

        st.subheader("Process Status")

        if enable_rules and 'violations' in locals():
            if len(violations) == 0:
                st.success("✅ Process In Control")
            elif len(violations) <= 2:
                st.warning("⚠️ Process Marginally Out of Control")
            else:
                st.error("🔴 Process Out of Control")

    # Detailed violation table
    if enable_rules and 'violations' in locals() and violations:
        st.subheader("Violation Summary Table")

        violation_df = pd.DataFrame(violations)
        st.dataframe(violation_df, use_container_width=True)

else:
    st.info("👈 Upload a CSV file or generate synthetic data to begin SPC monitoring")

    st.markdown("""
    ### CSV Format

    Your CSV should have two columns:
    ```
    timestamp,value
    2025-01-01 00:00:00,100.5
    2025-01-01 01:00:00,102.3
    2025-01-01 02:00:00,98.7
    ```

    ### Western Electric Rules

    1. **Rule 1:** One point beyond 3σ (CRITICAL)
    2. **Rule 2:** 2 of 3 points beyond 2σ (WARNING)
    3. **Rule 3:** 4 of 5 points beyond 1σ (WARNING)
    4. **Rule 4:** 8 consecutive points same side (WARNING)

    And more...
    """)

# Footer
st.markdown("---")
st.caption("SPECTRA Diffusion Module • Session 11 Dashboard • Powered by Streamlit")
