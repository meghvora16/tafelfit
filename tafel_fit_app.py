import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

st.set_page_config(page_title="Strict Tafel Analysis", layout="wide")
st.title("Strict Tafel Analysis (Decade Validation)")

st.markdown("""
**The 1-Decade Rule:**
Standard electrochemical theory requires at least **1 full decade** (10x change in current) of linearity for a valid Tafel fit.
* **Anodic Active-Passive:** Often fails this rule because the peak ($E_{pp}$) cuts the linear region short.
* **This Tool:** Will calculate the exact number of decades and **warn you** if your data is insufficient.
""")

# --- UTILS ---
def smooth_curve(y, window_length=15, polyorder=3):
    if len(y) < window_length: return y
    return savgol_filter(y, window_length, polyorder)

def find_ecorr(E, i):
    idx = np.argmin(np.abs(i))
    return E[idx], i[idx], idx

# --- MAIN APP ---
data_file = st.file_uploader("Upload LSV/Polarization data (CSV/Excel)", type=["csv", "xlsx", "xls"])

if data_file is not None:
    # Load Data
    if data_file.name.endswith(".csv"): df = pd.read_csv(data_file)
    else: df = pd.read_excel(data_file)
    
    # Columns
    c1, c2, c3, c4 = st.columns(4)
    col_E = c1.selectbox("Potential column", df.columns, index=0)
    col_I = c2.selectbox("Current column", df.columns, index=1)
    pot_units = c3.selectbox("Potential units", ["V", "mV"], 0)
    cur_units = c4.selectbox("Current units", ["A", "mA", "uA", "nA"], 0)
    area = st.number_input("Electrode Area (cm²)", value=1.0)

    # Process Data
    E_raw = df[col_E].astype(float).to_numpy()
    I_raw = df[col_I].astype(float).to_numpy()
    if pot_units == "mV": E_raw /= 1000.0
    unit_mult = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9}
    I_norm = I_raw * unit_mult[cur_units]
    i_dens = I_norm / area

    # Sort
    sort_idx = np.argsort(E_raw)
    E = E_raw[sort_idx]
    i_meas = i_dens[sort_idx]
    
    # Find Ecorr
    E_corr, i_corr_raw, idx_corr = find_ecorr(E, i_meas)

    # --- SIDEBAR CONTROLS ---
    st.sidebar.header("Fit Settings")
    
    # Anodic Settings
    st.sidebar.subheader("🔴 Anodic Fit Range")
    a_start_def = 0.02 
    a_end_def = 0.10 # Default smaller for active-passive
    
    anodic_start = st.sidebar.slider("Start (V > Ecorr)", 0.0, 0.5, a_start_def, 0.005, format="%.3f")
    anodic_end = st.sidebar.slider("End (V > Ecorr)", 0.0, 0.5, a_end_def, 0.005, format="%.3f")
    
    # Cathodic Settings
    st.sidebar.subheader("🔵 Cathodic Fit Range")
    c_start_def = 0.02
    c_end_def = 0.15
    
    cathodic_start = st.sidebar.slider("Start (V < Ecorr)", 0.0, 0.5, c_start_def, 0.005, format="%.3f")
    cathodic_end = st.sidebar.slider("End (V < Ecorr)", 0.0, 0.5, c_end_def, 0.005, format="%.3f")

    # --- ANALYSIS ---
    
    # 1. Anodic Peak Detection
    anodic_mask = E > E_corr
    E_anod = E[anodic_mask]
    i_anod = i_meas[anodic_mask]
    
    E_pp, i_crit = None, None
    if len(E_anod) > 10:
        i_smooth = smooth_curve(i_anod, 21)
        peaks = argrelextrema(i_smooth, np.greater)[0]
        valid_peaks = [p for p in peaks if i_smooth[p] > 0]
        if valid_peaks:
            best_p = valid_peaks[np.argmax(i_smooth[valid_peaks])]
            E_pp = E_anod[best_p]
            i_crit = i_anod[best_p]

    # 2. MANUAL TAFEL FITS & DECADE CHECKS
    
    # Anodic Fit
    mask_a_fit = (E > (E_corr + anodic_start)) & (E < (E_corr + anodic_end))
    b_a, i_corr_a, decades_a = None, None, 0.0
    
    if np.sum(mask_a_fit) > 2:
        i_fit = i_meas[mask_a_fit]
        if np.all(i_fit > 0):
            slope_a, int_a, r_a, _, _ = linregress(np.log10(i_fit), E[mask_a_fit])
            b_a = slope_a
            i_corr_a = 10**((E_corr - int_a)/slope_a)
            # DECADE CALCULATION
            decades_a = np.log10(i_fit.max()) - np.log10(i_fit.min())
        else:
            st.sidebar.error("Anodic range includes negative current!")

    # Cathodic Fit
    mask_c_fit = (E < (E_corr - cathodic_start)) & (E > (E_corr - cathodic_end))
    b_c, i_corr_c, decades_c = None, None, 0.0
    
    if np.sum(mask_c_fit) > 2:
        i_fit_c = np.abs(i_meas[mask_c_fit])
        if np.all(i_fit_c > 0):
            slope_c, int_c, r_c, _, _ = linregress(np.log10(i_fit_c), E[mask_c_fit])
            b_c = abs(slope_c)
            i_corr_c = 10**((E_corr - int_c)/slope_c)
            # DECADE CALCULATION
            decades_c = np.log10(i_fit_c.max()) - np.log10(i_fit_c.min())

    # --- PLOTS ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(E, np.abs(i_meas), 'k.', markersize=2, label='Data', alpha=0.4)
    ax.axvline(E_corr, color='gray', ls='--', alpha=0.5)

    # Plot Anodic Fit
    if b_a is not None:
        # Color code based on decade validity
        color_a = 'green' if decades_a >= 1.0 else 'red'
        E_seg = E[mask_a_fit]
        i_seg = 10**((E_seg - int_a)/slope_a)
        ax.semilogy(E_seg, i_seg, color=color_a, linestyle='-', lw=3, label=f'Anodic ({decades_a:.1f} dec)')
        # Extrapolate
        x_ex = np.linspace(E_corr, E_corr+anodic_end+0.1, 50)
        y_ex = 10**((x_ex - int_a)/slope_a)
        ax.semilogy(x_ex, y_ex, color=color_a, linestyle='--', lw=1, alpha=0.5)

    # Plot Cathodic Fit
    if b_c is not None:
        color_c = 'green' if decades_c >= 1.0 else 'red'
        E_seg = E[mask_c_fit]
        i_seg = 10**((E_seg - int_c)/slope_c)
        ax.semilogy(E_seg, i_seg, color=color_c, linestyle='-', lw=3, label=f'Cathodic ({decades_c:.1f} dec)')
        # Extrapolate
        x_ex = np.linspace(E_corr-cathodic_end-0.1, E_corr, 50)
        y_ex = 10**((x_ex - int_c)/slope_c)
        ax.semilogy(x_ex, y_ex, color=color_c, linestyle='--', lw=1, alpha=0.5)

    if E_pp is not None:
        ax.plot(E_pp, i_crit, 'ro', label='Passivation Peak')

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (A/cm²)")
    ax.legend(loc='lower right')
    ax.grid(True, which="both", alpha=0.2)
    st.pyplot(fig)

    # --- METRICS & VALIDATION ---
    st.subheader("Parameter Validation")
    
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("### 🔵 Cathodic")
        if b_c: 
            st.metric("Beta C", f"{b_c*1000:.1f} mV/dec")
            st.metric("i_corr (C)", f"{i_corr_c:.2e} A/cm²")
            
            if decades_c < 1.0:
                st.warning(f"⚠️ Linearity: {decades_c:.2f} Decades (< 1.0)")
            else:
                st.success(f"✅ Linearity: {decades_c:.2f} Decades")
        else:
            st.write("No fit")
        
    with m2:
        st.markdown("### 🔴 Anodic")
        if b_a: 
            st.metric("Beta A", f"{b_a*1000:.1f} mV/dec")
            st.metric("i_corr (A)", f"{i_corr_a:.2e} A/cm²")
            
            if decades_a < 1.0:
                st.error(f"⚠️ Linearity: {decades_a:.2f} Decades")
                st.caption("Standard requires > 1.0. This fit is statistically weak.")
            else:
                st.success(f"✅ Linearity: {decades_a:.2f} Decades")
        else:
            st.write("No fit")
            
    with m3:
        st.markdown("### ⚙️ Passivation")
        if E_pp:
            st.metric("E_pp", f"{E_pp:.3f} V")
            st.metric("i_crit", f"{i_crit:.2e} A/cm²")
        else:
            st.info("No active-passive peak found.")
