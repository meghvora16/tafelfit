import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

st.set_page_config(page_title="Complete Polarization Analysis", layout="wide")
st.title("Complete Polarization Analysis")

st.markdown("""
**Visual Guide:**
* **Background Shading:** Automatically detects the physical state of the metal.
    * 🔵 **Blue:** Cathodic Region.
    * 🔴 **Red:** Anodic Active Region (Dissolution).
    * 🟢 **Green:** Passive Region (Protection).
* **Thick Lines:** The specific data points you selected for the Tafel fit (via sidebar).
* **Dashed Lines:** Extrapolation of that fit.
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
    a_end_def = 0.10
    anodic_start = st.sidebar.slider("Start (V > Ecorr)", 0.0, 0.5, a_start_def, 0.005, format="%.3f")
    anodic_end = st.sidebar.slider("End (V > Ecorr)", 0.0, 0.5, a_end_def, 0.005, format="%.3f")
    
    # Cathodic Settings
    st.sidebar.subheader("🔵 Cathodic Fit Range")
    c_start_def = 0.02
    c_end_def = 0.15
    cathodic_start = st.sidebar.slider("Start (V < Ecorr)", 0.0, 0.5, c_start_def, 0.005, format="%.3f")
    cathodic_end = st.sidebar.slider("End (V < Ecorr)", 0.0, 0.5, c_end_def, 0.005, format="%.3f")

    # --- AUTOMATIC REGION DETECTION (For Shading) ---
    anodic_mask = E > E_corr
    E_anod = E[anodic_mask]
    i_anod = i_meas[anodic_mask]
    
    E_pp, i_crit, E_bd = None, None, None
    
    if len(E_anod) > 10:
        i_smooth = smooth_curve(i_anod, 21)
        peaks = argrelextrema(i_smooth, np.greater)[0]
        valid_peaks = [p for p in peaks if i_smooth[p] > 0]
        
        if valid_peaks:
            best_p = valid_peaks[np.argmax(i_smooth[valid_peaks])]
            E_pp = E_anod[best_p]
            i_crit = i_anod[best_p]
            
            # Detect Breakdown
            mask_pass = E_anod > E_pp
            if np.any(mask_pass):
                i_p = i_smooth[mask_pass]
                E_p = E_anod[mask_pass]
                min_idx = np.argmin(i_p)
                i_pass_min = i_p[min_idx]
                # Threshold: 5x min passive current
                rise = (i_p > i_pass_min*5) & (E_p > E_p[min_idx])
                if np.any(rise): E_bd = E_p[rise][0]

    # --- MANUAL TAFEL FITS ---
    # Anodic
    mask_a_fit = (E > (E_corr + anodic_start)) & (E < (E_corr + anodic_end))
    b_a, i_corr_a, decades_a = None, None, 0.0
    if np.sum(mask_a_fit) > 2:
        i_fit = i_meas[mask_a_fit]
        if np.all(i_fit > 0):
            s_a, int_a, _, _, _ = linregress(np.log10(i_fit), E[mask_a_fit])
            b_a = s_a
            i_corr_a = 10**((E_corr - int_a)/s_a)
            decades_a = np.log10(i_fit.max()) - np.log10(i_fit.min())

    # Cathodic
    mask_c_fit = (E < (E_corr - cathodic_start)) & (E > (E_corr - cathodic_end))
    b_c, i_corr_c, decades_c = None, None, 0.0
    if np.sum(mask_c_fit) > 2:
        i_fit_c = np.abs(i_meas[mask_c_fit])
        if np.all(i_fit_c > 0):
            s_c, int_c, _, _, _ = linregress(np.log10(i_fit_c), E[mask_c_fit])
            b_c = abs(s_c)
            i_corr_c = 10**((E_corr - int_c)/s_c)
            decades_c = np.log10(i_fit_c.max()) - np.log10(i_fit_c.min())

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. SHADING (The Physics)
    # Cathodic
    ax.axvspan(E.min(), E_corr, color='blue', alpha=0.05, label='Cathodic')
    
    # Anodic Active
    if E_pp is not None:
        ax.axvspan(E_corr, E_pp, color='red', alpha=0.1, label='Active')
        # Passive
        limit = E_bd if E_bd else E.max()
        ax.axvspan(E_pp, limit, color='green', alpha=0.1, label='Passive')
        # Breakdown line
        if E_bd:
            ax.axvline(E_bd, color='orange', ls='--', lw=1.5, label='Breakdown')
    else:
        # Fallback shading if no peak found
        ax.axvspan(E_corr, E.max(), color='red', alpha=0.05)

    # 2. DATA
    ax.semilogy(E, np.abs(i_meas), 'k.', markersize=3, alpha=0.6, label='Data')
    ax.axvline(E_corr, color='gray', ls='--', alpha=0.5)

    # 3. FITS (The Math)
    if b_a is not None:
        color = 'darkgreen' if decades_a >= 1.0 else 'firebrick'
        # Solid thick line for fitted range
        E_seg = E[mask_a_fit]
        i_seg = 10**((E_seg - int_a)/s_a)
        ax.semilogy(E_seg, i_seg, color=color, ls='-', lw=3, label=f'Anodic Fit ({decades_a:.1f} dec)')
        # Dashed extrapolation
        x_ex = np.linspace(E_corr, E_corr+anodic_end+0.1, 50)
        y_ex = 10**((x_ex - int_a)/s_a)
        ax.semilogy(x_ex, y_ex, color=color, ls='--', lw=1)

    if b_c is not None:
        color = 'darkblue' if decades_c >= 1.0 else 'firebrick'
        E_seg = E[mask_c_fit]
        i_seg = 10**((E_seg - int_c)/s_c)
        ax.semilogy(E_seg, i_seg, color=color, ls='-', lw=3, label=f'Cathodic Fit ({decades_c:.1f} dec)')
        x_ex = np.linspace(E_corr-cathodic_end-0.1, E_corr, 50)
        y_ex = 10**((x_ex - int_c)/s_c)
        ax.semilogy(x_ex, y_ex, color=color, ls='--', lw=1)

    if E_pp:
        ax.plot(E_pp, i_crit, 'ro', markersize=6)

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (A/cm²)")
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(True, which="both", alpha=0.2)
    
    st.pyplot(fig)

    # --- METRICS ---
    st.subheader("Parameter Validation")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("### 🔵 Cathodic")
        if b_c: 
            st.metric("Beta C", f"{b_c*1000:.1f} mV/dec")
            st.metric("i_corr", f"{i_corr_c:.2e} A/cm²")
            if decades_c < 1.0: st.warning(f"Linearity: {decades_c:.2f} Decades")
            else: st.success(f"Linearity: {decades_c:.2f} Decades")
            
    with m2:
        st.markdown("### 🔴 Anodic")
        if b_a: 
            st.metric("Beta A", f"{b_a*1000:.1f} mV/dec")
            st.metric("i_corr", f"{i_corr_a:.2e} A/cm²")
            if decades_a < 1.0: st.error(f"Linearity: {decades_a:.2f} Decades")
            else: st.success(f"Linearity: {decades_a:.2f} Decades")
            
    with m3:
        st.markdown("### ⚙️ Passivation")
        if E_pp:
            st.metric("E_pp", f"{E_pp:.3f} V")
            st.metric("i_crit", f"{i_crit:.2e} A/cm²")
            if E_bd: st.metric("E_bd", f"{E_bd:.3f} V")
