import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

st.set_page_config(page_title="Automated Smart Tafel", layout="wide")
st.title("Automated Smart Tafel Analysis")

st.markdown("""
**Status:** 🤖 **Auto-Pilot Engaged**
The algorithm is autonomously scanning your data to find the "most linear" Tafel regions.
* **Anodic Constraint:** It strictly searches *between* $E_{corr}$ and the Primary Passivation Potential.
* **Optimization:** It maximizes the $R^2$ (linearity) coefficient to ignore curved data points.
""")

# --- UTILS ---
def smooth_curve(y, window_length=15, polyorder=3):
    if len(y) < window_length: return y
    return savgol_filter(y, window_length, polyorder)

def find_ecorr(E, i):
    idx = np.argmin(np.abs(i))
    return E[idx], i[idx], idx

def get_best_linear_segment(E_segment, log_i_segment, min_points=10, min_r_squared=0.98):
    """
    Scans the provided segment to find the sub-region with the highest R^2.
    Uses a sliding window approach.
    """
    n = len(E_segment)
    if n < min_points:
        return None, None, None, None # Not enough data
    
    candidates = []
    # Check windows from 40% size
    window_size = int(n * 0.4)
    if window_size < min_points: window_size = n 
    
    steps = range(0, n - window_size + 1, max(1, int(n/20))) 
    
    for start_i in steps:
        end_i = start_i + window_size
        x = E_segment[start_i:end_i]
        y = log_i_segment[start_i:end_i]
        
        slope, intercept, r_val, _, _ = linregress(y, x) 
        candidates.append((r_val**2, start_i, end_i, slope, intercept))
    
    if not candidates:
        # Fallback: fit whole thing
        slope, intercept, r_val, _, _ = linregress(log_i_segment, E_segment)
        return slope, intercept, r_val**2, np.ones(n, dtype=bool)

    # Pick best window
    candidates.sort(key=lambda x: x[0], reverse=True)
    best_r2, start_idx, end_idx, best_slope, best_int = candidates[0]
    
    mask = np.zeros(n, dtype=bool)
    mask[start_idx:end_idx] = True
    
    return best_slope, best_int, best_r2, mask

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

    # --- 1. GLOBAL REGION IDENTIFICATION ---
    anodic_mask = E > E_corr
    E_anod_global = E[anodic_mask]
    i_anod_global = i_meas[anodic_mask]
    
    E_pp, i_crit, E_bd = None, None, None
    
    # Detect Passivation Peak
    if len(E_anod_global) > 10:
        i_smooth = smooth_curve(i_anod_global, 21)
        peaks = argrelextrema(i_smooth, np.greater)[0]
        valid_peaks = [p for p in peaks if i_smooth[p] > 0]
        
        if valid_peaks:
            best_p = valid_peaks[np.argmax(i_smooth[valid_peaks])]
            E_pp = E_anod_global[best_p]
            i_crit = i_anod_global[best_p]
            
            # Detect Breakdown
            mask_pass = E_anod_global > E_pp
            if np.any(mask_pass):
                i_p = i_smooth[mask_pass]
                E_p = E_anod_global[mask_pass]
                min_idx = np.argmin(i_p)
                i_pass_min = i_p[min_idx]
                rise = (i_p > i_pass_min*5) & (E_p > E_p[min_idx])
                if np.any(rise): E_bd = E_p[rise][0]

    # --- 2. SMART TAFEL FITTING ---
    
    # --- ANODIC AUTO-FIT ---
    b_a, i_corr_a, r2_a, decades_a, mask_a_best = None, None, 0, 0, None
    E_a_fit_plot, i_a_fit_plot = None, None
    
    search_start_a = E_corr + 0.02
    search_end_a = E_pp - 0.02 if E_pp else E.max()
    
    mask_search_a = (E > search_start_a) & (E < search_end_a)
    
    if np.sum(mask_search_a) > 10:
        E_search = E[mask_search_a]
        i_search = i_meas[mask_search_a]
        
        if np.sum(i_search > 0) > 10:
            E_use = E_search[i_search > 0]
            log_i_use = np.log10(i_search[i_search > 0])
            
            slope, intercept, r2, local_mask = get_best_linear_segment(E_use, log_i_use)
            
            if slope is not None:
                b_a = slope
                i_corr_a = 10**((E_corr - intercept)/slope)
                r2_a = r2
                E_a_fit_plot = E_use[local_mask]
                i_a_fit_plot = 10**log_i_use[local_mask]
                decades_a = np.log10(i_a_fit_plot.max()) - np.log10(i_a_fit_plot.min())

    # --- CATHODIC AUTO-FIT ---
    b_c, i_corr_c, r2_c, decades_c, mask_c_best = None, None, 0, 0, None
    E_c_fit_plot, i_c_fit_plot = None, None
    
    search_start_c = E_corr - 0.02
    search_end_c = E.min()
    
    mask_search_c = (E < search_start_c) & (E > search_end_c)
    
    if np.sum(mask_search_c) > 10:
        E_search_c = E[mask_search_c]
        i_search_c = np.abs(i_meas[mask_search_c])
        
        if np.sum(i_search_c > 0) > 10:
            E_use_c = E_search_c[i_search_c > 0]
            log_i_use_c = np.log10(i_search_c[i_search_c > 0])
            
            slope_c, intercept_c, r2_val_c, local_mask_c = get_best_linear_segment(E_use_c, log_i_use_c)
            
            if slope_c is not None:
                b_c = abs(slope_c)
                i_corr_c = 10**((E_corr - intercept_c)/slope_c) 
                r2_c = r2_val_c
                E_c_fit_plot = E_use_c[local_mask_c]
                i_c_fit_plot = 10**log_i_use_c[local_mask_c]
                decades_c = np.log10(i_c_fit_plot.max()) - np.log10(i_c_fit_plot.min())

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. SHADING
    ax.axvspan(E.min(), E_corr, color='blue', alpha=0.05, label='Cathodic Zone')
    
    if E_pp is not None:
        ax.axvspan(E_corr, E_pp, color='red', alpha=0.05, label='Anodic Active Zone')
        limit_p = E_bd if E_bd else E.max()
        ax.axvspan(E_pp, limit_p, color='green', alpha=0.05, label='Passive Zone')
        if E_bd: ax.axvline(E_bd, color='orange', ls=':', lw=2, label='Breakdown Potential')
    else:
        ax.axvspan(E_corr, E.max(), color='red', alpha=0.05)

    # 2. DATA
    ax.semilogy(E, np.abs(i_meas), 'k.', markersize=2, alpha=0.4, label='Raw Data')
    ax.axvline(E_corr, color='gray', ls='--', alpha=0.5, label='Corrosion Potential ($E_{corr}$)')

    # 3. SMART FITS
    if b_a is not None:
        ax.semilogy(E_a_fit_plot, i_a_fit_plot, 'r-', lw=3, label=f'Anodic Tafel Fit ($R^2$={r2_a:.3f})')
        y_ex = i_corr_a * 10**((E - E_corr)/b_a)
        # Limit extrapolation drawing to near Ecorr
        mask_ex = (E > E_corr) & (E < E_corr + 0.3)
        ax.semilogy(E[mask_ex], y_ex[mask_ex], 'r--', lw=1, alpha=0.5)

    if b_c is not None:
        ax.semilogy(E_c_fit_plot, i_c_fit_plot, 'b-', lw=3, label=f'Cathodic Tafel Fit ($R^2$={r2_c:.3f})')
        y_ex = i_corr_c * 10**((E_corr - E)/b_c)
        mask_ex = (E < E_corr) & (E > E_corr - 0.3)
        ax.semilogy(E[mask_ex], y_ex[mask_ex], 'b--', lw=1, alpha=0.5)

    if E_pp: ax.plot(E_pp, i_crit, 'ro', label='Primary Passivation Potential')

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (A/cm²)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    st.pyplot(fig)

    # --- METRICS ---
    st.subheader("Automated Results")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("### 🔵 Cathodic Auto-Fit")
        if b_c: 
            st.metric("Beta C", f"{b_c*1000:.1f} mV/dec")
            st.metric("Corrosion Current Density", f"{i_corr_c:.2e}")
            st.caption(f"Linearity ($R^2$): {r2_c:.4f}")
            if decades_c < 1.0: st.warning(f"Range: {decades_c:.2f} Decades (Weak)")
            else: st.success(f"Range: {decades_c:.2f} Decades")
            
    with m2:
        st.markdown("### 🔴 Anodic Auto-Fit")
        if b_a: 
            st.metric("Beta A", f"{b_a*1000:.1f} mV/dec")
            st.metric("Corrosion Current Density", f"{i_corr_a:.2e}")
            st.caption(f"Linearity ($R^2$): {r2_a:.4f}")
            if decades_a < 1.0: st.error(f"Range: {decades_a:.2f} Decades (Weak)")
            else: st.success(f"Range: {decades_a:.2f} Decades")
            
    with m3:
        st.markdown("### ⚙️ Passivation Detection")
        if E_pp:
            st.metric("Primary Passivation Potential", f"{E_pp:.3f} V")
            st.metric("Critical Current Density", f"{i_crit:.2e}")
            if E_bd: st.metric("Breakdown Potential", f"{E_bd:.3f} V")
        else:
            st.info("No passivation peak detected.")
