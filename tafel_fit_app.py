import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

st.set_page_config(page_title="Interactive Tafel Analysis", layout="wide")
st.title("Interactive Tafel Analysis")

st.markdown("""
**Instruction:** Use the **Sidebar** to manually adjust the Tafel fitting regions. 
If the automatic fit looks bad, tweak the sliders until the dashed lines match the straightest parts of your log-plot.
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
    # Defaults: 20mV to 200mV away from Ecorr
    a_start_def = 0.02 
    a_end_def = 0.15
    
    anodic_start = st.sidebar.slider("Start (V above Ecorr)", 0.0, 0.5, a_start_def, 0.01, format="%.2f")
    anodic_end = st.sidebar.slider("End (V above Ecorr)", 0.0, 1.0, a_end_def, 0.01, format="%.2f")
    
    # Cathodic Settings
    st.sidebar.subheader("🔵 Cathodic Fit Range")
    c_start_def = 0.02
    c_end_def = 0.15
    
    cathodic_start = st.sidebar.slider("Start (V below Ecorr)", 0.0, 0.5, c_start_def, 0.01, format="%.2f")
    cathodic_end = st.sidebar.slider("End (V below Ecorr)", 0.0, 1.0, c_end_def, 0.01, format="%.2f")

    # --- ANALYSIS ---
    
    # 1. Anodic Active / Passive Detection
    anodic_mask = E > E_corr
    E_anod = E[anodic_mask]
    i_anod = i_meas[anodic_mask]
    
    E_pp, i_crit, E_bd, i_pass_mean = None, None, None, None
    
    if len(E_anod) > 10:
        i_smooth = smooth_curve(i_anod, 21)
        peaks = argrelextrema(i_smooth, np.greater)[0]
        valid_peaks = [p for p in peaks if i_smooth[p] > 0]
        if valid_peaks:
            best_p = valid_peaks[np.argmax(i_smooth[valid_peaks])]
            E_pp = E_anod[best_p]
            i_crit = i_anod[best_p]
            
            # Passive
            mask_pass = E_anod > E_pp
            if np.any(mask_pass):
                i_p = i_smooth[mask_pass]
                E_p = E_anod[mask_pass]
                min_idx = np.argmin(i_p)
                i_pass_mean = i_p[min_idx]
                # Breakdown
                rise = (i_p > i_pass_mean*5) & (E_p > E_p[min_idx])
                if np.any(rise): E_bd = E_p[rise][0]

    # 2. MANUAL TAFEL FITS
    
    # Anodic Fit
    # Range: Ecorr + start to Ecorr + end
    mask_a_fit = (E > (E_corr + anodic_start)) & (E < (E_corr + anodic_end))
    b_a, i_corr_a = None, None
    
    if np.sum(mask_a_fit) > 5:
        # Check if current is positive
        i_fit = i_meas[mask_a_fit]
        if np.all(i_fit > 0):
            slope_a, int_a, r_a, _, _ = linregress(np.log10(i_fit), E[mask_a_fit])
            b_a = slope_a
            i_corr_a = 10**((E_corr - int_a)/slope_a)
        else:
            st.sidebar.warning("Anodic fit range contains negative currents!")

    # Cathodic Fit
    # Range: Ecorr - end to Ecorr - start (since we go downwards)
    mask_c_fit = (E < (E_corr - cathodic_start)) & (E > (E_corr - cathodic_end))
    b_c, i_corr_c = None, None
    
    if np.sum(mask_c_fit) > 5:
        i_fit_c = np.abs(i_meas[mask_c_fit])
        if np.all(i_fit_c > 0):
            slope_c, int_c, r_c, _, _ = linregress(np.log10(i_fit_c), E[mask_c_fit])
            b_c = abs(slope_c)
            i_corr_c = 10**((E_corr - int_c)/slope_c) # Intercept calculation slightly depends on sign convention

    # --- PLOTS ---
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogy(E, np.abs(i_meas), 'k.', markersize=2, label='Data', alpha=0.5)
    ax.axvline(E_corr, color='gray', ls='--', alpha=0.5)

    # Plot Anodic Fit Segment
    if b_a is not None:
        # Plot the segment actually fitted (thick line)
        E_seg = E[mask_a_fit]
        i_seg = 10**((E_seg - int_a)/slope_a)
        ax.semilogy(E_seg, i_seg, 'r-', lw=3, label=f'Anodic Fit ($b_a$={b_a*1000:.0f}mV)')
        # Extrapolate
        x_ex = np.linspace(E_corr, E_corr+anodic_end+0.1, 50)
        y_ex = 10**((x_ex - int_a)/slope_a)
        ax.semilogy(x_ex, y_ex, 'r--', lw=1, alpha=0.5)

    # Plot Cathodic Fit Segment
    if b_c is not None:
        E_seg = E[mask_c_fit]
        i_seg = 10**((E_seg - int_c)/slope_c) # slope_c is negative
        ax.semilogy(E_seg, i_seg, 'b-', lw=3, label=f'Cathodic Fit ($b_c$={b_c*1000:.0f}mV)')
        # Extrapolate
        x_ex = np.linspace(E_corr-cathodic_end-0.1, E_corr, 50)
        y_ex = 10**((x_ex - int_c)/slope_c)
        ax.semilogy(x_ex, y_ex, 'b--', lw=1, alpha=0.5)

    # Highlight Passivation
    if E_pp is not None:
        ax.plot(E_pp, i_crit, 'ro', label='Passivation Peak')
        ax.axvspan(E_corr, E_pp, color='red', alpha=0.05) # Active
        limit = E_bd if E_bd else E.max()
        ax.axvspan(E_pp, limit, color='green', alpha=0.05) # Passive

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (A/cm²)")
    ax.legend(loc='lower right')
    ax.grid(True, which="both", alpha=0.2)
    
    st.pyplot(fig)

    # --- METRICS ---
    m1, m2, m3 = st.columns(3)
    with m1:
        st.markdown("### 🔵 Cathodic")
        if b_c: st.metric("Beta C (mV/dec)", f"{b_c*1000:.1f}")
        if i_corr_c: st.metric("i_corr (Cathodic)", f"{i_corr_c:.2e}")
        
    with m2:
        st.markdown("### 🔴 Anodic")
        if b_a: st.metric("Beta A (mV/dec)", f"{b_a*1000:.1f}")
        if i_corr_a: st.metric("i_corr (Anodic)", f"{i_corr_a:.2e}")
        
    with m3:
        st.markdown("### ⚙️ Passivation")
        if E_pp: st.metric("E_pp (V)", f"{E_pp:.3f}")
        if i_crit: st.metric("i_crit", f"{i_crit:.2e}")
        if i_pass_mean: st.metric("i_pass", f"{i_pass_mean:.2e}")

    # Final Comparison
    st.info("""
    **How to use:**
    1. Look at the red/blue **thick lines** on the plot.
    2. If they are covering a curved part of the data, **move the sliders** in the sidebar.
    3. You want the thick line to sit on top of the "straightest" segment of the dots.
    """)
