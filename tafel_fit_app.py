import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

# --- CONFIGURATION ---
st.set_page_config(page_title="Full Polarization Analysis", layout="wide")
st.title("Full Polarization Analysis (Anodic & Cathodic)")

st.markdown("""
**Methodology:**
This tool splits the curve into three distinct physical zones:
1.  **🔵 Cathodic Region (Blue):** Where the reduction reaction occurs (e.g., $2H^+ + 2e^- \\to H_2$).
2.  **🔴 Anodic Active Region (Red):** Where the metal actively dissolves ($M \\to M^{n+} + ne^-$).
3.  **🟢 Passive Region (Green):** Where a protective oxide film forms.
""")

# --- UTILS ---
def smooth_curve(y, window_length=15, polyorder=3):
    """Smooths data using Savitzky-Golay filter."""
    if len(y) < window_length:
        return y
    return savgol_filter(y, window_length, polyorder)

def find_ecorr(E, i):
    """Finds Ecorr based on the minimum absolute current."""
    idx = np.argmin(np.abs(i))
    return E[idx], i[idx], idx

# --- MAIN APP ---
data_file = st.file_uploader("Upload LSV/Polarization data (CSV/Excel)", type=["csv", "xlsx", "xls"])

if data_file is not None:
    # Load Data
    if data_file.name.endswith(".csv"):
        df = pd.read_csv(data_file)
    else:
        df = pd.read_excel(data_file)
    
    st.success(f"Loaded {len(df)} rows.")
    
    # Column Selection
    c1, c2, c3, c4 = st.columns(4)
    col_E = c1.selectbox("Potential column", df.columns, index=0)
    col_I = c2.selectbox("Current column", df.columns, index=1)
    pot_units = c3.selectbox("Potential units", ["V", "mV"], 0)
    cur_units = c4.selectbox("Current units", ["A", "mA", "uA", "nA"], 0)
    
    area = st.number_input("Electrode Area (cm²)", value=1.0)

    # Data Preprocessing
    E_raw = df[col_E].astype(float).to_numpy()
    I_raw = df[col_I].astype(float).to_numpy()
    
    # Normalize Units
    if pot_units == "mV": E_raw /= 1000.0
    unit_mult = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9}
    I_norm = I_raw * unit_mult[cur_units]
    i_dens = I_norm / area  # A/cm²

    # Sort by Potential
    sort_idx = np.argsort(E_raw)
    E = E_raw[sort_idx]
    i_meas = i_dens[sort_idx]
    
    # 1. Find Ecorr
    E_corr, i_corr_raw, idx_corr = find_ecorr(E, i_meas)
    
    # --- SPLIT DATA ---
    # Anodic Branch (E > Ecorr)
    anodic_mask = E > E_corr
    E_anod = E[anodic_mask]
    i_anod = i_meas[anodic_mask]
    
    # Cathodic Branch (E < Ecorr)
    cathodic_mask = E < E_corr
    E_cath = E[cathodic_mask]
    i_cath = np.abs(i_meas[cathodic_mask]) # Work with absolute current

    # --- ANODIC ANALYSIS ---
    E_pp, i_crit, E_bd, i_pass_mean = None, None, None, None
    b_a = None
    
    if len(E_anod) > 10:
        i_anod_smooth = smooth_curve(i_anod, window_length=21)
        
        # Detect Passivation Peak
        peaks_idx = argrelextrema(i_anod_smooth, np.greater)[0]
        valid_peaks = [p for p in peaks_idx if i_anod_smooth[p] > 0]

        if valid_peaks:
            best_p = valid_peaks[np.argmax(i_anod_smooth[valid_peaks])]
            E_pp = E_anod[best_p]
            i_crit = i_anod[best_p]
        
        # Detect Passive/Breakdown
        if E_pp is not None:
            post_peak_mask = E_anod > E_pp
            E_post = E_anod[post_peak_mask]
            i_post = i_anod_smooth[post_peak_mask]
            
            if len(i_post) > 5:
                min_idx = np.argmin(i_post)
                i_pass_mean = i_post[min_idx]
                E_min = E_post[min_idx]
                
                # Breakdown threshold (5x passive current)
                threshold = i_pass_mean * 5
                rise_mask = (i_post > threshold) & (E_post > E_min)
                if np.any(rise_mask):
                    E_bd = E_post[rise_mask][0]

        # Anodic Tafel Fit (b_a)
        # Range: Ecorr + 20mV to E_pp - 50mV (or Ecorr + 250mV)
        upper_limit = E_pp - 0.05 if E_pp is not None else E_corr + 0.25
        tafel_mask_a = (E_anod > (E_corr + 0.02)) & (E_anod < upper_limit)
        E_tafel_a = E_anod[tafel_mask_a]
        i_tafel_a = i_anod[tafel_mask_a]
        
        if len(E_tafel_a) > 5:
            slope_a, intercept_a, _, _, _ = linregress(np.log10(i_tafel_a), E_tafel_a)
            b_a = slope_a

    # --- CATHODIC ANALYSIS ---
    b_c = None
    E_cath_limit = E_cath[0] if len(E_cath) > 0 else None # Default to start of scan
    
    if len(E_cath) > 10:
        # Cathodic Tafel Fit (b_c)
        # Range: Ecorr - 250mV to Ecorr - 20mV
        # Note: We fit E vs log(i). Slope is negative for cathodic in this convention? 
        # No, beta is usually reported positive. But mathematically the line is E = Ecorr - bc*log(i/icorr)
        # If we fit E = slope * log(i) + int, slope will be positive if we move away from Ecorr?
        # Wait. As E decreases (Cathodic), log(i) increases. So slope (dE/dlogi) is NEGATIVE.
        
        tafel_mask_c = (E_cath < (E_corr - 0.02)) & (E_cath > (E_corr - 0.25))
        E_tafel_c = E_cath[tafel_mask_c]
        i_tafel_c = i_cath[tafel_mask_c]
        
        if len(E_tafel_c) > 5:
            slope_c, intercept_c, _, _, _ = linregress(np.log10(i_tafel_c), E_tafel_c)
            b_c = abs(slope_c) # Report as positive magnitude
            
            # Check for diffusion limit on Cathode (Plateau)
            # If slope becomes very steep (dE/dlogi -> infinity) or i becomes constant
            # Simple check: if derivative of i w.r.t E is close to zero
            # Let's just use the full range for shading unless we want to be super fancy.
            # Shading: E_min to E_corr
            pass

    # --- DISPLAY RESULTS ---
    st.subheader("Extracted Electrochemical Parameters")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🔵 Cathodic & General")
        st.metric("E_corr (V)", f"{E_corr:.3f}")
        if b_c is not None:
            st.metric("Cathodic Slope (b_c) (mV/dec)", f"{b_c*1000:.1f}")
        else:
            st.write("Could not fit Cathodic Tafel.")

    with col2:
        st.markdown("### 🔴 Anodic Active")
        if b_a is not None:
            st.metric("Anodic Slope (b_a) (mV/dec)", f"{b_a*1000:.1f}")
        if E_pp is not None:
            st.metric("E_pp (Passivation) (V)", f"{E_pp:.3f}")
            st.metric("i_crit (A/cm²)", f"{i_crit:.2e}")
        else:
            st.write("No passivation peak detected.")

    # Passive Metrics if exist
    if i_pass_mean is not None:
        st.markdown("### 🟢 Passive Region")
        c3, c4 = st.columns(2)
        c3.metric("i_pass (A/cm²)", f"{i_pass_mean:.2e}")
        if E_bd is not None:
            c4.metric("E_bd (Breakdown) (V)", f"{E_bd:.3f}")

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot Raw Data
    ax.semilogy(E, np.abs(i_meas), 'k-', lw=1.5, label='Data', alpha=0.8)
    
    # Plot Ecorr Line
    ax.axvline(E_corr, color='gray', linestyle='--', alpha=0.5)

    # 1. CATHODIC REGION SHADING (Blue)
    if len(E_cath) > 0:
        ax.axvspan(E_cath.min(), E_corr, color='blue', alpha=0.1, label='Cathodic Region')

    # 2. ANODIC ACTIVE REGION SHADING (Red)
    if E_pp is not None:
        ax.axvspan(E_corr, E_pp, color='red', alpha=0.1, label='Anodic Active')
        ax.plot(E_pp, i_crit, 'ro') # Peak dot
    else:
        # If no passivation, shade active region arbitrarily 300mV
        ax.axvspan(E_corr, E_corr + 0.3, color='red', alpha=0.1, label='Anodic Active (Est.)')

    # 3. PASSIVE REGION SHADING (Green)
    if E_pp is not None:
        limit = E_bd if E_bd is not None else E_anod.max()
        ax.axvspan(E_pp, limit, color='green', alpha=0.1, label='Passive Region')
        if E_bd is not None:
            ax.axvline(E_bd, color='orange', linestyle='--', lw=2, label='Breakdown')

    # Tafel Fits Visualization
    # Anodic Fit
    if b_a is not None:
        # E = b*log(i) + a  -> log(i) = (E - a)/b -> i = 10^((E-a)/b)
        x_a = np.linspace(E_corr, E_corr+0.2, 50)
        y_a = 10**((x_a - intercept_a)/slope_a)
        ax.semilogy(x_a, y_a, 'r--', lw=1, label=f'b_a={b_a*1000:.0f}mV')
        
    # Cathodic Fit
    if b_c is not None:
        # Slope was negative in regression, but we used abs for reporting.
        # In regression: E = slope_c * log(i) + intercept_c.
        # We use the actual slope_c from regression (which is negative) to plot
        x_c = np.linspace(E_corr-0.2, E_corr, 50)
        y_c = 10**((x_c - intercept_c)/slope_c)
        ax.semilogy(x_c, y_c, 'b--', lw=1, label=f'b_c={b_c*1000:.0f}mV')

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (|A|/cm²)")
    ax.set_title("Global Polarization Analysis")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc='lower right')
    
    st.pyplot(fig)
