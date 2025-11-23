import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

# --- CONFIGURATION ---
st.set_page_config(page_title="Passivation & Tafel Analysis", layout="wide")
st.title("Passivation & Tafel Analysis")

st.markdown("""
**Methodology:**
Instead of forcing a monolithic Butler-Volmer fit (which fails for passivation), this tool performs **Region-Based Characterization**:
1.  **Active Region:** Fits the Tafel slope ($b_a$) before the peak.
2.  **Passivation Peak:** Detects $E_{pp}$ (Primary Passivation Potential) and $i_{crit}$.
3.  **Passive Region:** Identifies the passive current $i_{pass}$.
4.  **Transpassive/Breakdown:** Detects where the current rises again ($E_{bd}$/Transpassive).
""")

# --- UTILS ---
def smooth_curve(y, window_length=15, polyorder=3):
    """Smooths data using Savitzky-Golay filter to handle noise."""
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

    # Sort by Potential (Crucial for LSV)
    sort_idx = np.argsort(E_raw)
    E = E_raw[sort_idx]
    i_meas = i_dens[sort_idx]
    
    # 1. Find Ecorr
    E_corr, i_corr_raw, idx_corr = find_ecorr(E, i_meas)
    
    # 2. Split into Anodic and Cathodic
    # We are mostly interested in Anodic for Passivation
    anodic_mask = E > E_corr
    E_anodic = E[anodic_mask]
    i_anodic = i_meas[anodic_mask]
    
    if len(E_anodic) < 10:
        st.error("Not enough data in the anodic branch to analyze passivation.")
        st.stop()

    # Smooth the anodic current (log space is better for smoothing usually, but linear works for peak finding)
    # However, for peak detection in passivation, linear current is safer to see the drop.
    i_anodic_smooth = smooth_curve(i_anodic, window_length=21)

    # 3. Detect Passivation Peak (E_pp, i_crit)
    # Look for local maxima
    peaks_idx = argrelextrema(i_anodic_smooth, np.greater)[0]
    
    # Filter peaks: Must be at least X% higher than the final current to be considered a "passivation" peak
    # and must be somewhat close to Ecorr (active region)
    valid_peaks = []
    for p in peaks_idx:
        if i_anodic_smooth[p] > 0: # Positive anodic current
            valid_peaks.append(p)
            
    if not valid_peaks:
        st.warning("No clear anodic peak detected. Data might not show passivation.")
        E_pp, i_crit = None, None
    else:
        # Assume the first major peak is the primary passivation potential
        # Heuristic: The highest peak in the first half of the scan often corresponds to Active->Passive
        best_p = valid_peaks[np.argmax(i_anodic_smooth[valid_peaks])] 
        E_pp = E_anodic[best_p]
        i_crit = i_anodic[best_p] # Use raw or smooth? Smooth is safer.
        idx_peak_global = np.where(E == E_pp)[0][0]

    # 4. Detect Passive Region & Breakdown (E_bd / "Erp")
    E_bd = None
    i_pass_mean = None
    
    if E_pp is not None:
        # Look at data AFTER the peak
        post_peak_mask = E_anodic > E_pp
        E_post = E_anodic[post_peak_mask]
        i_post = i_anodic_smooth[post_peak_mask]
        
        if len(i_post) > 10:
            # Find the "valley" (minimum current after peak)
            min_idx_local = np.argmin(i_post)
            i_min_val = i_post[min_idx_local]
            E_at_min = E_post[min_idx_local]
            
            # Define Passive Region: area around the minimum
            # Or simply define passive current as the minimum current found
            i_pass_mean = i_min_val
            
            # Detect Breakdown: Where current rises significantly above i_pass
            # Heuristic: When current rises to 5x or 10x the passive current, or reaches 50% of i_crit again
            # Let's use a threshold.
            threshold = i_min_val * 5 
            # Find where it crosses this threshold AFTER the valley
            rise_mask = (i_post > threshold) & (E_post > E_at_min)
            if np.any(rise_mask):
                E_bd = E_post[rise_mask][0]
            else:
                # If it never rises, maybe the scan ended in the passive region
                st.info("No breakdown/transpassive region detected (current remained low).")

    # 5. Tafel Fit (Active Region Only)
    # Fit between Ecorr and E_pp (with some buffer)
    b_a = None
    if E_pp is not None:
        # Take a slice between Ecorr + 50mV and E_pp - 50mV
        tafel_mask = (E > (E_corr + 0.02)) & (E < (E_pp - 0.05))
    else:
        # Standard Tafel: Ecorr + 50mV to Ecorr + 250mV
        tafel_mask = (E > (E_corr + 0.05)) & (E < (E_corr + 0.25))
        
    E_tafel = E[tafel_mask]
    i_tafel = np.abs(i_meas[tafel_mask])
    
    if len(E_tafel) > 5 and np.all(i_tafel > 0):
        # Fit log(i) vs E
        slope, intercept, r_val, _, _ = linregress(np.log10(i_tafel), E_tafel)
        b_a = slope # slope is dE/dlogi = Beta
        i_corr_fit = 10**((E_corr - intercept) / slope)
    else:
        i_corr_fit = np.nan

    # --- DISPLAY RESULTS ---
    st.subheader("Extracted Electrochemical Parameters")
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.markdown("### 🟢 Active Region")
        st.metric("E_corr (V)", f"{E_corr:.3f}")
        if E_pp is not None:
            st.metric("E_pp (Primary Passivation) (V)", f"{E_pp:.3f}")
            st.metric("i_crit (Critical Current) (A/cm²)", f"{i_crit:.2e}")
        else:
            st.write("No passivation peak detected.")
            
    with res_col2:
        st.markdown("### 🔴 Passive/Transpassive")
        if i_pass_mean is not None:
            st.metric("i_pass (Min. Passive Current) (A/cm²)", f"{i_pass_mean:.2e}")
        
        if E_bd is not None:
            st.metric("E_bd (Breakdown/Transpassive) (V)", f"{E_bd:.3f}", help="Potential where current rises significantly after passivation. Often confused with Erp in LSV.")
        else:
            st.write("No breakdown detected.")
            
        if b_a is not None:
            st.write(f"**Anodic Tafel Slope (b_a):** {b_a*1000:.1f} mV/dec")

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot raw data
    ax.semilogy(E, np.abs(i_meas), 'k-', lw=1.5, label='Experimental Data', alpha=0.7)
    
    # Plot Ecorr
    ax.axvline(E_corr, color='gray', linestyle='--', alpha=0.5, label='E_corr')
    
    # Highlight E_pp
    if E_pp is not None:
        ax.plot(E_pp, i_crit, 'ro', markersize=8, label=f'E_pp = {E_pp:.2f} V')
        ax.axvline(E_pp, color='r', linestyle=':', alpha=0.5)
        
    # Highlight E_bd
    if E_bd is not None:
        ax.axvline(E_bd, color='orange', linestyle='--', lw=2, label=f'Breakdown/Transpassive = {E_bd:.2f} V')
        # Shade passive region
        if E_pp is not None:
            ax.axvspan(E_pp, E_bd, color='green', alpha=0.1, label='Passive Region')

    # Plot Tafel Fit line
    if b_a is not None and E_pp is not None:
        # Extrapolate line for visual
        x_line = np.linspace(E_corr, E_pp, 100)
        y_line = 10**((x_line - intercept) / slope) # Inverse of E = b*log(i) + a -> log(i) = (E-a)/b
        # Wait, linregress was (logi, E). So E = slope*logi + intercept.
        # logi = (E - intercept) / slope. i = 10^...
        y_line = 10**((x_line - intercept)/slope)
        ax.semilogy(x_line, y_line, 'b--', label=f'Tafel Fit ($b_a$={b_a*1000:.0f}mV)')

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (|A|/cm²)")
    ax.set_title("Global Polarization Analysis")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    
    st.pyplot(fig)

    # Download Processed Data
    st.download_button("Download Processed Data", data=df.to_csv(index=False), file_name="processed_data.csv")
