import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

# --- CONFIGURATION ---
st.set_page_config(page_title="Advanced Polarization Analysis", layout="wide")
st.title("Advanced Polarization Analysis: Passivation & Diffusion")

st.markdown("""
**Physical Regions Identified:**
1.  **🔵 Cathodic Region:** Reduction reactions. Checks for **Tafel** (activation) or **Limiting** (diffusion) behavior.
2.  **🔴 Anodic Active Region:** Metal dissolution ($E_{corr} \to E_{pp}$).
3.  **🟢 Passive Region:** Oxide film protection ($E_{pp} \to E_{bd}$).
4.  **⚠️ Limiting Current ($i_L$):** Pure diffusion control (flat plateau), extrapolated horizontally.
""")

# --- UTILS ---
def smooth_curve(y, window_length=15, polyorder=3):
    if len(y) < window_length: return y
    return savgol_filter(y, window_length, polyorder)

def find_ecorr(E, i):
    idx = np.argmin(np.abs(i))
    return E[idx], i[idx], idx

def detect_limiting_current(E_segment, i_segment, window_fraction=0.2, tolerance=0.05):
    """
    Detects a limiting current plateau at the end of a segment.
    Criteria: The standard deviation of the last X% of points is small relative to the mean.
    """
    n_points = len(i_segment)
    if n_points < 20: return None, None
    
    # Look at the 'tail' (last 20% of the scan direction)
    # For Cathodic: most negative potentials (start of array if sorted?)
    # We assume input segments are ordered away from Ecorr? 
    # Let's just take the last portion of the provided array.
    
    n_tail = int(n_points * window_fraction)
    tail_i = i_segment[-n_tail:] # Last 20%
    
    mean_i = np.mean(tail_i)
    std_i = np.std(tail_i)
    
    # If variation is less than 5% of the mean value (flat line)
    if abs(mean_i) > 1e-12 and (std_i / abs(mean_i)) < tolerance:
        return mean_i, (E_segment[-n_tail], E_segment[-1]) # Return value and E-range
    return None, None

# --- MAIN APP ---
data_file = st.file_uploader("Upload LSV/Polarization data (CSV/Excel)", type=["csv", "xlsx", "xls"])

if data_file is not None:
    # Load Data
    if data_file.name.endswith(".csv"): df = pd.read_csv(data_file)
    else: df = pd.read_excel(data_file)
    
    st.success(f"Loaded {len(df)} rows.")
    
    c1, c2, c3, c4 = st.columns(4)
    col_E = c1.selectbox("Potential column", df.columns, index=0)
    col_I = c2.selectbox("Current column", df.columns, index=1)
    pot_units = c3.selectbox("Potential units", ["V", "mV"], 0)
    cur_units = c4.selectbox("Current units", ["A", "mA", "uA", "nA"], 0)
    area = st.number_input("Electrode Area (cm²)", value=1.0)

    # Data Preprocessing
    E_raw = df[col_E].astype(float).to_numpy()
    I_raw = df[col_I].astype(float).to_numpy()
    
    if pot_units == "mV": E_raw /= 1000.0
    unit_mult = {"A": 1.0, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9}
    I_norm = I_raw * unit_mult[cur_units]
    i_dens = I_norm / area

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
    
    # Cathodic Branch (E < Ecorr) - Sort descending (away from Ecorr) for logic
    cathodic_mask = E < E_corr
    E_cath = E[cathodic_mask]
    i_cath = np.abs(i_meas[cathodic_mask]) 
    
    # We want to look at the "tails" (far from Ecorr)
    # Anodic tail is just the end of E_anod
    # Cathodic tail is the START of E_cath (since E is sorted ascending -1.0 -> -0.4)
    
    # --- ANALYSIS ---
    
    # 1. Detect Cathodic Limiting Current (i_L_c)
    # Pass the "deepest" cathodic part (first 20% of sorted array)
    i_Lc = None
    E_range_Lc = None
    
    if len(i_cath) > 20:
        # Look at the most negative potentials (start of the sorted array)
        # We use the detect function on the first 20% reversed? 
        # detect_limiting_current expects the segment of interest to be at the END of the array provided.
        # So let's pass E_cath[0:N] reversed so the "tail" is the "deepest" part.
        E_deep_cath = E_cath[::-1] 
        i_deep_cath = i_cath[::-1]
        val, rng = detect_limiting_current(E_deep_cath, i_deep_cath)
        
        if val is not None:
            i_Lc = val
            E_range_Lc = rng # This will be (E_deep, E_less_deep)
    
    # 2. Anodic Analysis (Passivation / Tafel)
    E_pp, i_crit, E_bd, i_pass_mean = None, None, None, None
    b_a = None
    
    if len(E_anod) > 10:
        i_anod_smooth = smooth_curve(i_anod, window_length=21)
        
        # Passivation Peak
        peaks_idx = argrelextrema(i_anod_smooth, np.greater)[0]
        valid_peaks = [p for p in peaks_idx if i_anod_smooth[p] > 0]

        if valid_peaks:
            best_p = valid_peaks[np.argmax(i_anod_smooth[valid_peaks])]
            E_pp = E_anod[best_p]
            i_crit = i_anod[best_p]
        
        # Passive Region
        if E_pp is not None:
            post_peak_mask = E_anod > E_pp
            E_post = E_anod[post_peak_mask]
            i_post = i_anod_smooth[post_peak_mask]
            
            if len(i_post) > 5:
                min_idx = np.argmin(i_post)
                i_pass_mean = i_post[min_idx]
                E_min = E_post[min_idx]
                
                # Breakdown
                threshold = i_pass_mean * 5
                rise_mask = (i_post > threshold) & (E_post > E_min)
                if np.any(rise_mask):
                    E_bd = E_post[rise_mask][0]

        # Anodic Tafel
        upper = E_pp - 0.05 if E_pp is not None else E_corr + 0.25
        mask_a = (E_anod > (E_corr + 0.02)) & (E_anod < upper)
        if np.sum(mask_a) > 5:
            s_a, int_a, _, _, _ = linregress(np.log10(i_anod[mask_a]), E_anod[mask_a])
            b_a = s_a

    # 3. Cathodic Tafel
    b_c = None
    # Only fit Tafel if NOT in limiting current region
    # If i_Lc is detected, fit Tafel between Ecorr and start of limiting
    lower_limit_c = E_range_Lc[1] if i_Lc is not None else E_cath.min()
    
    # Usually Tafel is Ecorr - 0.02 to Ecorr - 0.25, or until Limiting starts
    mask_c = (E_cath < (E_corr - 0.02)) & (E_cath > max(lower_limit_c, E_corr - 0.3))
    
    if np.sum(mask_c) > 5:
        s_c, int_c, _, _, _ = linregress(np.log10(i_cath[mask_c]), E_cath[mask_c])
        b_c = abs(s_c)

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # Raw Data
    ax.semilogy(E, np.abs(i_meas), 'k-', lw=1.5, label='Experimental Data', alpha=0.8)
    ax.axvline(E_corr, color='gray', linestyle='--', alpha=0.5)

    # 1. REGION SHADING
    # Cathodic
    if len(E_cath) > 0:
        ax.axvspan(E_cath.min(), E_corr, color='blue', alpha=0.05)
    # Anodic Active
    if E_pp is not None:
        ax.axvspan(E_corr, E_pp, color='red', alpha=0.1, label='Anodic Active')
        ax.plot(E_pp, i_crit, 'ro') 
    else:
        ax.axvspan(E_corr, E_corr + 0.3, color='red', alpha=0.05)
    # Passive
    if E_pp is not None:
        limit = E_bd if E_bd is not None else E_anod.max()
        ax.axvspan(E_pp, limit, color='green', alpha=0.1, label='Passive Region')

    # 2. LIMITING CURRENT (Extrapolation)
    if i_Lc is not None:
        # Draw the actual detected plateau region thick
        ax.plot([E_range_Lc[0], E_range_Lc[1]], [i_Lc, i_Lc], color='purple', lw=3, label=f'Limiting Current ($i_L$)')
        # Extrapolate horizontally across the whole plot
        ax.axhline(i_Lc, color='purple', linestyle='--', alpha=0.6, linewidth=1)
        ax.text(E.min(), i_Lc*1.1, f" $i_L$ = {i_Lc:.1e}", color='purple', fontsize=10)

    # 3. TAFEL FITS
    if b_a is not None:
        x = np.linspace(E_corr, E_corr+0.2, 50)
        y = 10**((x - int_a)/s_a)
        ax.semilogy(x, y, 'r--', lw=1, label=f'b_a={b_a*1000:.0f}mV')
    if b_c is not None:
        x = np.linspace(E_corr-0.2, E_corr, 50)
        y = 10**((x - int_c)/s_c) # s_c is negative
        ax.semilogy(x, y, 'b--', lw=1, label=f'b_c={b_c*1000:.0f}mV')

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (|A|/cm²)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc='lower right')
    st.pyplot(fig)

    # --- RESULTS TABLE ---
    st.subheader("Extracted Parameters")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        st.markdown("### 🔵 Cathodic / Diffusion")
        st.metric("E_corr (V)", f"{E_corr:.3f}")
        if i_Lc is not None:
            st.metric("Limiting Current ($i_L$) (A/cm²)", f"{i_Lc:.2e}")
        else:
            st.caption("No limiting plateau detected.")
        if b_c is not None:
            st.metric("Cathodic Slope ($b_c$) (mV/dec)", f"{b_c*1000:.0f}")

    with c2:
        st.markdown("### 🔴 Anodic Active")
        if E_pp is not None:
            st.metric("E_pp (Passivation) (V)", f"{E_pp:.3f}")
            st.metric("i_crit (A/cm²)", f"{i_crit:.2e}")
        if b_a is not None:
            st.metric("Anodic Slope ($b_a$) (mV/dec)", f"{b_a*1000:.0f}")

    with c3:
        st.markdown("### 🟢 Passive / Breakdown")
        if i_pass_mean is not None:
            st.metric("i_pass (Passive) (A/cm²)", f"{i_pass_mean:.2e}")
        if E_bd is not None:
            st.metric("E_bd (Breakdown) (V)", f"{E_bd:.3f}")
        else:
            st.caption("No breakdown detected.")
