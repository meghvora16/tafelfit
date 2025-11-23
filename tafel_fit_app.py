import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.stats import linregress

st.set_page_config(page_title="Full Physics Polarization", layout="wide")
st.title("Full Physics Polarization Analysis")

st.markdown("""
**Analysis Logic:**
1.  **Passivation Check:** Is there a peak ($E_{pp}$) followed by a drop?
2.  **Diffusion Check ($i_L$):** Is there a flat plateau at the end of the scan?
3.  **Activation Check (Tafel):** Find the straightest line in the remaining active regions.
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
    Checks if the 'tail' (end) of a segment is a flat plateau.
    Returns: (limiting_current_value, E_start_of_plateau) or (None, None)
    """
    n = len(i_segment)
    if n < 20: return None, None
    
    # Analyze the last 20% of points (furthest from Ecorr typically)
    n_tail = int(n * window_fraction)
    # We assume input arrays are sorted such that the 'end' is the tail
    # For Anodic: End is highest E. For Cathodic: End is lowest E? 
    # Note: usage below will ensure we pass the "end" of the scan as the end of the array.
    
    tail_i = i_segment[-n_tail:]
    tail_E = E_segment[-n_tail:]
    
    mean_i = np.mean(tail_i)
    std_i = np.std(tail_i)
    
    # Check for flatness (Std Dev is small fraction of Mean)
    # Also ensure mean_i is not effectively zero (noise)
    if abs(mean_i) > 1e-12 and (std_i / abs(mean_i)) < tolerance:
        return mean_i, tail_E[0]
    return None, None

def get_best_linear_segment(E_segment, log_i_segment, min_points=10):
    """Finds the segment with highest R^2."""
    n = len(E_segment)
    if n < min_points: return None, None, None, None
    
    candidates = []
    window_size = int(n * 0.4)
    if window_size < min_points: window_size = n 
    
    steps = range(0, n - window_size + 1, max(1, int(n/20))) 
    
    for start_i in steps:
        end_i = start_i + window_size
        x = E_segment[start_i:end_i]
        y = log_i_segment[start_i:end_i]
        
        # Linregress (E vs log i) -> E = b*log(i) + a
        # Note: R value is correlation.
        slope, intercept, r_val, _, _ = linregress(y, x) 
        candidates.append((r_val**2, start_i, end_i, slope, intercept))
    
    if not candidates:
        slope, intercept, r_val, _, _ = linregress(log_i_segment, E_segment)
        return slope, intercept, r_val**2, np.ones(n, dtype=bool)

    candidates.sort(key=lambda x: x[0], reverse=True)
    best_r2, start_idx, end_idx, best_slope, best_int = candidates[0]
    
    mask = np.zeros(n, dtype=bool)
    mask[start_idx:end_idx] = True
    
    return best_slope, best_int, best_r2, mask

# --- MAIN APP ---
data_file = st.file_uploader("Upload LSV/Polarization data (CSV/Excel)", type=["csv", "xlsx", "xls"])

if data_file is not None:
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

    # --- 1. ANALYZE CATHODIC BRANCH ---
    # Branch: E < Ecorr
    mask_c = E < E_corr
    E_c = E[mask_c]
    i_c = np.abs(i_meas[mask_c])
    
    # Sort descending E (from Ecorr down to most negative) for processing?
    # Usually arrays are sorted by E ascending (-1.0, -0.9 ...).
    # E_c will be [-1.0, -0.9 ... Ecorr].
    # The "Tail" is at the beginning of this array.
    # Let's reverse it so index 0 is Ecorr and index -1 is the deep tail.
    if len(E_c) > 10:
        E_c_rev = E_c[::-1] # Now starts near Ecorr, ends at -1.0V
        i_c_rev = i_c[::-1]
        
        # A. LIMITING CURRENT CHECK
        i_L_c, E_plateau_start_c = detect_limiting_current(E_c_rev, i_c_rev)
        
        # B. TAFEL FIT
        # Search zone: Start near Ecorr, Stop before Plateau (if exists)
        search_end_idx = len(E_c_rev)
        if i_L_c is not None:
            # Find index where E matches E_plateau_start_c
            # Since we know E_plateau_start_c came from the tail, we crop there.
            # Ideally, crop a bit before the plateau starts to avoid the bend.
            pass # The search function handles the mask
            
        # Define mask for Tafel Search
        # Logic: Between Ecorr-20mV and (Plateau Start OR End of Scan)
        start_E_tafel = E_corr - 0.02
        end_E_tafel = E_plateau_start_c + 0.05 if i_L_c else E_c_rev[-1]
        
        mask_search_c = (E_c_rev < start_E_tafel) & (E_c_rev > end_E_tafel)
        
        b_c, i_corr_c, r2_c, decades_c = None, None, 0, 0
        E_c_fit, i_c_fit = None, None
        
        if np.sum(mask_search_c) > 10:
            E_use = E_c_rev[mask_search_c]
            log_i_use = np.log10(i_c_rev[mask_search_c])
            
            slope, intercept, r2, mask_local = get_best_linear_segment(E_use, log_i_use)
            if slope is not None:
                b_c = abs(slope)
                i_corr_c = 10**((E_corr - intercept)/slope)
                r2_c = r2
                E_c_fit = E_use[mask_local]
                i_c_fit = 10**log_i_use[mask_local]
                decades_c = np.log10(i_c_fit.max()) - np.log10(i_c_fit.min())

    # --- 2. ANALYZE ANODIC BRANCH ---
    mask_a = E > E_corr
    E_a = E[mask_a]
    i_a = i_meas[mask_a]
    
    E_pp, i_crit, E_bd = None, None, None
    b_a, i_corr_a, r2_a, decades_a = None, None, 0, 0
    E_a_fit, i_a_fit = None, None
    i_L_a = None # Anodic limiting current (rare but possible)

    if len(E_a) > 10:
        i_smooth = smooth_curve(i_a, 21)
        
        # A. PASSIVATION DETECTION
        peaks = argrelextrema(i_smooth, np.greater)[0]
        valid_peaks = [p for p in peaks if i_smooth[p] > 0]
        
        if valid_peaks:
            best_p = valid_peaks[np.argmax(i_smooth[valid_peaks])]
            E_pp = E_a[best_p]
            i_crit = E_a[best_p] # Typo fix: i_crit should be current
            i_crit = i_a[best_p] 
            
            # Breakdown
            mask_pass = E_a > E_pp
            if np.any(mask_pass):
                i_p = i_smooth[mask_pass]
                E_p = E_a[mask_pass]
                min_idx = np.argmin(i_p)
                i_pass_min = i_p[min_idx]
                rise = (i_p > i_pass_min*5) & (E_p > E_p[min_idx])
                if np.any(rise): E_bd = E_p[rise][0]
        
        # B. TAFEL FIT
        # Search Zone: Ecorr + 20mV to (E_pp or End)
        start_E_tafel = E_corr + 0.02
        end_E_tafel = E_pp - 0.02 if E_pp else E_a[-1]
        
        # Limit Anodic fit to avoid Limiting Current if present?
        # Check for Anodic Limiting Current (only if no passivation detected usually)
        if not E_pp:
            i_L_a, E_plat_a = detect_limiting_current(E_a, i_a)
            if i_L_a: end_E_tafel = E_plat_a
            
        mask_search_a = (E_a > start_E_tafel) & (E_a < end_E_tafel)
        
        if np.sum(mask_search_a) > 10:
            E_use = E_a[mask_search_a]
            # Filter positive
            valid_i = i_a[mask_search_a] > 0
            if np.sum(valid_i) > 10:
                E_use = E_use[valid_i]
                log_i_use = np.log10(i_a[mask_search_a][valid_i])
                
                slope, intercept, r2, mask_local = get_best_linear_segment(E_use, log_i_use)
                if slope is not None:
                    b_a = slope
                    i_corr_a = 10**((E_corr - intercept)/slope)
                    r2_a = r2
                    E_a_fit = E_use[mask_local]
                    i_a_fit = 10**log_i_use[mask_local]
                    decades_a = np.log10(i_a_fit.max()) - np.log10(i_a_fit.min())

    # --- PLOTTING ---
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 1. Shading
    ax.axvspan(E.min(), E_corr, color='blue', alpha=0.05) # Cathodic
    if E_pp:
        ax.axvspan(E_corr, E_pp, color='red', alpha=0.05) # Active
        limit = E_bd if E_bd else E.max()
        ax.axvspan(E_pp, limit, color='green', alpha=0.05) # Passive
    else:
        ax.axvspan(E_corr, E.max(), color='red', alpha=0.05) # Active

    # 2. Data
    ax.semilogy(E, np.abs(i_meas), 'k.', markersize=2, alpha=0.4, label='Data')
    ax.axvline(E_corr, color='gray', ls='--', alpha=0.5)

    # 3. Limiting Currents (Purple)
    if 'i_L_c' in locals() and i_L_c is not None:
        ax.axhline(i_L_c, color='purple', ls='--', lw=1.5, label=f'Cathodic Limiting ($i_L$)')
        # Draw the plateau segment physically
        # E_plateau_start_c is where it starts. It ends at E_c.min()
        ax.plot([E_c.min(), E_plateau_start_c], [i_L_c, i_L_c], color='purple', lw=3)
        
    if 'i_L_a' in locals() and i_L_a is not None:
        ax.axhline(i_L_a, color='purple', ls='--', lw=1.5)
        ax.plot([E_plat_a, E_a.max()], [i_L_a, i_L_a], color='purple', lw=3)

    # 4. Tafel Fits
    if b_c is not None:
        ax.semilogy(E_c_fit, i_c_fit, 'b-', lw=3, label=f'Cathodic Tafel')
        # Extrapolate
        y_ex = i_corr_c * 10**((E_corr - E)/b_c)
        mask_ex = (E < E_corr) & (E > E_corr - 0.3)
        ax.semilogy(E[mask_ex], y_ex[mask_ex], 'b--', lw=1, alpha=0.5)
        
    if b_a is not None:
        ax.semilogy(E_a_fit, i_a_fit, 'r-', lw=3, label=f'Anodic Tafel')
        y_ex = i_corr_a * 10**((E - E_corr)/b_a)
        mask_ex = (E > E_corr) & (E < E_corr + 0.3)
        ax.semilogy(E[mask_ex], y_ex[mask_ex], 'r--', lw=1, alpha=0.5)

    if E_pp: ax.plot(E_pp, i_crit, 'ro')
    if E_bd: ax.axvline(E_bd, color='orange', ls=':', lw=2, label='Breakdown')

    ax.set_xlabel(f"Potential ({pot_units})")
    ax.set_ylabel("Current Density (A/cm²)")
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, which="both", alpha=0.2)
    st.pyplot(fig)

    # --- METRICS ---
    st.subheader("Extracted Parameters")
    m1, m2, m3 = st.columns(3)
    
    with m1:
        st.markdown("### 🔵 Cathodic")
        if 'i_L_c' in locals() and i_L_c:
            st.metric("Limiting Current ($i_L$)", f"{i_L_c:.2e}")
        if b_c:
            st.metric("Beta C", f"{b_c*1000:.1f} mV/dec")
            st.metric("Corrosion Current ($i_{corr}$)", f"{i_corr_c:.2e}")
            
    with m2:
        st.markdown("### 🔴 Anodic")
        if b_a:
            st.metric("Beta A", f"{b_a*1000:.1f} mV/dec")
            st.metric("Corrosion Current ($i_{corr}$)", f"{i_corr_a:.2e}")
            
    with m3:
        st.markdown("### ⚙️ Passivation")
        if E_pp:
            st.metric("Primary Passivation Pot.", f"{E_pp:.3f} V")
            st.metric("Critical Current ($i_{crit}$)", f"{i_crit:.2e}")
            if E_bd: st.metric("Breakdown Pot.", f"{E_bd:.3f} V")
