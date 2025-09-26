import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from scipy.stats import linregress

st.set_page_config(page_title="Global Implicit Tafel Fit", layout="wide")

F = 96485.33212
R = 8.314462618

def beta_from_alpha(alpha, n=1, T=298.15):
    return 2.303 * R * T / (max(alpha, 1e-6) * n * F)

def newton_current_for_E(E, pars, T=298.15, n=1, i_init=None):
    try:
        i0_a = pars["i0_a"]
        alpha_a = pars["alpha_a"]
        i0_c = pars["i0_c"]
        alpha_c = pars["alpha_c"]
        iL = pars["iL"]
        Ecorr = pars["Ecorr"]
        Ru = pars["Ru"]
    except KeyError:
        return np.nan
    i = 0.0 if i_init is None else float(i_init)
    k_a = (alpha_a * n * F) / (R * T)
    k_c = (alpha_c * n * F) / (R * T)
    for _ in range(10):
        try:
            eta = E - Ecorr - i * Ru
            i_a = i0_a * math.exp(k_a * eta)
            i_c_act = -i0_c * math.exp(-k_c * eta)
        except OverflowError:
            return np.nan
        denom = (i_c_act - iL)
        if abs(denom) < 1e-30:
            denom = 1e-30
        i_c = (i_c_act * -iL) / denom
        f = i - (i_a + i_c)
        di_a_deta = i_a * k_a
        di_cact_deta = (-i_c_act) * k_c
        di_c_dg = (iL ** 2) / (denom ** 2)
        di_c_deta = di_c_dg * di_cact_deta
        dfi = 1 - (di_a_deta + di_c_deta) * -Ru
        step = -f / (dfi + 1e-30)
        i += step * 0.5
        if abs(f) < 1e-12:
            break
    return i

def simulate_curve(E_arr, pars, T=298.15, n=1):
    out = []
    i_guess = 0.0
    for E in E_arr:
        val = newton_current_for_E(E, pars, T=T, n=n, i_init=i_guess)
        if not np.isfinite(val):
            val = np.nan
        i_guess = val if np.isfinite(val) else 0.0
        out.append(val)
    return np.array(out)

def downsample(E, I, n_points=100):
    if len(E) <= n_points:
        return E, I
    idx = np.linspace(0, len(E) - 1, n_points, dtype=int)
    return E[idx], I[idx]

def find_plateau_mask(E, i_meas, Ecorr, anodic=True, slope_tol=0.04, r2_min=0.98, window_size=7):
    """Returns Boolean mask for plateau (diffusion) region on either side of Ecorr."""
    if anodic:
        mask = (E > Ecorr) & (i_meas > 0)
    else:
        mask = (E < Ecorr) & (i_meas < 0)
    indices = np.where(mask)[0]
    logi = np.log10(np.abs(i_meas) + 1e-15)
    N = len(E)
    plateau_windows = []
    for start in range(len(indices) - window_size + 1):
        idx = indices[start:start+window_size]
        xw = E[idx]
        yw = logi[idx]
        slope, intercept, r, p, stderr = linregress(xw, yw)
        if abs(slope) < slope_tol and r**2 > r2_min:
            plateau_windows.append(idx)
    if not plateau_windows:
        return np.zeros(N, dtype=bool)
    # For anodic use last window; for cathodic use first
    if anodic:
        chosen_window = plateau_windows[-1]
        plateau_mask = np.zeros(N, dtype=bool)
        plateau_mask[chosen_window[0]:] = True
        return plateau_mask
    else:
        chosen_window = plateau_windows[0]
        plateau_mask = np.zeros(N, dtype=bool)
        plateau_mask[:chosen_window[-1]+1] = True
        return plateau_mask

# ----- Streamlit UI -----

st.title("Classify plot regions")

data_file = st.file_uploader("Upload polarization data (CSV/Excel)", type=["csv", "xlsx", "xls"])
plateau_slope_tol = st.slider("Plateau slope tolerance", min_value=0.005, max_value=0.10, value=0.04, step=0.005)
r2_min = st.slider("Plateau min R²", min_value=0.95, max_value=0.999, value=0.98, step=0.001)
window_size = st.slider("Plateau window size", min_value=4, max_value=15, value=7, step=1)

if data_file is not None:
    df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
    st.success(f"Loaded {len(df)} rows.")
    st.dataframe(df.head(8))

    col_E = st.selectbox("Potential column", df.columns)
    col_I = st.selectbox("Current column", df.columns)
    pot_units = st.selectbox("Potential units", ["V", "mV"], 0)
    cur_units = st.selectbox("Current units", ["A", "mA", "uA", "nA"], 1)

    area_val = st.number_input("Electrode area (cm²)", value=1.0)
    area_arr = np.full(len(df), area_val)

    E_raw = df[col_E].astype(float).to_numpy()
    if pot_units == "mV":
        E_raw /= 1000
    I_raw = df[col_I].astype(float).to_numpy()
    I = I_raw * {"A": 1, "mA": 1e-3, "uA": 1e-6, "nA": 1e-9}[cur_units]
    i_meas = I / area_arr

    idx = np.argsort(E_raw)
    E = E_raw[idx]
    i_meas = i_meas[idx]

    # --- Auto-detect Ecorr ---
    sign = np.sign(i_meas)
    zc = np.where(np.diff(sign) != 0)[0]
    if len(zc):
        j = zc[0]
        Ecorr_guess = E[j] - i_meas[j] * (E[j + 1] - E[j]) / (i_meas[j + 1] - i_meas[j])
    else:
        Ecorr_guess = E[np.argmin(np.abs(i_meas))]

    st.write(f"Data-driven Ecorr ≈ **{Ecorr_guess:.3f} V**")

    # Find diffusion (plateau) regions
    anodic_diff_mask = find_plateau_mask(E, i_meas, Ecorr_guess, anodic=True, slope_tol=plateau_slope_tol, r2_min=r2_min, window_size=window_size)
    cathodic_diff_mask = find_plateau_mask(E, i_meas, Ecorr_guess, anodic=False, slope_tol=plateau_slope_tol, r2_min=r2_min, window_size=window_size)

    # Assign regions according to rules
    region_labels = np.full(len(E), "ignored", dtype=object)
    region_labels[anodic_diff_mask] = "anodic_diffusion"
    region_labels[cathodic_diff_mask] = "cathodic_diffusion"
    # Active regions: between two diffusion regions
    # Anodic active: between Ecorr and start of anodic plateau
    anodic_active_mask = (~anodic_diff_mask) & (~cathodic_diff_mask) & (E > Ecorr_guess)
    region_labels[anodic_active_mask] = "anodic_active"
    # Cathodic active: between end of cathodic plateau and Ecorr
    cathodic_active_mask = (~anodic_diff_mask) & (~cathodic_diff_mask) & (E < Ecorr_guess)
    region_labels[cathodic_active_mask] = "cathodic_active"

    color_map = {
        "ignored": "gray",
        "cathodic_active": "dodgerblue",
        "anodic_active": "orange",
        "anodic_diffusion": "forestgreen",
        "cathodic_diffusion": "maroon"
    }
    marker_map = {
        "ignored": ".",
        "cathodic_active": "o",
        "anodic_active": "o",
        "anodic_diffusion": "o",
        "cathodic_diffusion": "o"
    }
    label_map = {
        "ignored": "Ignored data",
        "cathodic_active": "Cathodic active region",
        "anodic_active": "Anodic active region",
        "anodic_diffusion": "Anodic diffusion region",
        "cathodic_diffusion": "Cathodic diffusion region"
    }

    # Plateau value for horizontal line (anodic diffusion region)
    logi = np.log10(np.abs(i_meas) + 1e-15)
    if np.any(anodic_diff_mask):
        plateau_val = np.mean(logi[anodic_diff_mask])
    elif np.any(cathodic_diff_mask):
        plateau_val = np.mean(logi[cathodic_diff_mask])
    else:
        plateau_val = None

    # Make plot and explanation side by side
    colL, colR = st.columns([2,3])
    with colL:
        fig, ax = plt.subplots(figsize=(7, 5))
        # Plot regions as colored points
        for region in color_map.keys():
            mask = (region_labels == region)
            if np.any(mask):
                ax.plot(E[mask], logi[mask],
                        marker_map[region], color=color_map[region],
                        label=label_map[region],
                        linestyle="None", markersize=6 if region != "ignored" else 2)
        # Draw plateau line if found
        if plateau_val is not None:
            ax.axhline(plateau_val, color="red", linestyle="--", label="plateau")
        ax.set_xlabel("E/V")
        ax.set_ylabel(r"log$_{10}$(|i|/A)")
        ax.legend(fontsize=10)
        ax.grid(True)
        st.pyplot(fig)
    with colR:
        st.markdown("""
        ## Classify plot regions

        <span style="font-size:1.3em"><b>1</b></span> **Find plateaus**
        
        Chunk the data into subsections and fit a line to each. We only accept close to horizontal lines that explain the surrounding measurements well.

        <br>

        <span style="font-size:1.3em"><b>2</b></span> **Define active and diffusion regions**
        
        The diffusion regions are the last plateau on the cathodic and first on the anodic side.
        The active regions are measurements between the two diffusion regions.

        <br>

        <span style="font-size:1.3em"><b>3</b></span> **Remove tails**
        
        Since our models do not explain anything beyond the diffusion regions, we drop all those measurements.

        """, unsafe_allow_html=True)

    st.info(
        """**Legend:**  
        - Blue = cathodic active  
        - Orange = anodic active  
        - Green = anodic diffusion region  
        - Maroon = cathodic diffusion region  
        - Gray = ignored  
        - Red dashed line = detected plateau (mean log(|i|) in anodic diffusion region)
        """)
