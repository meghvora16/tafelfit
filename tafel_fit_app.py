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

st.title("Global Implicit Tafel Fit")

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

def longest_linear_tafel_region(E, i_meas, Ecorr, anodic=True, min_size=6, r2_threshold=0.995):
    if anodic:
        mask = (E > Ecorr) & (i_meas > 0)
    else:
        mask = (E < Ecorr) & (i_meas < 0)
    indices = np.where(mask)[0]
    best_len = 0
    best_seg = None
    for start in range(len(indices)):
        for end in range(start + min_size, len(indices) + 1):
            idx_window = indices[start:end]
            if len(idx_window) < min_size:
                continue
            logi = np.log10(np.abs(i_meas[idx_window]) + 1e-15)
            fit = linregress(E[idx_window], logi)
            fitvals = fit.intercept + fit.slope * E[idx_window]
            ss_res = np.sum((logi - fitvals) ** 2)
            ss_tot = np.sum((logi - np.mean(logi)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0
            if r2 > r2_threshold and len(idx_window) > best_len:
                best_len = len(idx_window)
                best_seg = idx_window
    if best_seg is not None:
        return best_seg
    else:
        return np.array([], dtype=int)

# --------- Automatic plateau detection ---------
def diffusion_plateau_mask(E, i_meas, Ecorr, anodic=True, slope_tol=0.04, r2_min=0.98, window_size=7):
    """
    Returns boolean mask for diffusion plateau region selected automatically:
    - For anodic: uses last valid plateau window (rightmost "flat" segment after Ecorr).
    - For cathodic: uses first valid plateau window (leftmost "flat" segment before Ecorr).
    """
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
    plateau_mask = np.zeros(N, dtype=bool)
    if anodic:
        # Last window
        last_win = plateau_windows[-1]
        plateau_mask[last_win[0]:] = True
    else:
        # First window
        first_win = plateau_windows[0]
        plateau_mask[:first_win[-1]+1] = True
    return plateau_mask

# --------- UI ---------
data_file = st.file_uploader("Upload polarization data (CSV/Excel)", type=["csv", "xlsx", "xls"])
plateau_slope_tol = st.slider("Log plateau slope (for diffusion plateau)", min_value=0.01, max_value=0.10, value=0.04, step=0.01)
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

    # ---- Fit & simulated curve (as in your code, not changed) ----
    log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ru_guess = -6, 0.5, -8, 0.5, -4, 0
    x0 = np.array([log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr_guess, Ru_guess])
    maskB = (E >= Ecorr_guess - 0.3) & (E <= Ecorr_guess + 0.3)
    E_B, i_B = downsample(E[maskB], i_meas[maskB], 120)
    def residuals_B(x):
        pars_local = {
            "i0_a": 10 ** x[0], "alpha_a": x[1],
            "i0_c": 10 ** x[2], "alpha_c": x[3],
            "iL": 10 ** x[4], "Ecorr": x[5],
            "Ru": max(x[6], 0)
        }
        i_model = simulate_curve(E_B, pars_local)
        mask = np.isfinite(i_model)
        eps = 1e-15
        r = (np.log10(np.abs(i_model[mask]) + eps) -
             np.log10(np.abs(i_B[mask]) + eps))
        return r

    resB = least_squares(
        residuals_B, x0,
        bounds=([-12, 0.3, -12, 0.3, -6, Ecorr_guess - 0.2, 0],
                [-2, 0.7, -3, 0.7, -3, Ecorr_guess + 0.2, 200]),
        loss="soft_l1", f_scale=0.2, max_nfev=500
    )
    x = resB.x
    pars = {
        "i0_a": 10 ** x[0], "alpha_a": x[1],
        "i0_c": 10 ** x[2], "alpha_c": x[3],
        "iL": 10 ** x[4], "Ecorr": x[5],
        "Ru": max(x[6], 0)
    }
    st.subheader("Extracted Parameters")
    st.json(pars)
    beta_a = beta_from_alpha(pars["alpha_a"])
    beta_c = beta_from_alpha(pars["alpha_c"])
    i_corr = abs(newton_current_for_E(pars["Ecorr"], pars))
    st.write(f"β_a = {beta_a:.3f} V/dec, β_c = {beta_c:.3f} V/dec")
    st.write(f"i_corr = {i_corr:.3e} A/cm²")
    st.write(f"Fitted Ecorr = **{pars['Ecorr']:.3f} V** (data-driven guess: {Ecorr_guess:.3f} V)")

    # --------- Automated region detection ---------
    anodic_diff_mask = diffusion_plateau_mask(E, i_meas, Ecorr_guess,
        anodic=True, slope_tol=plateau_slope_tol, r2_min=r2_min, window_size=window_size)
    cathodic_diff_mask = diffusion_plateau_mask(E, i_meas, Ecorr_guess,
        anodic=False, slope_tol=plateau_slope_tol, r2_min=r2_min, window_size=window_size)
    region_labels = np.full(len(E), "active", dtype=object)
    region_labels[anodic_diff_mask] = "anodic_diffusion"
    region_labels[cathodic_diff_mask] = "cathodic_diffusion"

    # For plotting span and start
    if np.any(anodic_diff_mask):
        anodic_plateau_start_E = E[np.where(anodic_diff_mask)[0][0]]
        anodic_plateau_end_E = E[np.where(anodic_diff_mask)[0][-1]]
    else:
        anodic_plateau_start_E = anodic_plateau_end_E = None
    if np.any(cathodic_diff_mask):
        cathodic_plateau_start_E = E[np.where(cathodic_diff_mask)[0][0]]
        cathodic_plateau_end_E = E[np.where(cathodic_diff_mask)[0][-1]]
    else:
        cathodic_plateau_start_E = cathodic_plateau_end_E = None

    # --- Main plot: |i| vs E with shaded regions and plateau start ---
    fig, ax = plt.subplots(figsize=(7, 5))
    # Anodic diffusion region
    if anodic_plateau_start_E is not None and anodic_plateau_end_E is not None:
        ax.axvspan(anodic_plateau_start_E, anodic_plateau_end_E, color='yellow', alpha=0.25, label="Anodic diffusion region")
        ax.axvline(anodic_plateau_start_E, color='orange', lw=2, linestyle='--', label="Anodic plateau start")
    # Cathodic diffusion region
    if cathodic_plateau_start_E is not None and cathodic_plateau_end_E is not None:
        ax.axvspan(cathodic_plateau_start_E, cathodic_plateau_end_E, color='yellow', alpha=0.25, label="Cathodic diffusion region")
        ax.axvline(cathodic_plateau_end_E, color='darkred', lw=2, linestyle='--', label="Cathodic plateau end")
    maskB = (E >= Ecorr_guess - 0.3) & (E <= Ecorr_guess + 0.3)
    E_B, i_B = downsample(E[maskB], i_meas[maskB], 120)
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas) + 1e-12), s=0.001)
    i_smooth = 10 ** spl(E_grid)
    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Fit")
    ax.axvline(Ecorr_guess, color="blue", linestyle="--", lw=2, label="Ecorr")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel(r"$|i|$ (A/cm²)")
    ax.grid(True, which="both", ls='--', alpha=0.6)
    ax.legend(fontsize=10, loc="lower right")
    st.pyplot(fig)

    # --- Log(|i|) plot region visualization ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    logi = np.log10(np.abs(i_meas) + 1e-15)
    active_mask = (region_labels == "active")
    ax2.plot(E[active_mask], logi[active_mask], "o", color="orange", label="Active region", markersize=5)
    if np.any(anodic_diff_mask):
        ax2.plot(E[anodic_diff_mask], logi[anodic_diff_mask], "o", color="forestgreen", label="Anodic diffusion region", markersize=6)
        ax2.axhline(np.mean(logi[anodic_diff_mask]), color="red", linestyle="--", lw=2, label="Anodic plateau (mean)")
        ax2.axvline(anodic_plateau_start_E, color='orange', linestyle='--', lw=2, label="Anodic plateau start")
    if np.any(cathodic_diff_mask):
        ax2.plot(E[cathodic_diff_mask], logi[cathodic_diff_mask], "o", color="maroon", label="Cathodic diffusion region", markersize=6)
        ax2.axhline(np.mean(logi[cathodic_diff_mask]), color="navy", linestyle="--", lw=2, label="Cathodic plateau (mean)")
        ax2.axvline(cathodic_plateau_end_E, color='darkred', linestyle='--', lw=2, label="Cathodic plateau end")
    ax2.axvline(Ecorr_guess, color="blue", linestyle="--", lw=2, label="Ecorr")
    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("log |i| (A/cm²)")
    ax2.grid(True, which="both", ls='--', alpha=0.5)
    ax2.legend(loc="lower right", fontsize=10)
    st.pyplot(fig2)

    # --- Raw data plot ---
    fig_raw, ax_raw = plt.subplots(figsize=(7, 5))
    ax_raw.plot(E, np.log10(np.abs(i_meas) + 1e-15), "ko", ms=4)
    ax_raw.set_xlabel("Potential (V)")
    ax_raw.set_ylabel("log |i| (A/cm²)")
    ax_raw.grid(True, which="both", ls='--', alpha=0.5)
    ax_raw.set_title("Raw Tafel Plot: log(|i|) vs. Potential")
    st.pyplot(fig_raw)

    st.info(
        "Yellow regions = diffusion plateau (automatically detected, not hardcoded). "
        "Orange dashed line = start of anodic plateau. Adjust sliders to fine-tune detection. "
        "Other overlays/fits as in original code."
    )
