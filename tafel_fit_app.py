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

data_file = st.file_uploader("Upload polarization data (CSV/Excel)", type=["csv", "xlsx", "xls"])

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

    # ---- Global fit ----
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
    #st.write(f"Fitted Ecorr = **{pars['Ecorr']:.3f} V** (data-driven guess: {Ecorr_guess:.3f} V)")

    # Fit curve for main plot
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas) + 1e-12), s=0.001)
    i_smooth = 10 ** spl(E_grid)

    # -------- REGION DEFINITION BASED ON LOG(|i|) --------
    logi = np.log10(np.abs(i_meas) + 1e-15)
    anodic_mask = (E > Ecorr_guess) & (i_meas > 0)
    cathodic_mask = (E < Ecorr_guess) & (i_meas < 0)

    # Tafel: log(|i|) between -8 and -7
    anodic_tafel_mask = anodic_mask & (logi > -8) & (logi <= -7)
    cathodic_tafel_mask = cathodic_mask & (logi > -8) & (logi <= -7)
    # Anodic diffusion: log(|i|) > -7
    anodic_diff_mask = anodic_mask & (logi > -7)

    # Potential boundaries
    if np.any(anodic_tafel_mask):
        anodic_tafel_start_E = E[anodic_tafel_mask][0]
        anodic_tafel_end_E = E[anodic_tafel_mask][-1]
    else:
        anodic_tafel_start_E = anodic_tafel_end_E = None

    if np.any(anodic_diff_mask):
        anodic_diff_start_E = E[anodic_diff_mask][0]
        anodic_diff_end_E = E[anodic_diff_mask][-1]
    else:
        anodic_diff_start_E = anodic_diff_end_E = None

    if np.any(cathodic_tafel_mask):
        cathodic_tafel_start_E = E[cathodic_tafel_mask][0]
        cathodic_tafel_end_E = E[cathodic_tafel_mask][-1]
    else:
        cathodic_tafel_start_E = cathodic_tafel_end_E = None

    # Ecorr (magenta)
    ecorr_window = 0.03
    ecorr_bounds = (Ecorr_guess - ecorr_window, Ecorr_guess + ecorr_window)

    # --- Main plot: |i| vs E with shaded regions
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.axvspan(*ecorr_bounds, color='magenta', alpha=0.14, label="Ecorr region")
    if cathodic_tafel_start_E is not None and cathodic_tafel_end_E is not None:
        ax.axvspan(cathodic_tafel_start_E, cathodic_tafel_end_E, color='blue', alpha=0.14, label="Cathodic Tafel region")
    if anodic_tafel_start_E is not None and anodic_tafel_end_E is not None:
        ax.axvspan(anodic_tafel_start_E, anodic_tafel_end_E, color='red', alpha=0.14, label="Anodic Tafel region")
    if anodic_diff_start_E is not None and anodic_diff_end_E is not None:
        ax.axvspan(anodic_diff_start_E, anodic_diff_end_E, color='yellow', alpha=0.24, label="Anodic diffusion-limited region")
        ax.axvline(anodic_diff_start_E, color='orange', linestyle='--', lw=2, label='Anodic diffusion start')

    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Fit")
    ax.axvline(Ecorr_guess, color="blue", linestyle="--", label="Ecorr")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel(r"$|i|$ (A/cm²)")
    ax.grid(True, which="both")
    ax.legend(loc="lower right", fontsize=9)
    st.pyplot(fig)

    # --- Log(|i|) plot: show fits on both Tafel regions ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    ax2.plot(E, logi, "k.", label="log(|i|) data")
    if cathodic_tafel_start_E is not None and cathodic_tafel_end_E is not None:
        ax2.axvspan(cathodic_tafel_start_E, cathodic_tafel_end_E, color='blue', alpha=0.14, label="Cathodic Tafel region")
        # Linear fit for cathodic region
        tafel_x_c = E[cathodic_tafel_mask]
        tafel_y_c = logi[cathodic_tafel_mask]
        if len(tafel_x_c) > 2:
            slope_c, intercept_c, r_c, p_c, stderr_c = linregress(tafel_x_c, tafel_y_c)
            ax2.plot(tafel_x_c, slope_c*tafel_x_c+intercept_c, "b--", linewidth=2, label="Cathodic Tafel fit")
    if anodic_tafel_start_E is not None and anodic_tafel_end_E is not None:
        ax2.axvspan(anodic_tafel_start_E, anodic_tafel_end_E, color='red', alpha=0.14, label="Anodic Tafel region")
        # Linear fit for anodic region
        tafel_x = E[anodic_tafel_mask]
        tafel_y = logi[anodic_tafel_mask]
        if len(tafel_x) > 2:
            slope, intercept, r, p, stderr = linregress(tafel_x, tafel_y)
            ax2.plot(tafel_x, slope*tafel_x+intercept, "r--", linewidth=2, label="Anodic Tafel fit")
    if anodic_diff_start_E is not None and anodic_diff_end_E is not None:
        ax2.axvspan(anodic_diff_start_E, anodic_diff_end_E, color='yellow', alpha=0.24, label="Anodic diffusion-limited region")
        ax2.axvline(anodic_diff_start_E, color='orange', linestyle='--', lw=2, label='Anodic diffusion start')
    ax2.axvline(Ecorr_guess, color="blue", linestyle="--", label="Ecorr")
    ax2.set_xlabel("Potential (V)")
    ax2.set_ylabel("log |i| (A/cm²)")
    ax2.grid(True, which="both")
    ax2.legend(loc="lower right", fontsize=9)
    st.pyplot(fig2)

    # --- Raw Tafel plot: log(|i|) vs E (just data, no overlays) ---
    fig_raw, ax_raw = plt.subplots(figsize=(7, 5))
    ax_raw.plot(E, np.log10(np.abs(i_meas) + 1e-15), "ko", ms=4, label="Raw data")
    ax_raw.set_xlabel("Potential (V)")
    ax_raw.set_ylabel("log |i| (A/cm²)")
    ax_raw.grid(True, which="both")
    ax_raw.legend(loc="best")
    ax_raw.set_title("Raw Tafel Plot: log(|i|) vs. Potential")
    st.pyplot(fig_raw)

    st.info(
        "Blue region: Cathodic Tafel region (1e-8 to 1e-7 A/cm²). "
        "Red region: Anodic Tafel region (1e-8 to 1e-7 A/cm²). "
        "Dashed blue/red lines: Tafel fits."
        "Yellow region: Anodic diffusion-limited region (>1e-7 A/cm²). "
        "Orange dashed line: Start of diffusion-limited region."
    )
