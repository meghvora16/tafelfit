import io
import json
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.interpolate import UnivariateSpline
from sklearn.metrics import r2_score

st.set_page_config(page_title="Global Implicit Tafel Fit", layout="wide")

F = 96485.33212
R = 8.314462618

st.title("Global Implicit Tafel Fit (Single-Stage)")

def beta_from_alpha(alpha, n=1, T=298.15):
    return 2.303 * R * T / (max(alpha, 1e-6) * n * F)

# ---- Physics solver with overflow protection ----
def newton_current_for_E(E, pars, T=298.15, n=1, i_init=None):
    try:
        i0_a = pars["i0_a"]; alpha_a = pars["alpha_a"]
        i0_c = pars["i0_c"]; alpha_c = pars["alpha_c"]
        iL = pars["iL"]; Ecorr = pars["Ecorr"]; Ru = pars["Ru"]
    except KeyError:
        return np.nan

    i = 0.0 if i_init is None else float(i_init)
    k_a = (alpha_a * n * F) / (R * T)
    k_c = (alpha_c * n * F) / (R * T)

    for _ in range(10):  # 10 Newton steps max
        try:
            eta = E - Ecorr - i * Ru
            i_a = i0_a * math.exp(k_a * eta)
            i_c_act = - i0_c * math.exp(-k_c * eta)
        except OverflowError:
            return np.nan

        denom = (i_c_act - iL)
        if abs(denom) < 1e-30: denom = 1e-30
        i_c = (i_c_act * -iL) / denom
        f = i - (i_a + i_c)

        di_a_deta = i_a * k_a
        di_cact_deta = (-i_c_act) * k_c
        di_c_dg = (iL**2) / (denom**2)
        di_c_deta = di_c_dg * di_cact_deta
        dfi = 1 - (di_a_deta + di_c_deta) * -Ru

        step = -f / (dfi + 1e-30)
        i += step * 0.5  # damped step
        if abs(f) < 1e-12:
            break
    return i

def simulate_curve(E_arr, pars, T=298.15, n=1):
    out = []
    i_guess = 0.0
    for E in E_arr:
        val = newton_current_for_E(E, pars, T=T, n=n, i_init=i_guess)
        if not np.isfinite(val): val = np.nan
        i_guess = val if np.isfinite(val) else 0.0
        out.append(val)
    return np.array(out)

# ---- Utility: downsample ----
def downsample(E, I, n_points=100):
    if len(E) <= n_points:
        return E, I
    idx = np.linspace(0, len(E)-1, n_points, dtype=int)
    return E[idx], I[idx]

# ---- Upload ----
data_file = st.file_uploader("Upload polarization data (CSV/Excel).", type=["csv","xlsx","xls"])
if data_file is not None:
    df = pd.read_csv(data_file) if data_file.name.endswith(".csv") else pd.read_excel(data_file)
    st.success(f"Loaded {len(df)} rows."); st.dataframe(df.head(8))

    col_E = st.selectbox("Potential column", df.columns)
    col_I = st.selectbox("Current column", df.columns)
    pot_units = st.selectbox("Potential units", ["V","mV"], 0)
    cur_units = st.selectbox("Current units", ["A","mA","uA","nA"], 1)

    area_val = st.number_input("Electrode area (cm²)", value=1.0)
    area_arr = np.full(len(df), area_val)

    # Corrosion rate inputs
    st.markdown("### Corrosion rate inputs")
    ew_mode = st.radio("Provide:", ["Equivalent weight (g/eq)", "Atomic weight + valence"], index=0, horizontal=True)
    if ew_mode == "Equivalent weight (g/eq)":
        EW = st.number_input("Equivalent weight, EW (g/equiv)", value=27.92, min_value=0.0, help="For Fe (valence 2), EW ≈ 27.92 g/eq")
    else:
        M = st.number_input("Atomic weight, M (g/mol)", value=55.845, min_value=0.0)
        z = st.number_input("Valence (electrons per atom)", value=2, min_value=1, step=1)
        EW = M / max(z, 1)

    rho = st.number_input("Density, ρ (g/cm³)", value=7.87, min_value=0.0)

    # Data prep
    E_raw = df[col_E].astype(float).to_numpy()
    if pot_units == "mV": E_raw /= 1000
    I_raw = df[col_I].astype(float).to_numpy()
    I = I_raw * {"A":1,"mA":1e-3,"uA":1e-6,"nA":1e-9}[cur_units]
    i_meas = I / area_arr  # A/cm²

    idx = np.argsort(E_raw)
    E = E_raw[idx]; i_meas = i_meas[idx]

    # ---- Auto-detect Ecorr ----
    sign = np.sign(i_meas)
    zc = np.where(np.diff(sign) != 0)[0]
    if len(zc):
        j = zc[0]
        Ecorr_guess = E[j] - i_meas[j]*(E[j+1]-E[j])/(i_meas[j+1]-i_meas[j])
    else:
        Ecorr_guess = E[np.argmin(np.abs(i_meas))]
    st.write(f"Data-driven Ecorr ≈ **{Ecorr_guess:.3f} V**")

    # ---- Single-stage global fit ----
    log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ru_guess = -6, 0.5, -8, 0.5, -4, 0
    x0 = np.array([log_i0a, alpha_a, log_i0c, alpha_c, log_iL, Ecorr_guess, Ru_guess])

    # Fit window (you can widen this if desired)
    maskB = (E >= Ecorr_guess - 0.3) & (E <= Ecorr_guess + 0.3)
    E_B, i_B = downsample(E[maskB], i_meas[maskB], 120)

    bounds_lo_B = [-12, 0.3, -12, 0.3, -6, Ecorr_guess-0.2, 0]
    bounds_hi_B = [-2,  0.7,  -3,  0.7,  -3, Ecorr_guess+0.2, 200]

    def residuals_B(x):
        pars_local = {
            "i0_a":10**x[0], "alpha_a":x[1],
            "i0_c":10**x[2], "alpha_c":x[3],
            "iL":10**x[4], "Ecorr":x[5],
            "Ru":max(x[6], 0)
        }
        i_model = simulate_curve(E_B, pars_local)
        mask = np.isfinite(i_model)
        eps = 1e-15
        # log-magnitude residuals for robustness across decades
        r = (np.log10(np.abs(i_model[mask]) + eps) -
             np.log10(np.abs(i_B[mask]) + eps))
        return r

    resB = least_squares(
        residuals_B, x0,
        bounds=(bounds_lo_B, bounds_hi_B),
        loss="soft_l1", f_scale=0.2, max_nfev=500
    )
    x = resB.x
    pars = {
        "i0_a":10**x[0], "alpha_a":x[1],
        "i0_c":10**x[2], "alpha_c":x[3],
        "iL":10**x[4], "Ecorr":x[5],
        "Ru":max(x[6],0)
    }

    # ---- Results ----
    st.subheader("Extracted Parameters")
    st.json(pars)

    beta_a = beta_from_alpha(pars["alpha_a"])
    beta_c = beta_from_alpha(pars["alpha_c"])

    # i_corr here is taken as |i| at E = Ecorr from the implicit model
    i_corr = abs(newton_current_for_E(pars["Ecorr"], pars))  # A/cm²

    # Corrosion rate in mm/year:
    # CR(mm/yr) = 3270 * i_corr(A/cm²) * EW(g/eq) / rho(g/cm³)
    if np.isfinite(i_corr) and rho > 0 and EW > 0:
        CR_mm_per_yr = 3270.0 * i_corr * EW / rho
    else:
        CR_mm_per_yr = np.nan

    st.write(f"β_a = {beta_a:.3f} V/dec, β_c = {beta_c:.3f} V/dec")
    st.write(f"i_corr = {i_corr:.3e} A/cm²")
    st.write(f"Fitted Ecorr = **{pars['Ecorr']:.3f} V** (data-driven guess: {Ecorr_guess:.3f} V)")
    st.write(f"Corrosion rate = **{CR_mm_per_yr:.3f} mm/year**  (EW = {EW:.3f} g/eq, ρ = {rho:.3f} g/cm³)")

    # ---- Cosmetic curve ----
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas) + 1e-12), s=0.001)
    i_smooth = 10**spl(E_grid)
    r2 = r2_score(np.log10(np.abs(i_meas) + 1e-12), spl(E))

    # ---- Plot ----
    fig, ax = plt.subplots()
    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Fit")
    ax.axvline(Ecorr_guess, color="b", linestyle="--", label="Ecorr")
    ax.axvline(pars["Ecorr"], color="g", linestyle="--", label="Fitted Ecorr")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel("|i| (A/cm²)")
    ax.grid(True, which="both")
    ax.legend()
    st.pyplot(fig)
