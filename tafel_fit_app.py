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

def longest_linear_tafel_region(E, i_meas, Ecorr, anodic=True, min_size=6, r2_threshold=0.995, min_decades=1.0):
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
            if np.ptp(logi) < min_decades:
                continue
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

def get_tafel_fit_deviation(E, i_meas, idx_fit, direction="right", deviation=0.10):
    if len(idx_fit) < 2:
        return None
    E_fit = E[idx_fit]
    logi_fit = np.log10(np.abs(i_meas[idx_fit]) + 1e-15)
    fit = linregress(E_fit, logi_fit)
    predicted = fit.intercept + fit.slope * E
    logi_meas = np.log10(np.abs(i_meas) + 1e-15)
    errors = np.abs(10 ** logi_meas - 10 ** predicted) / np.maximum(np.abs(10 ** predicted), 1e-30)
    last_idx = idx_fit[-1] if direction == "right" else idx_fit[0]
    search_range = range(last_idx + 1, len(E)) if direction == "right" else range(last_idx - 1, -1, -1)
    for i in search_range:
        if errors[i] > deviation:
            return E[i]
    return None

data_file = st.file_uploader("Upload polarization data (CSV/Excel).", type=["csv", "xlsx", "xls"])
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
    st.write(f"Fitted Ecorr = **{pars['Ecorr']:.3f} V** (data-driven guess: {Ecorr_guess:.3f} V)")

    # Corrosion rate
    st.markdown("### Corrosion rate")
    mode = st.radio("Material info:", ["I know V_m and z", "I don't know the material"], index=1, horizontal=True)
    if mode == "I know V_m and z":
        z = st.number_input("Valence z (electrons per metal atom)", value=2, min_value=1, step=1)
        Vm = st.number_input("Molar volume V_m (cm³/mol)", value=7.09, min_value=0.0)
        if np.isfinite(i_corr) and Vm > 0 and z > 0:
            CR_mm_per_yr = 3270.0 * i_corr * Vm / z
            CR_info = f"{CR_mm_per_yr:.3f} mm/year"
            st.write(f"Corrosion rate = **{CR_info}** (V_m = {Vm:.3f} cm³/mol, z = {int(z)})")
        else:
            CR_info = "N/A"
            st.warning("Provide positive V_m and z to compute corrosion rate.")
    else:
        materials = {
            "Steel-like (Fe)": (7.09, 2),
            "Aluminum (Al)": (10.0, 3),
            "Copper (Cu)": (7.11, 2),
            "Nickel (Ni)": (6.59, 2),
            "Zinc (Zn)": (9.16, 2),
            "Titanium (Ti)": (10.64, 4),
            "Magnesium (Mg)": (14.0, 2),
        }
        k_list = np.array([3.27e-3 * Vm / z for (Vm, z) in materials.values()])
        k_med = float(np.median(k_list))
        k_min = float(np.min(k_list))
        k_max = float(np.max(k_list))
        i_corr_uA = i_corr * 1e6
        cr_est = k_med * i_corr_uA
        CR_info = f"{cr_est:.3f} mm/year"
        cr_low = k_min * i_corr_uA
        cr_high = k_max * i_corr_uA
        st.write(f"Estimated corrosion rate = **{cr_est:.3f} mm/year**")
        st.write(f"Typical range across common metals: **{cr_low:.3f} – {cr_high:.3f} mm/year**")
        with st.expander("Assumptions and per-μA factors"):
            for name, (Vm, z) in materials.items():
                k = 3.27e-3 * Vm / z
                st.write(f"- {name}: V_m={Vm} cm³/mol, z={z} → {k:.5f} mm/year per μA/cm²")
        st.caption("Without material identity, this is a rough estimate; true CR depends on V_m and z.")

    # Fit curve for main plot
    E_grid = np.linspace(E.min(), E.max(), 600)
    spl = UnivariateSpline(E, np.log10(np.abs(i_meas) + 1e-12), s=0.001)
    i_smooth = 10 ** spl(E_grid)

    # Tafel regions
    anodic_idx = longest_linear_tafel_region(
        E, i_meas, Ecorr_guess, anodic=True, min_size=6, r2_threshold=0.995, min_decades=1.0
    )
    cathodic_idx = longest_linear_tafel_region(
        E, i_meas, Ecorr_guess, anodic=False, min_size=6, r2_threshold=0.995, min_decades=1.0
    )

    anodic_found = len(anodic_idx) > 0
    cathodic_found = len(cathodic_idx) > 0

    if not anodic_found:
        st.warning("No anodic Tafel region found (try less stringent min_size, min_decades, or r2_threshold).")
    if not cathodic_found:
        st.warning("No cathodic Tafel region found (try less stringent min_size, min_decades, or r2_threshold).")

    anodic_bounds = (E[anodic_idx[0]], E[anodic_idx[-1]]) if anodic_found else (None, None)
    cathodic_bounds = (E[cathodic_idx[0]], E[cathodic_idx[-1]]) if cathodic_found else (None, None)

    # Find where the diffusion region starts on anodic branch (>10% deviation from linear fit)
    anode_diff_start = get_tafel_fit_deviation(E, i_meas, anodic_idx, direction="right", deviation=0.10) if anodic_found else None

    # Diffusion-limited (cathodic branch)
    diff_limit_thr = 0.20
    ilim = np.nanmin(i_meas)
    mask_diff = (i_meas < 0) & (np.abs(i_meas - ilim) / np.abs(ilim) < diff_limit_thr) & (E < Ecorr_guess)
    diff_indices = np.where(mask_diff)[0]
    diff_bounds = (E[diff_indices[0]], E[diff_indices[-1]]) if len(diff_indices) > 0 else (None, None)

    # Anodic diffusion-limited region: from anode_diff_start to max E
    anodic_diff_bounds = (None, None)
    if anodic_found and anode_diff_start is not None:
        idx_start = np.argmin(np.abs(E - anode_diff_start))
        anodic_diff_bounds = (E[idx_start], E[-1])

    # Ecorr (magenta)
    ecorr_window = 0.03
    ecorr_bounds = (Ecorr_guess - ecorr_window, Ecorr_guess + ecorr_window)

    # -------------------------- MAIN PLOT ----------------------------
    fig, ax = plt.subplots(figsize=(7, 5))
    if cathodic_bounds[0] is not None:
        ax.axvspan(cathodic_bounds[0], cathodic_bounds[1], color='blue', alpha=0.15, label="Cathodic Tafel region")
    if anodic_bounds[0] is not None:
        ax.axvspan(anodic_bounds[0], anodic_bounds[1], color='red', alpha=0.14, label="Anodic Tafel region")
    if diff_bounds[0] is not None:
        ax.axvspan(diff_bounds[0], diff_bounds[1], color='green', alpha=0.10, label="Cathodic diffusion-limited")
    if anodic_diff_bounds[0] is not None:
        ax.axvspan(anodic_diff_bounds[0], anodic_diff_bounds[1], color='yellow', alpha=0.12, label="Anodic diffusion-limited")
    ax.axvspan(*ecorr_bounds, color='magenta', alpha=0.13, label="Ecorr region")
    if anode_diff_start is not None:
        ax.axvline(anode_diff_start, color='orange', lw=2, linestyle='--', label='Anodic diffusion onset (>10% deviation)')
    ax.semilogy(E, np.abs(i_meas), "k.", label="Data")
    ax.semilogy(E_grid, i_smooth, "r-", label="Fit")
    ax.axvline(Ecorr_guess, color="blue", linestyle="--", label="Ecorr")
    ax.set_xlabel("Potential (V)")
    ax.set_ylabel(r"$|i|$ (A/cm²)")
    ax.grid(True, which="both")
    ax.legend(loc="lower right", fontsize=9)
    st.pyplot(fig)

    # --- Log(|i|) vs E plot to show fitting windows ---
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    logi = np.log10(np.abs(i_meas) + 1e-15)
    ax2.plot(E, logi, "k.", label="log(|i|) data")
    if cathodic_bounds[0] is not None:
        ax2.axvspan(cathodic_bounds[0], cathodic_bounds[1], color='blue', alpha=0.15, label="Cathodic Tafel window")
        fitE = E[cathodic_idx]
        fitlogi = logi[cathodic_idx]
        res = linregress(fitE, fitlogi)
        ax2.plot(fitE, res.intercept + res.slope * fitE, color='blue', lw=2)
    if anodic_bounds[0] is not None:
        ax2.axvspan(anodic_bounds[0], anodic_bounds[1], color='red', alpha=0.15, label="Anodic Tafel window")
        fitE = E[anodic_idx]
        fitlogi = logi[anodic_idx]
        res = linregress(fitE, fitlogi)
        ax2.plot(fitE, res.intercept + res.slope * fitE, color='red', lw=2)
    if anode_diff_start is not None:
        ax2.axvline(anode_diff_start, color='orange', lw=2, linestyle='--', label='Anodic diffusion onset (>10% deviation)')
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
        "Shaded regions: Red=Anodic Tafel, Blue=Cathodic Tafel, Yellow=Anodic diffusion-limited, Green=Cathodic diffusion-limited, Magenta=Ecorr region.\n"
        "The longest contiguous, high-linearity region (≥1 decade, R²>0.995) is used for each Tafel slope. The diffusion onset (anodic) is marked by orange dashed line (>10% deviation)."
    )

    # --- Export parameters as CSV ---
    param_dict = {
        'i0_a (A/cm²)': [pars["i0_a"]],
        'alpha_a': [pars["alpha_a"]],
        'i0_c (A/cm²)': [pars["i0_c"]],
        'alpha_c': [pars["alpha_c"]],
        'iL (A/cm²)': [pars["iL"]],
        'Ecorr (V)': [pars["Ecorr"]],
        'Ru (ohm·cm²)': [pars["Ru"]],
        'beta_a (V/decade)': [beta_a],
        'beta_c (V/decade)': [beta_c],
        'i_corr (A/cm²)': [i_corr],
        'Fitted corrosion rate (mm/year)': [CR_info],
        'Tafel linear anodic E_start (V)': [anodic_bounds[0]],
        'Tafel linear anodic E_end (V)': [anodic_bounds[1]],
        'Diffusion onset anodic (V)': [anode_diff_start],
        'Tafel linear cathodic E_start (V)': [cathodic_bounds[0]],
        'Tafel linear cathodic E_end (V)': [cathodic_bounds[1]],
        'Anodic diffusion-limited region start (V)': [anodic_diff_bounds[0]],
        'Anodic diffusion-limited region end (V)': [anodic_diff_bounds[1]],
    }
    params_df = pd.DataFrame(param_dict)
    csv_bytes = params_df.to_csv(index=False).encode()
    st.download_button(
        "Download Fit Parameters (CSV)",
        data=csv_bytes,
        file_name="tafel_fit_parameters.csv",
        mime="text/csv"
    )
