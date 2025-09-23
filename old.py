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

st.title("Global Implicit Tafel Fit")

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
    st.success(f"Loaded {len(df)} rows.")
    st.dataframe(df.head(8))

    col_E = st.selectbox("Potential column", df.columns)
    col_I = st.selectbox("Current column", df.columns)
    pot_units = st.selectbox("Potential units", ["V","mV"], 0)
    cur_units = st.selectbox("Current units", ["A","mA","uA","nA"], 1)

    area_val = st.number_input("Electrode area (cm²)", value=1.0)
    area_arr = np.full(len(df), area_val)

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

    st.write(f"β_a = {beta_a:.3f} V/dec, β_c = {beta_c:.3f} V/dec")
    st.write(f"i_corr = {i_corr:.3e} A/cm²")
    st.write(f"Fitted Ecorr = **{pars['Ecorr']:.3f} V** (data-driven guess: {Ecorr_guess:.3f} V)")

    # ---- Corrosion rate (no EW or density) ----
    st.markdown("### Corrosion rate")
    mode = st.radio("Material info:", ["I know V_m and z", "I don't know the material"], index=1, horizontal=True)

    if mode == "I know V_m and z":
        z = st.number_input("Valence z (electrons per metal atom)", value=2, min_value=1, step=1)
        Vm = st.number_input("Molar volume V_m (cm³/mol)", value=7.09, min_value=0.0,
                             help="Examples: Fe≈7.09, Al≈10.0, Cu≈7.11, Ni≈6.59, Zn≈9.16, Ti≈10.64, Mg≈14.0 cm³/mol")
        # CR(mm/yr) = 3270 * i_corr(A/cm²) * V_m(cm³/mol) / z
        if np.isfinite(i_corr) and Vm > 0 and z > 0:
            CR_mm_per_yr = 3270.0 * i_corr * Vm / z
            st.write(f"Corrosion rate = **{CR_mm_per_yr:.3f} mm/year**  (V_m = {Vm:.3f} cm³/mol, z = {int(z)})")
        else:
            st.warning("Provide positive V_m and z to compute corrosion rate.")
    else:
        # Unknown material: estimate best value and range across common metals
        materials = {
            "Steel-like (Fe)": (7.09, 2),
            "Aluminum (Al)":   (10.0, 3),
            "Copper (Cu)":     (7.11, 2),
            "Nickel (Ni)":     (6.59, 2),
            "Zinc (Zn)":       (9.16, 2),
            "Titanium (Ti)":   (10.64, 4),
            "Magnesium (Mg)":  (14.0, 2),
        }
        # Factor per μA/cm²: k = 3.27e-3 * V_m / z  [mm/year per μA/cm²]
        k_list = np.array([3.27e-3 * Vm / z for (Vm, z) in materials.values()])
        k_med = float(np.median(k_list))
        k_min = float(np.min(k_list))
        k_max = float(np.max(k_list))

        i_corr_uA = i_corr * 1e6  # A/cm² -> μA/cm²
        cr_est  = k_med * i_corr_uA
        cr_low  = k_min * i_corr_uA
        cr_high = k_max * i_corr_uA

        st.write(f"Estimated corrosion rate = **{cr_est:.3f} mm/year**")
        st.write(f"Typical range across common metals: **{cr_low:.3f} – {cr_high:.3f} mm/year**")
        with st.expander("Assumptions and per-μA factors"):
            for name, (Vm, z) in materials.items():
                k = 3.27e-3 * Vm / z
                st.write(f"- {name}: V_m={Vm} cm³/mol, z={z} → {k:.5f} mm/year per μA/cm²")
        st.caption("Without material identity, this is a rough estimate; true CR depends on V_m and z.")

    # ---- Methods and equations (added for documentation only; outputs unchanged) ----
    with st.expander("Methods and equations used"):
        st.markdown(r"""
**Model**
- Overpotential with ohmic drop: \( \eta = E - E_{\text{corr}} - i\,R_u \)
- Anodic activation: \( i_a = i_{0,a}\,\exp\!\big(\tfrac{\alpha_a n F}{RT}\,\eta\big) \)
- Cathodic activation: \( i_{c}^{\text{act}} = -\,i_{0,c}\,\exp\!\big(-\tfrac{\alpha_c n F}{RT}\,\eta\big) \)
- Cathodic mass-transfer limit (Koutecký–Levich combination):
  \[
  \frac{1}{i_c}=\frac{1}{i_{c}^{\text{act}}}+\frac{1}{-i_L}
  \quad\Rightarrow\quad
  i_c=\frac{i_{c}^{\text{act}}(-i_L)}{\,i_{c}^{\text{act}}-i_L\,}
  \]
- Mixed current balance (implicit): \( i = i_a + i_c \), solved for \(i\) by Newton's method at each \(E\).

**Fitting objective**
- For each point \(E_j\), compute \(i_{\text{model}}(E_j)\) and minimize robust log-magnitude residuals:
  \[
  r_j=\log_{10}\!\big(|i_{\text{model}}|+\varepsilon\big)-\log_{10}\!\big(|i_{\text{meas}}|+\varepsilon\big)
  \]
  using `soft_l1` loss with bounds on parameters.

**Reported quantities**
- Tafel slope: \( \beta = \tfrac{2.303\,RT}{\alpha\,nF} \) (V/dec)
- Corrosion current density: \( i_{\text{corr}} = |\,i(E=E_{\text{corr}})\,| \)
- Corrosion rate (if \(V_m,z\) known): \( \text{CR}(\text{mm/yr}) = 3270 \; i_{\text{corr}}(\text{A/cm}^2)\; \tfrac{V_m(\text{cm}^3/\text{mol})}{z} \)

**Notes on sources**
- This is a generic mixed-kinetics model (Butler–Volmer branches + Koutecký–Levich limit + ohmic drop), consistent with textbook electrochemistry and many papers.
- It is conceptually aligned with:
  - M. C. van Ede & U. Angst — Tafel slopes and exchange current densities for ORR/HER on steel.
  - H. J. Flitt & D. P. Schweinsberg — Polarisation curve deconstruction for Fe/H\(_2\)O/H\(^+\)/O\(_2\).
- The implementation here is not a direct reproduction of those specific methodologies; it uses a single anodic and a single lumped cathodic branch with one limiting current.
""")

    with st.expander("References"):
        st.markdown("""
- M. C. van Ede and U. Angst, “Tafel slopes and exchange current densities of oxygen reduction and hydrogen evolution on steel.” (DOI provided in your source)
- H. J. Flitt and D. P. Schweinsberg, “A guide to polarisation curve interpretation: deconstruction of experimental curves typical of the Fe/H2O/H+/O2 corrosion system.”
- Standard texts on electrochemistry and corrosion kinetics (e.g., Butler–Volmer, Koutecký–Levich, mixed potential theory).
""")

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
