"""
Quantitative Methods in Finance — Final Exam 2025-2026
Counterfactual Inflation — Quarterly BQ-SVAR

Method : Bivariate SVAR, Blanchard-Quah (1989) identification
Freq   : Quarterly

Pipeline:
  1. Estimate SVAR on stationary QoQ variables (output + inflation)
  2. Replace UA demand shocks with EA demand shocks → counterfactual QoQ
  3. Apply time-varying treatment δ(t) from Part A regime chronology
  4. Reconstruct counterfactual price level → YoY for the figure

Data sources
------------
Ukraine CPI  : data_ukraine_cpi_raw.csv  (repo, SSSU SDMX, monthly MoM index)
EA HICP      : data_ecb_hicp_panel.csv   (repo, ECB, 11 countries YoY %, not used in the SVAR)
               + FRED CP0000EZCCM086NEST (monthly HICP level, index 2015=100)
Tthe official EA aggregate HICP level is preferred as it avoids equal-weighting of heterogeneous economies.
Ukraine IPI  : World Bank NV.IND.TOTL.KD (annual, downloaded programmatically)
               → quarterly linear interpolation
               No monthly/quarterly IPI for Ukraine available from public APIs.
               Linear interpolation induces smooth intra-year dynamics;
               results are interpreted with caution regarding magnitudes.
EA IPI       : FRED EA19PRINTO01IXOBSAM (monthly, SA, index 2015=100)

Note on δ(t)
------------
δ(t) ∈ [0,1] is a theory-informed calibrated parameter, not estimated.
It captures the fraction of monetary sovereignty Ukraine actually exercised
in each period, derived from the Part A regime chronology (Calvo & Reinhart 2002).
It cannot be estimated econometrically without additional identifying assumptions;
instead it is grounded in the qualitative IMF de facto regime classifications
documented in Part A. δ=1 during genuine float/devaluation episodes;
δ≈0.15 during de facto USD peg periods.
"""

import os
import warnings
import re
from io import StringIO

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import requests
from scipy.linalg import cholesky
from scipy.signal import lfilter
from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller

# ─────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────
# Repo CSV files are read locally (same directory as this script).
# Only truly external data (FRED, World Bank) is downloaded programmatically.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id="

# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
def from_local(filename):
    """Read a CSV from the same directory as this script."""
    return pd.read_csv(os.path.join(BASE_DIR, filename))

def from_fred(series_id):
    """Download a public FRED series (no API key required)."""
    r = requests.get(FRED_CSV + series_id, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), na_values=[".", ""])
    df.columns = ["date", series_id]
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")[series_id].dropna().astype(float).resample("MS").last()

def from_world_bank(indicator, country="UKR"):
    """Download an annual series from the World Bank API."""
    r = requests.get(
        f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
        f"?format=json&per_page=1000", timeout=30)
    r.raise_for_status()
    rows = [{"date": x["date"], "val": x["value"]}
            for x in r.json()[1] if x["value"] is not None]
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"] + "-01-01")
    return df.sort_values("date").set_index("date")["val"].astype(float)

def parse_ukraine_cpi(df):
    """
    Parse SSSU SDMX monthly MoM index (e.g. 101.5 = +1.5% MoM).
    Handles SDMX date format 2005-M02 and standard 2005-02.
    """
    pat = re.compile(r"^\d{4}-M?\d{2}$")
    date_col = "TIME_PERIOD" if "TIME_PERIOD" in df.columns else next(
        c for c in df.columns
        if df[c].dropna().astype(str).str.match(pat).mean() > 0.8)
    val_col = "OBS_VALUE" if "OBS_VALUE" in df.columns else next(
        c for c in df.columns if c != date_col
        and pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8)
    mask = df[date_col].astype(str).str.strip().str.match(pat)
    out  = df.loc[mask, [date_col, val_col]].copy()
    out[date_col] = pd.to_datetime(
        out[date_col].astype(str).str.strip()
            .str.replace("-M", "-", regex=False) + "-01")
    return out.set_index(date_col)[val_col].astype(float).sort_index()

def fit_bq_svar(df, max_lags=4, horizon=20, label=""):
    """
    Bivariate SVAR — Blanchard-Quah (1989) identification.

    Column order: [output_QoQ, inflation_QoQ_ann]  ← ORDER MATTERS

    Identification:
      Supply shock : permanent effect on output
      Demand shock : zero long-run effect on output  ← BQ restriction

    The restriction is imposed via lower-triangular Cholesky of C(1)·Σ·C(1)':
      C(1) = (I − ΣAⱼ)⁻¹    [long-run multiplier]
      D    = chol(C(1)·Σ·C(1)')   [lower triangular → D[0,1] = 0]
      B₀   = C(1)⁻¹·D
      εₜ   = B₀⁻¹·uₜ             [structural shocks]
      Θₕ   = Φₕ·B₀               [structural IRFs]
    """
    tmp = df.copy()
    tmp.columns = ["output", "inflation"]

    sel = VAR(tmp).select_order(maxlags=max_lags)
    p   = max(int(sel.selected_orders.get("aic", 1)), 1)
    res = VAR(tmp).fit(p, trend="c")
    if label:
        print(f"  [{label}] lags (AIC): p={p}")

    C1    = np.linalg.inv(np.eye(2) - res.coefs.sum(axis=0))
    Sigma = res.sigma_u.values if hasattr(res.sigma_u, "values") else res.sigma_u
    M     = C1 @ Sigma @ C1.T
    M     = (M + M.T) / 2
    D     = cholesky(M, lower=True)    # D[0,1] = 0 by construction
    B0    = np.linalg.inv(C1) @ D

    if label:
        print(f"  [{label}] D[0,1]={D[0,1]:.2e} (BQ restriction, should ≈ 0)")

    eps   = res.resid.values @ np.linalg.inv(B0).T
    eps   = pd.DataFrame(eps, index=res.resid.index, columns=["supply", "demand"])
    Theta = np.array([res.irf(horizon).irfs[h] @ B0 for h in range(horizon + 1)])

    return dict(result=res, eps=eps, Theta=Theta)

def make_delta(index):
    """
    Time-varying treatment intensity δ(t) ∈ [0,1].

    δ(t) is a theory-informed calibrated parameter derived from the Part A
    regime chronology. It is NOT estimated from the data — there is no
    standard econometric procedure to recover it without additional identifying
    assumptions. Instead, it is grounded in IMF de facto regime classifications
    (Calvo & Reinhart 2002) documented in Part A:

      δ = 1.00  genuine monetary autonomy: Ukraine exercised full control
                over monetary conditions (major devaluation episodes)
      δ = 0.80  high autonomy: IT framework operational, managed float
      δ = 0.55  partial: IT adopted but credibility not fully established,
                or managed float with significant capital controls
      δ = 0.25  limited: hard peg under martial law constraints, but some
                residual flexibility vs a full conventional peg
      δ = 0.15  near-zero: de facto USD peg, impossible trinity binding,
                monetary policy subordinated to exchange rate anchor

    The final counterfactual is:
      π_final(t) = δ(t)·π_cf_BQ(t) + (1−δ(t))·π_actual(t)

    This ensures the counterfactual tracks actual inflation when the treatment
    is small (peg periods) and diverges when the treatment is large (genuine float).
    """
    d = pd.Series(np.nan, index=index, dtype=float)
    d.loc[:"2007-12"]          = 0.15   # conventional USD peg (IMF: conv. peg)
    d.loc["2008-07":"2009-09"] = 1.00   # GFC devaluation ~60% — full autonomy
    d.loc["2009-10":"2013-12"] = 0.15   # stabilised arrangement (IMF reclassif.)
    d.loc["2014-01":"2015-12"] = 1.00   # Crimea/Donbas ~225% — full autonomy
    d.loc["2016-01":"2016-12"] = 0.55   # IT announced (Aug 2015), not yet credible
    d.loc["2017-01":"2021-12"] = 0.80   # IT operational; residual interventions
    d.loc["2022-01":"2023-09"] = 0.25   # wartime hard peg (Resolution No. 18)
    d.loc["2023-10":]          = 0.55   # managed float post-peg exit (Oct 2023)
    return d.ffill().bfill()

# ─────────────────────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────────────────────
print("── Loading data ──")

# Repo files — read locally
ua_cpi_mom = parse_ukraine_cpi(from_local("data_ukraine_cpi_raw.csv"))
ua_cpi_mom = ua_cpi_mom.loc["2005-02":]   # trim to exam-documented sample
print(f"  UA CPI (local)  : {ua_cpi_mom.index.min():%Y-%m} → {ua_cpi_mom.index.max():%Y-%m}")

# External data — downloaded programmatically
ea_hicp_lv = from_fred("CP0000EZCCM086NEST")   # EA HICP level, index 2015=100
print(f"  EA HICP (FRED)  : {ea_hicp_lv.index.min():%Y-%m} → {ea_hicp_lv.index.max():%Y-%m}")

ua_out_ann = from_world_bank("NV.IND.TOTL.KD", "UKR")   # annual industry VA
print(f"  UA output (WB)  : {ua_out_ann.index.min():%Y} → {ua_out_ann.index.max():%Y} (annual)")

ea_ipi_m = from_fred("EA19PRINTO01IXOBSAM")   # EA IPI, monthly, SA
print(f"  EA IPI (FRED)   : {ea_ipi_m.index.min():%Y-%m} → {ea_ipi_m.index.max():%Y-%m}")

# ─────────────────────────────────────────────────────────────
# 2. BUILD QUARTERLY SERIES
# ─────────────────────────────────────────────────────────────

# Ukraine price level and inflation
ua_price_m  = (ua_cpi_mom / 100).cumprod()
ua_price_q  = ua_price_m.resample("QS").last()
ua_pi_yoy_q = ua_price_q.pct_change(4) * 100       # YoY — for final figure
ua_pi_qoq_q = np.log(ua_price_q).diff(1) * 400     # QoQ annualised — for SVAR

# EA price level and inflation
ea_price_q  = ea_hicp_lv.resample("QS").last()
ea_pi_qoq_q = np.log(ea_price_q).diff(1) * 400     # QoQ annualised — for SVAR

# Ukraine output: annual → quarterly linear interpolation → QoQ log-diff
# Note: interpolation induces smooth intra-year dynamics (see docstring above)
ua_out_q    = ua_out_ann.resample("QS").interpolate(method="time")
ua_out_qoq  = np.log(ua_out_q).diff(1) * 100

# EA output: monthly → quarterly average → QoQ log-diff
ea_out_q    = ea_ipi_m.resample("QS").mean()
ea_out_qoq  = np.log(ea_out_q).diff(1) * 100

# Bivariate SVAR datasets — column order [output, inflation] is mandatory for BQ
ua_df = pd.concat([ua_out_qoq.rename("output"), ua_pi_qoq_q.rename("inflation")],
                  axis=1).dropna()
ea_df = pd.concat([ea_out_qoq.rename("output"), ea_pi_qoq_q.rename("inflation")],
                  axis=1).dropna()

t0 = max(ua_df.index.min(), ea_df.index.min())
t1 = min(ua_df.index.max(), ea_df.index.max())
ua_df = ua_df.loc[t0:t1]
ea_df = ea_df.loc[t0:t1]
print(f"\n✓ Common sample: {t0:%Y-%m} → {t1:%Y-%m} ({len(ua_df)} quarters)")

# ─────────────────────────────────────────────────────────────
# 3. ADF STATIONARITY TESTS
# ─────────────────────────────────────────────────────────────
print("\n── ADF tests (H₀: unit root) ──")
for lbl, s in [("UA output QoQ",   ua_df["output"]),
               ("UA infl QoQ ann",  ua_df["inflation"]),
               ("EA output QoQ",    ea_df["output"]),
               ("EA infl QoQ ann",  ea_df["inflation"])]:
    stat, pval, *_ = adfuller(s.dropna(), autolag="AIC")
    flag = "I(0) ✓" if pval < 0.05 else ("borderline" if pval < 0.10 else "I(1) ⚠")
    print(f"  {lbl:<20}  ADF={stat:7.3f}  p={pval:.3f}  {flag}")

# ─────────────────────────────────────────────────────────────
# 4. SVAR ESTIMATION
# ─────────────────────────────────────────────────────────────
print("\n── BQ-SVAR Ukraine ──")
ua_svar = fit_bq_svar(ua_df, label="Ukraine")
print("\n── BQ-SVAR Euro Area ──")
ea_svar = fit_bq_svar(ea_df, label="EA")

# ─────────────────────────────────────────────────────────────
# 5. COUNTERFACTUAL (QoQ annualised)
# ─────────────────────────────────────────────────────────────
# Bayoumi & Eichengreen (1993): under EA membership,
#   - Supply shocks remain Ukrainian (energy, agriculture, geopolitics)
#   - Demand shocks replaced by ECB-driven EA demand shocks
#   - Propagation follows Ukraine's structural IRFs
#
# π_cf_t = Σₕ Θ_UA[h,1,0]·ε_supply_UA[t-h]    (supply: unchanged)
#          + Σₕ Θ_UA[h,1,1]·ε_demand_EA[t-h]   (demand: ECB-driven)
idx = ua_svar["eps"].index.intersection(ea_svar["eps"].index)

pi_cf_qoq = pd.Series(
    lfilter(ua_svar["Theta"][:, 1, 0], [1.],
            ua_svar["eps"]["supply"].reindex(idx).fillna(0).values)
  + lfilter(ua_svar["Theta"][:, 1, 1], [1.],
            ea_svar["eps"]["demand"].reindex(idx).fillna(0).values),
    index=idx
)

# ─────────────────────────────────────────────────────────────
# 6. TIME-VARYING TREATMENT δ(t)
# ─────────────────────────────────────────────────────────────
# π_cf_final = δ(t)·π_cf_BQ + (1-δ(t))·π_actual
# See make_delta() docstring for full calibration rationale.
delta        = make_delta(idx)
pi_cf_qoq_w  = delta * pi_cf_qoq + (1 - delta) * ua_pi_qoq_q.reindex(idx)

# ─────────────────────────────────────────────────────────────
# 7. RECONSTRUCT COUNTERFACTUAL PRICE LEVEL → YoY
# ─────────────────────────────────────────────────────────────
# QoQ annualised (%) → quarterly log increment → cumulative price level → YoY
cf_price_q = pd.Series(np.nan, index=idx)
cf_price_q.iloc[0] = ua_price_q.reindex(idx).iloc[0]
for i in range(1, len(idx)):
    cf_price_q.iloc[i] = cf_price_q.iloc[i - 1] * np.exp(pi_cf_qoq_w.iloc[i] / 400)

plot_actual = ua_pi_yoy_q.reindex(idx)
plot_cf     = cf_price_q.pct_change(4) * 100

# ─────────────────────────────────────────────────────────────
# 8. SUMMARY TABLES
# ─────────────────────────────────────────────────────────────
print("\n── Mean YoY inflation by episode ──")
print(f"  {'Episode':<30} {'Actual':>8} {'CF':>8} {'Δ(pp)':>8}")
print("  " + "─" * 52)
for ep, (s, e) in {
    "USD peg 2005-2007":     ("2005-04", "2007-12"),
    "GFC 2008-2009":         ("2008-07", "2009-09"),
    "USD peg 2010-2013":     ("2010-01", "2013-12"),
    "Crimea/Donbas 2014-15": ("2014-01", "2015-12"),
    "IT + float 2017-2021":  ("2017-01", "2021-12"),
    "Full invasion 2022-23": ("2022-01", "2023-06"),
}.items():
    a, c = plot_actual.loc[s:e].mean(), plot_cf.loc[s:e].mean()
    if pd.notna(a) and pd.notna(c):
        print(f"  {ep:<30} {a:>8.1f} {c:>8.1f} {c-a:>+8.1f}")

print("\n── Treatment intensity δ(t) by episode ──")
print(f"  {'Episode':<30} {'δ(t)':>6}")
print("  " + "─" * 38)
for ep, (s, e) in {
    "USD peg 2005-2007":     ("2005-04", "2007-12"),
    "GFC 2008-2009":         ("2008-07", "2009-09"),
    "USD peg 2010-2013":     ("2010-01", "2013-12"),
    "Crimea/Donbas 2014-15": ("2014-01", "2015-12"),
    "IT adoption 2016":      ("2016-01", "2016-12"),
    "IT + float 2017-2021":  ("2017-01", "2021-12"),
    "Wartime peg 2022-23":   ("2022-01", "2023-09"),
    "Post-peg 2023+":        ("2023-10", "2023-12"),
}.items():
    d = delta.loc[s:e].mean()
    if pd.notna(d):
        print(f"  {ep:<30} {d:>6.2f}")

# ─────────────────────────────────────────────────────────────
# 9. FIGURE (single panel — as required by the exam)
# ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

ax.fill_between(idx, plot_actual, plot_cf,
                where=(plot_cf < plot_actual),
                alpha=0.13, color="#C0392B",
                label="EA membership → lower inflation")
ax.fill_between(idx, plot_actual, plot_cf,
                where=(plot_cf >= plot_actual),
                alpha=0.13, color="#2471A3",
                label="EA membership → higher inflation")
ax.plot(idx, plot_actual, color="#C0392B", lw=2.0,
        label="Ukraine — actual CPI YoY (%)")
ax.plot(idx, plot_cf,     color="#2471A3", lw=2.0, ls="--",
        label="Counterfactual: Ukraine in the Euro Area\n"
              "(BQ-SVAR: UA supply shocks + EA demand shocks × δ(t))")
ax.axhline(5, color="#27AE60", lw=0.8, ls=":", alpha=0.8,
           label="ECB target (5%)")
ax.axhline(0, color="black", lw=0.4, ls=":")

for s, e, lbl, col in [("2008-07", "2009-09", "GFC\n2008-09",    "#E74C3C"),
                        ("2014-01", "2015-12", "Crimea/\nDonbas", "#E74C3C"),
                        ("2022-01", "2023-06", "Full\ninvasion",  "#E67E22")]:
    ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.08, color=col, zorder=0)
    mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
    ax.text(mid, ax.get_ylim()[1] * 0.95, lbl,
            ha="center", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, ec="none"))

ax.set_title(
    "Ukraine: Actual vs Counterfactual Inflation Under Euro Area Membership\n"
    "Quarterly BQ-SVAR | UA supply shocks + EA demand shocks × time-varying δ(t)",
    fontsize=11, pad=8)
ax.set_ylabel("Year-on-year inflation (%)", fontsize=11)
ax.set_xlabel("Date", fontsize=11)
ax.legend(loc="upper right", fontsize=9, framealpha=0.92, edgecolor="lightgray")
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

plt.tight_layout()
plt.savefig("ukraine_counterfactual_inflation.png", dpi=150, bbox_inches="tight")
plt.show()
print("✓ Figure saved: ukraine_counterfactual_inflation.png")
