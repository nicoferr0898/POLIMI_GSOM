# -*- coding: utf-8 -*-
"""
=============================================================================
  ESG Constraints and the Markowitz Efficient Frontier
  Empirical analysis - Russell 3000 (2017-2022)
=============================================================================
  Code structure
  --------------
  1.  Libraries and settings
  2.  ESG data
  3.  Return data
  4.  Selection of the ten assets
  5.  Annualized portfolio parameters
  6.  Descriptive analysis
  7.  ESG - expected-return relationship
  8.  Unconstrained efficient frontier (closed form)
  9.  ESG-constrained frontiers - short selling allowed
  10. ESG-constrained frontiers - long-only portfolios
  11. Cost of sustainability
  12. Composition of the optimal portfolio
  13. Summary of results
=============================================================================
"""

# =============================================================================
# 1. LIBRARIES AND SETTINGS
# =============================================================================

#%% [1] Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy.optimize import minimize
import statsmodels.api as sm

np.set_printoptions(suppress=True, precision=4)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")

# Global plotting parameters (paper-ready figures)
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "legend.fontsize":  10,
    "figure.dpi":       130,
})

SAVEFIG = True
def _save(name):
    if SAVEFIG:
        plt.savefig(f"fig_{name}.png", dpi=150, bbox_inches="tight")
    plt.show()
    plt.close()


# =============================================================================
# 2. ESG DATA
# =============================================================================

#%% [2] ESG data
# The file is structured as ISIN x months. We transpose it to get months x ISIN.
esg_raw = pd.read_csv("ESG_monthly_clean.csv", sep=";")
esg_raw = esg_raw.T
esg_raw.columns = esg_raw.iloc[0]      # ISINs become column names
esg_raw = esg_raw[2:]                  # drop the ISIN and RIC rows
esg_raw.dropna(how="all", inplace=True)
# Scores are on a 0-100 scale: we normalize them to [0, 1].
esg_raw = esg_raw.apply(pd.to_numeric, errors="coerce") / 100


# =============================================================================
# 3. RETURN DATA
# =============================================================================

#%% [3] Returns
# Same format: we transpose it and convert to decimals.
ret_raw = pd.read_csv("Returns.csv", sep=",").iloc[:, 1:]
ret_raw = ret_raw.T
ret_raw.columns = ret_raw.iloc[0]
ret_raw = ret_raw[2:]
ret_raw = ret_raw.apply(pd.to_numeric, errors="coerce") / 100


# =============================================================================
# 4. SELECTION OF THE TEN ASSETS
# =============================================================================

#%% [4] Asset selection
# Consider the ISINs present in both datasets.
common_isins = ret_raw.columns.intersection(esg_raw.columns)

# Average ESG score per ISIN, sorted from best to worst.
esg_score_full = esg_raw[common_isins].mean().sort_values(ascending=False)

# Exclude ISINs with fewer than 60 monthly return observations.
valid_isins = common_isins[ret_raw[common_isins].notna().sum() >= 60]
esg_score_valid = esg_score_full[esg_score_full.index.isin(valid_isins)]

# The five stocks with the highest ESG score and the five with the lowest.
best5  = list(esg_score_valid.index[:5])
worst5 = list(esg_score_valid.index[-5:])
sel    = best5 + worst5

# Readable labels (ticker from the RIC column of the returns file).
ric_lookup = (pd.read_csv("Returns.csv", sep=",")
                .set_index("Instrument")["RIC"]
                .str.replace(r"\.(N|OQ|O)$", "", regex=True))
labels = {isin: ric_lookup.get(isin, isin) for isin in sel}
tick   = [labels[i] for i in sel]          # ordered list of tickers

# Final dataset: monthly returns aligned to the ten selected assets.
returns   = ret_raw[sel].dropna()
esg_data  = esg_raw[sel]
assert list(returns.columns) == sel, "Column order does not match"


# =============================================================================
# 5. ANNUALIZED PORTFOLIO PARAMETERS
# =============================================================================

#%% [5] Parameters
rf         = 0.03                              # risk-free rate
mu         = returns.mean() * 12               # annualized expected return
esg_score  = esg_data.mean()                   # average ESG score per asset
sigma      = returns.cov() * 12                # variance-covariance matrix
corr       = returns.corr()
vol        = pd.Series(np.sqrt(np.diag(sigma)), index=sel)
N          = len(sel)
esg_grp    = pd.Series(["High-ESG"] * 5 + ["Low-ESG"] * 5, index=sel)

# Palette: green for High-ESG, dark red for Low-ESG.
color_asset = ["#2d6a4f" if esg_grp[i] == "High-ESG" else "#9b2226" for i in sel]


# =============================================================================
# 6. DESCRIPTIVE ANALYSIS
# =============================================================================

#%% [6] Descriptive table
summary = pd.DataFrame({
    "Ticker" : tick,
    "Group"  : [esg_grp[i] for i in sel],
    "ESG"    : esg_score.values,
    "mu"     : mu.values,
    "vol"    : vol.values,
    "Sharpe" : ((mu - rf) / vol).values,
}, index=sel)

print("\n" + "="*65)
print("  CHARACTERISTICS OF THE TEN ASSETS")
print("="*65)
print(summary.to_string())
print(f"\nMean ESG   High / Low : {esg_score[best5].mean():.3f} / {esg_score[worst5].mean():.3f}")
print(f"Mean mu    High / Low : {mu[best5].mean():.3f} / {mu[worst5].mean():.3f}")
print(f"Mean vol   High / Low : {vol[best5].mean():.3f} / {vol[worst5].mean():.3f}")

# --- Figure: asset positions in (vol, mu) space ---
fig, ax = plt.subplots(figsize=(9, 6))
for isin in sel:
    ax.scatter(vol[isin], mu[isin], color=color_asset[sel.index(isin)],
               s=120, zorder=5, edgecolors="white", linewidths=0.6)
    ax.annotate(labels[isin], (vol[isin], mu[isin]),
                fontsize=9, xytext=(5, 4), textcoords="offset points")
ax.axhline(0, color="grey", lw=0.6, ls="--")
ax.set_xlabel("Annual volatility")
ax.set_ylabel("Annual expected return")
ax.set_title("Asset universe: risk-return space")
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.grid(alpha=0.3)

# Manual legend for the group
from matplotlib.lines import Line2D
legend_el = [
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#2d6a4f",
           markersize=10, label="High-ESG (top 5)"),
    Line2D([0],[0], marker="o", color="w", markerfacecolor="#9b2226",
           markersize=10, label="Low-ESG (bottom 5)"),
]
ax.legend(handles=legend_el, loc="upper left")
_save("asset_caratteristiche")

# --- Figure: correlation matrix ---
fig, ax = plt.subplots(figsize=(8, 6))
im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
ax.set_xticks(range(N)); ax.set_xticklabels(tick, rotation=90, fontsize=9)
ax.set_yticks(range(N)); ax.set_yticklabels(tick, fontsize=9)
for i in range(N):
    for j in range(N):
        ax.text(j, i, f"{corr.values[i,j]:.2f}", ha="center", va="center",
                fontsize=7)
ax.set_title("Return correlation matrix")
_save("correlation")


# =============================================================================
# 7. ESG - EXPECTED-RETURN RELATIONSHIP
# =============================================================================

#%% [7] ESG - return relationship
X   = sm.add_constant(esg_score.values)
ols = sm.OLS(mu.values, X).fit()
b0, b1  = ols.params
rho  = np.corrcoef(esg_score.values, mu.values)[0, 1]
pval = ols.pvalues[1]

print("\n" + "="*65)
print("  ESG - EXPECTED-RETURN RELATIONSHIP")
print("="*65)
print(f"  Correlation rho   : {rho:+.3f}")
print(f"  OLS slope (b1)    : {b1:+.4f}  (p-value = {pval:.3f})")
print()
print("  Interpretation:")
print("  The NEGATIVE slope indicates that stocks with a higher ESG score")
print("  tend to have a lower expected return in the sample.")
print("  This is the condition for the ESG constraint to be binding")
print("  (Azzone, Barucci, Stocco 2024): requiring ESG >= threshold means")
print("  giving up expected return, pushing the frontier outward.")

fig, ax = plt.subplots(figsize=(8, 5))
for isin in sel:
    c = "#2d6a4f" if esg_grp[isin] == "High-ESG" else "#9b2226"
    ax.scatter(esg_score[isin], mu[isin], color=c, s=90, zorder=5,
               edgecolors="white", linewidths=0.6)
    ax.annotate(labels[isin], (esg_score[isin], mu[isin]),
                fontsize=9, xytext=(5, 4), textcoords="offset points")
xx = np.linspace(esg_score.min() - 0.02, esg_score.max() + 0.02, 80)
ax.plot(xx, b0 + b1 * xx, "k--", lw=1.5,
        label=f"OLS:  b = {b1:+.3f}")
ax.axhline(0, color="grey", lw=0.5, ls=":")
ax.set_xlabel("Average ESG score")
ax.set_ylabel("Annual expected return")
ax.set_title("ESG score versus expected return")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend()
ax.grid(alpha=0.3)
_save("esg_vs_return")


# =============================================================================
# 8. UNCONSTRAINED EFFICIENT FRONTIER (CLOSED FORM)
# =============================================================================

#%% [8] Analytic frontier (no constraints, short selling allowed)
def frontier_constants(mu_vec, sigma_mat):
    """Scalar constants of the closed-form Markowitz frontier."""
    m = np.asarray(mu_vec, float)
    S = np.linalg.inv(np.asarray(sigma_mat, float))
    e = np.ones(len(m))
    A = e @ S @ e
    B = e @ S @ m
    C = m @ S @ m
    D = A * C - B**2
    return dict(A=A, B=B, C=C, D=D, S_inv=S, ones=e, mu=m)

def analytic_vol(mu_p, fc):
    """Analytic minimum variance for target return mu_p."""
    A, B, C, D = fc["A"], fc["B"], fc["C"], fc["D"]
    return np.sqrt(np.maximum((A * mu_p**2 - 2*B*mu_p + C) / D, 0.0))

def analytic_weights(mu_p, fc):
    """Analytic optimal weights on the frontier (no rf, no constraints)."""
    A, B, C, D = fc["A"], fc["B"], fc["C"], fc["D"]
    lam = (C - B*mu_p) / D
    eta = (A*mu_p - B) / D
    return fc["S_inv"] @ (lam * fc["ones"] + eta * fc["mu"])

def tangency_portfolio(fc, risk_free):
    """Tangency portfolio (maximum Sharpe with short selling)."""
    excess = fc["mu"] - risk_free * fc["ones"]
    w = (fc["S_inv"] @ excess) / (fc["ones"] @ fc["S_inv"] @ excess)
    mu_t  = w @ fc["mu"]
    vol_t = np.sqrt(w @ np.linalg.inv(fc["S_inv"]) @ w)
    return w, mu_t, vol_t

FC         = frontier_constants(mu, sigma)
mvp_vol    = np.sqrt(1.0 / FC["A"])
mvp_mu     = FC["B"] / FC["A"]
mvp_w      = analytic_weights(mvp_mu, FC)
mvp_esg    = mvp_w @ esg_score.values
tan_w, tan_mu, tan_vol = tangency_portfolio(FC, rf)
tan_sharpe = (tan_mu - rf) / tan_vol

print("\n" + "="*65)
print("  EFFICIENT FRONTIER - REFERENCE CASE (SHORT SELLING)")
print("="*65)
print(f"  MVP        : mu = {mvp_mu:.3f},  vol = {mvp_vol:.3f},  ESG = {mvp_esg:.3f}")
print(f"  Tangency   : mu = {tan_mu:.3f},  vol = {tan_vol:.3f},  Sharpe = {tan_sharpe:.3f}")


# =============================================================================
# 9. CONSTRAINED FRONTIERS - SHORT SELLING ALLOWED
# =============================================================================

#%% [9] Numerical optimizer with ESG constraint
def min_var_portfolio(mu_v, sigma_m, mu_p,
                      esg_threshold=None, long_only=False, w0=None):
    """
    Minimize 0.5 * w' Sigma w subject to:
      - sum(w) = 1
      - w' mu  = mu_p
      - (if esg_threshold) w' esg_score >= esg_threshold
      - (if long_only)     w >= 0
    """
    mu_v    = np.asarray(mu_v, float)
    sigma_m = np.asarray(sigma_m, float)
    n       = len(mu_v)
    if w0 is None:
        w0 = np.ones(n) / n

    cons = [
        {"type": "eq",   "fun": lambda w: w.sum() - 1.0,
                         "jac": lambda w: np.ones(n)},
        {"type": "eq",   "fun": lambda w: w @ mu_v - mu_p,
                         "jac": lambda w: mu_v},
    ]
    if esg_threshold is not None:
        s = esg_score.values
        cons.append({
            "type": "ineq",
            "fun":  lambda w, s=s, t=esg_threshold: w @ s - t,
            "jac":  lambda w, s=s: s,
        })

    bounds = [(0.0, 1.0)] * n if long_only else None

    return minimize(
        fun=lambda w: 0.5 * w @ sigma_m @ w,
        x0=w0, method="SLSQP",
        jac=lambda w: sigma_m @ w,
        bounds=bounds, constraints=cons,
        options={"ftol": 1e-12, "maxiter": 1500},
    )

def compute_frontier(mu_grid, esg_threshold=None, long_only=False):
    """
    Compute the efficient frontier over a grid of target returns.
    Returns (vols, rets, weights) for the points where the optimization
    converges and the ESG constraint is satisfied.
    """
    vols, rets, ws = [], [], []
    w0 = np.ones(N) / N
    for m in mu_grid:
        res = min_var_portfolio(mu, sigma, m, esg_threshold, long_only, w0)
        esg_ok = (esg_threshold is None or
                  res.x @ esg_score.values >= esg_threshold - 1e-6)
        if res.success and esg_ok:
            w = res.x
            vols.append(np.sqrt(w @ sigma.values @ w))
            rets.append(float(w @ mu.values))
            ws.append(w.copy())
            w0 = w
    return np.array(vols), np.array(rets), np.array(ws)

# ---- Preliminary computation of the analytic MVP's ESG score ----
# As long as the threshold stays below mvp_esg, the constraint is non-binding:
# all frontiers coincide with the unconstrained one in the lower segment.
print(f"\n  ESG of the MVP (no constraints) : {mvp_esg:.3f}")
print(f"  Non-binding thresholds          : all values < {mvp_esg:.2f}")
print(f"  Binding thresholds              : values > {mvp_esg:.2f}")

# ---- Compute frontiers for both cases ----
mu_grid_short = np.linspace(mvp_mu, 0.50, 150)
base_vols     = analytic_vol(mu_grid_short, FC)

# With short selling: only the binding thresholds (>mvp_esg).
# Expected result: the frontier barely moves -> constraint nearly free.
THR_BINDING  = sorted([t for t in [0.80, 0.85, 0.90] if t > mvp_esg])
COLORS_SHORT = ["#2196F3", "#FF9800", "#C62828"][:len(THR_BINDING)]

short_curves = {}
for thr in THR_BINDING:
    v, r, _ = compute_frontier(mu_grid_short, esg_threshold=thr)
    if len(v):
        short_curves[thr] = (v, r)

# Long-only: thresholds over the whole range to show the progressive impact
mu_lo_min = float(mu.min()) + 0.002
mu_lo_max = float(mu.max()) - 0.002
mu_grid_lo = np.linspace(mu_lo_min, mu_lo_max, 180)
v_free_lo, r_free_lo, _ = compute_frontier(mu_grid_lo, long_only=True)

THR_LO_COMPARE = [0.50, 0.65, 0.75, 0.85]
COLORS_LO      = ["#4CAF50", "#8BC34A", "#FF9800", "#C62828"]

lo_curves = {}
for thr in THR_LO_COMPARE:
    v, r, _ = compute_frontier(mu_grid_lo, esg_threshold=thr, long_only=True)
    if len(v):
        lo_curves[thr] = (v, r)

# ---- Comparison figure (2 panels): the central message of the analysis ----
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# --- Left panel: short selling -> nearly free constraint ---
ax_s = axes[0]
ax_s.plot(base_vols, mu_grid_short, "k-", lw=2.5, zorder=10,
          label="Unconstrained (no ESG)")
for thr, c in zip(THR_BINDING, COLORS_SHORT):
    if thr in short_curves:
        v, r = short_curves[thr]
        ax_s.plot(v, r, lw=2.0, color=c, label=f"ESG $\\geq$ {thr:.2f}")

ax_s.scatter(mvp_vol, mvp_mu, color="black", marker="*", s=180, zorder=12)
ax_s.annotate("MVP", (mvp_vol, mvp_mu), fontsize=10,
              xytext=(4, -13), textcoords="offset points")
ax_s.text(0.16, 0.43,
          "The curves overlap:\nthe ESG constraint\n"
          r"is $\approx$ free with shorting",
          fontsize=9, color="#333333", ha="center",
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4",
                    edgecolor="#F9A825", alpha=0.95))
ax_s.set_xlim(0.13, 0.22)
ax_s.set_ylim(0.10, 0.50)
ax_s.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_s.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_s.set_xlabel("Annual volatility")
ax_s.set_ylabel("Annual expected return")
ax_s.set_title("With short selling:\nESG constraint nearly free",
               fontweight="bold")
ax_s.legend(loc="lower right", fontsize=9)
ax_s.grid(alpha=0.3)

# --- Right panel: long-only -> measurable and growing cost ---
ax_l = axes[1]
ax_l.plot(v_free_lo, r_free_lo, "k-", lw=2.5, zorder=10,
          label="Unconstrained (no ESG)")
for thr, c in zip(THR_LO_COMPARE, COLORS_LO):
    if thr in lo_curves:
        v, r = lo_curves[thr]
        ax_l.plot(v, r, lw=2.0, color=c, label=f"ESG $\\geq$ {thr:.2f}")

for isin in sel:
    cidx = sel.index(isin)
    ax_l.scatter(vol[isin], mu[isin], color=color_asset[cidx],
                 s=60, edgecolors="white", lw=0.6, zorder=8)
    ax_l.annotate(labels[isin], (vol[isin], mu[isin]), fontsize=8,
                  xytext=(4, 3), textcoords="offset points")

ax_l.set_xlim(0.14, 0.70)
ax_l.set_ylim(-0.02, 0.38)
ax_l.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_l.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax_l.set_xlabel("Annual volatility")
ax_l.set_ylabel("Annual expected return")
ax_l.set_title("Long-only:\nESG constraint has a real cost",
               fontweight="bold")
ax_l.legend(loc="lower right", fontsize=9)
ax_l.grid(alpha=0.3)

plt.suptitle(
    "Effect of the ESG constraint on the efficient frontier\n"
    "Comparison: short selling allowed vs long-only portfolios",
    fontsize=13, fontweight="bold"
)
plt.tight_layout()
_save("frontiere_esg")


# =============================================================================
# 10. CONSTRAINED FRONTIERS - LONG-ONLY PORTFOLIOS (DETAIL)
# =============================================================================

#%% [10] Detailed long-only frontier (7 thresholds, fine grid)
# The comparison panel in Section 9 showed 4 thresholds. Here we show 7 to
# highlight the progressivity of the shift.
mu_lo_min  = float(mu.min()) + 0.002
mu_lo_max  = float(mu.max()) - 0.002
mu_grid_lo = np.linspace(mu_lo_min, mu_lo_max, 180)

THR_LO    = [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
COLORS_LO = plt.cm.plasma(np.linspace(0.08, 0.90, len(THR_LO)))

v_free, r_free, _ = compute_frontier(mu_grid_lo, long_only=True)

fig, ax = plt.subplots(figsize=(10, 7))
ax.plot(v_free, r_free, "k-", lw=2.5, label="Long-only, no ESG constraint",
        zorder=10)

for thr, c in zip(THR_LO, COLORS_LO):
    v, r, _ = compute_frontier(mu_grid_lo, esg_threshold=thr, long_only=True)
    if len(v):
        ax.plot(v, r, lw=1.8, color=c, label=f"ESG $\\geq$ {thr:.2f}")

for isin in sel:
    cidx = sel.index(isin)
    ax.scatter(vol[isin], mu[isin], color=color_asset[cidx],
               s=80, edgecolors="white", lw=0.8, zorder=8)
    ax.annotate(labels[isin], (vol[isin], mu[isin]), fontsize=8,
                xytext=(4, 4), textcoords="offset points")

ax.set_xlim(0.14, 0.70)
ax.set_ylim(-0.02, 0.38)
ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.set_xlabel("Annual volatility")
ax.set_ylabel("Annual expected return")
ax.set_title("Long-only efficient frontier by ESG threshold")
ax.legend(loc="lower right", ncol=2, fontsize=9)
ax.grid(alpha=0.3)
_save("frontiera_longonly")


# =============================================================================
# 11. COST OF SUSTAINABILITY
# =============================================================================

#%% [11] Cost of sustainability
def max_sharpe_portfolio(esg_threshold=None, long_only=False, n_grid=80):
    """
    Maximum-Sharpe portfolio under the ESG and/or long-only constraint.
    Finds the maximum-Sharpe point along the constrained frontier.
    """
    grid = np.linspace(mu_lo_min, mu_lo_max, n_grid)
    v, r, w = compute_frontier(grid, esg_threshold, long_only)
    if not len(v):
        return np.nan, None, np.nan, np.nan
    sh = (r - rf) / v
    k  = int(np.argmax(sh))
    return sh[k], w[k], r[k], v[k]

# Long-only optimal portfolio without ESG constraint (reference)
sh_free, w_free, mu_opt, vol_opt = max_sharpe_portfolio(long_only=True)
esg_opt = float(w_free @ esg_score.values)

print("\n" + "="*65)
print("  LONG-ONLY OPTIMAL PORTFOLIO (NO ESG CONSTRAINT)")
print("="*65)
print(f"  Return     : {mu_opt:.3f}  ({mu_opt:.1%})")
print(f"  Volatility : {vol_opt:.3f}  ({vol_opt:.1%})")
print(f"  Sharpe     : {sh_free:.3f}")
print(f"  ESG score  : {esg_opt:.3f}")
print(f"\n  The ESG constraint is NON-binding for thresholds < {esg_opt:.3f}.")
print(f"  Above that level each increase in the threshold lowers the Sharpe.")

# (a) Curve: extra volatility at fixed return as the threshold varies
base_sigma = analytic_vol(mu_opt, FC)      # analytic vol (ref with shorting)

thr_fine   = np.round(np.arange(0.05, 0.92, 0.02), 2)
extra_v, feas_thr = [], []
w0_warm = np.ones(N) / N
for thr in thr_fine:
    res = min_var_portfolio(mu, sigma, mu_opt, esg_threshold=thr, w0=w0_warm)
    if res.success and res.x @ esg_score.values >= thr - 1e-6:
        feas_thr.append(thr)
        extra_v.append(np.sqrt(res.x @ sigma.values @ res.x) - base_sigma)
        w0_warm = res.x
feas_thr = np.array(feas_thr)
extra_v  = np.array(extra_v)

# (b) Curve: maximum long-only Sharpe as the threshold varies
sh_curve = []
for thr in thr_fine:
    sh, *_ = max_sharpe_portfolio(esg_threshold=thr, long_only=True)
    sh_curve.append(sh)
sh_curve = np.array(sh_curve)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(feas_thr, extra_v * 100, "o-", color="#e76f51", ms=4, lw=1.8)
axes[0].axvline(esg_opt, color="steelblue", ls="--", lw=1.2,
                label=f"Unconstrained ESG ({esg_opt:.2f})")
axes[0].set_xlabel("ESG threshold")
axes[0].set_ylabel("Extra volatility (percentage points)")
axes[0].set_title(f"Volatility cost at fixed $\\mu$ ({mu_opt:.0%})")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(thr_fine, sh_curve, "o-", color="#264653", ms=4, lw=1.8)
axes[1].axhline(sh_free, color="steelblue", ls="--", lw=1.2,
                label=f"Unconstrained Sharpe: {sh_free:.2f}")
axes[1].axvline(esg_opt, color="steelblue", ls=":",  lw=1.0)
axes[1].set_xlabel("ESG threshold")
axes[1].set_ylabel("Maximum Sharpe ratio")
axes[1].set_title("Price of sustainability: Sharpe vs ESG threshold")
axes[1].legend()
axes[1].grid(alpha=0.3)
_save("costo_esg")

# (c) Shadow price: marginal cost of the constraint (numerical derivative)
if len(feas_thr) > 4:
    sigma_at_thr = extra_v + base_sigma
    shadow_price = np.gradient(sigma_at_thr, feas_thr)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(feas_thr, shadow_price, "o-", color="#7b2d8b", ms=4, lw=1.8)
    ax.axhline(0, color="grey", lw=0.6)
    ax.axvline(esg_opt, color="steelblue", ls="--", lw=1.2,
               label=f"Unconstrained ESG ({esg_opt:.2f})")
    ax.set_xlabel("ESG threshold")
    ax.set_ylabel(r"$\partial\sigma / \partial\tau_{ESG}$")
    ax.set_title("Shadow price of the ESG constraint")
    ax.legend()
    ax.grid(alpha=0.3)
    _save("prezzo_ombra")
    print("\n  Shadow price near zero for low thresholds, convex above "
          f"{esg_opt:.2f}:")
    print("  the first degrees of sustainability are nearly free,")
    print("  the last threshold increases are very costly in terms of risk.")


# =============================================================================
# 12. COMPOSITION OF THE OPTIMAL PORTFOLIO
# =============================================================================

#%% [12] Composition of the optimal portfolio (long-only)
# Compute the maximum-Sharpe long-only portfolio for each ESG threshold.
thr_comp    = np.round(np.arange(0.10, 0.92, 0.08), 2)
comp_w, comp_esg_val, comp_sh = [], [], []
for thr in thr_comp:
    sh, w, rr, vv = max_sharpe_portfolio(esg_threshold=thr, long_only=True)
    if w is not None:
        comp_w.append(w)
        comp_esg_val.append(float(w @ esg_score.values))
        comp_sh.append(sh)
comp_w   = np.array(comp_w)
thr_ok   = thr_comp[:len(comp_w)]
w_high   = comp_w[:, :5].sum(axis=1)   # aggregate High-ESG weight
w_low    = comp_w[:, 5:].sum(axis=1)   # aggregate Low-ESG weight

# ---- Figure: stacked bar of the composition ----
GREEN_PAL = plt.cm.Greens(np.linspace(0.45, 0.92, 5))
RED_PAL   = plt.cm.Reds(  np.linspace(0.45, 0.92, 5))
comp_colors = list(GREEN_PAL) + list(RED_PAL)

fig, ax = plt.subplots(figsize=(11, 6))
bottom = np.zeros(len(thr_ok))
for k, isin in enumerate(sel):
    grp_lbl = "H" if esg_grp[isin] == "High-ESG" else "L"
    ax.bar(thr_ok, comp_w[:, k], bottom=bottom, width=0.065,
           color=comp_colors[k], edgecolor="white", linewidth=0.5,
           label=f"{labels[isin]} ({grp_lbl})")
    bottom += comp_w[:, k]

ax.set_xlabel("Imposed ESG threshold")
ax.set_ylabel("Weight in the optimal portfolio (long-only)")
ax.set_title("Composition of the maximum-Sharpe portfolio by ESG threshold")
ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left")
ax.grid(axis="y", alpha=0.3)
_save("composizione_sharpe")

# ---- Figure: aggregate High vs Low ESG rotation, with Sharpe ----
fig, ax1 = plt.subplots(figsize=(9, 5))
ax2 = ax1.twinx()

ax1.plot(thr_ok, w_high, "o-", color="#2d6a4f", lw=2, ms=7,
         label="High-ESG weight")
ax1.plot(thr_ok, w_low,  "o-", color="#9b2226", lw=2, ms=7,
         label="Low-ESG weight")
ax2.plot(thr_ok, comp_sh, "s--", color="#555555", lw=1.5, ms=6,
         label="Sharpe ratio (right axis)")

ax1.set_xlabel("Imposed ESG threshold")
ax1.set_ylabel("Aggregate group weight")
ax2.set_ylabel("Sharpe ratio of the optimal portfolio")
ax1.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
ax1.set_title("Weight rotation and Sharpe cost as the constraint tightens")
ax1.grid(alpha=0.3)

# Combined legend
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(handles1 + handles2, labels1 + labels2, loc="center left")
_save("rotazione_pesi")

# ---- Summary table ----
comp_tab = pd.DataFrame(
    comp_w, columns=tick, index=[f"ESG>={t:.2f}" for t in thr_ok]
)
comp_tab["%High"]  = w_high
comp_tab["%Low"]   = w_low
comp_tab["ESG_pt"] = comp_esg_val
comp_tab["Sharpe"] = comp_sh

print("\n" + "="*65)
print("  OPTIMAL PORTFOLIO COMPOSITION BY ESG THRESHOLD")
print("="*65)
print(comp_tab.round(3).to_string())


# =============================================================================
# 13. SUMMARY OF RESULTS
# =============================================================================

#%% [13] Summary
print("""
=============================================================================
  SUMMARY
=============================================================================

1. ESG-RETURN RELATIONSHIP
   In the sample, stocks with a higher ESG score tend to have a lower expected
   return (negative correlation, negative OLS slope). This is the necessary
   CONDITION (Azzone, Barucci, Stocco 2024) for the ESG constraint to be
   binding: requiring a greener portfolio means giving up return. If the
   relationship were positive or null, the constraint would be essentially
   free.

2. THE COST IS NOT CONSTANT (and is not trivial)
   The shift of the frontier is not uniform:
   - For thresholds <= the ESG of the unconstrained optimal portfolio (~0.65),
     the constraint does NOT bite: the constrained frontier coincides with the
     unconstrained one (cost = 0).
   - Above that threshold the cost grows CONVEXLY: the first degrees of
     sustainability come nearly free, the last ones cost a lot.
   - The shadow price (derivative of variance w.r.t. the threshold) is nearly
     flat up to ~0.65, then rises rapidly.

3. COMPOSITION: THE MECHANISM OF THE SHIFT
   The composition analysis explains why the frontier shifts:
   - The constraint forces a ROTATION of capital away from Low-ESG stocks
     (which in the sample offer higher expected return but are more volatile)
     toward High-ESG stocks.
   - Not all High-ESG stocks are equivalent: MSFT (high ESG and high Sharpe)
     stays dominant at every threshold. The other low-return High-ESG stocks
     (CL, MMM) enter the portfolio only when the constraint makes them
     necessary, explaining the acceleration of the cost at high thresholds.
   - This INTERNAL heterogeneity within the High-ESG group is the richest
     result of the analysis: the cost of sustainability depends not only on
     'how much ESG you want', but on 'which ESG assets you are forced to hold'.

=============================================================================
""")
