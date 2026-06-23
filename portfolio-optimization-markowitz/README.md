# Portfolio Optimization on a Multi-Sector S&P 500 Subset

Mean–variance asset allocation on a six-stock, multi-sector subset of the S&P 500, built end to end in Python: robust covariance estimation, Markowitz efficient frontiers (with and without a risk-free asset and under allocation constraints), CAPM regressions, and a Black–Litterman overlay.

> Academic group project — Asset Management course, GSoM (Politecnico di Milano), January 2026.

---

## Business question

Given a small, deliberately diversified universe of large-cap U.S. equities, how should a rational investor allocate capital? The project walks through the full mean–variance workflow — estimating expected returns and risk, tracing the efficient frontier, identifying the tangency portfolio, measuring how real-world constraints degrade the risk–return trade-off, and finally testing the assets against the CAPM and incorporating subjective views via Black–Litterman.

## Dataset

Daily total returns and market caps for six S&P 500 stocks, plus the S&P 500 index close price, through 11 December 2025. The six tickers were chosen to span distinct sectors:

| Ticker | Company | Sector |
|--------|---------|--------|
| ADBE | Adobe Inc. | Technology |
| XEL | Xcel Energy Inc. | Utilities |
| CAG | Conagra Brands Inc. | Consumer Staples |
| BK | Bank of New York Mellon | Financials |
| PCG | PG&E Corporation | Utilities |
| WELL | Welltower Inc. | Healthcare REIT |

## Methodology

1. **Descriptive analysis** of each stock and its sector profile.
2. **Return & risk estimation** — expected returns via an exponentially weighted average (EWMA); covariance via a **shrinkage estimator toward a constant-correlation target** (shrinkage intensity *k = 0.35*, average pairwise correlation ≈ 0.3), trading a small bias for a large reduction in estimation noise.
3. **Efficient frontier with a risk-free asset** (annual Rf = 3%) and identification of the **tangency portfolio**.
4. **Constrained frontiers** — re-solving the optimization under (a) a 40% combined-weight constraint on two assets and (b) a 5% minimum weight per asset, then quantifying the efficiency loss versus the unconstrained frontier.
5. **CAPM** — time-series regressions of each stock's excess return on the market risk premium; estimation of α, β and R².
6. **Black–Litterman** — a Bayesian overlay blending market-implied equilibrium returns with subjective views.

## Selected results

- **Tangency portfolio** is highly leveraged long/short (expected return ≈ 89.9%, volatility ≈ 79.2% annualized), versus the **Global Minimum-Variance Portfolio** at ≈ 6.9% return and ≈ 16.7% volatility — illustrating how aggressively the unconstrained tangency solution tilts toward high-Sharpe assets.
- **Constraints dominate downward**, as theory predicts: both constrained frontiers are dominated by the unconstrained Markowitz frontier, since each constraint shrinks the feasible set. To reach a 6% annual target, unconstrained Markowitz achieves 2.74% volatility, versus 6.32% and 11.28% under the two constraint sets.
- **CAPM** — market beta is statistically significant for all six assets. BK (β ≈ 1.32) and ADBE (β ≈ 1.19) are the most aggressive; CAG (β ≈ 0.44) and XEL (β ≈ 0.55) the most defensive. Alphas are economically small and statistically insignificant, and R² ranges from ≈ 0.09 to ≈ 0.51 — consistent with CAPM, with a non-trivial idiosyncratic component remaining.

Full discussion and figures are in [`report/report_group1.pdf`](report/report_group1.pdf).

## Tech stack

Python · NumPy · pandas · SciPy (`optimize.minimize`) · statsmodels · Matplotlib · seaborn

## Repository structure

```
.
├── notebooks/
│   └── portfolio_optimization.ipynb   # Full analysis (exercises 1–7)
├── report/
│   └── report_group1.pdf          # Written report with figures and commentary
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/portfolio_optimization.ipynb
```

The notebook was originally developed in Google Colab. The first cell mounts Google Drive (`from google.colab import drive`); when running locally, skip that cell and point the data-loading cells to a local path. The underlying return dataset is course-provided and is not redistributed here.

## Team

Antonio Fontanella · Luca Palumbo · Lorenzo Meloncelli · Nicolò Ferrari

## License

Released under the MIT License — see [`LICENSE`](../LICENSE).
