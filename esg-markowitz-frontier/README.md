# ESG Constraints and the Markowitz Efficient Frontier

An empirical study of how imposing a minimum average **ESG score** on a portfolio reshapes the Markowitz efficient frontier — and what that sustainability constraint actually costs in risk-adjusted return.

> Academic group project — *Sustainable Finance* (Finanza Quantitativa), GSoM (Politecnico di Milano), April 2026.

---

## Research question

Adding a constraint *sᵀw ≥ τ* (average ESG score above threshold τ) can only shrink the feasible set, so the constrained frontier can only coincide with or sit outside the unconstrained one. The interesting question is **when the constraint actually bites** — and, when it does, how the cost of sustainability behaves as the threshold rises.

## Data & methodology

Ten Russell 3000 stocks over **Jan 2017 – Dec 2022** — the **five highest-ESG and five lowest-ESG** names from the universe (monthly ESG scores normalised to [0,1], monthly returns). Assets with fewer than 60 monthly observations are dropped. The analysis compares:

- the **closed-form unconstrained frontier** (short selling allowed),
- **ESG-constrained frontiers** with short selling, and
- **ESG-constrained frontiers, long-only** (`w ≥ 0`),

solving the constrained mean–variance problem numerically across a grid of ESG thresholds, then measuring the **shadow price** of the constraint and decomposing the optimal portfolio's weights.

*The ESG-score and return datasets come from a commercial data provider and are **not redistributed** here.*

## Key findings

- **Negative ESG–return relationship** in the sample (OLS slope ≈ −0.078, ρ ≈ −0.305) — the necessary condition for the constraint to be binding.
- **With short selling the constraint is nearly free**, even at very high thresholds (ESG ≥ 0.90): one can satisfy the average-score target by shorting low-ESG and going long high-ESG names without materially changing the variance structure (constrained MVPs differ by only 0.3–0.5 pp of volatility).
- **Long-only tells a different story.** The long-only maximum-Sharpe portfolio has a "natural" ESG score of ≈ **0.65** (μ ≈ 24.5%, σ ≈ 18.3%, Sharpe ≈ 1.18). Below that threshold the constraint is slack and costless; **above τ\* ≈ 0.65 the cost in volatility and Sharpe grows convexly**, with the shadow price rising sharply.
- **Mechanism:** the cost comes from a rotation of weights away from higher-return low-ESG stocks toward high-ESG stocks. Heterogeneity *within* the high-ESG group (e.g. MSFT, ESG ≈ 0.91 and the sample's best individual Sharpe ≈ 1.00) shapes how fast the cost accelerates.

Full derivations, figures and tables are in [`docs/esg_markowitz_frontier_report.pdf`](docs/esg_markowitz_frontier_report.pdf). The framework follows Azzone et al. (2024).

## Tech stack

Python · NumPy · pandas · SciPy (`optimize.minimize`) · statsmodels · Matplotlib

## How to run

```bash
pip install -r requirements.txt
python notebooks/esg_efficient_frontier.py
```

The script expects `ESG_monthly_clean.csv` and `Returns.csv` in the working directory (not included — see *Data* above) and writes paper-ready figures (`fig_*.png`).

## Team

Nicolò Ferrari · Antonio Fontanella · Luca Palumbo · Lorenzo Meloncelli

## License

MIT — see [`LICENSE`](../LICENSE).
