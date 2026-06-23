# POLIMI GSoM — Quantitative Finance Projects

A curated collection of group projects from my **M.Sc. in Quantitative Finance** at **GSoM – Politecnico di Milano**. Each project applies quantitative methods to a real financial problem, end to end in Python. 

## Projects

| Project | Field | What it does | Headline result |
|---|---|---|---|
| [📈 Portfolio Optimization (Markowitz)](portfolio-optimization-markowitz/) | Asset management | Mean–variance allocation on a multi-sector S&P 500 subset: shrinkage covariance, efficient frontiers, CAPM, Black–Litterman | Tangency vs. constrained frontiers; CAPM betas 0.44–1.32 |
| [🤖 Statistical Arbitrage EWS (LSTM)](statistical-arbitrage-lstm-ews/) | Quant / ML | Pairs-trading Early Warning System: OPTICS + cointegration pair selection, LSTM novelty detector | **Sharpe 2.60 vs 0.93** on COVID stress test |
| [📰 Financial News Sentiment (BERT + LoRA)](financial-news-sentiment-bert-lora/) | NLP | Explaining same-day stock moves from headlines via two-stage LoRA transfer learning | macro-F1 0.415, +30% over chance |
| [🌱 ESG & the Efficient Frontier](esg-markowitz-frontier/) | Sustainable finance | How an ESG-score constraint reshapes the Markowitz frontier on Russell 3000 stocks | Cost of ESG is ~free with shorting, convex above τ\*≈0.65 long-only |

Each subfolder has its own README with full methodology, results, and instructions.

## Tech across the portfolio

Python · PyTorch · scikit-learn · statsmodels · HuggingFace Transformers + PEFT (LoRA) · NumPy · pandas · SciPy · Matplotlib · seaborn

## Repository structure

```
POLIMI_GSOM/
├── portfolio-optimization-markowitz/   # Markowitz / CAPM / Black-Litterman
├── statistical-arbitrage-lstm-ews/     # Pairs trading + LSTM anomaly detection
├── financial-news-sentiment-bert-lora/ # BERT + LoRA financial NLP
├── esg-markowitz-frontier/             # ESG-constrained efficient frontier
├── LICENSE
└── README.md
```

## A note on data & reproducibility

Raw datasets are **not** included: some are course-provided or proprietary (e.g. Bloomberg-sourced macro data) and others are too large for GitHub (Kaggle headline corpus). API tokens and credentials are never committed. Each project README explains how to obtain or regenerate its data.

## Teams

These were group projects. Contributors are credited by name in each project's README, and will be added as repository collaborators on GitHub.

- **Portfolio Optimization** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella
- **Statistical Arbitrage EWS** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella
- **Financial News Sentiment** — Nicolò Ferrari, Luca Palumbo, Lorenzo Meloncelli, Antonio Fontanella
- **ESG & the Efficient Frontier** — Nicolò Ferrari, Antonio Fontanella, Luca Palumbo, Lorenzo Meloncelli

## License

Released under the MIT License — see [`LICENSE`](LICENSE).
