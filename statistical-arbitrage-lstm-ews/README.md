# Statistical Arbitrage Early Warning System — LSTM Anomaly Detection

A pairs-trading **Early Warning System (EWS)** that detects anomalous spread behaviour in weekly macro data. An LSTM novelty detector, trained only on calm ("risk-on") weeks, learns normal spread dynamics and flags divergences as potential stress events. Benchmarked against a classical ±2σ z-score strategy on a COVID-19 stress test.

> Academic group project — *New Technologies for Finance* course, GSoM (Politecnico di Milano) Hackathon.

---

## Headline result

On the COVID-19 stress test (Jan 2020 – Apr 2021), the LSTM EWS reached a **portfolio Sharpe of 2.60** and **+10.23% average ROI**, beating the threshold baseline by **+1.67 Sharpe points**. The best pair — European IG corporate bonds vs. MSCI Europe equities — delivered a test Sharpe of **4.01** and **+40.4% ROI** over ~70 weeks.

| Metric (COVID test) | Threshold baseline | LSTM EWS |
|---|:--:|:--:|
| Portfolio Sharpe | 0.93 | **2.60** |
| Avg ROI | +1.37% | **+10.23%** |
| Avg annual ROI | +1.03% | **+7.60%** |
| Total trades | 2 | 14 |

The threshold model fires only twice — the COVID shock pushes spreads so far beyond their historical norm that a fixed ±2σ band, calibrated on training variance, is rarely breached. The LSTM instead catches the March 2020 dislocation and profits from the subsequent mean reversion.

## Problem

Detect regime breaks in financial spread data, framed as **novelty detection**: identify the weeks where the spread between two instruments diverges anomalously from its historical norm, and trade the expected reversion.

## Data

1,111 weekly observations (Jan 2000 – Apr 2021) across 42 macro features — bond indices, equities, FX, rates, commodities — with a binary risk-on/risk-off label. *The dataset is course-provided (Bloomberg-sourced) and is **not redistributed** in this repository.*

## Methodology

1. **Pair selection** — OPTICS clustering on PCA-reduced returns shrinks 861 candidate pairs to 20 economically coherent ones, then a sequential **cointegration → Hurst (H < 0.55) → Ornstein–Uhlenbeck half-life** filter keeps the 4 that pass every test.
2. **Threshold baseline** — z-score normalised on training stats only (no leakage); enter long at z < −2σ / short at z > +2σ, exit at 0; 1-week execution delay, 5-bar stabilisation, 0.28% cost per leg, short-rental fees.
3. **LSTM anomaly detector** — a `SpreadLSTM` (PyTorch) trained **exclusively on risk-on weeks** to predict the next spread value; the prediction error is the anomaly score, with the threshold tuned via nested time-series cross-validation (never on the test set).

The 4 selected pairs:

| Pair | β | Half-life | Hurst |
|------|--:|----------:|------:|
| LG30TRUU / LP01TREU | 3.34 | 82 wk | 0.508 |
| LP01TREU / MXEU | 0.10 | 64 wk | 0.549 |
| EMUSTRUU / LMBITR | 0.78 | 25 wk | 0.505 |
| LMBITR / LUACTRUU | 0.35 | 32 wk | 0.535 |

Full numbers and ablations are in [`RESULTS.md`](RESULTS.md); the abstract and slides are in [`docs/`](docs/).

## Tech stack

Python · PyTorch · scikit-learn (OPTICS, PCA, StandardScaler) · statsmodels (cointegration) · NumPy · pandas · Matplotlib · seaborn

## Repository structure

```
.
├── notebooks/
│   ├── 01_pair_selection.ipynb     # PCA + OPTICS clustering, cointegration pipeline
│   ├── 02_threshold_model.ipynb    # ±2σ z-score baseline + backtest
│   └── 03_lstm_model.ipynb         # SpreadLSTM novelty detector + backtest
├── docs/
│   ├── statistical_arbitrage_lstm_ews_abstract.pdf   # 2-page technical abstract
├── RESULTS.md                      # Full result tables
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_pair_selection.ipynb
```

Notebooks expect the course dataset locally; update the data path in the first cells. The raw data is not included (see *Data* above).

## Team

Lorenzo Meloncelli · Nicolò Ferrari · Luca Palumbo · Antonio Fontanella

## License

MIT — see [`LICENSE`](../LICENSE).
