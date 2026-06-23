# Results Summary — Statistical Arbitrage EWS
**GSoM Hackathon · Zenti Dataset · Weekly data, Jan 2000 – Apr 2021**

---

## Quick Reference

|  | Exp 1 · Test 2018–2021 | Exp 2 · Test 2020–2021 *(COVID)* |
|---|:---:|:---:|
| **Threshold — Portfolio Sharpe** | 0.853 | 0.930 |
| **LSTM — Portfolio Sharpe** | -0.195 | **2.596** |
| **Threshold — Avg ROI** | +1.22% | +1.37% |
| **LSTM — Avg ROI** | -1.13% | **+10.23%** |
| **LSTM vs Threshold (Sharpe)** | -1.048 | **+1.666** |

---

## Setup

### Pairs (4 cointegrated, from `Zenti Pair Selection.ipynb`)

| Pair | β | Half-life | Hurst |
|------|--:|----------:|------:|
| LG30TRUU / LP01TREU | 3.343 | 82 wk | 0.508 |
| LP01TREU / MXEU | 0.101 | 64 wk | 0.549 |
| EMUSTRUU / LMBITR | 0.775 | 25 wk | 0.505 |
| LMBITR / LUACTRUU | 0.348 | 32 wk | 0.535 |

### Common settings

| Parameter | Value |
|-----------|-------|
| Entry / exit threshold | ±2σ z-score entry · 0 exit |
| Execution delay | 1 week |
| Stabilizing period | 5 bars |
| Commission + market impact | 0.28% per leg |
| Short rental | 1% / 52 per week |
| Normalisation | StandardScaler fit on train only |
| Sharpe formula | (μ / σ) × √52 · in-position weeks · rf = 0 |

### LSTM architecture

SpreadLSTM: `LSTM(hidden) → Dropout(0.2) → Dense(1)` · trained on Y=0 weeks only  
Anomaly score = |actual\_norm − predicted\_norm| · signal held while score > threshold  
Tuning via nested time-series CV: 3 expanding outer folds · inner val = last 20% of outer-train  
Grid: `n_in` ∈ {4, 8, 13, 26} · `n_units` ∈ {16, 32} · `threshold` ∈ {0.5, 1.0} → 16 combos

---

## Experiment 1 — Train Jan 2000 – Dec 2017 · Test Jan 2018 – Apr 2021

*Test window: ~175 weeks (3.3 yr) · COVID falls inside the test period*

### Threshold Model

| Pair | Train Sharpe | Train ROI | Test Sharpe | Test ROI |
|------|-------------:|----------:|------------:|---------:|
| LG30TRUU / LP01TREU | 0.513 | +17.56% | 0.000 | 0.00% |
| LP01TREU / MXEU | 0.914 | +198.48% | 0.000 | 0.00% |
| EMUSTRUU / LMBITR | 1.794 | +85.92% | 0.058 | +0.18% |
| LMBITR / LUACTRUU | -0.414 | -7.94% | 1.991 | +4.72% |
| **Portfolio** | **0.922** | **avg +73.5%** | **0.853** | **avg +1.22%** |

Train: 12 trades · 10W / 2L &nbsp;&nbsp;|&nbsp;&nbsp; Test: 2 trades · 2W / 0L

### LSTM Anomaly Detection

**Final params** (inner CV on 2000–2017):

| Pair | n_in | n_units | threshold |
|------|-----:|--------:|----------:|
| LG30TRUU / LP01TREU | 13 | 16 | 0.5 |
| LP01TREU / MXEU | 13 | 16 | 0.5 |
| EMUSTRUU / LMBITR | 8 | 32 | 1.0 |
| LMBITR / LUACTRUU | 8 | 32 | 0.5 |

| Pair | Train Sharpe | Train ROI | Test Sharpe | Test ROI |
|------|-------------:|----------:|------------:|---------:|
| LG30TRUU / LP01TREU | 2.239 | +10.72% | 0.850 | +0.78% |
| LP01TREU / MXEU | 0.771 | +78.53% | 0.149 | -0.35% |
| EMUSTRUU / LMBITR | 3.216 | +12.96% | -2.744 | -2.82% |
| LMBITR / LUACTRUU | 0.274 | +2.12% | -0.347 | -2.12% |
| **Portfolio** | **0.847** | **avg +26.1%** | **-0.195** | **avg -1.13%** |

Train: 89 trades · 44W / 45L &nbsp;&nbsp;|&nbsp;&nbsp; Test: 19 trades · 7W / 12L

---

## Experiment 2 — Train Jan 2000 – Dec 2019 · Test Jan 2020 – Apr 2021

*Test window: ~70 weeks (1.3 yr) · COVID is the entire test period*

### Threshold Model

| Pair | Train Sharpe | Train ROI | Test Sharpe | Test ROI |
|------|-------------:|----------:|------------:|---------:|
| LG30TRUU / LP01TREU | 0.518 | +17.77% | 0.000 | 0.00% |
| LP01TREU / MXEU | 0.901 | +197.38% | 0.000 | 0.00% |
| EMUSTRUU / LMBITR | 1.566 | +96.26% | 0.058 | +0.18% |
| LMBITR / LUACTRUU | -0.170 | -2.83% | 2.119 | +5.29% |
| **Portfolio** | **0.950** | **avg +77.2%** | **0.930** | **avg +1.37%** |

Train: 13 trades · 10W / 3L &nbsp;&nbsp;|&nbsp;&nbsp; Test: 2 trades · 2W / 0L

### LSTM Anomaly Detection

**Final params** (inner CV on 2000–2019):

| Pair | n_in | n_units | threshold | val Sharpe |
|------|-----:|--------:|----------:|-----------:|
| LG30TRUU / LP01TREU | 4 | 16 | 1.0 | 0.000 ⚠ |
| LP01TREU / MXEU | 4 | 16 | 0.5 | 0.000 ⚠ |
| EMUSTRUU / LMBITR | 4 | 16 | 1.0 | 33.99 |
| LMBITR / LUACTRUU | 13 | 16 | 1.0 | 108.53 |

> ⚠ val Sharpe = 0 means no trades were generated in the validation window (2016–2019),
> so hyperparameter selection for these pairs is effectively random within the grid.

| Pair | Train Sharpe | Train ROI | Test Sharpe | Test ROI |
|------|-------------:|----------:|------------:|---------:|
| LG30TRUU / LP01TREU | 1.155 | +6.97% | 3.018 | +0.35% |
| LP01TREU / MXEU | 0.499 | +175.16% | **4.010** | **+40.38%** |
| EMUSTRUU / LMBITR | 1.735 | +20.48% | -0.425 | -0.70% |
| LMBITR / LUACTRUU | 0.308 | +0.71% | 0.451 | +0.90% |
| **Portfolio** | **0.563** | **avg +50.8%** | **2.596** | **avg +10.23%** |

Train: 74 trades · 44W / 30L &nbsp;&nbsp;|&nbsp;&nbsp; Test: 14 trades · 9W / 5L

---

## Comparison

### Portfolio test metrics across all experiments

| Metric | Threshold Exp 1 | Threshold Exp 2 | LSTM Exp 1 | LSTM Exp 2 |
|--------|:-----------:|:-----------:|:------:|:------:|
| Portfolio Sharpe | 0.853 | 0.930 | -0.195 | **2.596** |
| Avg ROI | +1.22% | +1.37% | -1.13% | **+10.23%** |
| Avg Annual ROI | +0.37% | +1.03% | -0.34% | **+7.60%** |
| Total trades | 2 | 2 | 19 | 14 |
| Win rate | 100% | 100% | 37% | 64% |

### LSTM vs Threshold (test Sharpe delta)

| Experiment | Threshold | LSTM | LSTM advantage |
|------------|:---------:|:----:|:--------------:|
| Exp 1 · 2018–2021 | 0.853 | -0.195 | -1.048 |
| Exp 2 · 2020–2021 | 0.930 | 2.596 | **+1.666** |

---

## Key Takeaways

**1. Threshold model is robust but inert under COVID.**
The ±2σ threshold is calibrated on training variance. The COVID shock drives spreads far beyond 2σ, but the model produces only 2 trades in both test windows — LMBITR/LUACTRUU is the sole active pair. Performance is nearly identical across splits (+1.22% vs +1.37%) regardless of whether COVID is in the test.

**2. LSTM is purpose-built for regime breaks — but only when they are out-of-sample.**
In Exp 1 (COVID inside a 3.3 yr test mixed with normal periods), the LSTM loses money (Sh=–0.195), dragged down by EMUSTRUU/LMBITR (Sh=–2.744). In Exp 2, with COVID as the dedicated test window, the anomaly score spikes sharply on the shock, the model enters mean-reversion positions, and markets recover: Sharpe=2.596, ROI=+10.2%.

**3. LP01TREU / MXEU is the key driver of Exp 2.**
This pair (European IG corporate bonds vs MSCI Europe equities) compressed severely in the COVID crash and rebounded quickly. The LSTM caught the anomaly cleanly: ROI=+40.4% in ~70 weeks, Sharpe=4.01. In Exp 1 the same pair contributed nothing (ROI=–0.35%, Sh=0.149).

**4. Train Sharpe is not predictive of test Sharpe for the LSTM.**
Exp 1 train Sharpe (0.847) > Exp 2 train Sharpe (0.563), yet test Sharpe reverses completely (–0.195 vs +2.596). The LSTM learns normal-period spread dynamics well; its edge emerges only when the test contains a genuine out-of-distribution event.

**5. Hyperparameter selection is unreliable without an anomaly in the validation set.**
In Exp 2, two of four pairs return val Sharpe=0 in inner CV (2016–2019 contained no large dislocations), so selected params are effectively arbitrary for those pairs. The strong test result partly reflects luck. A more robust design would ensure at least one CV fold covers a stress period.
