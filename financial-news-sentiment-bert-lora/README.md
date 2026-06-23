# Explaining Same-Day Stock Moves from News Headlines — BERT + LoRA

Can a day's financial news headlines **explain** (after the fact) the direction of that day's stock price move? We frame it as three-class text classification — *positive / negative / no_change* — and tackle it with two-stage transfer learning on **BERT fine-tuned with LoRA**.

> Academic group project — *AI for Finance* course, GSoM (Politecnico di Milano).

---

## Approach

Price-derived labels are inherently noisy (one headline rarely causes a clean 1% move), so instead of throwing a raw model at noisy labels we use **two-stage transfer learning**:

1. **Stage 1 — sentiment backbone.** Fine-tune `bert-base-uncased` with **LoRA** (HuggingFace `peft`) on ~11.9k labelled finance tweets (Twitter Financial News Sentiment: Bearish / Bullish / Neutral). LoRA freezes BERT and trains only small low-rank matrices plus a head — fast and resistant to overfitting. **Result: 0.877 accuracy / 0.841 macro-F1** on held-out tweets.
2. **Stage 2 — transfer to the price task.** Two models of increasing capacity:
   - **Model 1 — linear probe:** freeze the Stage-1 backbone, train only a linear head on the price labels (tests whether the frozen sentiment features are already useful).
   - **Model 2 — stacked LoRA:** keep Stage 1 frozen and live, stack a *second* LoRA adapter on top so the deltas sum in the forward pass (tests whether *adapting* the representation helps).

## Data & labelling

A Kaggle set of ~1.4M financial news headlines, focused on five stocks chosen to mix volatility profiles — **Google, Tesla, Nvidia, Coca-Cola, Netflix**. Headlines ship unlabelled, so ground truth is built from prices: daily closes from Yahoo Finance, same-day % change, **±1% threshold** → positive / negative / no_change. Classes are imbalanced, so evaluation uses **macro-F1**, not accuracy. The split is strictly **chronological** (≈70/10/20, past → train, future → val/test) to avoid look-ahead leakage.

*The headline dataset is downloaded from Kaggle (not redistributed here); price data comes from Yahoo Finance via `yfinance`.*

## Results

| Model | Accuracy | Macro-F1 | vs. chance F1 (0.320) |
|---|:--:|:--:|:--:|
| Majority class | 0.236 | 0.127 | — |
| Stratified random (50 seeds) | 0.326 | 0.320 | floor |
| **Model 1 — linear probe** | **0.425** | **0.415** | **+30%** |
| Model 2 — stacked LoRA + head | 0.409 | 0.397 | +24% |

**Honest takeaway:** the frozen Stage-1 sentiment features beat the stacked-LoRA model — adding Stage-2 capacity did *not* help on this noisy task, a textbook case of more parameters not meaning more signal. A `k=3` expanding-window time-series cross-validation over four LoRA ranks confirmed the single-split result sits inside the CV band (i.e. it's stable, not a lucky seed). The honest conclusion: same-day headlines carry a *modest but real* signal — clearly above chance, but far from a trading edge.

## Tech stack

Python · PyTorch · HuggingFace Transformers + PEFT (LoRA) · Datasets · scikit-learn · yfinance · pandas · NumPy · Matplotlib

## Repository structure

```
.
├── notebooks/
│   └── sentiment_analysis.ipynb    # Full pipeline: data, labelling, Stage 1 & 2, CV, conclusions
├── docs/
│   └── presentation_notes.md       # Talk script walking through the notebook
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/sentiment_analysis.ipynb
```

You'll need (a) a Kaggle API token to download the headline dataset and (b) internet access for `yfinance`. **Store credentials in a local `.env` (git-ignored) — never commit API tokens.** The notebook was developed in Google Colab; the "fast path" cells load pre-trained adapters from Drive — when running locally, run the training cells instead.

## Team

Nicolò Ferrari · Luca Palumbo · Lorenzo Meloncelli · Antonio Fontanella

## License

MIT — see [`LICENSE`](../LICENSE).
