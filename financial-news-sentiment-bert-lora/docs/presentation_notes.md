# Presentation Speech — AI for Finance

**Project:** Explaining same-day stock price moves from financial news headlines
**Group:** Nicolò Ferrari · Luca Palumbo · Lorenzo Meloncelli · Antonio Fontanella
**Format:** ~7–8 min, presented live from `Sentiment Analysis 2.ipynb`

---

## How to use this script

- Each part is timed; the **total is ≈ 7.5 min**, leaving a small buffer.
- *Italic bracketed lines* are **stage cues** — what to scroll to / point at. They are **not spoken**.
- The plain paragraphs are the words to say. Speak slowly; the numbers are the point, let them land.
- Run the **fast path** before you start (Sections 1–5 loaded from Drive) so every result cell is already populated. **Skip Section 7 live** — it retrains 12 models; just show its output and conclusions.
- Suggested 4-way split below — reassign freely; if one person presents, ignore the speaker labels.

---

## Part 1 — Problem & Data  ·  Speaker 1  ·  ~1:45

*[SHOW: Section 1 title cell — the project title and the four names.]*

Good morning. Our project asks a simple but hard question: **can financial news headlines explain the stock market's move on the same day?** Not predict the future — explain, after the fact. If a stock moved, do that day's headlines carry the signal that says which way?

We frame it as **text classification with three labels: positive, negative, and no_change.** Given a day's headlines for a stock, the model predicts the direction of that day's price move.

*[SHOW: Section 2 — Kaggle download, and Section 3 — the per-ticker label distribution bar charts.]*

For the data, we took a Kaggle set of about **1.4 million financial headlines** and focused on **five stocks — Google, Tesla, Nvidia, Coca-Cola, and Netflix** — chosen to mix high-volatility names like Tesla and Nvidia with a stable one like Coca-Cola.

The headlines come unlabeled, so **we build the ground truth ourselves from prices.** We pull daily closes from Yahoo Finance, compute the same-day percentage change, and apply a **one-percent threshold**: up more than one percent is *positive*, down more than one percent is *negative*, and everything in between is *no_change*. As you can see, the classes are **imbalanced** — this matters a lot for how we measure success later.

*[Hand to Speaker 2.]*

---

## Part 2 — The approach & the sentiment backbone  ·  Speaker 2  ·  ~2:00

*[SHOW: Section 4 — the chronological split print-out (Train / Val / Test sizes and dates).]*

A quick but important point on the split. Because our labels come from prices, this is **time-series data** — so we split **chronologically: we train on the past, and validate and test on the future**, roughly seventy / ten / twenty percent. We never let the model see the future, which would be unrealistic and would leak information.

*[SHOW: Section 6 — "LoRA on Twitter Financial News Sentiment".]*

Now the core idea. The price labels are **noisy** — a single headline doesn't cleanly cause a one-percent move. So instead of throwing a raw model at noisy labels, we use **two-stage transfer learning.**

**Stage one builds a sentiment expert.** We take BERT and fine-tune it with **LoRA** on a clean dataset of about twelve thousand finance tweets labeled Bearish, Bullish, or Neutral. LoRA is the efficient part: it **freezes all of BERT and trains only small low-rank matrices** plus a classification head — a tiny fraction of the parameters, so it's fast and hard to overfit.

*[SHOW: Stage-1 confusion matrix and the macro-F1 print-out.]*

And Stage one **works**: on held-out tweets it reaches about **88% accuracy and 0.84 macro-F1**. So we now have a backbone that genuinely understands financial sentiment. The real question is whether *that* understanding transfers to our noisy price task.

*[Hand to Speaker 3.]*

---

## Part 3 — The two models  ·  Speaker 3  ·  ~1:30

*[SHOW: Section 6 — "Model 1 — Linear probe" markdown, then its confusion matrix.]*

We test transfer with **two models of increasing capacity.**

**Model 1 is a linear probe.** We **freeze the whole sentiment backbone** and train only a single linear layer on top, against the price labels. This asks the cleanest possible question: *are the frozen sentiment features already useful for predicting price direction, with no further adaptation?*

*[SHOW: Model 2 markdown — the stacked-LoRA explanation.]*

**Model 2 adds capacity.** We keep Stage one frozen but **stack a second, fresh LoRA adapter on top** and let it adapt the representation to our labels. Both adapters stay active, so their effects sum in the forward pass — BERT, plus the Stage-1 delta, plus the Stage-2 delta. Only the new adapter and a fresh head are trained.

So Model 1 tests the frozen features; Model 2 tests whether *adapting* them helps. Keep that contrast in mind — because the answer is the interesting part.

*[Hand to Speaker 4.]*

---

## Part 4 — Baselines, results & takeaway  ·  Speaker 4  ·  ~2:15

*[SHOW: Section 6 — "Dummy Classifier" baseline output, then the comparison table and bar chart.]*

Before trusting any score, we need an **honest baseline.** With imbalanced classes, accuracy lies, so we compare against **dummy classifiers** and we focus on **macro-F1**. One key point: under our class imbalance, **random chance is not 0.33** — measured empirically, the chance macro-F1 floor is about **0.32.**

*[Point at the comparison table / bar chart.]*

Now the results. **Model 1, the simple linear probe, is the best — 0.415 macro-F1, about thirty percent above chance.** Model 2, with all its extra capacity, actually does **slightly worse**, at 0.397. So adapting the representation didn't help — it **hurt a little.**

*[SHOW: Section 7 — the rank cross-validation plot (mean ± std per rank).]*

We checked that this isn't a fluke with **time-series cross-validation** across four LoRA sizes — ranks 8, 16, 32, and 64. The pattern is striking: the **means are basically flat**, but the **smallest adapter is the most stable**, and **bigger adapters mainly add variance** — the error bars grow. More capacity buys noise, not accuracy.

*[SHOW: Section 8 — Conclusions.]*

So our **takeaway** has two parts. First, the good news: **headlines do carry real signal** — both models clearly beat chance, so news genuinely helps explain same-day moves. Second, the deeper lesson: **the ceiling here is the labels, not the model.** Our ground truth — a fixed one-percent threshold applied across stocks with very different volatilities, plus near-duplicate headlines sharing one daily label — is inherently noisy. That's why the **cheapest model wins and extra capacity only adds variance.** The honest scientific result isn't a bigger model — it's knowing *where the wall is, and why.*

Thank you — we're happy to take questions.

---

## Appendix — likely questions (not spoken; for your defense)

- **Why macro-F1 and not accuracy?** Classes are imbalanced; a majority-class predictor gets decent accuracy but only ~0.13 macro-F1. Macro-F1 weights all three classes equally, so it actually measures learning.
- **Why BERT and not FinBERT?** We wanted Stage 1 to do real work and to show the transfer effect cleanly; plain `bert-base-uncased` makes the sentiment fine-tuning meaningful rather than pre-baked.
- **Why does the simpler model (Model 1) win?** With noisy labels, extra trainable capacity fits noise, not signal — classic bias–variance. The rank-CV shows variance rising with rank while the mean stays flat.
- **What exactly is "stacked LoRA"?** Two adapters are active at once and never merged: the frozen Stage-1 adapter keeps contributing its sentiment knowledge while the new Stage-2 adapter trains. Their low-rank deltas simply add in the forward pass.
- **Isn't same-day "explanation" just hindsight?** Yes — by design. We explain the move after the fact; we are not claiming a tradeable forecast.
- **Biggest limitation?** The label definition: one fixed % threshold across different-volatility stocks, and many headlines on the same day collapse to one label. A volatility-adjusted or per-stock threshold is the obvious next step.
- **Did you leak future data?** No — chronological split, and the CV uses expanding windows that only ever train on the past.
