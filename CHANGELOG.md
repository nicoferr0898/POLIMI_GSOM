# Changelog

All notable changes to the **POLIMI_GSOM** repository are recorded here.
Most recent entries first.

## 2026-06-23

### Added
- Initial publication of the portfolio as a public GitHub repo
  (`nicoferr0898/POLIMI_GSOM`), containing three projects:
  `portfolio-optimization-markowitz`, `statistical-arbitrage-lstm-ews`,
  and `financial-news-sentiment-bert-lora`, plus index `README.md`,
  `LICENSE`, and `.gitignore`.
- Repository description and topics (quantitative-finance,
  portfolio-optimization, statistical-arbitrage, lstm, nlp, bert, lora,
  pytorch, python).

### Changed
- Set the program name in the index README to *M.Sc. in Quantitative
  Finance, GSoM – Politecnico di Milano*.
- Revised README content and contributor names.
- Moved `portfolio-optimization-markowitz/portfolio_optimization.ipynb`
  into a `notebooks/` subfolder for consistency with the other projects;
  updated README path references.

### Fixed
- Sentiment-analysis notebook failed to render on GitHub because of an
  invalid `metadata.widgets` block (missing `state` key, left by
  ipywidgets/tqdm). Stripped the block; cells and outputs unaffected.

### Removed
- `financial-news-sentiment-bert-lora/docs/` directory.

### Excluded from publication
- `PUBLISH_GUIDE.md` (internal publishing runbook) is git-ignored and not
  committed.
