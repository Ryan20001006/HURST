# HURST_RC — Reservoir Computing with Hurst Exponent for Stock Price Prediction

Replication and extension of **Santos et al., *Physica D* 476 (2025) 134698**.  
The paper proposes augmenting an Echo State Network (ESN) with a rolling Hurst exponent
to forecast daily closing prices across 11 global stock indices.

---

## Repository structure

```
HURST/
├── hurst_rc.py          # Core pipeline: all models + next-day prediction
├── plot_figures.py      # Publication-quality figures for all 11 indices
├── download_data.py     # Download & cache price data from yfinance
├── pyESN.py             # Local copy of the pyESN Echo State Network library
├── CSV/
│   ├── price_data.csv   # Historical closing prices (downloaded by download_data.py)
│   └── predictions.csv  # Next-day predictions saved by hurst_rc.py
├── figures/             # Output figures (auto-generated)
├── HURST_RC 2/
│   └── SOURCE CODE HURST_RC.ipynb  # Authors' original notebook
└── Material/            # Lecture slides (ANN, RL)
```

---

## Requirements

All packages are available for **Python 3.13**. If `python3` on your machine points to a
different version (check with `which -a python3`), use `python3.13` explicitly.

```bash
pip install pandas numpy yfinance scikit-learn matplotlib statsmodels hurst pyESN torch
```

---

## Quickstart

### 1. Download price data

```bash
python3.13 download_data.py
```

Downloads closing prices for all 11 indices (2009-12-31 → 2026-05-12) from Yahoo Finance
and saves them to `CSV/price_data.csv`. Edit `START_DATE` / `END_DATE` at the top of
the file to change the window.

### 2. Run the core analysis (single index)

```bash
python3.13 hurst_rc.py
```

Runs HURST_RC, corrected HURST_RC, naïve persistence, RC (no Hurst), ARIMA, and LSTM on
**S&P 500** by default. Prints a metrics comparison table, saves figures to `figures/`,
and appends a next-day prediction to `CSV/predictions.csv`.

To change the index or timeframe, edit the `__main__` block at the bottom of `hurst_rc.py`:

```python
COL   = "SP500"          # any key from TICKER_MAP (see below)
START = "2009-12-31"
END   = "2026-05-12"
```

### 3. Generate all figures (all 11 indices)

```bash
python3.13 plot_figures.py
```

Produces six figures in `figures/` covering all indices. Per-index date ranges are
configured in the `DATE_RANGES` dict at the top of `plot_figures.py`.

---

## Supported indices

| Key | Index | Yahoo ticker |
|---|---|---|
| `DJI` | Dow Jones Industrial Average | `^DJI` |
| `SP500` | S&P 500 | `^GSPC` |
| `CAC40` | CAC 40 (France) | `^FCHI` |
| `NYSE` | NYSE Composite | `^NYA` |
| `NASDAQ` | NASDAQ Composite | `^IXIC` |
| `FTSE` | FTSE 100 (UK) | `^FTSE` |
| `N225` | Nikkei 225 (Japan) | `^N225` |
| `SSE` | Shanghai Composite | `000001.SS` |
| `SZ` | Shenzhen Component | `399001.SZ` |
| `SH` | Shanghai B-Share | `000002.SS` |
| `HSI` | Hang Seng (Hong Kong) | `^HSI` |

---

## Model specifications (S&P 500)

All models share the same date range (2009-12-31 → 2026-05-12) and 80/20 chronological
train/test split. The first 99 rows are consumed by the 100-day Hurst window burn-in,
so RC-based models effectively start from day 100.

### HURST_RC (main model)

Echo State Network augmented with a rolling Hurst exponent.

| Parameter | Value |
|---|---|
| Reservoir size | 200 |
| Spectral radius | 0.7 (S&P 500, DJI) / 0.9 (CAC40, NYSE) / 0.85 (NASDAQ, FTSE) |
| Sparsity | 0.2 |
| Noise | 0.1 |
| Inputs | 2 — normalised price + rolling Hurst exponent H |
| Hurst window | 100 trading days (`hurst.compute_Hc`, kind='random_walk') |
| Normalisation | MinMaxScaler fit on full series (train + test) |
| Train target | Next-step normalised price |
| Evaluation | **Author's version**: prediction at *t* vs actual at *t* (inflated R²) |

### RC baseline (no Hurst)

Identical to HURST_RC but with a single price input (no Hurst feature). Used to isolate
the contribution of the Hurst exponent.

### ARIMA(5,1,0)

| Parameter | Value |
|---|---|
| Order | p=5, d=1, q=0 |
| Fitting | One-shot on training set, no refit |
| Forecast | Rolling one-step-ahead (`refit=False`) |
| Evaluation | Honest: predict *t+1* vs actual *t+1* |

### LSTM

| Parameter | Value |
|---|---|
| Hidden size | 50 |
| Layers | 1 |
| Lookback | 20 days |
| Epochs | 50 |
| Learning rate | 0.001 |
| Input | Log returns (avoids non-stationarity of raw prices) |
| Evaluation | Honest: predict *t+1* price |

### Naïve persistence

Predicts tomorrow's price = today's price. Acts as a lower-bound benchmark.

---

## Output figures

| File | Description |
|---|---|
| `figure1_hurst_input.png` | Normalised price + rolling Hurst H for all indices, shaded train/test regions |
| `figure2_predictions_page*.png` | Actual vs predicted price with zoom inset (4 indices per page) |
| `figure3_metrics_bar.png` | HURST_RC vs naïve persistence — MAE, RMSE, MAPE, R² bar chart |
| `figure4_metrics_table.png` | Heatmap of all metrics across all indices |
| `figure5_hit_rate.png` | Directional accuracy: overall, rolling 30-day, cumulative, up/down breakdown |
| `figure6_model_comparison_bar.png` | HURST_RC vs RC vs ARIMA vs LSTM vs naïve — all indices |
| `hurst_rc_results.png` | Single-index summary (from `hurst_rc.py`) |
| `hurst_rc_sensitivity.png` | MAE and R² vs spectral radius sweep |

---

## Known issues in the original paper

Two methodological issues were identified during replication:

1. **Evaluation misalignment**: The paper compares the ESN prediction made using input at
   time *t* against the actual price at *t*, not *t+1*. Because the input already contains
   the current normalised price, the model trivially learns to echo it back. This inflates
   R² to ~0.997 due to autocorrelation rather than genuine forecasting skill.
   `run_hurst_rc_corrected()` in `hurst_rc.py` fixes this.

2. **Data leakage in normalisation**: `MinMaxScaler` is fit on the full series including
   the test set. The global min/max of the test period therefore leaks into the training
   normalisation. Correct practice is to fit the scaler on training data only.

---

## Replicating the paper's exact results

Exact numerical replication is not possible because:

- The paper does not disclose the random seed used for ESN initialisation.
- Yahoo Finance retroactively adjusts historical prices for splits and dividends; the same
  date range downloaded today returns different values than at the time of writing.
- The paper's data source (likely Bloomberg or Refinitiv) differs from Yahoo Finance.

---

## Reference

> Santos, T. et al. (2025). *Stock market prediction using reservoir computing augmented
> with Hurst exponents*. Physica D: Nonlinear Phenomena, 476, 134698.
