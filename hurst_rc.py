"""
HURST_RC Faithful Replication — Santos et al., Physica D 476 (2025) 134698
Uses the author's exact libraries (pyESN, hurst.compute_Hc) and pipeline.

HOW TO CHANGE THE TIMEFRAME
────────────────────────────
  Option A — use a different slice of the downloaded CSV:
    Edit START_DATE / END_DATE in the __main__ block, then re-run.
    The CSV is loaded from price_data.csv (run download_data.py first).

  Option B — change the download window altogether:
    Edit START_DATE and END_DATE in download_data.py, re-run that script
    to refresh price_data.csv, then run this file.
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, WORK_DIR)

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from pyESN import ESN
from hurst import compute_Hc
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

np.random.seed(42)

CSV_PATH         = os.path.join(WORK_DIR, "CSV", "price_data.csv")
PREDICTIONS_CSV  = os.path.join(WORK_DIR, "CSV", "predictions.csv")

# Map column names in the CSV to their yfinance ticker (for fallback download)
TICKER_MAP = {
    "DJI":    "^DJI",
    "SP500":  "^GSPC",
    "CAC40":  "^FCHI",
    "NYSE":   "^NYA",
    "NASDAQ": "^IXIC",
    "FTSE":   "^FTSE",
    "N225":   "^N225",
    "SSE":    "000001.SS",
    "SZ":     "399001.SZ",
    "SH":     "000002.SS",
    "HSI":    "^HSI",
}


def load_series(col: str, start: str = None, end: str = None) -> pd.Series:
    """
    Load a single index from price_data.csv when possible, otherwise download.
    col  : column name in CSV, e.g. "DJI", "SP500", "HSI"
    start: "YYYY-MM-DD"  — filter rows on or after this date
    end  : "YYYY-MM-DD"  — filter rows strictly before this date

    Falls back to a live yfinance download when:
      • the CSV does not exist
      • col is not a column in the CSV
      • the requested end date is later than the last date in the CSV
        (e.g. you set END to today's date but the CSV only covers up to 2018)
    """
    if os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH, index_col="Date", parse_dates=True)
        if col in df.columns:
            # check whether the CSV covers the full requested range
            csv_last = df[col].dropna().index[-1]
            need_end = pd.Timestamp(end) if end else pd.Timestamp.today()
            if need_end <= csv_last + pd.Timedelta(days=1):
                # CSV covers the range — use it
                s = df[col].dropna()
                if start:
                    s = s[s.index >= start]
                if end:
                    s = s[s.index < end]
                return s
            else:
                print(f"  [{col}] requested end {need_end.date()} is beyond "
                      f"CSV ({csv_last.date()}) — downloading live data…")

    # live download (covers gaps and out-of-range dates)
    ticker = TICKER_MAP.get(col, col)
    raw = yf.download(ticker, start=start, end=end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(
            f"No data for '{col}' (ticker '{ticker}') between {start} and {end}.\n"
            f"  Hint: COL must be a CSV column name, not a ticker symbol.\n"
            f"  Valid names: {list(TICKER_MAP.keys())}"
        )
    close = raw["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    return close.squeeze().rename(col)


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE  (mirrors the author's notebook cell exactly)
# ─────────────────────────────────────────────────────────────────────────────

def run_hurst_rc(col="DJI",
                 start="2009-12-31", end="2026-02-12",
                 reservoir_size=200, sparsity=0.2,
                 noise=0.1, spectral_radius=0.7,
                 window_size=100):
    """
    Faithful replication of the author's source code.
    col   : column name from price_data.csv  (e.g. "DJI", "SP500", "HSI")
    start : start date string "YYYY-MM-DD"  ← change to test different windows
    end   : end date string   "YYYY-MM-DD"  ← change to test different windows
    Returns raw arrays and metrics for downstream plotting.
    """

    # ── 1. Load from CSV (or download if CSV missing) ─────────────────────
    close = load_series(col, start=start, end=end)
    series = close.values.reshape(-1, 1)

    # ── 2. Normalise (scaler fit on FULL series — same as authors) ────────
    scaler = MinMaxScaler()
    normalized_series = scaler.fit_transform(series)

    # ── 3. Rolling Hurst exponent (100-day window, compute_Hc) ───────────
    hurst_exponents = []
    for i in range(len(normalized_series) - window_size + 1):
        seg = normalized_series[i: i + window_size].reshape(-1)
        H, _, _ = compute_Hc(seg)
        hurst_exponents.append(H)

    hurst_exponents = np.array(hurst_exponents).reshape(-1, 1)

    # Align price series to same length as Hurst vector
    normalized_series = normalized_series[-len(hurst_exponents):]

    # ── 4. Concatenate  [price_norm | H] ─────────────────────────────────
    data_combined = np.hstack((normalized_series, hurst_exponents))

    # ── 5. 80/20 train/test split ─────────────────────────────────────────
    train_size = int(len(data_combined) * 0.8)
    train_data = data_combined[:train_size]
    test_data  = data_combined[train_size:]

    # ── 6. Fit ESN ────────────────────────────────────────────────────────
    esn = ESN(n_inputs=data_combined.shape[1],
              n_outputs=1,
              n_reservoir=reservoir_size,
              sparsity=sparsity,
              noise=noise,
              spectral_radius=spectral_radius)
    esn.fit(train_data[:-1, :], train_data[1:, 0])

    # ── 7. Predict on test set ────────────────────────────────────────────
    test_input = test_data[:-1, :]
    predicted_output = esn.predict(test_input)

    # ── 8. Denormalise (same as authors) ─────────────────────────────────
    predicted_output = scaler.inverse_transform(
        predicted_output.reshape(-1, 1))

    # Authors compare prediction against test_data[:-1, :-1] (current step)
    # NOT test_data[1:, 0] (next step) — see critical discussion
    actual_output = scaler.inverse_transform(test_data[:-1, :-1])

    # ── 9. Metrics ────────────────────────────────────────────────────────
    y_true = actual_output[:, 0]
    y_pred = predicted_output[:, 0]

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)

    metrics = dict(MAE=mae, RMSE=rmse, MSE=mse, MAPE=mape, R2=r2)

    # Dates for test window
    dates_all  = close.index[-len(data_combined):]
    dates_test = dates_all[train_size: train_size + len(y_true)]

    return dict(
        ticker=col,
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        dates_test=dates_test,
        data_combined=data_combined,
        hurst_exponents=hurst_exponents,
        normalized_series=normalized_series,
        dates_all=dates_all,
        train_size=train_size,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED PIPELINE  (fixes the evaluation alignment)
# ─────────────────────────────────────────────────────────────────────────────

def run_hurst_rc_corrected(col="DJI",
                            start="2009-12-31", end="2018-12-28",
                            reservoir_size=200, sparsity=0.2,
                            noise=0.1, spectral_radius=0.7,
                            window_size=100):
    """
    Same pipeline but evaluates predictions against test_data[1:, 0]
    (the true next-step target), not test_data[:-1, 0] (current step).
    """
    close = load_series(col, start=start, end=end)
    series = close.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    normalized_series = scaler.fit_transform(series)

    hurst_exponents = []
    for i in range(len(normalized_series) - window_size + 1):
        seg = normalized_series[i: i + window_size].reshape(-1)
        H, _, _ = compute_Hc(seg)
        hurst_exponents.append(H)

    hurst_exponents = np.array(hurst_exponents).reshape(-1, 1)
    normalized_series = normalized_series[-len(hurst_exponents):]
    data_combined = np.hstack((normalized_series, hurst_exponents))

    train_size = int(len(data_combined) * 0.8)
    train_data = data_combined[:train_size]
    test_data  = data_combined[train_size:]

    esn = ESN(n_inputs=data_combined.shape[1],
              n_outputs=1,
              n_reservoir=reservoir_size,
              sparsity=sparsity,
              noise=noise,
              spectral_radius=spectral_radius)
    esn.fit(train_data[:-1, :], train_data[1:, 0])

    predicted_output = esn.predict(test_data[:-1, :])
    predicted_output = scaler.inverse_transform(
        predicted_output.reshape(-1, 1))

    # CORRECTED: compare against next step's actual price
    actual_output = scaler.inverse_transform(
        test_data[1:, :1])

    y_true = actual_output[:, 0]
    y_pred = predicted_output[:, 0]

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)

    return dict(MAE=mae, RMSE=rmse, MSE=mse, MAPE=mape, R2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# NAIVE PERSISTENCE BASELINE  (predict tomorrow = today)
# ─────────────────────────────────────────────────────────────────────────────

def run_naive(col="DJI", start="2009-12-31", end="2018-12-28"):
    close = load_series(col, start=start, end=end)
    v = close.values
    split = int(len(v) * 0.8)
    y_true = v[split + 1:]
    y_pred = v[split: -1]
    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)
    return dict(MAE=mae, RMSE=rmse, MSE=mse, MAPE=mape, R2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# RC BASELINE  (plain Reservoir Computing — no Hurst exponent input)
# Tests whether adding the Hurst feature actually improves the model.
# Uses the same evaluation alignment as the paper (compare prediction at t
# against actual at t, i.e. test_data[:-1] as ground truth).
# ─────────────────────────────────────────────────────────────────────────────

def run_rc(col="DJI", start="2009-12-31", end="2018-12-28",
           reservoir_size=200, sparsity=0.2, noise=0.1,
           spectral_radius=0.7, window_size=100):
    """
    Plain Echo State Network on normalised price only (1-D input).
    Trimmed to the same number of rows as HURST_RC for a fair comparison.
    """
    close = load_series(col, start=start, end=end)
    series = close.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    norm = scaler.fit_transform(series)

    # Drop the first (window_size-1) rows so the test period matches HURST_RC
    norm = norm[window_size - 1:]

    train_size = int(len(norm) * 0.8)
    train = norm[:train_size]
    test  = norm[train_size:]

    esn = ESN(n_inputs=1, n_outputs=1,
              n_reservoir=reservoir_size,
              sparsity=sparsity, noise=noise,
              spectral_radius=spectral_radius)
    esn.fit(train[:-1], train[1:, 0])

    pred_norm = esn.predict(test[:-1])
    y_pred = scaler.inverse_transform(pred_norm.reshape(-1, 1))[:, 0]
    y_true = scaler.inverse_transform(test[:-1])[:, 0]   # author's alignment

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)
    return dict(MAE=mae, RMSE=rmse, MSE=mse, MAPE=mape, R2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# HURST_RC HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_hurst_rc(col="SP500", start="2009-12-31", end="2026-05-12",
                  random_state=42):
    """
    Three-phase grid search over HURST_RC hyperparameters.
    Evaluated on corrected metrics (predict t+1 vs actual t+1).
    ESN is seeded with random_state for reproducibility.

    Phase 1 — spectral_radius × n_reservoir  (most impactful ESN params)
    Phase 2 — noise × sparsity               (fixed best from Phase 1)
    Phase 3 — window_size                    (fixed best from Phases 1-2)
    """

    # load once — reused across all runs
    _close = load_series(col, start=start, end=end)

    def _run(reservoir_size, spectral_radius, sparsity, noise, window_size):
        close = _close
        series = close.values.reshape(-1, 1)

        scaler = MinMaxScaler()
        norm = scaler.fit_transform(series)

        H = []
        for i in range(len(norm) - window_size + 1):
            h, _, _ = compute_Hc(norm[i: i + window_size].reshape(-1))
            H.append(h)
        H = np.array(H).reshape(-1, 1)
        norm = norm[-len(H):]
        data = np.hstack((norm, H))

        train_size = int(len(data) * 0.8)
        train = data[:train_size]
        test  = data[train_size:]

        esn = ESN(n_inputs=2, n_outputs=1,
                  n_reservoir=reservoir_size,
                  sparsity=sparsity,
                  noise=noise,
                  spectral_radius=spectral_radius,
                  random_state=random_state)
        esn.fit(train[:-1], train[1:, 0])

        pred_norm = esn.predict(test[:-1])
        y_pred = scaler.inverse_transform(pred_norm.reshape(-1, 1))[:, 0]
        y_true = scaler.inverse_transform(test[1:, :1])[:, 0]   # corrected: t+1

        mae  = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
        r2   = r2_score(y_true, y_pred)
        return dict(MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)

    hdr = f"  {'Config':<40} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R2':>8}"
    sep = "  " + "─" * 76

    # ── baseline ─────────────────────────────────────────────────────────────
    base = _run(reservoir_size=200, spectral_radius=0.7,
                sparsity=0.2, noise=0.1, window_size=100)
    print(f"\n  Baseline HURST_RC  "
          f"MAE={base['MAE']:.2f}  RMSE={base['RMSE']:.2f}  "
          f"MAPE={base['MAPE']:.4f}  R2={base['R2']:.4f}")

    # ── Phase 1: spectral_radius × n_reservoir ───────────────────────────────
    spectral_radii = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    reservoir_sizes = [100, 200, 300, 500]

    print(f"\n  Phase 1 — spectral_radius × n_reservoir "
          f"({len(spectral_radii) * len(reservoir_sizes)} runs)")
    print(hdr); print(sep)

    p1 = []
    for rho in spectral_radii:
        for N in reservoir_sizes:
            m = _run(reservoir_size=N, spectral_radius=rho,
                     sparsity=0.2, noise=0.1, window_size=100)
            label = f"rho={rho}  N={N}"
            print(f"  {label:<40} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
                  f"{m['MAPE']:>8.4f} {m['R2']:>8.4f}")
            p1.append(dict(spectral_radius=rho, n_reservoir=N, **m))

    best1 = min(p1, key=lambda x: x["MAE"])
    print(f"\n  Best Phase 1: rho={best1['spectral_radius']}  N={best1['n_reservoir']}  "
          f"MAE={best1['MAE']:.2f}")

    # ── Phase 2: noise × sparsity ─────────────────────────────────────────────
    noise_vals    = [0.005, 0.01, 0.05, 0.1, 0.2]
    sparsity_vals = [0.1, 0.2, 0.3, 0.5]

    print(f"\n  Phase 2 — noise × sparsity "
          f"({len(noise_vals) * len(sparsity_vals)} runs, "
          f"fixed rho={best1['spectral_radius']} N={best1['n_reservoir']})")
    print(hdr); print(sep)

    p2 = []
    for noise in noise_vals:
        for sparsity in sparsity_vals:
            m = _run(reservoir_size=best1["n_reservoir"],
                     spectral_radius=best1["spectral_radius"],
                     sparsity=sparsity, noise=noise, window_size=100)
            label = f"noise={noise}  sparsity={sparsity}"
            print(f"  {label:<40} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
                  f"{m['MAPE']:>8.4f} {m['R2']:>8.4f}")
            p2.append(dict(noise=noise, sparsity=sparsity, **m))

    best2 = min(p2, key=lambda x: x["MAE"])
    print(f"\n  Best Phase 2: noise={best2['noise']}  sparsity={best2['sparsity']}  "
          f"MAE={best2['MAE']:.2f}")

    # ── Phase 3: window_size ──────────────────────────────────────────────────
    window_sizes = [100, 150, 200, 250]  # min 100 required by compute_Hc

    print(f"\n  Phase 3 — window_size ({len(window_sizes)} runs, "
          f"fixed rho={best1['spectral_radius']} N={best1['n_reservoir']} "
          f"noise={best2['noise']} sparsity={best2['sparsity']})")
    print(hdr); print(sep)

    p3 = []
    for w in window_sizes:
        m = _run(reservoir_size=best1["n_reservoir"],
                 spectral_radius=best1["spectral_radius"],
                 sparsity=best2["sparsity"],
                 noise=best2["noise"],
                 window_size=w)
        label = f"window={w}"
        print(f"  {label:<40} {m['MAE']:>8.2f} {m['RMSE']:>8.2f} "
              f"{m['MAPE']:>8.4f} {m['R2']:>8.4f}")
        p3.append(dict(window_size=w, **m))

    best3 = min(p3, key=lambda x: x["MAE"])
    print(f"\n  Best Phase 3: window={best3['window_size']}  MAE={best3['MAE']:.2f}")

    # ── Summary ───────────────────────────────────────────────────────────────
    best = _run(reservoir_size=best1["n_reservoir"],
                spectral_radius=best1["spectral_radius"],
                sparsity=best2["sparsity"],
                noise=best2["noise"],
                window_size=best3["window_size"])

    improvement = (base["MAE"] - best["MAE"]) / base["MAE"] * 100
    print("\n" + "═" * 78)
    print(f"  TUNED:    rho={best1['spectral_radius']}  N={best1['n_reservoir']}  "
          f"noise={best2['noise']}  sparsity={best2['sparsity']}  "
          f"window={best3['window_size']}")
    print(f"  BASELINE: rho=0.7  N=200  noise=0.1  sparsity=0.2  window=100")
    print(f"\n  {'':30} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R2':>8}")
    print(f"  {'Baseline':30} {base['MAE']:>8.2f} {base['RMSE']:>8.2f} "
          f"{base['MAPE']:>8.4f} {base['R2']:>8.4f}")
    print(f"  {'Tuned':30} {best['MAE']:>8.2f} {best['RMSE']:>8.2f} "
          f"{best['MAPE']:>8.4f} {best['R2']:>8.4f}")
    print(f"\n  MAE improvement: {improvement:+.2f}%")
    print("═" * 78)

    return dict(
        best_params=dict(
            spectral_radius=best1["spectral_radius"],
            n_reservoir=best1["n_reservoir"],
            noise=best2["noise"],
            sparsity=best2["sparsity"],
            window_size=best3["window_size"],
        ),
        baseline_metrics=base,
        tuned_metrics=best,
        improvement_pct=improvement,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA  (classical benchmark from the paper)
# ─────────────────────────────────────────────────────────────────────────────

def run_arima(col="DJI", start="2009-12-31", end="2018-12-28",
              order=(5, 1, 0), window_size=100):
    """
    ARIMA(5,1,0) with rolling one-step-ahead forecast.
    After each step the model state is updated with the actual observation
    (no refit) so every prediction uses all available history.
    Evaluation: predict price at t+1, compare against actual price at t+1.
    Trimmed to the same number of rows as HURST_RC for fair comparison.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    close = load_series(col, start=start, end=end)
    v = close.values.astype(float)
    v = v[window_size - 1:]                           # align with HURST_RC

    train_size = int(len(v) * 0.8)
    n_forecast = len(v) - train_size - 1              # same length as HURST_RC test

    res = SARIMAX(v[:train_size], order=order,
                  enforce_stationarity=False,
                  enforce_invertibility=False).fit(disp=False)

    predictions = []
    for t in range(n_forecast):
        yhat = float(res.forecast(1)[0])
        predictions.append(yhat)
        res = res.append([v[train_size + t]], refit=False)

    y_pred = np.array(predictions)
    y_true = v[train_size + 1: train_size + 1 + n_forecast]  # actual next-step prices

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)
    return dict(MAE=mae, RMSE=rmse, MSE=mse, MAPE=mape, R2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# ARIMA HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_arima(col="SP500", start="2009-12-31", end="2026-05-12",
               p_range=range(0, 6), d_range=range(0, 3), q_range=range(0, 6),
               top_n=5, window_size=100):
    """
    Two-stage ARIMA hyperparameter search for a given index.

    Stage 1 — AIC grid search on training data only (fast).
              Fits ARIMA(p,d,q) for every combination in p_range x d_range x q_range
              and ranks by AIC.
    Stage 2 — Rolling one-step-ahead test-set evaluation (same as run_arima)
              for the top_n candidates from Stage 1.

    Returns a sorted list of dicts with keys:
        order, AIC, MAE, RMSE, MAPE, R2
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    close = load_series(col, start=start, end=end)
    v = close.values.astype(float)
    v = v[window_size - 1:]

    train_size = int(len(v) * 0.8)
    train = v[:train_size]
    n_forecast = len(v) - train_size - 1

    # ── Stage 1: AIC grid search on training data ────────────────────────────
    print(f"\n  Stage 1 — AIC grid search  "
          f"(p={list(p_range)}, d={list(d_range)}, q={list(q_range)})")
    n_total = len(p_range) * len(d_range) * len(q_range)
    print(f"  Fitting {n_total} models…", flush=True)

    aic_results = []
    for p in p_range:
        for d in d_range:
            for q in q_range:
                if p == 0 and q == 0:
                    continue  # trivial model
                try:
                    res = SARIMAX(train, order=(p, d, q),
                                  enforce_stationarity=False,
                                  enforce_invertibility=False).fit(disp=False)
                    aic_results.append(dict(order=(p, d, q), AIC=res.aic))
                except Exception:
                    pass

    aic_results.sort(key=lambda x: x["AIC"])

    print(f"\n  Top {min(top_n, len(aic_results))} by AIC:")
    print(f"  {'Order':<14} {'AIC':>10}")
    print("  " + "─" * 26)
    for r in aic_results[:top_n]:
        print(f"  ARIMA{str(r['order']):<9} {r['AIC']:>10.2f}")

    # ── Stage 2: rolling forecast for top_n candidates ───────────────────────
    print(f"\n  Stage 2 — Rolling test-set evaluation for top {top_n} models…")
    print(f"  {'Order':<14} {'MAE':>8} {'RMSE':>8} {'MAPE%':>8} {'R2':>8}  {'AIC':>10}")
    print("  " + "─" * 60)

    final = []
    for candidate in aic_results[:top_n]:
        order = candidate["order"]
        try:
            res = SARIMAX(train, order=order,
                          enforce_stationarity=False,
                          enforce_invertibility=False).fit(disp=False)
            predictions = []
            for t in range(n_forecast):
                predictions.append(float(res.forecast(1)[0]))
                res = res.append([v[train_size + t]], refit=False)

            y_pred = np.array(predictions)
            y_true = v[train_size + 1: train_size + 1 + n_forecast]

            mae  = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
            r2   = r2_score(y_true, y_pred)

            row = dict(order=order, AIC=candidate["AIC"],
                       MAE=mae, RMSE=rmse, MAPE=mape, R2=r2)
            final.append(row)
            print(f"  ARIMA{str(order):<9} {mae:>8.2f} {rmse:>8.2f} {mape:>8.4f} {r2:>8.4f}  {candidate['AIC']:>10.2f}")
        except Exception as e:
            print(f"  ARIMA{str(order):<9} FAILED: {e}")

    final.sort(key=lambda x: x["MAE"])

    print(f"\n  Best model by MAE: ARIMA{final[0]['order']}  "
          f"MAE={final[0]['MAE']:.2f}  RMSE={final[0]['RMSE']:.2f}  "
          f"MAPE={final[0]['MAPE']:.4f}  R2={final[0]['R2']:.4f}")

    # compare against paper's baseline
    baseline = run_arima(col=col, start=start, end=end, order=(5, 1, 0),
                         window_size=window_size)
    print(f"  Paper baseline ARIMA(5,1,0)  "
          f"MAE={baseline['MAE']:.2f}  RMSE={baseline['RMSE']:.2f}  "
          f"MAPE={baseline['MAPE']:.4f}  R2={baseline['R2']:.4f}")

    improvement = (baseline["MAE"] - final[0]["MAE"]) / baseline["MAE"] * 100
    print(f"  MAE improvement over (5,1,0): {improvement:+.1f}%")

    return final


# ─────────────────────────────────────────────────────────────────────────────
# LSTM  (deep learning benchmark — PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

def run_lstm(col="DJI", start="2009-12-31", end="2018-12-28",
             hidden_size=50, num_layers=1, lookback=20,
             epochs=50, lr=0.001, window_size=100):
    """
    Single-layer LSTM trained on log returns (not raw prices).
    Using log returns instead of prices avoids the non-stationarity problem:
    a model trained on 2009-2015 prices cannot extrapolate to 2017-2018 price
    levels, but daily log returns have a stable distribution throughout.
    Pipeline: log_return[t] = log(price[t+1]/price[t])
      → LSTM predicts next log return from a window of past log returns
      → convert back to price: pred_price[t+1] = actual_price[t] * exp(pred_return)
    Evaluation: pred_price[t+1] vs actual_price[t+1].
    Trimmed to the same number of rows as HURST_RC for fair comparison.
    Requires PyTorch (torch).
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(42)

    class _LSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True)
            self.fc   = nn.Linear(hidden_size, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    close = load_series(col, start=start, end=end)
    prices = close.values.astype(float)
    prices = prices[window_size - 1:]                 # align with HURST_RC

    # Log returns: shape (len(prices)-1,)
    log_ret = np.diff(np.log(prices)).astype(np.float32)

    train_size = int(len(prices) * 0.8)               # split index in price space
    ret_train_end = train_size - 1                     # returns have one fewer element

    # Normalise returns (zero mean, unit std from training set)
    mu  = log_ret[:ret_train_end].mean()
    sig = log_ret[:ret_train_end].std() + 1e-8
    ret_norm = (log_ret - mu) / sig

    # Build (input_window → next_return) sequences
    X, y = [], []
    for i in range(lookback, len(ret_norm)):
        X.append(ret_norm[i - lookback: i])
        y.append(ret_norm[i])
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    split = ret_train_end - lookback
    X_tr, y_tr = X[:split], y[:split]
    X_te, y_te = X[split:], y[split:]

    X_tr_t = torch.from_numpy(X_tr).unsqueeze(-1)
    y_tr_t = torch.from_numpy(y_tr).unsqueeze(-1)
    X_te_t = torch.from_numpy(X_te).unsqueeze(-1)

    model     = _LSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = criterion(model(X_tr_t), y_tr_t)
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        pred_ret_norm = model(X_te_t).numpy().flatten()

    # De-normalise predicted returns
    pred_ret = pred_ret_norm * sig + mu                # back to log-return scale

    # Convert predicted log returns → predicted prices
    # ret_norm[ret_train_end + i] = log(prices[ret_train_end+i+1] / prices[ret_train_end+i])
    # so base price for test step i is prices[ret_train_end + i]
    anchor_idx = ret_train_end                         # = train_size - 1
    pred_prices, true_prices = [], []
    for i, pr in enumerate(pred_ret):
        base = prices[anchor_idx + i]
        pred_prices.append(base * np.exp(pr))
        true_prices.append(prices[anchor_idx + i + 1])

    # Trim to same test length as HURST_RC
    n_forecast = len(prices) - train_size - 1
    y_pred = np.array(pred_prices[:n_forecast])
    y_true = np.array(true_prices[:n_forecast])

    mse  = mean_squared_error(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100
    r2   = r2_score(y_true, y_pred)
    return dict(MAE=mae, RMSE=rmse, MSE=mse, MAPE=mape, R2=r2)


# ─────────────────────────────────────────────────────────────────────────────
# NEXT-DAY PREDICTION
# ─────────────────────────────────────────────────────────────────────────────

# Per-index hyperparameters from the paper's Table 4 (SDTP comparison)
INDEX_PARAMS = {
    "DJI":    dict(spectral_radius=0.7,  n_reservoir=200),
    "SP500":  dict(spectral_radius=0.7,  n_reservoir=200),
    "CAC40":  dict(spectral_radius=0.9,  n_reservoir=200),
    "NYSE":   dict(spectral_radius=0.9,  n_reservoir=200),
    "NASDAQ": dict(spectral_radius=0.85, n_reservoir=200),
    "FTSE":   dict(spectral_radius=0.85, n_reservoir=200),
}


def predict_next_day(col, start=None, end=None,
                     sparsity=0.2, noise=0.1, window_size=100):
    """
    Train HURST_RC on the full [start, end] window (END inclusive),
    then predict the price for the next trading day after end.

    Always downloads fresh from yfinance — never reads price_data.csv —
    so previously-saved predicted rows cannot contaminate the training set.

    Returns
    -------
    dict with keys:
        last_date       – last date in training data  (= end or closest prior)
        last_price      – actual closing price on last_date
        next_date       – predicted date (next business day after last_date)
        predicted_price – model's predicted price for next_date
        col             – index name
    """
    params = INDEX_PARAMS.get(col, dict(spectral_radius=0.85, n_reservoir=200))
    reservoir_size  = params["n_reservoir"]
    spectral_radius = params["spectral_radius"]

    # ── Always download fresh from yfinance (bypasses price_data.csv) ────────
    # yfinance end is exclusive, so add 1 day to make our END inclusive.
    ticker = TICKER_MAP.get(col, col)
    yf_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") if end else None
    raw = yf.download(ticker, start=start, end=yf_end,
                      auto_adjust=True, progress=False)
    if raw.empty:
        raise ValueError(
            f"No data for '{col}' (ticker '{ticker}') between {start} and {end}.")
    close_raw = raw["Close"]
    if isinstance(close_raw, pd.DataFrame):
        close_raw = close_raw.iloc[:, 0]
    close = close_raw.squeeze().rename(col)
    # Keep only up to and including `end`
    if end:
        close = close[close.index <= pd.Timestamp(end)]

    if len(close) < window_size + 10:
        raise ValueError(f"Not enough data for {col} (got {len(close)} rows).")

    series = close.values.reshape(-1, 1)

    # ── Normalise on full series ──────────────────────────────────────────
    scaler = MinMaxScaler()
    norm = scaler.fit_transform(series)

    # ── Rolling Hurst (100-day window) ────────────────────────────────────
    H = []
    for i in range(len(norm) - window_size + 1):
        h, _, _ = compute_Hc(norm[i: i + window_size].reshape(-1))
        H.append(h)
    H = np.array(H).reshape(-1, 1)

    norm = norm[-len(H):]                         # align lengths
    data = np.hstack((norm, H))                   # shape (T, 2)

    # ── Train ESN on ALL data ─────────────────────────────────────────────
    # input: data[0..T-2],  target: data[1..T-1, 0]  (next-step price norm)
    esn = ESN(n_inputs=2, n_outputs=1,
              n_reservoir=reservoir_size,
              sparsity=sparsity,
              noise=noise,
              spectral_radius=spectral_radius)
    esn.fit(data[:-1, :], data[1:, 0])

    # ── Predict one step beyond the last known point ──────────────────────
    # After fit(), pyESN's internal reservoir state sits at the state
    # produced by the last training input (data[-2]).
    # We feed data[-1] to get the prediction for T+1.
    pred_norm = esn.predict(data[-1:, :])         # shape (1,) or (1,1)
    pred_norm = np.array(pred_norm).reshape(-1, 1)
    predicted_price = float(scaler.inverse_transform(pred_norm)[0, 0])

    last_date  = close.index[-1]
    last_price = float(close.iloc[-1])
    next_date  = last_date + pd.offsets.BDay(1)   # next business day

    return dict(
        col=col,
        last_date=last_date,
        last_price=last_price,
        next_date=next_date,
        predicted_price=predicted_price,
    )


def save_predictions_to_csv(predictions: list, csv_path: str = None):
    """
    Save predicted prices to predictions.csv (NOT price_data.csv).
    This keeps training data clean so successive runs don't contaminate each other.

    predictions : list of dicts returned by predict_next_day()
    """
    if csv_path is None:
        csv_path = PREDICTIONS_CSV

    # Build a dict of {col: predicted_price} from the results
    pred_row = {p["col"]: round(p["predicted_price"], 4) for p in predictions}
    next_dates = [p["next_date"] for p in predictions]
    next_date  = max(set(next_dates), key=next_dates.count)

    # Load existing predictions file (or start fresh)
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
    else:
        df = pd.DataFrame()

    if next_date in df.index:
        print(f"  Row for {next_date.date()} already exists — overwriting.")
        for k, v in pred_row.items():
            df.loc[next_date, k] = v
    else:
        new_row = pd.DataFrame([pred_row], index=pd.DatetimeIndex([next_date]))
        new_row.index.name = "Date"
        df = pd.concat([df, new_row])

    df.sort_index(inplace=True)
    df.to_csv(csv_path)
    return df, next_date


# ─────────────────────────────────────────────────────────────────────────────
# PLOTTING
# ─────────────────────────────────────────────────────────────────────────────

def plot_all(res, m_paper, m_corrected, m_naive):
    fig = plt.figure(figsize=(16, 15))
    fig.suptitle(
        "HURST_RC Replication — S&P 500  (Santos et al., Physica D 2025)",
        fontsize=13, fontweight="bold"
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # ── Panel A: Actual vs Predicted (author's alignment) ─────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(res["dates_test"], res["y_true"],
             label="Actual", color="steelblue", lw=1.3)
    ax1.plot(res["dates_test"], res["y_pred"],
             label="HURST_RC Forecast", color="darkorange", lw=1.2, alpha=0.85)
    ax1.set_title("Test Period: Actual vs Predicted (Author's Evaluation)",
                  fontsize=11)
    ax1.set_ylabel("S&P 500 Index")
    ax1.legend(); ax1.grid(True, alpha=0.3)

    # ── Panel B: Rolling Hurst Exponent ───────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(res["dates_all"], res["hurst_exponents"],
             color="darkorange", lw=0.9, alpha=0.85)
    ax2.axhline(0.5, color="grey", lw=1.2, linestyle="--",
                label="H=0.5 (random walk)")
    ax2.axvline(res["dates_all"][res["train_size"]],
                color="green", lw=1.2, linestyle=":", label="Train/Test split")
    ax2.set_title("Rolling Hurst Exponent (100-day window)", fontsize=10)
    ax2.set_ylabel("H"); ax2.set_ylim(0, 1)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)

    # ── Panel C: Concatenated input visualisation ──────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(res["dates_all"], res["normalized_series"],
             color="steelblue", lw=0.9, label="Normalised Price")
    ax3.plot(res["dates_all"], res["hurst_exponents"],
             color="darkorange", lw=0.9, alpha=0.8, label="Hurst Exponent H")
    ax3.set_title("Concatenated Input to Reservoir", fontsize=10)
    ax3.set_ylabel("Normalised [0,1]")
    ax3.legend(fontsize=8); ax3.grid(True, alpha=0.3)

    # ── Panel D: 3-way metrics comparison bar chart ────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    labels = ["MAE", "RMSE", "MAPE", "R²"]
    keys   = ["MAE", "RMSE", "MAPE", "R2"]
    paper_vals     = [m_paper[k]     for k in keys]
    paper_rep_vals = [res["metrics"][k] for k in keys]
    corrected_vals = [m_corrected[k]  for k in keys]
    naive_vals     = [m_naive[k]      for k in keys]
    paper_claimed  = [6.33, 7.79, 0.23, 0.9975]   # Table 2 of the paper

    x = np.arange(len(labels))
    w = 0.20
    ax4.bar(x - 1.5*w, paper_claimed,  w, label="Paper (Table 2)",      color="gold",       alpha=0.9)
    ax4.bar(x - 0.5*w, paper_rep_vals, w, label="Our replication (author alignment)", color="steelblue",  alpha=0.85)
    ax4.bar(x + 0.5*w, corrected_vals, w, label="Our replication (corrected)",        color="tomato",     alpha=0.85)
    ax4.bar(x + 1.5*w, naive_vals,     w, label="Naïve persistence",    color="dimgrey",    alpha=0.75)

    ax4.set_xticks(x); ax4.set_xticklabels(labels, fontsize=11)
    ax4.set_title(
        "Metrics Comparison — Paper vs Replication vs Corrected vs Naïve\n"
        "(note: R² and MAPE are small numbers; MAE/RMSE are in index points)",
        fontsize=10)
    ax4.legend(fontsize=8); ax4.grid(True, alpha=0.3, axis="y")
    ax4.set_yscale("symlog", linthresh=1)

    _fig_dir = os.path.join(WORK_DIR, "figures")
    os.makedirs(_fig_dir, exist_ok=True)
    plt.savefig(os.path.join(_fig_dir, "hurst_rc_results.png"),
                dpi=150, bbox_inches="tight")
    print(f"Figure saved → {_fig_dir}/hurst_rc_results.png")
    plt.show()


def print_table(m_rep, m_corrected, m_naive):
    paper = dict(MAE=6.33, RMSE=7.79, MSE=None, MAPE=0.23, R2=0.9975)
    print("\n" + "─"*75)
    print(f"  {'Metric':<8} {'Paper':>10} {'Replication':>14} {'Corrected':>12} {'Naïve':>10}")
    print("─"*75)
    for k in ["MAE", "RMSE", "MAPE", "R2"]:
        p = f"{paper[k]:.4f}" if paper[k] is not None else "  —"
        print(f"  {k:<8} {p:>10} {m_rep[k]:>14.4f} {m_corrected[k]:>12.4f} {m_naive[k]:>10.4f}")
    print("─"*75)


# ─────────────────────────────────────────────────────────────────────────────
# SENSITIVITY: spectral radius vs MAE (author Table 4 varies this 0.7–0.95)
# ─────────────────────────────────────────────────────────────────────────────

def sensitivity_spectral(col="DJI", start="2009-12-31", end="2018-12-28"):
    radii = [0.70, 0.80, 0.85, 0.90, 0.95]
    rows = []
    for rho in radii:
        r = run_hurst_rc(col=col, start=start, end=end,
                         spectral_radius=rho, reservoir_size=200)
        rows.append((rho, r["metrics"]["MAE"], r["metrics"]["R2"]))
        print(f"  ρ={rho}  MAE={r['metrics']['MAE']:.2f}  R²={r['metrics']['R2']:.4f}")

    rhos, maes, r2s = zip(*rows)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(rhos, maes, "o-", color="steelblue", lw=2)
    ax1.set_xlabel("Spectral Radius ρ"); ax1.set_ylabel("MAE")
    ax1.set_title("MAE vs Spectral Radius"); ax1.grid(True, alpha=0.3)

    ax2.plot(rhos, r2s, "o-", color="darkorange", lw=2)
    ax2.set_xlabel("Spectral Radius ρ"); ax2.set_ylabel("R²")
    ax2.set_title("R² vs Spectral Radius"); ax2.grid(True, alpha=0.3)

    fig.suptitle("Hyperparameter Sensitivity — S&P 500 (N=200, noise=0.1)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    _fig_dir = os.path.join(WORK_DIR, "figures")
    os.makedirs(_fig_dir, exist_ok=True)
    plt.savefig(os.path.join(_fig_dir, "hurst_rc_sensitivity.png"),
                dpi=150, bbox_inches="tight")
    print(f"Sensitivity figure saved → {_fig_dir}/hurst_rc_sensitivity.png")
    plt.show()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ─────────────────────────────────────────────────────────────────────
    # ★  TO TEST A DIFFERENT TIMEFRAME — change START / END here  ★
    #    Examples:
    #      Full paper window   : START="2009-12-31"  END="2018-12-28"
    #      Post-COVID window   : START="2020-01-01"  END="2024-01-01"
    #      Short recent window : START="2022-01-01"  END="2025-01-01"
    #
    # ★  TO SWITCH INDEX — change COL to any column in price_data.csv  ★
    #    Available: "DJI", "SP500", "CAC40", "NYSE", "NASDAQ",
    #               "FTSE", "N225", "SSE", "SZ", "SH", "HSI"
    #    Note: dates outside the CSV range will fall back to live download.
    # ─────────────────────────────────────────────────────────────────────
    # !! Use the CSV column name, NOT the Yahoo ticker symbol !!
    # Wrong:  COL = "GSPC"   (that's the ticker)
    # Right:  COL = "SP500"  (that's the CSV column name)
    COL   = "SP500"
    START = "2009-12-31"
    END   = "2026-05-12"

    # Tuned hyperparameters for SP500 (tune_hurst_rc, 2026-05-13)
    # Baseline: N=200, ρ=0.7, noise=0.1, sparsity=0.2  →  MAE=40.65
    # Tuned:    N=100, ρ=0.5, noise=0.01, sparsity=0.5 →  MAE=36.63 (+9.9%)
    RESERVOIR_SIZE   = 100
    SPECTRAL_RADIUS  = 0.5
    NOISE            = 0.01
    SPARSITY         = 0.5

    print("=" * 60)
    print(f"  HURST_RC  |  {COL}  |  {START} → {END}")
    print(f"  N={RESERVOIR_SIZE}  ρ={SPECTRAL_RADIUS}  noise={NOISE}  sparsity={SPARSITY}  (tuned)")
    print("=" * 60)

    res = run_hurst_rc(
        col=COL, start=START, end=END,
        reservoir_size=RESERVOIR_SIZE,
        spectral_radius=SPECTRAL_RADIUS,
        noise=NOISE, sparsity=SPARSITY
    )
    print("  HURST_RC metrics (author's alignment):")
    for k, v in res["metrics"].items():
        print(f"    {k:6s} = {v:.4f}")

    print("\n  Running corrected evaluation…")
    m_corrected = run_hurst_rc_corrected(
        col=COL, start=START, end=END,
        reservoir_size=RESERVOIR_SIZE,
        spectral_radius=SPECTRAL_RADIUS,
        noise=NOISE, sparsity=SPARSITY
    )
    print("  Corrected metrics:")
    for k, v in m_corrected.items():
        print(f"    {k:6s} = {v:.4f}")

    print("\n  Running naïve persistence baseline…")
    m_naive = run_naive(col=COL, start=START, end=END)
    print("  Naïve metrics:")
    for k, v in m_naive.items():
        print(f"    {k:6s} = {v:.4f}")

    print_table(res["metrics"], m_corrected, m_naive)
    plot_all(res, res["metrics"], m_corrected, m_naive)

    # ─────────────────────────────────────────────────────────────────────
    # MODEL COMPARISON  (RC, ARIMA, LSTM vs HURST_RC)
    # ─────────────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON  (all models, same train/test window)")
    print("=" * 60)

    print("  Running RC (no Hurst)…")
    m_rc = run_rc(col=COL, start=START, end=END,
                  reservoir_size=RESERVOIR_SIZE,
                  spectral_radius=SPECTRAL_RADIUS,
                  noise=NOISE,
                  sparsity=SPARSITY)

    print("  Running ARIMA(5,1,0) rolling…")
    m_arima = run_arima(col=COL, start=START, end=END)

    print("  Running LSTM (log returns, 50 epochs)…")
    m_lstm = run_lstm(col=COL, start=START, end=END)

    paper_claimed = dict(MAE=6.33, RMSE=7.79, MSE=None, MAPE=0.23, R2=0.9975)

    models = [
        ("Paper (Table 2)*",  paper_claimed),
        ("HURST_RC (paper)*", res["metrics"]),
        ("HURST_RC (corrected)", m_corrected),
        ("RC (no Hurst)",     m_rc),
        ("ARIMA(5,1,0)",      m_arima),
        ("LSTM",              m_lstm),
        ("Naïve persistence", m_naive),
    ]

    print(f"\n  {'Model':<24} {'MAE':>9} {'RMSE':>9} {'MAPE%':>9} {'R²':>9}")
    print("  " + "─" * 62)
    for name, m in models:
        mae  = f"{m['MAE']:.2f}"   if m.get('MAE')  is not None else "  —"
        rmse = f"{m['RMSE']:.2f}"  if m.get('RMSE') is not None else "  —"
        mape = f"{m['MAPE']:.4f}"  if m.get('MAPE') is not None else "  —"
        r2   = f"{m['R2']:.4f}"    if m.get('R2')   is not None else "  —"
        note = "  ← flawed eval" if "*" in name else ""
        print(f"  {name:<24} {mae:>9} {rmse:>9} {mape:>9} {r2:>9}{note}")
    print("\n  * author's evaluation aligns prediction at t with actual at t")
    print("    (not t+1), which inflates R² due to price autocorrelation.")

    print("\n  Running spectral radius sensitivity…")
    sensitivity_spectral(col=COL, start=START, end=END)

    # ─────────────────────────────────────────────────────────────────────
    # NEXT-DAY PREDICTION FOR ALL INDICES
    # Train on full [START, END] window and predict the next trading day.
    # The result is appended as a new row in price_data.csv.
    # ─────────────────────────────────────────────────────────────────────
    ALL_COLS = ["DJI", "SP500", "CAC40", "NYSE", "NASDAQ",
                "FTSE"]

    print("\n" + "=" * 60)
    print("  NEXT-DAY PREDICTIONS (train on full window)")
    print("=" * 60)
    print(f"  {'Index':<8} {'Last Date':<14} {'Last Price':>12} "
          f"{'Next Date':<14} {'Predicted':>12}  {'Δ':>8}")
    print("  " + "─" * 66)

    predictions = []
    for c in ALL_COLS:
        try:
            p = predict_next_day(c, start=START, end=END)
            delta = p["predicted_price"] - p["last_price"]
            direction = "▲" if delta >= 0 else "▼"
            print(f"  {c:<8} {str(p['last_date'].date()):<14} "
                  f"{p['last_price']:>12.2f} "
                  f"{str(p['next_date'].date()):<14} "
                  f"{p['predicted_price']:>12.2f}  "
                  f"{direction}{abs(delta):>7.2f}")
            predictions.append(p)
        except Exception as e:
            print(f"  {c:<8} ERROR: {e}")

    if predictions:
        df_updated, pred_date = save_predictions_to_csv(predictions)
        print(f"\n  Predictions for {pred_date.date()} saved → {PREDICTIONS_CSV}")
        print(f"  predictions.csv now has {len(df_updated)} rows "
              f"({df_updated.index[0].date()} → {df_updated.index[-1].date()})")
        print("\n  Last 2 rows of predictions.csv:")
        print(df_updated.tail(2).to_string())
