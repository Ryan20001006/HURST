
"""
Replicates the key figures from Santos et al., Physica D 476 (2025) 134698.

Figures produced:
  figure1_hurst_input.png   — Fig 2/11 style: normalised price + Hurst,
                              shaded train/test regions (one panel per index)
  figure2_predictions.png   — Figs 8-10 style: actual vs predicted with zoom
                              inset, 4 indices per page  (3 pages total)
  figure3_metrics_bar.png   — Fig 7 style: HURST_RC vs ARIMA bar charts for
                              all indices across MAE, RMSE, MAPE, R²
"""

import os, sys, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from hurst import compute_Hc
from pyESN import ESN

from hurst_rc import (load_series, INDEX_PARAMS,
                      run_rc, run_arima, run_lstm,
                      run_naive as _hrc_naive)

np.random.seed(42)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  — edit these to change indices / dates
# ─────────────────────────────────────────────────────────────────────────────
START      = "2009-12-31"   # default start — used for any index not in DATE_RANGES
END        = "2026-05-12"   # default end   — used for any index not in DATE_RANGES

# ★  Per-index date overrides.
#    To change one index, edit its ("start", "end") here.
#    To use the global START/END above, remove the entry or leave it matching.
DATE_RANGES = {
    "DJI":    ("2009-12-31", "2026-05-12"),
    "SP500":  ("2009-12-31", "2026-05-12"),
    "CAC40":  ("2009-12-31", "2026-05-12"),
    "NYSE":   ("2009-12-31", "2026-05-28"),
    "NASDAQ": ("2009-12-31", "2026-05-28"),
    "FTSE":   ("2009-12-31", "2026-05-28"),
    "N225":   ("2009-12-31", "2026-05-28"),
    "SSE":    ("2009-12-31", "2026-05-12"),
    "SZ":     ("2009-12-31", "2026-05-12"),
    "SH":     ("2009-12-31", "2026-05-12"),
    "HSI":    ("2009-12-31", "2026-05-12"),
}

WINDOW     = 100
SPARSITY   = 0.2
NOISE      = 0.1
ALL_COLS   = ["DJI", "SP500", "CAC40", "NYSE",
              "NASDAQ", "FTSE", "N225", "SSE",
              "SZ", "SH", "HSI"]

# ★  ALIGN_DATES = True  → every index is restricted to dates where ALL
#    indices have data (intersection).  The 80/20 split then lands on exactly
#    the same calendar dates for every index.
#    ALIGN_DATES = False → each index uses its own trading calendar (original
#    paper behaviour — different number of rows per index).
ALIGN_DATES = False

# Indices included in the multi-model comparison (Figures 6).
# ARIMA ≈ 2s/index, LSTM ≈ 20s/index — reduce to e.g. ["DJI","SP500"] for speed.
COMPARE_COLS = ALL_COLS

# Set False to skip the slow LSTM in the comparison (much faster for 11 indices)
INCLUDE_LSTM = True

# Pretty display names matching the paper's axis labels
LABEL = {
    "DJI":    "DJI",
    "SP500":  "S&P500",
    "CAC40":  "CAC40",
    "NYSE":   "NYSE",
    "NASDAQ": "NASDAQ",
    "FTSE":   "FTSE",
    "N225":   "N225",
    "SSE":    "SSE",
    "SZ":     "SZ",
    "SH":     "SH",
    "HSI":    "HSI",
}


# ─────────────────────────────────────────────────────────────────────────────
# COMMON DATE INDEX — intersection of all indices' trading days
# ─────────────────────────────────────────────────────────────────────────────

def get_common_dates(cols):
    """
    Return the sorted DatetimeIndex of dates where EVERY index has a price.
    This ensures all series share exactly the same rows (same train/test split
    calendar dates) regardless of individual market holidays.
    """
    print("  Computing common date intersection…", flush=True)
    series_list = {}
    for col in cols:
        s = load_series(col, start=START, end=END)
        series_list[col] = s

    df = pd.DataFrame(series_list)       # outer join — NaN where market closed
    common = df.dropna().index           # keep only rows with data for ALL cols
    print(f"  Common dates: {len(common)} days  "
          f"({common[0].date()} → {common[-1].date()})")
    for col in cols:
        full = series_list[col].index
        dropped = len(full) - len(common)
        print(f"    {col:8s}: {len(full)} own trading days → "
              f"{len(common)} common days  ({dropped} dropped)")
    return common


# ─────────────────────────────────────────────────────────────────────────────
# HELPER: run the full HURST_RC pipeline for one index
# ─────────────────────────────────────────────────────────────────────────────

def _build(col, common_dates=None):
    params = INDEX_PARAMS.get(col, dict(spectral_radius=0.85, n_reservoir=200))
    s, e   = DATE_RANGES.get(col, (START, END))
    close  = load_series(col, start=s, end=e)

    # restrict to common dates if alignment is enabled
    if common_dates is not None:
        close = close.reindex(common_dates).dropna()

    series = close.values.reshape(-1, 1)

    scaler = MinMaxScaler()
    norm   = scaler.fit_transform(series)

    H = []
    for i in range(len(norm) - WINDOW + 1):
        h, _, _ = compute_Hc(norm[i: i + WINDOW].reshape(-1))
        H.append(h)
    H    = np.array(H).reshape(-1, 1)
    norm = norm[-len(H):]
    data = np.hstack((norm, H))

    dates     = close.index[-len(data):]
    split     = int(len(data) * 0.8)
    train     = data[:split]
    test      = data[split:]

    esn = ESN(n_inputs=2, n_outputs=1,
              n_reservoir=params["n_reservoir"],
              sparsity=SPARSITY, noise=NOISE,
              spectral_radius=params["spectral_radius"])
    esn.fit(train[:-1], train[1:, 0])

    pred_norm = esn.predict(test[:-1])
    pred_raw  = scaler.inverse_transform(pred_norm.reshape(-1, 1))[:, 0]
    # author's alignment: compare against test[:-1, 0]
    true_raw  = scaler.inverse_transform(test[:-1, :1])[:, 0]
    # actual NEXT-step price (t+1) for correct hit-rate evaluation
    actual_next = scaler.inverse_transform(test[1:, :1])[:, 0]

    # metrics
    mae  = mean_absolute_error(true_raw, pred_raw)
    mse  = mean_squared_error(true_raw, pred_raw)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((true_raw - pred_raw) / (true_raw + 1e-9))) * 100
    r2   = r2_score(true_raw, pred_raw)

    # ── Hit rate ────────────────────────────────────────────────────────────
    # predicted direction: pred_raw[i] vs true_raw[i]  (current price)
    # actual direction:    actual_next[i] vs true_raw[i]
    pred_direction   = (pred_raw   > true_raw).astype(int)  # 1=up, 0=down
    actual_direction = (actual_next > true_raw).astype(int)
    hits             = (pred_direction == actual_direction).astype(int)
    hit_rate         = hits.mean()

    # naïve hit rate: always predict same direction as previous day
    prev_direction  = (true_raw[1:] > true_raw[:-1]).astype(int)
    naive_direction = (true_raw[1:] > true_raw[:-1]).astype(int)   # persist last move
    actual_dir_naive = (actual_next[1:] > true_raw[1:]).astype(int)
    naive_hit_rate  = (prev_direction == actual_dir_naive).mean()

    # random baseline is always 50 %
    dates_test = dates[split: split + len(true_raw)]

    return dict(
        col=col, close=close, norm=norm, H=H, data=data,
        dates=dates, split=split,
        true_raw=true_raw, pred_raw=pred_raw, actual_next=actual_next,
        dates_test=dates_test,
        hits=hits, hit_rate=hit_rate, naive_hit_rate=naive_hit_rate,
        pred_direction=pred_direction, actual_direction=actual_direction,
        metrics=dict(MAE=mae, MSE=mse, RMSE=rmse, MAPE=mape, R2=r2),
    )


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 1 — Normalised price + Hurst exponent  (paper Fig 2 / Fig 11 style)
# One subplot per index, shaded train/test intervals, same as paper's Fig 11
# ─────────────────────────────────────────────────────────────────────────────

def figure1_hurst_input(results):
    n   = len(results)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(18, nrows * 3.5))
    axes = axes.flatten()

    for i, r in enumerate(results):
        ax    = axes[i]
        dates = r["dates"]
        split = r["split"]
        norm  = r["norm"][:, 0]
        H     = r["H"][:, 0]

        ax.plot(dates, norm, color="steelblue",  lw=1.0, label="Time series of asset prices")
        ax.plot(dates, H,    color="darkorange", lw=0.9, label="Hurst exponents")

        # shaded train / test intervals (paper Fig 2 uses green/red shading)
        ax.axvspan(dates[0],       dates[split - 1], alpha=0.07,
                   color="green",  label="Train interval")
        ax.axvspan(dates[split],   dates[-1],        alpha=0.10,
                   color="salmon", label="Test interval")
        ax.axhline(0.5, color="grey", lw=0.7, linestyle="--")

        ax.set_title(f"Combined Time Series with Hurst Exponents\n({LABEL[r['col']]})",
                     fontsize=9)
        ax.set_xlabel("Date", fontsize=8)
        ax.set_ylabel("Values", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=7, loc="upper left")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Normalised Price + Hurst Exponent — All Indices\n"
                 f"({START} → {END}, 100-day rolling window)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figure1_hurst_input.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Actual vs Predicted with zoom inset  (paper Figs 8–10 style)
# 4 panels per page, blue rectangle + inset zoom on test indices 20–50
# ─────────────────────────────────────────────────────────────────────────────

def _prediction_panel(ax, fig, r, zoom_start=20, zoom_end=50):
    """Draw one actual-vs-predicted panel with a zoom inset, exactly as in the paper."""
    dates_test = r["dates_test"]
    true_raw   = r["true_raw"]
    pred_raw   = r["pred_raw"]
    col        = r["col"]

    # ── main plot ──────────────────────────────────────────────────────────
    ax.plot(dates_test, true_raw, color="steelblue",  lw=1.2, label="Real Data")
    ax.plot(dates_test, pred_raw, color="darkorange", lw=1.1,
            alpha=0.85, label="Forecast")
    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel(LABEL[col], fontsize=10, fontweight="bold")
    ax.set_title(f"{LABEL[col]} Stock Price Prediction (Reservoir Computing)",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=7)

    # rotate x-axis labels
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

    # ── zoom inset ─────────────────────────────────────────────────────────
    ze = min(zoom_end, len(true_raw))
    zs = min(zoom_start, ze - 5)
    if ze - zs < 3:
        return

    # inset axes position (paper places it in upper-left area)
    inset = ax.inset_axes([0.03, 0.55, 0.28, 0.38])
    inset.plot(dates_test[zs:ze], true_raw[zs:ze],
               color="steelblue",  lw=1.0)
    inset.plot(dates_test[zs:ze], pred_raw[zs:ze],
               color="darkorange", lw=0.9, alpha=0.9)
    inset.set_title(f"Zoom from index {zs} to {ze}", fontsize=6)
    inset.grid(True, alpha=0.3)
    inset.tick_params(labelsize=5)
    for tick in inset.get_xticklabels():
        tick.set_rotation(45)

    # blue rectangle on main chart showing zoom region
    y_lo = min(true_raw[zs:ze].min(), pred_raw[zs:ze].min())
    y_hi = max(true_raw[zs:ze].max(), pred_raw[zs:ze].max())
    rect = mpatches.Rectangle(
        (dates_test[zs], y_lo),
        width=(dates_test[ze - 1] - dates_test[zs]),
        height=(y_hi - y_lo),
        edgecolor="blue", facecolor="none", lw=1.2
    )
    ax.add_patch(rect)


def figure2_predictions(results):
    groups = [results[i: i + 4] for i in range(0, len(results), 4)]
    for page, group in enumerate(groups):
        fig, axes = plt.subplots(len(group), 1,
                                 figsize=(12, 5 * len(group)))
        if len(group) == 1:
            axes = [axes]
        for ax, r in zip(axes, group):
            _prediction_panel(ax, fig, r)
        fig.suptitle("Stock Price Prediction — Reservoir Computing (HURST_RC)\n"
                     f"({START} → {END})",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        path = os.path.join(OUT_DIR, f"figure2_predictions_page{page+1}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Bar chart comparison across all indices  (paper Fig 7 style)
# HURST_RC (blue) vs ARIMA naïve persistence (red) for MAE, RMSE, MAPE, R²
# ─────────────────────────────────────────────────────────────────────────────

def _naive_metrics(col, common_dates=None):
    s, e  = DATE_RANGES.get(col, (START, END))
    close = load_series(col, start=s, end=e)
    if common_dates is not None:
        close = close.reindex(common_dates).dropna()
    v = close.values
    split = int(len(v) * 0.8)
    y_true = v[split + 1:]
    y_pred = v[split: -1]
    return dict(
        MAE  = mean_absolute_error(y_true, y_pred),
        RMSE = np.sqrt(mean_squared_error(y_true, y_pred)),
        MAPE = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100,
        R2   = r2_score(y_true, y_pred),
    )


def figure3_metrics_bar(results, naive_metrics):
    metrics_keys = ["MAE", "RMSE", "MAPE", "R2"]
    titles = {
        "MAE":  "MAE",
        "RMSE": "RMSE",
        "MAPE": "MAPE (%)",
        "R2":   "R²",
    }
    cols  = [r["col"] for r in results]
    x     = np.arange(len(cols))
    w     = 0.38
    xlabels = [LABEL[c] for c in cols]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, key in zip(axes, metrics_keys):
        hrc_vals   = [r["metrics"][key] for r in results]
        naive_vals = [naive_metrics[c][key] for c in cols]

        bars1 = ax.bar(x - w / 2, hrc_vals,   w,
                       label="HURST_RC",        color="steelblue", alpha=0.88)
        bars2 = ax.bar(x + w / 2, naive_vals,  w,
                       label="Naïve persistence", color="tomato",  alpha=0.88)

        # value labels on top of each bar
        for bar in bars1:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)
        for bar in bars2:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_title(titles[key], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(titles[key], fontsize=9)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle("HURST_RC vs Naïve Persistence — All Indices\n"
                 f"({START} → {END})",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = os.path.join(OUT_DIR, "figure3_metrics_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Metrics summary table as a heatmap  (additional diagnostic)
# ─────────────────────────────────────────────────────────────────────────────

def figure4_metrics_table(results):
    cols  = [LABEL[r["col"]] for r in results]
    keys  = ["MAE", "RMSE", "MAPE", "R2"]
    data  = np.array([[r["metrics"][k] for k in keys] for r in results])

    fig, ax = plt.subplots(figsize=(10, 6))
    # normalise each column to [0,1] for colour mapping
    normed = np.zeros_like(data)
    for j in range(data.shape[1]):
        col_min, col_max = data[:, j].min(), data[:, j].max()
        if col_max > col_min:
            normed[:, j] = (data[:, j] - col_min) / (col_max - col_min)
        else:
            normed[:, j] = 0.5
    # for R², higher = better, so invert colour
    normed[:, 3] = 1 - normed[:, 3]

    im = ax.imshow(normed.T, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)

    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=10)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(["MAE", "RMSE", "MAPE (%)", "R²"], fontsize=10)

    # annotate each cell with the raw value
    for i in range(len(cols)):
        for j, key in enumerate(keys):
            val = data[i, j]
            fmt = f"{val:.4f}" if key == "R2" else f"{val:.2f}"
            ax.text(i, j, fmt, ha="center", va="center",
                    fontsize=8, color="black")

    ax.set_title("HURST_RC Metrics — All Indices\n"
                 "(green = better, red = worse within each metric)",
                 fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.04,
                 label="Relative performance (0=best, 1=worst)")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "figure4_metrics_table.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Hit rate analysis  (directional accuracy)
# Panel A: bar chart of hit rate per index vs naïve and random baselines
# Panel B: rolling 30-day hit rate over the test period for each index
# Panel C: cumulative hit rate over time per index
# Panel D: confusion-style up/down breakdown per index
# ─────────────────────────────────────────────────────────────────────────────

def figure5_hit_rate(results):
    cols      = [r["col"]           for r in results]
    xlabels   = [LABEL[c]           for c in cols]
    hit_rates = [r["hit_rate"] * 100        for r in results]
    naive_hrs = [r["naive_hit_rate"] * 100  for r in results]
    random_hr = 50.0
    ROLL      = 30    # rolling window in trading days

    fig = plt.figure(figsize=(18, 16))
    fig.suptitle("HURST_RC Directional Hit Rate Analysis\n"
                 f"(test period, 80/20 split, {START} → {END})",
                 fontsize=14, fontweight="bold")

    gs = fig.add_gridspec(3, 2, hspace=0.50, wspace=0.35)

    # ── Panel A: overall hit rate bar chart ───────────────────────────────
    ax_a = fig.add_subplot(gs[0, :])
    x = np.arange(len(cols))
    w = 0.28
    ax_a.bar(x - w,     hit_rates, w, label="HURST_RC",
             color="steelblue", alpha=0.88)
    ax_a.bar(x,         naive_hrs, w, label="Naïve (persist last direction)",
             color="darkorange", alpha=0.85)
    ax_a.bar(x + w, [random_hr] * len(cols), w,
             label="Random baseline (50%)", color="grey", alpha=0.55)

    ax_a.axhline(50, color="red", lw=1.2, linestyle="--", label="50% random")
    ax_a.axhline(55, color="green", lw=1.0, linestyle=":",
                 label="55% practical threshold")

    for i, v in enumerate(hit_rates):
        ax_a.text(x[i] - w, v + 0.4, f"{v:.1f}%",
                  ha="center", va="bottom", fontsize=8, fontweight="bold",
                  color="steelblue")

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(xlabels, fontsize=10)
    ax_a.set_ylabel("Hit Rate (%)", fontsize=10)
    ax_a.set_ylim(30, 80)
    ax_a.set_title("Overall Directional Hit Rate — HURST_RC vs Baselines",
                   fontsize=11)
    ax_a.legend(fontsize=8, loc="upper right")
    ax_a.grid(True, alpha=0.3, axis="y")

    # ── Panel B: rolling hit rate for every index ─────────────────────────
    ax_b = fig.add_subplot(gs[1, :])
    cmap = plt.get_cmap("tab10")
    for i, r in enumerate(results):
        roll = pd.Series(r["hits"]).rolling(ROLL).mean() * 100
        ax_b.plot(r["dates_test"][ROLL - 1:], roll.dropna(),
                  lw=1.1, alpha=0.85, color=cmap(i % 10),
                  label=LABEL[r["col"]])
    ax_b.axhline(50, color="red",   lw=1.2, linestyle="--", label="50% random")
    ax_b.axhline(55, color="green", lw=1.0, linestyle=":",  label="55% threshold")
    ax_b.set_ylabel(f"Rolling {ROLL}-day Hit Rate (%)", fontsize=10)
    ax_b.set_title(f"Rolling {ROLL}-day Directional Hit Rate Over Test Period",
                   fontsize=11)
    ax_b.legend(fontsize=7, ncol=4, loc="lower right")
    ax_b.grid(True, alpha=0.3)
    ax_b.set_ylim(20, 85)

    # ── Panel C: cumulative hit rate per index ────────────────────────────
    ax_c = fig.add_subplot(gs[2, 0])
    for i, r in enumerate(results):
        cum = np.cumsum(r["hits"]) / (np.arange(len(r["hits"])) + 1) * 100
        ax_c.plot(r["dates_test"], cum,
                  lw=1.0, alpha=0.85, color=cmap(i % 10),
                  label=LABEL[r["col"]])
    ax_c.axhline(50, color="red",   lw=1.2, linestyle="--")
    ax_c.axhline(55, color="green", lw=1.0, linestyle=":")
    ax_c.set_ylabel("Cumulative Hit Rate (%)", fontsize=10)
    ax_c.set_title("Cumulative Hit Rate Over Test Period", fontsize=11)
    ax_c.legend(fontsize=7, ncol=2)
    ax_c.grid(True, alpha=0.3)
    ax_c.set_ylim(35, 70)

    # ── Panel D: up / down breakdown per index ────────────────────────────
    ax_d = fig.add_subplot(gs[2, 1])

    # for each index: how many actual-up days did the model call correctly?
    up_hit, dn_hit = [], []
    for r in results:
        up_mask = r["actual_direction"] == 1   # days that actually went up
        dn_mask = r["actual_direction"] == 0   # days that actually went down
        up_hit.append(r["hits"][up_mask].mean() * 100 if up_mask.sum() else 0)
        dn_hit.append(r["hits"][dn_mask].mean() * 100 if dn_mask.sum() else 0)

    x2 = np.arange(len(cols))
    ax_d.bar(x2 - 0.2, up_hit, 0.38, label="Hit rate on UP days",
             color="mediumseagreen", alpha=0.88)
    ax_d.bar(x2 + 0.2, dn_hit, 0.38, label="Hit rate on DOWN days",
             color="tomato", alpha=0.88)
    ax_d.axhline(50, color="grey", lw=1.0, linestyle="--")
    ax_d.set_xticks(x2)
    ax_d.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    ax_d.set_ylabel("Hit Rate (%)", fontsize=10)
    ax_d.set_title("Hit Rate Breakdown: UP days vs DOWN days", fontsize=11)
    ax_d.legend(fontsize=9)
    ax_d.grid(True, alpha=0.3, axis="y")
    ax_d.set_ylim(30, 80)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(OUT_DIR, "figure5_hit_rate.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")

    # ── print summary table ───────────────────────────────────────────────
    print(f"\n  {'Index':<8} {'Hit Rate':>10} {'Naïve':>10} "
          f"{'vs Random':>12} {'Up-day HR':>12} {'Dn-day HR':>12}")
    print("  " + "─" * 65)
    for i, r in enumerate(results):
        diff = r["hit_rate"] * 100 - 50
        sign = "+" if diff >= 0 else ""
        print(f"  {r['col']:<8} {r['hit_rate']*100:>9.1f}%"
              f" {r['naive_hit_rate']*100:>9.1f}%"
              f" {sign}{diff:>10.1f}pp"
              f" {up_hit[i]:>11.1f}%"
              f" {dn_hit[i]:>11.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Multi-model metrics comparison  (extended paper Fig 7)
# HURST_RC / RC (no Hurst) / ARIMA(5,1,0) / LSTM / Naïve
# 4 subplots: MAE, MSE, MAPE (%), R²
# ─────────────────────────────────────────────────────────────────────────────

def _build_comparison(col):
    """Run RC, ARIMA, (optional LSTM), and Naïve for one index."""
    s, e   = DATE_RANGES.get(col, (START, END))
    params = INDEX_PARAMS.get(col, dict(spectral_radius=0.85, n_reservoir=200))
    out = {}
    out["RC"]    = run_rc(col=col, start=s, end=e,
                          reservoir_size=params["n_reservoir"],
                          spectral_radius=params["spectral_radius"])
    out["ARIMA"] = run_arima(col=col, start=s, end=e)
    out["Naive"] = _hrc_naive(col=col, start=s, end=e)
    if INCLUDE_LSTM:
        out["LSTM"] = run_lstm(col=col, start=s, end=e)
    return out


def figure6_model_comparison_bar(results, comp):
    """
    Extended paper Fig 7 — HURST_RC vs RC vs ARIMA vs LSTM vs Naïve
    for all indices across MAE, MSE, MAPE, R².
    """
    valid = [r for r in results if r["col"] in comp]
    if not valid:
        print("  figure6: no comparison data — skipping.")
        return

    cols    = [r["col"] for r in valid]
    xlabels = [LABEL[c] for c in cols]
    x       = np.arange(len(cols))

    # Only include a model if every index has it (avoids KeyError on partial failures)
    model_order = ["HURST_RC"] + [
        m for m in ["RC", "ARIMA", "LSTM", "Naive"]
        if all(m in comp.get(c, {}) for c in cols)
    ]
    model_colors = {
        "HURST_RC": "steelblue",
        "RC":       "mediumseagreen",
        "ARIMA":    "tomato",
        "LSTM":     "mediumpurple",
        "Naive":    "dimgrey",
    }
    model_labels = {
        "HURST_RC": "HURST_RC",
        "RC":       "RC (no Hurst)",
        "ARIMA":    "ARIMA(5,1,0)",
        "LSTM":     "LSTM",
        "Naive":    "Naïve",
    }

    n       = len(model_order)
    w       = 0.8 / n
    offsets = np.linspace(-(0.8 - w) / 2, (0.8 - w) / 2, n)

    metrics_keys = ["MAE", "MSE", "MAPE", "R2"]
    titles       = {"MAE": "MAE", "MSE": "MSE", "MAPE": "MAPE (%)", "R2": "R²"}

    fig, axes = plt.subplots(2, 2, figsize=(max(16, len(cols) * 1.4 + 4), 12))
    axes = axes.flatten()

    for ax, key in zip(axes, metrics_keys):
        for m_name, offset in zip(model_order, offsets):
            if m_name == "HURST_RC":
                vals = [r["metrics"][key] for r in valid]
            else:
                vals = [comp[c][m_name][key] for c in cols]

            bars = ax.bar(x + offset, vals, w,
                          label=model_labels[m_name],
                          color=model_colors[m_name], alpha=0.88)

            for bar in bars:
                h = bar.get_height()
                fmt = (f"{h:.4f}" if key == "R2"
                       else f"{h:.0f}" if h >= 100
                       else f"{h:.2f}")
                ax.text(bar.get_x() + bar.get_width() / 2, h,
                        fmt, ha="center", va="bottom", fontsize=5,
                        rotation=90 if h >= 1000 else 0)

        ax.set_title(titles[key], fontsize=12, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel(titles[key], fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

    model_str = " / ".join(model_labels[m] for m in model_order)
    fig.suptitle(f"Model Comparison — {model_str}\n"
                 f"All Indices  ({START} → {END})",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    path = os.path.join(OUT_DIR, "figure6_model_comparison_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 55)
    print("  Building HURST_RC results for all indices…")
    print(f"  Date range : {START} → {END}")
    print(f"  Align dates: {ALIGN_DATES}")
    print("=" * 55)

    common_dates = get_common_dates(ALL_COLS) if ALIGN_DATES else None

    if common_dates is not None:
        split_idx = int(len(common_dates) * 0.8)
        print(f"\n  Shared train period: {common_dates[0].date()} → "
              f"{common_dates[split_idx - 1].date()}")
        print(f"  Shared test  period: {common_dates[split_idx].date()} → "
              f"{common_dates[-1].date()}\n")

    results = []
    for col in ALL_COLS:
        print(f"  {col:8s}", end=" … ", flush=True)
        try:
            r = _build(col, common_dates=common_dates)
            m = r["metrics"]
            n_test = len(r["true_raw"])
            print(f"MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  "
                  f"MAPE={m['MAPE']:.3f}%  R²={m['R2']:.4f}  "
                  f"test={r['dates_test'][0].date()}→{r['dates_test'][-1].date()} "
                  f"({n_test} days)")
            results.append(r)
        except Exception as e:
            print(f"SKIP ({e})")

    print("\n  Computing naïve baselines…")
    naive = {}
    for col in [r["col"] for r in results]:
        naive[col] = _naive_metrics(col, common_dates=common_dates)

    print("\n  Drawing figures…")
    figure1_hurst_input(results)
    figure2_predictions(results)
    figure3_metrics_bar(results, naive)
    figure4_metrics_table(results)
    figure5_hit_rate(results)

    print("\n" + "=" * 55)
    print("  Multi-model comparison (Figure 6)…")
    print(f"  Indices : {COMPARE_COLS}")
    print(f"  LSTM    : {INCLUDE_LSTM}")
    print("=" * 55)
    comp = {}
    for col in COMPARE_COLS:
        if col not in [r["col"] for r in results]:
            continue
        print(f"  {col:8s}", end=" … ", flush=True)
        try:
            comp[col] = _build_comparison(col)
            print(f"done  ({', '.join(comp[col])})")
        except Exception as e:
            print(f"SKIP ({e})")

    if comp:
        figure6_model_comparison_bar(results, comp)

    print(f"\n  All figures saved to → {OUT_DIR}/")
