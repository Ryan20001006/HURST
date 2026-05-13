"""
Download all indices used in Santos et al. (2025) and save to CSV.
To change the time window, edit START_DATE and END_DATE below.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf

# ─────────────────────────────────────────────────────────────────────────────
# ★  ADJUST THE TIMEFRAME HERE  ★
#    Format: "YYYY-MM-DD"
#    Author's original window: 2009-12-31 → 2018-12-28
# ─────────────────────────────────────────────────────────────────────────────
START_DATE = "2009-12-31"
END_DATE   = "2026-05-12"

# ─────────────────────────────────────────────────────────────────────────────
# All 11 indices from the paper (DJI listed first as requested)
# ─────────────────────────────────────────────────────────────────────────────
TICKERS = {
    "DJI":    "^DJI",       # Dow Jones Industrial Average
    "SP500":  "^GSPC",      # Standard & Poor's 500
    "CAC40":  "^FCHI",      # CAC 40 (France)
    "NYSE":   "^NYA",       # NYSE Composite
    "NASDAQ": "^IXIC",      # NASDAQ Composite
    "FTSE":   "^FTSE",      # FTSE 100 (UK)      # Hang Seng Index (Hong Kong)
}

OUTPUT_CSV = "/Users/ryan/Desktop/Financial Analysis and Machine Learning/CSV/price_data.csv"


def download_all(tickers: dict, start: str, end: str) -> pd.DataFrame:
    frames = {}
    for name, symbol in tickers.items():
        print(f"  Downloading {name:6s} ({symbol}) …", end=" ", flush=True)
        try:
            raw = yf.download(symbol, start=start, end=end,
                              auto_adjust=True, progress=False)
            if raw.empty:
                print("NO DATA")
                continue
            # flatten MultiIndex columns (yfinance >= 0.2.31)
            close = raw["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
            close = close.squeeze().rename(name)
            frames[name] = close
            print(f"  {len(close):,} rows  "
                  f"({close.index[0].date()} → {close.index[-1].date()})")
        except Exception as e:
            print(f"ERROR: {e}")

    if not frames:
        raise RuntimeError("No data downloaded — check internet connection.")

    df = pd.DataFrame(frames)
    df.index.name = "Date"
    return df


def main():
    print("=" * 60)
    print(f"  Downloading price data  {START_DATE} → {END_DATE}")
    print("=" * 60)

    df = download_all(TICKERS, START_DATE, END_DATE)

    # ── Summary ────────────────────────────────────────────────────────────
    print(f"\n  Combined shape: {df.shape[0]:,} dates × {df.shape[1]} indices")
    print(f"  Missing values per column:")
    for col in df.columns:
        n_missing = df[col].isna().sum()
        pct = n_missing / len(df) * 100
        flag = "  ← has gaps (different trading calendars)" if n_missing > 0 else ""
        print(f"    {col:8s}: {n_missing:4d} NaN  ({pct:.1f}%){flag}")

    # ── Save ──────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV)
    print(f"\n  Saved → {OUTPUT_CSV}")

    # ── Preview (first 3 and last 3 rows) ─────────────────────────────────
    print("\n  First 3 rows:")
    print(df.head(3).to_string())
    print("\n  Last 3 rows:")
    print(df.tail(3).to_string())

    return df


if __name__ == "__main__":
    df = main()
