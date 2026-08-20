"""
data_cleaning.py

Loads raw VIX and SPX price data, cleans it, aligns it on a common trading
calendar, and derives the base series needed for the variance risk premium
(VRP) project:

    - spx_close        : cleaned SPX close price
    - spx_log_return    : daily log return of SPX
    - vix_close         : cleaned VIX close (already an annualized % vol)

Run directly (`python src/data_cleaning.py`) to fetch fresh data via
yfinance and write a cleaned CSV to data/clean_market_data.csv.

Import `load_clean_data()` elsewhere to skip the CLI and get a DataFrame.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_PATH = DATA_DIR / "raw_market_data.csv"
CLEAN_PATH = DATA_DIR / "clean_market_data.csv"

SPX_TICKER = "^GSPC"
VIX_TICKER = "^VIX"


# --------------------------------------------------------------------------- #
# 1. Fetch
# --------------------------------------------------------------------------- #
def fetch_raw_data(start: str = "2005-01-01", end: str | None = None) -> pd.DataFrame:
    """
    Pull daily SPX and VIX close prices via yfinance.

    Requires internet access and the yfinance package
    (`pip install yfinance`) -- intended to be run locally / in VS Code,
    not inside a sandboxed environment without external network access.
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required to fetch data. Install with: pip install yfinance"
        ) from e

    spx = yf.download(SPX_TICKER, start=start, end=end, progress=False)[["Close"]]
    spx.columns = ["spx_close"]

    vix = yf.download(VIX_TICKER, start=start, end=end, progress=False)[["Close"]]
    vix.columns = ["vix_close"]

    raw = spx.join(vix, how="outer")
    raw.index.name = "date"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_PATH)
    print(f"Saved raw data -> {RAW_PATH}  ({len(raw)} rows)")
    return raw


# --------------------------------------------------------------------------- #
# 2. Clean
# --------------------------------------------------------------------------- #
def clean_data(raw: pd.DataFrame, max_ffill_days: int = 3) -> pd.DataFrame:
    """
    Clean the raw SPX/VIX frame:

    1. Ensure a sorted DatetimeIndex.
    2. Drop rows where BOTH series are missing (non-trading days that
       slipped through).
    3. Forward-fill isolated gaps (holidays mismatched between feeds,
       single missing prints) up to `max_ffill_days`; longer gaps are
       left as NaN and then dropped so we never silently interpolate
       across a real data outage.
    4. Drop non-positive or clearly bad prices (data errors).
    5. Compute derived columns: log returns, realized volatility inputs.
    6. Drop any remaining rows with NaNs in the core columns.

    Returns a tidy DataFrame indexed by date with columns:
        spx_close, vix_close, spx_log_return
    """
    df = raw.copy()

    # --- index hygiene -----------------------------------------------------
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]

    # --- drop fully-empty rows ---------------------------------------------
    df = df.dropna(how="all")

    # --- fix bad / non-positive prices -> NaN so they get handled by fill --
    for col in ["spx_close", "vix_close"]:
        if col in df.columns:
            df.loc[df[col] <= 0, col] = np.nan

    # --- limited forward-fill for short gaps --------------------------------
    df[["spx_close", "vix_close"]] = df[["spx_close", "vix_close"]].ffill(
        limit=max_ffill_days
    )

    # --- drop rows that still have missing core data (real gaps/outages) ---
    before = len(df)
    df = df.dropna(subset=["spx_close", "vix_close"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows with unrecoverable gaps (> {max_ffill_days} days).")

    # --- sanity filter: VIX outside a plausible range is a data error ------
    df = df[(df["vix_close"] > 1) & (df["vix_close"] < 200)]

    # --- derived columns ------------------------------------------------
    df["spx_log_return"] = np.log(df["spx_close"] / df["spx_close"].shift(1))

    df = df.dropna(subset=["spx_log_return"])

    return df


# --------------------------------------------------------------------------- #
# 3. Public entry point
# --------------------------------------------------------------------------- #
def load_clean_data(
    start: str = "2005-01-01",
    end: str | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Convenience loader used by the rest of the project.

    - If a cleaned CSV already exists and force_refresh is False, load it.
    - Otherwise fetch raw data, clean it, cache it, and return it.
    """
    if CLEAN_PATH.exists() and not force_refresh:
        df = pd.read_csv(CLEAN_PATH, index_col="date", parse_dates=True)
        return df

    raw = fetch_raw_data(start=start, end=end)
    clean = clean_data(raw)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_PATH)
    print(f"Saved clean data -> {CLEAN_PATH}  ({len(clean)} rows)")
    return clean


if __name__ == "__main__":
    force = "--refresh" in sys.argv
    data = load_clean_data(force_refresh=force)
    print(data.head())
    print(data.tail())
    print(f"\nRows: {len(data)}  |  Date range: {data.index.min().date()} -> {data.index.max().date()}")
