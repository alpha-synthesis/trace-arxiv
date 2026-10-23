# -*- coding: utf-8 -*-
"""
Data Download & Preparation for the Causal SDE Generator.

Downloads real market data from Yahoo Finance and prepares it
for training the factor extraction + FIN pipeline.

All configuration is driven by the YAML config file — no CLI.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ── In-memory + disk cache ──────────────────────────────────────────
_real_data_cache: Dict[str, pd.DataFrame] = {}
_disk_cache_dir: Optional[str] = None


def _set_cache_dir(cache_dir: Optional[str]) -> None:
    """Set disk cache directory for downloaded data."""
    global _disk_cache_dir
    _disk_cache_dir = cache_dir
    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)


def _disk_cache_path(cache_key: str) -> Optional[Path]:
    """Return parquet path for a cache key, or None if disk cache is off."""
    if _disk_cache_dir is None:
        return None
    safe_name = hashlib.md5(cache_key.encode()).hexdigest()
    return Path(_disk_cache_dir) / f"{safe_name}.parquet"


def _download_yahoo(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "cache/causal_sde",
) -> pd.DataFrame:
    """
    Download data from Yahoo Finance with disk + memory cache.

    Returns
    -------
    pd.DataFrame
        Columns: ``date, tic, open, high, low, close, volume``.
    """
    import yfinance as yf  # lightweight; no heavy framework needed

    _set_cache_dir(cache_dir)
    cache_key = f"{start_date}|{end_date}|{','.join(sorted(tickers))}"

    # 1. In-memory cache
    if cache_key in _real_data_cache:
        print(f"  > Using cached real data {start_date}→{end_date}")
        return _real_data_cache[cache_key].copy()

    # 2. Disk cache
    disk_path = _disk_cache_path(cache_key)
    if disk_path is not None and disk_path.exists():
        print(f"  > Loading real data from disk cache {start_date}→{end_date}")
        df = pd.read_parquet(disk_path)
        _real_data_cache[cache_key] = df.copy()
        return df

    # 3. Download from Yahoo Finance
    frames = []
    for tic in tickers:
        raw = yf.download(tic, start=start_date, end=end_date, progress=False)
        if raw.empty:
            print(f"  ⚠ No data for {tic}")
            continue
        # Handle MultiIndex columns from yfinance
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.droplevel(1)
        tmp = raw.reset_index().rename(columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adjclose",
            "Volume": "volume",
        })
        tmp["tic"] = tic
        frames.append(tmp)

    if not frames:
        raise RuntimeError(f"No data downloaded for tickers: {tickers}")

    df = pd.concat(frames, ignore_index=True)

    # Persist to disk cache
    if disk_path is not None:
        disk_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(disk_path, index=False)
        print(f"  > Saved to disk cache: {disk_path}")

    _real_data_cache[cache_key] = df.copy()
    return df


# ══════════════════════════════════════════════════════════════════════
# Download
# ══════════════════════════════════════════════════════════════════════


def download_training_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "cache/causal_sde",
) -> pd.DataFrame:
    """
    Download real market data from Yahoo Finance.

    Parameters
    ----------
    tickers : list[str]
        Stock ticker symbols.
    start_date, end_date : str
        ISO date strings (e.g., ``"2004-01-01"``).
    cache_dir : str
        Directory for caching downloaded data.

    Returns
    -------
    pd.DataFrame
        Columns: ``date, tic, open, high, low, close, volume``.
    """
    print(f"\n  Downloading real data: {start_date} → {end_date}")
    print(f"  Tickers: {tickers}")

    df = _download_yahoo(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        cache_dir=cache_dir,
    )

    # Normalise columns
    if "timestamp" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"timestamp": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    needed = ["date", "tic", "open", "high", "low", "close", "volume"]
    available = [c for c in needed if c in df.columns]
    df = df[available].copy()

    print(f"  ✓ Downloaded {len(df)} rows, "
          f"{df['tic'].nunique()} tickers, "
          f"{df['date'].nunique()} trading days")

    return df


def download_test_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_dir: str = "cache/causal_sde",
) -> pd.DataFrame:
    """Download out-of-sample test data (same interface as training)."""
    print(f"\n  Downloading test data: {start_date} → {end_date}")
    return download_training_data(tickers, start_date, end_date, cache_dir)


# ══════════════════════════════════════════════════════════════════════
# Per-ticker splitting (generator trains one ticker at a time)
# ══════════════════════════════════════════════════════════════════════


def split_by_ticker(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Split a multi-ticker DataFrame into per-ticker DataFrames.

    Returns
    -------
    dict
        ``{ticker: DataFrame}`` with each frame sorted by date.
    """
    result = {}
    for tic in sorted(df["tic"].unique()):
        tic_df = (
            df[df["tic"] == tic]
            .sort_values("date")
            .reset_index(drop=True)
        )
        result[tic] = tic_df
    return result


# ══════════════════════════════════════════════════════════════════════
# Save / Load helpers
# ══════════════════════════════════════════════════════════════════════


def save_training_data(
    df: pd.DataFrame,
    output_dir: str,
    filename: str = "training_data.csv",
) -> str:
    """Save training data to CSV. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    df.to_csv(path, index=False)
    print(f"  ✓ Training data saved to {path}")
    return path


def load_training_data(path: str) -> pd.DataFrame:
    """Load previously saved training data."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    return df


# ══════════════════════════════════════════════════════════════════════
# Data quality checks
# ══════════════════════════════════════════════════════════════════════


def validate_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Quick data quality checks.

    Returns a dict with stats and any warnings.
    """
    report = {
        "n_rows": len(df),
        "n_tickers": df["tic"].nunique(),
        "n_dates": df["date"].nunique(),
        "date_range": (df["date"].min(), df["date"].max()),
        "tickers": sorted(df["tic"].unique().tolist()),
        "warnings": [],
    }

    # Missing values
    missing = df[["open", "high", "low", "close", "volume"]].isnull().sum()
    if missing.sum() > 0:
        report["warnings"].append(f"Missing values: {missing.to_dict()}")

    # Zero prices
    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            n_zero = (df[col] == 0).sum()
            if n_zero > 0:
                report["warnings"].append(f"{n_zero} zero values in {col}")

    # OHLC consistency
    if all(c in df.columns for c in ["open", "high", "low", "close"]):
        bad_high = (df["high"] < df[["open", "close"]].max(axis=1)).sum()
        bad_low = (df["low"] > df[["open", "close"]].min(axis=1)).sum()
        if bad_high > 0:
            report["warnings"].append(f"{bad_high} rows with high < max(open, close)")
        if bad_low > 0:
            report["warnings"].append(f"{bad_low} rows with low > min(open, close)")

    # Print summary
    print(f"\n  Data Quality Report:")
    print(f"    Rows:     {report['n_rows']}")
    print(f"    Tickers:  {report['n_tickers']} — {report['tickers']}")
    print(f"    Dates:    {report['n_dates']} ({report['date_range'][0]} → {report['date_range'][1]})")
    if report["warnings"]:
        for w in report["warnings"]:
            print(f"    ⚠ {w}")
    else:
        print(f"    ✓ No issues found")

    return report
