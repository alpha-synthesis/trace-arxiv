#!/usr/bin/env python3
"""Restore each window's return in the released label files from downloaded prices.

The released analyses.json files omit ``price_change_pct``, the window's return
derived from Yahoo Finance prices, which the generator and the B1 scripts read.
This script recomputes it with the generator's own rule, the window's last close
over its first open minus one, in percent (sde_causal_generator/factor_extractor.py),
and writes it back. For the paper's runs the recomputed values matched the
original ones to within 1e-4 percentage points when computed from the prices
saved at run time. Revised Yahoo data can flip the sign of a window whose return
is almost zero, so for the few windows with |return| < 0.01% the value recorded
at run time (tools/near_zero_window_returns.json) is used instead.

Run tools/download_prices.py first.

Usage:
  python tools/attach_window_returns.py [--prices results/r2_datasign/training_data.csv]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]


def window_return(df: pd.DataFrame, start: str, end: str) -> float:
    w = df[(df["date"] >= pd.Timestamp(start)) & (df["date"] <= pd.Timestamp(end))].sort_values("date")
    if w.empty:
        return 0.0
    first_open, last_close = float(w.iloc[0]["open"]), float(w.iloc[-1]["close"])
    return (last_close - first_open) / first_open * 100.0


NEAR_ZERO = json.loads((REPO / "tools" / "near_zero_window_returns.json").read_text())["runs"]


def recorded(run: str) -> dict:
    rows = NEAR_ZERO.get(run, NEAR_ZERO["r2_datasign"])
    return {(x["ticker"], x["start_date"], x["end_date"]): x["price_change_pct"] for x in rows}


def attach(analyses: Path, prices: pd.DataFrame, near: dict) -> tuple[int, int]:
    ticker = analyses.parent.name
    df = prices[prices["tic"] == ticker]
    windows = json.loads(analyses.read_text())
    changed = 0
    for w in windows:
        key = (ticker, w["start_date"], w["end_date"])
        r = near[key] if key in near else window_return(df, w["start_date"], w["end_date"])
        if w.get("price_change_pct") != r:
            w["price_change_pct"] = r
            changed += 1
    analyses.write_text(json.dumps(windows, indent=2))
    return len(windows), changed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--prices", default=str(REPO / "results" / "r2_datasign" / "training_data.csv"),
                    help="price file for label sets outside a run folder (e.g. the cross-model labels)")
    a = ap.parse_args()
    cache: dict[Path, pd.DataFrame] = {}

    def prices_for(path: Path) -> pd.DataFrame:
        run_prices = path.parent.parent / "training_data.csv"
        src = run_prices if run_prices.exists() else Path(a.prices)
        if not src.exists():
            sys.exit(f"Missing {src}: run tools/download_prices.py first.")
        if src not in cache:
            cache[src] = pd.read_csv(src, parse_dates=["date"])
        return cache[src]

    files = sorted((REPO / "results").glob("*/*/analyses.json")) + \
        sorted((REPO / "experiments" / "results").rglob("analyses.json"))
    n_files = n_windows = 0
    for f in files:
        run = f.parent.parent.name if f.parent.parent.parent == REPO / "results" else ""
        n, _ = attach(f, prices_for(f), recorded(run))
        n_files += 1
        n_windows += n
    print(f"window returns attached: {n_windows} windows in {n_files} label files")


if __name__ == "__main__":
    main()
