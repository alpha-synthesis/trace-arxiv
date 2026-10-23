#!/usr/bin/env python3
"""Download the daily prices of the 28-stock panel and place them in each run folder.

Market prices (Yahoo Finance, through yfinance) are not redistributed with this
repository. This script calls the generator's own download function with the
tickers and training period of each run's config and writes
results/<run>/training_data.csv for every run folder present, which is what the
generator and the analysis scripts read.

Yahoo revises dividend-adjusted history, so a fresh download can differ slightly
from the one used for the paper. tools/attach_window_returns.py and the checks in
README.md show whether the reported numbers are affected.

Usage:
  python tools/download_prices.py [--runs r0,r2_datasign,...]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sde_causal_generator.download_data import download_training_data, save_training_data  # noqa: E402

DEFAULT_CONFIG = REPO / "configs" / "r2_datasign.yaml"


def run_settings(run: str) -> tuple[tuple[str, ...], str, str]:
    cfg_path = REPO / "configs" / f"{run}.yaml"
    cfg = yaml.safe_load((cfg_path if cfg_path.exists() else DEFAULT_CONFIG).read_text())
    return tuple(cfg["tickers"]), str(cfg["train_start"]), str(cfg["train_end"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", default="", help="comma-separated run folders (default: every folder in results/)")
    a = ap.parse_args()
    results = REPO / "results"
    runs = a.runs.split(",") if a.runs else sorted(p.name for p in results.iterdir() if p.is_dir()) if results.exists() else []
    if not runs:
        sys.exit("No run folders in results/. Extract the data bundle first (see README.md).")
    downloads: dict[tuple, object] = {}
    for run in runs:
        key = run_settings(run)
        if key not in downloads:
            tickers, start, end = key
            print(f"Downloading {len(tickers)} tickers {start} -> {end}")
            downloads[key] = download_training_data(tickers=list(tickers), start_date=start, end_date=end,
                                                    cache_dir=str(REPO / "cache" / "prices"))
        (results / run).mkdir(parents=True, exist_ok=True)
        save_training_data(downloads[key], str(results / run), "training_data.csv")
        print(f"  wrote results/{run}/training_data.csv")


if __name__ == "__main__":
    main()
