# -*- coding: utf-8 -*-
"""Shared helpers for the experiments.

Reads the generator's saved run folders (read-only) and writes outputs only
inside experiments/. Never writes to the run folders.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance

# Environment overrides; unset = the defaults:
#   TRACE_ORIG  generator checkout (sys.path and default run root; default this repository)
#   TRACE_ROOT  directory holding the run folders (default ORIG/results)
#   TRACE_OUT   output directory (default experiments/results)
ORIG = Path(os.environ.get("TRACE_ORIG", str(Path(__file__).resolve().parents[2])))
ROOT = Path(os.environ.get("TRACE_ROOT", str(ORIG / "results")))
# TRACE_RUN selects the generator run (default: r0, the price-informed labels).
RUN_TAG = os.environ.get("TRACE_RUN", "r0")
RUN = ROOT / RUN_TAG
WS = Path(__file__).resolve().parent.parent          # experiments/
OUT = Path(os.environ.get("TRACE_OUT", str(WS / "results")))

if str(ORIG) not in sys.path:
    sys.path.insert(0, str(ORIG))

OOS_START = "2023-01-01"
SEED = 42
N_SAMPLES = 10
WINDOW_TRADING_DAYS = 63
CALIB_DAYS = 504  # calibration days of the out-of-sample evaluation

_train_df_cache: pd.DataFrame | None = None


def load_panel() -> pd.DataFrame:
    global _train_df_cache
    if _train_df_cache is None:
        _train_df_cache = pd.read_csv(RUN / "training_data.csv")
    return _train_df_cache


def tickers_28() -> list[str]:
    return sorted(load_panel()["tic"].unique().tolist())


def ticker_df(ticker: str) -> pd.DataFrame:
    df = load_panel()
    return df[df["tic"] == ticker].copy().sort_values("date").reset_index(drop=True)


def split_is_oos(df: pd.DataFrame, oos_start: str = OOS_START):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    is_df = df[df["date"] < oos_start].sort_values("date").reset_index(drop=True)
    oos_df = df[df["date"] >= oos_start].sort_values("date").reset_index(drop=True)
    return is_df, oos_df


def log_returns(close: np.ndarray) -> np.ndarray:
    return np.diff(np.log(np.clip(close.astype(np.float64), 1e-8, None)))


def score_paths(real_close: np.ndarray, synth_closes: list[np.ndarray]) -> dict:
    """Per-trajectory KS, W1 and sigma, averaged over trajectories."""
    real_r = log_returns(real_close)
    per_traj = []
    for sc in synth_closes:
        sr = log_returns(np.asarray(sc, dtype=np.float64))
        n = min(len(real_r), len(sr))
        if n < 10:
            continue
        ks = float(ks_2samp(real_r[:n], sr[:n]).statistic)
        w1 = float(wasserstein_distance(real_r[:n], sr[:n]))
        sig = (float(np.std(sr[:n]) / np.std(real_r[:n]))
               if np.std(real_r[:n]) > 0 else float("nan"))
        per_traj.append({"ks": ks, "wasserstein": w1, "sigma_ratio": sig})
    return {
        "per_trajectory": per_traj,
        "ks_mean": float(np.mean([m["ks"] for m in per_traj])),
        "ks_median": float(np.median([m["ks"] for m in per_traj])),
        "wasserstein_mean": float(np.mean([m["wasserstein"] for m in per_traj])),
        "sigma_ratio_mean": float(np.mean([m["sigma_ratio"] for m in per_traj])),
    }


def load_ticker_artifacts(ticker: str):
    """ImpactMatrix (full run), IS-only ImpactMatrix, analyses, prices df."""
    from sde_causal_generator.data_structures import ImpactMatrix, WindowAnalysis

    im = ImpactMatrix.load(str(RUN / ticker / "impact_matrix.json"))
    im_is_path = RUN / ticker / "impact_matrix_is.json"
    im_is = ImpactMatrix.load(str(im_is_path)) if im_is_path.exists() else None
    with open(RUN / ticker / "analyses.json") as f:
        analyses = [WindowAnalysis.from_dict(d) for d in json.load(f)]
    return im, im_is, analyses


def build_presence(analyses, df: pd.DataFrame, factor_names: list[str]) -> np.ndarray:
    """Presence matrix rebuilt from the labels, as the generator builds it.

    Vectorized over days (2026-10-08, positive control): each window's day
    range is located by binary search on the sorted dates (start <= d <= end,
    as before), and overlapping windows keep the maximum magnitude exactly as
    the original per-day ``max`` did (a NaN magnitude never replaces a value).
    Asserted bit-identical to the original loop (``_build_presence_loop``) on
    every stock of R1 and R2 before use.
    """
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    dates = pd.to_datetime(sorted(df["date"].unique()))
    presence = np.zeros((len(dates), len(factor_names)), dtype=np.float32)
    for a in analyses:
        lo = int(dates.searchsorted(pd.Timestamp(a.start_date), side="left"))
        hi = int(dates.searchsorted(pd.Timestamp(a.end_date), side="right"))
        if hi <= lo:
            continue
        for f in a.factors:
            if f.magnitude <= 0.0 or f.name not in name_to_idx:
                continue
            col = name_to_idx[f.name]
            seg = presence[lo:hi, col]
            m = np.float32(f.magnitude)
            presence[lo:hi, col] = np.where(m > seg, m, seg)
    return presence


def _build_presence_loop(analyses, df: pd.DataFrame, factor_names: list[str]) -> np.ndarray:
    """The original per-day loop, kept as the reference for the equality check."""
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    dates = pd.to_datetime(sorted(df["date"].unique()))
    presence = np.zeros((len(dates), len(factor_names)), dtype=np.float32)
    date_to_row = {d: i for i, d in enumerate(dates)}
    for a in analyses:
        start, end = pd.Timestamp(a.start_date), pd.Timestamp(a.end_date)
        for f in a.factors:
            if f.magnitude <= 0.0 or f.name not in name_to_idx:
                continue
            col = name_to_idx[f.name]
            for d in dates:
                if start <= d <= end:
                    r = date_to_row[d]
                    presence[r, col] = max(presence[r, col], f.magnitude)
    return presence


def save_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
