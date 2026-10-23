# -*- coding: utf-8 -*-
"""
Out-of-Sample Validation — Improvement #10.

Validates synthetic data quality by comparing statistical properties
of the generator against a *held-out* test period that was never seen
during training.

Approach
--------
1. Train the pipeline normally on ``[train_start, train_end]``.
2. Generate synthetic data conditioned on training-period statistics.
3. Download *real* test-period data ``[test_start, test_end]``.
4. Compare distributional moments, autocorrelation, tail behaviour,
   and financial metrics between:
   - *Synthetic* (generated from training model)
   - *Test-real*  (actual held-out data)

If the generator has truly captured the data-generating process,
the synthetic data should remain statistically close to the
unseen test data, not just the training data it was calibrated on.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ══════════════════════════════════════════════════════════════════════
# OOS metric computation
# ══════════════════════════════════════════════════════════════════════


def _log_returns(prices: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(prices[1:] / prices[:-1])
    return np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)


def _mmd_rbf(X: np.ndarray, Y: np.ndarray, gamma: float = 0.0) -> float:
    """
    Maximum Mean Discrepancy with RBF (Gaussian) kernel.

    A kernel-based two-sample test statistic.  Lower values indicate
    the two distributions are more similar.

    STAT-1: uses the median heuristic for bandwidth selection when
    gamma=0 (default).  gamma=1 was almost flat for financial returns
    with std ~ 10^-2.
    """
    from scipy.spatial.distance import cdist

    n = min(len(X), 5000)
    m = min(len(Y), 5000)
    if n < 5 or m < 5:
        return float("nan")

    rng = np.random.RandomState(42)
    X = X[rng.choice(len(X), n, replace=False)].reshape(-1, 1)
    Y = Y[rng.choice(len(Y), m, replace=False)].reshape(-1, 1)

    XX = cdist(X, X, "sqeuclidean")
    YY = cdist(Y, Y, "sqeuclidean")
    XY = cdist(X, Y, "sqeuclidean")

    # STAT-1: median heuristic — gamma = 1 / median(||x-y||^2)
    if gamma <= 0:
        all_dists = np.concatenate([XX.ravel(), YY.ravel(), XY.ravel()])
        med = float(np.median(all_dists[all_dists > 0]))
        gamma = 1.0 / med if med > 1e-15 else 1.0

    Kxx = np.exp(-gamma * XX)
    Kyy = np.exp(-gamma * YY)
    Kxy = np.exp(-gamma * XY)

    mmd2 = Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return float(np.sqrt(max(mmd2, 0.0)))


def _signature_distance(
    X: np.ndarray, Y: np.ndarray, depth: int = 4
) -> float:
    """
    Signature distance between two 1-D time series.

    Uses ``iisignature`` if available, otherwise falls back to a
    truncated rolling-statistics proxy.
    """
    try:
        import iisignature  # type: ignore[import-untyped]

        def _make_path(arr):
            t = np.linspace(0, 1, len(arr)).reshape(-1, 1)
            v = arr.reshape(-1, 1)
            return np.hstack([t, v]).astype(np.float64)

        sig_x = iisignature.sig(_make_path(X), depth)
        sig_y = iisignature.sig(_make_path(Y), depth)
        return float(np.linalg.norm(sig_x - sig_y))
    except ImportError:
        pass

    # Fallback: compare rolling statistics as path proxy
    window = min(20, len(X) // 3, len(Y) // 3)
    if window < 3:
        return float("nan")

    def _rolling_stats(arr, w):
        s = pd.Series(arr)
        return np.concatenate([
            s.rolling(w, min_periods=1).mean().values,
            s.rolling(w, min_periods=1).std().fillna(0).values,
            s.rolling(w, min_periods=1).skew().fillna(0).values,
        ])

    sx = _rolling_stats(X, window)
    sy = _rolling_stats(Y, window)
    n = min(len(sx), len(sy))
    return float(np.linalg.norm(sx[:n] - sy[:n]) / max(n, 1))


def oos_validate(
    synth_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    price_col: str = "close",
) -> Dict[str, Any]:
    """Compute out-of-sample validation metrics.

    Parameters
    ----------
    synth_df : DataFrame
        Full synthetic data (multi-sample). Must have ``date``, ``close``.
    test_df : DataFrame
        Held-out real test-period data. Must have ``date``, ``close``.
    train_df : DataFrame or None
        Training-period real data (for baseline comparison).
    price_col : str

    Returns
    -------
    report : dict
        Metrics comparing synth vs test (and optionally synth vs train).
    """
    report: Dict[str, Any] = {}

    # ── Synthetic returns (per-sample and pooled) ────────────────────
    synth_returns_per_sample: List[np.ndarray] = []
    if "sample" in synth_df.columns:
        sample_ids = sorted(synth_df["sample"].unique())
        for sid in sample_ids:
            s = synth_df[synth_df["sample"] == sid].sort_values("date")
            sr = _log_returns(s[price_col].values.astype(float))
            synth_returns_per_sample.append(sr)
        # Use first sample for distribution tests (avoids inflated N)
        synth_ret = synth_returns_per_sample[0]
    else:
        synth_ret = _log_returns(
            synth_df.sort_values("date")[price_col].values.astype(float)
        )
        synth_returns_per_sample = [synth_ret]

    # ── Test returns ────────────────────────────────────────────────
    test_sorted = test_df.sort_values("date")
    test_prices = test_sorted[price_col].values.astype(float)
    test_ret = _log_returns(test_prices)

    if len(test_ret) < 5 or len(synth_ret) < 5:
        report["error"] = "Insufficient data for OOS validation"
        return report

    # ── Distribution comparison ─────────────────────────────────────
    # STAT-9: test on single sample (not concatenated) to avoid
    # inflating sample size and over-powering the KS test.
    ks_stat, ks_pval = stats.ks_2samp(synth_ret, test_ret)
    wd = stats.wasserstein_distance(synth_ret, test_ret)

    report["oos_ks_stat"] = float(ks_stat)
    report["oos_ks_pval"] = float(ks_pval)
    report["oos_wasserstein"] = float(wd)

    # STAT-9: per-sample KS tests — report median p-value
    if len(synth_returns_per_sample) > 1:
        per_sample_ks_pvals = []
        for sr in synth_returns_per_sample:
            _, pv = stats.ks_2samp(sr, test_ret)
            per_sample_ks_pvals.append(pv)
        report["oos_ks_pval_median"] = float(np.median(per_sample_ks_pvals))
        report["oos_ks_pval_std"] = float(np.std(per_sample_ks_pvals))

    # ── Moment comparison ───────────────────────────────────────────
    for label, arr in [("synth", synth_ret), ("test", test_ret)]:
        report[f"{label}_mean"] = float(np.mean(arr))
        report[f"{label}_std"] = float(np.std(arr))
        report[f"{label}_skew"] = float(stats.skew(arr))
        report[f"{label}_kurt"] = float(stats.kurtosis(arr, fisher=True))

    # Relative differences
    for moment in ["mean", "std", "skew", "kurt"]:
        s_val = report[f"synth_{moment}"]
        t_val = report[f"test_{moment}"]
        denom = max(abs(t_val), 1e-10)
        report[f"oos_diff_{moment}"] = abs(s_val - t_val) / denom

    # ── Tail comparison ─────────────────────────────────────────────
    sigma_test = test_ret.std()
    if sigma_test > 1e-10:
        for threshold in [2, 3]:
            t_frac = float(np.mean(np.abs(test_ret) > threshold * sigma_test))
            s_frac = float(np.mean(np.abs(synth_ret) > threshold * sigma_test))
            report[f"oos_tail_{threshold}sigma_test"] = t_frac
            report[f"oos_tail_{threshold}sigma_synth"] = s_frac
            report[f"oos_tail_{threshold}sigma_diff"] = abs(t_frac - s_frac)

    # ── Rolling volatility comparison ───────────────────────────────
    window = 20
    if len(test_ret) >= window and len(synth_ret) >= window:
        # Use first sample's returns for rolling vol
        if "sample" in synth_df.columns:
            s0 = synth_df[synth_df["sample"] == 0].sort_values("date")
            s0_ret = _log_returns(s0[price_col].values.astype(float))
        else:
            s0_ret = synth_ret

        test_vol = pd.Series(test_ret).rolling(window, min_periods=1).std().values
        synth_vol = pd.Series(s0_ret[:len(test_ret)]).rolling(
            window, min_periods=1
        ).std().values

        n = min(len(test_vol), len(synth_vol))
        report["oos_vol_rmse"] = float(
            np.sqrt(np.mean((test_vol[:n] - synth_vol[:n]) ** 2))
        )
        if n > 2:
            report["oos_vol_corr"] = float(
                np.corrcoef(test_vol[:n], synth_vol[:n])[0, 1]
            )

    # ── Financial metrics (BUG-C4: per-sample, not concatenated) ───
    def _sharpe(ret):
        s = ret.std()
        return (ret.mean() / s * np.sqrt(252)) if s > 1e-10 else 0.0

    def _max_dd(ret):
        cum = np.cumsum(ret)
        return float((np.maximum.accumulate(cum) - cum).max())

    def _var95(ret):
        return float(np.percentile(ret, 5))

    report["oos_test_sharpe"] = float(_sharpe(test_ret))
    report["oos_test_maxdd"] = _max_dd(test_ret)
    report["oos_test_var95"] = _var95(test_ret)

    # Compute per-sample, then report mean ± std
    sharpes, maxdds, var95s = [], [], []
    for sr in synth_returns_per_sample:
        sharpes.append(_sharpe(sr))
        maxdds.append(_max_dd(sr))
        var95s.append(_var95(sr))

    report["oos_synth_sharpe"] = float(np.mean(sharpes))
    report["oos_synth_sharpe_std"] = float(np.std(sharpes))
    report["oos_sharpe_diff"] = abs(
        report["oos_test_sharpe"] - report["oos_synth_sharpe"]
    )

    report["oos_synth_maxdd"] = float(np.mean(maxdds))
    report["oos_synth_maxdd_std"] = float(np.std(maxdds))
    report["oos_maxdd_diff"] = abs(
        report["oos_test_maxdd"] - report["oos_synth_maxdd"]
    )

    report["oos_synth_var95"] = float(np.mean(var95s))
    report["oos_synth_var95_std"] = float(np.std(var95s))
    report["oos_var95_diff"] = abs(
        report["oos_test_var95"] - report["oos_synth_var95"]
    )

    # ── Train baseline (if available) ───────────────────────────────
    if train_df is not None:
        train_ret = _log_returns(
            train_df.sort_values("date")[price_col].values.astype(float)
        )
        if len(train_ret) > 5:
            ks_train, _ = stats.ks_2samp(synth_ret, train_ret)
            wd_train = stats.wasserstein_distance(synth_ret, train_ret)
            report["in_sample_ks"] = float(ks_train)
            report["in_sample_wasserstein"] = float(wd_train)

            # Generalisation gap: how much worse is OOS vs IS
            report["generalisation_gap_ks"] = float(ks_stat - ks_train)
            report["generalisation_gap_wasserstein"] = float(wd - wd_train)

    # ── MMD (Maximum Mean Discrepancy) ──────────────────────────────
    mmd_val = _mmd_rbf(synth_ret, test_ret)
    if np.isfinite(mmd_val):
        report["oos_mmd_rbf"] = float(mmd_val)
        # STAT-5: bootstrap CI for MMD
        rng = np.random.RandomState(42)
        n_r, n_s = len(test_ret), len(synth_ret)
        boot_mmds = []
        for _ in range(500):
            r_b = test_ret[rng.randint(0, n_r, n_r)]
            s_b = synth_ret[rng.randint(0, n_s, n_s)]
            bv = _mmd_rbf(r_b, s_b)
            if np.isfinite(bv):
                boot_mmds.append(bv)
        if boot_mmds:
            report["oos_mmd_ci_lo"] = float(np.percentile(boot_mmds, 2.5))
            report["oos_mmd_ci_hi"] = float(np.percentile(boot_mmds, 97.5))
            report["oos_mmd_se"] = float(np.std(boot_mmds, ddof=1))

    # ── Signature distance ──────────────────────────────────────────
    sig_val = _signature_distance(
        synth_ret[:min(1000, len(synth_ret))],
        test_ret[:min(1000, len(test_ret))],
    )
    if np.isfinite(sig_val):
        report["oos_signature_dist"] = float(sig_val)

    return report


# ══════════════════════════════════════════════════════════════════════
# OOS Plot
# ══════════════════════════════════════════════════════════════════════


def plot_oos_comparison(
    synth_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_df: Optional[pd.DataFrame] = None,
    ticker: str = "",
    output_dir: str = "plots",
    price_col: str = "close",
) -> None:
    """Generate OOS comparison plots.

    Creates a 2×2 figure:
        - Top-left : return distributions (synth vs test)
        - Top-right: Q-Q plot
        - Bottom-left : rolling volatility
        - Bottom-right: summary bar chart of key metrics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Returns
    test_sorted = test_df.sort_values("date")
    test_ret = _log_returns(test_sorted[price_col].values.astype(float))

    if "sample" in synth_df.columns:
        s0 = synth_df[synth_df["sample"] == 0].sort_values("date")
    else:
        s0 = synth_df.sort_values("date")
    synth_ret = _log_returns(s0[price_col].values.astype(float))

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # ── 1. Return distributions ─────────────────────────────────────
    ax = axes[0, 0]
    lo = min(test_ret.min(), synth_ret.min())
    hi = max(test_ret.max(), synth_ret.max())
    bins = np.linspace(lo, hi, 80)
    ax.hist(test_ret, bins=bins, density=True, alpha=0.5,
            label="Test (real)", color="#2196F3")
    ax.hist(synth_ret, bins=bins, density=True, alpha=0.5,
            label="Synthetic", color="#FF5722")
    if train_df is not None:
        train_ret = _log_returns(
            train_df.sort_values("date")[price_col].values.astype(float)
        )
        ax.hist(train_ret, bins=bins, density=True, alpha=0.3,
                label="Train (real)", color="#4CAF50")
    ax.set_title("Return Distributions (OOS)")
    ax.set_xlabel("Log Return")
    ax.legend(fontsize=8)

    # ── 2. Q-Q plot ─────────────────────────────────────────────────
    ax = axes[0, 1]
    n = min(len(test_ret), len(synth_ret))
    q_test = np.percentile(test_ret, np.linspace(0, 100, n))
    q_synth = np.percentile(synth_ret, np.linspace(0, 100, n))
    ax.scatter(q_test, q_synth, s=3, alpha=0.6, color="#9C27B0")
    lims = [min(q_test.min(), q_synth.min()),
            max(q_test.max(), q_synth.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_title("Q-Q Plot (Synth vs Test)")
    ax.set_xlabel("Test Quantiles")
    ax.set_ylabel("Synthetic Quantiles")
    ax.set_aspect("equal")

    # ── 3. Rolling volatility ──────────────────────────────────────
    ax = axes[1, 0]
    window = 20
    if len(test_ret) >= window:
        test_vol = pd.Series(test_ret).rolling(window, min_periods=1).std()
        ax.plot(test_vol.values, label="Test (real)", color="#2196F3", alpha=0.8)
    if len(synth_ret) >= window:
        synth_vol = pd.Series(synth_ret).rolling(window, min_periods=1).std()
        ax.plot(synth_vol.values, label="Synthetic", color="#FF5722",
                alpha=0.7, linestyle="--")
    ax.set_title(f"Rolling Volatility ({window}d)")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("σ")
    ax.legend(fontsize=8)

    # ── 4. Metric summary bars ──────────────────────────────────────
    ax = axes[1, 1]
    metrics = oos_validate(synth_df, test_df, train_df, price_col)
    bar_names = []
    bar_vals = []
    for key in ["oos_ks_stat", "oos_wasserstein",
                "oos_diff_std", "oos_diff_kurt",
                "oos_sharpe_diff", "oos_var95_diff"]:
        if key in metrics:
            bar_names.append(key.replace("oos_", "").replace("_", " ").title())
            bar_vals.append(metrics[key])

    if bar_names:
        colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0",
                  "#E91E63", "#607D8B"]
        ax.barh(bar_names, bar_vals, color=colors[:len(bar_names)], alpha=0.7)
        ax.set_title("OOS Metric Summary")
        ax.set_xlabel("Value (lower = better)")

    fig.suptitle(
        f"{ticker} — Out-of-Sample Validation" if ticker
        else "Out-of-Sample Validation",
        fontsize=14, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    path = os.path.join(output_dir, f"oos_validation_{ticker}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ OOS validation plot: {path}")


def print_oos_report(metrics: Dict[str, Any], ticker: str = "") -> None:
    """Print a formatted OOS validation report."""
    header = f"{ticker} — " if ticker else ""
    print(f"\n{'═' * 56}")
    print(f"  {header}OUT-OF-SAMPLE VALIDATION")
    print(f"{'═' * 56}")

    sections = [
        ("Distribution", ["oos_ks_stat", "oos_ks_pval", "oos_wasserstein"]),
        ("Advanced Distance", ["oos_mmd_rbf", "oos_signature_dist"]),
        ("Moments (rel. diff.)", [
            "oos_diff_mean", "oos_diff_std", "oos_diff_skew", "oos_diff_kurt"
        ]),
        ("Tail Risk", [
            "oos_tail_2sigma_diff", "oos_tail_3sigma_diff"
        ]),
        ("Financial", [
            "oos_sharpe_diff", "oos_maxdd_diff", "oos_var95_diff"
        ]),
        ("Generalisation Gap", [
            "generalisation_gap_ks", "generalisation_gap_wasserstein"
        ]),
    ]

    for name, keys in sections:
        vals = {k: metrics[k] for k in keys if k in metrics}
        if not vals:
            continue
        print(f"\n  ── {name} ──")
        for k, v in vals.items():
            label = k.replace("oos_", "").replace("_", " ").title()
            print(f"    {label:<35s} {v:>12.6f}")

    print(f"\n{'═' * 56}")
