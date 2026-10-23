# -*- coding: utf-8 -*-
"""
Evaluation Module for Causal SDE Generator.

Provides comprehensive quality assessment of synthetic data produced by the
Causal SDE pipeline.  Metrics are organised in three families:

1. **Statistical fidelity** — distributional similarity (KS, Wasserstein,
   KL, moments comparison).
2. **Temporal fidelity** — autocorrelation structure, volatility clustering,
   stylised fact preservation.
3. **Downstream utility** — discriminative score, predictive score, and
   (optionally) portfolio-level comparison.

Usage::

    from sde_causal_generator.evaluate_data import CausalSDEEvaluator

    evaluator = CausalSDEEvaluator(output_dir="results/causal_sde_eval")
    report = evaluator.evaluate(real_df, synth_df)
    evaluator.print_report(report)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ══════════════════════════════════════════════════════════════════════
# Helper: returns extraction (works on FinRL-format DataFrames)
# ══════════════════════════════════════════════════════════════════════


def _log_returns(prices: np.ndarray) -> np.ndarray:
    """Compute log-returns from a price series, dropping the first NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(prices[1:] / prices[:-1])
    lr = np.nan_to_num(lr, nan=0.0, posinf=0.0, neginf=0.0)
    return lr


# ══════════════════════════════════════════════════════════════════════
# STAT-5: Bootstrap confidence intervals
# ══════════════════════════════════════════════════════════════════════

def bootstrap_ci(
    metric_fn,
    real: np.ndarray,
    synth: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute bootstrap confidence interval for a two-sample metric.

    Parameters
    ----------
    metric_fn : callable(real, synth) -> float
        Scalar metric function.
    real, synth : 1-D arrays
    n_boot : int
        Number of bootstrap resamples.
    ci : float
        Confidence level (e.g. 0.95 for 95 %).

    Returns
    -------
    dict with keys ``point``, ``ci_lo``, ``ci_hi``, ``se``.
    """
    rng = np.random.RandomState(seed)
    point = float(metric_fn(real, synth))
    n_r, n_s = len(real), len(synth)
    boot_vals = np.empty(n_boot)
    for b in range(n_boot):
        r_b = real[rng.randint(0, n_r, n_r)]
        s_b = synth[rng.randint(0, n_s, n_s)]
        boot_vals[b] = metric_fn(r_b, s_b)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.percentile(boot_vals, 100 * alpha))
    hi = float(np.percentile(boot_vals, 100 * (1 - alpha)))
    se = float(np.std(boot_vals, ddof=1))
    return {"point": point, "ci_lo": lo, "ci_hi": hi, "se": se}


def _acf(x: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """Sample autocorrelation function."""
    n = len(x)
    x_centered = x - x.mean()
    var = np.var(x)
    if var < 1e-15:
        return np.zeros(max_lag + 1)
    result = np.correlate(x_centered, x_centered, mode="full")
    result = result[n - 1 :] / (var * n)
    return result[: max_lag + 1]


# ══════════════════════════════════════════════════════════════════════
# Individual metric functions
# ══════════════════════════════════════════════════════════════════════


def ks_test(real: np.ndarray, synth: np.ndarray) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov test on returns."""
    stat, pval = stats.ks_2samp(real, synth)
    return {"ks_stat": float(stat), "ks_pval": float(pval)}


def wasserstein_distance(real: np.ndarray, synth: np.ndarray) -> Dict[str, float]:
    """1-Wasserstein (earth-mover) distance on returns."""
    wd = float(stats.wasserstein_distance(real, synth))
    return {"wasserstein": wd}


def kl_divergence(
    real: np.ndarray, synth: np.ndarray, n_bins: int = 100
) -> Dict[str, float]:
    """Symmetric KL divergence on histogram-estimated densities."""
    eps = 1e-10
    lo = min(real.min(), synth.min())
    hi = max(real.max(), synth.max())
    bins = np.linspace(lo - 1e-6, hi + 1e-6, n_bins + 1)
    # STAT-4: use density=False to avoid double-normalisation.
    # Manually normalise counts into a proper probability distribution.
    p_raw, _ = np.histogram(real, bins=bins, density=False)
    q_raw, _ = np.histogram(synth, bins=bins, density=False)
    p = (p_raw + eps) / (p_raw + eps).sum()
    q = (q_raw + eps) / (q_raw + eps).sum()
    kl_pq = float(np.sum(p * np.log(p / q)))
    kl_qp = float(np.sum(q * np.log(q / p)))
    return {"kl_pq": kl_pq, "kl_qp": kl_qp, "kl_sym": (kl_pq + kl_qp) / 2}


def moments_comparison(real: np.ndarray, synth: np.ndarray) -> Dict[str, float]:
    """Compare first four moments of the return distributions."""
    result = {}
    for prefix, arr in [("real", real), ("synth", synth)]:
        result[f"{prefix}_mean"] = float(np.mean(arr))
        result[f"{prefix}_std"] = float(np.std(arr))
        result[f"{prefix}_skew"] = float(stats.skew(arr))
        result[f"{prefix}_kurt"] = float(stats.kurtosis(arr, fisher=True))

    # Relative differences
    for stat_name in ["mean", "std", "skew", "kurt"]:
        real_val = result[f"real_{stat_name}"]
        synth_val = result[f"synth_{stat_name}"]
        denom = abs(real_val) if abs(real_val) > 1e-10 else 1.0
        result[f"rel_diff_{stat_name}"] = abs(synth_val - real_val) / denom

    return result


def acf_distance(
    real: np.ndarray, synth: np.ndarray, max_lag: int = 20
) -> Dict[str, float]:
    """
    L2 distance between autocorrelation functions.

    Computes ACF on returns and on absolute returns (volatility clustering).
    """
    acf_real = _acf(real, max_lag)
    acf_synth = _acf(synth, max_lag)
    acf_dist = float(np.sqrt(np.mean((acf_real - acf_synth) ** 2)))

    acf_abs_real = _acf(np.abs(real), max_lag)
    acf_abs_synth = _acf(np.abs(synth), max_lag)
    acf_abs_dist = float(np.sqrt(np.mean((acf_abs_real - acf_abs_synth) ** 2)))

    return {
        "acf_dist_returns": acf_dist,
        "acf_dist_abs_returns": acf_abs_dist,
    }


def volatility_comparison(real: np.ndarray, synth: np.ndarray, window: int = 20) -> Dict[str, float]:
    """
    Compare rolling volatility profiles.

    Measures whether synthetic data replicates the heteroskedastic nature
    (volatility clustering) of real returns.
    """
    def rolling_vol(arr, w):
        padded = np.concatenate([np.zeros(w - 1), arr])
        return np.array([padded[i:i + w].std() for i in range(len(arr))])

    vol_real = rolling_vol(real, window)
    vol_synth = rolling_vol(synth, min(window, len(synth)))

    # Truncate to shorter
    n = min(len(vol_real), len(vol_synth))
    vol_real, vol_synth = vol_real[:n], vol_synth[:n]

    return {
        "vol_rmse": float(np.sqrt(np.mean((vol_real - vol_synth) ** 2))),
        "vol_corr": float(np.corrcoef(vol_real, vol_synth)[0, 1]) if n > 2 else 0.0,
        "real_avg_vol": float(vol_real.mean()),
        "synth_avg_vol": float(vol_synth.mean()),
    }


def tail_comparison(real: np.ndarray, synth: np.ndarray) -> Dict[str, float]:
    """
    Compare tail behaviour: extreme-event frequency.

    Checks if synthetic data produces a similar frequency of extreme moves
    (> 2σ, > 3σ), which is crucial for risk management applications.
    """
    sigma_real = real.std()
    if sigma_real < 1e-10:
        return {"tail_2sigma_diff": 0.0, "tail_3sigma_diff": 0.0}

    result = {}
    for threshold, label in [(2, "2sigma"), (3, "3sigma")]:
        real_frac = float(np.mean(np.abs(real) > threshold * sigma_real))
        synth_frac = float(np.mean(np.abs(synth) > threshold * sigma_real))
        result[f"real_{label}_frac"] = real_frac
        result[f"synth_{label}_frac"] = synth_frac
        result[f"tail_{label}_diff"] = abs(real_frac - synth_frac)

    return result


def financial_metrics(
    real_prices: np.ndarray,
    synth_prices: np.ndarray,
    annual_factor: float = 252,
) -> Dict[str, float]:
    """
    Portfolio / risk metrics comparing real and synthetic price series.

    Includes Sharpe ratio, max drawdown, and Value-at-Risk.
    """
    real_ret = _log_returns(real_prices)
    synth_ret = _log_returns(synth_prices)

    result = {}
    for prefix, ret in [("real", real_ret), ("synth", synth_ret)]:
        sr = 0.0
        if ret.std() > 1e-10:
            sr = (ret.mean() / ret.std()) * np.sqrt(annual_factor)
        result[f"{prefix}_sharpe"] = float(sr)

        # Max drawdown
        cum = np.cumsum(ret)
        running_max = np.maximum.accumulate(cum)
        drawdown = running_max - cum
        result[f"{prefix}_max_dd"] = float(drawdown.max())

        # VaR 5%
        result[f"{prefix}_var95"] = float(np.percentile(ret, 5))

    result["sharpe_diff"] = abs(result["real_sharpe"] - result["synth_sharpe"])
    result["max_dd_diff"] = abs(result["real_max_dd"] - result["synth_max_dd"])
    result["var95_diff"] = abs(result["real_var95"] - result["synth_var95"])

    return result


# ══════════════════════════════════════════════════════════════════════
# Discriminative & Predictive scores (lightweight versions)
# ══════════════════════════════════════════════════════════════════════


def discriminative_score(
    real_windows: np.ndarray,
    synth_windows: np.ndarray,
    epochs: int = 150,
    hidden_dim: int = 32,
    batch_size: int = 64,
    n_runs: int = 3,
) -> float:
    """
    Post-hoc RNN classifier to distinguish real vs synthetic.

    Score close to 0 ⇒ indistinguishable (ideal).
    Score close to 0.5 ⇒ perfectly distinguished.

    Parameters
    ----------
    real_windows, synth_windows : ndarray, shape (N, seq_len, features)
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  ⚠ PyTorch not available — skipping discriminative score")
        return float("nan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n = min(len(real_windows), len(synth_windows))
    real_windows = real_windows[:n]
    synth_windows = synth_windows[:n]
    input_dim = real_windows.shape[2]

    scores = []
    for _ in range(n_runs):
        idx = np.random.permutation(n)
        split = int(0.8 * n)
        train_i, test_i = idx[:split], idx[split:]

        X_train = np.vstack([real_windows[train_i], synth_windows[train_i]])
        y_train = np.array([1.0] * len(train_i) + [0.0] * len(train_i))
        X_test = np.vstack([real_windows[test_i], synth_windows[test_i]])
        y_test = np.array([1.0] * len(test_i) + [0.0] * len(test_i))

        perm = np.random.permutation(len(X_train))
        X_train, y_train = X_train[perm], y_train[perm]

        ds = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train).unsqueeze(1),
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        fc = nn.Linear(hidden_dim, 1)
        model = nn.Sequential()  # placeholder — use custom forward
        rnn.to(device)
        fc.to(device)
        optimizer = torch.optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=1e-3)
        criterion = nn.BCEWithLogitsLoss()

        rnn.train()
        fc.train()
        for _epoch in range(epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out, _ = rnn(Xb)
                logit = fc(out[:, -1, :])
                loss = criterion(logit, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rnn.eval()
        fc.eval()
        with torch.no_grad():
            Xt = torch.FloatTensor(X_test).to(device)
            out, _ = rnn(Xt)
            logit = fc(out[:, -1, :])
            preds = (torch.sigmoid(logit).cpu().numpy().flatten() > 0.5).astype(float)
            acc = np.mean(preds == y_test)
        scores.append(abs(acc - 0.5))

    return float(np.mean(scores))


def predictive_score(
    real_windows: np.ndarray,
    synth_windows: np.ndarray,
    epochs: int = 150,
    hidden_dim: int = 32,
    batch_size: int = 64,
    n_runs: int = 3,
) -> float:
    """
    Train next-step predictor on synthetic → evaluate MAE on real.

    Lower is better.
    """
    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("  ⚠ PyTorch not available — skipping predictive score")
        return float("nan")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = real_windows.shape[2]
    maes = []

    for _ in range(n_runs):
        X_synth = synth_windows[:, :-1, :]
        y_synth = synth_windows[:, 1:, -1:]

        ds = TensorDataset(torch.FloatTensor(X_synth), torch.FloatTensor(y_synth))
        loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

        rnn = nn.GRU(input_dim, hidden_dim, num_layers=2, batch_first=True)
        fc = nn.Linear(hidden_dim, 1)
        rnn.to(device)
        fc.to(device)
        optimizer = torch.optim.Adam(list(rnn.parameters()) + list(fc.parameters()), lr=1e-3)
        # BUG-C3: use MSE with linear output (targets are log-returns,
        # not probabilities — sigmoid would clip to (0,1)).
        criterion = nn.MSELoss()

        rnn.train()
        fc.train()
        for _epoch in range(epochs):
            for Xb, yb in loader:
                Xb, yb = Xb.to(device), yb.to(device)
                out, _ = rnn(Xb)
                pred = fc(out)  # BUG-C3: linear output, no sigmoid
                loss = criterion(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        rnn.eval()
        fc.eval()
        with torch.no_grad():
            Xr = torch.FloatTensor(real_windows[:, :-1, :]).to(device)
            yr = torch.FloatTensor(real_windows[:, 1:, -1:]).to(device)
            out, _ = rnn(Xr)
            pred = fc(out)  # BUG-C3: linear output, no sigmoid
            mae = float(torch.mean(torch.abs(pred - yr)).item())
        maes.append(mae)

    return float(np.mean(maes))


# ══════════════════════════════════════════════════════════════════════
# Plotting
# ══════════════════════════════════════════════════════════════════════


def plot_price_comparison(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    ticker: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Plot real vs synthetic close prices."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(real_df["date"], real_df["close"], label="Real", color="#2196F3", alpha=0.8)
    ax.plot(synth_df["date"], synth_df["close"], label="Synthetic", color="#FF5722", alpha=0.7, linestyle="--")
    ax.set_title(f"Price Comparison — {ticker}" if ticker else "Price Comparison")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price")
    ax.legend()
    plt.tight_layout()
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_return_distributions(
    real_ret: np.ndarray,
    synth_ret: np.ndarray,
    ticker: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Overlaid histograms + KDE of log-return distributions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax = axes[0]
    bins = np.linspace(
        min(real_ret.min(), synth_ret.min()),
        max(real_ret.max(), synth_ret.max()),
        80,
    )
    ax.hist(real_ret, bins=bins, density=True, alpha=0.5, label="Real", color="#2196F3")
    ax.hist(synth_ret, bins=bins, density=True, alpha=0.5, label="Synthetic", color="#FF5722")
    ax.set_title("Return Distributions")
    ax.set_xlabel("Log Return")
    ax.legend()

    # QQ-plot
    ax = axes[1]
    real_sorted = np.sort(real_ret)
    synth_sorted = np.sort(synth_ret)
    n = min(len(real_sorted), len(synth_sorted))
    real_q = np.percentile(real_sorted, np.linspace(0, 100, n))
    synth_q = np.percentile(synth_sorted, np.linspace(0, 100, n))
    ax.scatter(real_q, synth_q, s=3, alpha=0.6, color="#9C27B0")
    lims = [min(real_q.min(), synth_q.min()), max(real_q.max(), synth_q.max())]
    ax.plot(lims, lims, "k--", alpha=0.5, linewidth=1)
    ax.set_title("Q-Q Plot")
    ax.set_xlabel("Real Quantiles")
    ax.set_ylabel("Synthetic Quantiles")
    ax.set_aspect("equal")

    fig.suptitle(f"Returns: {ticker}" if ticker else "Returns Analysis", y=1.02)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_acf_comparison(
    real_ret: np.ndarray,
    synth_ret: np.ndarray,
    max_lag: int = 30,
    ticker: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """ACF comparison for returns and absolute returns."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, transform) in enumerate(
        [("Returns", lambda x: x), ("|Returns|", np.abs)]
    ):
        ax = axes[i]
        real_acf = _acf(transform(real_ret), max_lag)
        synth_acf = _acf(transform(synth_ret), max_lag)
        lags = np.arange(max_lag + 1)
        ax.bar(lags - 0.15, real_acf, width=0.3, alpha=0.7, label="Real", color="#2196F3")
        ax.bar(lags + 0.15, synth_acf, width=0.3, alpha=0.7, label="Synthetic", color="#FF5722")
        ax.set_title(f"ACF — {name}")
        ax.set_xlabel("Lag")
        ax.legend()

    fig.suptitle(f"Autocorrelation: {ticker}" if ticker else "Autocorrelation", y=1.02)
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


def plot_rolling_volatility(
    real_ret: np.ndarray,
    synth_ret: np.ndarray,
    window: int = 20,
    ticker: str = "",
    output_path: Optional[str] = None,
) -> plt.Figure:
    """Compare rolling volatility (std of returns)."""
    def roll_std(x, w):
        return pd.Series(x).rolling(w, min_periods=1).std().values

    vol_r = roll_std(real_ret, window)
    vol_s = roll_std(synth_ret, window)

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(vol_r, label="Real", color="#2196F3", alpha=0.8)
    ax.plot(vol_s, label="Synthetic", color="#FF5722", alpha=0.7, linestyle="--")
    ax.set_title(f"Rolling Volatility ({window}d) — {ticker}" if ticker else f"Rolling Volatility ({window}d)")
    ax.set_xlabel("Trading Day")
    ax.set_ylabel("Volatility (σ)")
    ax.legend()
    plt.tight_layout()

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Windowing helper for disc/pred scores
# ══════════════════════════════════════════════════════════════════════


def _create_windows(
    prices: np.ndarray, seq_len: int = 24, stride: int = 1
) -> np.ndarray:
    """Sliding-window (N, seq_len, 1) from a 1-D price array."""
    ret = _log_returns(prices)
    windows = []
    for i in range(0, len(ret) - seq_len + 1, stride):
        windows.append(ret[i : i + seq_len])
    if not windows:
        return np.empty((0, seq_len, 1))
    return np.array(windows)[:, :, np.newaxis]


# ══════════════════════════════════════════════════════════════════════
# Main evaluator class
# ══════════════════════════════════════════════════════════════════════


class CausalSDEEvaluator:
    """
    Comprehensive evaluator for Causal SDE synthetic data.

    Accepts FinRL-format DataFrames (date, tic, open, high, low, close, volume)
    or raw price arrays.

    Parameters
    ----------
    output_dir : str
        Directory for saving reports and plots.
    include_deep_metrics : bool
        Whether to compute discriminative/predictive scores (slower).
    """

    def __init__(
        self,
        output_dir: str = "evaluation_results_causal_sde",
        include_deep_metrics: bool = True,
    ):
        self.output_dir = output_dir
        self.include_deep_metrics = include_deep_metrics
        os.makedirs(output_dir, exist_ok=True)

    # ── Main entry point ────────────────────────────────────────────

    def evaluate(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        price_col: str = "close",
        date_col: str = "date",
        ticker_col: str = "tic",
    ) -> Dict[str, Any]:
        """
        Run all evaluation metrics on real vs synthetic data.

        Parameters
        ----------
        real_df, synth_df : pd.DataFrame
            Must contain at least ``date_col`` and ``price_col``.
            If ``ticker_col`` exists with multiple tickers,
            metrics are computed per-ticker and averaged.

        Returns
        -------
        report : dict
            Nested dict: ``{ticker: {metric: value, ...}, "_aggregate": {...}}``.
        """
        real_df = real_df.copy()
        synth_df = synth_df.copy()
        real_df[date_col] = pd.to_datetime(real_df[date_col])
        synth_df[date_col] = pd.to_datetime(synth_df[date_col])

        # Handle scenario column
        if "scenario" in synth_df.columns:
            n_scenarios = synth_df["scenario"].nunique()
            print(f"  ℹ  {n_scenarios} scenarios detected — using scenario 0 for evaluation.")
            synth_df = synth_df[synth_df["scenario"] == 0].copy()

        # Detect tickers
        if ticker_col in real_df.columns:
            tickers = sorted(real_df[ticker_col].unique())
        else:
            tickers = ["SINGLE"]
            real_df[ticker_col] = "SINGLE"
            synth_df[ticker_col] = "SINGLE"

        report: Dict[str, Any] = {}

        for tic in tickers:
            print(f"\n  ─── Evaluating: {tic} ───")
            r = real_df[real_df[ticker_col] == tic].sort_values(date_col).reset_index(drop=True)
            s = synth_df[synth_df[ticker_col] == tic].sort_values(date_col).reset_index(drop=True)

            if r.empty or s.empty:
                print(f"    ⚠ Skipping {tic}: empty data")
                continue

            real_prices = r[price_col].values.astype(float)
            synth_prices = s[price_col].values.astype(float)

            real_ret = _log_returns(real_prices)
            synth_ret = _log_returns(synth_prices)

            if len(real_ret) < 5 or len(synth_ret) < 5:
                print(f"    ⚠ Skipping {tic}: too few data points")
                continue

            # ── Statistical metrics ─────────────────────────────────
            m: Dict[str, float] = {}
            m.update(ks_test(real_ret, synth_ret))
            m.update(wasserstein_distance(real_ret, synth_ret))
            m.update(kl_divergence(real_ret, synth_ret))
            m.update(moments_comparison(real_ret, synth_ret))
            m.update(acf_distance(real_ret, synth_ret))
            m.update(volatility_comparison(real_ret, synth_ret))
            m.update(tail_comparison(real_ret, synth_ret))
            m.update(financial_metrics(real_prices, synth_prices))

            # ── STAT-5: bootstrap CIs for key metrics ──────────────
            _ks_fn = lambda r, s: float(stats.ks_2samp(r, s)[0])
            _wd_fn = lambda r, s: float(stats.wasserstein_distance(r, s))
            _acf_fn = lambda r, s: float(
                np.sqrt(np.mean((_acf(r) - _acf(s)) ** 2))
            )
            for name, fn in [
                ("ks_stat", _ks_fn),
                ("wasserstein", _wd_fn),
                ("acf_dist", _acf_fn),
            ]:
                ci = bootstrap_ci(fn, real_ret, synth_ret, n_boot=1000)
                m[f"{name}_ci_lo"] = ci["ci_lo"]
                m[f"{name}_ci_hi"] = ci["ci_hi"]
                m[f"{name}_se"] = ci["se"]

            # Price correlation / RMSE (aligned length)
            n_common = min(len(real_prices), len(synth_prices))
            rp, sp = real_prices[:n_common], synth_prices[:n_common]
            m["price_rmse"] = float(np.sqrt(np.mean((rp - sp) ** 2)))
            if n_common > 2:
                m["price_corr"] = float(np.corrcoef(rp, sp)[0, 1])
            else:
                m["price_corr"] = 0.0

            # ── Deep metrics ────────────────────────────────────────
            if self.include_deep_metrics:
                seq_len = min(24, len(real_ret) // 3, len(synth_ret) // 3)
                if seq_len >= 8:
                    real_w = _create_windows(real_prices, seq_len)
                    synth_w = _create_windows(synth_prices, seq_len)
                    if len(real_w) > 10 and len(synth_w) > 10:
                        m["disc_score"] = discriminative_score(real_w, synth_w)
                        m["pred_score"] = predictive_score(real_w, synth_w)
                    else:
                        m["disc_score"] = float("nan")
                        m["pred_score"] = float("nan")
                else:
                    m["disc_score"] = float("nan")
                    m["pred_score"] = float("nan")

            report[tic] = m

            # ── Plots per ticker ────────────────────────────────────
            plot_price_comparison(r, s, ticker=tic, output_path=os.path.join(self.output_dir, f"price_{tic}.png"))
            plot_return_distributions(real_ret, synth_ret, ticker=tic, output_path=os.path.join(self.output_dir, f"returns_{tic}.png"))
            plot_acf_comparison(real_ret, synth_ret, ticker=tic, output_path=os.path.join(self.output_dir, f"acf_{tic}.png"))
            plot_rolling_volatility(real_ret, synth_ret, ticker=tic, output_path=os.path.join(self.output_dir, f"vol_{tic}.png"))

        # ── Aggregate across tickers ────────────────────────────────
        if report:
            report["_aggregate"] = self._aggregate(report)
            self._save_report(report)
            self._create_summary_table(report)

        return report

    # ── Helpers ──────────────────────────────────────────────────────

    def _aggregate(self, report: Dict[str, Any]) -> Dict[str, float]:
        """Average numeric metrics across tickers."""
        ticker_keys = [k for k in report if not k.startswith("_")]
        if not ticker_keys:
            return {}

        agg: Dict[str, List[float]] = {}
        for tic in ticker_keys:
            for metric, val in report[tic].items():
                if isinstance(val, (int, float)) and not np.isnan(val):
                    agg.setdefault(metric, []).append(val)

        return {k: float(np.mean(v)) for k, v in agg.items()}

    def _save_report(self, report: Dict[str, Any]) -> None:
        """Save full report as JSON."""
        path = os.path.join(self.output_dir, "quality_report.json")
        # Convert numpy types
        def convert(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=convert)
        print(f"\n  ✓ Full report saved to {path}")

    def _create_summary_table(self, report: Dict[str, Any]) -> None:
        """Create and save a CSV summary table."""
        ticker_keys = [k for k in report if not k.startswith("_")]
        if not ticker_keys:
            return

        rows = []
        for tic in ticker_keys:
            row = {"ticker": tic}
            for metric, val in report[tic].items():
                if isinstance(val, (int, float)):
                    row[metric] = val
            rows.append(row)

        # Add aggregate row
        agg_row = {"ticker": "AGGREGATE"}
        agg_row.update(report.get("_aggregate", {}))
        rows.append(agg_row)

        df = pd.DataFrame(rows)
        path = os.path.join(self.output_dir, "metrics_summary.csv")
        df.to_csv(path, index=False, float_format="%.6f")
        print(f"  ✓ Summary table saved to {path}")

    # ── Pretty print ────────────────────────────────────────────────

    @staticmethod
    def print_report(report: Dict[str, Any]) -> None:
        """Print a human-readable summary to stdout."""
        agg = report.get("_aggregate", {})
        if not agg:
            print("  No metrics to display.")
            return

        print(f"\n{'═' * 60}")
        print("  CAUSAL SDE — SYNTHETIC DATA QUALITY REPORT")
        print(f"{'═' * 60}")

        sections = [
            (
                "Statistical Fidelity",
                ["ks_stat", "ks_pval", "wasserstein", "kl_sym"],
            ),
            (
                "Moments",
                ["rel_diff_mean", "rel_diff_std", "rel_diff_skew", "rel_diff_kurt"],
            ),
            (
                "Temporal Fidelity",
                ["acf_dist_returns", "acf_dist_abs_returns", "vol_rmse", "vol_corr"],
            ),
            (
                "Tail Risk",
                ["tail_2sigma_diff", "tail_3sigma_diff"],
            ),
            (
                "Financial Metrics",
                ["sharpe_diff", "max_dd_diff", "var95_diff", "price_rmse", "price_corr"],
            ),
            (
                "Deep Metrics",
                ["disc_score", "pred_score"],
            ),
        ]

        for section_name, keys in sections:
            vals = {k: agg[k] for k in keys if k in agg}
            if not vals:
                continue
            print(f"\n  ── {section_name} ──")
            for k, v in vals.items():
                label = k.replace("_", " ").title()
                print(f"    {label:<30s} {v:>12.6f}")

        print(f"\n{'═' * 60}")

    # ── Convenience: evaluate from files ────────────────────────────

    @classmethod
    def evaluate_from_files(
        cls,
        real_csv: str,
        synth_csv: str,
        output_dir: str = "evaluation_results_causal_sde",
        include_deep_metrics: bool = True,
    ) -> Dict[str, Any]:
        """
        Load CSVs and run evaluation.

        Convenience entry point for scripts / CLI.
        """
        real_df = pd.read_csv(real_csv)
        synth_df = pd.read_csv(synth_csv)

        evaluator = cls(
            output_dir=output_dir,
            include_deep_metrics=include_deep_metrics,
        )
        report = evaluator.evaluate(real_df, synth_df)
        evaluator.print_report(report)
        return report


# ══════════════════════════════════════════════════════════════════════
# Causal comparison plot — real vs synthetic with event annotations
# ══════════════════════════════════════════════════════════════════════


def plot_causal_comparison(
    real_df: pd.DataFrame,
    synth_df: pd.DataFrame,
    analyses: List[Dict],
    ticker: str = "",
    output_dir: str = "plots",
    top_n_events: int = 25,
    min_magnitude: float = 0.08,
    validated_events: Optional[List[Dict]] = None,
    generation_factors: Optional[List[Dict]] = None,
) -> None:
    """
    Generate annotated comparison plots of real vs synthetic prices.

    Creates up to three plots:
        1. **Full training period** (real close) with causal event markers.
        2. **Synthetic overlay** — synthetic samples overlaid on real
           price range, also annotated with the most impactful factors.
        3. **Generation factor schedule** — real close with coloured
           time-span boxes for each active generation factor (only
           produced when *generation_factors* is provided).

    Parameters
    ----------
    real_df : pd.DataFrame
        Real price data with ``date`` and ``close`` columns.
    synth_df : pd.DataFrame
        Synthetic data with ``date``, ``close``, and optionally ``sample``.
    analyses : list[dict]
        Window analyses dicts (from ``analyses.json``).
    ticker : str
        Ticker symbol for titles.
    output_dir : str
        Directory to save PNG files.
    top_n_events : int
        Max number of event annotations to show.
    min_magnitude : float
        Minimum factor magnitude to consider for annotation.
    validated_events : list[dict] | None
        If provided, use these user-validated events instead of
        re-extracting from ``analyses``.  Each dict should have keys:
        ``name``, ``start_date``, ``end_date``, ``direction``,
        ``probability``.
    generation_factors : list[dict] | None
        If provided, produce an additional PNG showing factor boxes.
        Each dict should have keys: ``name``, ``start_date``,
        ``end_date``, ``direction``, ``probability``
        (same format as generation editor export).
    """
    os.makedirs(output_dir, exist_ok=True)

    # ── Parse dates ─────────────────────────────────────────────────
    real = real_df.copy()
    real["date"] = pd.to_datetime(real["date"])
    real = real.sort_values("date")

    synth = synth_df.copy()
    synth["date"] = pd.to_datetime(synth["date"])

    # ── Extract / use events ────────────────────────────────────────
    # Priority: generation_factors > validated_events > raw analyses
    def _dir_str_to_num(d: str) -> float:
        return 0.5 if d == "bullish" else (-0.5 if d == "bearish" else 0.0)

    if generation_factors is not None and len(generation_factors) > 0:
        # Best source: generation factors (final factors used in synthesis)
        events = []
        for gf in generation_factors:
            start = pd.Timestamp(gf["start_date"])
            end = pd.Timestamp(gf["end_date"])
            mid_date = start + (end - start) / 2
            dir_num = _dir_str_to_num(gf.get("direction", "neutral"))
            events.append({
                "date": mid_date,
                "name": gf["name"].replace("_", " ").title(),
                "magnitude": gf.get("probability", 0.5),
                "direction": dir_num,
                "window_start": gf["start_date"],
                "window_end": gf["end_date"],
            })
        events.sort(key=lambda x: x["magnitude"], reverse=True)
    elif validated_events is not None:
        # Fallback 1: user-validated events from interactive editor
        events = []
        for ve in validated_events:
            start = pd.Timestamp(ve["start_date"])
            end = pd.Timestamp(ve["end_date"])
            mid_date = start + (end - start) / 2
            dir_num = _dir_str_to_num(ve.get("direction", "neutral"))
            events.append({
                "date": mid_date,
                "name": ve["name"].replace("_", " ").title(),
                "magnitude": ve.get("probability", 0.5),
                "direction": dir_num,
                "window_start": ve["start_date"],
                "window_end": ve["end_date"],
            })
        events.sort(key=lambda x: x["magnitude"], reverse=True)
    else:
        # Fallback 2: extract from raw analyses
        events = []
        for w in analyses:
            mid_date = pd.Timestamp(w["start_date"]) + (
                pd.Timestamp(w["end_date"]) - pd.Timestamp(w["start_date"])
            ) / 2
            for f in w.get("factors", []):
                name = f["name"]
                # skip noise/residual factors
                if any(skip in name.lower() for skip in ("noise", "residual")):
                    continue
                mag = abs(f["magnitude"])
                if mag < min_magnitude:
                    continue
                direction = f["direction"]
                events.append({
                    "date": mid_date,
                    "name": name.replace("_", " ").title(),
                    "magnitude": mag,
                    "direction": direction,
                    "window_start": w["start_date"],
                    "window_end": w["end_date"],
                })

        # Sort by magnitude
        events.sort(key=lambda x: x["magnitude"], reverse=True)

    # ────────────────────────────────────────────────────────────────
    # Post-process: de-duplicate and enforce temporal diversity
    # ────────────────────────────────────────────────────────────────
    def _dedup_events(events_list: List[Dict]) -> List[Dict]:
        """De-duplicate events by name, keeping the highest-magnitude
        occurrence of each unique factor name."""
        best: Dict[str, Dict] = {}
        for ev in events_list:
            key = ev["name"]
            if key not in best or ev["magnitude"] > best[key]["magnitude"]:
                best[key] = ev
        return sorted(best.values(), key=lambda x: x["magnitude"], reverse=True)

    events = _dedup_events(events)

    # ────────────────────────────────────────────────────────────────
    # Helper: draw one causal-events axes (reused for both variants)
    # ────────────────────────────────────────────────────────────────
    def _draw_causal_events(ax_in, events_list, *, draw_extent: bool,
                               force_log: bool = False):
        """Populate *ax_in* with price line + event markers/annotations.

        Parameters
        ----------
        draw_extent : bool
            If True, also draw horizontal extent lines (error-bar style)
            from ``window_start`` to ``window_end``.
        force_log : bool
            If True, use logarithmic Y-axis.  If False, use linear.
        """
        use_log = force_log
        n_events = len(events_list)
        # Adaptive: smaller font / narrower stagger with many events
        ann_fontsize = max(4.5, 6.5 - 0.02 * max(0, n_events - 40))
        stagger_k = max(5, min(n_events // 3, 12))

        ax_in.plot(
            real["date"], real["close"],
            color="#1565C0", linewidth=1.2, alpha=0.9, label="Real Close",
        )

        if use_log:
            ax_in.set_yscale("log")
            from matplotlib.ticker import ScalarFormatter
            ax_in.yaxis.set_major_formatter(ScalarFormatter())
            ax_in.yaxis.get_major_formatter().set_scientific(False)

        y_lo, y_hi = real["close"].min(), real["close"].max()
        y_rng = y_hi - y_lo

        for i, ev in enumerate(events_list):
            idx = (real["date"] - ev["date"]).abs().idxmin()
            price_at_event = real.loc[idx, "close"]
            event_date = real.loc[idx, "date"]

            ws = pd.Timestamp(ev["window_start"])
            we = pd.Timestamp(ev["window_end"])

            if ev["direction"] > 0.15:
                color, marker = "#2E7D32", "^"
            elif ev["direction"] < -0.15:
                color, marker = "#C62828", "v"
            else:
                color, marker = "#757575", "o"

            # ── Horizontal extent line (optional) ──
            if draw_extent:
                ax_in.hlines(
                    y=price_at_event, xmin=ws, xmax=we,
                    colors=color, linewidth=1.8, alpha=0.6, zorder=4,
                )
                if use_log:
                    cap_h_lo = price_at_event * 0.95
                    cap_h_hi = price_at_event * 1.05
                else:
                    cap_h = y_rng * 0.012
                    cap_h_lo = price_at_event - cap_h
                    cap_h_hi = price_at_event + cap_h
                for cap_x in (ws, we):
                    ax_in.vlines(
                        x=cap_x,
                        ymin=cap_h_lo,
                        ymax=cap_h_hi,
                        colors=color, linewidth=1.4, alpha=0.6, zorder=4,
                    )

            # ── Center marker ──
            size = 30 + ev["magnitude"] * 200
            ax_in.scatter(
                event_date, price_at_event,
                s=size, color=color, zorder=5, alpha=0.85,
                marker=marker, edgecolors="white", linewidth=0.6,
            )

            # ── Stagger text annotation ──
            if use_log:
                import math
                log_lo = math.log10(max(y_lo, 0.01))
                log_hi = math.log10(max(y_hi, 0.01))
                log_rng = log_hi - log_lo
                log_price = math.log10(max(price_at_event, 0.01))
                offset_sign = 1 if (i % 2 == 0) else -1
                log_text_y = log_price + offset_sign * log_rng * (0.06 + 0.03 * (i % stagger_k))
                log_text_y = max(log_lo - log_rng * 0.05, min(log_hi + log_rng * 0.15, log_text_y))
                text_y = 10 ** log_text_y
            else:
                offset_sign = 1 if (i % 2 == 0) else -1
                text_y = price_at_event + offset_sign * y_rng * (0.06 + 0.03 * (i % stagger_k))
                text_y = max(y_lo - y_rng * 0.05, min(y_hi + y_rng * 0.15, text_y))

            ax_in.annotate(
                ev["name"],
                xy=(event_date, price_at_event),
                xytext=(event_date, text_y),
                fontsize=ann_fontsize, color=color, fontweight="bold",
                ha="center",
                va="bottom" if offset_sign > 0 else "top",
                arrowprops=dict(arrowstyle="-", color=color, alpha=0.4, linewidth=0.5),
                bbox=dict(
                    boxstyle="round,pad=0.2", facecolor="white",
                    edgecolor=color, alpha=0.7, linewidth=0.5,
                ),
            )

        # Decorations
        ax_in.set_xlabel("Date", fontsize=11)
        ax_in.set_ylabel("Close Price ($)", fontsize=11)
        ax_in.xaxis.set_major_locator(mdates.YearLocator())
        ax_in.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax_in.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
        plt.setp(ax_in.get_xticklabels(), rotation=45, ha="right")
        ax_in.grid(True, alpha=0.3, linestyle="--", which="both")

    # ────────────────────────────────────────────────────────────────
    # PLOT 1a/1b: Causal events — extent lines + markers only
    # Each variant gets both linear and log scale versions.
    # ────────────────────────────────────────────────────────────────
    from matplotlib.lines import Line2D

    legend_extent = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2E7D32",
               markersize=8, label="Bullish"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#C62828",
               markersize=8, label="Bearish"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#757575",
               markersize=8, label="Neutral"),
        Line2D([0], [0], color="#1565C0", linewidth=2, label="Real Close"),
        Line2D([0, 1], [0, 0], color="#555555", linewidth=1.8,
               marker="|", markersize=6, label="Event Window"),
    ]
    legend_markers = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#2E7D32",
               markersize=8, label="Bullish"),
        Line2D([0], [0], marker="v", color="w", markerfacecolor="#C62828",
               markersize=8, label="Bearish"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#757575",
               markersize=8, label="Neutral"),
        Line2D([0], [0], color="#1565C0", linewidth=2, label="Real Close"),
    ]

    _event_plot_specs = [
        # (draw_extent, force_log, suffix, title_extra, legend)
        (True,  False, f"causal_events_{ticker}",            "with extent lines",             legend_extent),
        (True,  True,  f"causal_events_log_{ticker}",        "with extent lines — log scale", legend_extent),
        (False, False, f"causal_events_markers_{ticker}",    "markers only",                  legend_markers),
        (False, True,  f"causal_events_markers_log_{ticker}","markers only — log scale",      legend_markers),
    ]

    # Adaptive figure height based on event count
    _fig_h = max(8, min(14, 8 + 0.06 * len(events)))

    for draw_ext, use_log, fname, title_extra, legend_handles in _event_plot_specs:
        fig, ax = plt.subplots(figsize=(22, _fig_h))
        _draw_causal_events(ax, events, draw_extent=draw_ext, force_log=use_log)
        ax.set_title(
            f"{ticker} — Causal Events ({title_extra})"
            if ticker else f"Causal Events ({title_extra})",
            fontsize=14, fontweight="bold",
        )
        ax.legend(handles=legend_handles, loc="upper left", fontsize=9)
        plt.tight_layout()
        path = os.path.join(output_dir, f"{fname}.png")
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"    ✓ {title_extra}: {path}")

    # ────────────────────────────────────────────────────────────────
    # PLOT 2: Real vs Synthetic – per-component OHLCV envelopes
    # Generate both linear and log-scale versions.
    # ────────────────────────────────────────────────────────────────
    has_samples = "sample" in synth.columns
    samples = sorted(synth["sample"].unique()) if has_samples else [0]

    # Components to plot (each gets its own envelope + row)
    price_cols = [c for c in ["open", "high", "low", "close"] if c in real.columns and c in synth.columns]
    has_volume = "volume" in real.columns and "volume" in synth.columns

    def _draw_real_vs_synth(*, use_log: bool, suffix: str):
        """Draw and save a real-vs-synth OHLCV envelope plot."""
        n_rows = len(price_cols) + (1 if has_volume else 0)
        height_ratios = [3] * len(price_cols) + ([1] if has_volume else [])
        fig_s, axes_s = plt.subplots(
            n_rows, 1, figsize=(22, 4 * n_rows),
            height_ratios=height_ratios, sharex=True,
            gridspec_kw={"hspace": 0.08},
        )
        if n_rows == 1:
            axes_s = [axes_s]

        col_colors = {
            "open":  ("#1565C0", "#FF9800", "#E65100"),
            "high":  ("#2E7D32", "#66BB6A", "#1B5E20"),
            "low":   ("#C62828", "#EF5350", "#B71C1C"),
            "close": ("#1565C0", "#FF9800", "#E65100"),
        }

        for row_idx, col_name in enumerate(price_cols):
            ax_s = axes_s[row_idx]
            c_real, c_fill, c_mean = col_colors.get(
                col_name, ("#1565C0", "#FF9800", "#E65100"))

            ax_s.plot(
                real["date"], real[col_name],
                color=c_real, linewidth=1.8, alpha=0.95,
                label=f"Real {col_name.title()}", zorder=10,
            )

            if has_samples and len(samples) > 1:
                synth_pivot = synth.pivot_table(
                    index="date", columns="sample",
                    values=col_name, aggfunc="first",
                ).sort_index()

                mean_vals = synth_pivot.mean(axis=1)
                p10 = synth_pivot.quantile(0.10, axis=1)
                p25 = synth_pivot.quantile(0.25, axis=1)
                p75 = synth_pivot.quantile(0.75, axis=1)
                p90 = synth_pivot.quantile(0.90, axis=1)
                dates_synth = synth_pivot.index

                ax_s.fill_between(
                    dates_synth, p10, p90,
                    color=c_fill, alpha=0.15, label="10–90th pctl",
                )
                ax_s.fill_between(
                    dates_synth, p25, p75,
                    color=c_fill, alpha=0.30, label="25–75th pctl",
                )
                ax_s.plot(
                    dates_synth, mean_vals,
                    color=c_mean, linewidth=1.4, alpha=0.9,
                    linestyle="--", label=f"Synth Mean (n={len(samples)})",
                )
            else:
                s_data = synth.sort_values("date")
                ax_s.plot(
                    s_data["date"], s_data[col_name],
                    color=c_fill, linewidth=1.0, alpha=0.8,
                    linestyle="--", label="Synthetic",
                )

            if use_log:
                ax_s.set_yscale("log")
                from matplotlib.ticker import ScalarFormatter
                ax_s.yaxis.set_major_formatter(ScalarFormatter())
                ax_s.yaxis.get_major_formatter().set_scientific(False)

            # Event markers (vertical lines)
            for ev in events[:15]:
                date_min_s = min(real["date"].min(), synth["date"].min())
                date_max_s = max(real["date"].max(), synth["date"].max())
                if ev["date"] < date_min_s or ev["date"] > date_max_s:
                    continue
                idx = (real["date"] - ev["date"]).abs().idxmin()
                event_date = real.loc[idx, "date"]
                if ev["direction"] > 0.15:
                    ecolor = "#2E7D32"
                elif ev["direction"] < -0.15:
                    ecolor = "#C62828"
                else:
                    ecolor = "#757575"
                ax_s.axvline(
                    event_date, color=ecolor, alpha=0.2,
                    linewidth=0.8, linestyle=":",
                )

            ax_s.set_ylabel(f"{col_name.title()} ($)", fontsize=11)
            ax_s.legend(loc="upper left", fontsize=8, ncol=2)
            ax_s.grid(True, alpha=0.3, linestyle="--", which="both" if use_log else "major")

            if row_idx == 0:
                scale_label = " (log scale)" if use_log else ""
                ax_s.set_title(
                    f"{ticker} — Real vs Synthetic OHLCV Envelope{scale_label}"
                    if ticker else f"Real vs Synthetic OHLCV Envelope{scale_label}",
                    fontsize=14, fontweight="bold",
                )

        # Volume subplot
        if has_volume:
            ax_vol = axes_s[-1]
            ax_vol.bar(
                real["date"], real["volume"],
                width=1.5, color="#1565C0", alpha=0.4, label="Real Vol",
            )
            if has_samples and len(samples) > 1:
                vol_pivot = synth.pivot_table(
                    index="date", columns="sample",
                    values="volume", aggfunc="first",
                ).sort_index()
                mean_vol = vol_pivot.mean(axis=1)
                ax_vol.bar(
                    mean_vol.index, mean_vol.values,
                    width=1.5, color="#FF5722", alpha=0.3, label="Synth Mean Vol",
                )
            else:
                s0 = synth.sort_values("date")
                ax_vol.bar(
                    s0["date"], s0["volume"],
                    width=1.5, color="#FF5722", alpha=0.3, label="Synth Vol",
                )
            ax_vol.set_ylabel("Volume", fontsize=10)
            ax_vol.legend(fontsize=8)
            ax_vol.grid(True, alpha=0.3, linestyle="--")

        axes_s[-1].set_xlabel("Date", fontsize=11)
        axes_s[-1].xaxis.set_major_locator(mdates.YearLocator())
        axes_s[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        plt.setp(axes_s[-1].get_xticklabels(), rotation=45, ha="right")

        fig_s.subplots_adjust(hspace=0.08)
        path_s = os.path.join(output_dir, f"real_vs_synth{suffix}_{ticker}.png")
        fig_s.savefig(path_s, dpi=200, bbox_inches="tight")
        plt.close(fig_s)
        scale_tag = "log" if use_log else "linear"
        print(f"    ✓ Real vs Synthetic ({scale_tag}): {path_s}")

    _draw_real_vs_synth(use_log=False, suffix="")
    _draw_real_vs_synth(use_log=True, suffix="_log")

    # ────────────────────────────────────────────────────────────────
    # PLOT 3: Generation Factor Schedule (box-style, like browser)
    # ────────────────────────────────────────────────────────────────
    # Determine the source of factors for the box plot.
    # Priority: generation_factors > validated_events > raw analyses
    box_factors: Optional[List[Dict]] = None
    box_label = ""
    if generation_factors is not None:
        box_factors = generation_factors
        box_label = "Generation Factors"
    elif validated_events is not None:
        box_factors = validated_events
        box_label = "Validated Factors"

    if box_factors:
        _plot_factor_boxes(
            real, box_factors, ticker=ticker,
            output_dir=output_dir, label=box_label,
        )


def _plot_factor_boxes(
    real_df: pd.DataFrame,
    factor_configs: List[Dict],
    ticker: str = "",
    output_dir: str = "plots",
    label: str = "Generation Factors",
) -> None:
    """Create a PNG with coloured time-span boxes for active factors.

    Similar to the Dash browser view but as a static matplotlib plot.
    """
    real = real_df.copy()
    real["date"] = pd.to_datetime(real["date"])
    real = real.sort_values("date")

    fig, ax = plt.subplots(figsize=(22, 8))
    ax.plot(
        real["date"], real["close"],
        color="#1565C0", linewidth=1.4, alpha=0.95,
        label="Real Close", zorder=10,
    )

    y_min, y_max = real["close"].min(), real["close"].max()
    y_range = y_max - y_min

    dir_colors = {
        "bearish": ("#C62828", "#FFCDD2"),
        "bullish": ("#2E7D32", "#C8E6C9"),
        "neutral": ("#F9A825", "#FFF9C4"),
    }

    # Sort factors by start_date for consistent staggering
    sorted_factors = sorted(
        factor_configs,
        key=lambda f: f.get("start_date", ""),
    )

    n_factors = len(sorted_factors)
    for i, fc in enumerate(sorted_factors):
        direction = fc.get("direction", "neutral")
        if isinstance(direction, (int, float)):
            direction = (
                "bullish" if direction > 0.15
                else "bearish" if direction < -0.15
                else "neutral"
            )
        edge_col, fill_col = dir_colors.get(direction, dir_colors["neutral"])
        prob = fc.get("probability", 1.0)
        alpha = 0.10 + 0.20 * prob  # range 0.10 – 0.30

        start = pd.Timestamp(fc["start_date"])
        end = pd.Timestamp(fc["end_date"])

        # Draw shaded box spanning full price range
        ax.axvspan(
            start, end,
            alpha=alpha, color=fill_col, zorder=1,
        )
        # Draw left/right border lines
        ax.axvline(start, color=edge_col, alpha=0.3, linewidth=0.5, zorder=2)
        ax.axvline(end, color=edge_col, alpha=0.3, linewidth=0.5, zorder=2)

        # Label: stagger vertically to reduce overlap
        mid = start + (end - start) / 2
        text_y_frac = 0.92 - 0.06 * (i % 8)
        text_y = y_min + text_y_frac * y_range

        name = fc.get("name", f"Factor {i}")
        name_display = name.replace("_", " ").title()

        ax.annotate(
            f"{name_display} ({prob:.0%})",
            xy=(mid, text_y),
            fontsize=6, color=edge_col, fontweight="bold",
            ha="center", va="center",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white", edgecolor=edge_col,
                alpha=0.8, linewidth=0.5,
            ),
            zorder=15,
        )

    ax.set_title(
        f"{ticker} — {label} (Box Schedule)"
        if ticker else f"{label} (Box Schedule)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("Close Price ($)", fontsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=[4, 7, 10]))
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.grid(True, alpha=0.3, linestyle="--")

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#C8E6C9", edgecolor="#2E7D32",
              label="Bullish"),
        Patch(facecolor="#FFCDD2", edgecolor="#C62828",
              label="Bearish"),
        Patch(facecolor="#FFF9C4", edgecolor="#F9A825",
              label="Neutral"),
        plt.Line2D([0], [0], color="#1565C0", linewidth=2,
                   label="Real Close"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    plt.tight_layout()

    safe_label = label.lower().replace(" ", "_")
    path = os.path.join(output_dir, f"{safe_label}_{ticker}.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"    ✓ Factor box schedule plot: {path}")

