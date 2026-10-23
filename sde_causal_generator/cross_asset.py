# -*- coding: utf-8 -*-
"""
Cross-asset correlations — Improvement #9.

Captures and re-imposes the return-level correlation structure
between multiple tickers so that synthetic samples preserve
realistic co-movement patterns.

Overview
--------
1. **Estimation** – from real multi-ticker returns, compute a
   Pearson correlation matrix across tickers (using close returns
   as the representative price signal).
2. **Cholesky decomposition** – decompose the correlation matrix
   into L L^T so that correlated noise can be generated as L·ε.
3. **Application** – after generating independent trajectories per
   ticker, post-hoc "re-correlate" the innovations, or provide L
   to the SDE sampler upstream.

This module can be used:
- *Upstream* (provided to `sample_neural_sde` or
  `ImpactDrivenGenerator` as `corr_cholesky`) so that noise is
  correlated during generation.
- *Downstream* (applied as a post-processing step that reshuffles
  returns across tickers to match the target correlation matrix).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# Estimation
# ══════════════════════════════════════════════════════════════════════


def estimate_cross_asset_correlation(
    ticker_returns: Dict[str, np.ndarray],
    method: str = "pearson",
) -> Tuple[np.ndarray, List[str]]:
    """Estimate a correlation matrix across tickers.

    Parameters
    ----------
    ticker_returns : dict[str, ndarray]
        Mapping from ticker name to (T,) array of daily log-returns.
        All arrays must share the same date alignment and length.
    method : str
        'pearson' or 'spearman'.

    Returns
    -------
    corr_matrix : (N, N)  — correlation matrix
    tickers : list[str]  — ordered ticker names
    """
    tickers = sorted(ticker_returns.keys())
    N = len(tickers)
    if N < 2:
        return np.eye(1), tickers

    # Build DataFrame
    df = pd.DataFrame({t: ticker_returns[t] for t in tickers})

    if method == "spearman":
        corr = df.corr(method="spearman").values
    else:
        corr = df.corr(method="pearson").values

    # Ensure positive-semi-definiteness (numerical stability)
    corr = _nearest_psd(corr)

    return corr, tickers


def extract_returns_from_dataframes(
    ticker_dfs: Dict[str, pd.DataFrame],
    price_col: str = "close",
) -> Dict[str, np.ndarray]:
    """Extract log-return series from per-ticker DataFrames.

    Parameters
    ----------
    ticker_dfs : dict[str, DataFrame]
        Each DF must have a ``date`` column and the price column.
    price_col : str

    Returns
    -------
    ticker_returns : dict[str, ndarray]
        Log-returns aligned to the common date range.
    """
    merged = None
    for t, df in ticker_dfs.items():
        series = df.set_index("date")[[price_col]].rename(
            columns={price_col: t}
        )
        if merged is None:
            merged = series
        else:
            merged = merged.join(series, how="inner")

    if merged is None or len(merged) < 2:
        return {}

    log_ret = np.log(merged / merged.shift(1)).dropna()
    return {t: log_ret[t].values for t in log_ret.columns}


# ══════════════════════════════════════════════════════════════════════
# Cholesky + PSD helpers
# ══════════════════════════════════════════════════════════════════════


def cholesky_factor(corr_matrix: np.ndarray) -> np.ndarray:
    """Compute the lower Cholesky factor of a correlation matrix.

    Parameters
    ----------
    corr_matrix : (N, N)

    Returns
    -------
    L : (N, N). Satisfies  L @ L.T = corr_matrix.
    """
    corr_matrix = _nearest_psd(corr_matrix)
    return np.linalg.cholesky(corr_matrix)


def _nearest_psd(C: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the nearest PSD matrix."""
    C = (C + C.T) / 2.0
    eigvals, eigvecs = np.linalg.eigh(C)
    eigvals = np.maximum(eigvals, eps)
    C_psd = (eigvecs * eigvals) @ eigvecs.T
    # Re-normalise to correlation
    d = np.sqrt(np.diag(C_psd))
    d[d == 0] = 1.0
    C_psd = C_psd / np.outer(d, d)
    np.fill_diagonal(C_psd, 1.0)
    return C_psd


# ══════════════════════════════════════════════════════════════════════
# DCC-GARCH  — Improvement F6
# ══════════════════════════════════════════════════════════════════════


def estimate_dcc_garch(
    ticker_returns: Dict[str, np.ndarray],
    garch_p: int = 1,
    garch_q: int = 1,
    dcc_a: float = 0.0,
    dcc_b: float = 0.0,
) -> "DccResult":
    """Estimate time-varying correlations using DCC-GARCH(1,1).

    Steps
    -----
    1. Fit univariate GARCH(p,q) per ticker → standardised residuals.
    2. Run DCC dynamics on residuals → time-varying R_t.
    3. If ``dcc_a`` and ``dcc_b`` are both 0 (default), they are
       estimated by maximum-likelihood on a coarse grid search.

    Parameters
    ----------
    ticker_returns : dict[str, ndarray]
        Mapping from ticker to ``(T,)`` daily log-returns.
    garch_p, garch_q : int
        GARCH lag orders per univariate model.
    dcc_a, dcc_b : float
        DCC parameters.  If both zero, they are estimated.

    Returns
    -------
    result : DccResult
        Named tuple with fields ``corr_matrices  (T, N, N)``,
        ``tickers``, ``unconditional_corr  (N, N)``,
        ``dcc_params  (a, b)``, ``cond_vol  (T, N)``.
    """
    from collections import namedtuple

    DccResult = namedtuple(
        "DccResult",
        ["corr_matrices", "tickers", "unconditional_corr",
         "dcc_params", "cond_vol"],
    )

    tickers = sorted(ticker_returns.keys())
    N = len(tickers)

    if N < 2:
        T = len(next(iter(ticker_returns.values())))
        return DccResult(
            corr_matrices=np.ones((T, 1, 1)),
            tickers=tickers,
            unconditional_corr=np.eye(1),
            dcc_params=(0.0, 0.0),
            cond_vol=np.zeros((T, 1)),
        )

    T = min(len(ticker_returns[t]) for t in tickers)

    # ── Step 1: Univariate GARCH ─────────────────────────────────────
    try:
        from arch import arch_model as _arch_model
    except ImportError:
        print("    ⚠ `arch` package not installed — falling back to Pearson.")
        corr, tks = estimate_cross_asset_correlation(ticker_returns)
        return DccResult(
            corr_matrices=np.tile(corr, (T, 1, 1)),
            tickers=tks,
            unconditional_corr=corr,
            dcc_params=(0.0, 0.0),
            cond_vol=np.ones((T, N)) * 0.01,
        )

    resids = np.zeros((T, N))
    cond_vol = np.zeros((T, N))

    for i, t in enumerate(tickers):
        ret = ticker_returns[t][:T] * 100          # scale for GARCH stability
        try:
            am = _arch_model(ret, vol="Garch", p=garch_p, q=garch_q,
                             dist="normal", mean="Zero", rescale=False)
            result = am.fit(disp="off", show_warning=False)
            cv = result.conditional_volatility
            cv = np.where(cv < 1e-8, 1e-8, cv)
            resids[:, i] = result.resid / cv
            cond_vol[:, i] = cv / 100.0            # un-scale
        except Exception:
            # fallback: standardised returns
            std = np.std(ret)
            std = std if std > 1e-8 else 1e-8
            resids[:, i] = ret / std
            cond_vol[:, i] = std / 100.0

    # ── Step 2: DCC dynamics ──────────────────────────────────────────
    Q_bar = np.corrcoef(resids.T)
    Q_bar = _nearest_psd(Q_bar)

    # Estimate (a, b) via grid search if not provided
    if dcc_a == 0.0 and dcc_b == 0.0:
        dcc_a, dcc_b = _dcc_grid_search(resids, Q_bar)

    # Run DCC
    corr_matrices = np.zeros((T, N, N))
    Q_t = Q_bar.copy()

    for t in range(T):
        if t > 0:
            eps = resids[t - 1 : t, :].T                            # (N, 1)
            Q_t = (1.0 - dcc_a - dcc_b) * Q_bar + dcc_a * (eps @ eps.T) + dcc_b * Q_t

        # Normalise Q_t → correlation
        diag_q = np.diag(Q_t).clip(1e-8)
        D_inv = np.diag(1.0 / np.sqrt(diag_q))
        R_t = D_inv @ Q_t @ D_inv
        np.fill_diagonal(R_t, 1.0)
        R_t = _nearest_psd(R_t)
        corr_matrices[t] = R_t

    return DccResult(
        corr_matrices=corr_matrices,
        tickers=tickers,
        unconditional_corr=Q_bar,
        dcc_params=(dcc_a, dcc_b),
        cond_vol=cond_vol,
    )


def _dcc_grid_search(
    resids: np.ndarray,
    Q_bar: np.ndarray,
    n_grid: int = 10,
) -> tuple:
    """Coarse grid search for DCC (a, b) maximising quasi-likelihood."""
    T, N = resids.shape
    best_ll = -np.inf
    best_a, best_b = 0.05, 0.90

    for ia in range(1, n_grid):
        for ib in range(1, n_grid):
            a = ia * 0.02
            b = ib * 0.10
            if a + b >= 1.0:
                continue
            ll = _dcc_loglik(resids, Q_bar, a, b)
            if ll > best_ll:
                best_ll = ll
                best_a, best_b = a, b

    return best_a, best_b


def _dcc_loglik(
    resids: np.ndarray,
    Q_bar: np.ndarray,
    a: float,
    b: float,
) -> float:
    """Quasi-log-likelihood for DCC parameters."""
    T, N = resids.shape
    Q_t = Q_bar.copy()
    ll = 0.0

    for t in range(T):
        if t > 0:
            eps = resids[t - 1 : t, :].T
            Q_t = (1 - a - b) * Q_bar + a * (eps @ eps.T) + b * Q_t

        diag_q = np.diag(Q_t).clip(1e-8)
        D_inv = np.diag(1.0 / np.sqrt(diag_q))
        R_t = D_inv @ Q_t @ D_inv
        np.fill_diagonal(R_t, 1.0)
        R_t = _nearest_psd(R_t)

        try:
            sign, logdet = np.linalg.slogdet(R_t)
            if sign <= 0:
                continue
            R_inv = np.linalg.inv(R_t)
            e = resids[t]
            ll += -0.5 * (logdet + e @ R_inv @ e - e @ e)
        except np.linalg.LinAlgError:
            continue

    return ll


def recorrelate_returns_dynamic(
    independent_returns: Dict[str, np.ndarray],
    cholesky_sequence: np.ndarray,
    tickers: List[str],
) -> Dict[str, np.ndarray]:
    """Re-correlate returns using *time-varying* Cholesky factors.

    Parameters
    ----------
    independent_returns : dict[str, (T,) ndarray]
    cholesky_sequence : (T, N, N) — per-step lower Cholesky factors
    tickers : list[str]

    Returns
    -------
    correlated_returns : dict[str, (T,) ndarray]
    """
    N = len(tickers)
    if N < 2:
        return independent_returns

    T = min(len(independent_returns[t]) for t in tickers)
    Z = np.column_stack([independent_returns[t][:T] for t in tickers])

    means = Z.mean(axis=0, keepdims=True)
    stds = Z.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    Z_white = (Z - means) / stds

    Z_out = np.zeros_like(Z_white)
    for t in range(T):
        t_idx = min(t, cholesky_sequence.shape[0] - 1)
        Z_out[t] = Z_white[t] @ cholesky_sequence[t_idx].T

    Z_out = Z_out * stds + means
    return {t: Z_out[:, i] for i, t in enumerate(tickers)}


# ══════════════════════════════════════════════════════════════════════
# Post-hoc re-correlation
# ══════════════════════════════════════════════════════════════════════


def recorrelate_returns(
    independent_returns: Dict[str, np.ndarray],
    target_cholesky: np.ndarray,
    tickers: List[str],
) -> Dict[str, np.ndarray]:
    """Re-correlate independently generated returns to match the
    target correlation structure.

    Steps:
        1. Whiten each ticker's returns (standardise to unit variance).
        2. Stack → (T, N).
        3. Multiply by the Cholesky factor L (so Cov = L L^T).
        4. Rescale back to original scale.

    Parameters
    ----------
    independent_returns : dict[str, (T,) ndarray]
    target_cholesky : (N, N) — lower Cholesky factor
    tickers : list[str] — ordered ticker names

    Returns
    -------
    correlated_returns : dict[str, (T,) ndarray]
    """
    N = len(tickers)
    if N < 2:
        return independent_returns

    # Collect
    T = min(len(independent_returns[t]) for t in tickers)
    Z = np.column_stack([independent_returns[t][:T] for t in tickers])

    # Standardise per ticker
    means = Z.mean(axis=0, keepdims=True)
    stds = Z.std(axis=0, keepdims=True)
    stds[stds == 0] = 1.0
    Z_white = (Z - means) / stds

    # Apply target correlation
    Z_corr = Z_white @ target_cholesky.T

    # Restore scale
    Z_out = Z_corr * stds + means

    return {t: Z_out[:, i] for i, t in enumerate(tickers)}


def recorrelate_prices(
    independent_prices: Dict[str, np.ndarray],
    target_cholesky: np.ndarray,
    tickers: List[str],
) -> Dict[str, np.ndarray]:
    """Re-correlate independently generated *price* series.

    Converts to returns → re-correlates → converts back.

    Parameters
    ----------
    independent_prices : dict[str, (T,) ndarray]
    target_cholesky : (N, N)
    tickers : list[str]

    Returns
    -------
    correlated_prices : dict[str, (T,) ndarray]
    """
    N = len(tickers)
    if N < 2:
        return independent_prices

    # To log-returns
    returns = {}
    init_prices = {}
    for t in tickers:
        p = np.asarray(independent_prices[t], dtype=np.float64)
        p = np.clip(p, 1e-8, None)
        init_prices[t] = p[0]
        returns[t] = np.diff(np.log(p))

    # Re-correlate
    corr_returns = recorrelate_returns(returns, target_cholesky, tickers)

    # Back to prices
    result = {}
    for t in tickers:
        lr = corr_returns[t]
        cum = np.concatenate([[0.0], np.cumsum(lr)])
        result[t] = init_prices[t] * np.exp(cum)

    return result
