# -*- coding: utf-8 -*-
"""
Factor Relevance Scoring — Improvement F4.

Validates LLM-extracted factors by measuring their statistical
association with real returns, producing a relevance score per
factor that can be used to prune noise factors before FIN training.

Three scoring methods are combined:

1. **Granger-causality p-value** — does factor presence Granger-cause
   returns at lag 1..5?  Score = 1 − p.
2. **Point-biserial correlation** — correlation between binary factor
   presence and next-day absolute returns.
3. **Mutual information** — non-linear association between factor
   presence and discretised returns (5 bins).

The composite score is a weighted average of the three.

Usage
-----
::

    from sde_causal_generator.factor_relevance import (
        score_factors,
        prune_irrelevant_factors,
    )

    scores = score_factors(presence, log_returns, factor_names)
    presence2, names2 = prune_irrelevant_factors(
        presence, factor_names, scores, min_score=0.15,
    )
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ══════════════════════════════════════════════════════════════════════
# Individual scoring methods
# ══════════════════════════════════════════════════════════════════════


def _granger_score(
    presence_col: np.ndarray,
    returns_col: np.ndarray,
    max_lag: int = 5,
) -> float:
    """Granger-causality score (1 − min p-value across lags).

    Uses statsmodels OLS-based Granger test.
    Returns 0.0 when the test cannot be run.
    """
    try:
        from statsmodels.tsa.stattools import grangercausalitytests
    except ImportError:
        return 0.0

    T = min(len(presence_col), len(returns_col))
    if T < max_lag + 20:
        return 0.0

    data = np.column_stack([returns_col[:T], presence_col[:T]])
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            results = grangercausalitytests(data, maxlag=max_lag, verbose=False)
        min_p = min(
            results[lag][0]["ssr_ftest"][1]
            for lag in range(1, max_lag + 1)
        )
        # STAT-2: Bonferroni correction for multiple comparisons
        # across lags.  Without correction, min(p) over 5 lags
        # inflates false-positive rate ~5x.
        corrected_p = min(min_p * max_lag, 1.0)
        return float(np.clip(1.0 - corrected_p, 0.0, 1.0))
    except Exception:
        return 0.0


def _pointbiserial_score(
    presence_col: np.ndarray,
    returns_col: np.ndarray,
) -> float:
    """Point-biserial correlation between binary presence and |returns|.

    Returns the absolute correlation value (0–1).
    """
    from scipy import stats as sp_stats

    T = min(len(presence_col), len(returns_col))
    if T < 20:
        return 0.0

    binary = (presence_col[:T] > 0).astype(float)
    abs_ret = np.abs(returns_col[:T])

    # Need at least 2 values in each group
    # DISC-2: reduce threshold from 5 to 2 to protect rare events
    if binary.sum() < 2 or (1 - binary).sum() < 2:
        return 0.0

    try:
        corr, _ = sp_stats.pointbiserialr(binary, abs_ret)
        return float(np.clip(abs(corr), 0.0, 1.0))
    except Exception:
        return 0.0


def _mutual_info_score(
    presence_col: np.ndarray,
    returns_col: np.ndarray,
    n_bins: int = 5,
) -> float:
    """Mutual information between factor presence and discretised returns.

    Returns normalised MI in [0, 1].
    """
    T = min(len(presence_col), len(returns_col))
    if T < 20:
        return 0.0

    binary = (presence_col[:T] > 0).astype(int)

    # Discretise returns into quantile bins
    try:
        ret_bins = pd.qcut(returns_col[:T], q=n_bins, labels=False,
                           duplicates="drop")
    except Exception:
        return 0.0

    # Compute MI via contingency table
    try:
        from sklearn.metrics import normalized_mutual_info_score
        nmi = normalized_mutual_info_score(binary, ret_bins)
        return float(np.clip(nmi, 0.0, 1.0))
    except ImportError:
        # Fallback: manual computation
        return _manual_nmi(binary, ret_bins)


def _manual_nmi(x: np.ndarray, y: np.ndarray) -> float:
    """Manual normalised mutual information (no sklearn dependency)."""
    from collections import Counter

    n = len(x)
    if n == 0:
        return 0.0

    # Joint and marginal distributions
    joint = Counter(zip(x, y))
    px = Counter(x)
    py = Counter(y)

    mi = 0.0
    for (xi, yi), nxy in joint.items():
        pxy = nxy / n
        pxi = px[xi] / n
        pyi = py[yi] / n
        if pxy > 0 and pxi > 0 and pyi > 0:
            mi += pxy * np.log(pxy / (pxi * pyi))

    # Normalise by max(H(X), H(Y))
    hx = -sum((c / n) * np.log(c / n) for c in px.values() if c > 0)
    hy = -sum((c / n) * np.log(c / n) for c in py.values() if c > 0)
    norm = max(hx, hy)
    if norm < 1e-10:
        return 0.0
    return float(np.clip(mi / norm, 0.0, 1.0))


# ══════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════


def score_factors(
    presence: np.ndarray,
    log_returns: np.ndarray,
    factor_names: List[str],
    weights: Tuple[float, float, float] = (0.4, 0.3, 0.3),
    price_col: int = 3,
) -> Dict[str, Dict[str, float]]:
    """Score each factor's relevance to real returns.

    Parameters
    ----------
    presence : (T, K) — binary factor presence matrix
    log_returns : (T-1, M) or (T, M) — real daily log-returns
    factor_names : list[str] — K factor names
    weights : (w_granger, w_pb, w_mi) — scoring weights (sum to 1)
    price_col : int — which return column to use (default: close=3)

    Returns
    -------
    scores : dict[str, dict] — per-factor scores::

        {
            "factor_name": {
                "granger": 0.85,
                "pointbiserial": 0.12,
                "mutual_info": 0.07,
                "composite": 0.45,
            },
            ...
        }
    """
    T_pres, K = presence.shape
    col = min(price_col, log_returns.shape[1] - 1)
    ret = log_returns[:, col]

    # Align lengths
    T = min(T_pres, len(ret))
    pres = presence[:T]
    ret = ret[:T]

    w_g, w_p, w_m = weights

    # First pass: collect raw scores for each method
    raw_g, raw_pb, raw_mi = [], [], []
    factor_raw: Dict[str, Tuple[float, float, float]] = {}
    for k in range(K):
        name = factor_names[k] if k < len(factor_names) else f"factor_{k}"
        p_col = pres[:, k]

        g = _granger_score(p_col, ret)
        pb = _pointbiserial_score(p_col, ret)
        mi = _mutual_info_score(p_col, ret)
        raw_g.append(g)
        raw_pb.append(pb)
        raw_mi.append(mi)
        factor_raw[name] = (g, pb, mi)

    # STAT-3: rank-normalize each score before combining to prevent
    # Granger (bimodal ~0 or ~1) from dominating point-biserial
    # (|r| < 0.05) and MI (< 0.1).
    def _rank_normalize(values):
        """Convert values to rank-based scores in [0, 1]."""
        n = len(values)
        if n <= 1:
            return [0.5] * n
        order = np.argsort(values)
        ranks = np.empty(n)
        ranks[order] = np.linspace(0, 1, n)
        return ranks.tolist()

    rn_g = _rank_normalize(raw_g)
    rn_pb = _rank_normalize(raw_pb)
    rn_mi = _rank_normalize(raw_mi)

    scores: Dict[str, Dict[str, float]] = {}
    for k in range(K):
        name = factor_names[k] if k < len(factor_names) else f"factor_{k}"
        g, pb, mi = factor_raw[name]
        composite = w_g * rn_g[k] + w_p * rn_pb[k] + w_m * rn_mi[k]

        scores[name] = {
            "granger": round(g, 4),
            "pointbiserial": round(pb, 4),
            "mutual_info": round(mi, 4),
            "composite": round(composite, 4),
        }

    return scores


def prune_irrelevant_factors(
    presence: np.ndarray,
    factor_names: List[str],
    scores: Dict[str, Dict[str, float]],
    min_score: float = 0.10,
    max_factors: int = 0,
) -> Tuple[np.ndarray, List[str]]:
    """Remove factors below a relevance threshold.

    Parameters
    ----------
    presence : (T, K)
    factor_names : list[str]
    scores : output of ``score_factors()``
    min_score : float — minimum composite score to keep
    max_factors : int — keep at most this many (0 = no limit)

    Returns
    -------
    presence_pruned : (T, K')
    names_pruned : list[str]
    """
    # Build (name, index, composite_score) list
    scored = []
    for k, name in enumerate(factor_names):
        s = scores.get(name, {}).get("composite", 0.0)
        scored.append((name, k, s))

    # Filter by min_score
    filtered = [(n, k, s) for n, k, s in scored if s >= min_score]

    # Sort by composite score descending
    filtered.sort(key=lambda x: x[2], reverse=True)

    # Cap to max_factors
    if max_factors > 0 and len(filtered) > max_factors:
        filtered = filtered[:max_factors]

    if not filtered:
        # Keep at least top-5 even if below threshold
        scored.sort(key=lambda x: x[2], reverse=True)
        filtered = scored[:min(5, len(scored))]

    # Rebuild presence matrix with selected factors
    idx = sorted([k for _, k, _ in filtered])
    names_out = [factor_names[k] for k in idx]
    presence_out = presence[:, idx]

    return presence_out, names_out


def print_relevance_report(
    scores: Dict[str, Dict[str, float]],
    top_n: int = 20,
) -> None:
    """Print a formatted relevance report to stdout."""
    items = sorted(
        scores.items(),
        key=lambda x: x[1]["composite"],
        reverse=True,
    )

    print("\n" + "═" * 60)
    print("  FACTOR RELEVANCE SCORES (F4)")
    print("═" * 60)
    print(f"  {'Factor':<30s} {'Granger':>8s} {'PtBis':>8s} "
          f"{'MI':>8s} {'Score':>8s}")
    print("  " + "─" * 56)

    for name, sc in items[:top_n]:
        print(f"  {name:<30s} {sc['granger']:8.4f} "
              f"{sc['pointbiserial']:8.4f} {sc['mutual_info']:8.4f} "
              f"{sc['composite']:8.4f}")

    if len(items) > top_n:
        print(f"  ... ({len(items) - top_n} more factors omitted)")

    # Summary stats
    composites = [s["composite"] for s in scores.values()]
    print(f"\n  Total factors: {len(scores)}")
    print(f"  Score range: [{min(composites):.4f}, {max(composites):.4f}]")
    print(f"  Mean score:  {np.mean(composites):.4f}")
    above_10 = sum(1 for c in composites if c >= 0.10)
    print(f"  Factors ≥ 0.10: {above_10}")
    print("═" * 60)
