# -*- coding: utf-8 -*-
"""Settings and helpers shared by the B3 tests, the five-seed estimates and the sign-recovery check."""
from __future__ import annotations

import copy

import numpy as np

# TFT refits in the rolling-origin test (the generator's own runs use 300 epochs, configs/*.yaml)
EPOCHS = 200
BATCH = 64
LR = 1e-3
PATIENCE = 30
FEATS = ["open", "high", "low", "close", "volume"]

MIN_WINDOW_DAYS = 5   # active days a factor needs on each side of an origin
D_NULL = 1000         # permutations in agreement_stats


def _llm_directions(analyses, factor_names):
    idx = {n: i for i, n in enumerate(factor_names)}
    sums = np.zeros(len(factor_names))
    counts = np.zeros(len(factor_names))
    for a in analyses:
        for f in a.factors:
            if f.name in idx:
                sums[idx[f.name]] += f.direction
                counts[idx[f.name]] += 1
    return np.where(counts > 0, sums / np.maximum(counts, 1), 0.0).astype(np.float32)


def _permute_windows(analyses, rng):
    """Shuffle the (start_date, end_date) slots among the analyses."""
    shuffled = [copy.deepcopy(a) for a in analyses]
    slots = [(a.start_date, a.end_date) for a in analyses]
    perm = rng.permutation(len(slots))
    for a, j in zip(shuffled, perm):
        a.start_date, a.end_date = slots[j]
    return shuffled


def agreement_stats(rbar: np.ndarray, d: np.ndarray, rng: np.random.RandomState):
    """Sign agreement + Spearman + permutation null over eligible factors."""
    from scipy.stats import spearmanr
    elig = (~np.isnan(rbar)) & (np.abs(d) > 1e-12)
    n = int(elig.sum())
    if n < 5:
        return None
    rb, dd = rbar[elig], d[elig]
    agree = float((np.sign(rb) == np.sign(dd)).mean())
    rho = float(spearmanr(rb, dd).statistic)
    null_agrees = np.empty(D_NULL)
    for i in range(D_NULL):
        null_agrees[i] = (np.sign(rb) == np.sign(dd[rng.permutation(n)])).mean()
    p = float((null_agrees >= agree).mean())
    return {"n_factors": n, "agreement": round(agree, 4),
            "spearman": round(rho, 4),
            "null_mean": round(float(null_agrees.mean()), 4),
            "p_perm": round(p, 4)}
