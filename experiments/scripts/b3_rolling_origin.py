#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 rolling origin: out-of-sample sign test on the corrected pipeline.

Criteria v2 and v2.1. Differences from the first version of the test:
  * impact network: impact_estimator="ablation" (daily units, no mu),
    data-sign variant (llm_directions=None) by default; 5 training seeds,
    d = median over seeds (sign = majority sign);
  * verdict arms: control vs window_permuted only (direction_shuffled is
    degenerate without a prior: identical to control);
  * presence on day t is paired with the return from t to t+1 (training
    convention), both for eligibility and for the realized excess return;
  * baseline factors (market_noise, noise_residual, >80% of training days)
    are excluded from eligibility;
  * reported, not in the verdict: a label-only predictor (training-period mean
    LLM direction) against the same realized excess returns;
  * LLM-sign variant (secondary): the original three arms, including
    direction_shuffled.
Factor set: no pruning (every extracted factor is kept; asserted), so the set
does not depend on test-year labels; eligibility uses training days only.
Origins, eligibility thresholds, null, epochs (200) and verdict rule as in v1.

Secondary (label-information test): --estimator ols replaces
the 5-seed TFT median by OLS of the close return t -> t+1 on an intercept and
the presence indicators of every factor, fitted on the days before the origin
(each arm on its own presence); factors with no active training day get 0.

Usage:
  TRACE_RUN=<run> python b3_rolling_origin.py --origin 2021-01-01 [--variant data] [--estimator ols]
  TRACE_RUN=<run> python b3_rolling_origin.py --aggregate [--variant data]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RUN_TAG, build_presence, load_ticker_artifacts, save_json, ticker_df, tickers_28  # noqa: E402
from shared import FEATS, EPOCHS, BATCH, LR, PATIENCE, _llm_directions, _permute_windows  # noqa: E402
from shared import agreement_stats, MIN_WINDOW_DAYS  # noqa: E402

ORIGINS = ["2021-01-01", "2022-01-01", "2023-01-01"]
SEEDS = (0, 1, 2, 3, 4)
ARM_SEED = 7
MIN_IS_DAYS = 5
MIN_FACTORS = 5
ALPHA = 0.01
BASELINE_NAMES = {"market_noise", "noise_residual"}


ESTIMATOR = "tft"


def out_dir(variant):
    return OUT / (f"b3_rolling_origin__{RUN_TAG}__{variant}" + ("" if ESTIMATOR == "tft" else f"__{ESTIMATOR}"))


def ols_signs_input(P_train, close_train):
    """Secondary OLS: y = r(t -> t+1), X = [1, 1[P_k(t) > 0]], days of P_train."""
    r = np.diff(np.log(np.clip(close_train.astype(np.float64), 1e-8, None)))
    A = (P_train[:-1] > 0).astype(np.float64)
    keep = A.sum(axis=0) > 0
    coef = np.zeros(A.shape[1])
    c = np.linalg.lstsq(np.column_stack([np.ones(len(r)), A[:, keep]]), r, rcond=None)[0][1:]
    coef[keep] = c
    return coef


def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def realized_excess(P_eval, close, n_is, n_end, eligible):
    r = np.diff(np.log(np.clip(close.astype(np.float64), 1e-8, None)))   # r[j]: day j -> j+1
    r_t = r[n_is:n_end - 1]
    P_t = P_eval[n_is:n_end - 1]
    overall = r_t.mean()
    rbar = np.full(P_eval.shape[1], np.nan)
    for k in np.where(eligible)[0]:
        mask = P_t[:, k] > 0
        if int(mask.sum()) >= MIN_WINDOW_DAYS:
            rbar[k] = r_t[mask].mean() - overall
    return rbar


def run_ticker(ticker, origin, variant, device):
    from sde_causal_generator.factor_impact_network import train_impact_network
    im0, _, analyses = load_ticker_artifacts(ticker)
    names = im0.factor_names
    df = ticker_df(ticker)
    prices = np.clip(df[FEATS].values.astype(np.float64), 1e-8, None)
    dates = pd.to_datetime(df["date"])
    o = pd.Timestamp(origin)
    n_is = int((dates < o).sum())
    n_end = int((dates < o + pd.DateOffset(years=1)).sum())

    an_train = [a for a in analyses if pd.Timestamp(a.end_date) < o]
    rng_arm = np.random.RandomState(ARM_SEED)
    P_true = build_presence(an_train, df, names)
    dirs_true = _llm_directions(an_train, names)
    P_perm = build_presence(_permute_windows(an_train, rng_arm), df, names)
    dirs_shuf = dirs_true[rng_arm.permutation(len(dirs_true))]   # v1 RNG call order
    assert set(names) == {f.name for x in analyses for f in x.factors if f.magnitude > 0}, \
        f"{ticker}: factor set was pruned on full-period labels"
    P_eval = build_presence(analyses, df, names)

    occ_is = (P_true[:n_is] > 0).mean(axis=0)
    is_days = (P_true[:n_is] > 0).sum(axis=0)
    eligible = np.array([(n not in BASELINE_NAMES) and (occ_is[k] <= 0.8) and (is_days[k] >= MIN_IS_DAYS)
                         for k, n in enumerate(names)])
    rbar = realized_excess(P_eval, prices[:, 3], n_is, n_end, eligible)

    out = {"ticker": ticker, "origin": origin, "variant": variant,
           "n_eligible_factors": int((~np.isnan(rbar)).sum())}
    arms = [("control", P_true, dirs_true), ("window_permuted", P_perm, dirs_true)]
    if variant == "llm":
        arms.append(("direction_shuffled", P_true, dirs_shuf))
    for arm, P, dirs in arms:
        t0 = time.time()
        if ESTIMATOR == "ols":
            if arm == "direction_shuffled":
                continue                    # OLS has no direction prior
            d = ols_signs_input(P[:n_is], prices[:n_is, 3])
            out[arm] = {"metric": agreement_stats(rbar, d, np.random.RandomState(5000)),
                        "elapsed_s": round(time.time() - t0, 1)}
            continue
        ds = []
        for s in SEEDS:
            seed_all(s)
            _, im_new, _ = train_impact_network(
                factor_matrix=P[:n_is], price_data=prices[:n_is], factor_names=names,
                feature_names=FEATS, n_lags=10, n_epochs=EPOCHS, batch_size=BATCH,
                learning_rate=LR, val_fraction=0.15, device=device, patience=PATIENCE,
                architecture="tft", llm_directions=(dirs if variant == "llm" else None),
                impact_estimator="ablation")
            ds.append(im_new.base_impact[:, 3])
        D = np.stack(ds)
        d = np.median(D, axis=0)
        signs = np.sign(D)
        out[arm] = {"metric": agreement_stats(rbar, d, np.random.RandomState(5000)),
                    "seed_sign_agreement": float(np.abs(signs.mean(axis=0))[~np.isnan(rbar)].mean())
                    if (~np.isnan(rbar)).any() else None,
                    "elapsed_s": round(time.time() - t0, 1)}
    out["label_only"] = {"metric": agreement_stats(rbar, dirs_true.astype(float), np.random.RandomState(5000))}
    m = out["control"]["metric"] or {}
    print(f"  {ticker}/{origin[:4]}: control agree={m.get('agreement')} "
          f"wp={((out['window_permuted']['metric'] or {}).get('agreement'))} "
          f"label={((out['label_only']['metric'] or {}).get('agreement'))} "
          f"(n={m.get('n_factors')}, {out['control']['elapsed_s']}+{out['window_permuted']['elapsed_s']}s)", flush=True)
    return out


def aggregate(variant):
    from scipy.stats import wilcoxon
    od = out_dir(variant)
    per = {o: json.loads((od / f"origin_{o[:4]}.json").read_text())["per_ticker"]
           for o in ORIGINS if (od / f"origin_{o[:4]}.json").exists()}

    def agree(row, arm):
        m = (row.get(arm) or {}).get("metric") or {}
        return m["agreement"] if m and m.get("n_factors", 0) >= MIN_FACTORS else None

    rep = {"run": RUN_TAG, "variant": variant, "estimator": ESTIMATOR,
           "origins_present": sorted(per), "per_origin": {}, "pooled": {}}
    comps = ["window_permuted", "label_only"] + (["direction_shuffled"] if variant == "llm" and ESTIMATOR == "tft" else [])
    for comp in comps:
        by_t, pos = {}, 0
        for o, rows in per.items():
            diffs = [agree(r, "control") - agree(r, comp) for r in rows
                     if agree(r, "control") is not None and agree(r, comp) is not None]
            for r in rows:
                c, a = agree(r, "control"), agree(r, comp)
                if c is not None and a is not None:
                    by_t.setdefault(r["ticker"], []).append(c - a)
            med = float(np.median(diffs)) if diffs else None
            pos += int((med or 0) > 0)
            rep["per_origin"].setdefault(o, {})[f"control_minus_{comp}"] = {
                "n_tickers": len(diffs), "median_diff": med,
                "p_one_sided": float(wilcoxon(diffs, alternative="greater").pvalue)
                if len(diffs) >= 6 and any(diffs) else None}
        pooled = [float(np.mean(v)) for v in by_t.values() if len(v) >= 2]
        p = float(wilcoxon(pooled, alternative="greater").pvalue) if len(pooled) >= 6 and any(pooled) else None
        rep["pooled"][f"control_minus_{comp}"] = {"n_tickers": len(pooled),
                                                  "mean_diff": float(np.mean(pooled)) if pooled else None,
                                                  "p_one_sided": p, "origins_with_positive_median": pos,
                                                  "pass": bool(p is not None and p < ALPHA and pos >= 2)}
    rep["verdict"] = ("INCOMPLETE" if len(per) < len(ORIGINS) else
                      "PASS" if rep["pooled"]["control_minus_window_permuted"]["pass"] else "FAIL")
    save_json(rep, od / "verdict.json")
    print(json.dumps({k: rep[k] for k in ("run", "variant", "verdict", "pooled")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--origin", choices=ORIGINS)
    ap.add_argument("--variant", choices=["data", "llm"], default="data")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--estimator", choices=["tft", "ols"], default="tft")
    a = ap.parse_args()
    global ESTIMATOR
    ESTIMATOR = a.estimator
    if a.aggregate:
        aggregate(a.variant); return
    if not a.origin:
        ap.error("--origin required")
    rows = [run_ticker(t, a.origin, a.variant, a.device) for t in tickers_28()]
    save_json({"origin": a.origin, "variant": a.variant, "estimator": ESTIMATOR,
               "seeds": list(SEEDS) if ESTIMATOR == "tft" else None, "per_ticker": rows},
              out_dir(a.variant) / f"origin_{a.origin[:4]}.json")


if __name__ == "__main__":
    main()
