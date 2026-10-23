#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sign-recovery check (criteria v2; required before any B3 verdict).

Planted-effect recovery on realistic inputs. Per ticker:
  * presence = the run's real presence matrix (TRACE_RUN labels);
  * close returns simulated from a GJR-GARCH(1,1) with Student-t errors fitted
    to that ticker's real close log-returns (arch);
  * open/high/low keep the real ratios to close; volume is the real volume;
  * planted effects beta * sigma (sigma = std of the real close log-return)
    on 10 factors drawn at random among non-baseline factors with >= 20
    active days, random signs; presence on day t moves the return t -> t+1
    (training convention);
  * corrected TFT (impact_estimator="ablation", direction_prior none), same
    hyperparameters as the pipeline, training seed 0.
Reported per beta in {0.05, 0.1, 0.2, 0.3}: sign-recovery rate over planted
factors, Pearson correlation of recovered vs planted (in sigma units), and
the spread of the non-planted factors' b (false-signal scale).

One training seed per cell, whereas the B3 tests use the 5-seed majority sign,
so these recovery rates are a conservative (lower-bound) power estimate.

Added after a smoke run on AAPL was seen (2026-10-01 00:40; TFT numbers
unchanged by this addition): two reference estimators on the same synthetic
returns, to separate estimator power from design power:
  * OLS of the close return on all presence indicators (+ intercept), which
    is the correctly specified model for the planted data;
  * univariate in-window mean minus overall mean.
The simulated series' own s.d. is reported next to sigma.

Usage: TRACE_RUN=<run> python sign_recovery.py [--tickers AAPL,MSFT] [--device cuda] [--seed S]
       TRACE_RUN=<run> python sign_recovery.py --aggregate
       TRACE_RUN=<run> python sign_recovery.py --aggregate-majority

Seeds 1-4 (added after the seed-0 result and the estimator comparison had been
seen, so that the check measures the 5-seed majority sign that the B3 tests
use): identical panels (the panel RNG depends on the ticker only),
training seed S; outputs in sign_recovery__<RUN>/seed<S>/. --aggregate-majority
applies the 80% rule to the sign of the median b over seeds 0-4, the
test estimator fixed in advance, with a ticker-cluster bootstrap CI.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RUN_TAG, build_presence, load_ticker_artifacts, save_json, ticker_df, tickers_28  # noqa: E402
from shared import FEATS  # noqa: E402

BETAS = (0.05, 0.1, 0.2, 0.3)
N_PLANT = 10
MIN_ACTIVE = 20
BASELINE_NAMES = {"market_noise", "noise_residual"}
OUT_DIR = OUT / f"sign_recovery__{RUN_TAG}"


def seed_all(s):
    import torch
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def gjr_t_returns(r_real, T, rng):
    from arch import arch_model
    am = arch_model(r_real * 100, vol="Garch", p=1, o=1, q=1, dist="t", mean="Zero", rescale=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = am.fit(disp="off", show_warning=False)
    # simulated with our own RNG (seeded per ticker), 500-step burn-in
    w, a, g, b = (float(res.params[k]) for k in ("omega", "alpha[1]", "gamma[1]", "beta[1]"))
    nu = float(res.params["nu"])
    z = rng.standard_t(nu, size=T + 500) * np.sqrt((nu - 2) / nu)
    h = np.empty(T + 500); e = np.empty(T + 500)
    h[0] = w / max(1 - a - g / 2 - b, 1e-3)
    for t in range(T + 500):
        if t:
            h[t] = w + (a + g * (e[t - 1] < 0)) * e[t - 1] ** 2 + b * h[t - 1]
        e[t] = np.sqrt(h[t]) * z[t]
    return e[500:] / 100, {"omega": w, "alpha": a, "gamma": g, "beta": b, "nu": nu}


def run_ticker(ticker, device, seed=0):
    from sde_causal_generator.factor_impact_network import train_impact_network
    im0, _, analyses = load_ticker_artifacts(ticker)
    names = im0.factor_names
    df = ticker_df(ticker)
    real = np.clip(df[FEATS].values.astype(np.float64), 1e-8, None)
    P = build_presence(analyses, df, names)
    T = len(real)
    r_real = np.diff(np.log(real[:, 3]))
    sigma = float(r_real.std())
    occ = (P > 0).mean(axis=0)
    cand = [k for k, n in enumerate(names)
            if n not in BASELINE_NAMES and occ[k] <= 0.8 and (P[:, k] > 0).sum() >= MIN_ACTIVE]
    rng = np.random.RandomState(zlib.crc32(ticker.encode()))
    plant = rng.choice(cand, size=min(N_PLANT, len(cand)), replace=False)
    signs = rng.choice([-1.0, 1.0], size=len(plant))
    eps, garch = gjr_t_returns(r_real, T - 1, rng)
    ratios = real[:, :3] / real[:, [3]]
    out = {"ticker": ticker, "sigma": sigma, "sim_sd": float(eps.std()), "garch": garch,
           "active_days_planted": [int((P[:, k] > 0).sum()) for k in plant], "n_candidates": len(cand),
           "planted": [names[k] for k in plant], "signs": signs.tolist(), "per_beta": {}}
    for beta in BETAS:
        r = eps.copy()
        act = (P[:-1, plant] > 0).astype(float)            # presence on day t -> return t -> t+1
        r += (act * (beta * sigma * signs)).sum(axis=1)
        close = real[0, 3] * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
        prices = np.column_stack([ratios * close[:, None], close, real[:, 4]])
        t0 = time.time()
        seed_all(seed)
        _, im, _ = train_impact_network(
            factor_matrix=P, price_data=prices, factor_names=names, feature_names=FEATS,
            n_lags=10, n_epochs=300, batch_size=64, learning_rate=1e-3, val_fraction=0.15,
            device=device, patience=30, architecture="tft", llm_directions=None,
            impact_estimator="ablation")
        b = im.base_impact[:, 3] / sigma
        rec = b[plant]
        A = (P[:-1] > 0).astype(float)
        ols = np.linalg.lstsq(np.column_stack([np.ones(len(r)), A]), r, rcond=None)[0][1:] / sigma
        uni = np.array([(r[A[:, k] > 0].mean() - r.mean()) / sigma for k in plant])
        others = np.delete(b, np.concatenate([plant, [k for k, n in enumerate(names)
                                                      if n in BASELINE_NAMES or occ[k] > 0.8]]).astype(int))
        out["per_beta"][str(beta)] = {
            "recovered_sigma": rec.tolist(),
            "sign_recovery": float((np.sign(rec) == signs).mean()),
            "corr": float(np.corrcoef(rec, beta * signs)[0, 1]) if len(rec) > 2 else None,
            "others_sd_sigma": float(others.std()) if len(others) else None,
            "ols_sign_recovery": float((np.sign(ols[plant]) == signs).mean()),
            "ols_corr": float(np.corrcoef(ols[plant], signs)[0, 1]),
            "univariate_sign_recovery": float((np.sign(uni) == signs).mean()),
            "elapsed_s": round(time.time() - t0, 1)}
        pb = out["per_beta"][str(beta)]
        print(f"  {ticker} beta={beta}: sign={pb['sign_recovery']:.2f} corr={pb['corr']:.2f} "
              f"ols={pb['ols_sign_recovery']:.2f} uni={pb['univariate_sign_recovery']:.2f} "
              f"others_sd={pb['others_sd_sigma']:.3f}σ ({pb['elapsed_s']}s)", flush=True)
    return out


def aggregate():
    rows = [json.loads(f.read_text()) for f in sorted(OUT_DIR.glob("ticker_*.json"))]
    rep = {"run": RUN_TAG, "n_tickers": len(rows), "per_beta": {}}
    for beta in BETAS:
        s = str(beta)
        rec = np.concatenate([np.sign(r["per_beta"][s]["recovered_sigma"]) == np.array(r["signs"]) for r in rows])
        x = np.concatenate([np.array(r["per_beta"][s]["recovered_sigma"]) for r in rows])
        y = np.concatenate([beta * np.array(r["signs"]) for r in rows])
        rep["per_beta"][s] = {
            "pooled_sign_recovery": float(rec.mean()),
            "median_ticker_sign_recovery": float(np.median([r["per_beta"][s]["sign_recovery"] for r in rows])),
            "pooled_corr": float(np.corrcoef(x, y)[0, 1]),
            "pooled_ols_sign_recovery": float(np.mean([r["per_beta"][s]["ols_sign_recovery"] for r in rows])),
            "pooled_univariate_sign_recovery": float(np.mean([r["per_beta"][s]["univariate_sign_recovery"] for r in rows])),
            "median_others_sd_sigma": float(np.median([r["per_beta"][s]["others_sd_sigma"] for r in rows])),
            "powered_at_80pct": bool(rec.mean() >= 0.80)}
    save_json(rep, OUT_DIR / "summary.json")
    print(json.dumps(rep, indent=2))


def aggregate_majority(seeds=(0, 1, 2, 3, 4), n_boot=4000):
    dirs = [OUT_DIR if s == 0 else OUT_DIR / f"seed{s}" for s in seeds]
    tick = sorted(f.name for f in OUT_DIR.glob("ticker_*.json"))
    tick = [t for t in tick if all((d / t).exists() for d in dirs)]
    rep = {"run": RUN_TAG, "seeds": list(seeds), "n_tickers": len(tick), "per_beta": {}}
    rng = np.random.RandomState(0)
    for beta in BETAS:
        s_ = str(beta)
        per_t = []
        for t in tick:
            rows = [json.loads((d / t).read_text()) for d in dirs]
            assert all(r["planted"] == rows[0]["planted"] and r["signs"] == rows[0]["signs"] for r in rows)
            B = np.stack([r["per_beta"][s_]["recovered_sigma"] for r in rows])     # (seeds, planted)
            signs = np.array(rows[0]["signs"])
            per_t.append({"maj": (np.sign(np.median(B, axis=0)) == signs).astype(float),
                          "seed0": (np.sign(B[0]) == signs).astype(float),
                          "seed_corr": float(np.mean([np.corrcoef(B[i], B[j])[0, 1]
                                                      for i in range(len(seeds)) for j in range(i + 1, len(seeds))]))})
        maj = np.concatenate([x["maj"] for x in per_t])
        boots = []
        for _ in range(n_boot):
            pick = rng.randint(0, len(per_t), len(per_t))
            boots.append(np.concatenate([per_t[i]["maj"] for i in pick]).mean())
        lo, hi = np.percentile(boots, [2.5, 97.5])
        rep["per_beta"][s_] = {
            "majority_sign_recovery": float(maj.mean()), "ci95_ticker_bootstrap": [float(lo), float(hi)],
            "seed0_sign_recovery": float(np.concatenate([x["seed0"] for x in per_t]).mean()),
            "median_seed_corr_of_b": float(np.median([x["seed_corr"] for x in per_t])),
            "powered_at_80pct": bool(maj.mean() >= 0.80)}
    save_json(rep, OUT_DIR / "summary_majority5.json")
    print(json.dumps(rep, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tickers", default="")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--aggregate", action="store_true")
    ap.add_argument("--aggregate-majority", action="store_true")
    a = ap.parse_args()
    if a.aggregate:
        aggregate(); return
    if a.aggregate_majority:
        aggregate_majority(); return
    out = OUT_DIR if a.seed == 0 else OUT_DIR / f"seed{a.seed}"
    for t in (a.tickers.split(",") if a.tickers else tickers_28()):
        f = out / f"ticker_{t}.json"
        if f.exists():
            continue
        save_json(run_ticker(t, a.device, a.seed), f)


if __name__ == "__main__":
    main()
