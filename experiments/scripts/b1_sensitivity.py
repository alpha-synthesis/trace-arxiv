#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 sensitivity: lead-lag profile and leak dose-response (Section 4.2).

Reads the saved labels only (analyses.json of A0, A1 and R2).

  lead_lag   excess agreement of each label set's net direction with the
             return of window w+j, j = -3..3, with calendar-window cluster
             bootstrap 95% intervals (as make_figures._cluster_ci)
  dose       share of draws in which B1's calendar-window interval excludes
             zero when the A0 (price-informed) labels replace the labels of a
             share pi of calendar windows; two baselines: R2's own labels,
             and a null labeller (R2 directions permuted across windows within
             each stock, which keeps each stock's marginals and removes any
             association with returns)
  one_stock  each stock in turn fully replaced by its A0 labels, on R2
  panel_perm calendar-window permutation null: the realized returns of whole
             calendar windows are permuted jointly across stocks (one
             permutation for all 28), labels fixed; keeps shared news and the
             cross-stock dependence of returns; one-sided p for the excess
  per_stock  the independence baseline computed within each stock and pooled
             with the stock's weight, A_ind = sum_i n_i (u_i b_i + (1-u_i)(1-b_i)) / n,
             for the same window and the next window (placebo), with
             calendar-window cluster bootstrap intervals that recompute the
             per-stock rates in each draw
  other_llms B1 on the labels of other LLMs saved by the multi-LLM extension
             (experiments/results/cross_model_labels/runs_m1, runs_m2; prompt with
             prices, uncurated corpus, as A0), with the same scoring

Usage: python b1_sensitivity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ORIG, OUT, save_json  # noqa: E402

NOISE = {"market_noise", "noise_residual"}
ARMS = {"A0": "r0", "A1": "a1", "R2": "r2_datasign"}
SHARES = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20)
REPS = 200


def load(run, root=None):
    rows = []
    for tdir in sorted((root or ORIG / "results" / run).iterdir()):
        f = tdir / "analyses.json"
        if not f.exists():
            continue
        for i, w in enumerate(sorted(json.load(open(f)), key=lambda x: x["start_date"])):
            fs = [x for x in w["factors"] if x["name"] not in NOISE and x["magnitude"] > 0]
            rows.append({"ticker": tdir.name, "win": i, "s": sum(x["direction"] * x["magnitude"] for x in fs),
                         "r": float(w["price_change_pct"])})
    return pd.DataFrame(rows).sort_values(["ticker", "win"]).reset_index(drop=True)


def _scored(s, r, win):
    m = (s != 0) & (r != 0) & ~np.isnan(r)
    return np.sign(s[m]), np.sign(r[m]), win[m]


def _excess(s, r):
    up, bull = (r > 0).mean(), (s > 0).mean()
    return float((s == r).mean() - (up * bull + (1 - up) * (1 - bull)))


def _boot(s, r, win, n_boot, seed):
    wins = np.unique(win)
    idx = np.searchsorted(wins, win)
    W = len(wins)
    n_w = np.bincount(idx, minlength=W).astype(float)
    a_w = np.bincount(idx, weights=(s == r).astype(float), minlength=W)
    u_w = np.bincount(idx, weights=(r > 0).astype(float), minlength=W)
    b_w = np.bincount(idx, weights=(s > 0).astype(float), minlength=W)
    rng = np.random.RandomState(seed)
    C = np.stack([np.bincount(rng.randint(0, W, W), minlength=W) for _ in range(n_boot)])
    n = C @ n_w
    up, bull = (C @ u_w) / n, (C @ b_w) / n
    return (C @ a_w) / n - (up * bull + (1 - up) * (1 - bull))


def lead_lag(df, n_boot=2000):
    out = {}
    for j in range(-3, 4):
        rj = df.groupby("ticker").r.shift(-j).values
        s, r, w = _scored(df.s.values, rj, df.win.values)
        bt = _boot(s, r, w, n_boot, 0)
        out[str(j)] = {"excess": _excess(s, r), "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))]}
    return out


def dose(base_s, a0_s, r, win, seed=1):
    rng = np.random.RandomState(seed)
    W = np.unique(win)
    out = {}
    for pi in SHARES:
        det, ex = 0, []
        for rep in range(REPS):
            leak = np.isin(win, rng.choice(W, int(round(pi * len(W))), replace=False))
            s_mix = np.where(leak & ~np.isnan(a0_s), a0_s, base_s)
            s, rr, w = _scored(s_mix, r, win)
            ex.append(_excess(s, rr))
            det += np.percentile(_boot(s, rr, w, 500, rep), 2.5) > 0
        out[str(pi)] = {"mean_excess": float(np.mean(ex)), "detect_rate": det / REPS}
    return out


def panel_perm(df, n_perm=10000, seed=0):
    piv_s = df.pivot(index="win", columns="ticker", values="s")
    piv_r = df.pivot(index="win", columns="ticker", values="r")
    S, R = piv_s.values, piv_r.values
    s, r, _ = _scored(S.ravel(), R.ravel(), np.zeros(S.size))
    obs = _excess(s, r)
    rng = np.random.RandomState(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        Rp = R[rng.permutation(R.shape[0])]
        s, r, _ = _scored(S.ravel(), Rp.ravel(), np.zeros(S.size))
        null[i] = _excess(s, r)
    return {"excess": obs, "p_one_sided": float((1 + (null >= obs).sum()) / (1 + n_perm)),
            "null_mean": float(null.mean()), "null_sd": float(null.std()), "n_perm": n_perm}


def _per_stock_excess(C, n, a, u, b):
    """C: (draws, W) window counts; n, a, u, b: (W, S) per window and stock."""
    N, A, U, B = C @ n, C @ a, C @ u, C @ b
    with np.errstate(invalid="ignore", divide="ignore"):
        uu, bb = np.where(N > 0, U / N, 0), np.where(N > 0, B / N, 0)
    ind = (N * (uu * bb + (1 - uu) * (1 - bb))).sum(axis=1) / N.sum(axis=1)
    return A.sum(axis=1) / N.sum(axis=1) - ind


def per_stock(df, n_boot=2000, seed=0):
    out = {}
    for name, j in (("same_window", 0), ("next_window", 1)):
        d = df.copy()
        d["rj"] = d.groupby("ticker").r.shift(-j)
        d = d[(d.s != 0) & (d.rj != 0) & d.rj.notna()]
        tick = sorted(d.ticker.unique()); wins = sorted(d.win.unique())
        ti, wi = np.searchsorted(tick, d.ticker), np.searchsorted(wins, d.win)
        shape = (len(wins), len(tick))
        n, a, u, b = (np.zeros(shape) for _ in range(4))
        np.add.at(n, (wi, ti), 1)
        np.add.at(a, (wi, ti), (np.sign(d.s) == np.sign(d.rj)).astype(float))
        np.add.at(u, (wi, ti), (d.rj > 0).astype(float))
        np.add.at(b, (wi, ti), (d.s > 0).astype(float))
        obs = float(_per_stock_excess(np.ones((1, len(wins))), n, a, u, b)[0])
        rng = np.random.RandomState(seed)
        C = np.stack([np.bincount(rng.randint(0, len(wins), len(wins)), minlength=len(wins)) for _ in range(n_boot)])
        bt = _per_stock_excess(C.astype(float), n, a, u, b)
        out[name] = {"excess": obs, "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                     "p_one_sided": float((bt <= 0).mean())}
    return out


def other_llms():
    base = OUT / "cross_model_labels"
    out = {}
    for model in ("gpt-4o-mini", "claude-haiku-45", "gemini-25-flash", "gemini-20-flash", "claude-35-haiku"):
        dirs = [base / m / model for m in ("runs_m1", "runs_m2")]
        dfs = [load(None, d) for d in dirs if any(d.glob("*/analyses.json"))]
        if not dfs:
            continue
        df = pd.concat(dfs).drop_duplicates(["ticker", "win"]).sort_values(["ticker", "win"])
        s, r, w = _scored(df.s.values, df.r.values, df.win.values)
        a = float((s == r).mean())
        bt = _boot(s, r, w, 2000, 0)
        out[model] = {"stocks": int(df.ticker.nunique()), "n_scored": int(len(s)), "agreement": a,
                      "independence": a - _excess(s, r), "excess": _excess(s, r),
                      "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
                      "tickers": sorted(df.ticker.unique())}
    return out


def main():
    D = {k: load(v) for k, v in ARMS.items()}
    res = {"lead_lag": {k: lead_lag(df) for k, df in D.items()}}
    r2 = D["R2"]
    a0 = D["A0"].set_index(["ticker", "win"]).s.reindex(pd.MultiIndex.from_frame(r2[["ticker", "win"]])).values
    rng = np.random.RandomState(7)
    null_s = r2.groupby("ticker").s.transform(lambda x: x.values[rng.permutation(len(x))]).values
    res["dose"] = {"on_R2": dose(r2.s.values, a0, r2.r.values, r2.win.values),
                   "on_null": dose(null_s, a0, r2.r.values, r2.win.values)}
    one = {}
    for t in r2.ticker.unique():
        s_mix = np.where((r2.ticker.values == t) & ~np.isnan(a0), a0, r2.s.values)
        s, rr, w = _scored(s_mix, r2.r.values, r2.win.values)
        one[t] = {"excess": _excess(s, rr), "detected": bool(np.percentile(_boot(s, rr, w, 1000, 0), 2.5) > 0)}
    res["panel_perm"] = {k: panel_perm(df) for k, df in D.items()}
    res["per_stock"] = {k: per_stock(df) for k, df in D.items()}
    res["other_llms"] = other_llms()
    res["one_stock"] = {"detected": sum(v["detected"] for v in one.values()), "stocks": len(one), "by_stock": one}
    save_json(res, OUT / "b1_sensitivity" / "summary.json")
    print(json.dumps({k: res[k] for k in ("dose",)}, indent=1))
    print({k: {j: round(100 * v["excess"], 2) for j, v in p.items()} for k, p in res["lead_lag"].items()})
    print({k: (v["stocks"], round(v["agreement"], 3), round(v["independence"], 3)) for k, v in res["other_llms"].items()})
    print({k: (round(100 * v["excess"], 2), v["p_one_sided"]) for k, v in res["panel_perm"].items()})
    print({k: {w: (round(100 * x["excess"], 2), [round(100 * c, 2) for c in x["ci95"]]) for w, x in v.items()}
           for k, v in res["per_stock"].items()})
    print("one stock leaked: detected", res["one_stock"]["detected"], "of", res["one_stock"]["stocks"])


if __name__ == "__main__":
    main()
