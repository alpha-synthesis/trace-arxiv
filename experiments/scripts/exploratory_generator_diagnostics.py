#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory diagnostics that need the generator, recomputed with released code.

Companion to exploratory_diagnostics.py (which only reads saved artifacts).
Each stock is simulated with the arm's saved impact matrix and labels; no
model is trained. Quoted in Sections 4.1 and 4.3 of the TRACE paper:

  guard_variants   fidelity guard (30 paths, seed 7, against the initial
                   pipeline regenerated with current code) for the arm's own
                   effects b and for b = 0, -b and b shuffled across factors;
                   KS change relative to b
  tracking         terminal tracking error per seed (seeds 1-10, 10 paths)
                   for R1 and R2; whole-path RMS gap to history (seeds 1-3)
  group_ratio      cumulative effect of removing the eligible positive-effect
                   group (b > 0), over the s.d. of an unpaired factual path
  cum_ratio        the same ratio for each single-factor removal (seed 1)
  timing           share of a single-factor removal's effect that falls on
                   event days, in the 20 trading days before an episode
                   starts, and before the factor's first event (seed 1)
  shifted          per-factor B2 criteria (sign, ratio in [0.5, 2], above
                   twice the seed s.d.; seeds 1-5) with presence shifted by
                   756 trading days (three years)

Usage: python exploratory_generator_diagnostics.py [--workers 4] [--tickers AAPL,MSFT]
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ORIG, OUT, build_presence, save_json  # noqa: E402

REF = "r0"
ARMS = {"R1": "r1_datasign", "R2": "r2_datasign"}
NOISE = {"market_noise", "noise_residual"}
SEED_GUARD, S_GUARD = 7, 30
SEEDS, S = tuple(range(1, 11)), 10
SEEDS_FACTOR = (1, 2, 3, 4, 5)
SHIFT = 756
DST = OUT / "exploratory_generator_diagnostics"


def _load(run, t):
    from sde_causal_generator.data_structures import ImpactMatrix, WindowAnalysis
    im = ImpactMatrix.load(str(ORIG / "results" / run / t / "impact_matrix.json"))
    an = [WindowAnalysis.from_dict(d) for d in json.load(open(ORIG / "results" / run / t / "analyses.json"))]
    return im, an


def _paths(im, rp, rlr, P, n, seed, **kw):
    from sde_causal_generator.generate import ImpactDrivenGenerator
    g = ImpactDrivenGenerator(im)
    np.random.seed(seed)
    o = g.generate_scenario(n_steps=len(rp), initial_prices=rp[0], n_samples=n, real_log_returns=rlr,
                            real_volume=rp[:, 4], historical_presence=P, real_prices=rp, **kw)
    return o[:, :, 3], g


def _acf_abs(r, lags=range(1, 11)):
    a = np.abs(r) - np.abs(r).mean()
    den = (a * a).sum()
    return float(np.mean([(a[l:] * a[:-l]).sum() / den for l in lags]))


def _fidelity(close, rr):
    sr = np.diff(np.log(np.clip(close, 1e-8, None)), axis=1)
    m = np.array([(s.std() / rr.std(), ks_2samp(rr, s).statistic, _acf_abs(s)) for s in sr])
    return m.mean(axis=0)


def _variant(im, kind):
    v = copy.deepcopy(im)
    if kind == "zero":
        v.base_impact = np.zeros_like(v.base_impact)
    elif kind == "neg":
        v.base_impact = -v.base_impact
    elif kind == "shuffle":
        perm = np.random.RandomState(0).permutation(v.base_impact.shape[0])
        v.base_impact = v.base_impact[perm]
    return v


def _cum(c):
    return np.log(c[:, -1] / c[:, 0])


def _criteria(d_path, d_exp, sd, b_sign):
    ratio = d_path / d_exp if abs(d_exp) > 0 else np.nan
    return (bool(np.sign(d_path) == -b_sign), bool(0.5 <= ratio <= 2.0), bool(abs(d_path) > 2 * sd))


def run_ticker(t):
    t0 = time.time()
    panel = pd.read_csv(ORIG / "results" / REF / "training_data.csv")
    df = panel[panel.tic == t].sort_values("date").reset_index(drop=True)
    rp = np.clip(df[["open", "high", "low", "close", "volume"]].values.astype(float), 1e-8, None)
    rlr = np.diff(np.log(rp), axis=0)
    rr = rlr[:, 3]
    hist = np.log(rp[:, 3] / rp[0, 3])
    im0, an0 = _load(REF, t)
    f0 = _fidelity(_paths(im0, rp, rlr, build_presence(an0, df, im0.factor_names), S_GUARD, SEED_GUARD)[0], rr)
    out = {"ticker": t, "reference": {"sig": f0[0], "ks": f0[1], "acf": f0[2]}}
    for arm, run in ARMS.items():
        im, an = _load(run, t)
        names = list(im.factor_names)
        P = build_presence(an, df, names)
        act = P > 0
        b = im.base_impact[:, 3]
        occ = act.mean(axis=0)
        base = np.array([(n in NOISE) or (o > 0.8) for n, o in zip(names, occ)])
        elig = [k for k in range(len(names)) if not base[k] and act[:, k].sum() >= 20 and abs(b[k]) >= 1e-8]
        res = {"n_eligible": len(elig)}

        guard = {}
        for kind in ("b", "zero", "neg", "shuffle"):
            f = _fidelity(_paths(_variant(im, kind), rp, rlr, P, S_GUARD, SEED_GUARD)[0], rr)
            guard[kind] = {"sig": f[0], "ks": f[1], "acf": f[2],
                           "pass": bool(abs(f[0] - f0[0]) <= 0.05 and f[1] - f0[1] <= 0.02 and f[2] >= 0.7 * f0[2])}
        for kind in ("zero", "neg", "shuffle"):
            guard[kind]["d_ks_vs_b"] = guard[kind]["ks"] - guard["b"]["ks"]
        res["guard"] = guard

        facs, refs = {}, {}
        for s in SEEDS:
            c, g = _paths(im, rp, rlr, P, S, s, record_ref=True)
            facs[s], refs[s] = c, g._last_ref
        res["track_by_seed"] = {s: float(_cum(facs[s]).mean() - hist[-1]) for s in SEEDS}
        res["rms_whole_path"] = float(np.mean([np.sqrt(np.mean((np.log(facs[s] / facs[s][:, :1]).mean(axis=0) - hist) ** 2))
                                               for s in (1, 2, 3)]))

        bulls = [names[k] for k in elig if b[k] > 0]
        if bulls:
            eff = [float((_cum(_paths(im, rp, rlr, P, S, s, matched_ref=refs[s], remove_factors=bulls)[0])
                          - _cum(facs[s])).mean()) for s in SEEDS]
            sd = float(np.concatenate([_cum(facs[s]) for s in (1, 2, 3)]).std(ddof=1))
            res["group"] = {"n": len(bulls), "cum_effect": float(np.mean(eff)), "unpaired_sd": sd,
                            "ratio": abs(float(np.mean(eff))) / sd}
        else:
            res["group"] = None

        timing = []
        fac1 = facs[1]
        sd_unpaired = float(np.concatenate([_cum(facs[s]) for s in (1, 2, 3)]).std(ddof=1))
        for k in elig:
            cf = _paths(im, rp, rlr, P, S, 1, matched_ref=refs[1], remove_factors=[names[k]])[0]
            d = (np.diff(np.log(cf), axis=1) - np.diff(np.log(fac1), axis=1)).mean(axis=0)  # return into day t, t>=1
            a = act[1:, k]
            tot = d.sum()
            if abs(tot) < 1e-15:
                continue
            days = np.where(act[:, k])[0]
            starts = days[np.r_[True, np.diff(days) > 1]]
            pre = np.zeros(len(d), bool)
            for st in starts:
                if st == 0:
                    continue
                lo = max(st - 20, 1)
                pre[lo - 1:st - 1] = True
            pre &= ~a
            first = days.min()
            timing.append({"on_event": float(d[a].sum() / tot), "before_onset": float(d[pre].sum() / tot),
                           "before_first": float(d[:max(first - 1, 0)].sum() / tot),
                           "cum_ratio": float(abs(tot) / sd_unpaired)})
        res["timing"] = timing

        Ps = np.roll(P, SHIFT, axis=0)
        sfac, sref = {}, {}
        for s in SEEDS_FACTOR:
            c, g = _paths(im, rp, rlr, Ps, S, s, record_ref=True)
            sfac[s], sref[s] = c, g._last_ref
        crit = []
        for k in elig:
            ds = [float((np.diff(np.log(_paths(im, rp, rlr, Ps, S, s, matched_ref=sref[s], remove_factors=[names[k]])[0]), axis=1)
                         - np.diff(np.log(sfac[s]), axis=1)).mean()) for s in SEEDS_FACTOR]
            crit.append(_criteria(float(np.mean(ds)), float(-occ[k] * b[k]), float(np.std(ds, ddof=1)), np.sign(b[k])))
        crit = np.array(crit, bool) if crit else np.zeros((0, 3), bool)
        res["shifted"] = {"n": int(len(crit)), "share_sign": float(crit[:, 0].mean()) if len(crit) else None,
                          "share_ratio": float(crit[:, 1].mean()) if len(crit) else None,
                          "share_visible": float(crit[:, 2].mean()) if len(crit) else None}
        out[arm] = res
    out["elapsed_s"] = round(time.time() - t0, 1)
    save_json(out, DST / f"ticker_{t}.json")
    return out


def aggregate(rows):
    agg = {}
    for arm in ARMS:
        g = {kind: sum(r[arm]["guard"][kind]["pass"] for r in rows) for kind in ("b", "zero", "neg", "shuffle")}
        dks = np.array([abs(r[arm]["guard"][k]["d_ks_vs_b"]) for r in rows for k in ("zero", "neg", "shuffle")])
        grp = [r[arm]["group"]["ratio"] for r in rows if r[arm]["group"]]
        tim = [x for r in rows for x in r[arm]["timing"]]
        sh = [r[arm]["shifted"] for r in rows if r[arm]["shifted"]["n"]]
        agg[arm] = {
            "guard_pass_counts": g, "max_abs_d_ks_vs_b": float(dks.max()), "median_abs_d_ks_vs_b": float(np.median(dks)),
            "rms_whole_path_median": float(np.median([r[arm]["rms_whole_path"] for r in rows])),
            "group_ratio_median": float(np.median(grp)) if grp else None, "group_n_stocks": len(grp),
            "timing_median": {k: float(np.median([x[k] for x in tim])) for k in ("on_event", "before_onset", "before_first", "cum_ratio")},
            "timing_n_factors": len(tim),
            "shifted_stocks_passing": {c: int(sum(x[f"share_{c}"] >= 0.9 for x in sh)) for c in ("sign", "ratio", "visible")},
            "shifted_n_stocks": len(sh),
        }
    # R2 vs R1 terminal tracking, as in b2_counterfactual: |mean signed error over seeds|, R2 worse by
    # more than 0.05 log units fails; seeds 1-3 (the criterion) and 4-10 (independent seeds)
    def crit(seeds):
        fails = []
        for r in rows:
            e1 = abs(np.mean([r["R1"]["track_by_seed"][s] for s in seeds]))
            e2 = abs(np.mean([r["R2"]["track_by_seed"][s] for s in seeds]))
            if e2 - e1 > 0.05:
                fails.append(r["ticker"])
        return {"pass": len(rows) - len(fails), "fails": fails}
    gaps = {}
    for r in rows:
        d = np.array([abs(r["R2"]["track_by_seed"][s]) - abs(r["R1"]["track_by_seed"][s]) for s in SEEDS])
        gaps[r["ticker"]] = {"mean": float(d.mean()), "sd": float(d.std(ddof=1)), "t": float(d.mean() / (d.std(ddof=1) / np.sqrt(len(d))))}
    agg["tracking"] = {"seeds_1_3": crit((1, 2, 3)), "seeds_4_10": crit(tuple(range(4, 11))),
                       "seeds_1_10": crit(SEEDS), "gap_by_stock": gaps}
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tickers", default="")
    ap.add_argument("--aggregate", action="store_true", help="re-aggregate saved per-stock files")
    args = ap.parse_args()
    panel = pd.read_csv(ORIG / "results" / REF / "training_data.csv")
    tickers = args.tickers.split(",") if args.tickers else sorted(panel.tic.unique())
    if args.aggregate:
        rows = []
        for t in tickers:
            r = json.load(open(DST / f"ticker_{t}.json"))
            for arm in ARMS:
                r[arm]["track_by_seed"] = {int(k): v for k, v in r[arm]["track_by_seed"].items()}
            rows.append(r)
    else:
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(run_ticker, tickers))
    if not args.tickers:
        agg = aggregate(rows)
        save_json(agg, DST / "summary.json")
        print(json.dumps({k: v for k, v in agg.items() if k != "tracking"}, indent=1, default=str))
        print(json.dumps({k: v for k, v in agg["tracking"].items() if k != "gap_by_stock"}, indent=1, default=str))
    else:
        print(json.dumps(rows, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
