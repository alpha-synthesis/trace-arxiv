#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B2 counterfactual consistency: removal changes the paths as specified, and the factual path tracks history.

Criteria v2 and v2.1. Stated as verified plumbing, not evidence about reality.

Generator: the run's corrected impact matrix (linear drift, daily units).
Matched counterfactual: the factual run records background variance,
calibration and its schedule (record_ref); each removal replays them with the
same seed (matched_ref) and switches the removed factors fully off.

Eligible factors: non-baseline (not market_noise / noise_residual, active on
<= 80% of days) with >= 20 active days. Comparator (expected daily Δ of a
removal set R): -Σ_{k∈R} p_k b_k, p_k = share of days with presence > 0.
Path Δ: mean daily close log-return difference (counterfactual - factual),
averaged over S paths, per seed; seed s.d. across seeds.

Per factor (single-factor removal, primary per criteria v2.1), 5 seeds:
  (a) sign(Δ_path) = -sign(b_k); (b) Δ_path / Δ_expected in [0.5, 2];
  (c) |Δ_path| > 2 x seed s.d.
  A ticker passes a criterion if it holds for >= 90% of its eligible factors
  with |b_k| >= 1e-8 (definition fixed here, before running).
Groups (bulls = eligible b > 0, bears = eligible b < 0), 10 seeds: the same
  three criteria for both groups; reported.
Factual fidelity: KS(factual, real) and |cumulative tracking error| per
  ticker; for runs other than R1 they must lie within 0.02 (KS) and 0.05 log
  units (tracking) of R1's (requires R1's output first).
Each criterion must hold on >= 24/28 tickers. P3: corr(expected, path) for
  the bull group across tickers >= 0.8. Calibration share is reported.

Usage: TRACE_RUN=<run> python b2_counterfactual.py [--workers 4]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RUN_TAG, build_presence, load_ticker_artifacts, save_json, ticker_df, tickers_28  # noqa: E402

BASELINE_NAMES = {"market_noise", "noise_residual"}
S = 10
SEEDS_FACTOR = (1, 2, 3, 4, 5)
SEEDS_GROUP = tuple(range(1, 11))
# reference run for the fidelity check; TRACE_B2_REF overrides it (positive control, 2026-10-08)
R1 = os.environ.get("TRACE_B2_REF", "r1_datasign")
OUT_DIR = OUT / f"b2_counterfactual__{RUN_TAG}"


def _paths(gen, rp, rlr, P, seed, **kw):
    np.random.seed(seed)
    o = gen.generate_scenario(n_steps=len(rp), initial_prices=rp[0], n_samples=S,
                              real_log_returns=rlr, real_volume=rp[:, 4],
                              historical_presence=P, real_prices=rp, **kw)
    return o[:, :, 3]


def _dlr(cf, fac):
    return float((np.diff(np.log(cf), axis=1) - np.diff(np.log(fac), axis=1)).mean())


def _criteria(d_path, d_exp, sd, b_sign):
    ratio = d_path / d_exp if abs(d_exp) > 0 else np.nan
    return {"sign": bool(np.sign(d_path) == -b_sign), "ratio": float(ratio),
            "ratio_ok": bool(0.5 <= ratio <= 2.0), "visible": bool(abs(d_path) > 2 * sd)}


def run_ticker(t):
    from sde_causal_generator.generate import ImpactDrivenGenerator
    t0 = time.time()
    im, _, analyses = load_ticker_artifacts(t)
    assert im.units == "daily", t
    names = list(im.factor_names)
    df = ticker_df(t)
    rp = np.clip(df[["open", "high", "low", "close", "volume"]].values.astype(float), 1e-8, None)
    rlr = np.diff(np.log(rp), axis=0)
    rr = rlr[:, 3]
    P = build_presence(analyses, df, names)
    b = im.base_impact[:, 3]
    occ = (P > 0).mean(axis=0)
    base = np.array([(n in BASELINE_NAMES) or (o > 0.8) for n, o in zip(names, occ)])
    elig = [k for k in range(len(names)) if not base[k] and (P[:, k] > 0).sum() >= 20 and abs(b[k]) >= 1e-8]
    gen = ImpactDrivenGenerator(im)

    refs = {}
    facs = {}
    for s in sorted(set(SEEDS_FACTOR) | set(SEEDS_GROUP)):
        facs[s] = _paths(gen, rp, rlr, P, s, record_ref=True)
        refs[s] = gen._last_ref

    # single-factor removals
    per_factor = []
    for k in elig:
        ds = [_dlr(_paths(gen, rp, rlr, P, s, matched_ref=refs[s], remove_factors=[names[k]]), facs[s])
              for s in SEEDS_FACTOR]
        c = _criteria(float(np.mean(ds)), float(-occ[k] * b[k]), float(np.std(ds, ddof=1)), np.sign(b[k]))
        per_factor.append({"factor": names[k], "b": float(b[k]), "p": float(occ[k]),
                           "d_path": float(np.mean(ds)), "d_exp": float(-occ[k] * b[k]), **c})
    pf = pd.DataFrame(per_factor)

    # groups
    groups = {}
    for g, mask in (("bulls", [k for k in elig if b[k] > 0]), ("bears", [k for k in elig if b[k] < 0])):
        if not mask:
            groups[g] = None
            continue
        ds = [_dlr(_paths(gen, rp, rlr, P, s, matched_ref=refs[s], remove_factors=[names[k] for k in mask]), facs[s])
              for s in SEEDS_GROUP]
        d_exp = float(-(occ[mask] * b[mask]).sum())
        groups[g] = {"n": len(mask), "d_path": float(np.mean(ds)), "d_exp": d_exp,
                     **_criteria(float(np.mean(ds)), d_exp, float(np.std(ds, ddof=1)), 1.0 if g == "bulls" else -1.0)}

    # factual fidelity (first 3 seeds) and calibration share
    ks = float(np.mean([np.mean([ks_2samp(rr, np.diff(np.log(p))).statistic for p in facs[s]]) for s in (1, 2, 3)]))
    track = float(np.mean([np.log(facs[s][:, -1] / facs[s][:, 0]).mean() - np.log(rp[-1, 3] / rp[0, 3])
                           for s in (1, 2, 3)]))
    calib = float(np.mean([np.abs(refs[s]["calib_delta"].sum(1)).mean() for s in (1, 2, 3)]))

    def share(col):
        return float(pf[col].mean()) if len(pf) else None

    res = {"ticker": t, "n_eligible": len(elig),
           "share_sign": share("sign"), "share_ratio_ok": share("ratio_ok"), "share_visible": share("visible"),
           "median_ratio": float(pf["ratio"].median()) if len(pf) else None,
           "groups": groups, "ks_factual": ks, "track_err_factual": track, "calib_abs_cum_log": calib,
           "elapsed_s": round(time.time() - t0, 1)}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pf.to_csv(OUT_DIR / f"per_factor_{t}.csv", index=False)
    print(f"{t}: n={len(elig)} sign={res['share_sign']} ratio_ok={res['share_ratio_ok']} "
          f"visible={res['share_visible']} med_ratio={res['median_ratio']} ({res['elapsed_s']}s)", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run_ticker, tickers_28()))
    d = pd.DataFrame([{k: v for k, v in r.items() if k != "groups"} for r in rows])
    ok = lambda col: (d[col] >= 0.9)                                         # noqa: E731
    crit = {"sign": ok("share_sign"), "ratio": ok("share_ratio_ok"), "visible": ok("share_visible")}
    if RUN_TAG != R1:
        r1 = pd.read_csv(OUT / f"b2_counterfactual__{R1}" / "per_ticker.csv").set_index("ticker")
        dd = d.set_index("ticker")
        crit["fidelity_vs_R1"] = ((dd.ks_factual - r1.ks_factual).abs() <= 0.02) & \
            ((dd.track_err_factual.abs() - r1.track_err_factual.abs()) <= 0.05)
        crit["fidelity_vs_R1"] = crit["fidelity_vs_R1"].values
    g_ok = {c: [bool(r["groups"]["bulls"] and r["groups"]["bears"] and r["groups"]["bulls"][c] and r["groups"]["bears"][c])
                for r in rows] for c in ("sign", "ratio_ok", "visible")}
    bulls = [(r["groups"]["bulls"]["d_exp"], r["groups"]["bulls"]["d_path"]) for r in rows if r["groups"]["bulls"]]
    p3 = float(np.corrcoef(*np.array(bulls).T)[0, 1]) if len(bulls) > 2 else None
    summary = {"run": RUN_TAG, "n_tickers": len(d),
               "per_factor_criteria_pass": {c: f"{int(np.sum(v))}/{len(d)}" for c, v in crit.items()},
               "group_criteria_pass": {c: f"{int(np.sum(v))}/{len(rows)}" for c, v in g_ok.items()},
               "P3_corr_expected_vs_path_bulls": p3,
               "median_ks_factual": float(d.ks_factual.median()),
               "median_abs_track_err_factual": float(d.track_err_factual.abs().median()),
               "median_calib_abs_cum_log": float(d.calib_abs_cum_log.median())}
    summary["b2_verdict"] = ("PASS" if all(np.sum(v) >= 24 for v in crit.values()) and (p3 or 0) >= 0.8
                                     else "FAIL")
    d.to_csv(OUT_DIR / "per_ticker.csv", index=False)
    save_json({"summary": summary, "rows": rows}, OUT_DIR / "report.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
