#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Implementation checks and fidelity guard (criteria v2 and v2.1).

1. Fidelity guard: corrected run (TRACE_RUN, e.g. r1_datasign) vs R0
   REGENERATED with today's code (legacy mode, r0 matrices), same
   seed, S paths. Pass per ticker: |Δσ-ratio| <= 0.05, ACF(|r|, lags 1-10)
   >= 70% of R0, ΔKS <= 0.02. Gate: >= 26/28.
2. Attribution consistency (plumbing): for non-baseline factors with >= 20
   active days, a matched single-factor removal changes the daily close
   log-return by exactly -b_k on active days (atol 1e-9) in >= 99.9% of
   active entries; misses only where the ±0.20 clip binds.
3. Reported: Σ p·b (all / non-baseline factors), share of cumulative drift
   carried by calibration.

Usage: TRACE_RUN=r1_datasign python implementation_checks.py [--paths 30]

Version 2 of this script, written after its first run (whose output is not
part of this release):
  * attribution implemented strictly as fixed in advance: active days are the
    generator's own factual schedule (``_last_ref["schedule"]``); on active
    days |Δ + b_k| <= 1e-9 (absolute, rtol 0); on inactive days |Δ| <= 1e-9;
    clip exception when either path's return reaches the ±0.20 clip; a
    factor whose removal is never visible fails the ticker. The first
    version accepted Δ ∈ {0, -b_k} on every day, so a no-op removal passed.
  * ACF on |r| (the first version used |r - mean|; negligible difference).
  * also reported: the comparison against the SAVED R0 synthetic paths
    from each run's synthetic_data_causal_sde.csv, skipped when those files are absent
    (they are not part of the public release).
Output: results/implementation_checks__<RUN>/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ORIG, OUT, RUN_TAG, build_presence, save_json  # noqa: E402

from sde_causal_generator.data_structures import ImpactMatrix, WindowAnalysis  # noqa: E402
from sde_causal_generator.generate import ImpactDrivenGenerator  # noqa: E402

REF = "r0"
BASELINE_NAMES = {"market_noise", "noise_residual"}
SEED = 7


def acf_abs(r, lags=range(1, 11)):
    a = np.abs(r)
    a = a - a.mean()
    den = (a * a).sum()
    return float(np.mean([(a[l:] * a[:-l]).sum() / den for l in lags]))


def load(run, t):
    im = ImpactMatrix.load(str(ORIG / "results" / run / t / "impact_matrix.json"))
    an = [WindowAnalysis.from_dict(d) for d in
          json.load(open(ORIG / "results" / run / t / "analyses.json"))]
    return im, an


def paths(gen, rp, rlr, P, S, **kw):
    np.random.seed(SEED)
    o = gen.generate_scenario(n_steps=len(rp), initial_prices=rp[0], n_samples=S,
                              real_log_returns=rlr, real_volume=rp[:, 4],
                              historical_presence=P, real_prices=rp, **kw)
    return o[:, :, 3]


def fidelity(close, rr):
    sr = np.diff(np.log(np.clip(close, 1e-8, None)), axis=1)
    return np.array([(s.std() / rr.std(), ks_2samp(rr, s).statistic, acf_abs(s)) for s in sr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--paths", type=int, default=30)
    args = ap.parse_args()
    panel = pd.read_csv(ORIG / "results" / REF / "training_data.csv")
    rows, attr_rows = [], []
    for t in sorted(panel["tic"].unique()):
        if not (ORIG / "results" / RUN_TAG / t / "impact_matrix.json").exists():
            continue
        df = panel[panel.tic == t].sort_values("date").reset_index(drop=True)
        rp = np.clip(df[["open", "high", "low", "close", "volume"]].values.astype(float), 1e-8, None)
        rlr = np.diff(np.log(rp), axis=0)
        rr = rlr[:, 3]

        im0, an0 = load(REF, t)
        im1, an1 = load(RUN_TAG, t)
        assert im1.units == "daily" and im1.estimator == "ablation", t
        P0 = build_presence(an0, df, im0.factor_names)
        P1 = build_presence(an1, df, im1.factor_names)
        f0 = fidelity(paths(ImpactDrivenGenerator(im0), rp, rlr, P0, args.paths), rr)
        g1 = ImpactDrivenGenerator(im1)
        c1 = paths(g1, rp, rlr, P1, args.paths, record_ref=True)
        f1 = fidelity(c1, rr)

        # drift accounting on the factual corrected paths
        b = im1.base_impact[:, 3]
        occ = (P1 > 0).mean(axis=0)
        base_mask = np.array([(n in BASELINE_NAMES) or (o > 0.8)
                              for n, o in zip(im1.factor_names, occ)])
        calib = g1._last_ref["calib_delta"]                     # (S, T)
        # factor drift = factual minus the matched run with every factor removed
        c_none = paths(g1, rp, rlr, P1, args.paths, matched_ref=g1._last_ref,
                       remove_factors=list(im1.factor_names))
        fac_drift = (np.diff(np.log(c1), axis=1) - np.diff(np.log(c_none), axis=1))
        calib_share = float(np.abs(calib.sum(1)).mean()
                            / (np.abs(calib.sum(1)).mean() + np.abs(fac_drift.sum(1)).mean() + 1e-12))

        rows.append(dict(
            ticker=t,
            sig_ratio_r0=f0[:, 0].mean(), sig_ratio_r1=f1[:, 0].mean(),
            d_sig=f1[:, 0].mean() - f0[:, 0].mean(),
            ks_r0=f0[:, 1].mean(), ks_r1=f1[:, 1].mean(), d_ks=f1[:, 1].mean() - f0[:, 1].mean(),
            acf_r0=f0[:, 2].mean(), acf_r1=f1[:, 2].mean(), acf_ratio=f1[:, 2].mean() / f0[:, 2].mean(),
            sum_pb_all_pct_yr=float((occ * b).sum() * 252 * 100),
            sum_pb_nonbaseline_pct_yr=float((occ * b)[~base_mask].sum() * 252 * 100),
            n_baseline=int(base_mask.sum()), calib_share=calib_share,
        ))

        # attribution consistency (strict): single-factor matched removals, 1 path each
        elig = [k for k in range(len(b)) if not base_mask[k] and (P1[:, k] > 0).sum() >= 20]
        n_act = n_ok = n_clip = n_inact_bad = n_invisible = 0
        max_err = 0.0
        for k in elig:
            g = ImpactDrivenGenerator(im1)
            fac = paths(g, rp, rlr, P1, 1, record_ref=True)
            ref = g._last_ref
            act = ref["schedule"][0, 1:, k] > 0                 # schedule of the return into day t
            cf = paths(g, rp, rlr, P1, 1, matched_ref=ref, remove_factors=[im1.factor_names[k]])
            fr = np.diff(np.log(fac), axis=1).ravel()
            cr = np.diff(np.log(cf), axis=1).ravel()
            d = cr - fr
            clip = (np.abs(fr) >= 0.2 - 1e-12) | (np.abs(cr) >= 0.2 - 1e-12)
            good = np.abs(d + b[k]) <= 1e-9
            n_act += int(act.sum()); n_ok += int((good & act).sum())
            n_clip += int((~good & act & clip).sum())
            n_inact_bad += int((np.abs(d[~act]) > 1e-9).sum())
            n_invisible += int(abs(b[k]) > 1e-12 and not (good & act).any())
            if (act & ~clip).any():
                max_err = max(max_err, float(np.abs(d[act & ~clip] + b[k]).max()))
        attr_rows.append(dict(ticker=t, n_factors=len(elig), n_active_entries=n_act,
                              share_exact=n_ok / max(n_act, 1), clip_exceptions=n_clip,
                              misses_other=n_act - n_ok - n_clip, inactive_nonzero=n_inact_bad,
                              invisible_removals=n_invisible, max_err_nonclip=max_err))

        # comparison against the SAVED R0 / R1 synthetic paths (reported, not gated)
        def saved(run):
            sd = pd.read_csv(ORIG / "results" / run / t / "synthetic_data_causal_sde.csv")
            return fidelity(np.stack([g_.close.values for _, g_ in sd.groupby("sample")]), rr)
        try:
            s0, s1 = saved(REF), saved(RUN_TAG)
            rows[-1].update(saved_d_sig=s1[:, 0].mean() - s0[:, 0].mean(),
                            saved_acf_ratio=s1[:, 2].mean() / s0[:, 2].mean(),
                            saved_d_ks=s1[:, 1].mean() - s0[:, 1].mean())
        except FileNotFoundError:
            pass  # saved synthetic paths are not part of the public release
        print(f"{t}: dσ={rows[-1]['d_sig']:+.3f} acf×{rows[-1]['acf_ratio']:.2f} dKS={rows[-1]['d_ks']:+.3f} "
              f"Σpb={rows[-1]['sum_pb_all_pct_yr']:+.1f}%/yr calib={calib_share:.2f} "
              f"attr={attr_rows[-1]['share_exact']:.4f}", flush=True)

    d = pd.DataFrame(rows); a = pd.DataFrame(attr_rows)
    ok = (d.d_sig.abs() <= 0.05) & (d.acf_ratio >= 0.7) & (d.d_ks <= 0.02)
    fid_pass = int(ok.sum())
    attr_pass = bool((((a.share_exact >= 0.999) | (a.misses_other == 0))
                      & (a.inactive_nonzero == 0) & (a.invisible_removals == 0)).all())
    ok_saved = ((d.saved_d_sig.abs() <= 0.05) & (d.saved_acf_ratio >= 0.7) & (d.saved_d_ks <= 0.02)) if "saved_d_sig" in d else None
    summary = {
        "run": RUN_TAG, "reference": f"{REF} regenerated (legacy mode, today's code)",
        "paths": args.paths, "seed": SEED, "n_tickers": len(d),
        "fidelity_pass": f"{fid_pass}/{len(d)}", "fidelity_gate": fid_pass >= 26,
        "sigma_fails": int((d.d_sig.abs() > 0.05).sum()), "acf_fails": int((d.acf_ratio < 0.7).sum()),
        "ks_fails": int((d.d_ks > 0.02).sum()),
        "attribution_gate": attr_pass,
        "attribution_min_share_exact": float(a.share_exact.min()),
        "attribution_clip_exceptions": int(a.clip_exceptions.sum()),
        "fidelity_vs_saved_R0_pass": f"{int(ok_saved.sum())}/{len(d)}" if ok_saved is not None else "not available (saved paths not released)",
        "median_sum_pb_all_pct_yr": float(d.sum_pb_all_pct_yr.median()),
        "median_abs_sum_pb_all_pct_yr": float(d.sum_pb_all_pct_yr.abs().median()),
        "median_abs_sum_pb_nonbaseline_pct_yr": float(d.sum_pb_nonbaseline_pct_yr.abs().median()),
        "median_calib_share": float(d.calib_share.median()),
        "verdict": "PASS" if (fid_pass >= 26 and attr_pass) else "FAIL",
    }
    out = OUT / f"implementation_checks__{RUN_TAG}"
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "fidelity_per_ticker.csv", index=False)
    a.to_csv(out / "attribution_per_ticker.csv", index=False)
    save_json(summary, out / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
