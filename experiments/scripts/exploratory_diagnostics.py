#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exploratory diagnostics quoted in the TRACE paper, recomputed from frozen artifacts.

Several exploratory numbers in the paper were first computed by ad hoc
code. This script recomputes them from the saved
labels, impact matrices, contexts and test results, so that every quoted
number comes from a released script. Part 1 reads results only. Part 2 calls
the generator's own calibration functions on each run's saved price panel;
it draws innovations but trains nothing and simulates no price paths.

Part 1 (artifacts only):
  label_information   R2 labels: kappa, P(bull) vs P(up), alignment with the
                      equal-weighted market return of the other 27 stocks,
                      cross-sectional rank IC with stock-specific returns
                      (Section 4.2)
  crash_window        R2 labels of the window starting 2020-03-01: share of
                      bullish stocks and the equal-weighted return
  company_contexts    share of contexts with a company-specific section
  factor_agreement    factor-level direction agreement, R1 and R2 (Section 4.5)
  presence            R2 coverage of the 20-active-day rule, factor-day share,
                      median eligible factors per stock, identical presence
  loo_rolling_origin  leave-one-origin-out p-values, R1 primary TFT test
  effect_scale        a 1% one-day move spread over ten days, in s.d.
  excess_upper_bound  in-sample excess returns of labelled factors, corrected
                      for sampling noise (spread of effect sizes)
  shrinkage           slope of recovered on planted effects (seed 0)
  b2_summaries        from the b2_counterfactual outputs: realized-to-expected ratio
                      range, median single-factor effect (%/yr), share under
                      1%/yr, largest per-stock KS difference R2 - R1 (Section 4.3)
  name_normalization  R2 recurrence after normalizing spelling (case,
                      separators, plurals): single-window share of names and
                      stock-origins with five eligible factors (Section 4.5)
  tft_best_epoch      best epoch of the generator's TFT (the run's own impact
                      network, pipeline seed 42, NLL selection), from each
                      stock's pipeline_report.json, R1 and R2 (Appendix,
                      implementation pitfalls)
  planted_sd_ratio    s.d. of the simulated GJR-GARCH-t returns of the
                      planted panels (before planting) over the real s.d.
                      (Appendix, planted-effect check)
  initial_noise_profile  R0 (r0): per-factor noise
                      (impact_std, close) over the daily return s.d., and the
                      share of each effect re-added by the temporal profile
                      over lags 1-9 (Appendix, implementation pitfalls)
  b2_two_sided        stocks whose |terminal tracking error| differs from
                      R1's by at most 0.05 in either direction (Section 4.3)
  documented_rows     R2 documented-effect rows: 3-to-2 seed splits among the
                      tested rows, and calendar windows per class (Section 4.5)
  run_time            R2 data-sign run: total and extraction time (Appendix)
  ideal_reference     Phi(beta * sqrt(10)) at the quoted beta (Eq. ideal)
  b1_bootstrap_p      R2 excess agreement: calendar-window cluster bootstrap
                      (as make_figures._cluster_ci, 4,000 draws, seed 0),
                      percentile p-values; A0 agreement without the five
                      fallback stocks (Section 4.2)
  fallback_stocks     A0 stocks labelled by the no-news fallback prompt, whose
                      windows carry its noise_residual factor (Appendix)

Part 2 (generator calibration code, run's own training_data.csv):
  innovation_kurtosis excess kurtosis of the skewed-t innovations after the
                      generator standardizes the 10 paths drawn at each date
                      (per-stock df and skew from the generator's estimator)
  jump_intensity      segments whose jump intensity is raised (mean below
                      -0.02%/day or s.d. above 1.3x overall): stocks and
                      intensity multipliers
  drift_correction    share of the segment mean-return gap closed by the
                      regime drift calibration, by sign of the real mean

Usage: python exploratory_diagnostics.py
"""
from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ORIG, OUT, build_presence, save_json, ticker_df, tickers_28  # noqa: E402

NOISE = {"market_noise", "noise_residual"}
R1, R2 = "r1_datasign", "r2_datasign"
CTX = ORIG / "cache" / "contexts_v2.1"
DST = OUT / "exploratory_diagnostics"


def _windows(run, ticker):
    f = ORIG / "results" / run / ticker / "analyses.json"
    out = []
    for w in sorted(json.load(open(f)), key=lambda x: x["start_date"]):
        fs = [x for x in w["factors"] if x["name"] not in NOISE and x["magnitude"] > 0]
        out.append({"start": w["start_date"], "r": float(w["price_change_pct"]),
                    "s": sum(x["direction"] * x["magnitude"] for x in fs), "factors": fs})
    return out


def _agree(s, r):
    s, r = np.asarray(s, float), np.asarray(r, float)
    keep = (s != 0) & (r != 0)
    s, r = s[keep], r[keep]
    up, bull = (r > 0).mean(), (s > 0).mean()
    a = float((np.sign(s) == np.sign(r)).mean())
    ind = float(up * bull + (1 - up) * (1 - bull))
    return a, ind, int(keep.sum()), float(bull), float(up)


def label_information(tickers):
    rows = []
    for t in tickers:
        for w in _windows(R2, t):
            rows.append({"ticker": t, "start": w["start"], "s": w["s"], "r": w["r"]})
    df = pd.DataFrame(rows)
    a, ind, n, bull, up = _agree(df.s, df.r)
    kappa = (a - ind) / (1 - ind)
    g = df.groupby("start")["r"]
    df["mkt"] = g.transform("mean")
    df["mkt_other"] = (g.transform("sum") - df.r) / (g.transform("count") - 1)
    am, indm, nm, _, _ = _agree(df.s, df.mkt)
    ao, indo, _, _, _ = _agree(df.s, df.mkt_other)
    df["spec"] = df.r - df.mkt
    ics = []
    for _, g in df[df.s != 0].groupby("start"):
        if len(g) >= 5 and g.s.nunique() > 1:
            ics.append(spearmanr(g.s, g.spec).correlation)
    ics = np.array([x for x in ics if np.isfinite(x)])
    return {"agreement": a, "independence": ind, "n_scored": n, "kappa": kappa,
            "p_bull": bull, "p_up": up,
            "market_alignment_excess": am - indm, "market_alignment_n": nm,
            "market_other27_alignment_excess": ao - indo,
            "rank_ic_mean": float(ics.mean()), "rank_ic_t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))),
            "rank_ic_windows": int(len(ics))}


def crash_window(tickers, start="2020-03-01"):
    rows = [w for t in tickers for w in _windows(R2, t) if w["start"] == start]
    s = np.array([w["s"] for w in rows]); r = np.array([w["r"] for w in rows])
    return {"start": start, "stocks": len(rows), "bullish": int((s > 0).sum()), "bearish": int((s < 0).sum()),
            "bullish_share": float((s > 0).mean()), "equal_weighted_return_pct": float(r.mean())}


def company_contexts():
    n = hit = 0
    for f in sorted(glob.glob(str(CTX / "[A-Z]*.json"))):
        j = json.load(open(f))
        if "windows" not in j:
            continue
        tag = f"── {j['ticker']}-SPECIFIC NEWS ──"
        for w in j["windows"]:
            n += 1
            hit += tag in w["context"]
    return {"contexts": n, "with_company_section": hit, "share": hit / n}


def factor_agreement(run, tickers):
    agree = n = 0
    for t in tickers:
        for w in _windows(run, t):
            if w["r"] == 0:
                continue
            for x in w["factors"]:
                if x["direction"] != 0:
                    n += 1
                    agree += np.sign(x["direction"]) == np.sign(w["r"])
    return {"occurrences": n, "agreement": agree / n}


def _presence(run, ticker):
    from sde_causal_generator.data_structures import ImpactMatrix, WindowAnalysis
    im = ImpactMatrix.load(str(ORIG / "results" / run / ticker / "impact_matrix.json"))
    analyses = [WindowAnalysis.from_dict(d) for d in json.load(open(ORIG / "results" / run / ticker / "analyses.json"))]
    names = list(im.factor_names)
    df = ticker_df(ticker)
    P = build_presence(analyses, df, names)
    return names, P, df


def presence(run, tickers):
    n_nb = n_el = days_nb = days_el = n_shared = 0
    per_stock = []
    for t in tickers:
        names, P, _ = _presence(run, t)
        act = P > 0
        occ = act.mean(axis=0)
        base = np.array([(nm in NOISE) or (o > 0.8) for nm, o in zip(names, occ)])
        nb = np.where(~base & act.any(axis=0))[0]
        el = [k for k in nb if act[:, k].sum() >= 20]
        n_nb += len(nb); n_el += len(el); per_stock.append(len(el))
        days_nb += int(act[:, nb].sum()); days_el += int(act[:, el].sum()) if el else 0
        cols = {}
        for k in range(len(names)):
            if act[:, k].any():
                cols.setdefault(act[:, k].tobytes(), []).append(k)
        n_shared += sum(len(cols[act[:, k].tobytes()]) > 1 for k in nb)
    return {"non_baseline_factors": n_nb, "eligible_20d": n_el, "eligible_share": n_el / n_nb,
            "eligible_factor_day_share": days_el / days_nb, "median_eligible_per_stock": float(np.median(per_stock)),
            "identical_presence_share": n_shared / n_nb}


def loo_rolling_origin():
    od = OUT / f"b3_rolling_origin__{R1}__data"
    per = {o: json.loads((od / f"origin_{o}.json").read_text())["per_ticker"] for o in ("2021", "2022", "2023")}

    def agree(row, arm):
        m = (row.get(arm) or {}).get("metric") or {}
        return m["agreement"] if m and m.get("n_factors", 0) >= 5 else None

    diffs = {o: {r["ticker"]: agree(r, "control") - agree(r, "window_permuted") for r in rows
                 if agree(r, "control") is not None and agree(r, "window_permuted") is not None}
             for o, rows in per.items()}
    out = {}
    for left in per:
        by_t = {}
        for o, d in diffs.items():
            if o == left:
                continue
            for t, v in d.items():
                by_t.setdefault(t, []).append(v)
        pooled = [float(np.mean(v)) for v in by_t.values() if len(v) >= 2]
        p = float(wilcoxon(pooled, alternative="greater").pvalue) if len(pooled) >= 6 and any(pooled) else None
        out[f"without_{left}"] = {"n_stocks": len(pooled), "p_one_sided": p}
    by_t = {}
    for d in diffs.values():
        for t, v in d.items():
            by_t.setdefault(t, []).append(v)
    pooled = [float(np.mean(v)) for v in by_t.values() if len(v) >= 2]
    out["all_origins"] = {"n_stocks": len(pooled), "mean_diff": float(np.mean(pooled)),
                          "p_one_sided": float(wilcoxon(pooled, alternative="greater").pvalue)}
    return out


def effect_scale():
    out = {}
    for run in (R1, R2):
        sig, days = [], []
        for f in sorted(glob.glob(str(OUT / f"sign_recovery__{run}" / "ticker_*.json"))):
            j = json.load(open(f))
            sig.append(j["sigma"])
            adp = j["active_days_planted"]
            days += list(adp.values()) if isinstance(adp, dict) else list(adp)
        eff = 0.01 / 10 / np.array(sig)
        out[run] = {"median_sd": float(np.median(eff)), "min_sd": float(eff.min()), "max_sd": float(eff.max()),
                    "one_day_move_sd_median": float(np.median(0.01 / np.array(sig))),
                    "planted_active_days_median": float(np.median(days)), "planted_active_days_min": int(min(days))}
    return out


def excess_upper_bound(run, tickers, min_days=20):
    z, inv_n = [], []
    for t in tickers:
        names, P, df = _presence(run, t)
        close = df["close"].values.astype(float)
        r = np.diff(np.log(close))          # r[j]: day j -> j+1 (training convention)
        act = P[:-1] > 0
        sd, mu = r.std(), r.mean()
        occ = act.mean(axis=0)
        for k, nm in enumerate(names):
            nk = int(act[:, k].sum())
            if nm in NOISE or occ[k] > 0.8 or nk < min_days:
                continue
            z.append((r[act[:, k]].mean() - mu) / sd)
            inv_n.append(1.0 / nk)
    z, inv_n = np.array(z), np.array(inv_n)
    var_true = max(0.0, z.var(ddof=1) - inv_n.mean())
    return {"factors": int(len(z)), "sd_observed": float(z.std(ddof=1)), "noise_sd": float(np.sqrt(inv_n.mean())),
            "sd_true_effect": float(np.sqrt(var_true)),
            "p90_abs_observed": float(np.quantile(np.abs(z), 0.9))}


def shrinkage(run=R1):
    out = {}
    files = sorted(glob.glob(str(OUT / f"sign_recovery__{run}" / "ticker_*.json")))
    for beta in ("0.05", "0.1", "0.2", "0.3"):
        num = den = 0.0
        for f in files:
            j = json.load(open(f))
            rec = np.array(j["per_beta"][beta]["recovered_sigma"], float)
            sg = np.array(j["signs"], float)
            num += float((rec * sg).sum()); den += float(len(sg)) * float(beta)
        out[beta] = num / den
    return out


def b2_summaries():
    out = {}
    pt = {}
    for arm, run in (("R1", R1), ("R2", R2)):
        d = OUT / f"b2_counterfactual__{run}"
        pf = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(str(d / "per_factor_*.csv")))])
        eff = pf.d_path.abs() * 252 * 100          # mean daily change -> % per year
        out[arm] = {"factors": int(len(pf)), "ratio_min": float(pf.ratio.min()), "ratio_max": float(pf.ratio.max()),
                    "median_effect_pct_yr": float(eff.median()), "share_under_1pct_yr": float((eff < 1).mean())}
        pt[arm] = pd.read_csv(d / "per_ticker.csv").set_index("ticker")
    diff = (pt["R2"].ks_factual - pt["R1"].ks_factual).abs()
    out["ks_max_abs_diff_R2_R1"] = float(diff.max())
    return out


def _normalize(name):
    toks = [t for t in re.split(r"[^a-z0-9]+", name.lower()) if t]
    return "_".join(t[:-1] if len(t) > 3 and t.endswith("s") and not t.endswith("ss") else t for t in toks)


def _merge(P, names, key):
    keys = sorted({key(n) for n in names})
    idx = {k: i for i, k in enumerate(keys)}
    M = np.zeros((P.shape[0], len(keys)), dtype=P.dtype)
    for k, n in enumerate(names):
        M[:, idx[key(n)]] = np.maximum(M[:, idx[key(n)]], P[:, k])
    return M, keys


def name_normalization(run=R2, tickers=None):
    """Same eligibility as b3_rolling_origin (>=5 training days, <=80% occupancy, non-baseline,
    >=5 test-year days, >=5 factors per stock-origin), before and after normalizing."""
    from sde_causal_generator.data_structures import WindowAnalysis
    out = {}
    for label, key in (("raw", lambda n: n), ("normalized", _normalize)):
        single = total = 0
        qualifying = n_so = 0
        for t in tickers:
            raw = json.load(open(ORIG / "results" / run / t / "analyses.json"))
            c = {}
            for w in raw:
                for k in {key(x["name"]) for x in w["factors"] if x["magnitude"] > 0}:
                    c[k] = c.get(k, 0) + 1
            single += sum(v == 1 for v in c.values()); total += len(c)
            analyses = [WindowAnalysis.from_dict(d) for d in raw]
            names = sorted({f.name for a in analyses for f in a.factors if f.magnitude > 0})
            df = ticker_df(t)
            dates = pd.to_datetime(df["date"])
            P_all, keys = _merge(build_presence(analyses, df, names), names, key)
            for origin in ("2021-01-01", "2022-01-01", "2023-01-01"):
                o = pd.Timestamp(origin)
                n_is = int((dates < o).sum()); n_end = int((dates < o + pd.DateOffset(years=1)).sum())
                an_train = [a for a in analyses if pd.Timestamp(a.end_date) < o]
                P_tr, _ = _merge(build_presence(an_train, df, names), names, key)
                act_is = P_tr[:n_is] > 0
                occ, is_days = act_is.mean(axis=0), act_is.sum(axis=0)
                test_days = (P_all[n_is:n_end - 1] > 0).sum(axis=0)
                base = {key(n) for n in NOISE}
                n_el = sum((k not in base) and occ[i] <= 0.8 and is_days[i] >= 5 and test_days[i] >= 5
                           for i, k in enumerate(keys))
                n_so += 1; qualifying += n_el >= 5
        out[label] = {"names": total, "single_window_share": single / total,
                      "stock_origins": n_so, "with_five_eligible": int(qualifying)}
    return out


def tft_best_epoch(run, tickers):
    """The saved best_epoch is the 0-based epoch index of train_impact_network."""
    ep, seeds = {}, set()
    for t in tickers:
        j = json.load(open(ORIG / "results" / run / t / "pipeline_report.json"))
        ep[t] = int(j["phases"]["impact_learning"]["best_epoch"])
        seeds.add(j["config"]["seed"])
    e = np.array(list(ep.values()))
    return {"seed": sorted(seeds), "stocks": len(e), "median_zero_based": float(np.median(e)),
            "at_most_7_one_based": int((e + 1 <= 7).sum()), "at_most_7_zero_based": int((e <= 7).sum()),
            "best_epoch_zero_based": ep}


def planted_sd_ratio(run):
    r = np.array([j["sim_sd"] / j["sigma"] for j in
                  (json.load(open(f)) for f in sorted(glob.glob(str(OUT / f"sign_recovery__{run}" / "ticker_*.json"))))])
    return {"stocks": len(r), "median": float(np.median(r)), "min": float(r.min()), "max": float(r.max())}


def initial_noise_profile(tickers, run="r0"):
    ratio, prof = [], []
    for t in tickers:
        im = json.load(open(ORIG / "results" / run / t / "impact_matrix.json"))
        sd = float(np.diff(np.log(ticker_df(t)["close"].values.astype(float))).std())
        ratio.append(float(np.median([x[3] for x in im["impact_std"]])) / sd)
        prof += [float(np.sum(tp[1:10])) for tp in im["temporal_profile"]]
    return {"noise_over_return_sd_median": float(np.median(ratio)), "noise_over_return_sd_min": float(min(ratio)),
            "noise_over_return_sd_max": float(max(ratio)), "factors": len(prof),
            "profile_lags_1_9_median": float(np.median(prof)), "profile_lags_1_9_mean": float(np.mean(prof))}


def b2_two_sided(tol=0.05):
    a = pd.read_csv(OUT / f"b2_counterfactual__{R1}" / "per_ticker.csv").set_index("ticker").track_err_factual.abs()
    b = pd.read_csv(OUT / f"b2_counterfactual__{R2}" / "per_ticker.csv").set_index("ticker").track_err_factual.abs()
    return {"stocks": int(len(b)), "one_sided_pass": int(((b - a) <= tol).sum()),
            "two_sided_pass": int(((b - a).abs() <= tol).sum())}


def documented_rows():
    rows = pd.read_csv(OUT / f"b3_documented_effects__{R2}__data" / "rows.csv")
    tested = rows[rows["class"].isin(["E1_tightening", "E2_easing"])]
    starts = [w["start_date"] for w in sorted(json.load(open(ORIG / "results" / R2 / "AAPL" / "analyses.json")),
                                              key=lambda x: x["start_date"])]
    windows = {c: {starts[int(k)]: int(n) for k, n in g.groupby("cluster").size().items()}
               for c, g in rows.groupby("class")}
    return {"tested_rows": int(len(tested)), "split_3_2": int(np.isclose(tested.seed_agreement, 0.2).sum()),
            "split_3_2_share": float(np.isclose(tested.seed_agreement, 0.2).mean()),
            "median_seed_agreement": float(tested.seed_agreement.median()), "rows_by_window_start": windows}


def run_time(tickers, run=R2):
    g = json.load(open(ORIG / "results" / run / "generation_summary.json"))
    ext = sum(json.load(open(ORIG / "results" / run / t / "pipeline_report.json"))["phases"]["extraction"]["time_seconds"]
              for t in tickers)
    return {"total_min": g["total_time_seconds"] / 60, "extraction_min": ext / 60}


def fallback_stocks(tickers, run="r0"):
    out = {}
    for t in tickers:
        a = json.load(open(ORIG / "results" / run / t / "analyses.json"))
        n = sum(any(x["name"] == "noise_residual" for x in w["factors"]) for w in a)
        if n:
            out[t] = {"windows_with_fallback": n, "windows": len(a)}
    return out


def b1_bootstrap_p(n_boot=4000, seed=0):
    from make_figures import _arm_windows, _excess
    df = _arm_windows(R2)
    d = df[(df.s != 0) & (df.r != 0)]
    s, r = np.sign(d.s.values), np.sign(d.r.values)
    wins = np.array(sorted(d.win.unique()))
    idx = np.searchsorted(wins, d.win.values)
    W = len(wins)
    n_w = np.bincount(idx, minlength=W).astype(float)
    a_w = np.bincount(idx, weights=(s == r).astype(float), minlength=W)
    u_w = np.bincount(idx, weights=(r > 0).astype(float), minlength=W)
    b_w = np.bincount(idx, weights=(s > 0).astype(float), minlength=W)
    rng = np.random.RandomState(seed)
    ex = np.empty(n_boot)
    for i in range(n_boot):
        c = np.bincount(rng.randint(0, W, W), minlength=W)
        n = c @ n_w
        up, bull = (c @ u_w) / n, (c @ b_w) / n
        ex[i] = (c @ a_w) / n - (up * bull + (1 - up) * (1 - bull))
    one = float((ex <= 0).mean())
    a0 = json.load(open(OUT / "b1_event_grounding" / "summary.json"))["A0"]["per_ticker"]
    five = set(fallback_stocks(tickers_28()))
    split = {}
    for name, keep in (("fallback_five", lambda t: t in five), ("other", lambda t: t not in five)):
        rows = [v for t, v in a0.items() if keep(t)]
        n = sum(v["n_scored"] for v in rows)
        split[name] = {"stocks": len(rows), "n_scored": n,
                       "agreement": sum(v["agreement"] * v["n_scored"] for v in rows) / n,
                       "independence": sum(v["independence"] * v["n_scored"] for v in rows) / n}
    return {"excess": _excess(df, "r")[2], "ci95": [float(np.percentile(ex, 2.5)), float(np.percentile(ex, 97.5))],
            "p_one_sided": one, "p_two_sided": float(min(1.0, 2 * min(one, (ex >= 0).mean()))),
            "A0_split": split}


def ideal_reference(n=10):
    from scipy.stats import norm
    return {str(b): float(norm.cdf(b * np.sqrt(n))) for b in (0.06, 0.2, 0.3)}


# ── Part 2: the generator's calibration code on each run's saved prices ──

N_PATHS = 10          # paths simulated together (b2_counterfactual, the pipeline's gen_n_samples)


def _real_returns(run):
    panel = pd.read_csv(ORIG / "results" / run / "training_data.csv")
    for t, df in panel.groupby("tic"):
        rp = np.clip(df.sort_values("date")[["open", "high", "low", "close", "volume"]].values.astype(float), 1e-8, None)
        yield t, np.diff(np.log(rp), axis=0)


def innovation_kurtosis(run, n_dates=20000, seed=0):
    """Per stock, n_dates draws of _sample_skewed_t(df, skew, 10), as in the GARCH loop.
    Each date is standardized to mean 0 and variance 1, so the pooled excess
    kurtosis is mean(z^4) - 3."""
    from sde_causal_generator.generate import _estimate_skewed_t_params, _sample_skewed_t
    rows = []
    for t, rlr in _real_returns(run):
        df_, skew = _estimate_skewed_t_params(rlr[:, 3])
        rng = np.random.RandomState(seed)
        z = np.stack([_sample_skewed_t(df_, skew, N_PATHS, rng=rng) for _ in range(n_dates)])
        m4 = (z ** 4).mean(axis=1)
        rows.append((t, df_, skew, m4.mean() - 3, m4.std() / np.sqrt(n_dates), np.abs(z).max()))
    k = np.array([x[3] for x in rows])
    return {"stocks": len(rows), "n_paths": N_PATHS, "n_dates": n_dates,
            "excess_kurtosis_median": float(np.median(k)), "excess_kurtosis_min": float(k.min()),
            "excess_kurtosis_max": float(k.max()), "mc_se_max": float(max(x[4] for x in rows)),
            "max_abs_z": float(max(x[5] for x in rows)), "bound_sqrt_n_minus_1": float(np.sqrt(N_PATHS - 1)),
            "gaussian_reference": 3 * (N_PATHS - 1) / (N_PATHS + 1) - 3,
            "df_median": float(np.median([x[1] for x in rows])), "df_at_floor_5": int(sum(x[1] == 5.0 for x in rows)),
            "df_max": float(max(x[1] for x in rows)), "skew_min": float(min(x[2] for x in rows)),
            "skew_max": float(max(x[2] for x in rows)),
            "by_stock": {x[0]: {"df": x[1], "skew": x[2], "excess_kurtosis": float(x[3])} for x in rows}}


def jump_intensity(run):
    # mirrors the inline rule of ImpactDrivenGenerator.generate_scenario (step 6, R5)
    from sde_causal_generator.generate import _detect_regimes, _estimate_jump_params
    raised, n_seg, jumps_off = {}, [], []
    for t, rlr in _real_returns(run):
        lam, _, sig = _estimate_jump_params(rlr[:, :4])
        if lam <= 0 or sig <= 0:
            jumps_off.append(t)
        r = rlr[:, 3]
        g = float(r.std())
        regs = _detect_regimes(r)
        n_seg.append(len(regs))
        m = [float(np.clip(3.0 + 2.0 * max(0, sd / g - 1.0), 3.0, 5.0))
             for _, _, mu, sd in regs if sd > g * 1.3 or mu < -0.0002]
        if m and lam > 0 and sig > 0:
            raised[t] = {"multiplier": m, "by_volatility": any(sd > g * 1.3 for _, _, _, sd in regs)}
    mult = [x for v in raised.values() for x in v["multiplier"]]
    return {"stocks": len(n_seg), "segments_per_stock": sorted(set(n_seg)), "jumps_off": jumps_off,
            "stocks_raised": len(raised), "multiplier_min": min(mult), "multiplier_max": max(mult), "raised": raised}


def drift_correction(run, seed=0):
    """_regime_calibrate (blend 0.6, as called by the generator) applied to 10 i.i.d.
    zero-mean normal paths with the stock's s.d.; per segment and path, the share of
    (real mean - synthetic mean) that the shift closes. The share does not depend on
    the input, except for paths reset to the real mean (R3), whose count does."""
    from sde_causal_generator.generate import ImpactDrivenGenerator, _detect_regimes
    share = {"nonneg": [], "neg": []}
    n_seg = {"nonneg": 0, "neg": 0}
    full = 0
    for _, rlr in _real_returns(run):
        r = rlr[:, 3]
        T = len(r) + 1                                   # n_steps = number of price rows
        x = np.random.RandomState(seed).normal(0.0, r.std(), (N_PATHS, T))
        y = ImpactDrivenGenerator._regime_calibrate(x, r, blend=0.6)
        for s, e, mu, _ in _detect_regimes(r):
            a = max(0, min(int(round(s / len(r) * T)), T - 1))
            b = max(a + 1, min(int(round(e / len(r) * T)), T))
            pre, post = x[:, a:b].mean(axis=1), y[:, a:b].mean(axis=1)
            key = "neg" if mu < 0 else "nonneg"
            n_seg[key] += 1
            reset = np.isclose(post, mu, rtol=0, atol=1e-12) & (mu < 0)
            full += int(reset.sum())
            share[key] += list((post - pre)[~reset] / (mu - pre)[~reset])
    return {"segments": n_seg, "paths_per_segment": N_PATHS,
            **{f"share_{k}_min": float(min(v)) for k, v in share.items()},
            **{f"share_{k}_max": float(max(v)) for k, v in share.items()},
            "paths_reset_to_real_mean_on_noise_input": full}


def main():
    tickers = tickers_28()
    res = {
        "label_information_R2": label_information(tickers),
        "company_contexts": company_contexts(),
        "factor_agreement": {"R1": factor_agreement(R1, tickers), "R2": factor_agreement(R2, tickers)},
        "presence_R2": presence(R2, tickers),
        "loo_rolling_origin_R1": loo_rolling_origin(),
        "effect_scale": effect_scale(),
        "excess_upper_bound": {"R1": excess_upper_bound(R1, tickers), "R2": excess_upper_bound(R2, tickers)},
        "shrinkage_R1_seed0": shrinkage(R1),
        "b2_summaries": b2_summaries(),
        "crash_window_R2": crash_window(tickers),
        "name_normalization_R2": name_normalization(R2, tickers),
        "initial_noise_profile": initial_noise_profile(tickers),
        "b2_two_sided": b2_two_sided(),
        "documented_rows_R2": documented_rows(),
        "run_time_R2": run_time(tickers),
        "ideal_reference_n10": ideal_reference(),
        "fallback_stocks_A0": fallback_stocks(tickers),
        "b1_bootstrap_p_R2": b1_bootstrap_p(),
        "tft_best_epoch": {"R1": tft_best_epoch(R1, tickers), "R2": tft_best_epoch(R2, tickers)},
        "planted_sd_ratio": {"R1": planted_sd_ratio(R1), "R2": planted_sd_ratio(R2)},
        "innovation_kurtosis": {"R1": innovation_kurtosis(R1), "R2": innovation_kurtosis(R2)},
        "jump_intensity": {"R1": jump_intensity(R1), "R2": jump_intensity(R2)},
        "drift_correction": {"R1": drift_correction(R1), "R2": drift_correction(R2)},
    }
    save_json(res, DST / "summary.json")
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
