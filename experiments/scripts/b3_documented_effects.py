#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B3 documented effects: learned event effects against documented effects.

Criteria v2 and v2.1. Two steps:

  1. --classify  : classify factor NAMES (not descriptions) into E1..E5 with
                   word-boundary regexes; write the matched list for an
                   audit blind to the learned signs. No signs are computed.
  2. --score     : after the list is frozen (audited CSV), compute 5-seed
                   majority signs from five-seed impact matrices and apply the
                   rule fixed in advance (pooled >= 0.60, cluster-bootstrap
                   p < 0.01, no class with n >= 20 below 0.50; classes need
                   n >= 15 to be tested). Primary = raw sign; secondary = sign
                   relative to the ticker's median b over non-baseline factors.

Clusters: calendar window of the factor's first activity (errors clustered by
calendar window).

Secondary: --estimator ols takes signs from OLS of the close
return t -> t+1 on an intercept and all presence indicators, all days (no seeds).

Usage:
  TRACE_RUN=r2_datasign python b3_documented_effects.py --classify
  TRACE_RUN=r2_datasign python b3_documented_effects.py --score --variant data \
      --frozen ../results/b3_documented_effects__r2_datasign/classified_FROZEN.csv
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RUN_TAG, build_presence, load_ticker_artifacts, save_json, ticker_df, tickers_28  # noqa: E402

from sde_causal_generator.data_structures import ImpactMatrix  # noqa: E402

FINANCIALS = {"JPM", "GS", "AXP", "TRV", "V"}
ENERGY = {"XOM", "CVX"}
COMPANY = {
    "AAPL": "apple", "AMGN": "amgen", "AMZN": "amazon", "AXP": "american_express|amex",
    "BA": "boeing", "CAT": "caterpillar", "CRM": "salesforce", "CSCO": "cisco",
    "CVX": "chevron", "DIS": "disney", "GS": "goldman", "HD": "home_depot",
    "HON": "honeywell", "IBM": "ibm", "JNJ": "johnson|jnj|j_j", "JPM": "jpmorgan|jp_morgan|chase",
    "KO": "coca_cola|coke", "MCD": "mcdonald", "MMM": "3m|mmm", "MRK": "merck",
    "MSFT": "microsoft", "NKE": "nike", "PG": "procter|p_g", "TRV": "travelers",
    "UNH": "unitedhealth|united_health", "V": "visa", "WMT": "walmart", "XOM": "exxon",
}
BASELINE_NAMES = {"market_noise", "noise_residual"}
FED = r"\b(fed|fomc|federal reserve|powell)\b"
EXCL = re.compile(r"\b(stab\w*|steady|hold|holds|unchanged|pause\w*|fluctuat\w*|uncertain\w*|expect\w*|speculat\w*)\b")
P_TIGHT = re.compile(FED + r".*\b(hike|hikes|hiked|hiking|raise|raises|raised|raising|tighten\w*|hawkish|increase|increases|increased)\b")
P_EASE = re.compile(FED + r".*\b(cut|cuts|cutting|ease|eases|eased|easing|lower|lowers|lowered|lowering|dovish|stimulus|qe)\b")
P_OIL = re.compile(r"\b(oil|crude|opec|brent|wti)\b.*\b(rise|rises|rising|rose|increase|increases|increased|surge\w*|higher|spike\w*|rall\w*|jump\w*)\b"
                   r"|\bopec\b.*\b(cut|cuts)\b|\b(production|output|supply) cuts?\b")
P_BEAT = re.compile(r"\b(earnings|revenue|revenues|profit|profits|eps|results|sales)\b.*\b(beat|beats|strong|stronger|exceed\w*|record|surpass\w*|better|upbeat)\b")
P_MISS = re.compile(r"\b(earnings|revenue|revenues|profit|profits|eps|results|sales)\b.*\b(miss|misses|missed|weak|weaker|below|disappoint\w*|shortfall|worse)\b")
EXPECTED = {"E1_tightening": -1, "E2_easing": +1, "E3_oil_up": +1, "E4_earnings_beat": +1, "E5_earnings_miss": -1}
MIN_N, BIG_N = 15, 20


def classify(name: str, ticker: str):
    txt = name.lower().replace("_", " ")
    if EXCL.search(txt):
        return None
    hits = []
    if ticker not in FINANCIALS:
        if P_TIGHT.search(txt): hits.append("E1_tightening")
        if P_EASE.search(txt): hits.append("E2_easing")
    if ticker in ENERGY and P_OIL.search(txt):
        hits.append("E3_oil_up")
    own = re.search(r"\b(" + COMPANY[ticker].replace("_", " ") + r")\b", txt) is not None
    if own and P_BEAT.search(txt): hits.append("E4_earnings_beat")
    if own and P_MISS.search(txt): hits.append("E5_earnings_miss")
    if len(hits) != 1:
        return "AMBIGUOUS" if hits else None
    return hits[0]


def first_window(P_k, dates, windows):
    idx = np.where(P_k > 0)[0]
    if len(idx) == 0:
        return None
    d0 = dates[idx[0]]
    for i, (s, e) in enumerate(windows):
        if s <= d0 <= e:
            return i
    return None


def do_classify():
    rows = []
    for t in tickers_28():
        im, _, analyses = load_ticker_artifacts(t)
        df = ticker_df(t)
        P = build_presence(analyses, df, im.factor_names)
        occ = (P > 0).mean(axis=0)
        for k, n in enumerate(im.factor_names):
            if n in BASELINE_NAMES or occ[k] > 0.8:
                continue
            c = classify(n, t)
            if c is None:
                continue
            rows.append({"ticker": t, "factor": n, "class": c,
                         "n_active_days": int((P[:, k] > 0).sum()), "keep": int(c != "AMBIGUOUS")})
    out = OUT / f"b3_documented_effects__{RUN_TAG}"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "classified_FOR_AUDIT.csv", index=False)
    print(pd.DataFrame(rows).groupby("class").size().to_string())
    print(f"wrote {out / 'classified_FOR_AUDIT.csv'} — audit 'keep' blind to signs, save as classified_FROZEN.csv")


def _ols_coefs(P, close):
    r = np.diff(np.log(np.clip(close.astype(np.float64), 1e-8, None)))
    A = (P[:-1] > 0).astype(np.float64)
    keep = A.sum(axis=0) > 0
    coef = np.zeros(A.shape[1])
    coef[keep] = np.linalg.lstsq(np.column_stack([np.ones(len(r)), A[:, keep]]), r, rcond=None)[0][1:]
    return coef


def do_score(variant: str, frozen: str, estimator: str = "tft"):
    fz = pd.read_csv(frozen)
    fz = fz[(fz["keep"] == 1) & (fz["class"] != "AMBIGUOUS")]
    ms_root = OUT / f"five_seed__{RUN_TAG}__{variant}"
    rows = []
    for t, g in fz.groupby("ticker"):
        im0, _, analyses = load_ticker_artifacts(t)
        df = ticker_df(t)
        if estimator == "ols":
            names = im0.factor_names
            Pf = build_presence(analyses, df, names)
            B = _ols_coefs(Pf, df["close"].values)[None, :]           # (1, K)
        else:
            mats = [ImpactMatrix.load(str(f)) for f in sorted((ms_root / t).glob("impact_matrix_seed*.json"))]
            assert len(mats) == 5, f"{t}: need 5 seeds, found {len(mats)}"
            names = mats[0].factor_names
            B = np.stack([m.base_impact[:, 3] for m in mats])      # (5, K)
        P = build_presence(analyses, df, names)
        occ = (P > 0).mean(axis=0)
        nonbase = np.array([(n not in BASELINE_NAMES) and (o <= 0.8) for n, o in zip(names, occ)])
        med = np.median(B[:, nonbase], axis=1, keepdims=True)
        dates = pd.to_datetime(sorted(df["date"].unique()))
        windows = [(pd.Timestamp(a.start_date), pd.Timestamp(a.end_date)) for a in analyses]
        for _, r in g.iterrows():
            k = names.index(r["factor"])
            maj = int(np.sign(np.sign(B[:, k]).sum()))
            maj_c = int(np.sign(np.sign(B[:, k] - med[:, 0]).sum()))
            rows.append({"ticker": t, "factor": r["factor"], "class": r["class"],
                         "expected": EXPECTED[r["class"]], "sign": maj, "sign_centered": maj_c,
                         "seed_agreement": float(np.abs(np.sign(B[:, k]).mean())),
                         "cluster": first_window(P[:, k], dates, windows)})
    d = pd.DataFrame(rows)
    rng = np.random.RandomState(0)

    def stats(df_, col):
        v = df_[df_[col] != 0]
        if len(v) == 0:
            return None
        agree = (v[col] == v["expected"]).astype(float)
        cl = v["cluster"].fillna(-1).values
        uc = np.unique(cl)
        boots = []
        for _ in range(2000):
            pick = rng.choice(uc, size=len(uc), replace=True)
            m = np.concatenate([agree.values[cl == c] for c in pick])
            boots.append(m.mean())
        boots = np.array(boots)
        return {"n": int(len(v)), "agree": float(agree.mean()),
                "p_cluster_boot": float((boots <= 0.5).mean()), "n_clusters": int(len(uc))}

    report = {"run": RUN_TAG, "variant": variant, "estimator": estimator, "frozen_list": frozen}
    for col in ("sign", "sign_centered"):
        per = {c: stats(d[d["class"] == c], col) for c in EXPECTED}
        pooled = stats(d[d["class"].isin([c for c, s in per.items() if s and s["n"] >= MIN_N])], col)
        bad = [c for c, s in per.items() if s and s["n"] >= BIG_N and s["agree"] < 0.5]
        verdict = ("PASS" if pooled and pooled["agree"] >= 0.60 and pooled["p_cluster_boot"] < 0.01 and not bad
                   else "FAIL")
        report[col] = {"per_class": per, "pooled_tested_classes": pooled, "classes_n20_below_half": bad,
                       "verdict": verdict}
    report["primary_verdict"] = report["sign"]["verdict"]
    report["median_seed_agreement"] = float(d["seed_agreement"].median())
    out = OUT / (f"b3_documented_effects__{RUN_TAG}__{variant}" + ("" if estimator == "tft" else f"__{estimator}"))
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "rows.csv", index=False)
    save_json(report, out / "report.json")
    print(json.dumps(report, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--variant", choices=["data", "llm"], default="data")
    ap.add_argument("--frozen", default="")
    ap.add_argument("--estimator", choices=["tft", "ols"], default="tft")
    a = ap.parse_args()
    if a.classify:
        do_classify()
    elif a.score:
        do_score(a.variant, a.frozen, a.estimator)
    else:
        ap.error("--classify or --score")


if __name__ == "__main__":
    main()
