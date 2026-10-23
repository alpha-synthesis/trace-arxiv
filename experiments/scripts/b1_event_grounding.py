#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""B1 event grounding: agreement of label directions with returns, against the independence baseline.

Criteria v2 and v2.2. Label sets:
  A0   = r0       (prices in the prompt; R1's labels)
  A1   = a1   (price-blind, old store: duplicated, FRED-contaminated)
  R2 = r2_datasign (price-blind, corpus contexts v2.1; primary)

Same measure as v1: over windows with at least one non-noise factor, the
share whose magnitude-weighted direction has the sign of the realized window
return (close over open), against the independence baseline
P(up)P(bull) + P(down)P(bear). Pooled over windows; readings unchanged:
<= 0.70 "prices/move sentences drove most of A0's agreement", >= 0.80
"agreement survives; channel unresolved", otherwise partial.

Added in v2:
  * market-reading share: share of non-noise factor occurrences whose NAME
    matches MARKET_READING (defined below, fixed before any R2 number);
  * share of windows labelled only with market_noise / noise_residual;
  * next-window placebo: window i's direction against window i+1's return
    (expected at the independence baseline);
  * a label-shuffle null: agreement with window returns permuted within
    ticker (1,000 draws), for a p-value of agreement - independence.
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT, save_json  # noqa: E402

# TRACE_B1_ARMS (JSON object, arm -> run tag under common.ROOT) replaces the
# arms (positive control, 2026-10-08); the tickers are those of the first arm.
ARMS = json.loads(os.environ["TRACE_B1_ARMS"]) if os.environ.get("TRACE_B1_ARMS") else \
    {"A0": "r0", "A1": "a1", "R2": "r2_datasign"}
NOISE = {"market_noise", "noise_residual"}
MARKET_READING = re.compile(
    r"(vix|volatility_index|implied_vol|treasury_(yield|rate|note|bond)|bond_yield|yield_(curve|spread|rise|fall|increase|decrease)|"
    r"t10y|dgs10|10y|2y|stock_market|equity_market|market_(sell|rally|crash|decline|downturn|correction|rebound|recovery|volatility|sentiment|momentum|performance)|"
    r"technical|momentum|sell_?off|rally|s_?p_?500|sp500|dow_jones|\bdow\b|nasdaq|index_(gain|loss|decline|rise)|"
    r"price_(decline|drop|increase|rally|surge|fall|rise|momentum|movement|action|volatility)|share_price|stock_price|"
    r"stock_(decline|rally|surge|drop|performance|volatility|rebound)|bull_market|bear_market|investor_sentiment|risk_(on|off))"
)


def windows(run: str, ticker: str):
    f = ROOT / run / ticker / "analyses.json"
    if not f.exists():
        return None
    out = []
    for w in sorted(json.load(open(f)), key=lambda x: x["start_date"]):
        fs = [x for x in w["factors"] if x["name"] not in NOISE and x["magnitude"] > 0]
        s = sum(x["direction"] * x["magnitude"] for x in fs)
        r = float(w["price_change_pct"])
        out.append({"scored": bool(fs) and s != 0 and r != 0, "s": s, "r": r,
                    "noise_only": not fs,
                    "names": [x["name"] for x in fs]})
    return out


def _agree(pairs):
    pairs = [(s, r) for s, r in pairs if s != 0 and r != 0]
    if not pairs:
        return None, None, 0
    s, r = np.array(pairs).T
    up, bull = (r > 0).mean(), (s > 0).mean()
    return float((np.sign(s) == np.sign(r)).mean()), float(up * bull + (1 - up) * (1 - bull)), len(pairs)


def arm_report(run, tickers, rng):
    per, all_pairs, nxt_pairs, by_t = {}, [], [], {}
    n_occ = n_mkt = n_win = n_noise_only = 0
    for t in tickers:
        ws = windows(run, t)
        if ws is None:
            continue
        pairs = [(w["s"], w["r"]) for w in ws if w["scored"]]
        nxt = [(ws[i]["s"], ws[i + 1]["r"]) for i in range(len(ws) - 1) if ws[i]["scored"]]
        a, ind, n = _agree(pairs)
        per[t] = {"agreement": a, "independence": ind, "n_scored": n}
        all_pairs += pairs; nxt_pairs += nxt; by_t[t] = pairs
        n_win += len(ws); n_noise_only += sum(w["noise_only"] for w in ws)
        for w in ws:
            n_occ += len(w["names"]); n_mkt += sum(bool(MARKET_READING.search(x)) for x in w["names"])
    a, ind, n = _agree(all_pairs)
    an, indn, nn = _agree(nxt_pairs)
    null = []
    for _ in range(1000):
        sh = []
        for p in by_t.values():
            if p:
                s, r = np.array(p).T
                sh += list(zip(s, rng.permutation(r)))
        null.append(_agree(sh)[0])
    null = np.array(null)
    reading = (None if a is None else "<=0.70: prices/move sentences drove most of A0's agreement" if a <= 0.70
               else ">=0.80: agreement survives; channel unresolved" if a >= 0.80 else "partial (0.70-0.80)")
    return {"pooled": {"agreement": a, "independence": ind, "n_scored": n,
                       "p_label_shuffle": float((null >= a).mean()) if a is not None else None,
                       "null_mean": float(null.mean())},
            "next_window_placebo": {"agreement": an, "independence": indn, "n": nn},
            "market_reading_share": n_mkt / max(n_occ, 1), "n_factor_occurrences": n_occ,
            "noise_only_window_share": n_noise_only / max(n_win, 1), "n_windows": n_win,
            "reading": reading, "per_ticker": per}


def main():
    tickers = sorted(p.name for p in (ROOT / next(iter(ARMS.values()))).iterdir()
                     if (p / "analyses.json").exists())
    rep = {"arms": ARMS}
    for arm, run in ARMS.items():
        rep[arm] = arm_report(run, tickers, np.random.RandomState(0))
    save_json(rep, OUT / "b1_event_grounding" / "summary.json")
    for arm in ARMS:
        r = rep[arm]
        print(arm, json.dumps({k: r[k] for k in ("pooled", "next_window_placebo", "market_reading_share",
                                                  "noise_only_window_share", "reading")}))


if __name__ == "__main__":
    main()
