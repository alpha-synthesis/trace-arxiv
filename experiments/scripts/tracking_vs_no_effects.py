#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal tracking of the factual paths against the engine without effects.

The factual-fidelity criterion compares R2 with R1, whose labels were written
with prices in view. This script adds the engine with b = 0 (the vanilla
engine: same calibration and random numbers, no factor term) as a third
reference, using the same seeds (1-3, 10 paths) and the same statistic as
b2_counterfactual: |mean over seeds of the mean signed terminal error|. A stock passes
against a reference when its error is at most 0.05 log units worse.

Reads the per-stock R1/R2 errors saved by exploratory_generator_diagnostics.py
and simulates only the b = 0 paths. Output: results/tracking_vs_no_effects/summary.json

Usage: python tracking_vs_no_effects.py
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ORIG, OUT, build_presence, save_json, tickers_28  # noqa: E402
from exploratory_generator_diagnostics import REF, S, _cum, _load, _paths, _variant  # noqa: E402

SEEDS = (1, 2, 3)
TOL = 0.05
GEN = OUT / "exploratory_generator_diagnostics"


def run_ticker(t):
    panel = pd.read_csv(ORIG / "results" / REF / "training_data.csv")
    df = panel[panel.tic == t].sort_values("date").reset_index(drop=True)
    rp = np.clip(df[["open", "high", "low", "close", "volume"]].values.astype(float), 1e-8, None)
    rlr = np.diff(np.log(rp), axis=0)
    hist = np.log(rp[:, 3] / rp[0, 3])
    im, an = _load("r2_datasign", t)
    P = build_presence(an, df, list(im.factor_names))
    zero = _variant(im, "zero")
    return t, {s: float(_cum(_paths(zero, rp, rlr, P, S, s)[0]).mean() - hist[-1]) for s in SEEDS}


def main():
    tickers = tickers_28()
    with ProcessPoolExecutor(4) as ex:
        vanilla = dict(ex.map(run_ticker, tickers))
    rows = []
    for t in tickers:
        r = json.load(open(GEN / f"ticker_{t}.json"))
        e = {arm: abs(np.mean([r[arm]["track_by_seed"][str(s)] for s in SEEDS])) for arm in ("R1", "R2")}
        e["vanilla"] = abs(np.mean([vanilla[t][s] for s in SEEDS]))
        rows.append({"ticker": t, **e})
    d = pd.DataFrame(rows).set_index("ticker")
    res = {"seeds": list(SEEDS), "tolerance": TOL,
           "R2_vs_R1_pass": int(((d.R2 - d.R1) <= TOL).sum()),
           "R2_vs_vanilla_pass": int(((d.R2 - d.vanilla) <= TOL).sum()),
           "R1_vs_vanilla_pass": int(((d.R1 - d.vanilla) <= TOL).sum()),
           "R1_better_than_vanilla_by_tol": int(((d.vanilla - d.R1) > TOL).sum()),
           "R2_better_than_vanilla_by_tol": int(((d.vanilla - d.R2) > TOL).sum()),
           "median_abs_error": {c: float(d[c].median()) for c in d.columns},
           "by_stock": d.round(4).to_dict(orient="index")}
    save_json(res, OUT / "tracking_vs_no_effects" / "summary.json")
    print(json.dumps({k: v for k, v in res.items() if k != "by_stock"}, indent=1))


if __name__ == "__main__":
    main()
