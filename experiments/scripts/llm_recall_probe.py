#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Outcome-recall probe: does the LLM remember the sign of each window's return?

For every stock and two-week window of the R2 panel (28 x 155), the model is
asked, with no news and no prices, whether the stock rose or fell between the
window's first and last day. Agreement with the realized sign is scored as in
B1 (independence baseline, calendar-window cluster bootstrap). An excess shows
that the parametric channel of outcome leakage is open for this model and
period; it does not show that the R2 labels used it.

Calls go to OpenRouter (key from OPENROUTER_API_KEY or ~/.config/trace/openrouter.env),
temperature 0, one call per stock-window, cached in a JSONL file so that a run
can resume. Output: results/llm_recall_probe/<model>/{calls.jsonl,summary.json}

Usage: python llm_recall_probe.py --model openai/gpt-4o-mini [--limit N]
       python llm_recall_probe.py --model openai/gpt-4o-mini --score
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from b1_sensitivity import _boot, _excess, _scored, load  # noqa: E402
from common import ORIG, OUT, save_json  # noqa: E402

RUN = "r2_datasign"
URL = "https://openrouter.ai/api/v1/chat/completions"
NAMES = {"AAPL": "Apple", "AMGN": "Amgen", "AMZN": "Amazon", "AXP": "American Express", "BA": "Boeing",
         "CAT": "Caterpillar", "CRM": "Salesforce", "CSCO": "Cisco Systems", "CVX": "Chevron", "DIS": "Walt Disney",
         "GS": "Goldman Sachs", "HD": "Home Depot", "HON": "Honeywell", "IBM": "IBM", "JNJ": "Johnson & Johnson",
         "JPM": "JPMorgan Chase", "KO": "Coca-Cola", "MCD": "McDonald's", "MMM": "3M", "MRK": "Merck",
         "MSFT": "Microsoft", "NKE": "Nike", "PG": "Procter & Gamble", "TRV": "Travelers",
         "UNH": "UnitedHealth Group", "V": "Visa", "WMT": "Walmart", "XOM": "ExxonMobil"}
PROMPT = ("Between {start} and {end}, did the share price of {name} (ticker {ticker}) rise or fall? "
          "Answer with one word: UP or DOWN.")


def _key():
    k = os.environ.get("OPENROUTER_API_KEY", "")
    f = Path.home() / ".config" / "trace" / "openrouter.env"
    if not k and f.exists():
        for line in f.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                k = line.split("=", 1)[1].strip().strip('"')
    if not k:
        sys.exit("OPENROUTER_API_KEY not set")
    return k


def _windows():
    rows = []
    for tdir in sorted((ORIG / "results" / RUN).iterdir()):
        f = tdir / "analyses.json"
        if f.exists():
            for i, w in enumerate(sorted(json.load(open(f)), key=lambda x: x["start_date"])):
                rows.append({"ticker": tdir.name, "win": i, "start": w["start_date"], "end": w["end_date"]})
    return rows


def _call(model, key, w):
    msg = PROMPT.format(start=w["start"], end=w["end"], name=NAMES[w["ticker"]], ticker=w["ticker"])
    for attempt in range(5):
        try:
            r = requests.post(URL, headers={"Authorization": f"Bearer {key}"}, timeout=60,
                              json={"model": model, "temperature": 0, "max_tokens": 5,
                                    "messages": [{"role": "user", "content": msg}]})
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"].strip().upper()
            return {**w, "model": model, "answer": text}
        except Exception as e:  # noqa: BLE001
            err = str(e)
            time.sleep(2 ** attempt)
    return {**w, "model": model, "answer": None, "error": err}


def run(model, limit=None, workers=8):
    d = OUT / "llm_recall_probe" / model.replace("/", "_")
    d.mkdir(parents=True, exist_ok=True)
    f = d / "calls.jsonl"
    done = {(r["ticker"], r["win"]) for r in map(json.loads, f.read_text().splitlines())} if f.exists() else set()
    todo = [w for w in _windows() if (w["ticker"], w["win"]) not in done][:limit]
    key = _key()
    with ThreadPoolExecutor(workers) as ex, open(f, "a") as out:
        for r in ex.map(lambda w: _call(model, key, w), todo):
            out.write(json.dumps(r) + "\n")
    print(f"{len(todo)} calls written to {f}")


def score(model):
    d = OUT / "llm_recall_probe" / model.replace("/", "_")
    calls = pd.DataFrame(map(json.loads, (d / "calls.jsonl").read_text().splitlines()))
    calls = calls.dropna(subset=["answer"]).drop_duplicates(["ticker", "win"], keep="last")
    calls["s"] = np.where(calls.answer.str.startswith("UP"), 1.0, np.where(calls.answer.str.startswith("DOWN"), -1.0, 0.0))
    lab = load(RUN)[["ticker", "win", "r"]]
    df = lab.merge(calls[["ticker", "win", "s"]], on=["ticker", "win"], how="inner")
    s, r, w = _scored(df.s.values, df.r.values, df.win.values)
    bt = _boot(s, r, w, 2000, 0)
    res = {"model": model, "windows": int(len(df)), "answered_up_or_down": int((df.s != 0).sum()),
           "n_scored": int(len(s)), "agreement": float((s == r).mean()), "excess": _excess(s, r),
           "independence": float((s == r).mean()) - _excess(s, r),
           "ci95": [float(np.percentile(bt, 2.5)), float(np.percentile(bt, 97.5))],
           "p_one_sided": float((bt <= 0).mean())}
    # where the recall sits: market direction, stock-specific returns, next window, R2 labels
    df = df.sort_values(["ticker", "win"])
    g = df.groupby("win").r
    df["mkt"] = (g.transform("sum") - df.r) / (g.transform("count") - 1)
    df["r_next"] = df.groupby("ticker").r.shift(-1)
    for key, col in (("market_other27", "mkt"), ("next_window", "r_next")):
        s2, r2, w2 = _scored(df.s.values, df[col].values, df.win.values)
        bt2 = _boot(s2, r2, w2, 2000, 0)
        res[key] = {"excess": _excess(s2, r2), "ci95": [float(np.percentile(bt2, 2.5)), float(np.percentile(bt2, 97.5))]}
    from scipy.stats import spearmanr
    ics = [spearmanr(x.s, x.r - x.mkt).correlation for _, x in df[df.s != 0].groupby("win")
           if x.s.nunique() > 1 and len(x) >= 5]
    ics = np.array([v for v in ics if np.isfinite(v)])
    res["stock_specific_rank_ic"] = {"mean": float(ics.mean()), "t": float(ics.mean() / (ics.std(ddof=1) / np.sqrt(len(ics)))),
                                     "windows": int(len(ics))}
    from b1_sensitivity import per_stock
    res["per_stock_baseline"] = per_stock(df[["ticker", "win", "s", "r"]].copy())
    r2lab = load(RUN)[["ticker", "win", "s"]].rename(columns={"s": "s_r2"})
    m = df.merge(r2lab, on=["ticker", "win"])
    s3, r3, w3 = _scored(np.sign(m.s_r2.values), m.s.values, m.win.values)
    bt3 = _boot(s3, r3, w3, 2000, 0)
    res["R2_labels_vs_recall"] = {"excess": _excess(s3, r3), "ci95": [float(np.percentile(bt3, 2.5)), float(np.percentile(bt3, 97.5))],
                                  "n": int(len(s3))}
    save_json(res, d / "summary.json")
    print(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    score(a.model) if a.score else run(a.model, a.limit)


if __name__ == "__main__":
    main()
