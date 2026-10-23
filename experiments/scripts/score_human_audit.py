#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Score the human audit of the R2 news contexts (workbook from build_human_audit.py).

Reads audit/human_context_audit.xlsx as filled by the auditor (blind to the
windows' returns) and reports, with one-sided 95% Clopper-Pearson upper bounds:
  - items and contexts with a stock-market move (Q1) or an event after the
    window (Q3), over all items, the flagged items (every item of the sampled
    contexts that mentions a market, index, price or percentage term) and the
    random unflagged items;
  - non-equity price moves (Q2), which the design keeps: as marked, and with
    the coding rule applied to every currency, commodity and crypto move
    (items listed by row with the reason), because repeated items were marked
    inconsistently;
  - consistency of repeated items and the auditor's notes.

Needs openpyxl.
Usage: python score_human_audit.py
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

from openpyxl import load_workbook
from scipy.stats import beta

ROOT = Path(__file__).resolve().parents[1]
WB = ROOT / "audit" / "human_context_audit.xlsx"
DST = ROOT / "results" / "human_audit" / "summary.json"

# rows (1-based worksheet rows) whose text reports a market price move of a currency,
# commodity or index of commodity prices but were marked Q2 = N; listed after reading
# each text against the coding rule given to the auditor (non-equity market prices: Y)
Q2_RULE = {
    "Argentina hikes interest rate to 60%": "currency plunges",
    "The Turkish lira rises after": "currency rises",
    "The Turkish lira falls to a record low": "currency falls",
    "files for bankruptcy due to the oil price collapse": "oil price collapse",
    "the value of the Argentine peso at a record low": "currency at a record low",
    "The Turkish lira falls to an all-time low of 13.44": "currency falls",
    "The Turkish lira plunges by 15%": "currency falls",
    "international food prices reached a record high": "international commodity prices (FAO)",
}


def upper(k, n, conf=0.95):
    return float(beta.ppf(conf, k + 1, n - k)) if k < n else 1.0


def main():
    ws = load_workbook(WB, data_only=True)["Items"]
    rows = []
    for i, r in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
        rows.append({"row": i, "context": int(r[0]), "stock": r[1], "selection": r[4], "text": (r[7] or "").strip(),
                     "q1": str(r[8]).strip().upper(), "q2": str(r[9]).strip().upper(),
                     "q3": str(r[10]).strip().upper(), "note": r[11]})
    assert all(r[q] in ("Y", "N") for r in rows for q in ("q1", "q2", "q3")), "unfilled or invalid cells"
    ctx = sorted({r["context"] for r in rows})
    leak_ctx = {r["context"] for r in rows if r["q1"] == "Y" or r["q3"] == "Y"}
    out = {"workbook": str(WB.relative_to(ROOT)), "contexts": len(ctx), "items": len(rows)}
    for sel in ("flagged", "random", "all"):
        sub = [r for r in rows if sel == "all" or r["selection"] == sel]
        k1 = sum(r["q1"] == "Y" for r in sub); k3 = sum(r["q3"] == "Y" for r in sub)
        out[sel] = {"items": len(sub), "q1_yes": k1, "q3_yes": k3,
                    "q1_or_q3_upper95_items": upper(sum(r["q1"] == "Y" or r["q3"] == "Y" for r in sub), len(sub))}
    out["contexts_with_q1_or_q3"] = len(leak_ctx)
    out["contexts_upper95"] = upper(len(leak_ctx), len(ctx))
    q2 = [r for r in rows if r["q2"] == "Y"]
    rule = []
    for r in rows:
        if r["q2"] == "N":
            for key, why in Q2_RULE.items():
                if key in r["text"]:
                    rule.append({"row": r["row"], "context": r["context"], "reason": why, "text": r["text"][:160]})
    out["q2_as_marked"] = {"items": len(q2), "contexts": len({r["context"] for r in q2})}
    out["q2_rule_applied"] = {"items": len(q2) + len(rule), "contexts": len({r["context"] for r in q2} | {x["context"] for x in rule}),
                              "recoded_rows": rule}
    by = collections.defaultdict(list)
    for r in rows:
        by[r["text"]].append((r["q1"], r["q2"], r["q3"]))
    out["repeated_texts"] = sum(len(v) > 1 for v in by.values())
    out["repeated_texts_inconsistent"] = sum(len(set(v)) > 1 for v in by.values())
    out["notes"] = [{"row": r["row"], "note": r["note"], "marks": [r["q1"], r["q2"], r["q3"]], "text": r["text"][:120]}
                    for r in rows if r["note"]]
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(out, indent=1, ensure_ascii=False))
    print(json.dumps({k: v for k, v in out.items() if k not in ("notes",)}, indent=1, ensure_ascii=False)[:2500])


if __name__ == "__main__":
    main()
