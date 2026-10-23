#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit of the strict price-blind redaction on real corpus-v2 contexts.

``sample``: collect every redactable sentence of the raw (unredacted)
precomputed contexts (``cache/contexts_v2/_raw/*.json``), de-duplicate by
normalised text, split them by the redaction decision under ``--mode``
(default ``strict``) and draw a seeded random sample of ``--n`` dropped and
``--n`` kept sentences.  Writes ``logs/redaction_audit_v2_sample.json``
with an empty ``label`` per item.

``score``: read the sample with labels filled in (``1`` = the sentence
reports an outcome, ``0`` = it does not) and write precision, recall and
their Wilson intervals to ``logs/redaction_audit_v2.md``.

Recall is estimated from the two strata: TP = precision x N_dropped,
FN = (outcome share among kept) x N_kept, recall = TP / (TP + FN).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sde_causal_generator.data_structures import normalize_article_text  # noqa: E402
from sde_causal_generator.factor_extractor import (  # noqa: E402
    _MARKET_TERMS_V1,
    _MOVE_TERMS,
    iter_context_sentences,
    sentence_is_redacted,
)


def decide(sentence: str, mode: str) -> bool:
    """Redaction decision; ``strict_v1`` = strict rule with the v1 term list
    (audit round 1 only, not an extraction mode)."""
    if mode == "strict_v1":
        return bool(_MARKET_TERMS_V1.search(sentence))
    return sentence_is_redacted(sentence, mode)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 3
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


# "[12] (2020-03-12) [wikipedia_current_events] " numbering prefix of the
# article header lines: it differs between contexts for the same text.
_PREFIX = re.compile(r"^\[\d+\]\s*\(\d{4}-\d{2}-\d{2}\)\s*\[[^\]]*\]\s*")


def collect(raw_dir: Path, mode: str):
    """Unique redactable sentences (numbering prefix stripped)."""
    pop = {}
    occ = Counter()
    for f in sorted(raw_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for w in data["windows"]:
            if "[No news articles found" in w["context"]:
                continue
            for _, sent in iter_context_sentences(w["context"]):
                sent = _PREFIX.sub("", sent)
                if not sent.strip():
                    continue
                key = normalize_article_text(sent)
                occ[key] += 1
                if key not in pop:
                    pop[key] = {"sentence": sent, "ticker": data["ticker"],
                                "window": f"{w['window_start']}..{w['window_end']}",
                                "dropped": decide(sent, mode)}
    return pop, occ


# Kept sentences that MIGHT report a market move: a v2 move term plus a
# market-ish word.  Labelled as a census, so that recall is not estimated
# from a uniform sample alone (outcome sentences are rare among kept ones).
CANDIDATE = re.compile(
    r"\b(markets?|index|indices|indexes|trading|traders?|investors?|exchanges?|"
    r"valu\w*|worth|IPO|listed|benchmark|composite|futures|bourses?|securities|"
    r"nasdaq|ibex|kospi|sensex|nifty|bovespa|merval|tsx|asx|msci|stoxx|csi|topix|"
    r"tadawul|moex|rts|psei|jse|bist|borsa)\b", re.IGNORECASE)


# Equity-ish words that can report an outcome without any move term
# ("becomes the first public company to be worth US$1 trillion").
EQUITY_ISH = re.compile(
    r"\b(trillion|valuation|market value|market cap|most valuable|publicly[- ]traded|"
    r"IPO|listed|listing|investors?|traders?|trading|index|indices|indexes|composite|"
    r"benchmark|bourses?|securities|nasdaq|ibex|kospi|sensex|nifty|bovespa|merval|tsx|"
    r"asx|msci|topix|tadawul|moex|rts|jse|bist|borsa)\b", re.IGNORECASE)


def is_candidate(sentence: str) -> bool:
    return bool((_MOVE_TERMS.search(sentence) and CANDIDATE.search(sentence))
                or EQUITY_ISH.search(sentence))


def cmd_sample(args):
    root = Path(args.data_root)
    pop, occ = collect(root / args.raw, args.mode)
    dropped = sorted(k for k, v in pop.items() if v["dropped"])
    kept = sorted(k for k, v in pop.items() if not v["dropped"])
    rng = random.Random(args.seed)
    sd = rng.sample(dropped, min(args.n, len(dropped)))
    sk = rng.sample(kept, min(args.n, len(kept)))
    cand = sorted(k for k in kept if is_candidate(pop[k]["sentence"]))
    items = []
    for stratum, keys in (("dropped", sd), ("kept", sk), ("kept_candidate", cand)):
        for k in keys:
            items.append({"id": len(items), "stratum": stratum, **pop[k],
                          "candidate": is_candidate(pop[k]["sentence"]),
                          "occurrences": occ[k], "label": None})
    out = {"mode": args.mode, "seed": args.seed,
           "population": {"n_unique_sentences": len(pop),
                          "n_unique_dropped": len(dropped), "n_unique_kept": len(kept),
                          "n_unique_kept_candidates": len(cand),
                          "n_occurrences_dropped": sum(occ[k] for k in dropped),
                          "n_occurrences_kept": sum(occ[k] for k in kept)},
           "items": items}
    path = root / args.sample
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["population"], indent=2))
    print(f"wrote {path}")


def cmd_score(args):
    root = Path(args.data_root)
    data = json.loads((root / args.sample).read_text(encoding="utf-8"))
    if args.labels:
        labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
        for it in data["items"]:
            if str(it["id"]) in labels:
                it["label"] = int(labels[str(it["id"])])
    if args.labels_by_text:
        labels = json.loads(Path(args.labels_by_text).read_text(encoding="utf-8"))
        for it in data["items"]:
            key = normalize_article_text(it["sentence"])
            if key in labels:
                it["label"] = int(labels[key])
    if args.labels or args.labels_by_text:
        (root / args.sample).write_text(json.dumps(data, indent=1, ensure_ascii=False),
                                        encoding="utf-8")
    items = data["items"]
    missing = [it for it in items if it["label"] is None]
    if missing:
        if args.dump_unlabelled:
            Path(args.dump_unlabelled).write_text("\n".join(
                f"{it['id']} | {it['stratum']} | {it['sentence']}" for it in missing),
                encoding="utf-8")
        raise SystemExit(f"{len(missing)} items unlabelled, e.g. "
                         f"{[it['id'] for it in missing[:10]]}")
    d = [it for it in items if it["stratum"] == "dropped"]
    k = [it for it in items if it["stratum"] == "kept"]
    tp_s = sum(it["label"] for it in d)
    fn_s = sum(it["label"] for it in k)
    prec = wilson(tp_s, len(d))
    fnr = wilson(fn_s, len(k))
    pop = data["population"]
    nd, nk = pop["n_unique_dropped"], pop["n_unique_kept"]

    def recall(p, f):
        tp, fn = p * nd, f * nk
        return tp / (tp + fn) if tp + fn > 0 else float("nan")

    rec = recall(prec[0], fnr[0])
    rec_lo = recall(prec[1], fnr[2])
    rec_hi = recall(prec[2], fnr[1])

    # stratified: census of kept candidates + uniform kept sample restricted
    # to non-candidates for the remaining kept sentences
    c = [it for it in items if it["stratum"] == "kept_candidate"]
    k0 = [it for it in k if not it.get("candidate")]
    nc = pop.get("n_unique_kept_candidates", 0)
    fn_c = sum(it["label"] for it in c)
    r0 = wilson(sum(it["label"] for it in k0), len(k0))
    nk0 = nk - nc

    def srecall(p, f0):
        tp, fn = p * nd, fn_c + f0 * nk0
        return tp / (tp + fn) if tp + fn > 0 else float("nan")

    strat = (srecall(prec[0], r0[0]), srecall(prec[1], r0[2]), srecall(prec[2], r0[1]))
    fp_examples = [it for it in d if it["label"] == 0]
    fn_examples = [it for it in k + c if it["label"] == 1]

    lines = [
        "# Redaction audit v2 (strict price-blind mode)", "",
        f"Date: 2026-09-30. Mode: `{data['mode']}` "
        "(`sde_causal_generator/factor_extractor.py`, `sentence_is_redacted`).",
        "Contexts: corpus-v2 raw contexts, `cache/contexts_v2/_raw/*.json` "
        "(all tickers, all 2-week windows), sentences de-duplicated by normalised text.",
        f"Sample: seed {data['seed']}, {len(d)} dropped + {len(k)} kept sentences, "
        "drawn uniformly from the unique sentences of each stratum.", "",
        "## Labelling rule", "",
        args.rule.strip(), "",
        "## Population", "",
        f"- unique sentences: {pop['n_unique_sentences']}",
        f"- dropped (unique / occurrences): {nd} / {pop['n_occurrences_dropped']}",
        f"- kept (unique / occurrences): {nk} / {pop['n_occurrences_kept']}", "",
        "## Result", "",
        "| quantity | estimate | 95% CI |", "|---|---|---|",
        f"| precision (outcome-reporting among dropped) | {prec[0]:.3f} ({tp_s}/{len(d)}) "
        f"| [{prec[1]:.3f}, {prec[2]:.3f}] |",
        f"| outcome-reporting among kept | {fnr[0]:.3f} ({fn_s}/{len(k)}) "
        f"| [{fnr[1]:.3f}, {fnr[2]:.3f}] |",
        f"| recall, uniform design (spec) | {rec:.3f} | [{rec_lo:.3f}, {rec_hi:.3f}] |",
    ]
    if c:
        lines += [
            f"| kept candidates labelled outcome-reporting (census) | {fn_c}/{len(c)} | — |",
            f"| outcome-reporting among kept non-candidates | {r0[0]:.3f} "
            f"({sum(it['label'] for it in k0)}/{len(k0)}) | [{r0[1]:.3f}, {r0[2]:.3f}] |",
            f"| **recall, stratified** | **{strat[0]:.3f}** | [{strat[1]:.3f}, {strat[2]:.3f}] |",
        ]
    lines += [
        "",
        "Uniform design: recall = TP / (TP + FN) with TP = precision x N_dropped "
        "and FN = (outcome share among the uniform kept sample) x N_kept.",
        "Stratified design: kept sentences are split into candidates (a v2 move "
        "term plus a market-ish word, or an equity-ish word; `is_candidate` in "
        "the script), labelled as a "
        f"census ({nc} unique), and the rest ({nk0} unique), whose outcome share "
        "is estimated from the uniform kept sample restricted to non-candidates; "
        "FN = census misses + share x N_rest.",
        "Intervals combine the Wilson intervals at their extremes (conservative); "
        "all counts are over unique sentences.", "",
        "## Kept sentences labelled outcome-reporting (misses)", "",
    ]
    lines += [f"- [{it['id']}] {it['sentence']}" for it in fn_examples] or ["- none"]
    lines += ["", "## Dropped sentences labelled not outcome-reporting (false positives)", ""]
    lines += [f"- [{it['id']}] {it['sentence']}" for it in fp_examples] or ["- none"]
    if args.notes:
        lines += ["", "## Notes", "", Path(args.notes).read_text(encoding="utf-8").strip()]
    lines += ["", f"Labelled sample: `{args.sample}` (per-item labels).", ""]
    out = root / args.report
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:40]))
    print(f"wrote {out}")


RULE = """\
A sentence is labelled **outcome-reporting (1)** when it reports a realised
movement, level or milestone of equity prices: a stock index (Dow, S&P 500,
Nasdaq, FTSE, Nikkei, ...), a company's share price or market value, the
"stock market" as a whole (crash, rally, sell-off, record, bear market), or
trading halts / circuit breakers triggered by price moves.  These are the
outcomes that the price-blind prompt must not see.  Everything else is
**0**, including policy or corporate events that merely mention shares or
stocks (share buybacks, stock offerings, IPO announcements, shareholder
votes, "shares a border"), and moves of non-equity prices (oil, currencies,
bond yields, crypto), which are candidate causes rather than the equity
outcome."""


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["sample", "score"])
    ap.add_argument("--data-root", default=str(REPO))
    ap.add_argument("--raw", default="cache/contexts_v2/_raw")
    ap.add_argument("--sample", default="logs/redaction_audit_v2_sample.json")
    ap.add_argument("--report", default="logs/redaction_audit_v2.md")
    ap.add_argument("--labels", default=None, help="score: JSON {id: 0|1}")
    ap.add_argument("--labels-by-text", default=None,
                    help="score: JSON {normalised sentence: 0|1}")
    ap.add_argument("--dump-unlabelled", default=None,
                    help="score: write unlabelled items to this file")
    ap.add_argument("--notes", default=None, help="score: markdown appended to the report")
    ap.add_argument("--mode", default="strict")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--rule", default=RULE)
    args = ap.parse_args()
    cmd_sample(args) if args.cmd == "sample" else cmd_score(args)


if __name__ == "__main__":
    main()
