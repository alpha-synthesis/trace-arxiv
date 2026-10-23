#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Precompute the redacted RAG news context of every (ticker, window).

Corpus v2: the LLM extraction process must never open the Chroma store
(threaded litellm calls in the same process corrupted ``chroma.sqlite3`` on
2026-09-30).  This separate process

* opens ONLY the v2 store, read-only (``NewsRAGManager(cache_mode=
  "readonly")``: Chroma queries a private copy; the store's sha256 is
  checked against its MANIFEST before and after);
* makes NO network calls (Hugging Face offline mode, and a socket guard
  that raises on any non-local connection);
* for each ticker of the config and each window of the pipeline (the
  extractor's own window logic, cross-checked against
  ``results/<run>/<TICKER>/analyses.json``), retrieves the context exactly
  as the extractor would (``get_context``: same query, ``max_articles``,
  ``content_limit``), applies the price-blind redaction (default
  ``strict``), and writes ``cache/contexts_v2/<TICKER>.json``.

Each window record: ``{window_start, window_end, context, n_retrieved,
n_unique, sources_count, n_sentences_dropped, n_sentences_kept}`` plus
``n_duplicates, n_excluded, n_ticker_specific, context_sha1``.  The raw
(unredacted) contexts go to ``cache/contexts_v2/_raw/<TICKER>.json`` for
audits.  ``INDEX.json`` holds the corpus tag and the context statistics.

``--context-version 2.1`` (after a review of the v2 contexts)
keeps the store unchanged and changes only how a context is assembled:

* whole items are dropped, BEFORE the cut to ``max_articles`` (the context is
  refilled from surviving items; no ``[redacted]`` marker, so the number of
  drops cannot signal a large market move):
  - Wikipedia topic-header items (link-only lines: no trailing source tag
    and under 140 characters);
  - any item with a sentence redacted under ``strict2`` (= strict + outcome
    phrases with no market term; a continuation sentence goes with its item);
* ``sec_edgar`` items are excluded (content-free "X filed with the SEC" stubs,
  half of them tagged to an unrelated issuer);
* the query names the company, and items that name it lexically are ranked
  first (``names_company``);
* up to 1000 candidates per window (600 capped 504 of 4,340 v2 windows);
* corpus tag ``<store tag>+ctx2.1``; a post-check re-applies ``strict2`` to the
  assembled context and must drop nothing.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import socket
import sys
from pathlib import Path

# ── no network: set before any HF / Chroma import ────────────────────
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

_real_connect = socket.socket.connect


def _guarded_connect(self, address):
    if self.family in (socket.AF_INET, socket.AF_INET6):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise ConnectionRefusedError(
                f"precompute_contexts.py makes no network calls (blocked {address!r})")
    return _real_connect(self, address)


socket.socket.connect = _guarded_connect

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import re  # noqa: E402

from sde_causal_generator.factor_extractor import (  # noqa: E402
    _SENTENCE_SPLIT,
    LLMFactorExtractor,
    redact_market_moves,
    sentence_is_redacted,
)
from sde_causal_generator.news_rag import (  # noqa: E402
    STORE_MANIFEST,
    NewsRAGManager,
    file_sha256,
    has_source_tag,
    item_body,
    item_text,
)


# ── context v2.1 ─────────────────────────────────────────────────────
#: Company names for the query and the lexical "names the company" rank.
#: Patterns avoid common-word collisions (Amazon rainforest, travel visas,
#: travelers, Dow index).
COMPANY = {
    "AAPL": ("Apple", r"(?<!Lost )\bApple(?:'s| Inc\.?)?\b(?!\s+(?:Daily|Fire|Project|trees?|orchards?|cultivars?|varieties))|\biPhones?\b|\bTim Cook\b"),
    "AMGN": ("Amgen", r"\bAmgen\b"),
    "AMZN": ("Amazon", r"\bAmazon(?:'s|\.com)?\b(?![\s-]+(?i:rainforest|basin|river|region|forest|fund|fires?|deforestation|jungle|tribes?|indigenous|states?))|\bAWS\b|\bWhole Foods\b"),
    "AXP": ("American Express", r"\bAmerican Express\b|\bAmex\b"),
    "BA": ("Boeing", r"\bBoeing\b"),
    "CAT": ("Caterpillar", r"\bCaterpillar Inc\.?\b|\bCaterpillar's\b|\bCaterpillar\b(?=\s+(?:says|said|announces|announced|reports|reported|cuts|plans|will|to))"),
    "CRM": ("Salesforce", r"\bSalesforce\b(?! Tower)"),
    "CSCO": ("Cisco", r"\bCisco\b"),
    "CVX": ("Chevron", r"\bChevron\b(?!\s+(?:deference|doctrine))"),
    "DIS": ("Disney", r"\bDisney(?:land|\+)?\b|\bPixar\b|\bMarvel Studios\b"),
    "GS": ("Goldman Sachs", r"\bGoldman Sachs\b|\bGoldman\b"),
    "HD": ("Home Depot", r"\bHome Depot\b"),
    "HON": ("Honeywell", r"\bHoneywell\b"),
    "IBM": ("IBM", r"\bIBM\b"),
    "JNJ": ("Johnson & Johnson", r"\bJohnson (?:&|&amp;|and) Johnson\b|\bJ&J\b|\bJanssen\b"),
    "JPM": ("JPMorgan Chase", r"\bJP ?Morgan\b|\bJ\.P\. Morgan\b|\bJamie Dimon\b"),
    "KO": ("Coca-Cola", r"\bCoca-Cola\b"),
    "MCD": ("McDonald's", r"\bMcDonald's\b|\bMcDonald’s\b"),
    "MMM": ("3M", r"\b3M\b"),
    "MRK": ("Merck", r"\bMerck\b"),
    "MSFT": ("Microsoft", r"\bMicrosoft\b(?! Theat(?:er|re))|\bSatya Nadella\b"),
    "NKE": ("Nike", r"\bNike\b"),
    "PG": ("Procter & Gamble", r"\bProcter (?:&|&amp;|and) Gamble\b|\bP&G\b"),
    "TRV": ("Travelers Companies", r"\bTravelers (?:Companies|Insurance)\b"),
    "UNH": ("UnitedHealth", r"\bUnitedHealth(?:care)?\b|\bOptum\b"),
    "V": ("Visa", r"\bVisa Inc\.?\b|\bVisa (?:and|&) Mastercard\b|\bMastercard (?:and|&) Visa\b|\bVisa's\b"),
    "WMT": ("Walmart", r"\bWal-?[Mm]art\b"),
    "XOM": ("ExxonMobil", r"\bExxon(?: ?Mobil)?\b"),
}
_SENT_END = re.compile(r"[a-z0-9)\"'’]\.(?:\s|$)")


def is_topic_header(a):
    """Link-only Wikipedia topic line: no source tag and either short or with
    no sentence ending (long comma-separated lists of article titles)."""
    if not a["source"].startswith("wikipedia") or has_source_tag(a):
        return False
    body = item_body(a)
    return len(body) < 140 or not _SENT_END.search(body)


def make_item_filter(mode):
    """Item filter for retrieval; ``.dropped`` keeps the redacted items for audit."""
    def item_filter(a):
        if is_topic_header(a):
            return "topic_header"
        for sent in _SENTENCE_SPLIT.split(item_text(a)):
            if sentence_is_redacted(sent, mode):
                item_filter.dropped.append({"date": a["date"], "text": item_body(a)[:600],
                                            "sentence": sent[:300]})
                return "redacted"
        return None
    item_filter.dropped = []
    return item_filter


def make_prefer(ticker):
    pat = re.compile(COMPANY[ticker][1])
    return lambda a: bool(pat.search(item_text(a)))


def company_query(ticker):
    return (f"Market moving events affecting {COMPANY[ticker][0]} ({ticker}) "
            f"stock price macroeconomic policy interest rates earnings")


def ticker_windows(ticker, prices: pd.DataFrame, window_sizes, analyses_path: Path):
    """Pipeline windows for a ticker; cross-checked with analyses.json."""
    ref = None
    if analyses_path.is_file():
        with open(analyses_path, "r", encoding="utf-8") as fh:
            ref = [(a["start_date"], a["end_date"]) for a in json.load(fh)]
    if prices is not None and ticker in set(prices["tic"]):
        dates = prices.loc[prices["tic"] == ticker, "date"]
        wins = LLMFactorExtractor.window_pairs(dates, window_sizes)
        if ref is not None and wins != ref:
            raise SystemExit(f"{ticker}: extractor windows differ from {analyses_path}")
        return wins, "extractor.window_pairs (matches analyses.json)" if ref else \
            "extractor.window_pairs"
    if ref is None:
        raise SystemExit(f"{ticker}: no price dates and no analyses.json")
    return ref, "analyses.json"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO))
    ap.add_argument("--config", default="configs/a1.yaml")
    ap.add_argument("--run", default="a1",
                    help="results/<run>/ holds training_data.csv and analyses.json")
    ap.add_argument("--store", default="cache/rag_store_v2")
    ap.add_argument("--out", default="cache/contexts_v2")
    ap.add_argument("--redaction", default="strict",
                    choices=["off", "legacy", "moves", "strict", "strict2"])
    ap.add_argument("--readonly-workdir", default=None)
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--min-median-unique", type=float, default=30.0)
    ap.add_argument("--context-version", default="2", choices=["2", "2.1"])
    args = ap.parse_args()
    v21 = args.context_version == "2.1"
    if v21:
        if args.redaction not in ("strict2",):
            args.redaction = "strict2"
        if args.out == "cache/contexts_v2":
            args.out = "cache/contexts_v2.1"

    root = Path(args.data_root)
    with open(root / args.config, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    tickers = args.tickers or cfg["tickers"]
    window_sizes = cfg.get("window_sizes", ["1ME", "1QE"])
    max_articles = int(cfg.get("rag_max_articles", 50))
    # The extractor calls get_context(...) without content_limit, i.e. 500.
    content_limit = 500

    store_dir = root / args.store
    with open(store_dir / STORE_MANIFEST, "r", encoding="utf-8") as fh:
        manifest = json.load(fh)
    store_sha = manifest["sqlite_sha256"]
    corpus_tag = manifest["corpus_tag"] + ("+ctx2.1" if v21 else "")
    sqlite_path = store_dir / "chroma.sqlite3"
    st0 = sqlite_path.stat()

    prices_path = root / "results" / args.run / "training_data.csv"
    prices = pd.read_csv(prices_path, usecols=["date", "tic"]) if prices_path.is_file() else None

    out_dir = root / args.out
    raw_dir = out_dir / "_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rag = NewsRAGManager(persist_dir=str(store_dir), cache_mode="readonly",
                         readonly_workdir=args.readonly_workdir,
                         expected_sha256=store_sha,
                         **({"exclude_sources": ["sec_edgar"], "overfetch": 1000} if v21 else {}))
    excluded = sorted(rag.store.exclude_sources)
    index = {"corpus_tag": corpus_tag, "store": str(store_dir),
             "store_sha256": store_sha, "redaction_mode": args.redaction,
             "max_articles": max_articles, "content_limit": content_limit,
             "window_sizes": window_sizes, "excluded_sources": excluded,
             "context_version": args.context_version,
             "overfetch": rag.overfetch,
             "created": _dt.datetime.now().isoformat(timespec="seconds"),
             "tickers": {}}
    all_unique, n_fred_in_context, n_excluded_hits = [], 0, 0
    n_sec_in_context = n_post_check_drops = n_capped = 0
    n_names_company_ctx = 0
    drop_reasons = {}
    for t in tickers:
        wins, win_src = ticker_windows(
            t, prices, window_sizes, root / "results" / args.run / t / "analyses.json")
        query = company_query(t) if v21 else NewsRAGManager.default_query(t)
        recs, raws = [], []
        for s, e in wins:
            if v21:
                filt = make_item_filter(args.redaction)
                ctx, st = rag.get_context_with_stats(
                    t, s, e, max_articles=max_articles, content_limit=content_limit,
                    query_template=query, item_filter=filt, prefer=make_prefer(t),
                    compact=True)
                ctx_raw = ctx                       # items were dropped whole
                _, n_post, n_keep = redact_market_moves(ctx, args.redaction)
                n_post_check_drops += n_post
                n_drop = int(st["n_dropped_by_reason"].get("redacted", 0))
                for k, v in st["n_dropped_by_reason"].items():
                    drop_reasons[k] = drop_reasons.get(k, 0) + v
                n_capped += st["n_candidates_capped"]
                n_names_company_ctx += int(st["n_names_company"] > 0)
            else:
                ctx_raw, st = rag.get_context_with_stats(
                    t, s, e, max_articles=max_articles, content_limit=content_limit)
                has_news = "[No news articles found" not in ctx_raw
                if has_news and args.redaction != "off":
                    ctx, n_drop, n_keep = redact_market_moves(ctx_raw, args.redaction)
                else:
                    ctx, n_drop, n_keep = ctx_raw, 0, 0
            n_fred_in_context += sum(v for k, v in st["sources_count"].items()
                                     if k.lower() == "fred")
            n_sec_in_context += sum(v for k, v in st["sources_count"].items()
                                    if k.lower() == "sec_edgar")
            n_excluded_hits += st["n_excluded"]
            recs.append({
                "window_start": s, "window_end": e, "context": ctx,
                "n_retrieved": st["n_retrieved"], "n_unique": st["n_unique"],
                "sources_count": st["sources_count"],
                "n_sentences_dropped": 0 if v21 else n_drop,
                "n_items_redacted": n_drop if v21 else 0,
                "n_sentences_kept": n_keep,
                "n_dropped_by_reason": st.get("n_dropped_by_reason"),
                "n_names_company": st.get("n_names_company"),
                "n_duplicates": st["n_duplicates"], "n_excluded": st["n_excluded"],
                "n_ticker_specific": st["n_ticker_specific"],
                "context_sha1": hashlib.sha1(ctx.encode("utf-8")).hexdigest(),
            })
            raws.append({"window_start": s, "window_end": e, "context": ctx_raw,
                         **({"dropped_items": filt.dropped} if v21 else {})})
        uniq = [r["n_unique"] for r in recs]
        all_unique.extend(uniq)
        meta = {"ticker": t, "corpus_tag": corpus_tag, "store_sha256": store_sha,
                "redaction_mode": args.redaction, "max_articles": max_articles,
                "content_limit": content_limit, "query": query,
                "window_sizes": window_sizes, "windows_from": win_src}
        with open(out_dir / f"{t}.json", "w", encoding="utf-8") as fh:
            json.dump({**meta, "windows": recs}, fh, ensure_ascii=False)
        with open(raw_dir / f"{t}.json", "w", encoding="utf-8") as fh:
            json.dump({**meta, "redaction_mode": "off", "windows": raws}, fh,
                      ensure_ascii=False)
        index["tickers"][t] = {
            "n_windows": len(recs),
            "median_unique": float(np.median(uniq)) if uniq else 0.0,
            "min_unique": int(min(uniq)) if uniq else 0,
            "n_no_news": sum(1 for r in recs if r["n_unique"] == 0),
            "sentences_dropped": int(sum(r["n_sentences_dropped"] for r in recs)),
            "sentences_kept": int(sum(r["n_sentences_kept"] for r in recs)),
        }
        print(f"  {t:5s} {len(recs)} windows  median unique="
              f"{index['tickers'][t]['median_unique']:.0f}  "
              f"min={index['tickers'][t]['min_unique']}  "
              f"dropped/kept={index['tickers'][t]['sentences_dropped']}/"
              f"{index['tickers'][t]['sentences_kept']}", flush=True)
    rag.close()

    st1 = sqlite_path.stat()
    sha_after = file_sha256(sqlite_path)
    med = float(np.median(all_unique)) if all_unique else 0.0
    index.update({
        "n_contexts": len(all_unique),
        "median_unique_per_context": med,
        "mean_unique_per_context": float(np.mean(all_unique)) if all_unique else 0.0,
        "min_unique_per_context": int(min(all_unique)) if all_unique else 0,
        "n_fred_documents_in_contexts": n_fred_in_context,
        "n_excluded_hits_at_retrieval": n_excluded_hits,
        "store_unchanged": (st0.st_size, st0.st_mtime_ns) == (st1.st_size, st1.st_mtime_ns)
                           and sha_after == store_sha,
        "gate_G1_median_unique_ge_30": med >= args.min_median_unique,
        "gate_G1_zero_fred_retrieved": n_fred_in_context == 0 and (v21 or n_excluded_hits == 0),
    })
    if v21:
        index.update({
            "n_sec_documents_in_contexts": n_sec_in_context,
            "n_items_dropped_by_reason": drop_reasons,
            "n_contexts_naming_company": n_names_company_ctx,
            "n_windows_candidates_capped": n_capped,
            "n_post_check_drops": n_post_check_drops,
            "gate_G1_zero_sec_in_contexts": n_sec_in_context == 0,
            "gate_G1_post_check_clean": n_post_check_drops == 0,
        })
    with open(out_dir / "INDEX.json", "w", encoding="utf-8") as fh:
        json.dump(index, fh, indent=1)
    print(json.dumps({k: v for k, v in index.items() if k != "tickers"}, indent=2))
    if not index["store_unchanged"]:
        raise SystemExit("STORE CHANGED during precompute")


if __name__ == "__main__":
    main()
