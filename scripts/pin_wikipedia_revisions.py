#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Re-fetch Wikipedia Current Events day pages pinned to a past revision.

Corpus v2: the v1 corpus fetched every day page at its CURRENT
revision (``prop=revisions&rvprop=content`` with no ``rvstart``), so text
added to a page weeks or years later (hindsight) was visible to the LLM.

Pin rule
--------
Every day page from ``--start`` to ``--end`` is fetched at the latest
revision at or before ``pin(d) T23:59:59Z``
(``rvstart=<pin>&rvdir=older&rvlimit=1``), where

* if all tickers share the same 2-week analysis windows (read from
  ``results/<run>/<TICKER>/analyses.json``): ``pin(d)`` is the END of the
  earliest-ending window that contains ``d``.  A boundary day belongs to
  two windows (end of one, start of the next); the earlier end is used so
  that no window ever sees a revision saved after its own end.
* otherwise (windows differ between tickers), or for a day outside every
  window: ``pin(d) = d + 13 days`` (maximum in-window hindsight).

The parsed events use the existing parser
(``WikipediaCurrentEventsIngestor._parse_day_wikitext``).

Outputs (``<data-root>/cache/rag_store_v2/article_cache_pinned/``; nothing
existing is overwritten except this script's own files):

* ``wikipedia_pinned_wikitext.jsonl`` – one record per day: pin, revision
  ids/timestamp/sha1, raw wikitext (append-only; the script resumes).
* ``wikipedia_pinned_<start>_<end>.json`` – parsed NewsArticle dicts.
* ``wikipedia_pinned_manifest.json`` – per-day metadata (no wikitext),
  current revision ids, and the change statistics.

Network: read-only GETs to the MediaWiki API, generic User-Agent, about
2-3 requests/s with exponential back-off.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sde_causal_generator.data_structures import normalize_article_text  # noqa: E402
from sde_causal_generator.news_rag import WikipediaCurrentEventsIngestor as W  # noqa: E402

import yaml  # noqa: E402


def load_windows(data_root: Path, run: str, tickers):
    """{ticker: [(start, end), ...]} from analyses.json."""
    out = {}
    for t in tickers:
        p = data_root / "results" / run / t / "analyses.json"
        if p.is_file():
            with open(p, "r", encoding="utf-8") as fh:
                out[t] = [(a["start_date"], a["end_date"]) for a in json.load(fh)]
    return out


def pin_dates(days, windows_by_ticker):
    """Return ({iso_day: iso_pin_date}, rule, n_outside)."""
    distinct = {tuple(w) for w in windows_by_ticker.values()}
    shared = len(distinct) == 1 and len(windows_by_ticker) > 0
    pins, n_outside = {}, 0
    windows = list(next(iter(distinct))) if shared else []
    for d in days:
        iso = d.strftime("%Y-%m-%d")
        ends = [e for s, e in windows if s <= iso <= e]
        if ends:
            pins[iso] = min(ends)
        else:
            if shared:
                n_outside += 1
            pins[iso] = (d + pd.Timedelta(days=13)).strftime("%Y-%m-%d")
    rule = ("earliest end of the shared 2-week window containing the day; "
            "day+13 for days outside every window") if shared else \
        "day+13 (windows differ across tickers)"
    return pins, rule, n_outside, shared


def read_done(jsonl: Path):
    done = {}
    if jsonl.is_file():
        with open(jsonl, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("status") in ("ok", "missing", "no_revision", "hidden"):
                    done[rec["date"]] = rec
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO),
                    help="checkout holding cache/ and results/ (default: this repo)")
    ap.add_argument("--config", default="configs/a1.yaml")
    ap.add_argument("--run", default="a1",
                    help="results/<run>/<TICKER>/analyses.json holds the windows")
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2023-12-31")
    ap.add_argument("--delay", type=float, default=0.3,
                    help="seconds slept before each request (plus latency)")
    ap.add_argument("--reference-cache",
                    default="cache/rag_store/article_cache/"
                            "wikipedia_AAPL_2018-01-02_2023-12-29.json",
                    help="v1 cache (current revisions) to compare against; read only")
    ap.add_argument("--limit", type=int, default=0, help="debug: only N days")
    args = ap.parse_args()

    root = Path(args.data_root)
    out_dir = root / "cache" / "rag_store_v2" / "article_cache_pinned"
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "wikipedia_pinned_wikitext.jsonl"

    with open(root / args.config, "r", encoding="utf-8") as fh:
        tickers = yaml.safe_load(fh)["tickers"]
    windows = load_windows(root, args.run, tickers)
    days = list(pd.date_range(args.start, args.end, freq="D"))
    if args.limit:
        days = days[: args.limit]
    pins, rule, n_outside, shared = pin_dates(days, windows)
    print(f"{len(days)} day pages; windows from {len(windows)} tickers; "
          f"shared={shared}; rule: {rule}; {n_outside} days outside every window")

    done = read_done(jsonl)
    todo = [d for d in days if d.strftime("%Y-%m-%d") not in done]
    print(f"{len(done)} already fetched, {len(todo)} to fetch")
    with open(jsonl, "a", encoding="utf-8") as fh:
        for i, d in enumerate(todo, 1):
            iso = d.strftime("%Y-%m-%d")
            pin_time = f"{pins[iso]}T23:59:59Z"
            try:
                rec = W.fetch_day_revision(d, pin_time, request_delay=args.delay)
            except Exception as e:  # keep going; the day is retried on resume
                rec = {"date": iso, "title": W.day_page_title(d),
                       "pin_time": pin_time, "status": "error", "error": repr(e)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fh.flush()
            if rec["status"] in ("ok", "missing", "no_revision", "hidden"):
                done[iso] = rec
            if i % 100 == 0 or i == len(todo):
                print(f"  {i}/{len(todo)} fetched "
                      f"({Counter(r['status'] for r in done.values())})", flush=True)

    missing_days = [d.strftime("%Y-%m-%d") for d in days
                    if d.strftime("%Y-%m-%d") not in done]
    if missing_days:
        print(f"ERROR: {len(missing_days)} days not fetched (re-run to resume): "
              f"{missing_days[:5]} …")
        sys.exit(1)

    print("Fetching current revision ids …")
    current = W.fetch_current_revision_ids(days, request_delay=args.delay)

    # ── parse pinned content with the existing parser ─────────────────
    articles, per_day = [], []
    for d in days:
        iso = d.strftime("%Y-%m-%d")
        rec = done[iso]
        arts = (W._parse_day_wikitext(rec.get("wikitext", ""), iso)
                if rec["status"] == "ok" else [])
        articles.extend(arts)
        cur = current.get(iso, {})
        per_day.append({
            "date": iso, "title": rec["title"], "pin_time": rec["pin_time"],
            "status": rec["status"], "revid": rec.get("revid"),
            "timestamp": rec.get("timestamp"), "sha1": rec.get("sha1"),
            "current_revid": cur.get("revid"),
            "current_timestamp": cur.get("timestamp"),
            "changed_since_pin": (rec.get("revid") is not None
                                  and cur.get("revid") is not None
                                  and rec.get("revid") != cur.get("revid")),
            "redirect": rec.get("wikitext", "").lstrip().upper().startswith("#REDIRECT"),
            "n_articles": len(arts),
        })

    # ── compare with the v1 cache (current revisions at fetch time) ───
    ref_path = root / args.reference_cache
    cmp_stats = {}
    if ref_path.is_file():
        with open(ref_path, "r", encoding="utf-8") as fh:
            ref = json.load(fh)
        ref_by_day, pin_by_day = {}, {}
        for a in ref:
            ref_by_day.setdefault(a["published_date"], Counter())[
                normalize_article_text(a["content"])] += 1
        for a in articles:
            pin_by_day.setdefault(a.published_date, Counter())[
                normalize_article_text(a.content)] += 1
        common_days = sorted(set(ref_by_day) | set(pin_by_day))
        common_days = [x for x in common_days
                       if days[0].strftime("%Y-%m-%d") <= x <= days[-1].strftime("%Y-%m-%d")]
        n_diff = only_ref = only_pin = 0
        for x in common_days:
            r = set(ref_by_day.get(x, Counter()))
            p = set(pin_by_day.get(x, Counter()))
            if r != p:
                n_diff += 1
            only_ref += len(r - p)
            only_pin += len(p - r)
        ref_range = (min(ref_by_day), max(ref_by_day)) if ref_by_day else None
        cmp_stats = {
            "reference": str(ref_path),
            "reference_date_range": ref_range,
            "days_compared": len(common_days),
            "days_with_different_event_sets": n_diff,
            "events_only_in_reference_current": only_ref,
            "events_only_in_pinned": only_pin,
            "n_reference_articles": len(ref),
        }

    status = Counter(p["status"] for p in per_day)
    summary = {
        "start": args.start, "end": args.end, "n_days": len(days),
        "pin_rule": rule, "windows_shared_across_tickers": shared,
        "n_tickers_with_windows": len(windows),
        "n_days_outside_windows": n_outside,
        "status_counts": dict(status),
        "n_days_with_pinned_revision": sum(1 for p in per_day if p["revid"]),
        "n_days_changed_since_pin": sum(1 for p in per_day if p["changed_since_pin"]),
        "n_days_unchanged_since_pin": sum(
            1 for p in per_day if p["revid"] and p["current_revid"]
            and not p["changed_since_pin"]),
        "n_redirect_pages": sum(1 for p in per_day if p["redirect"]),
        "n_articles_pinned": len(articles),
        "comparison_with_v1_cache": cmp_stats,
        "user_agent": W._USER_AGENT,
        "fetched_with": "action=query&prop=revisions&rvprop=ids|timestamp|sha1|content"
                        "&rvslots=main&rvstart=<pin>T23:59:59Z&rvdir=older&rvlimit=1",
    }

    art_path = out_dir / f"wikipedia_pinned_{args.start}_{args.end}.json"
    with open(art_path, "w", encoding="utf-8") as fh:
        json.dump([a.to_dict() for a in articles], fh, ensure_ascii=False)
    with open(out_dir / "wikipedia_pinned_manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"summary": summary, "days": per_day}, fh, indent=1)
    print(json.dumps(summary, indent=2))
    print(f"wrote {art_path}")


if __name__ == "__main__":
    main()
