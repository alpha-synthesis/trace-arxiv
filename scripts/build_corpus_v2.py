#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build the corpus-v2 RAG store once: ``cache/rag_store_v2``.

Inputs (read only):

* revision-pinned Wikipedia Current Events
  (``cache/rag_store_v2/article_cache_pinned/wikipedia_pinned_*.json``,
  from ``scripts/pin_wikipedia_revisions.py``);
* the SEC EDGAR caches of the run's tickers
  (``cache/rag_store/article_cache/sec_edgar_<T>_<start>_<end>.json``).

FRED is excluded at the store level: no ``fred_*`` cache file is ever read,
the store refuses ``source=fred`` documents, and the manifest lists
``fred`` in ``excluded_sources`` (applied again at retrieval time).

Ids are content-deterministic (``md5(source|date|title|normalized_content)``,
recomputed on load; cached ids are ignored).  Articles are de-duplicated on
the id and on ``(date, normalized content)``; ticker lists of duplicates
are merged.

Phases (each in its own process, so Chroma has flushed and exited before
anything inspects the files):

1. ``build``  – one long-lived Chroma client, one ingestion pass.  Refuses
   to run if ``chroma.sqlite3`` already exists.
2. ``verify`` – without Chroma: ``PRAGMA integrity_check`` (sqlite opened
   read-only), document counts per source and per ticker, zero FRED,
   sha256 of ``chroma.sqlite3``, ``chmod a-w`` on it, ``MANIFEST.json``.
   Then a read-only smoke query through ``NewsRAGManager(cache_mode=
   "readonly")`` and a check that the store bytes did not change.

``--phase all`` (default) runs ``build`` in a child process, then ``verify``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import sqlite3
import stat
import subprocess
import sys
from collections import Counter, OrderedDict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from sde_causal_generator.data_structures import (  # noqa: E402
    NewsArticle,
    normalize_article_text,
)

EXCLUDED_SOURCES = ("fred",)
STORE_DIRNAME = "rag_store_v2"


# ══════════════════════════════════════════════════════════════════════
# Loading
# ══════════════════════════════════════════════════════════════════════


def _load_json_articles(path: Path):
    with open(path, "r", encoding="utf-8") as fh:
        return [NewsArticle.from_dict(d) for d in json.load(fh)]


def input_files(data_root: Path, tickers, sec_range: str, pinned_glob: str):
    """The exact input files: pinned Wikipedia + per-ticker SEC EDGAR.

    Never returns a ``fred_*`` file.
    """
    pinned_dir = data_root / "cache" / STORE_DIRNAME / "article_cache_pinned"
    wiki = sorted(pinned_dir.glob(pinned_glob))
    sec_dir = data_root / "cache" / "rag_store" / "article_cache"
    sec = []
    for t in tickers:
        p = sec_dir / f"sec_edgar_{t}_{sec_range}.json"
        if p.is_file():
            sec.append(p)
    files = wiki + sec
    assert not any(f.name.lower().startswith("fred") for f in files)
    return wiki, sec


def load_corpus(files, excluded=EXCLUDED_SOURCES):
    """Load, recompute ids, drop excluded sources, de-duplicate.

    De-duplication: by content id, then by ``(date, normalized content)``.
    Duplicates' ``tickers_mentioned`` / ``categories`` are merged into the
    first copy.  Returns ``(articles, stats)``.
    """
    excluded = {s.lower() for s in excluded}
    stats = {"n_input": 0, "n_input_by_source": Counter(),
             "n_excluded": 0, "n_dup_id": 0, "n_dup_date_text": 0,
             "files": {}}
    by_key: "OrderedDict[tuple, NewsArticle]" = OrderedDict()
    by_id = {}
    for f in files:
        arts = _load_json_articles(Path(f))
        stats["files"][str(f)] = len(arts)
        for a in arts:
            stats["n_input"] += 1
            stats["n_input_by_source"][a.source] += 1
            if a.source.strip().lower() in excluded:
                stats["n_excluded"] += 1
                continue
            key = (a.published_date.strip(), normalize_article_text(a.content))
            first = by_id.get(a.article_id)
            if first is not None:
                stats["n_dup_id"] += 1
            else:
                first = by_key.get(key)
                if first is not None:
                    stats["n_dup_date_text"] += 1
            if first is None:
                a.tickers_mentioned = sorted(set(a.tickers_mentioned))
                a.categories = list(dict.fromkeys(a.categories))
                by_key[key] = a
                by_id[a.article_id] = a
                continue
            first.tickers_mentioned = sorted(
                set(first.tickers_mentioned) | set(a.tickers_mentioned))
            for c in a.categories:
                if c not in first.categories:
                    first.categories.append(c)
    articles = list(by_key.values())
    stats["n_unique"] = len(articles)
    stats["n_unique_by_source"] = Counter(a.source for a in articles)
    stats["n_input_by_source"] = dict(stats["n_input_by_source"])
    stats["n_unique_by_source"] = dict(stats["n_unique_by_source"])
    return articles, stats


# ══════════════════════════════════════════════════════════════════════
# Phases
# ══════════════════════════════════════════════════════════════════════


def phase_build(args, store_dir: Path, tickers):
    from sde_causal_generator.news_rag import TemporalNewsStore

    sqlite_path = store_dir / "chroma.sqlite3"
    if sqlite_path.exists():
        raise SystemExit(f"{sqlite_path} already exists: corpus v2 is built once; "
                         "refusing to ingest again")
    wiki, sec = input_files(Path(args.data_root), tickers, args.sec_range,
                            args.pinned_glob)
    if not wiki:
        raise SystemExit("no pinned Wikipedia cache found; run "
                         "scripts/pin_wikipedia_revisions.py first")
    articles, stats = load_corpus(wiki + sec)
    print(json.dumps({k: v for k, v in stats.items() if k != "files"}, indent=2))

    store = TemporalNewsStore(persist_dir=str(store_dir),
                              exclude_sources=EXCLUDED_SOURCES)
    counts = store.ingest_articles(articles, batch_size=args.batch_size)
    n = store._collection.count()
    by_source = {}
    for s in sorted({a.source for a in articles} | set(EXCLUDED_SOURCES)):
        got = store._collection.get(where={"source": s}, include=[])
        by_source[s] = len(got["ids"])
    report = {"ingest": counts, "collection_count": n,
              "collection_by_source": by_source, "load": stats}
    with open(store_dir / "build_report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1, default=str)
    print(json.dumps({"collection_count": n, "by_source": by_source,
                      "ingest": counts}, indent=2))
    store.close()


def _sqlite_ro(path: Path):
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def sqlite_counts(path: Path, tickers):
    """Counts read straight from chroma.sqlite3 (read-only, no Chroma)."""
    con = _sqlite_ro(path)
    try:
        integrity = [r[0] for r in con.execute("PRAGMA integrity_check").fetchall()]
        n = con.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
        by_source = dict(con.execute(
            "SELECT string_value, COUNT(*) FROM embedding_metadata "
            "WHERE key='source' GROUP BY string_value").fetchall())
        tick_rows = con.execute(
            "SELECT string_value FROM embedding_metadata WHERE key='tickers'"
        ).fetchall()
    finally:
        con.close()
    per_ticker = Counter()
    n_general = 0
    for (val,) in tick_rows:
        ts = [t for t in (val or "").split(",") if t]
        if not ts:
            n_general += 1
        for t in ts:
            per_ticker[t] += 1
    rel = {t: {"ticker_specific": per_ticker.get(t, 0), "general": n_general,
               "retrievable": per_ticker.get(t, 0) + n_general}
           for t in tickers}
    return {"integrity_check": integrity, "count": n, "by_source": by_source,
            "n_general": n_general, "per_ticker": rel,
            "n_multi_ticker_docs": sum(
                1 for (v,) in tick_rows if len([t for t in (v or "").split(",") if t]) > 1)}


def phase_verify(args, store_dir: Path, tickers):
    from sde_causal_generator.news_rag import STORE_MANIFEST, file_sha256

    sqlite_path = store_dir / "chroma.sqlite3"
    if not sqlite_path.is_file():
        raise SystemExit(f"{sqlite_path} not found")
    counts = sqlite_counts(sqlite_path, tickers)
    ok_integrity = counts["integrity_check"] == ["ok"]
    n_fred = sum(v for k, v in counts["by_source"].items()
                 if k.lower() in EXCLUDED_SOURCES)
    sha = file_sha256(sqlite_path)
    os.chmod(sqlite_path, os.stat(sqlite_path).st_mode
             & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))

    build_report = {}
    if (store_dir / "build_report.json").is_file():
        with open(store_dir / "build_report.json", "r", encoding="utf-8") as fh:
            build_report = json.load(fh)
    wiki, sec = input_files(Path(args.data_root), tickers, args.sec_range,
                            args.pinned_glob)
    try:
        import chromadb
        chroma_version = chromadb.__version__
    except Exception:  # pragma: no cover
        chroma_version = "unknown"
    corpus_tag = f"{STORE_DIRNAME}@{sha[:16]}"
    manifest = {
        "corpus_tag": corpus_tag,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "sqlite_sha256": sha,
        "sqlite_size": sqlite_path.stat().st_size,
        "integrity_check": counts["integrity_check"],
        "count": counts["count"],
        "count_by_source": counts["by_source"],
        "n_general": counts["n_general"],
        "per_ticker": counts["per_ticker"],
        "n_multi_ticker_docs": counts["n_multi_ticker_docs"],
        "excluded_sources": list(EXCLUDED_SOURCES),
        "n_excluded_docs_in_store": n_fred,
        "inputs": {str(f): file_sha256(f) for f in wiki + sec},
        "load_stats": build_report.get("load"),
        "ingest": build_report.get("ingest"),
        "collection_count_at_build": build_report.get("collection_count"),
        "embedding_model": "all-MiniLM-L6-v2",
        "document_text": "f'{title}. {content[:500]}'",
        "article_id": "md5(source|date|title|normalized_content), "
                      "normalized = lowercase, collapsed whitespace, stripped",
        "chromadb_version": chroma_version,
        "readonly_note": "chromadb writes on open and cannot open a write-"
                         "protected sqlite; use NewsRAGManager(cache_mode="
                         "'readonly'), which queries a private copy",
    }
    with open(store_dir / STORE_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1, default=str)

    # ── read-only smoke query; the store must not change ─────────────
    from sde_causal_generator.news_rag import NewsRAGManager

    before = (sqlite_path.stat().st_size, sqlite_path.stat().st_mtime_ns)
    rag = NewsRAGManager(persist_dir=str(store_dir), cache_mode="readonly",
                         readonly_workdir=args.readonly_workdir)
    ctx, st = rag.get_context_with_stats("AAPL", "2020-03-08", "2020-03-22")
    n_ro = rag.store.get_stats()["total_articles"]
    rag.close()
    after = (sqlite_path.stat().st_size, sqlite_path.stat().st_mtime_ns)
    sha_after = file_sha256(sqlite_path)

    summary = {
        "corpus_tag": corpus_tag,
        "count": counts["count"],
        "count_by_source": counts["by_source"],
        "integrity_check": counts["integrity_check"],
        "fred_docs": n_fred,
        "sqlite_sha256": sha,
        "sqlite_mode": oct(sqlite_path.stat().st_mode & 0o777),
        "readonly_smoke": {"count": n_ro, "window": "AAPL 2020-03-08..2020-03-22",
                           **{k: st[k] for k in ("n_retrieved", "n_unique",
                                                  "n_duplicates", "n_excluded")},
                           "sources_count": st["sources_count"]},
        "store_unchanged_by_readonly_open": before == after and sha == sha_after,
    }
    print(json.dumps(summary, indent=2))
    print("\nper-ticker relevance (ticker-specific + general = retrievable):")
    for t, r in counts["per_ticker"].items():
        print(f"  {t:5s} specific={r['ticker_specific']:4d} general={r['general']} "
              f"retrievable={r['retrievable']}")
    if not ok_integrity or n_fred or not summary["store_unchanged_by_readonly_open"]:
        raise SystemExit("VERIFY FAILED")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=str(REPO))
    ap.add_argument("--config", default="configs/a1.yaml")
    ap.add_argument("--sec-range", default="2018-01-02_2023-12-29",
                    help="date range in the SEC cache file names (the run's corpus range)")
    ap.add_argument("--pinned-glob", default="wikipedia_pinned_2018-01-01_2023-12-31.json")
    ap.add_argument("--phase", choices=["all", "build", "verify"], default="all")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--readonly-workdir", default=None,
                    help="parent dir for the read-only private copy (default: tmp)")
    args = ap.parse_args()

    import yaml

    root = Path(args.data_root)
    with open(root / args.config, "r", encoding="utf-8") as fh:
        tickers = yaml.safe_load(fh)["tickers"]
    store_dir = root / "cache" / STORE_DIRNAME
    store_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "build":
        phase_build(args, store_dir, tickers)
    elif args.phase == "verify":
        phase_verify(args, store_dir, tickers)
    else:
        cmd = [sys.executable, __file__, "--phase", "build",
               "--data-root", args.data_root, "--config", args.config,
               "--sec-range", args.sec_range, "--pinned-glob", args.pinned_glob,
               "--batch-size", str(args.batch_size)]
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit(f"build phase failed (rc={rc})")
        phase_verify(args, store_dir, tickers)


if __name__ == "__main__":
    main()
