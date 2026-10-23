# -*- coding: utf-8 -*-
"""
Temporal News RAG for LLM Factor Extraction.

Provides time-bounded retrieval of news articles to ground LLM
causal factor extraction in contemporaneous information, mitigating
hindsight bias and narrative fallacy.

Supported vector-store backend: ChromaDB (local, persistent).

News sources (ingestors):
    - Wikipedia Current Events  (curated daily world-event summaries)
    - FRED                      (macroeconomic indicator releases)
    - SEC EDGAR                 (8-K / 10-K / 10-Q filings)
    - Custom CSV / JSON         (user-provided corpora)
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import tempfile
import time as _time
import urllib.error
import urllib.parse
import urllib.request
import uuid
import weakref
import zipfile
import shutil
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .data_structures import NewsArticle, normalize_article_text


#: Number of candidates fetched from Chroma before de-duplication and the
#: ticker filter.  A 2-week window holds ~350-400 Wikipedia events, so 600
#: returns essentially every in-window document, ranked by similarity.
DEFAULT_OVERFETCH = 600

#: Name of the manifest written next to a pre-built store (corpus v2).
STORE_MANIFEST = "MANIFEST.json"


def file_sha256(path, chunk: int = 1 << 20) -> str:
    """sha256 of a file, streamed."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _is_uuid(name: str) -> bool:
    try:
        uuid.UUID(name)
        return True
    except ValueError:
        return False


def _remove_tree(path: str) -> None:
    shutil.rmtree(path, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════
# Vector Store
# ══════════════════════════════════════════════════════════════════════


# ── context v2.1 item text ──────────────────────────────────────────────
#: trailing source tag, e.g. "(Reuters)", tolerating wiki remnants after it
SOURCE_TAG = re.compile(r"\s*\([^()]{2,80}\)\]?\s*(?:<!--.*)?[\].\s\"]*$", re.S)
_WIKI_PREFIX = re.compile(r"^World Event: .*?\. (.*)$", re.S)


def item_body(a: Dict) -> str:
    """Event text of a stored item. Wikipedia items are stored as
    "<title>. <text>" with title = "World Event: " + the first 80 characters
    of the text (some titles carry a trailing space); wiki comments removed."""
    c, t = a["content"].strip(), (a.get("title") or "").strip()
    body = None
    for sep in (". ", " . "):
        if t and c.startswith(t + sep):
            body = c[len(t) + len(sep):]
            break
    if body is None:
        m = _WIKI_PREFIX.match(c)
        body = m.group(1) if m else c
    return re.sub(r"<!--.*?(-->|$)", "", body, flags=re.S).strip()


def has_source_tag(a: Dict) -> bool:
    return bool(SOURCE_TAG.search(item_body(a)))


def item_text(a: Dict) -> str:
    """Item body without its trailing source tag (what a v2.1 prompt shows)."""
    return SOURCE_TAG.sub("", item_body(a)).strip()


class TemporalNewsStore:
    """
    ChromaDB-backed vector store with **strict temporal filtering**.

    Design principle: retrieval *never* returns articles published
    after the query ``end_date``, preventing information leakage.

    Read-only mode
    --------------
    chromadb 1.5.x writes to ``chroma.sqlite3`` every time a client opens
    it (even for a pure query), and it refuses to open a write-protected
    file.  ``readonly=True`` therefore never lets Chroma touch
    ``persist_dir``: the sqlite file and the segment directories are copied
    to a private, disposable working directory and Chroma opens the copy.
    The canonical store's bytes, size and mtime stay unchanged.  If a
    ``MANIFEST.json`` sits next to the store, the canonical sqlite file is
    checked against its sha256 before the copy, and its
    ``excluded_sources`` are applied at retrieval time.
    """

    def __init__(
        self,
        persist_dir: str = "cache/news_vectorstore",
        embedding_model: str = "all-MiniLM-L6-v2",
        readonly: bool = False,
        readonly_workdir: Optional[str] = None,
        exclude_sources: Optional[Iterable[str]] = None,
        expected_sha256: Optional[str] = None,
    ):
        self.persist_dir = persist_dir
        self.embedding_model_name = embedding_model
        self.readonly = bool(readonly)
        self.exclude_sources = frozenset(
            s.strip().lower() for s in (exclude_sources or ())
        )
        self.manifest: Dict = {}
        self.working_dir = persist_dir
        self._workdir_root: Optional[str] = None
        self._finalizer = None

        self._client = None
        self._embedder = None
        self._collection = None
        self._query_embeddings: Dict[str, list] = {}

        if self.readonly:
            self._open_private_copy(readonly_workdir, expected_sha256)
        else:
            os.makedirs(persist_dir, exist_ok=True)
        self._init_backend()

    # ── backend ─────────────────────────────────────────────────────

    def _open_private_copy(
        self, readonly_workdir: Optional[str], expected_sha256: Optional[str]
    ) -> None:
        src = Path(self.persist_dir)
        sqlite_path = src / "chroma.sqlite3"
        if not sqlite_path.is_file():
            raise FileNotFoundError(
                f"read-only RAG store not found: {sqlite_path} "
                "(build it first, e.g. scripts/build_corpus_v2.py)"
            )
        manifest_path = src / STORE_MANIFEST
        if manifest_path.is_file():
            with open(manifest_path, "r", encoding="utf-8") as fh:
                self.manifest = json.load(fh)
            if expected_sha256 is None:
                expected_sha256 = self.manifest.get("sqlite_sha256")
            self.exclude_sources = self.exclude_sources | frozenset(
                s.lower() for s in self.manifest.get("excluded_sources", [])
            )
        if expected_sha256:
            actual = file_sha256(sqlite_path)
            if actual != expected_sha256:
                raise RuntimeError(
                    f"RAG store {sqlite_path} sha256 {actual} does not match "
                    f"the expected {expected_sha256}; refusing to use it"
                )

        if readonly_workdir:
            os.makedirs(readonly_workdir, exist_ok=True)
        root = tempfile.mkdtemp(prefix="rag_ro_", dir=readonly_workdir or None)
        self._workdir_root = root
        self._finalizer = weakref.finalize(self, _remove_tree, root)
        dst = Path(root) / "store"
        dst.mkdir()
        for child in sorted(src.iterdir()):
            if child.name.startswith("chroma.sqlite3") and child.is_file():
                shutil.copyfile(child, dst / child.name)
            elif child.is_dir() and _is_uuid(child.name):
                shutil.copytree(child, dst / child.name,
                                copy_function=shutil.copyfile)
        self.working_dir = str(dst)

    def _init_backend(self):
        try:
            import chromadb
            from chromadb.config import Settings
        except ImportError:
            raise ImportError(
                "chromadb is required for the news RAG store. "
                "Install it with:  pip install chromadb"
            )

        self._client = chromadb.PersistentClient(
            path=self.working_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        if self.readonly:
            # Never create anything in read-only mode.
            self._collection = self._client.get_collection(name="news_temporal")
        else:
            self._collection = self._client.get_or_create_collection(
                name="news_temporal",
                metadata={"hnsw:space": "cosine"},
            )

    def close(self) -> None:
        """Release the Chroma handles; delete the private copy if any."""
        self._collection = None
        self._client = None
        if self._finalizer is not None:
            self._finalizer()
            self._finalizer = None

    def _get_embedder(self):
        if self._embedder is None:
            try:
                import logging
                from sentence_transformers import SentenceTransformer

                # Suppress noisy "UNEXPECTED key" warnings from the
                # BertModel weight-loading report (position_ids).
                _st_logger = logging.getLogger("sentence_transformers.SentenceTransformer")
                _prev = _st_logger.level
                _st_logger.setLevel(logging.ERROR)

                self._embedder = SentenceTransformer(
                    self.embedding_model_name
                )

                _st_logger.setLevel(_prev)
            except ImportError:
                raise ImportError(
                    "sentence-transformers is required. "
                    "Install it with:  pip install sentence-transformers"
                )
        return self._embedder

    # ── ingest ──────────────────────────────────────────────────────

    @staticmethod
    def document_text(article: NewsArticle) -> str:
        """The text that is embedded and stored for an article."""
        return f"{article.title}. {article.content[:500]}"

    @staticmethod
    def _metadata(article: NewsArticle, tickers: List[str],
                  categories: List[str]) -> Dict:
        return {
            "published_date": article.published_date,
            "published_date_epoch": int(
                pd.Timestamp(article.published_date).timestamp()
            ),
            "source": article.source,
            "tickers": ",".join(tickers),
            "categories": ",".join(categories),
            "title": article.title,
        }

    def ingest_articles(
        self, articles: List[NewsArticle], batch_size: int = 100
    ) -> Dict[str, int]:
        """Ingest news articles into the vector store (idempotent).

        * Articles whose ``source`` is in ``exclude_sources`` are dropped.
        * Articles sharing an ``article_id`` (content-deterministic, see
          :func:`~sde_causal_generator.data_structures.compute_article_id`)
          are stored once; their ``tickers_mentioned`` and ``categories``
          are merged.
        * Ids already in the store are not re-embedded; only their ticker
          and category lists are merged if they gained new entries.  So
          ingesting the same articles twice leaves the count unchanged.

        Returns counts ``{n_input, n_excluded, n_unique, n_added, n_updated}``.
        """
        if self.readonly:
            raise RuntimeError(
                "TemporalNewsStore was opened read-only; ingestion is disabled"
            )
        counts = {"n_input": len(articles), "n_excluded": 0, "n_unique": 0,
                  "n_added": 0, "n_updated": 0}
        if not articles:
            return counts

        merged: Dict[str, list] = {}
        for a in articles:
            if a.source.strip().lower() in self.exclude_sources:
                counts["n_excluded"] += 1
                continue
            entry = merged.get(a.article_id)
            if entry is None:
                merged[a.article_id] = [a, list(a.tickers_mentioned),
                                        list(a.categories)]
            else:
                entry[1].extend(t for t in a.tickers_mentioned if t not in entry[1])
                entry[2].extend(c for c in a.categories if c not in entry[2])
        counts["n_unique"] = len(merged)
        if not merged:
            return counts

        print(f"    Ingesting {len(merged)} unique articles "
              f"({len(articles)} given, {counts['n_excluded']} excluded) …")
        embedder = None
        items = list(merged.items())
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            ids = [aid for aid, _ in batch]
            existing = self._collection.get(ids=ids, include=["metadatas"])
            existing_meta = dict(zip(existing["ids"], existing["metadatas"]))

            new_ids, new_texts, new_meta = [], [], []
            upd_ids, upd_meta = [], []
            for aid, (art, tickers, cats) in batch:
                old = existing_meta.get(aid)
                if old is None:
                    new_ids.append(aid)
                    new_texts.append(self.document_text(art))
                    new_meta.append(self._metadata(art, sorted(set(tickers)),
                                                   sorted(set(cats))))
                    continue
                old_t = [t for t in (old.get("tickers") or "").split(",") if t]
                old_c = [c for c in (old.get("categories") or "").split(",") if c]
                all_t = sorted(set(old_t) | set(tickers))
                all_c = sorted(set(old_c) | set(cats))
                if all_t != sorted(set(old_t)) or all_c != sorted(set(old_c)):
                    upd_ids.append(aid)
                    upd_meta.append(self._metadata(art, all_t, all_c))

            if new_ids:
                if embedder is None:
                    embedder = self._get_embedder()
                embeddings = embedder.encode(new_texts).tolist()
                self._collection.add(
                    ids=new_ids,
                    documents=new_texts,
                    embeddings=embeddings,
                    metadatas=new_meta,
                )
                counts["n_added"] += len(new_ids)
            if upd_ids:
                self._collection.update(ids=upd_ids, metadatas=upd_meta)
                counts["n_updated"] += len(upd_ids)

        print(f"    ✓ Ingested: {counts['n_added']} added, "
              f"{counts['n_updated']} metadata-merged, "
              f"{counts['n_unique'] - counts['n_added'] - counts['n_updated']}"
              f" already present")
        return counts

    # ── retrieve ────────────────────────────────────────────────────

    def _embed_query(self, query: str) -> list:
        emb = self._query_embeddings.get(query)
        if emb is None:
            emb = self._get_embedder().encode([query]).tolist()
            self._query_embeddings[query] = emb
        return emb

    def retrieve_with_stats(
        self,
        query: str,
        ticker: str,
        start_date: str,
        end_date: str,
        max_results: int = 20,
        include_general: bool = True,
        overfetch: int = DEFAULT_OVERFETCH,
        item_filter: Optional[Callable[[Dict], Optional[str]]] = None,
        prefer: Optional[Callable[[Dict], bool]] = None,
    ) -> Tuple[List[Dict], Dict]:
        """Retrieve news with strict temporal bounds and de-duplication.

        Context v2.1: ``item_filter(article) -> reason | None`` drops whole
        candidates (e.g. topic headers, items with a redacted sentence)
        BEFORE the cut to ``max_results``, so the context is refilled from
        surviving items and no marker is left; dropped counts by reason go
        to ``stats["n_dropped_by_reason"]``.  ``prefer(article) -> bool``
        ranks matching items first (e.g. items naming the company); they are
        flagged ``names_company``.

        Only articles published within ``[start_date, end_date]`` are
        considered.  Up to ``max(overfetch, 3 * max_results)`` candidates
        are fetched; those that are neither ticker-specific nor general,
        or whose source is excluded, are dropped; duplicates by
        ``(date, normalised document text)`` are collapsed to their
        best-ranked copy; then up to ``max_results`` unique articles are
        kept (ticker-specific first, then by relevance).

        Returns ``(articles, stats)`` with ``stats`` holding
        ``n_retrieved`` (raw Chroma hits), ``n_excluded``, ``n_eligible``,
        ``n_duplicates``, ``n_unique_candidates`` and ``n_unique``
        (articles returned).
        """
        stats = {"n_retrieved": 0, "n_excluded": 0, "n_eligible": 0,
                 "n_duplicates": 0, "n_unique_candidates": 0, "n_unique": 0}
        if item_filter is not None or prefer is not None:
            stats.update({"n_dropped_by_reason": {}, "n_names_company": 0,
                          "n_candidates_capped": 0})
        if self._collection is None:
            return [], stats
        total = self._collection.count()
        if total == 0:
            return [], stats

        query_embedding = self._embed_query(query)

        # ChromaDB $gte/$lte require numeric operands → use epoch ints
        start_epoch = int(pd.Timestamp(start_date).timestamp())
        end_epoch = int(pd.Timestamp(end_date).timestamp())

        where_filter = {
            "$and": [
                {"published_date_epoch": {"$gte": start_epoch}},
                {"published_date_epoch": {"$lte": end_epoch}},
            ]
        }

        n_results = min(max(int(overfetch), max_results * 3), total)
        results = self._collection.query(
            query_embeddings=query_embedding,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        articles: List[Dict] = []
        seen = set()
        if results and results["documents"]:
            for doc, meta, dist in zip(
                results["documents"][0],
                results["metadatas"][0],
                results["distances"][0],
            ):
                stats["n_retrieved"] += 1
                source = meta.get("source", "")
                if source.strip().lower() in self.exclude_sources:
                    stats["n_excluded"] += 1
                    continue
                tickers_in_article = (meta.get("tickers") or "").split(",")
                is_ticker_relevant = ticker in tickers_in_article
                is_general = not tickers_in_article or tickers_in_article == [
                    ""
                ]
                if not (is_ticker_relevant or (include_general and is_general)):
                    continue
                stats["n_eligible"] += 1
                key = (str(meta.get("published_date", "")).strip(),
                       normalize_article_text(doc))
                if key in seen:
                    stats["n_duplicates"] += 1
                    continue
                seen.add(key)
                art = {
                    "title": meta.get("title", ""),
                    "content": doc,
                    "date": meta["published_date"],
                    "source": source,
                    "relevance": 1.0 - dist,
                    "ticker_specific": is_ticker_relevant,
                }
                if item_filter is not None:
                    reason = item_filter(art)
                    if reason:
                        d = stats["n_dropped_by_reason"]
                        d[reason] = d.get(reason, 0) + 1
                        continue
                if prefer is not None:
                    art["names_company"] = bool(prefer(art))
                    stats["n_names_company"] += art["names_company"]
                articles.append(art)
            if "n_candidates_capped" in stats:
                stats["n_candidates_capped"] = int(stats["n_retrieved"] >= n_results)

        stats["n_unique_candidates"] = len(articles)
        articles.sort(
            key=lambda x: (x.get("names_company", False), x["ticker_specific"], x["relevance"]),
            reverse=True,
        )
        out = articles[:max_results]
        stats["n_unique"] = len(out)
        return out, stats

    def retrieve(
        self,
        query: str,
        ticker: str,
        start_date: str,
        end_date: str,
        max_results: int = 20,
        include_general: bool = True,
        overfetch: int = DEFAULT_OVERFETCH,
    ) -> List[Dict]:
        """
        Retrieve news with **strict temporal bounds**, de-duplicated.

        Only returns articles published within ``[start_date, end_date]``.
        See :meth:`retrieve_with_stats`.
        """
        articles, _ = self.retrieve_with_stats(
            query, ticker, start_date, end_date, max_results=max_results,
            include_general=include_general, overfetch=overfetch,
        )
        return articles

    # ── stats ───────────────────────────────────────────────────────

    def get_stats(self) -> Dict:
        out = {"backend": "chroma", "total_articles": 0,
               "readonly": self.readonly,
               "excluded_sources": sorted(self.exclude_sources)}
        if self._collection is not None:
            out["total_articles"] = self._collection.count()
        return out


# ══════════════════════════════════════════════════════════════════════
# News Source Ingestors
# ══════════════════════════════════════════════════════════════════════


class WikipediaCurrentEventsIngestor:
    """
    Ingest Wikipedia's 'Portal:Current events' entries.

    Human-curated daily summaries of world events — high quality,
    temporally precise, and freely available (2004–present).

    Uses the MediaWiki batch API (``action=query``) to fetch up to
    50 day-pages per request, avoiding 429 rate-limit errors.
    """

    # Polite delay (seconds) between Wikipedia API requests.
    _REQUEST_DELAY: float = 1.0
    _API_URL = "https://en.wikipedia.org/w/api.php"
    _USER_AGENT = os.environ.get(
        "TRACE_USER_AGENT", "trace-research/1.0 (https://github.com/alpha-synthesis/trace-arxiv)"
    )

    @staticmethod
    def _api_get(
        params: dict,
        max_retries: int = 3,
        request_delay: Optional[float] = None,
    ) -> dict:
        """Call the MediaWiki API (read-only GET) with exponential back-off.

        Retries on HTTP 429 / 5xx, network errors and the API's ``maxlag``
        error.  ``request_delay`` (seconds) is slept before the request;
        it defaults to ``_REQUEST_DELAY``.
        """
        params = dict(params)
        params.setdefault("format", "json")
        qs = urllib.parse.urlencode(params, doseq=True)
        url = f"{WikipediaCurrentEventsIngestor._API_URL}?{qs}"

        delay = (WikipediaCurrentEventsIngestor._REQUEST_DELAY
                 if request_delay is None else request_delay)
        _time.sleep(delay)
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(
                    url,
                    headers={
                        "User-Agent": WikipediaCurrentEventsIngestor._USER_AGENT,
                    },
                )
                resp = urllib.request.urlopen(req, timeout=30)
                data = json.loads(resp.read().decode("utf-8"))
                err = data.get("error") if isinstance(data, dict) else None
                if err and err.get("code") == "maxlag":
                    raise TimeoutError("maxlag")
                return data
            except urllib.error.HTTPError as he:
                if he.code == 429 or he.code >= 500:
                    _time.sleep(2 ** (attempt + 1))
                else:
                    raise
            except (urllib.error.URLError, TimeoutError, OSError):
                _time.sleep(2 ** (attempt + 1))
        return {}

    # ── revision-pinned day pages (corpus v2) ──────────────────────

    @staticmethod
    def day_page_title(date) -> str:
        """``Portal:Current events/2020 March 11`` for a date."""
        d = pd.Timestamp(date)
        return f"Portal:Current events/{d.year} {d.strftime('%B')} {d.day}"

    @staticmethod
    def _revision_wikitext(rev: dict) -> Optional[str]:
        slots = rev.get("slots", {}).get("main", {})
        if "content" in slots:          # formatversion=2
            return slots["content"]
        if "*" in slots:                # formatversion=1
            return slots["*"]
        return None

    @staticmethod
    def fetch_day_revision(
        date,
        pin_time: str,
        request_delay: float = 0.34,
        max_retries: int = 6,
    ) -> Dict:
        """Fetch a day page at the latest revision at or before ``pin_time``.

        ``pin_time`` is an ISO timestamp such as ``2020-03-22T23:59:59Z``.
        Uses ``rvstart=pin_time&rvdir=older&rvlimit=1``.  Returns a dict
        with ``date, title, pin_time, status`` and, when a revision exists,
        ``pageid, revid, parentid, timestamp, sha1, wikitext``.
        ``status`` is ``ok``, ``missing`` (no such page), ``no_revision``
        (page created after ``pin_time``), ``hidden`` (content suppressed)
        or ``error``.
        """
        title = WikipediaCurrentEventsIngestor.day_page_title(date)
        out = {"date": pd.Timestamp(date).strftime("%Y-%m-%d"),
               "title": title, "pin_time": pin_time}
        data = WikipediaCurrentEventsIngestor._api_get(
            {
                "action": "query",
                "titles": title,
                "prop": "revisions",
                "rvprop": "ids|timestamp|sha1|content",
                "rvslots": "main",
                "rvstart": pin_time,
                "rvdir": "older",
                "rvlimit": 1,
                "formatversion": 2,
                "maxlag": 5,
            },
            max_retries=max_retries,
            request_delay=request_delay,
        )
        pages = data.get("query", {}).get("pages", [])
        if isinstance(pages, dict):
            pages = list(pages.values())
        if not pages:
            out["status"] = "error"
            out["error"] = json.dumps(data)[:300]
            return out
        page = pages[0]
        if page.get("missing") or page.get("invalid"):
            out["status"] = "missing"
            return out
        out["pageid"] = page.get("pageid")
        revs = page.get("revisions") or []
        if not revs:
            out["status"] = "no_revision"
            return out
        rev = revs[0]
        out.update(revid=rev.get("revid"), parentid=rev.get("parentid"),
                   timestamp=rev.get("timestamp"), sha1=rev.get("sha1"))
        text = WikipediaCurrentEventsIngestor._revision_wikitext(rev)
        if text is None:
            out["status"] = "hidden"
            out["wikitext"] = ""
        else:
            out["status"] = "ok"
            out["wikitext"] = text
        return out

    @staticmethod
    def fetch_current_revision_ids(
        dates: List,
        request_delay: float = 0.34,
        max_retries: int = 6,
    ) -> Dict[str, Dict]:
        """Latest revision id/timestamp of each day page (50 per request).

        Returns ``{iso_date: {revid, timestamp, sha1}}``; pages that do not
        exist are absent.
        """
        title_to_date = {
            WikipediaCurrentEventsIngestor.day_page_title(d):
                pd.Timestamp(d).strftime("%Y-%m-%d")
            for d in dates
        }
        titles = list(title_to_date)
        out: Dict[str, Dict] = {}
        for i in range(0, len(titles), 50):
            data = WikipediaCurrentEventsIngestor._api_get(
                {
                    "action": "query",
                    "titles": "|".join(titles[i : i + 50]),
                    "prop": "revisions",
                    "rvprop": "ids|timestamp|sha1",
                    "formatversion": 2,
                    "maxlag": 5,
                },
                max_retries=max_retries,
                request_delay=request_delay,
            )
            pages = data.get("query", {}).get("pages", [])
            if isinstance(pages, dict):
                pages = list(pages.values())
            for page in pages:
                iso = title_to_date.get(page.get("title", ""))
                revs = page.get("revisions") or []
                if iso and revs and not page.get("missing"):
                    out[iso] = {"revid": revs[0].get("revid"),
                                "timestamp": revs[0].get("timestamp"),
                                "sha1": revs[0].get("sha1")}
        return out

    @staticmethod
    def _parse_day_wikitext(wikitext: str, iso_date: str) -> List[NewsArticle]:
        """Extract event articles from a single day's raw wikitext."""
        articles: List[NewsArticle] = []

        # Each top-level bullet (*) or sub-bullet (**) is an event
        for line in wikitext.split("\n"):
            line = line.strip()
            if not line.startswith("*"):
                continue

            # Strip leading *, **, etc.
            clean = re.sub(r"^\*+\s*", "", line)
            # Remove wikilinks: [[Target|Display]] → Display, [[Target]] → Target
            clean = re.sub(
                r"\[\[([^|\]]*\|)?([^\]]*)\]\]", r"\2", clean
            )
            # Remove bold/italic markup
            clean = re.sub(r"'{2,}", "", clean)
            # Remove external links: [https://... (Source)] → (Source)
            clean = re.sub(r"\[https?://\S+\s*", "", clean)
            clean = clean.strip().rstrip("]")

            if len(clean) > 25:
                articles.append(
                    NewsArticle(
                        title=f"World Event: {clean[:80]}",
                        content=clean,
                        source="wikipedia_current_events",
                        published_date=iso_date,
                        categories=["geopolitical", "world_events"],
                    )
                )
        return articles

    @staticmethod
    def fetch_month(
        year: int, month: int, max_retries: int = 3
    ) -> List[NewsArticle]:
        """Fetch all days in a month using the MediaWiki batch API."""
        import calendar

        n_days = calendar.monthrange(year, month)[1]
        # Build page titles for each day
        all_titles = [
            f"Portal:Current events/{year}_{datetime(year, month, 1).strftime('%B')}_{d}"
            for d in range(1, n_days + 1)
        ]

        articles: List[NewsArticle] = []

        # MediaWiki API allows up to 50 titles per request
        for batch_start in range(0, len(all_titles), 50):
            batch_titles = all_titles[batch_start : batch_start + 50]
            titles_str = "|".join(batch_titles)

            data = WikipediaCurrentEventsIngestor._api_get(
                {
                    "action": "query",
                    "titles": titles_str,
                    "prop": "revisions",
                    "rvprop": "content",
                    "rvslots": "main",
                },
                max_retries=max_retries,
            )

            pages = data.get("query", {}).get("pages", {})
            for _pid, page in pages.items():
                title = page.get("title", "")
                # Skip missing pages (e.g., future dates)
                if "missing" in page:
                    continue

                revisions = page.get("revisions", [])
                if not revisions:
                    continue
                wikitext = (
                    revisions[0]
                    .get("slots", {})
                    .get("main", {})
                    .get("*", "")
                )
                if not wikitext:
                    continue

                # Extract date from title: "Portal:Current events/2020 March 11"
                match = re.search(
                    r"(\d{4})[\s_](\w+)[\s_](\d+)$", title
                )
                if not match:
                    continue
                try:
                    iso_date = datetime(
                        int(match.group(1)),
                        datetime.strptime(match.group(2), "%B").month,
                        int(match.group(3)),
                    ).strftime("%Y-%m-%d")
                except (ValueError, KeyError):
                    continue

                day_articles = WikipediaCurrentEventsIngestor._parse_day_wikitext(
                    wikitext, iso_date
                )
                articles.extend(day_articles)

        return articles

    @staticmethod
    def fetch_range(start_date: str, end_date: str) -> List[NewsArticle]:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)

        all_articles: List[NewsArticle] = []
        current = start.replace(day=1)

        total_months = (
            (end.year - start.year) * 12 + end.month - start.month + 1
        )
        fetched = 0

        while current <= end:
            fetched += 1
            month_label = current.strftime("%B %Y")
            monthly = WikipediaCurrentEventsIngestor.fetch_month(
                current.year, current.month
            )
            filtered = [
                a
                for a in monthly
                if start_date <= a.published_date <= end_date
            ]
            all_articles.extend(filtered)

            if fetched % 12 == 0 or fetched == total_months:
                print(
                    f"      Progress: {fetched}/{total_months} months, "
                    f"{len(all_articles)} events so far"
                )

            current += pd.DateOffset(months=1)

        return all_articles


class FREDIngestor:
    """
    Ingest macroeconomic data releases from FRED as news events.

    Captures release dates of key indicators with their values.
    Requires the ``fredapi`` package and ``FRED_API_KEY`` env var.
    """

    KEY_SERIES = {
        "FEDFUNDS": "Federal Funds Effective Rate",
        "CPIAUCSL": "Consumer Price Index",
        "UNRATE": "Unemployment Rate",
        "GDP": "Gross Domestic Product",
        "INDPRO": "Industrial Production Index",
        "T10Y2Y": "10Y-2Y Treasury Spread",
        "VIXCLS": "VIX Volatility Index",
        "DGS10": "10-Year Treasury Rate",
    }

    @staticmethod
    def fetch_releases(
        start_date: str,
        end_date: str,
        fred_api_key: Optional[str] = None,
    ) -> List[NewsArticle]:
        api_key = fred_api_key or os.environ.get("FRED_API_KEY", "")
        if not api_key:
            print(
                "      WARNING: No FRED_API_KEY set. Skipping FRED ingest."
            )
            return []

        articles: List[NewsArticle] = []
        try:
            from fredapi import Fred  # type: ignore[import-untyped]

            fred = Fred(api_key=api_key)

            for series_id, description in FREDIngestor.KEY_SERIES.items():
                try:
                    data = fred.get_series(
                        series_id,
                        observation_start=start_date,
                        observation_end=end_date,
                    )
                    for date, value in data.items():
                        if pd.notna(value):
                            articles.append(
                                NewsArticle(
                                    title=f"FRED Release: {description}",
                                    content=(
                                        f"{description} ({series_id}) "
                                        f"reported at {value:.4f} "
                                        f"on {date.strftime('%Y-%m-%d')}."
                                    ),
                                    source="fred",
                                    published_date=date.strftime("%Y-%m-%d"),
                                    categories=["macroeconomic"],
                                )
                            )
                except Exception as e:
                    print(f"      FRED series {series_id} failed: {e}")

        except ImportError:
            print("      Install fredapi: pip install fredapi")

        return articles


class SECEdgarIngestor:
    """
    Ingest SEC EDGAR filings (10-K, 10-Q, 8-K) as news events.

    8-K filings capture material events (earnings, M&A, management
    changes) in near-real-time.
    """

    @staticmethod
    def fetch_filings(
        ticker: str,
        start_date: str,
        end_date: str,
        max_filings: int = 50,
    ) -> List[NewsArticle]:
        articles: List[NewsArticle] = []
        url = (
            f"https://efts.sec.gov/LATEST/search-index?"
            f"q=%22{ticker}%22&dateRange=custom"
            f"&startdt={start_date}&enddt={end_date}"
            f"&forms=8-K,10-K,10-Q"
            f"&from=0&size={max_filings}"
        )
        headers = {"User-Agent": os.environ.get("TRACE_USER_AGENT", "trace-research/1.0 (https://github.com/alpha-synthesis/trace-arxiv)")}

        try:
            req = urllib.request.Request(url, headers=headers)
            response = urllib.request.urlopen(req, timeout=30)
            data = json.loads(response.read().decode())

            for hit in data.get("hits", {}).get("hits", []):
                source = hit.get("_source", {})
                filing_date = source.get("file_date", "")
                form_type = source.get("form_type", "")
                display_names = source.get("display_names", [])
                entity = display_names[0] if display_names else ticker

                articles.append(
                    NewsArticle(
                        title=f"SEC {form_type}: {entity}",
                        content=(
                            f"{entity} filed {form_type} with the SEC "
                            f"on {filing_date}."
                        ),
                        source="sec_edgar",
                        published_date=filing_date,
                        tickers_mentioned=[ticker],
                        categories=["company_specific", "regulatory"],
                    )
                )
        except Exception as e:
            print(f"      SEC EDGAR fetch failed for {ticker}: {e}")

        return articles


# ══════════════════════════════════════════════════════════════════════
# RAG Manager (unified interface)
# ══════════════════════════════════════════════════════════════════════


class NewsRAGManager:
    """
    Manages news ingestion from multiple sources and provides a
    unified retrieval interface with strict temporal bounds.

    Usage::

        rag = NewsRAGManager(persist_dir="cache/news")
        rag.build_corpus(ticker="AAPL",
                         start_date="2005-01-01",
                         end_date="2006-12-31")
        context = rag.get_context(ticker="AAPL",
                                  start_date="2006-03-01",
                                  end_date="2006-03-31")
    """

    CACHE_MODES = ("reuse", "refresh", "readonly")

    def __init__(
        self,
        persist_dir: str = "cache/news_rag",
        embedding_model: str = "all-MiniLM-L6-v2",
        cache_mode: str = "reuse",
        readonly_workdir: Optional[str] = None,
        exclude_sources: Optional[Iterable[str]] = None,
        expected_sha256: Optional[str] = None,
        overfetch: int = DEFAULT_OVERFETCH,
    ):
        """
        Parameters
        ----------
        cache_mode : str
            ``"reuse"`` / ``"refresh"``: see :meth:`build_corpus`.
            ``"readonly"``: open a pre-built store (e.g. ``cache/rag_store_v2``)
            for queries only.  :meth:`build_corpus` becomes a no-op and the
            store on disk is never written (Chroma opens a private copy; see
            :class:`TemporalNewsStore`).
        readonly_workdir : str | None
            Parent directory of the private copy (default: system temp dir).
        exclude_sources : iterable of str | None
            Sources never ingested nor returned (e.g. ``["fred"]``).
        expected_sha256 : str | None
            Read-only mode: required sha256 of ``chroma.sqlite3``.  Defaults
            to the value in the store's ``MANIFEST.json`` when present.
        overfetch : int
            Candidates fetched per query before de-duplication.
        """
        if cache_mode not in self.CACHE_MODES:
            raise ValueError(
                f"cache_mode must be one of {self.CACHE_MODES}, got {cache_mode!r}"
            )
        self.cache_mode = cache_mode
        self.readonly = cache_mode == "readonly"
        self.overfetch = int(overfetch)
        self.store = TemporalNewsStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model,
            readonly=self.readonly,
            readonly_workdir=readonly_workdir,
            exclude_sources=exclude_sources,
            expected_sha256=expected_sha256,
        )
        self.corpus_built = self.readonly

    def close(self) -> None:
        self.store.close()

    # ── build ───────────────────────────────────────────────────────

    def build_corpus(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        sources: Optional[List[str]] = None,
        fred_api_key: Optional[str] = None,
        cache_mode: Optional[str] = None,
    ):
        """Build the news corpus from multiple sources.

        Default sources: ``["wikipedia", "fred", "sec_edgar"]``.

        Parameters
        ----------
        cache_mode : str | None
            ``"reuse"``  – load articles from disk cache if available,
            only download what is missing.
            ``"refresh"`` – delete existing cache and re-download
            everything from scratch.
            ``"readonly"`` – only valid when the manager itself was opened
            with ``cache_mode="readonly"``; then nothing is loaded,
            downloaded or ingested.
            ``None`` uses the manager's ``cache_mode``.
        """
        if self.readonly:
            print("    RAG cache_mode=readonly → build_corpus skipped "
                  "(no ingestion, store opened for queries only)")
            return
        cache_mode = cache_mode or self.cache_mode
        if cache_mode == "readonly":
            raise ValueError(
                "cache_mode='readonly' must be passed to NewsRAGManager(...), "
                "so that the store is opened read-only from the start"
            )
        if sources is None:
            sources = ["wikipedia", "fred", "sec_edgar"]
        sources = [s for s in sources
                   if s.strip().lower() not in self.store.exclude_sources]

        cache_dir = Path(self.store.persist_dir) / "article_cache"

        # ── refresh mode: wipe caches ──────────────────────────────
        if cache_mode == "refresh" and cache_dir.exists():
            print("    Cache mode = refresh → deleting old article cache …")
            shutil.rmtree(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)

        all_articles: List[NewsArticle] = []
        print(
            f"  > Building news corpus for {ticker} "
            f"({start_date} to {end_date}) …"
        )

        for src in sources:
            cache_file = (
                cache_dir
                / f"{src}_{ticker}_{start_date}_{end_date}.json"
            )

            # ── try loading from cache ─────────────────────────────
            if cache_mode == "reuse" and cache_file.exists():
                print(f"    Loading cached {src} articles from {cache_file.name} …")
                with open(cache_file, "r", encoding="utf-8") as fh:
                    cached = json.load(fh)
                arts = [NewsArticle.from_dict(d) for d in cached]
                print(f"      → {len(arts)} articles (cached)")
                all_articles.extend(arts)
                continue

            # ── download from source ───────────────────────────────
            arts: List[NewsArticle] = []
            if src == "wikipedia":
                print("    Fetching Wikipedia Current Events …")
                arts = WikipediaCurrentEventsIngestor.fetch_range(
                    start_date, end_date
                )
                print(f"      → {len(arts)} events")
            elif src == "fred":
                print("    Fetching FRED releases …")
                arts = FREDIngestor.fetch_releases(
                    start_date, end_date, fred_api_key
                )
                print(f"      → {len(arts)} releases")
            elif src == "sec_edgar":
                print("    Fetching SEC EDGAR filings …")
                arts = SECEdgarIngestor.fetch_filings(
                    ticker, start_date, end_date
                )
                print(f"      → {len(arts)} filings")
            else:
                print(f"    ⚠ Unknown source '{src}', skipping.")
                continue

            # ── save to cache ──────────────────────────────────────
            if arts:
                with open(cache_file, "w", encoding="utf-8") as fh:
                    json.dump(
                        [a.to_dict() for a in arts], fh,
                        ensure_ascii=False,
                    )
                print(f"      Cached → {cache_file.name}")

            all_articles.extend(arts)

        if all_articles:
            self.store.ingest_articles(all_articles)
            self.corpus_built = True

        stats = self.store.get_stats()
        print(f"    ✓ Corpus ready: {stats['total_articles']} total articles")

    # ── retrieve ────────────────────────────────────────────────────

    @staticmethod
    def default_query(ticker: str) -> str:
        return (
            f"Market moving events affecting {ticker} stock price "
            f"macroeconomic policy interest rates earnings"
        )

    def get_context(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_articles: int = 50,
        query_template: Optional[str] = None,
        content_limit: int = 500,
    ) -> str:
        """Retrieve and format news context for a time window.

        Returns a formatted string ready for LLM prompt injection,
        containing **only** articles published within
        ``[start_date, end_date]``, with duplicates collapsed.

        Parameters
        ----------
        content_limit : int
            Maximum characters per article content in the output.
        """
        text, _ = self.get_context_with_stats(
            ticker, start_date, end_date, max_articles=max_articles,
            query_template=query_template, content_limit=content_limit,
        )
        return text

    def get_context_with_stats(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
        max_articles: int = 50,
        query_template: Optional[str] = None,
        content_limit: int = 500,
        item_filter: Optional[Callable[[Dict], Optional[str]]] = None,
        prefer: Optional[Callable[[Dict], bool]] = None,
        compact: bool = False,
    ) -> Tuple[str, Dict]:
        """:meth:`get_context` plus retrieval statistics.

        ``stats`` holds the :meth:`TemporalNewsStore.retrieve_with_stats`
        counts and ``sources_count`` (articles per source in the context).
        """
        if query_template is None:
            query_template = self.default_query(ticker)

        articles, stats = self.store.retrieve_with_stats(
            query=query_template,
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            max_results=max_articles,
            overfetch=self.overfetch,
            item_filter=item_filter,
            prefer=prefer,
        )
        stats["sources_count"] = dict(
            sorted(Counter(a["source"] for a in articles).items())
        )
        stats["n_ticker_specific"] = sum(
            1 for a in articles if a.get("ticker_specific"))
        return self._format_context(
            ticker, start_date, end_date, articles, content_limit,
            compact=compact), stats

    @staticmethod
    def _format_context(ticker, start_date, end_date, articles,
                        content_limit, compact: bool = False) -> str:
        if not articles:
            return (
                f"[No news articles found for {ticker} "
                f"between {start_date} and {end_date}]"
            )

        lines = [
            f"=== NEWS CONTEXT FOR {ticker} "
            f"({start_date} to {end_date}) ===",
            f"Total articles retrieved: {len(articles)}",
            "",
        ]

        specific = [a for a in articles
                    if a.get("ticker_specific") or a.get("names_company")]
        general = [a for a in articles
                   if not (a.get("ticker_specific") or a.get("names_company"))]

        def _item(i, a):
            if compact:
                # context v2.1: one line per item, the event text only (no
                # duplicated, truncated title line and no source tag), so the
                # prompt shows exactly the text the item filter checked
                lines.append(f"[{i}] ({a['date']}) {item_text(a)[:content_limit]}")
            else:
                lines.append(
                    f"[{i}] ({a['date']}) [{a['source']}] {a['title']}"
                )
                lines.append(f"    {a['content'][:content_limit]}")
            lines.append("")

        if specific:
            lines.append(f"── {ticker}-SPECIFIC NEWS ──")
            for i, a in enumerate(specific, 1):
                _item(i, a)

        if general:
            lines.append("── MACRO/GEOPOLITICAL CONTEXT ──")
            for i, a in enumerate(general, 1):
                _item(i, a)

        lines.append("=== END NEWS CONTEXT ===")
        return "\n".join(lines)

    # ── custom corpus ───────────────────────────────────────────────

    def ingest_custom_csv(
        self,
        csv_path: str,
        date_col: str = "date",
        title_col: str = "title",
        content_col: str = "content",
        ticker_col: Optional[str] = "ticker",
        source_name: str = "custom",
    ):
        """Ingest a custom CSV of news articles."""
        df = pd.read_csv(csv_path)
        articles: List[NewsArticle] = []
        for _, row in df.iterrows():
            tickers: List[str] = []
            if ticker_col and ticker_col in df.columns:
                tickers = [str(row[ticker_col])]

            articles.append(
                NewsArticle(
                    title=str(row[title_col]),
                    content=str(row.get(content_col, "")),
                    source=source_name,
                    published_date=str(row[date_col]),
                    tickers_mentioned=tickers,
                )
            )

        self.store.ingest_articles(articles)
        print(f"    ✓ Ingested {len(articles)} articles from {csv_path}")
