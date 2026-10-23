# -*- coding: utf-8 -*-
"""Corpus v2: deterministic article ids, idempotent ingestion, retrieval
de-duplication, FRED exclusion and the read-only store mode.

Chroma tests use a small hash-based embedder (no model download, no GPU).
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sde_causal_generator.data_structures import (
    NewsArticle,
    compute_article_id,
    normalize_article_text,
)

REPO = Path(__file__).resolve().parents[1]


# ══════════════════════════════════════════════════════════════════════
# Content-deterministic ids
# ══════════════════════════════════════════════════════════════════════


def _art(title="World Event: Fed raises rates", content="The Fed raises rates.",
         source="wikipedia_current_events", date="2020-03-16", tickers=None):
    return NewsArticle(title=title, content=content, source=source,
                       published_date=date, tickers_mentioned=tickers or [])


class TestArticleIds:

    def test_same_article_same_id_across_instances(self):
        assert _art().article_id == _art().article_id
        # no hidden counter: creating many others does not change the id
        ids = {_art(content=f"x {i} long enough content").article_id for i in range(50)}
        assert len(ids) == 50
        assert _art().article_id == _art().article_id

    def test_id_formula(self):
        a = _art()
        raw = "wikipedia_current_events|2020-03-16|World Event: Fed raises rates|the fed raises rates."
        assert a.article_id == hashlib.md5(raw.encode()).hexdigest()
        assert a.article_id == compute_article_id(a.source, a.published_date,
                                                  a.title, a.content)

    def test_normalisation_lowercase_whitespace_strip(self):
        assert normalize_article_text("  The  FED\n raises\t rates.  ") == \
            "the fed raises rates."
        a = _art(content="The Fed raises rates.")
        b = _art(content="  the   FED\nraises rates.  ")
        assert a.article_id == b.article_id

    def test_id_depends_on_source_date_title_content(self):
        base = _art().article_id
        assert _art(source="sec_edgar").article_id != base
        assert _art(date="2020-03-17").article_id != base
        assert _art(title="Other").article_id != base
        assert _art(content="Something else entirely.").article_id != base

    def test_same_id_across_processes(self):
        code = (
            "from sde_causal_generator.data_structures import NewsArticle as N;"
            "print(N(title='World Event: Fed raises rates', content='The Fed raises rates.',"
            "source='wikipedia_current_events', published_date='2020-03-16').article_id)"
        )
        ids = {
            subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                           capture_output=True, text=True, check=True).stdout.strip()
            for _ in range(2)
        }
        assert ids == {_art().article_id}

    def test_from_dict_recomputes_cached_id(self):
        d = _art().to_dict()
        d["article_id"] = "deadbeef-counter-based-id"
        a = NewsArticle.from_dict(d)
        assert a.article_id == _art().article_id

    def test_tickers_do_not_change_id(self):
        assert _art(tickers=["AAPL"]).article_id == _art(tickers=["MSFT"]).article_id


# ══════════════════════════════════════════════════════════════════════
# Chroma-backed store
# ══════════════════════════════════════════════════════════════════════

chromadb = pytest.importorskip("chromadb")

from sde_causal_generator.news_rag import (  # noqa: E402
    NewsRAGManager,
    TemporalNewsStore,
    file_sha256,
)


class HashEmbedder:
    """Deterministic 32-d embedding from token hashes (bag of words)."""

    dim = 32

    def encode(self, texts):
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            for tok in normalize_article_text(t).split():
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
            out[i] += 1e-3
        return out


@pytest.fixture(autouse=True)
def _fake_embedder(monkeypatch):
    monkeypatch.setattr(TemporalNewsStore, "_get_embedder",
                        lambda self: HashEmbedder())


def _corpus():
    arts = []
    for d in range(1, 15):
        date = f"2020-03-{d:02d}"
        for k in range(5):
            c = f"Event {k} on day {d}: the central bank and parliament act on policy item {k}."
            arts.append(NewsArticle(title=f"World Event: {c[:80]}", content=c,
                                    source="wikipedia_current_events",
                                    published_date=date,
                                    categories=["geopolitical"]))
    arts.append(NewsArticle(title="SEC 8-K: Apple Inc", content="Apple Inc filed 8-K with the SEC on 2020-03-05.",
                            source="sec_edgar", published_date="2020-03-05",
                            tickers_mentioned=["AAPL"]))
    arts.append(NewsArticle(title="SEC 8-K: Apple Inc", content="Apple Inc filed 8-K with the SEC on 2020-03-05.",
                            source="sec_edgar", published_date="2020-03-05",
                            tickers_mentioned=["MSFT"]))
    for d in range(1, 15):
        arts.append(NewsArticle(title="FRED Release: VIX Volatility Index",
                                content=f"VIX Volatility Index (VIXCLS) reported at {20 + d}.0 on 2020-03-{d:02d}.",
                                source="fred", published_date=f"2020-03-{d:02d}",
                                categories=["macroeconomic"]))
    return arts


def _count_source(store, source):
    return len(store._collection.get(where={"source": source}, include=[])["ids"])


class TestIngestion:

    def test_ingest_twice_leaves_count_unchanged(self, tmp_path):
        store = TemporalNewsStore(persist_dir=str(tmp_path / "s"))
        arts = [a for a in _corpus() if a.source != "fred"]
        c1 = store.ingest_articles(arts)
        n1 = store._collection.count()
        # same articles rebuilt from dicts (ids recomputed), ingested again
        again = [NewsArticle.from_dict(a.to_dict()) for a in arts]
        c2 = store.ingest_articles(again)
        assert store._collection.count() == n1
        assert c2["n_added"] == 0
        assert n1 == c1["n_unique"] == 70 + 1  # 70 wiki + 1 merged SEC

    def test_duplicates_merge_tickers(self, tmp_path):
        store = TemporalNewsStore(persist_dir=str(tmp_path / "s"))
        store.ingest_articles([a for a in _corpus() if a.source == "sec_edgar"])
        got = store._collection.get(where={"source": "sec_edgar"}, include=["metadatas"])
        assert len(got["ids"]) == 1
        assert got["metadatas"][0]["tickers"] == "AAPL,MSFT"
        # a later ingest with a third ticker merges into the same document
        store.ingest_articles([_art(title="SEC 8-K: Apple Inc",
                                    content="Apple Inc filed 8-K with the SEC on 2020-03-05.",
                                    source="sec_edgar", date="2020-03-05",
                                    tickers=["IBM"])])
        got = store._collection.get(where={"source": "sec_edgar"}, include=["metadatas"])
        assert len(got["ids"]) == 1
        assert got["metadatas"][0]["tickers"] == "AAPL,IBM,MSFT"

    def test_fred_never_ingested(self, tmp_path):
        store = TemporalNewsStore(persist_dir=str(tmp_path / "s"),
                                  exclude_sources=["fred"])
        counts = store.ingest_articles(_corpus())
        assert counts["n_excluded"] == 14
        assert _count_source(store, "fred") == 0
        assert _count_source(store, "wikipedia_current_events") == 70


class TestRetrievalDedup:

    def _dup_store(self, tmp_path):
        """A store with every article ingested 5 times under distinct ids,
        as the v1 counter-based ids did across tickers/processes."""
        store = TemporalNewsStore(persist_dir=str(tmp_path / "dup"))
        base = [a for a in _corpus() if a.source != "fred"]
        copies = []
        for r in range(5):
            for a in base:
                d = a.to_dict()
                copies.append(NewsArticle(**{**{k: d[k] for k in (
                    "title", "content", "source", "published_date",
                    "tickers_mentioned", "categories", "url")},
                    "article_id": f"{a.article_id}-{r}"}))
        store.ingest_articles(copies)
        assert store._collection.count() == 5 * 71
        return store

    def test_no_duplicate_title_date_content_returned(self, tmp_path):
        store = self._dup_store(tmp_path)
        for tic in ("AAPL", "MSFT", "KO"):
            arts, st = store.retrieve_with_stats(
                "central bank policy", tic, "2020-03-01", "2020-03-14",
                max_results=50)
            keys = [(a["title"], a["date"], a["content"]) for a in arts]
            assert len(keys) == len(set(keys))
            assert st["n_duplicates"] > 0
            assert st["n_unique"] == len(arts) == 50

    def test_keeps_up_to_max_unique(self, tmp_path):
        store = self._dup_store(tmp_path)
        # 3 days x 5 events = 15 unique wiki + SEC (AAPL) on 03-05
        arts = store.retrieve("central bank policy", "AAPL", "2020-03-04",
                              "2020-03-06", max_results=50)
        assert len(arts) == 16
        assert arts[0]["ticker_specific"] and arts[0]["source"] == "sec_edgar"
        arts_ko = store.retrieve("central bank policy", "KO", "2020-03-04",
                                 "2020-03-06", max_results=50)
        assert len(arts_ko) == 15
        assert not any(a["ticker_specific"] for a in arts_ko)

    def test_temporal_bounds(self, tmp_path):
        store = self._dup_store(tmp_path)
        arts = store.retrieve("policy", "AAPL", "2020-03-10", "2020-03-12",
                              max_results=200)
        assert arts and all("2020-03-10" <= a["date"] <= "2020-03-12" for a in arts)

    def test_get_context_lists_unique_articles(self, tmp_path):
        store = self._dup_store(tmp_path)
        rag = NewsRAGManager.__new__(NewsRAGManager)
        rag.store, rag.overfetch, rag.readonly, rag.cache_mode = store, 600, False, "reuse"
        text, st = rag.get_context_with_stats("AAPL", "2020-03-01", "2020-03-14",
                                              max_articles=30)
        body = [ln.strip() for ln in text.split("\n") if ln.startswith("    ")]
        assert len(body) == 30 == len(set(body)) == st["n_unique"]
        assert st["sources_count"] == {"sec_edgar": 1, "wikipedia_current_events": 29}

    def test_excluded_source_never_returned(self, tmp_path):
        # a store that DOES contain FRED (like v1) but excludes it at retrieval
        s = TemporalNewsStore(persist_dir=str(tmp_path / "v1like"))
        s.ingest_articles(_corpus())
        assert _count_source(s, "fred") == 14
        s.exclude_sources = frozenset({"fred"})
        arts, st = s.retrieve_with_stats("VIX volatility", "AAPL", "2020-03-01",
                                         "2020-03-14", max_results=100)
        assert not any(a["source"] == "fred" for a in arts)
        assert st["n_excluded"] == 14


def _snapshot(directory):
    out = {}
    for root, _, files in os.walk(directory):
        for f in files:
            p = os.path.join(root, f)
            st = os.stat(p)
            out[os.path.relpath(p, directory)] = (st.st_size, st.st_mtime_ns,
                                                  file_sha256(p))
    return out


@pytest.fixture
def built_store(tmp_path):
    """A small v2-like store, closed, with a manifest and chmod a-w."""
    d = tmp_path / "rag_store_v2"
    code = (
        "import sys, json; sys.path.insert(0, %r);"
        "import tests.test_corpus_v2_store as T;"
        "from sde_causal_generator.news_rag import TemporalNewsStore;"
        "TemporalNewsStore._get_embedder = lambda self: T.HashEmbedder();"
        "s = TemporalNewsStore(persist_dir=%r, exclude_sources=['fred']);"
        "s.ingest_articles(T._corpus()); print(s._collection.count())"
    ) % (str(REPO), str(d))
    out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "71"
    sqlite_path = d / "chroma.sqlite3"
    sha = file_sha256(sqlite_path)
    with open(d / "MANIFEST.json", "w") as fh:
        json.dump({"sqlite_sha256": sha, "excluded_sources": ["fred"],
                   "corpus_tag": f"rag_store_v2@{sha[:16]}"}, fh)
    os.chmod(sqlite_path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    yield d
    os.chmod(sqlite_path, stat.S_IRUSR | stat.S_IWUSR)


class TestReadOnlyMode:

    def test_readonly_never_writes(self, built_store, tmp_path):
        before = _snapshot(built_store)
        rag = NewsRAGManager(persist_dir=str(built_store), cache_mode="readonly",
                             readonly_workdir=str(tmp_path / "ro"))
        assert rag.readonly and rag.corpus_built
        # build_corpus is skipped entirely (no cache dir, no download, no ingest)
        rag.build_corpus("AAPL", "2020-03-01", "2020-03-14",
                         sources=["wikipedia", "fred", "sec_edgar"])
        assert not (built_store / "article_cache").exists()
        with pytest.raises(RuntimeError):
            rag.store.ingest_articles([_art()])
        text, st = rag.get_context_with_stats("AAPL", "2020-03-01", "2020-03-14")
        assert st["n_unique"] == 50 and "fred" not in st["sources_count"]
        assert rag.store.get_stats()["total_articles"] == 71
        rag.close()
        assert _snapshot(built_store) == before
        assert not any((tmp_path / "ro").iterdir())  # private copy removed

    def test_readonly_checks_manifest_hash(self, built_store, tmp_path):
        with pytest.raises(RuntimeError, match="sha256"):
            NewsRAGManager(persist_dir=str(built_store), cache_mode="readonly",
                           expected_sha256="0" * 64,
                           readonly_workdir=str(tmp_path / "ro"))

    def test_readonly_missing_store_is_not_created(self, tmp_path):
        missing = tmp_path / "nope"
        with pytest.raises(FileNotFoundError):
            NewsRAGManager(persist_dir=str(missing), cache_mode="readonly")
        assert not missing.exists()

    def test_readonly_must_be_set_at_construction(self, tmp_path):
        rag = NewsRAGManager(persist_dir=str(tmp_path / "rw"))
        with pytest.raises(ValueError, match="readonly"):
            rag.build_corpus("AAPL", "2020-01-01", "2020-01-31",
                             sources=[], cache_mode="readonly")

    def test_unknown_cache_mode(self, tmp_path):
        with pytest.raises(ValueError):
            NewsRAGManager(persist_dir=str(tmp_path / "x"), cache_mode="ro")
