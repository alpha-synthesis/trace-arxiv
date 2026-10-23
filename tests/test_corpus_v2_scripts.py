# -*- coding: utf-8 -*-
"""Corpus v2 scripts: Wikipedia revision pinning (mocked API) and the
store loader (FRED exclusion, recomputed ids, de-duplication)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from sde_causal_generator.data_structures import NewsArticle
from sde_causal_generator.news_rag import WikipediaCurrentEventsIngestor as W

REPO = Path(__file__).resolve().parents[1]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ══════════════════════════════════════════════════════════════════════
# Revision pinning
# ══════════════════════════════════════════════════════════════════════


class TestPinnedFetch:

    def test_request_is_pinned(self, monkeypatch):
        seen = {}

        def fake(params, max_retries=3, request_delay=None):
            seen.update(params)
            return {"query": {"pages": [{
                "pageid": 1, "title": params["titles"],
                "revisions": [{"revid": 11, "parentid": 10,
                               "timestamp": "2020-03-20T10:00:00Z", "sha1": "s",
                               "slots": {"main": {"content":
                                   "* The [[Federal Reserve]] cuts its rate to near zero today.\n"}}}],
            }]}}

        monkeypatch.setattr(W, "_api_get", staticmethod(fake))
        rec = W.fetch_day_revision("2020-03-15", "2020-03-22T23:59:59Z")
        assert seen["rvstart"] == "2020-03-22T23:59:59Z"
        assert seen["rvdir"] == "older" and seen["rvlimit"] == 1
        assert "content" in seen["rvprop"] and seen["titles"] == \
            "Portal:Current events/2020 March 15"
        assert rec["status"] == "ok" and rec["revid"] == 11
        arts = W._parse_day_wikitext(rec["wikitext"], rec["date"])
        assert len(arts) == 1 and "Federal Reserve" in arts[0].content

    @pytest.mark.parametrize("page,status", [
        ({"title": "x", "missing": True}, "missing"),
        ({"title": "x", "pageid": 3, "revisions": []}, "no_revision"),
        ({"title": "x", "pageid": 3, "revisions": [{"revid": 1, "slots": {"main": {}}}]},
         "hidden"),
    ])
    def test_statuses(self, monkeypatch, page, status):
        monkeypatch.setattr(W, "_api_get", staticmethod(
            lambda params, max_retries=3, request_delay=None: {"query": {"pages": [page]}}))
        assert W.fetch_day_revision("2020-03-15", "2020-03-22T23:59:59Z")["status"] == status

    def test_day_page_title(self):
        assert W.day_page_title("2018-01-01") == "Portal:Current events/2018 January 1"


class TestPinRule:

    def setup_method(self):
        self.mod = _load_script("pin_wikipedia_revisions")

    def test_shared_windows_use_earliest_window_end(self):
        wins = [("2020-01-05", "2020-01-19"), ("2020-01-19", "2020-02-02")]
        days = list(pd.date_range("2020-01-03", "2020-02-03"))
        pins, rule, n_out, shared = self.mod.pin_dates(days, {"AAPL": wins, "MSFT": wins})
        assert shared
        assert pins["2020-01-05"] == "2020-01-19"
        assert pins["2020-01-18"] == "2020-01-19"
        assert pins["2020-01-19"] == "2020-01-19"   # boundary: earlier end
        assert pins["2020-01-20"] == "2020-02-02"
        assert pins["2020-01-03"] == "2020-01-16"   # outside: day + 13
        assert pins["2020-02-03"] == "2020-02-16"
        assert n_out == 3
        # no pin is ever before the day itself
        assert all(p >= d for d, p in pins.items())

    def test_differing_windows_use_day_plus_13(self):
        days = list(pd.date_range("2020-01-05", "2020-01-10"))
        pins, rule, _, shared = self.mod.pin_dates(
            days, {"AAPL": [("2020-01-05", "2020-01-19")],
                   "MSFT": [("2020-01-06", "2020-01-20")]})
        assert not shared and "day+13" in rule
        assert pins["2020-01-05"] == "2020-01-18"


# ══════════════════════════════════════════════════════════════════════
# Store loader
# ══════════════════════════════════════════════════════════════════════


def _write(path: Path, arts):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([a.to_dict() | {"article_id": "stale-cached-id"}
                                for a in arts]))


@pytest.fixture
def fake_root(tmp_path):
    rng = "2018-01-02_2023-12-29"
    wiki = [NewsArticle(title=f"World Event: event {i}", content=f"event {i} happens somewhere",
                        source="wikipedia_current_events", published_date="2020-01-0%d" % (1 + i % 3))
            for i in range(6)]
    # same event, different case/whitespace in title and content
    wiki.append(NewsArticle(title="World Event: EVENT 0", content="  Event 0   happens somewhere ",
                            source="wikipedia_current_events", published_date="2020-01-01"))
    _write(tmp_path / "cache/rag_store_v2/article_cache_pinned/"
           "wikipedia_pinned_2018-01-01_2023-12-31.json", wiki)
    filing = dict(title="SEC 8-K: Acme", content="Acme filed 8-K with the SEC on 2020-01-02.",
                  source="sec_edgar", published_date="2020-01-02")
    _write(tmp_path / f"cache/rag_store/article_cache/sec_edgar_AAPL_{rng}.json",
           [NewsArticle(**filing, tickers_mentioned=["AAPL"])])
    _write(tmp_path / f"cache/rag_store/article_cache/sec_edgar_MSFT_{rng}.json",
           [NewsArticle(**filing, tickers_mentioned=["MSFT"])])
    _write(tmp_path / f"cache/rag_store/article_cache/fred_AAPL_{rng}.json",
           [NewsArticle(title="FRED Release: VIX", content="VIX at 20", source="fred",
                        published_date="2020-01-02")])
    # a FRED-sourced record smuggled into a SEC file is still excluded
    _write(tmp_path / f"cache/rag_store/article_cache/sec_edgar_KO_{rng}.json",
           [NewsArticle(title="FRED Release: DGS10", content="DGS10 at 1.9", source="fred",
                        published_date="2020-01-02")])
    return tmp_path


class TestCorpusLoader:

    def test_inputs_never_include_fred_files(self, fake_root):
        mod = _load_script("build_corpus_v2")
        wiki, sec = mod.input_files(fake_root, ["AAPL", "MSFT", "KO", "MCD"],
                                    "2018-01-02_2023-12-29",
                                    "wikipedia_pinned_2018-01-01_2023-12-31.json")
        names = [p.name for p in wiki + sec]
        assert len(wiki) == 1 and len(sec) == 3
        assert not any(n.startswith("fred") for n in names)

    def test_load_dedupes_and_excludes(self, fake_root):
        mod = _load_script("build_corpus_v2")
        wiki, sec = mod.input_files(fake_root, ["AAPL", "MSFT", "KO"],
                                    "2018-01-02_2023-12-29",
                                    "wikipedia_pinned_2018-01-01_2023-12-31.json")
        arts, st = mod.load_corpus(wiki + sec)
        assert st["n_excluded"] == 1
        assert not any(a.source == "fred" for a in arts)
        assert st["n_unique_by_source"] == {"wikipedia_current_events": 6, "sec_edgar": 1}
        assert st["n_dup_date_text"] == 1 and st["n_dup_id"] == 1
        sec_art = [a for a in arts if a.source == "sec_edgar"][0]
        assert sec_art.tickers_mentioned == ["AAPL", "MSFT"]
        assert all(a.article_id != "stale-cached-id" for a in arts)
        assert all(a.article_id == a.content_id() for a in arts)
