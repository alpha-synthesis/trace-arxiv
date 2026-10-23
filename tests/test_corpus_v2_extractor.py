# -*- coding: utf-8 -*-
"""Corpus v2 extractor hooks: legacy cache keys, corpus-tagged keys,
precomputed context provider (no Chroma), long-lived litellm HTTP client,
retry logging, and the redaction modes.  No network, no LLM calls."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
from pathlib import Path

import pandas as pd
import pytest

from sde_causal_generator.factor_extractor import (
    LLMFactorExtractor,
    PrecomputedContexts,
    _MARKET_TERMS,
    _MOVE_TERMS,
    _MOVE_TERMS_V1,
    normalize_redaction_mode,
    redact_market_moves,
    sentence_is_redacted,
)

REPO = Path(__file__).resolve().parents[1]

_FAKE_JSON = json.dumps([
    {"name": "fed_rate_cut", "category": "macroeconomic", "direction": 0.6,
     "magnitude": 0.7, "persistence": "medium", "confidence": 0.9,
     "source_articles": [1], "description": "Fed cut [1]."},
    {"name": "market_noise", "category": "market_sentiment", "direction": 0.0,
     "magnitude": 0.3, "persistence": "transient", "confidence": 1.0,
     "source_articles": [], "description": "noise"},
])

CTX = (
    "=== NEWS CONTEXT FOR AAPL (2020-03-08 to 2020-03-22) ===\n"
    "Total articles retrieved: 2\n\n"
    "── MACRO/GEOPOLITICAL CONTEXT ──\n"
    "[1] (2020-03-15) [wikipedia_current_events] World Event: The Fed cuts rates to zero.\n"
    "    World Event: The Fed cuts rates to zero.. The Fed cuts rates to zero.\n\n"
    "=== END NEWS CONTEXT ==="
)


def _ext(tmp_path, **kw):
    base = dict(llm_model="openrouter/openai/gpt-4o-mini", cache_dir=str(tmp_path / "c"),
                temperature=0.05, n_samples=5, max_articles=50)
    base.update(kw)
    return LLMFactorExtractor(**base)


def _df():
    dates = pd.bdate_range("2020-03-02", "2020-03-31")
    return pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "open": 100.0,
                         "high": 101.0, "low": 99.0, "close": 100.5,
                         "volume": 1e6})


class DictProvider:
    corpus_tag = "test-corpus@1"

    def __init__(self, ctx=CTX):
        self.ctx = ctx
        self.calls = []

    def get(self, ticker, start, end):
        self.calls.append((ticker, start, end))
        return self.ctx


# ══════════════════════════════════════════════════════════════════════
# Cache keys
# ══════════════════════════════════════════════════════════════════════


class TestCacheKeys:

    def test_legacy_keys_unchanged(self, tmp_path):
        """Keys of the existing A0 / A1 caches (replay with corpus_tag=None)."""
        w = ("AAPL", "2018-01-07", "2018-01-21")
        assert _ext(tmp_path)._cache_key(*w) == "42110d1f89b044e15399a4fa2fe0bf09"
        a1 = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves=True)
        assert a1._prompt_hash == "578a2b62"
        assert a1._cache_key(*w) == "8f04244314602bc7d161193194fcd1b7"
        nored = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves=False)
        assert nored._cache_key(*w) == "2b027ff9f7d33d8f5aa0977a819df2b5"

    def test_corpus_tag_and_context_in_key(self, tmp_path):
        w = ("AAPL", "2018-01-07", "2018-01-21")
        a = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves="strict",
                 corpus_tag="v2@aaa")
        b = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves="strict",
                 corpus_tag="v2@bbb")
        assert a._cache_key(*w, "ctx") != b._cache_key(*w, "ctx")
        assert a._cache_key(*w, "ctx") != a._cache_key(*w, "ctx2")
        assert a._cache_key(*w, "ctx") == a._cache_key(*w, "ctx")
        with pytest.raises(ValueError):
            a._cache_key(*w)

    def test_strict_mode_has_new_prompt_hash(self, tmp_path):
        a1 = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves=True)
        st = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves="strict")
        mv = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves="moves")
        assert len({a1._prompt_hash, st._prompt_hash, mv._prompt_hash}) == 3


# ══════════════════════════════════════════════════════════════════════
# Context provider
# ══════════════════════════════════════════════════════════════════════


class TestContextProvider:

    def test_provider_used_and_cache_tied_to_context(self, tmp_path, monkeypatch):
        prompts = []
        monkeypatch.setattr(LLMFactorExtractor, "_call_llm",
                            lambda self, p: prompts.append(p) or _FAKE_JSON)
        prov = DictProvider()
        ext = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves="strict",
                   context_provider=prov)
        assert ext.corpus_tag == "test-corpus@1"
        a = ext.extract_window(_df(), "AAPL", "2020-03-08", "2020-03-22")
        assert prov.calls == [("AAPL", "2020-03-08", "2020-03-22")]
        assert len(prompts) == 5 and "The Fed cuts rates to zero" in prompts[0]
        assert {f.name for f in a.factors} == {"fed_rate_cut", "market_noise"}

        # same corpus + same context → cache hit, no LLM call
        ext.extract_window(_df(), "AAPL", "2020-03-08", "2020-03-22")
        assert len(prompts) == 5
        # different context (another corpus revision) → never reuses labels
        ext2 = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves="strict",
                    context_provider=DictProvider(CTX.replace("zero", "0.25%")))
        ext2.extract_window(_df(), "AAPL", "2020-03-08", "2020-03-22")
        assert len(prompts) == 10

    def test_provider_requires_corpus_tag(self, tmp_path):
        class NoTag:
            def get(self, t, s, e):
                return CTX
        with pytest.raises(ValueError, match="corpus_tag"):
            _ext(tmp_path, context_provider=NoTag())
        _ext(tmp_path, context_provider=NoTag(), corpus_tag="x")  # ok

    def test_provider_and_rag_are_exclusive(self, tmp_path):
        with pytest.raises(ValueError):
            _ext(tmp_path, context_provider=DictProvider(), news_rag=object())

    def test_strict_redaction_applied_to_provider_context(self, tmp_path, monkeypatch):
        prompts = []
        monkeypatch.setattr(LLMFactorExtractor, "_call_llm",
                            lambda self, p: prompts.append(p) or _FAKE_JSON)
        ctx = CTX.replace("The Fed cuts rates to zero.. The Fed cuts rates to zero.",
                          "The Fed cuts rates to zero. Stocks plunge on Monday.")
        ext = _ext(tmp_path, prompt_mode="price_blind", redact_market_moves="strict",
                   context_provider=DictProvider(ctx))
        ext.extract_window(_df(), "AAPL", "2020-03-08", "2020-03-22")
        assert "Stocks plunge" not in prompts[0]
        assert "The Fed cuts rates to zero." in prompts[0]


def _write_contexts(d: Path, tag="rag_store_v2@abc"):
    d.mkdir(parents=True, exist_ok=True)
    (d / "INDEX.json").write_text(json.dumps({"corpus_tag": tag}))
    (d / "AAPL.json").write_text(json.dumps({
        "ticker": "AAPL", "corpus_tag": tag,
        "windows": [{"window_start": "2020-03-08", "window_end": "2020-03-22",
                     "context": CTX, "n_retrieved": 2, "n_unique": 1,
                     "sources_count": {"wikipedia_current_events": 1},
                     "n_sentences_dropped": 0, "n_sentences_kept": 3}]}))


class TestPrecomputedContexts:

    def test_get_and_missing_window(self, tmp_path):
        _write_contexts(tmp_path / "ctx")
        pc = PrecomputedContexts(str(tmp_path / "ctx"))
        assert pc.corpus_tag == "rag_store_v2@abc"
        assert pc.get("AAPL", "2020-03-08", "2020-03-22") == CTX
        with pytest.raises(KeyError):
            pc.get("AAPL", "2020-03-22", "2020-04-05")
        with pytest.raises(KeyError):
            pc.get("MSFT", "2020-03-08", "2020-03-22")

    def test_corpus_tag_mismatch(self, tmp_path):
        _write_contexts(tmp_path / "ctx")
        with pytest.raises(ValueError):
            PrecomputedContexts(str(tmp_path / "ctx"), expected_corpus_tag="other")

    def test_llm_process_never_imports_chroma(self, tmp_path):
        """A fresh process: extractor + PrecomputedContexts, fake LLM."""
        _write_contexts(tmp_path / "ctx")
        code = f"""
import sys, json
import pandas as pd
from sde_causal_generator.factor_extractor import LLMFactorExtractor, PrecomputedContexts
LLMFactorExtractor._call_llm = lambda self, p: {_FAKE_JSON!r}
pc = PrecomputedContexts({str(tmp_path / 'ctx')!r})
ext = LLMFactorExtractor(llm_model="openrouter/openai/gpt-4o-mini",
                         cache_dir={str(tmp_path / 'c')!r}, n_samples=2,
                         prompt_mode="price_blind", redact_market_moves="strict",
                         context_provider=pc)
dates = pd.bdate_range("2020-03-02", "2020-03-31")
df = pd.DataFrame({{"date": dates.strftime("%Y-%m-%d"), "open": 1.0, "high": 1.0,
                   "low": 1.0, "close": 1.0, "volume": 1.0}})
a = ext.extract_window(df, "AAPL", "2020-03-08", "2020-03-22")
ext.close()
bad = sorted(m for m in sys.modules if m.split(".")[0] in ("chromadb", "hnswlib", "sentence_transformers"))
print(json.dumps({{"bad": bad, "n": len(a.factors)}}))
"""
        out = subprocess.run([sys.executable, "-c", code], cwd=str(REPO),
                             capture_output=True, text=True, check=True)
        res = json.loads(out.stdout.strip().splitlines()[-1])
        assert res == {"bad": [], "n": 2}


# ══════════════════════════════════════════════════════════════════════
# Long-lived HTTP client and retry logging
# ══════════════════════════════════════════════════════════════════════

litellm = pytest.importorskip("litellm")


class _Resp:
    class _C:
        class message:
            content = _FAKE_JSON
    choices = [_C()]


class TestHttpClient:

    def test_one_client_per_instance_passed_to_litellm(self, tmp_path, monkeypatch):
        from litellm.llms.custom_httpx.http_handler import HTTPHandler

        seen = []
        monkeypatch.setattr(litellm, "completion",
                            lambda **kw: seen.append(kw) or _Resp())
        ext = _ext(tmp_path)
        threads = [threading.Thread(target=ext._call_llm, args=("p",))
                   for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        clients = {id(kw["client"]) for kw in seen}
        assert len(seen) == 8 and len(clients) == 1
        client = seen[0]["client"]
        assert isinstance(client, HTTPHandler)
        assert seen[0]["timeout"] == 600.0
        assert ext.n_llm_calls == 8
        assert not client.client.is_closed
        # another extractor gets its own client
        ext_b = _ext(tmp_path)
        ext_b._call_llm("p")
        assert seen[-1]["client"] is not client
        ext.close()
        assert client.client.is_closed
        with pytest.raises(RuntimeError):
            ext._call_llm("p")
        ext_b.close()

    def test_non_openrouter_model_gets_no_client(self, tmp_path, monkeypatch):
        seen = []
        monkeypatch.setattr(litellm, "completion",
                            lambda **kw: seen.append(kw) or _Resp())
        ext = _ext(tmp_path, llm_model="openai/gpt-4o-mini")
        ext._call_llm("p")
        assert "client" not in seen[0]

    def test_retry_logs_repr(self, tmp_path, monkeypatch, capsys):
        calls = {"n": 0}

        err = litellm.exceptions.RateLimitError(
            message="slow down", llm_provider="openrouter", model="m")

        def flaky(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise err
            return _Resp()

        monkeypatch.setattr(litellm, "completion", flaky)
        import sde_causal_generator.factor_extractor as fe
        monkeypatch.setattr(fe.time, "sleep", lambda s: None)
        log = tmp_path / "calls.jsonl"
        ext = _ext(tmp_path, call_log_path=str(log))
        assert ext._call_llm("p") == _FAKE_JSON
        out = capsys.readouterr().out
        assert repr(err) in out and "[RateLimitError]" in out
        recs = [json.loads(x) for x in log.read_text().splitlines()]
        assert [r["status"] for r in recs] == ["sent", "error", "sent"]
        assert recs[1]["error"] == repr(err)
        assert recs[1]["error_type"] == "RateLimitError"
        assert ext.n_llm_calls == 2 and ext.n_llm_failures == 1
        ext.close()


# ══════════════════════════════════════════════════════════════════════
# Redaction modes
# ══════════════════════════════════════════════════════════════════════


class TestRedaction:

    def test_mode_mapping(self):
        assert normalize_redaction_mode(False) == "off"
        assert normalize_redaction_mode(None) == "off"
        assert normalize_redaction_mode(True) == "legacy"
        assert normalize_redaction_mode("strict") == "strict"
        assert normalize_redaction_mode("STRICT") == "strict"
        with pytest.raises(ValueError):
            normalize_redaction_mode("aggressive")

    def test_strict_drops_any_market_term(self):
        s = "Apple shareholders approve the new board."
        assert not sentence_is_redacted("Apple and the board meet.", "strict")
        assert sentence_is_redacted("Apple shares were halted.", "strict")
        assert sentence_is_redacted("The Dow Jones index is discussed.", "strict")
        assert sentence_is_redacted("Shares of Boeing were halted.", "strict")
        assert not sentence_is_redacted("The Dow Jones index is discussed.", "legacy")
        # word boundary: "shareholders" is not "shares"
        assert not sentence_is_redacted(s, "strict")

    def test_new_move_words(self):
        for s in ("The S&P 500 increased 3% on Friday.",
                  "Wall Street had its worst day since 1987.",
                  "Stocks had their best day in a decade.",
                  "The Nasdaq decreased sharply."):
            assert _MOVE_TERMS.search(s), s
            assert sentence_is_redacted(s, "moves"), s
        assert not _MOVE_TERMS_V1.search("Wall Street had its worst day since 1987.")
        assert not sentence_is_redacted("Wall Street had its worst day since 1987.",
                                        "legacy")

    def test_legacy_function_default_unchanged(self):
        ctx = "    The Dow fell 500 points. Congress passes a bill."
        red, d, k = redact_market_moves(ctx)
        assert (d, k) == (1, 1) and "Congress passes a bill." in red

    def test_strict_is_idempotent_and_keeps_headers(self):
        ctx = ("=== NEWS CONTEXT FOR AAPL (a to b) ===\nTotal articles retrieved: 2\n\n"
               "── MACRO/GEOPOLITICAL CONTEXT ──\n"
               "[1] (2020-03-16) [wikipedia_current_events] World Event: Stocks plunge.\n"
               "    World Event: Stocks plunge.. Stock markets fall. The Fed acts.\n"
               "=== END NEWS CONTEXT ===")
        once, d1, k1 = redact_market_moves(ctx, "strict")
        twice, d2, k2 = redact_market_moves(once, "strict")
        assert once == twice and d2 == 0
        assert "Stock" not in once and "The Fed acts." in once
        assert once.splitlines()[0] == ctx.splitlines()[0]
        assert "[redacted]" in once

    @pytest.mark.parametrize("s", [
        # round-1 audit misses of the v1 term list
        "All three major United States trading indexes fall 7% during early trading.",
        "Spain's IBEX 35 falls 14%, its worst drop in history in a single day.",
        "At one point during the day, the benchmark index loses a record 1,579 points.",
        "This follows benchmark indices falling sharply in Europe after steep declines across Asia.",
        "Asian markets fall substantially and European markets open with huge losses.",
        "Markets are on pace for their worst week since the 2008 financial crisis.",
        "Uber's market cap drops from over $80 billion to about $72 billion.",
        "This increased the company's worth to US$1.88 trillion, making it the most "
        "valuable publicly-traded company in the world.",
        "becomes the first public company to be worth US$1 trillion.",
        "Trading was halted multiple times due to the climb and the crash.",
        "the S&P fell 3% in late trading.",
    ])
    def test_strict_v2_catches_round1_misses(self, s):
        assert sentence_is_redacted(s, "strict")

    @pytest.mark.parametrize("s", [
        "(The Wall Street Journal)",
        "(Nikkei Asia)",
        "S&P Global Ratings upgrades Argentina's long term sovereign-credit rating.",
        "Intelligence officials in the United States share evidence that North Korea has acted.",
        "Her party obtains its highest voter share in decades.",
        "The Blackstone Group, a multinational private equity and asset management firm, buys Geo-Young.",
        "The Federal Reserve raises its benchmark interest rate by 0.75 percentage point.",
        "The Food and Agriculture Organisation's food price index increases by 13% in March.",
        "The United States imposes a 25% tariff on goods worth $50 billion.",
    ])
    def test_strict_v2_keeps_round1_false_positives(self, s):
        assert not sentence_is_redacted(s, "strict")

    def test_v1_patterns_frozen(self):
        from sde_causal_generator.factor_extractor import _MARKET_TERMS_V1
        assert _MARKET_TERMS.pattern != _MARKET_TERMS_V1.pattern
        # legacy mode still uses v1: the WSJ citation is not a v1 "move"
        assert not sentence_is_redacted("(The Wall Street Journal)", "legacy")
        assert sentence_is_redacted("Wall Street falls 3%.", "legacy")


def test_window_pairs_match_extract_all_windows(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(LLMFactorExtractor, "extract_window",
                        lambda self, df, t, s, e: seen.append((s, e)) or None)
    dates = pd.bdate_range("2019-01-02", "2019-12-31")
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d")})
    ext = _ext(tmp_path)
    ext.extract_all_windows(df, "AAPL", window_sizes=["2W"], max_workers=1)
    assert sorted(seen) == sorted(LLMFactorExtractor.window_pairs(df["date"], ["2W"]))
    assert len(seen) == 25
