"""Context v2.1 (2026-10-01): item-level redaction, header filter, company rank.

Pins the fixes from the review of the v2 contexts. The v2 behaviour (strict,
sentence-level, ``[redacted]`` marker) stays reproducible and is covered by
tests/test_corpus_v2_*.py.
"""
import importlib.util
import json
from pathlib import Path

import pytest

from sde_causal_generator.factor_extractor import (
    LLMFactorExtractor, PrecomputedContexts, sentence_is_redacted)

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("precompute_contexts", ROOT / "scripts" / "precompute_contexts.py")
pc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pc)
import socket  # noqa: E402
socket.socket.connect = pc._real_connect   # the script installs a no-network guard


@pytest.mark.parametrize("s", [
    "They all closed over 9% down. (Business Insider)",
    "Amazon becomes the second trillion-dollar company in the United States.",
    "Entities such as Gazprom and Lukoil have lost almost all of their value.",
    "Black Thursday; socio-economic impact of the COVID-19 pandemic",
])
def test_strict2_catches_outcome_phrases_strict_missed(s):
    assert not sentence_is_redacted(s, "strict")
    assert sentence_is_redacted(s, "strict2")


@pytest.mark.parametrize("s", [
    "The Federal Reserve raises its benchmark interest rate by 0.75 percentage points.",
    "Apple reports quarterly earnings that beat analyst estimates. (Reuters)",
    "Brent crude rises 19.5% after OPEC+ agrees to production cuts. (Reuters)",
])
def test_strict2_keeps_causes(s):
    assert not sentence_is_redacted(s, "strict2")


def _wiki(text, title=None):
    return {"title": "World Event: " + (title or text[:80]), "source": "wikipedia_current_events",
            "content": f"World Event: {(title or text[:80])}. {text}", "date": "2020-03-12"}


def test_item_filter_drops_headers_and_whole_redacted_items():
    f = pc.make_item_filter("strict2")
    assert f(_wiki("Coronavirus recession, COVID-19 pandemic in the United States")) == "topic_header"
    # the continuation sentence has no market term; the whole item goes
    item = _wiki("All three major U.S. stock indexes fall 7%. They all closed over 9% down. (Business Insider)")
    assert f(item) == "redacted"
    assert len(f.dropped) == 1
    keep = _wiki("Lebanon's banking association rejects a plan proposed by the government. (Reuters)")
    assert f(keep) is None
    long_event = _wiki("The government of Finland resigns after a failure to deliver on healthcare reform, "
                       "a month ahead of the general election. The coalition had held office since 2015.")
    assert f(long_event) is None
    link_list = _wiki("Russian interference in the 2016 United States elections, Social media in the United "
                      "States presidential election, 2016, Russian interference in the 2018 United States elections")
    assert f(link_list) == "topic_header"
    assert f(_wiki("Syrian attacks on Idlib kill 12 civilians. (SOHR)].")) is None


@pytest.mark.parametrize("t,yes,no", [
    ("AMZN", "Amazon announces it will hire 100,000 workers.", "Fires in the Amazon rainforest reach a record."),
    ("V", "Visa and Mastercard suspend operations in Russia.", "Canada lifts visa requirements for Mexicans."),
    ("TRV", "Travelers Companies reports a catastrophe loss.", "Travelers are stranded at airports."),
    ("CVX", "Chevron agrees to buy Noble Energy.", "The Chevron doctrine is debated."),
])
def test_company_patterns(t, yes, no):
    pref = pc.make_prefer(t)
    assert pref(_wiki(yes)) and not pref(_wiki(no))


def test_company_table_covers_the_panel():
    import yaml
    tickers = yaml.safe_load(open(ROOT / "configs" / "r2_datasign.yaml"))["tickers"]
    assert set(tickers) <= set(pc.COMPANY)


def test_provider_mode_mismatch_is_refused(tmp_path):
    (tmp_path / "INDEX.json").write_text(json.dumps({"corpus_tag": "x@1", "redaction_mode": "strict"}))
    prov = PrecomputedContexts(str(tmp_path))
    with pytest.raises(ValueError, match="redacted with"):
        LLMFactorExtractor(cache_dir=str(tmp_path / "c"), prompt_mode="price_blind",
                           redact_market_moves="strict2", context_provider=prov)


def test_provider_contexts_are_not_redacted_twice(tmp_path):
    ctx = "=== NEWS CONTEXT ===\n[1] (2020-03-12) [w] x\n    stocks fall 7%.\n=== END NEWS CONTEXT ==="
    (tmp_path / "INDEX.json").write_text(json.dumps({"corpus_tag": "x@1", "redaction_mode": "strict2"}))
    (tmp_path / "AAA.json").write_text(json.dumps({"corpus_tag": "x@1", "windows": [
        {"window_start": "2020-03-01", "window_end": "2020-03-15", "context": ctx,
         "n_items_redacted": 3, "n_sentences_kept": 40}]}))
    ex = LLMFactorExtractor(cache_dir=str(tmp_path / "c"), prompt_mode="price_blind",
                            redact_market_moves="strict2", context_provider=PrecomputedContexts(str(tmp_path)))
    out, use_rag, red = ex._build_context("AAA", "2020-03-01", "2020-03-15")
    assert out == ctx and use_rag and red == (3, 40)


# ── review of the first v2.1 build (2026-10-01) ─────────────────────────

def test_source_tag_neither_redacts_nor_flags():
    f = pc.make_item_filter("strict2")
    assert f(_wiki("The COVID-19 death toll in the U.S. passes 100,000. (Nasdaq)")) is None
    assert not pc.make_prefer("DIS")(_wiki("The Lakers beat the Heat to win the NBA Finals. (ESPN)"))
    assert pc.make_prefer("DIS")(_wiki("Disney announces 28,000 layoffs at its theme parks. (Reuters)"))


def test_apple_false_positives():
    pref = pc.make_prefer("AAPL")
    assert not pref(_wiki("The Lost Apple Project rediscovers ten apple cultivars. (AP)"))
    assert not pref(_wiki("The Apple Fire burns 20,000 acres in California. (LA Times)"))
    assert pref(_wiki("Apple is fined 25 million euros by France. (Reuters)"))


def test_strict2_net_worth():
    assert sentence_is_redacted("Elon Musk overtakes Jeff Bezos to become the richest person in the world.", "strict2")
    assert sentence_is_redacted("The fine amounts to less than 0.003% of Apple Inc.'s current worth.", "strict2")


def test_compact_format_shows_the_checked_text():
    from sde_causal_generator.news_rag import NewsRAGManager
    arts = [dict(_wiki("Lebanon's banking association rejects a government plan. (Reuters)"),
                 relevance=1.0, ticker_specific=False)]
    ctx = NewsRAGManager._format_context("AAA", "2020-01-01", "2020-01-14", arts, 500, compact=True)
    assert "[1] (2020-03-12) Lebanon's banking association rejects a government plan." in ctx
    assert "World Event" not in ctx and "(Reuters)" not in ctx and "wikipedia" not in ctx


@pytest.mark.parametrize("t,no", [
    ("AAPL", "Police in Hong Kong arrest five executives of the Apple Daily newspaper."),
    ("AMZN", "Fires in the Amazon Forest reach a record. Norway freezes the Amazon Fund."),
    ("MSFT", "The Microsoft Theater hosts the Emmy Awards."),
    ("KO", "Russian forces shell the Avdiivka Coke Plant."),
    ("CRM", "A fire alarm empties the Salesforce Tower."),
])
def test_company_false_positives_review2(t, no):
    assert not pc.make_prefer(t)(_wiki(no))
