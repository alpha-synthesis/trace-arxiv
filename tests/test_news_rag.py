# -*- coding: utf-8 -*-
"""Tests for news_rag.py — WikipediaCurrentEventsIngestor parsing, NewsArticle.

B6 from the gap analysis: Tests the RAG module that is the foundation
of the data pipeline (Etapa 1). Uses mocks — no network access needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from sde_causal_generator.data_structures import NewsArticle


# ══════════════════════════════════════════════════════════════════════
# NewsArticle data structure
# ══════════════════════════════════════════════════════════════════════


class TestNewsArticle:

    def test_roundtrip_dict(self):
        """NewsArticle should survive to_dict / from_dict roundtrip."""
        article = NewsArticle(
            title="Fed Raises Rates",
            content="The Federal Reserve raised interest rates by 0.25%",
            source="wikipedia_current_events",
            published_date="2023-07-26",
            tickers_mentioned=["AAPL", "GOOG"],
            categories=["macroeconomic"],
            url="https://example.com/article",
        )
        d = article.to_dict()
        a2 = NewsArticle.from_dict(d)
        assert a2.title == article.title
        assert a2.content == article.content
        assert a2.source == article.source
        assert a2.published_date == article.published_date
        assert a2.tickers_mentioned == ["AAPL", "GOOG"]
        assert a2.categories == ["macroeconomic"]

    def test_auto_id_generation(self):
        """article_id should be auto-generated if not provided."""
        a = NewsArticle(
            title="Test", content="Test content",
            source="test", published_date="2023-01-01",
        )
        assert len(a.article_id) > 0

    def test_unique_ids(self):
        """Different articles should get different IDs."""
        a1 = NewsArticle(
            title="Article A", content="Content A",
            source="test", published_date="2023-01-01",
        )
        a2 = NewsArticle(
            title="Article B", content="Content B",
            source="test", published_date="2023-01-02",
        )
        assert a1.article_id != a2.article_id

    def test_explicit_id_preserved(self):
        """Explicitly set article_id should be preserved."""
        a = NewsArticle(
            title="Test", content="Test",
            source="test", published_date="2023-01-01",
            article_id="my_custom_id_123",
        )
        assert a.article_id == "my_custom_id_123"

    def test_empty_defaults(self):
        """Optional fields should default correctly."""
        a = NewsArticle(
            title="Test", content="Test",
            source="test", published_date="2023-01-01",
        )
        assert a.tickers_mentioned == []
        assert a.categories == []
        assert a.url == ""


# ══════════════════════════════════════════════════════════════════════
# WikipediaCurrentEventsIngestor — wikitext parsing
# ══════════════════════════════════════════════════════════════════════


class TestWikipediaParser:

    def test_parse_day_wikitext_basic(self):
        """Should extract events from wikitext bullet points."""
        from sde_causal_generator.news_rag import (
            WikipediaCurrentEventsIngestor,
        )

        wikitext = (
            "* The [[Federal Reserve]] raises interest rates "
            "for the first time in four years.\n"
            "* [[Apple Inc.]] announces record quarterly earnings "
            "of $120 billion in revenue.\n"
            "** The results exceeded analyst expectations.\n"
        )
        articles = WikipediaCurrentEventsIngestor._parse_day_wikitext(
            wikitext, "2023-07-26"
        )
        assert len(articles) >= 2
        assert all(isinstance(a, NewsArticle) for a in articles)
        assert all(a.published_date == "2023-07-26" for a in articles)
        assert all(a.source == "wikipedia_current_events" for a in articles)

    def test_parse_removes_wikilinks(self):
        """Wikilinks [[Target|Display]] should be cleaned to Display."""
        from sde_causal_generator.news_rag import (
            WikipediaCurrentEventsIngestor,
        )

        wikitext = (
            "* The [[Federal Reserve|Fed]] raises the rate significantly "
            "across multiple sectors.\n"
        )
        articles = WikipediaCurrentEventsIngestor._parse_day_wikitext(
            wikitext, "2023-01-01"
        )
        assert len(articles) >= 1
        # Should contain "Fed", not "[[Federal Reserve|Fed]]"
        assert "[[" not in articles[0].content
        assert "]]" not in articles[0].content

    def test_parse_empty_wikitext(self):
        """Empty wikitext should return no articles."""
        from sde_causal_generator.news_rag import (
            WikipediaCurrentEventsIngestor,
        )

        articles = WikipediaCurrentEventsIngestor._parse_day_wikitext("", "2023-01-01")
        assert articles == []

    def test_parse_skips_short_lines(self):
        """Very short bullet points (< 25 chars) should be skipped."""
        from sde_causal_generator.news_rag import (
            WikipediaCurrentEventsIngestor,
        )

        wikitext = "* Short.\n* Also short line here.\n"
        articles = WikipediaCurrentEventsIngestor._parse_day_wikitext(
            wikitext, "2023-01-01"
        )
        # Both lines are short after cleanup (< 25 chars)
        assert len(articles) == 0

    def test_parse_no_bullets_returns_empty(self):
        """Text without bullets should return empty."""
        from sde_causal_generator.news_rag import (
            WikipediaCurrentEventsIngestor,
        )

        wikitext = "This is a regular paragraph without any bullet points."
        articles = WikipediaCurrentEventsIngestor._parse_day_wikitext(
            wikitext, "2023-01-01"
        )
        assert articles == []
