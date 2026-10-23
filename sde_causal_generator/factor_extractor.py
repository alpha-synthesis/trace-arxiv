# -*- coding: utf-8 -*-
"""
LLM-based Causal Factor Extraction.

Extracts market-moving factors from historical price data using an LLM
grounded by a temporal news RAG, producing structured factor analyses
and a binary factor-presence matrix.

Prompt Engineering
------------------
Two prompt variants are used:

* **RAG-aware** (preferred): the LLM receives only contemporaneous news
  articles and must anchor its analysis in them — mitigates hindsight
  bias and narrative fallacy.
* **Fallback**: used when no news context is available.  The LLM is
  instructed to avoid future knowledge, but has no enforcement.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .data_structures import (
    FACTOR_CATEGORIES,
    CausalFactor,
    WindowAnalysis,
)

if TYPE_CHECKING:  # typing only: the extractor never imports Chroma itself
    from .news_rag import NewsRAGManager


# ══════════════════════════════════════════════════════════════════════
# Prompt templates
# ══════════════════════════════════════════════════════════════════════

_RAG_PROMPT = """\
You are a senior financial market analyst performing a CONTEMPORANEOUS
causal attribution analysis.  Your goal is to identify the specific
factors that CAUSED price movements — not factors that merely coincided.

ASSET: {ticker}
PERIOD: {start_date} to {end_date}

PRICE DATA:
- Opening price: {open_price}
- Closing price: {close_price}
- High: {high_price}
- Low: {low_price}
- Price change: {price_change_pct:.2f}%
- Average daily volume: {avg_volume}
{intra_window_stats}

{news_context}

ANALYSIS METHODOLOGY — Think step-by-step:
1. First, identify the DOMINANT REGIME during this window (bull/bear/sideways).
2. Examine the intra-window statistics: high volatility, drawdowns, or weekly
   reversals suggest multiple competing factors, not a single driver.
3. For each candidate factor, ask: "Would the price have moved THIS WAY
   and THIS MUCH without this factor?"  If not, it is causal.
4. Separate structural factors (e.g., Fed policy) from transient catalysts
   (e.g., a single earnings miss).  Mark persistence accordingly.
5. Attribute unexplained variance honestly to "market_noise".

CRITICAL RULES:
1. Base your analysis EXCLUSIVELY on the news articles provided above.
2. If no relevant news explains the price movement, attribute it to
   "market_noise" with high magnitude.
3. Do NOT reference any events that are NOT in the provided news context.
4. Do NOT use knowledge of what happened AFTER {end_date}.
5. For each factor, cite the article number [N] that supports it.
6. Keep magnitude and confidence SEPARATE: magnitude = how much market
   impact; confidence = how certain you are of the causal link.
7. NAMING RULE: Factor names must be SPECIFIC CAUSAL EVENTS, not generic
   descriptions of indicator movements.  BAD: "rising_treasury_yields",
   "declining_vix", "stable_unemployment_rate".  GOOD: "fed_tightening_cycle_2022",
   "post_covid_vix_normalization", "labor_market_recovery_q3".
   Always tie the name to the CAUSE or EVENT, not the indicator reading.
   If yields rose because of a specific event, name the event, not the yield move.

FEW-SHOT EXAMPLES (for format reference only — do NOT copy these factors):

Example 1 — Bearish window with clear cause:
[
  {{"name": "fed_rate_hike_75bps", "category": "macroeconomic",
    "direction": -0.85, "magnitude": 0.55, "persistence": "medium",
    "confidence": 0.95, "source_articles": [2, 5],
    "description": "Fed raised rates by 75bps [2], markets sold off as tighter monetary conditions reduced equity valuations [5]."}},
  {{"name": "tech_earnings_miss", "category": "company_specific",
    "direction": -0.6, "magnitude": 0.25, "persistence": "transient",
    "confidence": 0.80, "source_articles": [7],
    "description": "Major tech companies missed revenue estimates [7], adding sector-wide selling pressure."}},
  {{"name": "market_noise", "category": "market_sentiment",
    "direction": 0.0, "magnitude": 0.20, "persistence": "transient",
    "confidence": 1.0, "source_articles": [],
    "description": "Residual variance not attributable to identified factors."}}
]

Example 2 — Sideways window with mixed signals:
[
  {{"name": "stimulus_package", "category": "macroeconomic",
    "direction": 0.7, "magnitude": 0.30, "persistence": "long",
    "confidence": 0.85, "source_articles": [1, 3],
    "description": "Congress passed $1.9T fiscal stimulus [1], supporting consumer spending expectations [3]."}},
  {{"name": "supply_chain_disruption", "category": "sector_specific",
    "direction": -0.5, "magnitude": 0.25, "persistence": "medium",
    "confidence": 0.70, "source_articles": [4],
    "description": "Global shipping delays raised input costs [4], offsetting positive fiscal sentiment."}},
  {{"name": "market_noise", "category": "market_sentiment",
    "direction": 0.0, "magnitude": 0.45, "persistence": "transient",
    "confidence": 1.0, "source_articles": [],
    "description": "Large unexplained component — price was range-bound with no dominant driver."}}
]

NOW RESPOND for {ticker} ({start_date} to {end_date}).

Respond with a JSON array of factors.  Each factor must have:
- "name": short snake_case identifier (e.g., "fed_rate_hike")
- "category": one of {categories}
- "direction": float from -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float from 0.0 to 1.0
- "persistence": "transient" | "short" | "medium" | "long"
- "confidence": float 0.0 to 1.0
- "source_articles": list of article numbers that support this factor
- "description": one-sentence explanation citing specific evidence

The magnitudes across all factors MUST sum to 1.0.
Include a "market_noise" factor for truly unexplained random variance.

IMPORTANT — DISTINGUISH NOISE FROM CONSOLIDATION (DISC-3):
- "market_noise" = purely stochastic, no identifiable cause, direction=0.0.
- If the market is SIDEWAYS or RANGE-BOUND due to opposing forces
  balancing out, do NOT dump that into "market_noise".  Instead, name a
  "consolidation" or "range_bound" factor with category="market_sentiment",
  direction near 0.0, and describe the competing forces.  This is
  qualitatively different from noise: it signals low-volatility regimes
  with mean-reverting dynamics.

Respond ONLY with valid JSON.  No markdown, no commentary.
"""

_RAG_PROMPT_BLIND = """\
You are a senior financial market analyst.  From the news articles below,
identify the events and conditions reported during this period that would
plausibly affect the stock price of {ticker}, and for each one state the
direction and strength of the effect you would EXPECT from the content of
the news alone.

You are NOT given the price data for this period.  Do not try to infer,
recall or reconstruct how {ticker} or the market actually moved.  Judge
each event only on its economic content.

ASSET: {ticker}
PERIOD: {start_date} to {end_date}

{news_context}

ANALYSIS METHODOLOGY — Think step-by-step:
1. List the concrete events and conditions reported in the articles
   (policy decisions, earnings, macroeconomic releases, sector news,
   geopolitical events).
2. For each one, ask: "From its economic content alone, would this event
   be expected to push {ticker} up or down, and how strongly?"
3. Separate structural factors (e.g., Fed policy) from transient catalysts
   (e.g., a single earnings miss).  Mark persistence accordingly.
4. Give the remaining weight to "market_noise" when the articles contain
   little that is relevant to {ticker}.

CRITICAL RULES:
1. Base your analysis EXCLUSIVELY on the news articles provided above.
2. Do NOT reference any events that are NOT in the provided news context.
3. Do NOT use knowledge of what happened AFTER {end_date}, and do NOT use
   any recollection of how {ticker} or the market performed in this period.
4. For each factor, cite the article number [N] that supports it.
5. Keep magnitude and confidence SEPARATE: magnitude = expected market
   impact; confidence = how certain you are that the event affects {ticker}.
6. NAMING RULE: Factor names must be SPECIFIC CAUSAL EVENTS, not generic
   descriptions of indicator movements.  BAD: "rising_treasury_yields",
   "declining_vix", "stable_unemployment_rate".  GOOD: "fed_tightening_cycle_2022",
   "post_covid_vix_normalization", "labor_market_recovery_q3".
   Always tie the name to the CAUSE or EVENT, not the indicator reading.

FEW-SHOT EXAMPLES (for format reference only — do NOT copy these factors):

Example 1 — events with a clear expected effect:
[
  {{"name": "fed_rate_hike_75bps", "category": "macroeconomic",
    "direction": -0.85, "magnitude": 0.55, "persistence": "medium",
    "confidence": 0.95, "source_articles": [2, 5],
    "description": "The Fed raised rates by 75bps [2]; tighter monetary conditions are expected to weigh on equity valuations [5]."}},
  {{"name": "tech_earnings_miss", "category": "company_specific",
    "direction": -0.6, "magnitude": 0.25, "persistence": "transient",
    "confidence": 0.80, "source_articles": [7],
    "description": "Major tech companies missed revenue estimates [7], which is expected to weigh on sector sentiment."}},
  {{"name": "market_noise", "category": "market_sentiment",
    "direction": 0.0, "magnitude": 0.20, "persistence": "transient",
    "confidence": 1.0, "source_articles": [],
    "description": "Weight not attributable to events identified in the articles."}}
]

Example 2 — mixed signals:
[
  {{"name": "stimulus_package", "category": "macroeconomic",
    "direction": 0.7, "magnitude": 0.30, "persistence": "long",
    "confidence": 0.85, "source_articles": [1, 3],
    "description": "Congress passed a $1.9T fiscal stimulus [1], expected to support consumer spending [3]."}},
  {{"name": "supply_chain_disruption", "category": "sector_specific",
    "direction": -0.5, "magnitude": 0.25, "persistence": "medium",
    "confidence": 0.70, "source_articles": [4],
    "description": "Global shipping delays raised input costs [4], expected to pressure margins."}},
  {{"name": "market_noise", "category": "market_sentiment",
    "direction": 0.0, "magnitude": 0.45, "persistence": "transient",
    "confidence": 1.0, "source_articles": [],
    "description": "Much of the period's news is not relevant to this asset."}}
]

NOW RESPOND for {ticker} ({start_date} to {end_date}).

Respond with a JSON array of factors.  Each factor must have:
- "name": short snake_case identifier (e.g., "fed_rate_hike")
- "category": one of {categories}
- "direction": float from -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float from 0.0 to 1.0
- "persistence": "transient" | "short" | "medium" | "long"
- "confidence": float 0.0 to 1.0
- "source_articles": list of article numbers that support this factor
- "description": one-sentence explanation citing specific evidence

The magnitudes across all factors MUST sum to 1.0.
Include a "market_noise" factor for the weight not explained by the articles.

Respond ONLY with valid JSON.  No markdown, no commentary.
"""


# Sentences that report how markets or prices moved.  In price-blind mode
# they are removed from the news context, because the prompt must not see
# the outcome it is asked to explain.
#
# The v1 patterns are FROZEN: the A1 price-blind run used them and they are
# hashed into its LLM cache keys (``legacy`` mode below reproduces both the
# behaviour and the keys).  Corpus-v2 changes go into ``_MARKET_TERMS`` /
# ``_MOVE_TERMS``.
_MARKET_TERMS_V1_PATTERN = (
    r"\b(Dow Jones|Dow|DJIA|S&P|S&amp;P|Nasdaq|NYSE|Russell 2000|FTSE|Nikkei|"
    r"Hang Seng|DAX|CAC 40|Stoxx|stock markets?|stocks?|shares?|equit(?:y|ies)|"
    r"Wall Street|bourses?|VIX|volatility index|fear gauge|market capitali[sz]ation)\b"
)
_MOVE_TERMS_V1_PATTERN = (
    r"\b(fell|falls?|falling|drop(?:s|ped|ping)?|plung(?:e|es|ed|ing)|plummet(?:s|ed|ing)?|"
    r"tumbl(?:e|es|ed|ing)|slump(?:s|ed|ing)?|sank|sinks?|sinking|crash(?:es|ed|ing)?|"
    r"lost|loses?|losing|declin(?:e|es|ed|ing)|rose|rises?|rising|gain(?:s|ed|ing)?|"
    r"jump(?:s|ed|ing)?|surg(?:e|es|ed|ing)|soar(?:s|ed|ing)?|rall(?:y|ies|ied|ying)|"
    r"rebound(?:s|ed|ing)?|climb(?:s|ed|ing)?|record (?:high|low|close)|all-time (?:high|low)|"
    r"points?|percent|sell-?offs?|bear market|bull market|correction|circuit breakers?|"
    r"trading (?:halt|suspended)|closed (?:up|down|higher|lower|at)|spik(?:e|es|ed|ing))\b|%"
)
_MARKET_TERMS_V1 = re.compile(_MARKET_TERMS_V1_PATTERN, re.IGNORECASE)
_MOVE_TERMS_V1 = re.compile(_MOVE_TERMS_V1_PATTERN, re.IGNORECASE)

# Corpus v2.  Market/equity terms (the strict mode drops every sentence that
# contains one).  Changes vs v1, from the redaction audit on real v2
# contexts (logs/redaction_audit_v2.md, round 1):
#   recall  + equity-index phrases ("benchmark index", "trading indexes",
#             "share index"), more named indices (IBEX 35, KOSPI, ...),
#             regional "<X> markets", market cap/value, "most valuable /
#             publicly traded / public company", trading halts;
#   precision – "Wall Street Journal", "Nikkei Asia", "S&P Global" (rating
#             agency) and bare "S&P" only with a move term; singular
#             "share" only as "share price/value/index"; "equity" only as
#             "equities" / "equity market(s)/price(s)/index" (not "private
#             equity").
_MARKET_TERMS_V2_PATTERN = (
    r"\b(Dow Jones|Dow|DJIA|S&P ?500|S&amp;P ?500|S&P (?:futures|index|indices)|S&P/(?:TSX|ASX)|"
    r"Nasdaq|NYSE|Russell 2000|FTSE|Nikkei(?! Asia)|Hang Seng|DAX|CAC 40|Stoxx|"
    r"IBEX(?: 35)?|KOSPI|Sensex|Nifty|I?Bovespa|Merval|TSX|ASX|MSCI|TOPIX|Tadawul|TASI|MOEX|"
    r"RTS index|Shanghai Composite|Shenzhen Component|CSI 300|SSE Composite|"
    r"Straits Times Index|KLCI|Jakarta Composite|PSEi|BIST(?: 100)?|Borsa Istanbul|EGX ?30|"
    r"TA-35|Tel Aviv 35|"
    r"stock markets?|stocks?|shares|share (?:prices?|values?|index|indices|indexes|markets?)|"
    r"equities|equity (?:markets?|prices?|index|indices|indexes|valuations?)|"
    r"Wall Street(?! Journal)|bourses?|VIX|volatility index|fear gauge|"
    r"market (?:capitali[sz]ation|caps?|value|valuation)|"
    r"(?:stock|share|equity|benchmark|main|major|leading|trading) (?:market )?(?:index|indices|indexes)|"
    r"(?:asian|european|global|world|emerging|international|financial|american|u\.s\.|us|"
    r"chinese|japanese|indian|russian) (?:financial |stock )?markets|"
    r"(?:most valuable|publicly[- ]traded|publicly listed|public) compan(?:y|ies)|"
    r"trading (?:was |is |were )?(?:halted|suspended|halts?)|"
    r"(?:halts?|halted|suspends?|suspended) (?:all )?trading|circuit breakers?)\b"
)
_MARKET_TERMS = re.compile(_MARKET_TERMS_V2_PATTERN, re.IGNORECASE)
# Strict mode also drops: generic "markets" / bare "S&P" with a move term,
# and "market(s)" with an extreme-move phrase ("worst week", "crash", ...).
_GENERIC_MARKETS = re.compile(r"\bmarkets\b|\bS&(?:amp;)?P\b(?!\s*Global)", re.IGNORECASE)
_MARKET_WORD = re.compile(r"\bmarkets?\b", re.IGNORECASE)
_EXTREME_MOVE = re.compile(
    r"\b((?:worst|best|biggest|largest) (?:single-|one-)?(?:day|week|month|quarter|year|session)s?|"
    r"record (?:high|low|close)s?|all-time (?:high|low)s?|crash(?:es|ed)?|sell-?offs?|"
    r"bear market|bull market|correction territory|circuit breakers?|trading halts?|"
    r"wip(?:e|es|ed) out|plung(?:e|es|ed|ing)|plummet(?:s|ed|ing)?|tumbl(?:e|es|ed|ing)|"
    r"rall(?:y|ies|ied|ying))\b",
    re.IGNORECASE,
)
_MOVE_TERMS_ADDED_PATTERN = (
    r"\b(increas(?:e|es|ed|ing)|decreas(?:e|es|ed|ing)|"
    r"(?:worst|best|biggest|largest) (?:single-|one-)?(?:day|week|month|quarter|year|session)s?|"
    r"slid(?:e|es|ing)?|slipp?(?:s|ed|ing)?|tank(?:ed|ing)|shed(?:s|ding)?|"
    r"wip(?:e|es|ed|ing) (?:out|off)|eras(?:e|es|ed|ing)|"
    r"(?:lowest|highest) (?:level|close|point)s?|hit (?:a )?record)\b"
)
_MOVE_TERMS = re.compile(
    _MOVE_TERMS_ADDED_PATTERN + "|" + _MOVE_TERMS_V1_PATTERN, re.IGNORECASE
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# Context v2.1: outcome phrases with
# no market/equity term.  ``strict2`` = ``strict`` + these.
_STRICT2_EXTRA = re.compile(
    r"\btrillion-dollar compan(?:y|ies)\b|"
    r"\blost\b[^.]{0,30}\bof (?:their|its) (?:market )?value\b|"
    r"\bBlack (?:Monday|Tuesday|Wednesday|Thursday|Friday)\b|"
    r"\b(?:richest|wealthiest) (?:person|man|woman|people)\b|\bnet worth\b|\bcurrent worth\b|"
    r"\bclosed\b[^.]{0,20}\d+(?:\.\d+)? ?(?:%|percent)[^.]{0,10}\b(?:up|down|higher|lower)\b",
    re.IGNORECASE,
)

#: Redaction modes (price-blind prompt only).
#:   off    – no redaction
#:   legacy – v1: drop a sentence with a market term AND a v1 move term
#:            (the A1 run; ``redact_market_moves=True`` maps here)
#:   moves  – drop a sentence with a v2 market term AND a v2 move term
#:   strict – corpus v2: drop EVERY sentence with a v2 market/equity term,
#:            with or without a move term; also generic "markets" (or bare
#:            "S&P") with a move term, and "market(s)" with an extreme move
#:   strict2 – context v2.1: strict + outcome phrases with no market term
#:            ("lost ... of their value", "trillion-dollar company", "Black
#:            Monday", "closed 9% down"); used with item-level dropping
REDACTION_MODES = ("off", "legacy", "moves", "strict", "strict2")

#: Everything a non-legacy mode depends on (hashed into the prompt hash).
_V2_REDACTION_SIGNATURE = "|".join((
    _MARKET_TERMS_V2_PATTERN, _GENERIC_MARKETS.pattern, _MARKET_WORD.pattern,
    _EXTREME_MOVE.pattern,
))


def normalize_redaction_mode(value: Union[bool, str, None]) -> str:
    """Map ``redact_market_moves`` config values to a redaction mode."""
    if value is None or value is False:
        return "off"
    if value is True:
        return "legacy"
    v = str(value).strip().lower()
    v = {"false": "off", "none": "off", "no": "off", "0": "off",
         "true": "legacy", "yes": "legacy", "1": "legacy",
         "v1": "legacy"}.get(v, v)
    if v not in REDACTION_MODES:
        raise ValueError(
            f"unknown redaction mode {value!r}; expected one of {REDACTION_MODES}"
        )
    return v


def sentence_is_redacted(sentence: str, mode: str = "legacy") -> bool:
    """Whether ``redact_market_moves`` drops this sentence under ``mode``."""
    if mode == "off":
        return False
    if mode == "legacy":
        return bool(_MARKET_TERMS_V1.search(sentence)
                    and _MOVE_TERMS_V1.search(sentence))
    if mode == "moves":
        return bool(_MARKET_TERMS.search(sentence)
                    and _MOVE_TERMS.search(sentence))
    if mode == "strict2":
        return sentence_is_redacted(sentence, "strict") or bool(_STRICT2_EXTRA.search(sentence))
    if mode == "strict":
        return bool(
            _MARKET_TERMS.search(sentence)
            or (_GENERIC_MARKETS.search(sentence) and _MOVE_TERMS.search(sentence))
            or (_MARKET_WORD.search(sentence) and _EXTREME_MOVE.search(sentence))
        )
    raise ValueError(f"unknown redaction mode {mode!r}")


def _is_header_line(line: str) -> bool:
    return (not line.strip() or line.startswith("===") or line.startswith("──")
            or line.startswith("Total articles"))


def iter_context_sentences(context: str):
    """Yield ``(line_index, sentence)`` for every redactable sentence."""
    for i, line in enumerate(context.split("\n")):
        if _is_header_line(line):
            continue
        for sent in _SENTENCE_SPLIT.split(line.strip()):
            yield i, sent


def redact_market_moves(context: str, mode: str = "legacy") -> tuple:
    """Drop sentences that report market or price moves.

    ``mode`` is one of :data:`REDACTION_MODES` (default ``legacy``: a
    sentence is dropped when it names a market or equity term AND a v1
    movement term; ``strict`` drops every sentence with a market or equity
    term).  Returns ``(redacted_context, n_dropped, n_kept)``.
    Header lines (``===`` / ``──``) are kept untouched.
    """
    mode = normalize_redaction_mode(mode)
    out_lines = []
    n_dropped = n_kept = 0
    for line in context.split("\n"):
        if _is_header_line(line):
            out_lines.append(line)
            continue
        indent = line[: len(line) - len(line.lstrip())]
        kept = []
        for sent in _SENTENCE_SPLIT.split(line.strip()):
            if sentence_is_redacted(sent, mode):
                n_dropped += 1
            else:
                kept.append(sent)
                n_kept += 1
        out_lines.append(indent + " ".join(kept) if kept else indent + "[redacted]")
    return "\n".join(out_lines), n_dropped, n_kept


# ══════════════════════════════════════════════════════════════════════
# Precomputed news contexts (corpus v2)
# ══════════════════════════════════════════════════════════════════════


class PrecomputedContexts:
    """Context provider backed by ``scripts/precompute_contexts.py`` files.

    ``<dir>/<TICKER>.json`` holds ``{"ticker", "corpus_tag", ...,
    "windows": [{window_start, window_end, context, ...}, ...]}``.  Loading
    is pure JSON: no Chroma, no embedder, no network, so an LLM process
    that uses this provider never touches the vector store.

    ``get(ticker, start, end)`` returns the stored (already redacted)
    context and raises ``KeyError`` for a window that was not precomputed,
    so a missing context can never silently become an empty prompt.
    """

    def __init__(self, directory: str, expected_corpus_tag: Optional[str] = None):
        self.directory = directory
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"precomputed context dir not found: {directory}")
        self.index: Dict = {}
        index_path = os.path.join(directory, "INDEX.json")
        if os.path.isfile(index_path):
            with open(index_path, "r", encoding="utf-8") as fh:
                self.index = json.load(fh)
        self.corpus_tag: Optional[str] = self.index.get("corpus_tag")
        #: redaction already applied to the stored contexts (INDEX), if known
        self.redaction_mode: Optional[str] = self.index.get("redaction_mode")
        if expected_corpus_tag is not None:
            if self.corpus_tag is None:
                self.corpus_tag = expected_corpus_tag
            elif self.corpus_tag != expected_corpus_tag:
                raise ValueError(
                    f"precomputed contexts have corpus_tag {self.corpus_tag!r}, "
                    f"expected {expected_corpus_tag!r}"
                )
        self._records: Dict[str, Dict[Tuple[str, str], Dict]] = {}
        self._lock = threading.Lock()

    def _load(self, ticker: str) -> Dict[Tuple[str, str], Dict]:
        with self._lock:
            recs = self._records.get(ticker)
            if recs is not None:
                return recs
            path = os.path.join(self.directory, f"{ticker}.json")
            if not os.path.isfile(path):
                raise KeyError(f"no precomputed contexts for {ticker} in {self.directory}")
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            tag = data.get("corpus_tag")
            if self.corpus_tag is None:
                self.corpus_tag = tag
            elif tag is not None and tag != self.corpus_tag:
                raise ValueError(
                    f"{path}: corpus_tag {tag!r} differs from {self.corpus_tag!r}"
                )
            recs = {(r["window_start"], r["window_end"]): r
                    for r in data.get("windows", [])}
            self._records[ticker] = recs
            return recs

    def record(self, ticker: str, start_date: str, end_date: str) -> Dict:
        recs = self._load(ticker)
        try:
            return recs[(start_date, end_date)]
        except KeyError:
            raise KeyError(
                f"no precomputed context for {ticker} {start_date}→{end_date} "
                f"in {self.directory}"
            ) from None

    def get(self, ticker: str, start_date: str, end_date: str) -> str:
        return self.record(ticker, start_date, end_date)["context"]

    def windows(self, ticker: str) -> List[Tuple[str, str]]:
        return sorted(self._load(ticker))


_FALLBACK_PROMPT = """\
You are a financial market analyst.  You are given real historical
price data for {ticker} during the period {start_date} to {end_date}.

Price data summary:
- Opening price: {open_price}
- Closing price: {close_price}
- High: {high_price}
- Low: {low_price}
- Price change: {price_change_pct:.2f}%
- Average daily volume: {avg_volume}

IMPORTANT CONSTRAINTS:
1. Only cite factors that were KNOWN and OBSERVABLE during this period.
   Do NOT use hindsight from future events.
2. Distinguish between factors that CAUSED movement vs. those that
   merely COINCIDED with it.
3. If the movement was likely random noise, say so explicitly.
4. Be honest about uncertainty in your attributions.

Respond with a JSON array of factors.  Each factor must have:
- "name": short snake_case identifier
- "category": one of {categories}
- "direction": float from -1.0 (bearish) to 1.0 (bullish)
- "magnitude": float from 0.0 to 1.0
- "persistence": "transient" | "short" | "medium" | "long"
- "confidence": float 0.0 to 1.0
- "description": one-sentence explanation

The magnitudes across all factors should sum to approximately 1.0.
Include a "noise_residual" factor for truly unexplained random variance.
If the market is sideways or range-bound, name a separate "consolidation"
factor — do NOT collapse it into noise.

Respond ONLY with valid JSON.  No markdown, no commentary.
"""


# ══════════════════════════════════════════════════════════════════════
# Extractor
# ══════════════════════════════════════════════════════════════════════


class LLMFactorExtractor:
    """
    Extracts causal factors from historical price data using an LLM.

    * Supports optional RAG grounding (``news_rag`` parameter).
    * Runs ``n_samples`` LLM calls per window and averages factor
      magnitudes for robustness.
    * Results are cached to disk so repeated runs are free.

    Parameters
    ----------
    llm_model : str
        LiteLLM-compatible model identifier.
    api_key : str | None
        API key; falls back to ``OPENROUTER_API_KEY`` env var.
    cache_dir : str
        Directory for JSON-cached window analyses.
    temperature : float
        LLM sampling temperature (low → consistent).
    n_samples : int
        Number of LLM calls per window (averaged).
    news_rag : NewsRAGManager | None
        If provided, retrieves contemporaneous news context.
    context_provider : object | None
        Alternative to ``news_rag``: any object with
        ``get(ticker, start_date, end_date) -> str`` (e.g.
        :class:`PrecomputedContexts`).  The extractor then never imports
        or opens Chroma.  Requires a ``corpus_tag`` (given, or taken from
        ``context_provider.corpus_tag``).
    corpus_tag : str | None
        Identifies the news corpus.  When set, the cache key also includes
        the corpus tag and the sha1 of the exact context text sent to the
        LLM, so labels cached for another corpus can never be reused.
        ``None`` keeps the historical key (replay of existing caches).
    redact_market_moves : bool | str
        Price-blind redaction mode (:data:`REDACTION_MODES`): ``False`` /
        ``"off"``, ``True`` / ``"legacy"`` (v1), ``"moves"``, ``"strict"``.
    llm_timeout : float
        Per-request timeout (s) for the LLM HTTP call.
    call_log_path : str | None
        If set, one JSON line per LLM attempt (time, model, prompt sha1,
        attempt, status, ``repr`` of the error) is appended to this file.

    The OpenRouter (litellm) path uses ONE long-lived sync HTTP client per
    extractor, passed to ``litellm.completion(client=...)``.  It is closed
    only by :meth:`close` (or leaving a ``with`` block), never by litellm's
    1-hour client-cache eviction.
    """

    def __init__(
        self,
        llm_model: str = "openrouter/anthropic/claude-sonnet-4",
        api_key: Optional[str] = None,
        cache_dir: str = "cache/llm_factors",
        temperature: float = 0.1,
        n_samples: int = 3,
        news_rag: Optional["NewsRAGManager"] = None,
        max_articles: int = 50,
        llm_provider: str = "openrouter",
        ollama_base_url: str = "",
        prompt_mode: str = "with_prices",
        redact_market_moves: Union[bool, str] = False,
        context_provider=None,
        corpus_tag: Optional[str] = None,
        llm_timeout: float = 600.0,
        call_log_path: Optional[str] = None,
    ):
        if prompt_mode not in ("with_prices", "price_blind"):
            raise ValueError(f"unknown prompt_mode: {prompt_mode}")
        if news_rag is not None and context_provider is not None:
            raise ValueError("pass either news_rag or context_provider, not both")
        if context_provider is not None and not callable(
                getattr(context_provider, "get", None)):
            raise TypeError("context_provider must have get(ticker, start, end) -> str")
        self.prompt_mode = prompt_mode
        self.redact_market_moves = redact_market_moves
        self.redaction_mode = normalize_redaction_mode(redact_market_moves)
        provider_mode = getattr(context_provider, "redaction_mode", None)
        if (context_provider is not None and provider_mode is not None
                and prompt_mode == "price_blind"
                and normalize_redaction_mode(provider_mode) != self.redaction_mode):
            raise ValueError(
                f"precomputed contexts were redacted with {provider_mode!r} but the "
                f"extractor is configured for {self.redaction_mode!r}")
        self._redaction_stats: List[Dict] = []
        self.llm_model = llm_model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self.cache_dir = cache_dir
        self.temperature = temperature
        self.n_samples = n_samples
        self.news_rag = news_rag
        self.context_provider = context_provider
        if corpus_tag is None and context_provider is not None:
            corpus_tag = getattr(context_provider, "corpus_tag", None)
        if context_provider is not None and not corpus_tag:
            raise ValueError(
                "context_provider requires a corpus_tag (argument or "
                "context_provider.corpus_tag), so cached labels are tied to "
                "the corpus"
            )
        self.corpus_tag = corpus_tag
        self.max_articles = max_articles
        self.llm_provider = llm_provider
        self.ollama_base_url = ollama_base_url.rstrip("/") if ollama_base_url else ""
        self.llm_timeout = float(llm_timeout)
        self.call_log_path = call_log_path
        os.makedirs(cache_dir, exist_ok=True)

        # ── long-lived HTTP client (litellm path) ───────────────────
        self._http_client = None
        self._http_lock = threading.Lock()
        self._closed = False
        self._log_lock = threading.Lock()
        self.n_llm_calls = 0          # attempts sent
        self.n_llm_failures = 0       # attempts that raised

        # ── prompt version hash (included in cache key) ─────────────
        # The default mode keeps the historical hash so existing caches stay
        # valid; price-blind mode hashes its own template and flags.  The
        # ``off`` / ``legacy`` modes reproduce the A1 keys exactly (v1
        # patterns, ``|redact=False`` / ``|redact=True``).
        if prompt_mode == "with_prices":
            prompt_src = _RAG_PROMPT + _FALLBACK_PROMPT
        elif self.redaction_mode in ("off", "legacy"):
            prompt_src = (_RAG_PROMPT_BLIND
                          + f"|redact={self.redaction_mode == 'legacy'}"
                          + _MARKET_TERMS_V1_PATTERN + _MOVE_TERMS_V1_PATTERN)
        else:
            prompt_src = (_RAG_PROMPT_BLIND + f"|redact={self.redaction_mode}"
                          + _V2_REDACTION_SIGNATURE + _MOVE_TERMS.pattern
                          + (_STRICT2_EXTRA.pattern if self.redaction_mode == "strict2" else ""))
        self._prompt_hash = hashlib.md5(prompt_src.encode()).hexdigest()[:8]

        # ── LLM consistency tracking ────────────────────────────────
        self._consistency_stats: List[Dict] = []

    # ── lifecycle ───────────────────────────────────────────────────

    def _uses_litellm_http_client(self) -> bool:
        return (self.llm_provider != "ollama"
                and self.llm_model.startswith("openrouter/"))

    def _get_http_client(self):
        """The extractor's single sync HTTP handler (created on first use)."""
        with self._http_lock:
            if self._closed:
                raise RuntimeError("LLMFactorExtractor was closed")
            if self._http_client is None:
                import httpx
                from litellm.llms.custom_httpx.http_handler import HTTPHandler

                self._http_client = HTTPHandler(
                    timeout=httpx.Timeout(self.llm_timeout, connect=30.0)
                )
            return self._http_client

    def close(self) -> None:
        """Close the long-lived HTTP client.  The extractor cannot call the
        LLM afterwards (cache hits still work)."""
        with self._http_lock:
            client, self._http_client = self._http_client, None
            self._closed = True
        if client is not None:
            client.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def _log_call(self, **fields) -> None:
        with self._log_lock:
            self.n_llm_calls += fields.get("status") == "sent"
            self.n_llm_failures += fields.get("status") == "error"
            if not self.call_log_path:
                return
            rec = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "model": self.llm_model, **fields}
            with open(self.call_log_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec) + "\n")

    # ── LLM call ────────────────────────────────────────────────────

    def _call_llm(self, prompt: str, _max_retries: int = 4) -> str:
        """Call the LLM API via LiteLLM (default) or Ollama.

        Retries on transient errors (Timeout, RateLimitError, APIError)
        with exponential backoff: 5s → 10s → 20s → 40s.  Every retry logs
        ``repr(e)``.
        """
        if self.llm_provider == "ollama":
            return self._call_ollama(prompt)

        import litellm  # type: ignore[import-untyped]

        litellm.suppress_debug_info = True
        extra = {}
        if self._uses_litellm_http_client():
            # litellm/main.py:3243 forwards ``client`` to the OpenRouter
            # handler, and llm_http_handler.py:551-556 uses it instead of
            # the cached client that litellm evicts (and closes) after 1 h.
            extra["client"] = self._get_http_client()
        prompt_sha1 = hashlib.sha1(prompt.encode("utf-8")).hexdigest()

        for attempt in range(_max_retries + 1):
            self._log_call(status="sent", attempt=attempt, prompt_sha1=prompt_sha1)
            try:
                response = litellm.completion(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    api_key=self.api_key,
                    max_tokens=4096,
                    timeout=self.llm_timeout,
                    **extra,
                )
                return response.choices[0].message.content
            except ImportError:
                raise ImportError("Install litellm:  pip install litellm")
            except (
                litellm.exceptions.Timeout,
                litellm.exceptions.RateLimitError,
                litellm.exceptions.ServiceUnavailableError,
                litellm.exceptions.APIConnectionError,
                litellm.exceptions.APIError,
                litellm.exceptions.InternalServerError,
            ) as e:
                self._log_call(status="error", attempt=attempt,
                               prompt_sha1=prompt_sha1, error=repr(e),
                               error_type=type(e).__name__)
                if attempt == _max_retries:
                    print(f"      ✗ LLM call failed after {attempt + 1} attempts: "
                          f"[{type(e).__name__}] {e!r}")
                    raise
                wait = 5 * (2 ** attempt)  # 5, 10, 20, 40s
                print(
                    f"      ⟳ Retry {attempt + 1}/{_max_retries} after "
                    f"[{type(e).__name__}] {e!r} — waiting {wait}s …"
                )
                time.sleep(wait)
            except Exception as e:
                self._log_call(status="error", attempt=attempt,
                               prompt_sha1=prompt_sha1, error=repr(e),
                               error_type=type(e).__name__)
                print(f"      ✗ LLM call failed (not retried): "
                      f"[{type(e).__name__}] {e!r}")
                raise

    def _call_ollama(self, prompt: str) -> str:
        """Call a self-hosted Ollama instance."""
        import requests  # stdlib-adjacent, always available

        url = f"{self.ollama_base_url}/api/generate"
        payload = {
            "model": self.llm_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": 4096,
            },
        }
        resp = requests.post(url, json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["response"]

    # ── parsing ─────────────────────────────────────────────────────

    @staticmethod
    def _parse_factors(raw_json: str) -> List[CausalFactor]:
        """Parse LLM JSON response into ``CausalFactor`` objects.

        Includes a sanitization step that fixes common LLM JSON issues
        (trailing commas, single quotes, unquoted keys) before parsing.
        """
        try:
            cleaned = raw_json.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1])

            # ── JSON sanitization ───────────────────────────────────
            import re
            # Remove trailing commas before } or ]
            cleaned = re.sub(r',\s*([\]}])', r'\1', cleaned)
            # Replace single-quoted strings with double-quoted
            # (only when it looks like JSON keys/values)
            cleaned = re.sub(r"(?<=[{,\[])\s*'([^']+)'\s*:", r' "\1":', cleaned)
            cleaned = re.sub(r":\s*'([^']*)'", r': "\1"', cleaned)

            factors_data = json.loads(cleaned)
            factors: List[CausalFactor] = []
            for f in factors_data:
                # BUG-C6: keep magnitude and confidence separate.
                # Multiplying them violates the sum-to-1 invariant
                # that the prompt asks the LLM to obey.  Confidence
                # is preserved in the description for downstream use.
                magnitude = float(f["magnitude"])
                confidence = f.get("confidence", 1.0)
                desc = f.get("description", "")
                if confidence < 1.0:
                    desc = f"[conf={confidence:.2f}] {desc}"
                factors.append(
                    CausalFactor(
                        name=f["name"],
                        category=f.get("category", "unknown"),
                        direction=float(f["direction"]),
                        magnitude=magnitude,
                        persistence=f.get("persistence", "medium"),
                        description=desc,
                    )
                )
            return factors
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            print(f"      WARNING: Failed to parse LLM response: {e}")
            return []

    # ── caching ─────────────────────────────────────────────────────

    def _cache_key(self, ticker: str, start: str, end: str,
                   context: Optional[str] = None) -> str:
        # BUG-C7: include hyperparams so changing temperature/n_samples/
        # max_articles invalidates stale cache entries.
        raw = (
            f"{ticker}_{start}_{end}_{self.llm_model}"
            f"_t{self.temperature}_ns{self.n_samples}"
            f"_ma{self.max_articles}_{self._prompt_hash}"
        )
        if self.corpus_tag is not None:
            # B12: tie cached labels to the corpus and to the exact context.
            if context is None:
                raise ValueError("a corpus-tagged cache key needs the context text")
            ctx_sha1 = hashlib.sha1(context.encode("utf-8")).hexdigest()
            raw += f"_corpus={self.corpus_tag}_ctx={ctx_sha1}"
        return hashlib.md5(raw.encode()).hexdigest()

    def _build_context(
        self, ticker: str, start_date: str, end_date: str
    ) -> Tuple[str, bool, Optional[Tuple[int, int]]]:
        """Fetch the news context and apply price-blind redaction.

        Returns ``(context, use_rag, redaction)`` where ``redaction`` is
        ``(n_dropped, n_kept)`` when redaction ran, else ``None``.
        """
        news_context = ""
        if self.context_provider is not None:
            news_context = self.context_provider.get(ticker, start_date, end_date)
            if getattr(self.context_provider, "redaction_mode", None) is not None:
                # Already redacted upstream with the same mode (checked in
                # __init__): report the stored counts, do not redact again.
                rec = getattr(self.context_provider, "record", None)
                r = rec(ticker, start_date, end_date) if callable(rec) else {}
                use_rag = bool(news_context
                               and "[No news articles found" not in news_context)
                return news_context, use_rag, (
                    int(r.get("n_sentences_dropped", 0)) + int(r.get("n_items_redacted", 0)),
                    int(r.get("n_sentences_kept", 0)))
        elif self.news_rag is not None:
            news_context = self.news_rag.get_context(
                ticker=ticker,
                start_date=start_date,
                end_date=end_date,
                max_articles=self.max_articles,
            )
        use_rag = bool(
            news_context
            and "[No news articles found" not in news_context
        )
        redaction = None
        if (self.prompt_mode == "price_blind" and use_rag
                and self.redaction_mode != "off"):
            news_context, n_drop, n_keep = redact_market_moves(
                news_context, self.redaction_mode)
            redaction = (n_drop, n_keep)
        return news_context, use_rag, redaction

    def _save_cache(self, path: str, analysis: WindowAnalysis):
        with open(path, "w") as fh:
            json.dump(analysis.to_dict(), fh, indent=2)

    def _load_cache(self, path: str) -> WindowAnalysis:
        with open(path, "r") as fh:
            return WindowAnalysis.from_dict(json.load(fh))

    # ── single window ──────────────────────────────────────────────

    def extract_window(
        self,
        df: pd.DataFrame,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> WindowAnalysis:
        """
        Extract causal factors for a single time window.

        Runs ``n_samples`` LLM calls and averages factor magnitudes
        for robustness.  Results are cached.
        """
        built = None
        if self.corpus_tag is None:
            # Historical key: checked before any retrieval.
            cache_key = self._cache_key(ticker, start_date, end_date)
        else:
            built = self._build_context(ticker, start_date, end_date)
            cache_key = self._cache_key(ticker, start_date, end_date, built[0])
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.json")

        if os.path.exists(cache_path):
            return self._load_cache(cache_path)

        # ── prepare price summary ───────────────────────────────────
        mask = (df["date"] >= start_date) & (df["date"] <= end_date)
        window_df = df[mask].sort_values("date")

        if window_df.empty:
            return WindowAnalysis(
                start_date=start_date,
                end_date=end_date,
                window_type="unknown",
                price_change_pct=0.0,
            )

        open_price = window_df.iloc[0]["open"]
        close_price = window_df.iloc[-1]["close"]
        price_change = ((close_price - open_price) / open_price) * 100

        # ── compute intra-window statistics ─────────────────────────
        intra_stats_lines = []
        closes = window_df["close"].values.astype(float)
        if len(closes) > 2:
            log_ret = np.diff(np.log(closes))
            log_ret = log_ret[np.isfinite(log_ret)]
            if len(log_ret) > 1:
                realized_vol = float(np.std(log_ret) * np.sqrt(252)) * 100
                max_drawdown = float(
                    (np.maximum.accumulate(closes) - closes).max()
                    / max(np.maximum.accumulate(closes).max(), 1e-10)
                    * 100
                )
                # Weekly returns (every 5 trading days)
                weekly_rets = []
                for wi in range(0, len(closes) - 5, 5):
                    wr = (closes[wi + 5] - closes[wi]) / closes[wi] * 100
                    weekly_rets.append(wr)
                intra_stats_lines.append("")
                intra_stats_lines.append("INTRA-WINDOW STATISTICS:")
                intra_stats_lines.append(
                    f"- Annualised volatility: {realized_vol:.1f}%"
                )
                intra_stats_lines.append(
                    f"- Max intra-window drawdown: {max_drawdown:.1f}%"
                )
                intra_stats_lines.append(
                    f"- Trading days in window: {len(closes)}"
                )
                if weekly_rets:
                    up_weeks = sum(1 for w in weekly_rets if w > 0)
                    dn_weeks = len(weekly_rets) - up_weeks
                    intra_stats_lines.append(
                        f"- Weekly returns: {up_weeks} up / {dn_weeks} down "
                        f"(best: {max(weekly_rets):+.1f}%, "
                        f"worst: {min(weekly_rets):+.1f}%)"
                    )
        intra_window_stats = "\n".join(intra_stats_lines)

        # ── build prompt ────────────────────────────────────────────
        if built is None:
            built = self._build_context(ticker, start_date, end_date)
        news_context, use_rag, redaction = built

        fmt_kwargs = dict(
            ticker=ticker,
            start_date=start_date,
            end_date=end_date,
            open_price=f"{open_price:.2f}",
            close_price=f"{close_price:.2f}",
            high_price=f"{window_df['high'].max():.2f}",
            low_price=f"{window_df['low'].min():.2f}",
            price_change_pct=price_change,
            avg_volume=f"{window_df['volume'].mean():,.0f}",
            intra_window_stats=intra_window_stats,
            categories=FACTOR_CATEGORIES,
        )

        if self.prompt_mode == "price_blind":
            if not use_rag:
                # No news and no prices: there is nothing to attribute.
                analysis = WindowAnalysis(
                    start_date=start_date,
                    end_date=end_date,
                    window_type=self._infer_window_type(start_date, end_date),
                    price_change_pct=price_change,
                    factors=[CausalFactor(
                        name="market_noise", category="market_sentiment",
                        direction=0.0, magnitude=1.0, persistence="transient",
                        description="No news retrieved; no LLM call (price-blind mode).",
                    )],
                    raw_llm_response="[price_blind: no news, no LLM call]",
                )
                self._redaction_stats.append({"window": f"{start_date}→{end_date}",
                                              "no_news": True, "dropped": 0, "kept": 0})
                self._save_cache(cache_path, analysis)
                return analysis
            n_drop, n_keep = redaction if redaction is not None else (0, 0)
            self._redaction_stats.append({"window": f"{start_date}→{end_date}",
                                          "no_news": False, "dropped": n_drop, "kept": n_keep})
            prompt = _RAG_PROMPT_BLIND.format(
                ticker=ticker, start_date=start_date, end_date=end_date,
                news_context=news_context, categories=FACTOR_CATEGORIES,
            )
        elif use_rag:
            fmt_kwargs["news_context"] = news_context
            prompt = _RAG_PROMPT.format(**fmt_kwargs)
        else:
            prompt = _FALLBACK_PROMPT.format(**fmt_kwargs)

        # ── multiple LLM samples (parallel) ──────────────────────
        all_factors: Dict[str, List[CausalFactor]] = {}
        raw_responses: List[str] = []
        parsed_per_sample: List[List[CausalFactor]] = []

        with ThreadPoolExecutor(max_workers=self.n_samples) as pool:
            futures = [
                pool.submit(self._call_llm, prompt)
                for _ in range(self.n_samples)
            ]
            for fut in futures:
                raw = fut.result()
                raw_responses.append(raw)
                factors = self._parse_factors(raw)
                parsed_per_sample.append(factors)
                for f in factors:
                    all_factors.setdefault(f.name, []).append(f)

        # ── LLM consistency tracking ──────────────────────────────
        n_parsed_ok = sum(1 for s in parsed_per_sample if len(s) > 0)
        factor_sets = [set(f.name for f in s) for s in parsed_per_sample if len(s) > 0]
        if len(factor_sets) >= 2:
            # Pairwise Jaccard similarity
            jaccards = []
            for i in range(len(factor_sets)):
                for j in range(i + 1, len(factor_sets)):
                    inter = len(factor_sets[i] & factor_sets[j])
                    union = len(factor_sets[i] | factor_sets[j])
                    jaccards.append(inter / union if union > 0 else 0.0)
            avg_jaccard = float(np.mean(jaccards))
            # Direction agreement for shared factors
            dir_agreements = []
            for name, flist in all_factors.items():
                if len(flist) >= 2:
                    dirs = [1 if f.direction > 0 else (-1 if f.direction < 0 else 0) for f in flist]
                    dir_agreements.append(1.0 if len(set(dirs)) == 1 else 0.0)
            avg_dir_agree = float(np.mean(dir_agreements)) if dir_agreements else 0.0
            # Magnitude CV for shared factors
            mag_cvs = []
            for name, flist in all_factors.items():
                if len(flist) >= 2:
                    mags = [f.magnitude for f in flist]
                    mean_m = np.mean(mags)
                    if mean_m > 0:
                        mag_cvs.append(float(np.std(mags) / mean_m))
            avg_mag_cv = float(np.mean(mag_cvs)) if mag_cvs else 0.0
        else:
            avg_jaccard = 0.0
            avg_dir_agree = 0.0
            avg_mag_cv = 0.0

        self._consistency_stats.append({
            "window": f"{start_date}→{end_date}",
            "n_samples": self.n_samples,
            "n_parsed_ok": n_parsed_ok,
            "parse_rate": n_parsed_ok / self.n_samples,
            "n_unique_factors": len(all_factors),
            "jaccard_similarity": avg_jaccard,
            "direction_agreement": avg_dir_agree,
            "magnitude_cv": avg_mag_cv,
        })

        # ── average across samples ──────────────────────────────────
        averaged: List[CausalFactor] = []
        for name, factor_list in all_factors.items():
            if len(factor_list) < self.n_samples / 2:
                continue  # appeared in fewer than half the samples
            avg_mag = float(np.mean([f.magnitude for f in factor_list]))
            avg_dir = float(np.mean([f.direction for f in factor_list]))
            averaged.append(
                CausalFactor(
                    name=name,
                    category=factor_list[0].category,
                    direction=avg_dir,
                    magnitude=avg_mag,
                    persistence=factor_list[0].persistence,
                    description=factor_list[0].description,
                )
            )

        # re-normalize magnitudes
        total_mag = sum(f.magnitude for f in averaged)
        if total_mag > 0:
            for f in averaged:
                f.magnitude /= total_mag

        analysis = WindowAnalysis(
            start_date=start_date,
            end_date=end_date,
            window_type=self._infer_window_type(start_date, end_date),
            price_change_pct=price_change,
            factors=averaged,
            raw_llm_response=raw_responses[0] if raw_responses else "",
        )

        self._save_cache(cache_path, analysis)
        return analysis

    # ── all windows ─────────────────────────────────────────────────

    @staticmethod
    def window_pairs(dates, window_sizes: List[str]) -> List[Tuple[str, str]]:
        """The ``(start, end)`` windows that :meth:`extract_all_windows`
        analyses for these trading dates (same order)."""
        dates = sorted(pd.to_datetime(pd.Series(dates).unique()))
        pairs: List[Tuple[str, str]] = []
        for ws in window_sizes:
            if ws == "1BD":
                # True business-day daily: every trading day
                for d in dates:
                    s = d.strftime("%Y-%m-%d")
                    pairs.append((s, s))
            elif ws == "1D":
                sampled = dates[::5]
                for d in sampled:
                    s = d.strftime("%Y-%m-%d")
                    pairs.append((s, s))
            else:
                periods = pd.date_range(
                    start=dates[0], end=dates[-1], freq=ws
                )
                for i in range(len(periods) - 1):
                    s = periods[i].strftime("%Y-%m-%d")
                    e = periods[i + 1].strftime("%Y-%m-%d")
                    pairs.append((s, e))
        return pairs

    def extract_all_windows(
        self,
        df: pd.DataFrame,
        ticker: str,
        window_sizes: Optional[List[str]] = None,
        max_workers: int = 4,
    ) -> List[WindowAnalysis]:
        """
        Extract factors across multiple time-window granularities.

        Parameters
        ----------
        df : pd.DataFrame
            Price DataFrame (date, open, high, low, close, volume).
        ticker : str
            Stock ticker.
        window_sizes : list[str] | None
            Pandas offset aliases.  Default ``["1ME", "1QE"]``.
        max_workers : int
            Max concurrent window extractions (default 4).

        Returns
        -------
        list[WindowAnalysis]
        """
        if window_sizes is None:
            window_sizes = ["1ME", "1QE"]

        # ── pre-compute all (start, end) pairs for progress bar ─────
        window_pairs = self.window_pairs(df["date"], window_sizes)

        # ── parallel extraction with progress bar ───────────────────
        try:
            from tqdm import tqdm  # type: ignore[import-untyped]
            pbar = tqdm(
                total=len(window_pairs),
                desc="    Extracting factors",
                unit="win",
                ncols=90,
            )
        except ImportError:
            pbar = None

        # Maintain original ordering via index
        results: List[Optional[WindowAnalysis]] = [None] * len(window_pairs)

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_to_idx = {
                pool.submit(self.extract_window, df, ticker, s, e): idx
                for idx, (s, e) in enumerate(window_pairs)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()
                if pbar is not None:
                    pbar.update(1)

        if pbar is not None:
            pbar.close()

        # ── print LLM consistency report ────────────────────────────
        self._print_consistency_report()

        return [r for r in results if r is not None]

    # ── LLM consistency report ──────────────────────────────────────

    def _print_consistency_report(self) -> None:
        """Print a summary of LLM consistency metrics across all windows."""
        stats = self._consistency_stats
        if not stats:
            return

        n_windows = len(stats)
        parse_rates = [s["parse_rate"] for s in stats]
        jaccards = [s["jaccard_similarity"] for s in stats]
        dir_agrees = [s["direction_agreement"] for s in stats]
        mag_cvs = [s["magnitude_cv"] for s in stats]

        print(f"\n{'─' * 60}")
        print(f"  LLM CONSISTENCY REPORT  ({n_windows} windows)")
        print(f"{'─' * 60}")
        print(f"  Parse success rate:      {np.mean(parse_rates)*100:.1f}%"
              f"  (min {np.min(parse_rates)*100:.0f}%,"
              f" max {np.max(parse_rates)*100:.0f}%)")
        print(f"  Factor Jaccard (avg):    {np.mean(jaccards):.3f}"
              f"  (std {np.std(jaccards):.3f})")
        print(f"  Direction agreement:     {np.mean(dir_agrees)*100:.1f}%"
              f"  (std {np.std(dir_agrees)*100:.1f}%)")
        print(f"  Magnitude CV (avg):      {np.mean(mag_cvs):.3f}"
              f"  (std {np.std(mag_cvs):.3f})")
        print()

        # Interpretation guide
        j = np.mean(jaccards)
        d = np.mean(dir_agrees)
        m = np.mean(mag_cvs)
        verdicts = []
        if j >= 0.6:
            verdicts.append("Factor identity: GOOD (≥0.6 Jaccard)")
        elif j >= 0.4:
            verdicts.append("Factor identity: MODERATE (0.4-0.6 Jaccard)")
        else:
            verdicts.append("Factor identity: LOW (<0.4 Jaccard) — consider reducing temperature")
        if d >= 0.85:
            verdicts.append("Direction consensus: STRONG (≥85%)")
        elif d >= 0.70:
            verdicts.append("Direction consensus: MODERATE (70-85%)")
        else:
            verdicts.append("Direction consensus: WEAK (<70%) — factors direction is noisy")
        if m <= 0.3:
            verdicts.append("Magnitude stability: GOOD (CV≤0.3)")
        elif m <= 0.6:
            verdicts.append("Magnitude stability: MODERATE (CV 0.3-0.6)")
        else:
            verdicts.append("Magnitude stability: POOR (CV>0.6) — magnitude estimates vary widely")

        for v in verdicts:
            print(f"  → {v}")
        print(f"{'─' * 60}\n")

    # ── embedding matrix ────────────────────────────────────────────

    @staticmethod
    def build_factor_embedding(
        analyses: List[WindowAnalysis],
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Build the factor embedding matrix from window analyses.

        Returns
        -------
        embedding : ndarray, shape (n_windows, n_factors)
            Signed magnitude (direction × magnitude) per (window, factor).
        factor_names : list[str]
        window_labels : list[str]
            ``"start|end"`` label per window.
        """
        all_names: set = set()
        for a in analyses:
            for f in a.factors:
                all_names.add(f.name)
        factor_names = sorted(all_names)
        name_to_idx = {n: i for i, n in enumerate(factor_names)}

        n_windows = len(analyses)
        n_factors = len(factor_names)
        embedding = np.zeros((n_windows, n_factors), dtype=np.float32)
        window_labels: List[str] = []

        for w_idx, analysis in enumerate(analyses):
            window_labels.append(
                f"{analysis.start_date}|{analysis.end_date}"
            )
            for f in analysis.factors:
                f_idx = name_to_idx[f.name]
                embedding[w_idx, f_idx] = f.direction * f.magnitude

        return embedding, factor_names, window_labels

    # ── presence matrix (impact-weighted) ─────────────────────────────

    @staticmethod
    def build_presence_matrix(
        analyses: List[WindowAnalysis],
        df: pd.DataFrame,
        threshold: float = 0.0,
    ) -> Tuple[np.ndarray, List[str], pd.DatetimeIndex]:
        """
        Build the impact-weighted factor-presence matrix F ∈ [0,1]^(T×K).

        DISC-2: Instead of collapsing magnitude to binary {0,1}, preserves
        the LLM-assigned magnitude so that rare-but-intense events (e.g.
        COVID, flash crashes) are not treated identically to frequent-but-
        mild factors (e.g. market_noise).  The FIN/TFT architecture
        already multiplies ``factor_presence * embeddings``, so continuous
        values in [0,1] propagate naturally as intensity weights.

        Expands window-level factors to daily resolution by assigning
        each day within a window the factor's magnitude (if the factor
        was identified for that window with magnitude > ``threshold``).
        When windows overlap, the maximum magnitude is kept.

        Parameters
        ----------
        analyses : list[WindowAnalysis]
            Output of ``extract_all_windows``.
        df : pd.DataFrame
            Price DataFrame with a ``date`` column.
        threshold : float
            Minimum magnitude to consider a factor present.

        Returns
        -------
        presence : ndarray, shape (T, K)
            Impact-weighted presence matrix in [0.0, 1.0].
        factor_names : list[str]
        dates : pd.DatetimeIndex
        """
        all_names: set = set()
        for a in analyses:
            for f in a.factors:
                if f.magnitude > threshold:
                    all_names.add(f.name)

        factor_names = sorted(all_names)
        name_to_idx = {n: i for i, n in enumerate(factor_names)}

        dates = pd.to_datetime(sorted(df["date"].unique()))
        T = len(dates)
        K = len(factor_names)
        presence = np.zeros((T, K), dtype=np.float32)

        date_to_row = {d: i for i, d in enumerate(dates)}

        for analysis in analyses:
            start = pd.Timestamp(analysis.start_date)
            end = pd.Timestamp(analysis.end_date)
            for f in analysis.factors:
                if f.magnitude <= threshold:
                    continue
                if f.name not in name_to_idx:
                    continue
                col = name_to_idx[f.name]
                for d in dates:
                    if start <= d <= end:
                        row = date_to_row[d]
                        # DISC-2: keep the maximum magnitude across
                        # overlapping windows (not binary 1.0).
                        presence[row, col] = max(
                            presence[row, col], f.magnitude
                        )

        return presence, factor_names, dates

    # ── helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _infer_window_type(start: str, end: str) -> str:
        delta = (pd.Timestamp(end) - pd.Timestamp(start)).days
        if delta <= 1:
            return "daily"
        elif delta <= 35:
            return "monthly"
        else:
            return "quarterly"
