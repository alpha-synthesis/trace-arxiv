# -*- coding: utf-8 -*-
"""
Interactive Event Validation UI  (v2 — full CRUD).

A Dash-based web application that lets the user:

1. **View** historical price data as a candlestick chart with
   pan / zoom / range-slider navigation.
2. **See** LLM-extracted factors as semi-transparent rectangles
   on the chart (pre-populated and individually editable).
3. **Add** new event boxes by filling the sidebar form.
4. **Delete** individual events via per-row ✕ buttons.
5. **Edit** probability and direction inline for every event.
6. **Export** validated events as a JSON + binary presence matrix
   (.npz) ready for the FIN training pipeline.

Launched automatically by the pipeline when
``interactive.post_extraction: true`` in the YAML config,
or programmatically::

    from sde_causal_generator.event_validator import create_dash_app
    app = create_dash_app(df, ticker="AAPL")
    app.run(port=8050)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data_structures import CausalFactor, ImpactMatrix, Scenario, ScenarioSet


# ══════════════════════════════════════════════════════════════════════
# Event data model
# ══════════════════════════════════════════════════════════════════════


class UserEvent:
    """An event manually defined (or imported from LLM) on the chart."""

    def __init__(
        self,
        name: str,
        start_date: str,
        end_date: str,
        direction: str = "bearish",       # "bullish" | "bearish" | "neutral"
        probability: float = 1.0,         # P(occurs) 0-1
        category: str = "geopolitical",
        description: str = "",
        source: str = "user",             # "llm" | "user"
    ):
        self.name = name
        self.start_date = start_date
        self.end_date = end_date
        self.direction = direction
        self.probability = probability
        self.category = category
        self.description = description
        self.source = source

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "direction": self.direction,
            "probability": self.probability,
            "category": self.category,
            "description": self.description,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "UserEvent":
        valid_keys = {
            "name", "start_date", "end_date", "direction",
            "probability", "category", "description", "source",
        }
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


# ══════════════════════════════════════════════════════════════════════
# Conversion helpers
# ══════════════════════════════════════════════════════════════════════


def events_to_presence_matrix(
    events: List[UserEvent],
    dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, List[str]]:
    """Convert user-defined events -> binary presence matrix (T, K)."""
    factor_names = sorted(set(e.name for e in events))
    name_to_idx = {n: i for i, n in enumerate(factor_names)}
    T = len(dates)
    K = len(factor_names)
    presence = np.zeros((T, K), dtype=np.float32)
    date_to_row = {d: i for i, d in enumerate(dates)}

    for event in events:
        col = name_to_idx[event.name]
        start = pd.Timestamp(event.start_date)
        end = pd.Timestamp(event.end_date)
        for d in dates:
            if start <= d <= end:
                presence[date_to_row[d], col] = 1.0

    return presence, factor_names


def events_to_scenario_set(events: List[UserEvent]) -> ScenarioSet:
    """Build a single-scenario ScenarioSet from user events."""
    factor_probs = {e.name: e.probability for e in events}
    factor_names = sorted(factor_probs.keys())
    sc = Scenario(
        name="user_defined",
        description="Manually validated scenario from interactive UI.",
        factor_probs=factor_probs,
        scenario_prob=1.0,
    )
    return ScenarioSet(scenarios=[sc], factor_names=factor_names)


def merge_llm_and_user_events(
    llm_presence: np.ndarray,
    llm_factor_names: List[str],
    user_events: List[UserEvent],
    dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, List[str]]:
    """Merge LLM-extracted presence with user-validated events.

    User events **override** LLM factors with the same name.
    New user events are appended.
    """
    user_presence, user_names = events_to_presence_matrix(user_events, dates)
    merged_names = list(llm_factor_names)
    merged = llm_presence.copy()

    for u_idx, uname in enumerate(user_names):
        if uname in merged_names:
            m_idx = merged_names.index(uname)
            merged[:, m_idx] = user_presence[:, u_idx]
        else:
            merged_names.append(uname)
            merged = np.column_stack([merged, user_presence[:, u_idx]])

    return merged, merged_names


# ══════════════════════════════════════════════════════════════════════
# Event deduplication & pre-filtering
# ══════════════════════════════════════════════════════════════════════

_NOISE_KEYWORDS = {
    "noise", "residual", "random", "unexplained", "unknown",
    "other", "misc", "general",
}


def deduplicate_events(
    events: List[UserEvent],
    min_probability: float = 0.08,
    max_events: int = 0,
) -> List[UserEvent]:
    """Merge same-name events across windows and filter noise.

    For each unique factor name:
    * Merges overlapping or separate windows into the widest span
      (earliest start, latest end).
    * Averages probability across occurrences.
    * Keeps the majority direction.
    * Filters out noise/residual factors and low-probability ones.
    * Returns at most ``max_events`` sorted by occurrence count
      (descending), then by probability.

    Parameters
    ----------
    events : list[UserEvent]
    min_probability : float
        Drop factors with mean probability below this threshold.
    max_events : int
        Keep at most this many events (0 = no limit).
    """
    from collections import defaultdict

    groups: dict = defaultdict(list)
    for e in events:
        # Skip noise factors
        name_lower = e.name.lower().replace("_", " ")
        if any(kw in name_lower for kw in _NOISE_KEYWORDS):
            continue
        groups[e.name].append(e)

    merged: List[UserEvent] = []
    for name, group in groups.items():
        starts = sorted(e.start_date for e in group)
        ends = sorted(e.end_date for e in group)
        avg_prob = sum(e.probability for e in group) / len(group)

        if avg_prob < min_probability:
            continue

        # Majority direction
        dir_counts: dict = defaultdict(int)
        for e in group:
            dir_counts[e.direction] += 1
        direction = max(dir_counts, key=dir_counts.get)

        # Use first event's metadata
        first = group[0]
        merged.append(UserEvent(
            name=name,
            start_date=starts[0],
            end_date=ends[-1],
            direction=direction,
            probability=round(avg_prob, 3),
            category=first.category,
            description=(
                first.description
                or f"Merged from {len(group)} window(s)"
            ),
            source=first.source,
        ))

    # Sort: most frequent first, then by probability
    occurrence_count = {name: len(g) for name, g in groups.items()}
    merged.sort(
        key=lambda e: (occurrence_count.get(e.name, 0), e.probability),
        reverse=True,
    )

    if max_events > 0 and len(merged) > max_events:
        merged = merged[:max_events]

    return merged


# ══════════════════════════════════════════════════════════════════════
# Static Plotly chart (reusable by both apps)
# ══════════════════════════════════════════════════════════════════════


def plot_price_with_events(
    df: pd.DataFrame,
    events: Optional[List[UserEvent]] = None,
    ticker: str = "",
    output_html: Optional[str] = None,
    show_rangeslider: bool = True,
    log_scale: bool = False,
) -> go.Figure:
    """Create a Plotly candlestick chart with event overlays."""
    # -- robust data preparation (prevents blank chart) ---------------
    _df = df.copy()
    _df["date"] = pd.to_datetime(_df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        if c in _df.columns:
            _df[c] = pd.to_numeric(_df[c], errors="coerce")
    _df = _df.dropna(subset=["open", "high", "low", "close"]).sort_values("date")
    x_dates = _df["date"].dt.strftime("%Y-%m-%d")

    from plotly.subplots import make_subplots

    has_volume = "volume" in _df.columns
    if has_volume:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.75, 0.25],
        )
    else:
        fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=x_dates,
            open=_df["open"], high=_df["high"],
            low=_df["low"], close=_df["close"],
            name=ticker or "Price",
        ),
        row=1, col=1,
    )
    if has_volume:
        fig.add_trace(
            go.Bar(
                x=x_dates, y=_df["volume"],
                name="Volume",
                marker_color="rgba(100,100,200,0.4)",
            ),
            row=2, col=1,
        )

    # Event boxes with direction-dependent coloring
    dir_colors = {
        "bearish": ("rgba(255,50,50,{a})", "#C62828"),
        "bullish": ("rgba(50,200,50,{a})", "#2E7D32"),
        "neutral": ("rgba(200,200,50,{a})", "#F9A825"),
    }

    # Limit annotations when there are many events (prevents rendering stall)
    n_events = len(events) if events else 0
    show_annotations = n_events <= 40
    # Cap vrects to prevent Plotly from choking
    MAX_VRECTS = 60

    if events:
        chart_events = events[:MAX_VRECTS] if n_events > MAX_VRECTS else events
        for ev in chart_events:
            fill_tpl, line_col = dir_colors.get(
                ev.direction, dir_colors["neutral"]
            )
            alpha = 0.12 * ev.probability
            fill = fill_tpl.format(a=alpha)
            is_llm = getattr(ev, "source", "user") == "llm"
            tag = "LLM" if is_llm else "USR"

            kwargs = dict(
                x0=ev.start_date, x1=ev.end_date,
                fillcolor=fill,
                opacity=0.9,
                layer="below",
                line_width=1,
                line_color=line_col,
                row=1, col=1,
            )
            if show_annotations:
                kwargs.update(
                    annotation_text=(
                        f"[{tag}] {ev.name} (P={ev.probability:.0%})"
                    ),
                    annotation_position="top left",
                    annotation_font_size=9,
                )
            fig.add_vrect(**kwargs)

    fig.update_layout(
        title=f"{ticker} — Event Validation" if ticker else "Event Validation",
        yaxis_title="Price",
        yaxis_type="log" if log_scale else "linear",
        xaxis_rangeslider_visible=show_rangeslider,
        template="plotly_white",
        dragmode="zoom",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=40),
    )
    if has_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    if output_html:
        fig.write_html(output_html)
    return fig


# ══════════════════════════════════════════════════════════════════════
#  CSS constants
# ══════════════════════════════════════════════════════════════════════

_BTN = {
    "padding": "10px",
    "color": "white",
    "border": "none",
    "borderRadius": "4px",
    "cursor": "pointer",
    "width": "100%",
    "marginBottom": "8px",
}

_CARD = {
    "padding": "8px 10px",
    "marginBottom": "6px",
    "borderRadius": "4px",
    "fontSize": "12px",
}


# ══════════════════════════════════════════════════════════════════════
# Dash Application — Post-Extraction Editor  (Point 1)
# ══════════════════════════════════════════════════════════════════════


def create_dash_app(
    df: pd.DataFrame,
    ticker: str = "",
    initial_events: Optional[List[UserEvent]] = None,
    llm_factors: Optional[List[CausalFactor]] = None,
    output_dir: str = "cache/validated_events",
    port: int = 8050,
):
    """
    Dash app for interactive event editing **before FIN training**.

    Features added in v2
    --------------------
    * Per-event **x delete** button.
    * Inline **probability slider** per event.
    * Inline **direction dropdown** per event.
    * LLM/User **source badge** on every event.
    * Plotly range-slider for long time series navigation.
    * Pattern-matching callbacks for dynamic component IDs.
    """
    try:
        from dash import (
            Dash, Input, Output, State, ALL, MATCH,
            callback_context, dcc, html,
        )
        from dash.exceptions import PreventUpdate
    except ImportError:
        raise ImportError(
            "Dash is required for the interactive UI. "
            "Install it with:  pip install dash"
        )

    os.makedirs(output_dir, exist_ok=True)

    app = Dash(
        __name__,
        title=f"Event Validator — {ticker}",
        suppress_callback_exceptions=True,
    )

    if initial_events is None:
        initial_events = []
    events_json = json.dumps([e.to_dict() for e in initial_events])

    min_date = str(df["date"].min())[:10]
    max_date = str(df["date"].max())[:10]

    from .data_structures import FACTOR_CATEGORIES

    # -- layout -------------------------------------------------------
    app.layout = html.Div(
        style={
            "fontFamily": "'Segoe UI', 'Roboto', 'Helvetica Neue', Arial, sans-serif",
            "padding": "12px 20px",
            "backgroundColor": "#f5f6fa",
            "height": "95vh",
            "boxSizing": "border-box",
            "display": "flex",
            "flexDirection": "column",
            "overflow": "hidden",
        },
        children=[
            html.H2(
                f"Causal SDE — Event Editor (Post-Extraction) [{ticker}]",
                style={"color": "#1a1a2e", "margin": "0 0 4px 0",
                       "fontSize": "18px", "flexShrink": "0",
                       "fontWeight": "700"},
            ),
            html.P(
                "LLM-extracted factors are shown in the chart. "
                "Add new events, delete or edit any event, then Export.",
                style={"color": "#555", "marginBottom": "8px",
                       "fontSize": "13px", "flexShrink": "0"},
            ),

            # --- main: chart | sidebar (fills remaining height) ---
            html.Div(
                style={"display": "flex", "gap": "20px",
                       "flex": "1", "minHeight": "0"},
                children=[
                    # Chart (75 %)
                    html.Div(
                        style={"flex": "3", "minHeight": "0",
                               "display": "flex", "flexDirection": "column"},
                        children=[
                            # Chart toolbar
                            html.Div(
                                style={"display": "flex", "gap": "12px",
                                       "alignItems": "center",
                                       "marginBottom": "4px",
                                       "flexShrink": "0"},
                                children=[
                                    html.Span("Y-axis:",
                                              style={"fontSize": "12px",
                                                     "color": "#555",
                                                     "fontWeight": "600"}),
                                    dcc.RadioItems(
                                        id="y-scale-toggle",
                                        options=[
                                            {"label": "Linear", "value": "linear"},
                                            {"label": "Log", "value": "log"},
                                        ],
                                        value="linear",
                                        inline=True,
                                        style={"fontSize": "12px"},
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "12px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="price-chart",
                                style={"flex": "1", "minHeight": "0"},
                                responsive=True,
                                config={
                                    "scrollZoom": True,
                                    "displayModeBar": True,
                                    "modeBarButtonsToAdd": [
                                        "zoomIn2d", "zoomOut2d",
                                        "resetScale2d",
                                    ],
                                },
                            ),
                        ],
                    ),

                    # Sidebar (25 %)
                    html.Div(
                        style={
                            "flex": "1",
                            "backgroundColor": "#ffffff",
                            "padding": "15px",
                            "borderRadius": "8px",
                            "color": "#1a1a2e",
                            "overflowY": "auto",
                            "minHeight": "0",
                            "boxShadow": "0 2px 8px rgba(0,0,0,0.08)",
                            "border": "1px solid #e0e0e0",
                        },
                        children=[
                            # -- Add form ----
                            html.H4("Add Event",
                                    style={"marginTop": 0}),
                            html.Label("Name:"),
                            dcc.Input(
                                id="evt-name", type="text",
                                placeholder="e.g. subprime_crisis",
                                style={"width": "100%", "marginBottom": "6px"},
                            ),
                            html.Label("Start:"),
                            dcc.DatePickerSingle(
                                id="evt-start",
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                style={"marginBottom": "6px"},
                            ),
                            html.Label("End:"),
                            dcc.DatePickerSingle(
                                id="evt-end",
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                style={"marginBottom": "6px"},
                            ),
                            html.Label("Direction:"),
                            dcc.Dropdown(
                                id="evt-direction",
                                options=[
                                    {"label": "Bearish", "value": "bearish"},
                                    {"label": "Bullish", "value": "bullish"},
                                    {"label": "Neutral", "value": "neutral"},
                                ],
                                value="bearish",
                                clearable=False,
                                style={"marginBottom": "6px"},
                            ),
                            html.Label("Category:"),
                            dcc.Dropdown(
                                id="evt-category",
                                options=[
                                    {"label": c, "value": c}
                                    for c in FACTOR_CATEGORIES
                                ],
                                value="geopolitical",
                                clearable=False,
                                style={"marginBottom": "6px"},
                            ),
                            html.Label("Probability:"),
                            dcc.Slider(
                                id="evt-prob",
                                min=0, max=1, step=0.05, value=1.0,
                                marks={0: "0", 0.25: ".25",
                                       0.5: ".5", 0.75: ".75", 1: "1"},
                            ),
                            html.Label("Description:"),
                            dcc.Textarea(
                                id="evt-desc",
                                placeholder="Optional ...",
                                style={"width": "100%", "height": "50px",
                                       "marginBottom": "10px"},
                            ),
                            html.Button(
                                "Add Event", id="btn-add", n_clicks=0,
                                style={**_BTN, "backgroundColor": "#0f3460"},
                            ),
                            html.Hr(),

                            # -- Search / filter ---
                            html.H4("Events"),
                            dcc.Input(
                                id="evt-search", type="text",
                                placeholder="Search events...",
                                debounce=True,
                                style={"width": "100%", "marginBottom": "8px",
                                       "padding": "8px", "borderRadius": "6px",
                                       "border": "1px solid #ccc",
                                       "backgroundColor": "#f9f9f9",
                                       "color": "#222", "fontSize": "13px"},
                            ),
                            dcc.Dropdown(
                                id="evt-filter-source",
                                options=[
                                    {"label": "All sources", "value": "all"},
                                    {"label": "LLM only", "value": "llm"},
                                    {"label": "User only", "value": "user"},
                                ],
                                value="all",
                                clearable=False,
                                style={"marginBottom": "6px", "fontSize": "12px"},
                            ),
                            dcc.Dropdown(
                                id="evt-filter-direction",
                                options=[
                                    {"label": "All directions", "value": "all"},
                                    {"label": "\u25b2 Bullish", "value": "bullish"},
                                    {"label": "\u25bc Bearish", "value": "bearish"},
                                    {"label": "\u25cf Neutral", "value": "neutral"},
                                ],
                                value="all",
                                clearable=False,
                                style={"marginBottom": "6px", "fontSize": "12px"},
                            ),
                            dcc.Dropdown(
                                id="evt-filter-category",
                                options=(
                                    [{"label": "All categories", "value": "all"}]
                                    + [{"label": c.replace('_', ' ').title(),
                                        "value": c}
                                       for c in FACTOR_CATEGORIES]
                                ),
                                value="all",
                                clearable=False,
                                style={"marginBottom": "8px", "fontSize": "12px"},
                            ),
                            html.Div(id="event-list"),
                            html.Hr(),

                            # -- Export ---
                            html.Button(
                                "Export & Continue",
                                id="btn-export", n_clicks=0,
                                style={**_BTN, "backgroundColor": "#1a5276"},
                            ),
                            html.Div(
                                id="export-status",
                                style={"color": "#2ecc71",
                                       "marginTop": "6px"},
                            ),
                        ],
                    ),
                ],
            ),

            # -- hidden stores ---
            dcc.Store(id="events-store", data=events_json),
            # Used to propagate delete / inline-edit actions
            dcc.Store(id="delete-trigger", data=""),
        ],
    )

    # ================================================================
    # Callbacks
    # ================================================================

    @app.callback(
        Output("events-store", "data"),
        [
            Input("btn-add", "n_clicks"),
            Input("delete-trigger", "data"),
        ],
        [
            State("events-store", "data"),
            State("evt-name", "value"),
            State("evt-start", "date"),
            State("evt-end", "date"),
            State("evt-direction", "value"),
            State("evt-category", "value"),
            State("evt-prob", "value"),
            State("evt-desc", "value"),
        ],
        prevent_initial_call=True,
    )
    def modify_store(
        add_clicks, delete_payload,
        store_json, name, start, end, direction, category, prob, desc,
    ):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        trigger_id = ctx.triggered[0]["prop_id"].split(".")[0]

        events = json.loads(store_json) if store_json else []

        if trigger_id == "btn-add":
            if not name or not start or not end:
                raise PreventUpdate
            events.append(UserEvent(
                name=name,
                start_date=str(start)[:10],
                end_date=str(end)[:10],
                direction=direction or "bearish",
                probability=prob if prob is not None else 1.0,
                category=category or "geopolitical",
                description=desc or "",
                source="user",
            ).to_dict())

        elif trigger_id == "delete-trigger" and delete_payload:
            try:
                payload = json.loads(delete_payload)
                action = payload.get("action")
                idx = payload.get("index")
                if action == "delete" and 0 <= idx < len(events):
                    events.pop(idx)
                elif action == "update_prob" and 0 <= idx < len(events):
                    events[idx]["probability"] = payload["value"]
                elif action == "update_dir" and 0 <= idx < len(events):
                    events[idx]["direction"] = payload["value"]
                elif action == "update_start" and 0 <= idx < len(events):
                    if payload.get("value"):
                        events[idx]["start_date"] = payload["value"]
                elif action == "update_end" and 0 <= idx < len(events):
                    if payload.get("value"):
                        events[idx]["end_date"] = payload["value"]
            except (json.JSONDecodeError, KeyError, TypeError):
                raise PreventUpdate

        return json.dumps(events)

    # -- chart (responds to search/filter so plot matches sidebar) --
    @app.callback(
        Output("price-chart", "figure"),
        [
            Input("events-store", "data"),
            Input("evt-search", "value"),
            Input("evt-filter-source", "value"),
            Input("evt-filter-direction", "value"),
            Input("evt-filter-category", "value"),
            Input("y-scale-toggle", "value"),
        ],
    )
    def update_chart(store_json, search_text, filter_source,
                     filter_direction, filter_category, y_scale):
        events = []
        if store_json:
            all_events = json.loads(store_json)
            search_lower = (search_text or "").strip().lower()
            for e in all_events:
                src = e.get("source", "user")
                if filter_source and filter_source != "all" and src != filter_source:
                    continue
                if filter_direction and filter_direction != "all" and e.get("direction") != filter_direction:
                    continue
                if filter_category and filter_category != "all" and e.get("category") != filter_category:
                    continue
                if search_lower:
                    searchable = " ".join([
                        e.get("name", ""),
                        e.get("category", ""),
                        e.get("description", ""),
                        e.get("start_date", ""),
                        e.get("end_date", ""),
                    ]).lower()
                    if search_lower not in searchable:
                        continue
                events.append(UserEvent.from_dict(e))
        return plot_price_with_events(
            df, events=events, ticker=ticker, show_rangeslider=True,
            log_scale=(y_scale == "log"),
        )

    # -- event list with inline controls ---
    @app.callback(
        Output("event-list", "children"),
        [
            Input("events-store", "data"),
            Input("evt-search", "value"),
            Input("evt-filter-source", "value"),
            Input("evt-filter-direction", "value"),
            Input("evt-filter-category", "value"),
        ],
    )
    def render_event_list(store_json, search_text, filter_source,
                          filter_direction, filter_category):
        if not store_json:
            return html.P("No events.", style={"color": "#666"})

        events = json.loads(store_json)
        if not events:
            return html.P("No events.", style={"color": "#666"})

        # Apply filters
        search_lower = (search_text or "").strip().lower()

        dir_sym = {"bearish": "\u25bc Bear", "bullish": "\u25b2 Bull", "neutral": "\u25cf Neut"}
        src_colors = {"llm": "#1565C0", "user": "#2E7D32"}

        items = []
        shown = 0
        for i, e in enumerate(events):
            src = e.get("source", "user")

            # Source filter
            if filter_source and filter_source != "all" and src != filter_source:
                continue
            # Direction filter
            if filter_direction and filter_direction != "all" and e.get("direction") != filter_direction:
                continue
            # Category filter
            if filter_category and filter_category != "all" and e.get("category") != filter_category:
                continue
            # Text search (name, category, description, dates)
            if search_lower:
                searchable = " ".join([
                    e.get("name", ""),
                    e.get("category", ""),
                    e.get("description", ""),
                    e.get("start_date", ""),
                    e.get("end_date", ""),
                ]).lower()
                if search_lower not in searchable:
                    continue

            shown += 1
            sym = dir_sym.get(e.get("direction", ""), "?")
            src = e.get("source", "user")
            bg = "#e8eef6" if src == "llm" else "#e6f4ea"
            items.append(
                html.Div(
                    style={**_CARD, "backgroundColor": bg,
                           "borderLeft": f"3px solid {src_colors.get(src, '#555')}"},
                    children=[
                        # Row 1: name + delete
                        html.Div(
                            style={"display": "flex",
                                   "justifyContent": "space-between",
                                   "alignItems": "center"},
                            children=[
                                html.B(
                                    f"{sym} {e['name']}",
                                    style={"fontSize": "13px", "color": "#1a1a2e"},
                                ),
                                html.Button(
                                    "X",
                                    id={"type": "btn-del", "index": i},
                                    n_clicks=0,
                                    style={
                                        "background": "transparent",
                                        "border": "1px solid #c0392b",
                                        "color": "#c0392b",
                                        "borderRadius": "3px",
                                        "cursor": "pointer",
                                        "fontSize": "10px",
                                        "padding": "1px 5px",
                                    },
                                ),
                            ],
                        ),
                        # Row 2: inline date pickers + source badge
                        html.Div(
                            style={"display": "flex",
                                   "alignItems": "center",
                                   "gap": "4px",
                                   "marginTop": "3px",
                                   "flexWrap": "wrap"},
                            children=[
                                html.Span("From:", style={
                                    "fontSize": "11px", "color": "#666"}),
                                dcc.DatePickerSingle(
                                    id={"type": "evt-start-edit",
                                        "index": i},
                                    date=e.get("start_date", min_date),
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    style={"fontSize": "10px"},
                                ),
                                html.Span("To:", style={
                                    "fontSize": "11px", "color": "#666"}),
                                dcc.DatePickerSingle(
                                    id={"type": "evt-end-edit",
                                        "index": i},
                                    date=e.get("end_date", max_date),
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    style={"fontSize": "10px"},
                                ),
                                html.Span(
                                    src.upper(),
                                    style={
                                        "fontSize": "9px",
                                        "backgroundColor":
                                            src_colors.get(src, "#555"),
                                        "color": "white",
                                        "padding": "1px 5px",
                                        "borderRadius": "3px",
                                        "marginLeft": "auto",
                                    },
                                ),
                            ],
                        ),
                        # Row 3: inline prob slider
                        html.Div(
                            style={"marginTop": "4px",
                                   "display": "flex",
                                   "alignItems": "center",
                                   "gap": "6px"},
                            children=[
                                html.Span("P:", style={
                                    "fontSize": "12px", "color": "#555"}),
                                dcc.Slider(
                                    id={"type": "prob-slider", "index": i},
                                    min=0, max=1, step=0.05,
                                    value=e.get("probability", 1.0),
                                    marks={0: "0", 0.5: ".5", 1: "1"},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                ),
                            ],
                        ),
                        # Row 4: inline direction
                        html.Div(
                            style={"marginTop": "2px",
                                   "display": "flex",
                                   "alignItems": "center",
                                   "gap": "6px"},
                            children=[
                                html.Span("Dir:", style={
                                    "fontSize": "12px", "color": "#555"}),
                                dcc.Dropdown(
                                    id={"type": "dir-dropdown", "index": i},
                                    options=[
                                        {"label": "Bear", "value": "bearish"},
                                        {"label": "Bull", "value": "bullish"},
                                        {"label": "Neut", "value": "neutral"},
                                    ],
                                    value=e.get("direction", "bearish"),
                                    clearable=False,
                                    style={"flex": "1", "fontSize": "11px"},
                                ),
                            ],
                        ),
                    ],
                )
            )

        if not items:
            return html.P(
                f"No matches (0/{len(events)} events).",
                style={"color": "#888", "fontSize": "11px"},
            )

        header = html.P(
            f"Showing {shown}/{len(events)} events",
            style={"color": "#888", "fontSize": "11px", "marginBottom": "6px"},
        )
        return [header] + items

    # -- pattern-matching: delete / inline edit ----------------------
    @app.callback(
        Output("delete-trigger", "data"),
        [
            Input({"type": "btn-del", "index": ALL}, "n_clicks"),
            Input({"type": "prob-slider", "index": ALL}, "value"),
            Input({"type": "dir-dropdown", "index": ALL}, "value"),
            Input({"type": "evt-start-edit", "index": ALL}, "date"),
            Input({"type": "evt-end-edit", "index": ALL}, "date"),
        ],
        prevent_initial_call=True,
    )
    def handle_inline_actions(
        del_clicks, prob_values, dir_values,
        start_dates, end_dates,
    ):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        prop = ctx.triggered[0]["prop_id"]  # JSON-encoded id
        triggered_value = ctx.triggered[0]["value"]
        try:
            trigger = json.loads(prop.rsplit(".", 1)[0])
        except json.JSONDecodeError:
            raise PreventUpdate

        t = trigger.get("type")
        idx = trigger.get("index")

        if t == "btn-del":
            # Use triggered value directly (idx may not match
            # array position when sidebar is filtered)
            if triggered_value and triggered_value > 0:
                return json.dumps({"action": "delete", "index": idx})
        elif t == "prob-slider":
            return json.dumps({
                "action": "update_prob", "index": idx,
                "value": triggered_value,
            })
        elif t == "dir-dropdown":
            return json.dumps({
                "action": "update_dir", "index": idx,
                "value": triggered_value,
            })
        elif t == "evt-start-edit":
            return json.dumps({
                "action": "update_start", "index": idx,
                "value": str(triggered_value)[:10] if triggered_value else None,
            })
        elif t == "evt-end-edit":
            return json.dumps({
                "action": "update_end", "index": idx,
                "value": str(triggered_value)[:10] if triggered_value else None,
            })

        raise PreventUpdate

    # -- export ------------------------------------------------------
    @app.callback(
        Output("export-status", "children"),
        Input("btn-export", "n_clicks"),
        State("events-store", "data"),
        prevent_initial_call=True,
    )
    def export_events(n_clicks, store_json):
        if not store_json:
            return "No events to export."

        events = json.loads(store_json)
        json_path = os.path.join(output_dir, f"{ticker}_events.json")
        with open(json_path, "w") as f:
            json.dump(events, f, indent=2)

        user_events = [UserEvent.from_dict(d) for d in events]
        dates = pd.to_datetime(sorted(df["date"].unique()))
        presence, factor_names = events_to_presence_matrix(
            user_events, dates,
        )
        npz_path = os.path.join(output_dir, f"{ticker}_presence.npz")
        np.savez(
            npz_path,
            presence=presence,
            factor_names=factor_names,
            dates=dates.strftime("%Y-%m-%d").tolist(),
        )

        return f"Exported {len(events)} events -> {output_dir}/"

    return app
