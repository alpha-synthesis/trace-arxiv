# -*- coding: utf-8 -*-
"""
Interactive Generation Editor  (Point 2 — Pre-Generation).

A Dash-based web application launched **after FIN training** and
**before synthetic data generation**.  It lets the user:

1. **Toggle** trained causal factors on/off.
2. **Adjust** occurrence probability per factor via sliders.
3. **Set** temporal position (start/end dates) for each factor.
4. **Preview** the resulting factor schedule overlaid on the
   historical price chart.
5. **Export** the edited configuration for the generator.

Launched automatically by the pipeline when
``interactive.pre_generation: true`` in the YAML config,
or programmatically::

    from sde_causal_generator.generation_editor import (
        create_generation_editor_app,
    )
    app = create_generation_editor_app(
        df, ticker="AAPL",
        impact_matrix=impact_matrix,
    )
    app.run(port=8051)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from .data_structures import FACTOR_CATEGORIES, ImpactMatrix


# ══════════════════════════════════════════════════════════════════════
# CSS constants (shared style)
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
    "padding": "10px 12px",
    "marginBottom": "8px",
    "borderRadius": "6px",
    "fontSize": "12px",
}


# ══════════════════════════════════════════════════════════════════════
# Helper: build a presence matrix from user-edited factor configs
# ══════════════════════════════════════════════════════════════════════


def _build_edited_presence(
    factor_configs: List[dict],
    dates: pd.DatetimeIndex,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build a (T, K_active) binary presence matrix from the factor
    configurations exported by the generation editor.

    Each element in *factor_configs* is expected to have:
        name, active (bool), probability (0-1),
        start_date (str), end_date (str), direction (str).

    Only factors with ``active == True`` are included.
    """
    active = [f for f in factor_configs if f.get("active", True)]
    if not active:
        return np.zeros((len(dates), 0), dtype=np.float32), []

    factor_names = [f["name"] for f in active]
    K = len(factor_names)
    T = len(dates)
    presence = np.zeros((T, K), dtype=np.float32)
    date_to_row = {d: i for i, d in enumerate(dates)}

    for k, fc in enumerate(active):
        start = pd.Timestamp(fc["start_date"])
        end = pd.Timestamp(fc["end_date"])
        for d in dates:
            if start <= d <= end:
                presence[date_to_row[d], k] = 1.0

    return presence, factor_names


# ══════════════════════════════════════════════════════════════════════
# Plotly chart with factor schedule preview
# ══════════════════════════════════════════════════════════════════════


def _preview_chart(
    df: pd.DataFrame,
    factor_configs: List[dict],
    ticker: str = "",
    log_scale: bool = False,
) -> go.Figure:
    """Candlestick + factor schedule rectangles for preview."""
    # Robust data preparation
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

    dir_colors = {
        "bearish": ("rgba(255,50,50,{a})", "#C62828"),
        "bullish": ("rgba(50,200,50,{a})", "#2E7D32"),
        "neutral": ("rgba(200,200,50,{a})", "#F9A825"),
    }

    active = [fc for fc in factor_configs if fc.get("active", True)]
    n_active = len(active)
    show_annotations = n_active <= 40

    for fc in active:
        direction = fc.get("direction", "neutral")
        fill_tpl, line_col = dir_colors.get(direction, dir_colors["neutral"])
        alpha = 0.12 * fc.get("probability", 1.0)
        fill = fill_tpl.format(a=alpha)

        kwargs = dict(
            x0=fc["start_date"], x1=fc["end_date"],
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
                    f"{fc['name']} (P={fc.get('probability', 1.0):.0%})"
                ),
                annotation_position="top left",
                annotation_font_size=9,
            )
        fig.add_vrect(**kwargs)

    fig.update_layout(
        title=(
            f"{ticker} — Generation Factor Schedule"
            if ticker else "Generation Factor Schedule"
        ),
        yaxis_title="Price",
        yaxis_type="log" if log_scale else "linear",
        xaxis_rangeslider_visible=True,
        template="plotly_white",
        dragmode="zoom",
        autosize=True,
        margin=dict(l=50, r=50, t=50, b=40),
    )
    if has_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════
# Dash Application — Pre-Generation Editor  (Point 2)
# ══════════════════════════════════════════════════════════════════════


def create_generation_editor_app(
    df: pd.DataFrame,
    ticker: str = "",
    impact_matrix: Optional[ImpactMatrix] = None,
    historical_presence: Optional[np.ndarray] = None,
    output_dir: str = "cache/generation_config",
    port: int = 8051,
    validated_directions: Optional[Dict[str, str]] = None,
):
    """
    Dash app for editing which trained factors are used during
    synthetic data generation, and with what temporal positioning.

    Parameters
    ----------
    df : pd.DataFrame
        Historical price data (date, open, high, low, close, volume).
    ticker : str
        Ticker symbol for display.
    impact_matrix : ImpactMatrix | None
        Trained impact matrix with factor_names and occurrence_prob.
        If None, starts with an empty factor list.
    historical_presence : np.ndarray | None
        Original (T, K) presence matrix from training.
    output_dir : str
        Where to save the exported generation config.
    port : int
        Port for the Dash app (default 8051).
    validated_directions : dict[str, str] | None
        Mapping factor_name -> direction ("bullish"/"bearish"/"neutral")
        from the user-validated events in Point 1. When available,
        these take priority over the FIN-inferred direction.
    """
    try:
        from dash import (
            Dash, Input, Output, State, ALL,
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
        title=f"Generation Editor — {ticker}",
        suppress_callback_exceptions=True,
    )

    # -- initialise factor configs from impact matrix ----------------
    min_date = str(df["date"].min())[:10]
    max_date = str(df["date"].max())[:10]
    dates = pd.to_datetime(sorted(df["date"].unique()))

    initial_factors: List[dict] = []
    if impact_matrix is not None:
        _val_dirs = validated_directions or {}
        for k, fname in enumerate(impact_matrix.factor_names):
            # Priority: user-validated direction > FIN-inferred
            if fname in _val_dirs:
                direction = _val_dirs[fname]
            else:
                # Fallback: infer from base_impact on close only
                # (index 3 = close in OHLCV), not mean across all
                # features which can be misleading.
                n_features = impact_matrix.base_impact.shape[1]
                close_idx = min(3, n_features - 1)
                close_impact = float(
                    impact_matrix.base_impact[k, close_idx]
                )
                direction = "bearish" if close_impact < 0 else "bullish"

            # Infer temporal window from historical presence
            start_d, end_d = min_date, max_date
            if historical_presence is not None and k < historical_presence.shape[1]:
                active_rows = np.where(historical_presence[:, k] > 0)[0]
                if len(active_rows) > 0:
                    start_d = str(dates[active_rows[0]])[:10]
                    end_d = str(dates[active_rows[-1]])[:10]

            initial_factors.append({
                "name": fname,
                "active": True,
                "probability": float(
                    impact_matrix.occurrence_prob[k]
                    if impact_matrix.occurrence_prob is not None
                    else 1.0
                ),
                "start_date": start_d,
                "end_date": end_d,
                "direction": direction,
            })

    factors_json = json.dumps(initial_factors)

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
                f"Causal SDE — Generation Editor (Pre-Generation) [{ticker}]",
                style={"color": "#1a1a2e", "margin": "0 0 4px 0",
                       "fontSize": "18px", "flexShrink": "0",
                       "fontWeight": "700"},
            ),
            html.P(
                "Toggle factors on/off, adjust probabilities and "
                "temporal windows, then Export to proceed with generation.",
                style={"color": "#555", "marginBottom": "8px",
                       "fontSize": "13px", "flexShrink": "0"},
            ),

            # --- main: chart | sidebar (fills remaining height) ---
            html.Div(
                style={"display": "flex", "gap": "20px",
                       "flex": "1", "minHeight": "0"},
                children=[
                    # Chart (70 %)
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
                                        id="gen-y-scale-toggle",
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
                                id="gen-chart",
                                style={"flex": "1", "minHeight": "0"},
                                responsive=True,
                                config={
                                    "scrollZoom": True,
                                    "displayModeBar": True,
                                },
                            ),
                        ],
                    ),

                    # Sidebar (30 %)
                    html.Div(
                        style={
                            "flex": "1.2",
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
                            html.H4("Trained Factors",
                                    style={"marginTop": 0}),
                            html.P(
                                f"{len(initial_factors)} factors loaded "
                                f"from ImpactMatrix.",
                                style={"color": "#555", "fontSize": "12px"},
                            ),
                            dcc.Input(
                                id="gen-search", type="text",
                                placeholder="Search factors...",
                                debounce=True,
                                style={"width": "100%", "marginBottom": "6px",
                                       "padding": "8px", "borderRadius": "6px",
                                       "border": "1px solid #ccc",
                                       "backgroundColor": "#f9f9f9",
                                       "color": "#222", "fontSize": "13px"},
                            ),
                            dcc.Dropdown(
                                id="gen-filter-direction",
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
                                id="gen-filter-category",
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
                            html.Hr(),
                            html.Div(id="factor-list"),
                            html.Hr(),
                            html.Button(
                                "Export & Continue",
                                id="btn-gen-export", n_clicks=0,
                                style={**_BTN, "backgroundColor": "#1a5276"},
                            ),
                            html.Div(
                                id="gen-export-status",
                                style={"color": "#2ecc71",
                                       "marginTop": "6px"},
                            ),
                        ],
                    ),
                ],
            ),

            # hidden stores
            dcc.Store(id="factors-store", data=factors_json),
            dcc.Store(id="factor-edit-trigger", data=""),
        ],
    )

    # ================================================================
    # Callbacks
    # ================================================================

    # -- factor list rendering with inline controls ------------------
    @app.callback(
        Output("factor-list", "children"),
        [
            Input("factors-store", "data"),
            Input("gen-search", "value"),
            Input("gen-filter-direction", "value"),
            Input("gen-filter-category", "value"),
        ],
    )
    def render_factor_list(store_json, search_text,
                           filter_direction, filter_category):
        if not store_json:
            return html.P("No factors.", style={"color": "#666"})

        factors = json.loads(store_json)
        if not factors:
            return html.P("No factors.", style={"color": "#666"})

        search_lower = (search_text or "").strip().lower()

        dir_colors = {
            "bearish": "#C62828",
            "bullish": "#2E7D32",
            "neutral": "#F9A825",
        }

        items = []
        shown = 0
        for i, fc in enumerate(factors):
            # Apply direction filter
            if filter_direction and filter_direction != "all" and fc.get("direction") != filter_direction:
                continue
            # Apply category filter
            if filter_category and filter_category != "all" and fc.get("category") != filter_category:
                continue
            # Apply search filter
            if search_lower:
                searchable = fc.get("name", "").lower()
                if search_lower not in searchable:
                    continue
            shown += 1
            active = fc.get("active", True)
            direction = fc.get("direction", "neutral")
            bg = "#eef2f7" if active else "#f0f0f0"
            opacity = "1" if active else "0.4"

            items.append(
                html.Div(
                    style={
                        **_CARD,
                        "backgroundColor": bg,
                        "borderLeft": f"3px solid {dir_colors.get(direction, '#555')}",
                        "opacity": opacity,
                    },
                    children=[
                        # Row 1: toggle + name
                        html.Div(
                            style={"display": "flex",
                                   "alignItems": "center",
                                   "gap": "8px"},
                            children=[
                                dcc.Checklist(
                                    id={"type": "factor-toggle", "index": i},
                                    options=[{"label": "", "value": "on"}],
                                    value=["on"] if active else [],
                                    style={"marginRight": "4px"},
                                ),
                                html.B(
                                    fc["name"],
                                    style={"fontSize": "13px", "color": "#1a1a2e"},
                                ),
                            ],
                        ),
                        # Row 2: probability slider
                        html.Div(
                            style={"marginTop": "4px",
                                   "display": "flex",
                                   "alignItems": "center",
                                   "gap": "6px"},
                            children=[
                                html.Span("P:", style={
                                    "fontSize": "12px", "color": "#555"}),
                                dcc.Slider(
                                    id={"type": "factor-prob", "index": i},
                                    min=0, max=1, step=0.05,
                                    value=fc.get("probability", 1.0),
                                    marks={0: "0", 0.5: ".5", 1: "1"},
                                    tooltip={
                                        "placement": "bottom",
                                        "always_visible": False,
                                    },
                                    disabled=not active,
                                ),
                            ],
                        ),
                        # Row 3: direction dropdown
                        html.Div(
                            style={"marginTop": "2px",
                                   "display": "flex",
                                   "alignItems": "center",
                                   "gap": "6px"},
                            children=[
                                html.Span("Dir:", style={
                                    "fontSize": "12px", "color": "#555"}),
                                dcc.Dropdown(
                                    id={"type": "factor-dir", "index": i},
                                    options=[
                                        {"label": "Bear", "value": "bearish"},
                                        {"label": "Bull", "value": "bullish"},
                                        {"label": "Neut", "value": "neutral"},
                                    ],
                                    value=direction,
                                    clearable=False,
                                    disabled=not active,
                                    style={"flex": "1", "fontSize": "11px"},
                                ),
                            ],
                        ),
                        # Row 4: date range
                        html.Div(
                            style={"marginTop": "4px",
                                   "display": "flex",
                                   "alignItems": "center",
                                   "gap": "4px",
                                   "flexWrap": "wrap"},
                            children=[
                                html.Span("From:", style={
                                    "fontSize": "10px", "color": "#aaa"}),
                                dcc.DatePickerSingle(
                                    id={"type": "factor-start", "index": i},
                                    date=fc.get("start_date", min_date),
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    disabled=not active,
                                    style={"fontSize": "10px"},
                                ),
                                html.Span("To:", style={
                                    "fontSize": "10px", "color": "#aaa"}),
                                dcc.DatePickerSingle(
                                    id={"type": "factor-end", "index": i},
                                    date=fc.get("end_date", max_date),
                                    min_date_allowed=min_date,
                                    max_date_allowed=max_date,
                                    disabled=not active,
                                    style={"fontSize": "10px"},
                                ),
                            ],
                        ),
                    ],
                )
            )

        if not items:
            return html.P(
                f"No matches (0/{len(factors)} factors).",
                style={"color": "#888", "fontSize": "11px"},
            )

        header = html.P(
            f"Showing {shown}/{len(factors)} factors",
            style={"color": "#888", "fontSize": "11px", "marginBottom": "6px"},
        )
        return [header] + items

    # -- collect inline edits via pattern-matching -------------------
    @app.callback(
        Output("factor-edit-trigger", "data"),
        [
            Input({"type": "factor-toggle", "index": ALL}, "value"),
            Input({"type": "factor-prob", "index": ALL}, "value"),
            Input({"type": "factor-dir", "index": ALL}, "value"),
            Input({"type": "factor-start", "index": ALL}, "date"),
            Input({"type": "factor-end", "index": ALL}, "date"),
        ],
        prevent_initial_call=True,
    )
    def handle_factor_edits(
        toggles, probs, dirs, starts, ends,
    ):
        ctx = callback_context
        if not ctx.triggered:
            raise PreventUpdate
        prop = ctx.triggered[0]["prop_id"]
        try:
            trigger = json.loads(prop.rsplit(".", 1)[0])
        except json.JSONDecodeError:
            raise PreventUpdate

        t = trigger.get("type")
        idx = trigger.get("index")

        if t == "factor-toggle":
            active = "on" in (toggles[idx] or [])
            return json.dumps({
                "action": "toggle", "index": idx, "value": active,
            })
        elif t == "factor-prob":
            return json.dumps({
                "action": "update_prob", "index": idx,
                "value": probs[idx],
            })
        elif t == "factor-dir":
            return json.dumps({
                "action": "update_dir", "index": idx,
                "value": dirs[idx],
            })
        elif t == "factor-start":
            return json.dumps({
                "action": "update_start", "index": idx,
                "value": str(starts[idx])[:10] if starts[idx] else None,
            })
        elif t == "factor-end":
            return json.dumps({
                "action": "update_end", "index": idx,
                "value": str(ends[idx])[:10] if ends[idx] else None,
            })

        raise PreventUpdate

    # -- apply edits to store ----------------------------------------
    @app.callback(
        Output("factors-store", "data"),
        Input("factor-edit-trigger", "data"),
        State("factors-store", "data"),
        prevent_initial_call=True,
    )
    def apply_factor_edit(trigger_json, store_json):
        if not trigger_json:
            raise PreventUpdate

        try:
            payload = json.loads(trigger_json)
        except json.JSONDecodeError:
            raise PreventUpdate

        factors = json.loads(store_json) if store_json else []
        action = payload.get("action")
        idx = payload.get("index")
        value = payload.get("value")

        if idx is None or idx < 0 or idx >= len(factors):
            raise PreventUpdate

        if action == "toggle":
            factors[idx]["active"] = bool(value)
        elif action == "update_prob":
            factors[idx]["probability"] = value
        elif action == "update_dir":
            factors[idx]["direction"] = value
        elif action == "update_start" and value:
            factors[idx]["start_date"] = value
        elif action == "update_end" and value:
            factors[idx]["end_date"] = value
        else:
            raise PreventUpdate

        return json.dumps(factors)

    # -- chart preview (responds to search so plot matches sidebar) -
    @app.callback(
        Output("gen-chart", "figure"),
        [
            Input("factors-store", "data"),
            Input("gen-search", "value"),
            Input("gen-filter-direction", "value"),
            Input("gen-filter-category", "value"),
            Input("gen-y-scale-toggle", "value"),
        ],
    )
    def update_gen_chart(store_json, search_text,
                         filter_direction, filter_category, y_scale):
        factor_configs = []
        if store_json:
            all_factors = json.loads(store_json)
            search_lower = (search_text or "").strip().lower()
            for fc in all_factors:
                if filter_direction and filter_direction != "all" and fc.get("direction") != filter_direction:
                    continue
                if filter_category and filter_category != "all" and fc.get("category") != filter_category:
                    continue
                if search_lower:
                    if search_lower not in fc.get("name", "").lower():
                        continue
                factor_configs.append(fc)
        return _preview_chart(
            df, factor_configs, ticker=ticker,
            log_scale=(y_scale == "log"),
        )

    # -- export configuration ----------------------------------------
    @app.callback(
        Output("gen-export-status", "children"),
        Input("btn-gen-export", "n_clicks"),
        State("factors-store", "data"),
        prevent_initial_call=True,
    )
    def export_gen_config(n_clicks, store_json):
        if not store_json:
            return "No factors to export."

        factors = json.loads(store_json)
        active_factors = [f for f in factors if f.get("active", True)]

        # Save full config
        config_path = os.path.join(
            output_dir, f"{ticker}_generation_config.json",
        )
        with open(config_path, "w") as f:
            json.dump(factors, f, indent=2)

        # Save edited presence matrix
        presence, factor_names = _build_edited_presence(
            active_factors, dates,
        )
        npz_path = os.path.join(
            output_dir, f"{ticker}_generation_presence.npz",
        )
        np.savez(
            npz_path,
            presence=presence,
            factor_names=factor_names,
            dates=dates.strftime("%Y-%m-%d").tolist(),
        )

        return (
            f"Exported {len(active_factors)}/{len(factors)} active "
            f"factors -> {output_dir}/"
        )

    return app
