"""
analytics/charts.py — Plotly chart builder.

Produces JSON-serialised Plotly figures for bar, pie, and line/histogram
charts based on the column types present in the uploaded dataset.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
import plotly
import plotly.express as px

# ── Visual constants ───────────────────────────────────────────────────────────

# Brand palette (matches the front-end JS palette)
PALETTE = [
    "#6366f1", "#8b5cf6", "#06b6d4", "#10b981",
    "#f59e0b", "#ec4899", "#3b82f6", "#a78bfa",
]

# Base Plotly layout applied to every chart
_LAYOUT_BASE: dict[str, Any] = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#e2e8f0",
    title_font_size=15,
    colorway=PALETTE,
    margin=dict(l=40, r=20, t=50, b=60),
)

# Axis overrides for x/y charts
_AXIS_STYLE: dict[str, Any] = dict(
    gridcolor="rgba(255,255,255,0.07)",
    linecolor="rgba(255,255,255,0.07)",
    tickfont_color="#64748b",
    tickfont_size=11,
)

# Max categories shown in bar/pie charts (keeps charts readable)
_MAX_CATEGORIES = 15
_MAX_PIE_SLICES = 10

# Max data points for the line chart (performance guard for large CSVs)
_MAX_LINE_POINTS = 300


def _to_json(fig: Any) -> str:
    """Serialise a Plotly figure to a JSON string."""
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _build_bar_chart(
    df: pd.DataFrame,
    cat_col: str,
    num_col: str,
) -> str:
    """
    Bar chart: aggregate `num_col` values grouped by the top N categories
    in `cat_col`.

    Args:
        df:      Source DataFrame.
        cat_col: Categorical column for the x-axis.
        num_col: Numeric column for the y-axis.

    Returns:
        JSON string of the Plotly figure.
    """
    top_cats = df[cat_col].value_counts().head(_MAX_CATEGORIES).index
    plot_df  = df[df[cat_col].isin(top_cats)]

    fig = px.bar(
        plot_df,
        x=cat_col,
        y=num_col,
        title=f"{num_col} by {cat_col}",
        color=cat_col,
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(**_LAYOUT_BASE, showlegend=False)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return _to_json(fig)


def _build_pie_chart(df: pd.DataFrame, cat_col: str) -> str:
    """
    Pie chart: distribution of the top N values in `cat_col`.

    Args:
        df:      Source DataFrame.
        cat_col: Categorical column to visualise.

    Returns:
        JSON string of the Plotly figure.
    """
    counts = df[cat_col].value_counts().head(_MAX_PIE_SLICES)

    fig = px.pie(
        values=counts.values,
        names=counts.index,
        title=f"Distribution of {cat_col}",
        color_discrete_sequence=PALETTE,
    )
    pie_layout = {**_LAYOUT_BASE, "margin": dict(l=20, r=20, t=50, b=20)}
    fig.update_layout(**pie_layout)
    return _to_json(fig)


def _build_line_chart(df: pd.DataFrame, x_col: str, y_col: str) -> str:
    """
    Line chart plotting `y_col` over `x_col` (first N rows after sorting).

    Args:
        df:    Source DataFrame.
        x_col: Numeric column for the x-axis.
        y_col: Numeric column for the y-axis.

    Returns:
        JSON string of the Plotly figure.
    """
    plot_df = (
        df[[x_col, y_col]]
        .dropna()
        .sort_values(x_col)
        .head(_MAX_LINE_POINTS)
    )

    fig = px.line(
        plot_df,
        x=x_col,
        y=y_col,
        title=f"{y_col} over {x_col}",
        color_discrete_sequence=["#6366f1"],
    )
    fig.update_layout(**_LAYOUT_BASE)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return _to_json(fig)


def _build_histogram(df: pd.DataFrame, col: str) -> str:
    """
    Histogram for a single numeric column (used when only one numeric column
    exists, as a stand-in for the line chart slot).

    Args:
        df:  Source DataFrame.
        col: Numeric column to histogram.

    Returns:
        JSON string of the Plotly figure.
    """
    fig = px.histogram(
        df,
        x=col,
        title=f"Distribution of {col}",
        color_discrete_sequence=["#6366f1"],
        nbins=30,
    )
    fig.update_layout(**_LAYOUT_BASE)
    fig.update_xaxes(**_AXIS_STYLE)
    fig.update_yaxes(**_AXIS_STYLE)
    return _to_json(fig)


def build_charts(
    df: pd.DataFrame,
    numeric_cols: list[str],
    categorical_cols: list[str],
) -> dict[str, str]:
    """
    Build all available charts for the dashboard.

    Produces up to three chart types depending on what column types are
    available:
      - "bar"  → requires at least one categorical and one numeric column
      - "pie"  → requires at least one categorical column
      - "line" → requires at least two numeric columns (falls back to a
                 histogram when only one numeric column is present)

    Args:
        df:               Source DataFrame.
        numeric_cols:     List of numeric column names.
        categorical_cols: List of categorical column names.

    Returns:
        Dict mapping chart keys ("bar", "pie", "line") to JSON strings.
    """
    charts: dict[str, str] = {}

    if numeric_cols and categorical_cols:
        charts["bar"] = _build_bar_chart(
            df, cat_col=categorical_cols[0], num_col=numeric_cols[0]
        )

    if categorical_cols:
        charts["pie"] = _build_pie_chart(df, cat_col=categorical_cols[0])

    if len(numeric_cols) >= 2:
        charts["line"] = _build_line_chart(
            df, x_col=numeric_cols[0], y_col=numeric_cols[1]
        )
    elif len(numeric_cols) == 1:
        charts["line"] = _build_histogram(df, col=numeric_cols[0])

    return charts
