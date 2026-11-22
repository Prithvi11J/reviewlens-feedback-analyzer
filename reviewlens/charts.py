"""Plotly charts for the dashboard.

Sentiment is a polarity (good <-> bad), so it uses a DIVERGING scheme from PJ's
palette: peacock blue for positive, chilli red for negative, neutral gray in the
middle. Both pairs were run through a colour-blindness validator, and dark mode
gets its own lighter steps instead of reusing the light colours. Every bar carries
a visible number, so colour is never the only way to read a value.
"""

import pandas as pd
import plotly.graph_objects as go

from reviewlens.aggregate import WeeklyStats
from reviewlens.schemas import CATEGORIES, SENTIMENTS

SENTIMENT_COLORS = {
    "light": {"positive": "#0f78a8", "neutral": "#a39e93", "negative": "#c4432f"},
    "dark": {"positive": "#3d9cbf", "neutral": "#7d776c", "negative": "#e0654f"},
}
CATEGORY_LABELS = {c: c.replace("_", " ").capitalize() for c in CATEGORIES}


def _colors(dark: bool) -> dict[str, str]:
    return SENTIMENT_COLORS["dark" if dark else "light"]


def _base_layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=8, b=8),
        barmode="stack",
        bargap=0.35,
        # traceorder="normal": legend reads Positive, Neutral, Negative, same as the bars
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None, traceorder="normal"),
        hoverlabel=dict(namelength=-1),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, zeroline=False, title=None)
    fig.update_yaxes(showgrid=False, title=None)
    return fig


def sentiment_chart(stats: WeeklyStats, dark: bool = False) -> go.Figure:
    """One 100% bar split into positive / neutral / negative."""
    colors = _colors(dark)
    fig = go.Figure()
    for s in SENTIMENTS:
        pct, count = stats.sentiment_pct[s], stats.sentiment_counts[s]
        fig.add_bar(
            name=s.capitalize(), y=["Reviews"], x=[pct], orientation="h",
            marker=dict(color=colors[s]),
            text=[f"{s.capitalize()} {pct}%" if pct >= 8 else ""], textposition="inside",
            insidetextanchor="middle", textfont=dict(color="#ffffff"),
            hovertemplate=f"<b>{s.capitalize()}</b><br>{count} reviews ({pct}%)<extra></extra>",
        )
    fig = _base_layout(fig, height=130)
    fig.update_xaxes(range=[0, 100], ticksuffix="%", showgrid=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def category_chart(enriched: pd.DataFrame, dark: bool = False) -> go.Figure:
    """Reviews per category, stacked by sentiment, biggest category on top."""
    colors = _colors(dark)
    ok = enriched[enriched["analysis_status"] == "ok"]
    table = (ok.groupby(["category", "sentiment"]).size().unstack(fill_value=0)
             .reindex(columns=list(SENTIMENTS), fill_value=0))
    table = table.loc[table.sum(axis=1).sort_values().index]  # ascending -> largest at the top
    labels = [CATEGORY_LABELS.get(c, c) for c in table.index]

    fig = go.Figure()
    for s in SENTIMENTS:
        values = table[s].tolist()
        fig.add_bar(
            name=s.capitalize(), y=labels, x=values, orientation="h",
            marker=dict(color=colors[s]),
            text=[v if v >= 5 else "" for v in values], textposition="inside", insidetextanchor="middle",
            textfont=dict(color="#ffffff"),
            hovertemplate="<b>%{y}</b><br>" + s.capitalize() + ": %{x} reviews<extra></extra>",
        )
    return _base_layout(fig, height=60 + 42 * len(labels))
