"""Insights: the charts, for the whole restaurant or one outlet at a time."""

import streamlit as st

from reviewlens.charts import category_chart, sentiment_chart
from ui import state

st.title("Insights")
st.caption(state.week_label(state.stats()))

df = state.enriched()
if state.has_outlets():
    outlets = sorted(df["outlet"].dropna().unique())
    choice = st.segmented_control("Outlet", ["All outlets", *outlets], default="All outlets", key="insight_outlet")
    if choice and choice != "All outlets":
        df = df[df["outlet"] == choice]

s = state.stats(df)
with st.container(horizontal=True):
    st.metric("Reviews", s.total_reviews, border=True)
    st.metric("Average rating", f"{s.average_rating:.1f} ★" if s.average_rating else "-", border=True)
    st.metric("Negative", f"{s.sentiment_pct['negative']}%", border=True)
    st.metric("High urgency", s.urgency_counts["high"], border=True)

dark = state.is_dark()
st.subheader("Sentiment split")
st.plotly_chart(sentiment_chart(s, dark), width="stretch", config={"displayModeBar": False})
st.subheader("Reviews by category")
st.plotly_chart(category_chart(df, dark), width="stretch", config={"displayModeBar": False})
