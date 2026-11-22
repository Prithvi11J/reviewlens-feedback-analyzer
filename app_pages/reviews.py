"""All reviews: the enriched table with filters and Excel-ready downloads."""

import streamlit as st

from reviewlens.charts import CATEGORY_LABELS
from reviewlens.schemas import CATEGORIES, SENTIMENTS
from ui import state

TABLE_COLUMNS = ["review_id", "date", "outlet", "platform", "rating", "urgency", "sentiment", "category",
                 "language", "review_text", "summary", "suggested_reply", "analysis_status"]

st.title("All reviews")
df = state.enriched()

high_only = st.toggle("Show only HIGH urgency reviews (reply within 24 hours)", key="high_only")
filters = st.columns(4 if state.has_outlets() else 3)
cats = filters[0].multiselect("Category", CATEGORIES, format_func=CATEGORY_LABELS.get, key="f_cat")
sents = filters[1].multiselect("Sentiment", SENTIMENTS, key="f_sent")
plats = filters[2].multiselect("Platform", sorted(df["platform"].dropna().unique()), key="f_plat")
outs = filters[3].multiselect("Outlet", sorted(df["outlet"].dropna().unique()), key="f_out") \
    if state.has_outlets() else []

view = df
if high_only:
    view = view[view["urgency"] == "high"]
for column, chosen in [("category", cats), ("sentiment", sents), ("platform", plats), ("outlet", outs)]:
    if chosen:
        view = view[view[column].isin(chosen)]

st.caption(f"Showing {len(view)} of {len(df)} reviews")
st.dataframe(
    view[[c for c in TABLE_COLUMNS if c in view.columns]],
    width="stretch", hide_index=True, height=480,
    column_config={
        "review_text": st.column_config.TextColumn("Review", width="large"),
        "summary": st.column_config.TextColumn("Summary (EN)", width="large"),
        "suggested_reply": st.column_config.TextColumn("Suggested reply", width="large"),
        "rating": st.column_config.NumberColumn("★", format="%d"),
    },
)
with st.container(horizontal=True):
    st.download_button("Download filtered rows (CSV)", state.to_excel_csv(view), "reviews_filtered.csv",
                       mime="text/csv", icon=":material/download:")
    st.download_button("Download all enriched reviews (CSV)", state.to_excel_csv(df), "reviews_enriched.csv",
                       mime="text/csv", icon=":material/download:")
