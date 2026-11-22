"""ReviewLens Streamlit app.

Run from the project root:
    streamlit run app.py

How Streamlit works: the WHOLE script re-runs from top to bottom every time
the user clicks or types anything. Anything that must survive a re-run
(analysed reviews, the report) lives in st.session_state, so a filter click
never re-calls the API.
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from reviewlens.aggregate import compute_stats
from reviewlens.analyze import STATUS_OK, analyze_reviews
from reviewlens.charts import CATEGORY_LABELS, category_chart, sentiment_chart
from reviewlens.config import ConfigError, get_api_key, get_model_name
from reviewlens.data import InvalidCSVError, load_reviews
from reviewlens.report import generate_report
from reviewlens.schemas import CATEGORIES, SENTIMENTS

DEMO_DIR = Path(__file__).parent / "data" / "demo"
SAMPLE_CSV = Path(__file__).parent / "data" / "sample" / "reviews_week47.csv"
TABLE_COLUMNS = ["review_id", "date", "platform", "rating", "urgency", "sentiment", "category",
                 "language", "review_text", "summary", "suggested_reply", "analysis_status"]

st.set_page_config(page_title="ReviewLens", page_icon="🔍", layout="wide")


def has_api_key() -> bool:
    try:
        get_api_key()
        return True
    except ConfigError:
        return False


def reset_results(source: str) -> None:
    st.session_state.update(source=source, enriched=None, report_md=None)


def to_excel_csv(df: pd.DataFrame) -> bytes:
    """utf-8-sig adds a BOM so Excel shows Tamil text correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.title("🔍 ReviewLens")
    st.caption("Turn a week of customer reviews into tagged insights and a 5-minute report.")

    uploaded = st.file_uploader(
        "Upload reviews CSV", type="csv", key="upload",
        help="Columns: review_id, date, platform, rating, review_text",
    )
    st.download_button("Download a sample CSV", SAMPLE_CSV.read_bytes(), "reviews_week47.csv",
                       mime="text/csv", width="stretch")
    st.divider()
    demo_clicked = st.button("▶ Explore the demo week (no API key needed)", key="demo", width="stretch")
    # ?demo=1 in the URL opens straight into the demo: handy for sharing a link.
    demo_link = st.query_params.get("demo") == "1" and st.session_state.get("source") is None
    if demo_clicked or demo_link:
        reset_results("demo")
        st.session_state.enriched = pd.read_csv(DEMO_DIR / "reviews_week47_enriched.csv",
                                                encoding="utf-8-sig", dtype={"review_id": str})
        st.session_state.report_md = (DEMO_DIR / "weekly_report.md").read_text(encoding="utf-8")

    st.divider()
    st.caption(f"Model: `{get_model_name()}` · API key: {'✅ found' if has_api_key() else '❌ missing'}")

# ---------------------------------------------------------------- upload + analyse
if uploaded is not None:
    signature = f"upload:{uploaded.name}:{uploaded.size}"
    if st.session_state.get("source") != signature:
        reset_results(signature)  # a new file clears the old results

    try:
        reviews, validation = load_reviews(uploaded)
    except (InvalidCSVError, pd.errors.ParserError, UnicodeDecodeError) as err:
        st.error(f"Could not read this file: {err}")
        st.stop()

    msg = (f"**{uploaded.name}**: {validation.rows_in_file} rows, **{validation.rows_kept} ready to analyse** "
           f"({validation.empty_text_dropped} empty and {validation.duplicates_dropped} duplicate rows removed)")
    st.info(msg + "".join(f"  \n⚠️ {n}" for n in validation.notes))

    if st.session_state.get("enriched") is None:
        if not has_api_key():
            st.error("No OpenAI API key found. Add OPENAI_API_KEY to your .env file, "
                     "or try the demo week from the sidebar.")
            st.stop()
        if st.button(f"Analyse {validation.rows_kept} reviews", type="primary", key="analyse"):
            bar = st.progress(0.0, text="Starting...")
            enriched = analyze_reviews(
                reviews, on_progress=lambda done, total: bar.progress(done / total, text=f"Analysed {done} of {total} reviews"))
            bar.empty()
            st.session_state.enriched = enriched
        else:
            st.stop()

enriched: pd.DataFrame | None = st.session_state.get("enriched")
if enriched is None:
    st.header("Customer Feedback Analyzer")
    st.markdown(
        "1. **Upload** a CSV of reviews (English, Tamil or Tanglish), or explore the **demo week** from the sidebar.\n"
        "2. Every review is tagged with **sentiment, category, urgency**, an English **summary** and a **suggested reply**.\n"
        "3. Get a **dashboard**, a **weekly report** and an **Excel-ready CSV**."
    )
    st.stop()

# ---------------------------------------------------------------- results
stats = compute_stats(enriched)
if st.session_state.get("source") == "demo":
    st.caption("📦 Showing pre-computed results for the demo week (16 to 22 Nov 2026, synthetic data).")

failed = int((enriched["analysis_status"] != STATUS_OK).sum())
if failed:
    st.warning(f"{failed} review(s) could not be analysed and are marked 'needs_review' in the table.")

tab_dash, tab_reviews, tab_report = st.tabs(["📊 Dashboard", "📝 Reviews", "📄 Weekly report"])

with tab_dash:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews analysed", stats.total_reviews)
    c2.metric("Negative", f"{stats.sentiment_pct['negative']}%")
    c3.metric("🚨 High urgency", stats.urgency_counts["high"], help="Reply within 24 hours")
    c4.metric("Average rating", f"{stats.average_rating:.1f} / 5" if stats.average_rating else "-")

    st.subheader("Sentiment split")
    st.plotly_chart(sentiment_chart(stats), width="stretch", config={"displayModeBar": False})
    st.subheader("Reviews by category")
    st.plotly_chart(category_chart(enriched), width="stretch", config={"displayModeBar": False})

with tab_reviews:
    high_only = st.toggle("🚨 Show only HIGH urgency reviews (reply within 24 hours)", key="high_only")
    f2, f3, f4 = st.columns(3)
    cats = f2.multiselect("Category", CATEGORIES, format_func=CATEGORY_LABELS.get, key="f_cat")
    sents = f3.multiselect("Sentiment", SENTIMENTS, key="f_sent")
    plats = f4.multiselect("Platform", sorted(enriched["platform"].dropna().unique()), key="f_plat")

    view = enriched
    if high_only:
        view = view[view["urgency"] == "high"]
    if cats:
        view = view[view["category"].isin(cats)]
    if sents:
        view = view[view["sentiment"].isin(sents)]
    if plats:
        view = view[view["platform"].isin(plats)]

    st.caption(f"Showing {len(view)} of {len(enriched)} reviews")
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
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Download filtered rows (CSV)", to_excel_csv(view), "reviews_filtered.csv",
                       mime="text/csv", width="stretch")
    d2.download_button("⬇ Download all enriched reviews (CSV)", to_excel_csv(enriched), "reviews_enriched.csv",
                       mime="text/csv", width="stretch")

with tab_report:
    if st.session_state.get("report_md") is None:
        st.write("The report is written from the exact numbers above, in one AI call.")
        if st.button("Generate weekly report", type="primary", key="make_report"):
            with st.spinner("Writing the report..."):
                st.session_state.report_md = generate_report(stats).markdown
    if st.session_state.get("report_md"):
        # Shrink headings by two levels so the report fits inside the app's own layout.
        st.markdown(st.session_state.report_md.replace("\n## ", "\n#### ").replace("# ", "### ", 1),
                    unsafe_allow_html=True)
        st.download_button("⬇ Download report (Markdown)", st.session_state.report_md.encode("utf-8"),
                           "weekly_report.md", mime="text/markdown")
