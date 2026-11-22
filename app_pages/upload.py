"""Analyse a week: upload a CSV and run the tagging pipeline on it."""

import pandas as pd
import streamlit as st

from reviewlens.analyze import analyze_reviews
from reviewlens.config import ConfigError, get_api_key
from reviewlens.data import OPTIONAL_COLUMNS, REQUIRED_COLUMNS, InvalidCSVError, load_reviews
from ui import state

st.title("Analyse a week")
st.write(f"Upload a CSV with the columns `{', '.join(REQUIRED_COLUMNS)}`. "
         f"Add an optional `{OPTIONAL_COLUMNS[0]}` column to compare branches.")

with st.container(horizontal=True):
    st.download_button("Download a sample CSV", state.SAMPLE_CSV.read_bytes(), "reviews_week47.csv",
                       mime="text/csv", icon=":material/download:")
    if not state.is_demo() and st.button("Back to the demo week", icon=":material/science:", key="back_demo"):
        state.load_demo()
        st.switch_page("app_pages/briefing.py")

try:
    get_api_key()
except ConfigError:
    # The public demo runs without an API key (visitors shouldn't spend the owner's credits).
    st.info("This online demo runs on a pre-analysed week, so it can't analyse new files. To analyse your own "
            f"reviews, run ReviewLens on your computer with your own OpenAI key. "
            f"[The setup guide]({state.REPO_URL}#quick-start) takes about 5 minutes.",
            icon=":material/info:")
    st.stop()

uploaded = st.file_uploader("Reviews CSV", type="csv", key="upload")
if uploaded is None:
    st.stop()

try:
    reviews, validation = load_reviews(uploaded)
except (InvalidCSVError, pd.errors.ParserError, UnicodeDecodeError) as err:
    st.error(f"Couldn't read this file. {err}", icon=":material/error:")
    st.stop()

st.info(f"**{uploaded.name}**: {validation.rows_in_file} rows, **{validation.rows_kept} ready to analyse** "
        f"({validation.empty_text_dropped} empty and {validation.duplicates_dropped} duplicate rows removed)"
        + "".join(f"  \n{n}" for n in validation.notes), icon=":material/fact_check:")

if st.button(f"Analyse {validation.rows_kept} reviews", type="primary", icon=":material/auto_awesome:",
             key="analyse"):
    bar = st.progress(0.0, text="Starting...")
    result = analyze_reviews(reviews, on_progress=lambda done, total: bar.progress(
        done / total, text=f"Analysed {done} of {total} reviews"))
    bar.empty()
    state.set_results(f"upload:{uploaded.name}", result)
    st.success("Reviews analysed. Write the weekly report next, then open the briefing.",
               icon=":material/check_circle:")
    st.page_link("app_pages/report.py", label="Write the weekly report", icon=":material/arrow_forward:")
