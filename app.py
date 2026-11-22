"""ReviewLens for PJ's: entry point.

Run from the project root:
    streamlit run app.py

This file only sets up the frame (page config, logo, sidebar, navigation).
Each screen is its own script in app_pages/, and shared state lives in ui/state.py.
"""

from pathlib import Path

import streamlit as st

from reviewlens.config import ConfigError, get_api_key, get_model_name, get_restaurant_name
from ui import state

ASSETS = Path(__file__).parent / "assets"  # absolute, so the app works from any working directory

st.set_page_config(page_title=f"{get_restaurant_name()} · ReviewLens", page_icon=str(ASSETS / "pj_mark.svg"),
                   layout="wide", initial_sidebar_state="collapsed")  # navigation lives at the top
st.logo(str(ASSETS / "pj_logo.svg"), icon_image=str(ASSETS / "pj_mark.svg"), size="large")

state.ensure_data()

pages = [
    st.Page("app_pages/briefing.py", title="Briefing", icon=":material/wb_sunny:", default=True),
    st.Page("app_pages/replies.py", title="Reply queue", icon=":material/inbox:"),
    st.Page("app_pages/insights.py", title="Insights", icon=":material/insights:"),
    st.Page("app_pages/reviews.py", title="All reviews", icon=":material/table_rows:"),
    st.Page("app_pages/report.py", title="Weekly report", icon=":material/description:"),
    st.Page("app_pages/upload.py", title="Analyse a week", icon=":material/upload_file:"),
]
page = st.navigation(pages, position="top")

with st.sidebar:
    st.caption("ReviewLens turns a week of customer reviews into tagged insights, "
               "ready-to-post replies and a 5-minute briefing.")
    if state.is_demo():
        st.badge("Demo week · synthetic reviews", icon=":material/science:", color="orange")
    try:
        get_api_key()
        key_status = "API key found"
    except ConfigError:
        key_status = "No API key: demo only"
    st.caption(f"Model `{get_model_name()}` · {key_status}")

page.run()
