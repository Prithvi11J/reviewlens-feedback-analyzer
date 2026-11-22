"""Weekly report: the full written briefing, downloadable as Markdown."""

import streamlit as st

from reviewlens.config import ConfigError, get_api_key
from reviewlens.report import generate_report
from ui import state

st.title("Weekly report")

if state.report_markdown() is None:
    st.write("The report is written from this week's exact numbers, in one AI call.")
    try:
        get_api_key()
        if st.button("Write the weekly report", type="primary", icon=":material/edit_note:", key="make_report"):
            with st.spinner("Writing the report..."):
                report = generate_report(state.stats())
            state.set_report(report.markdown, report.insights)
            st.rerun()
    except ConfigError:
        st.warning("Add an OpenAI API key to .env to write reports for your own data.", icon=":material/key:")
else:
    md = state.report_markdown()
    # Shrink headings by two levels so the report sits inside the page's own title.
    st.markdown(md.replace("\n## ", "\n#### ").replace("# ", "### ", 1), unsafe_allow_html=True)
    st.download_button("Download report (Markdown)", md.encode("utf-8"), "weekly_report.md",
                       mime="text/markdown", icon=":material/download:")
