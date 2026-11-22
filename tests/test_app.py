"""Smoke test for the Streamlit app using the demo data (no API calls).

AppTest runs app.py headlessly, clicks widgets and checks for errors.
"""

from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_landing_page_renders():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.header[0].value == "Customer Feedback Analyzer"


def test_demo_week_dashboard_and_high_filter():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="demo").click().run()
    assert not at.exception

    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Reviews analysed"] == "240"

    at.toggle(key="high_only").set_value(True).run()
    assert not at.exception
    shown = at.caption.values
    assert any(c.startswith("Showing 33 of 240") for c in shown)

    assert any("Weekly Feedback Report" in md.value for md in at.markdown)  # report tab is rendered
