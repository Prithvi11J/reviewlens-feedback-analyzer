"""Smoke tests for the Streamlit app using the demo data (no API calls).

AppTest runs the app headlessly, switches pages, clicks widgets and checks for errors.
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
PAGES = ["app_pages/briefing.py", "app_pages/replies.py", "app_pages/insights.py", "app_pages/reviews.py",
         "app_pages/report.py", "app_pages/upload.py"]


def start() -> AppTest:
    return AppTest.from_file(APP, default_timeout=30).run()


def test_opens_on_briefing_with_demo_week():
    at = start()
    assert not at.exception
    assert at.title[0].value == "Monday briefing"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Reviews this week"] == "240"
    badges = " ".join(md.value for md in at.markdown)
    assert "Fish fry" in badges and "complaints at the Peelamedu outlet" in badges  # fix-first cards


@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders(page):
    at = start()
    at.switch_page(page).run()
    assert not at.exception, page


def test_high_urgency_filter():
    at = start()
    at.switch_page("app_pages/reviews.py").run()
    at.toggle(key="high_only").set_value(True).run()
    assert not at.exception
    assert any(c.startswith("Showing 35 of 240") for c in at.caption.values)


def test_reply_queue_mark_replied_and_undo():
    at = start()
    at.switch_page("app_pages/replies.py").run()
    assert "0 of 240 replied" in at.get("progress")[0].proto.text
    first = at.code[0].value  # the top-priority draft
    assert "Team PJ's" in first

    done_buttons = [b for b in at.button if b.key and b.key.startswith("done_")]
    assert len(done_buttons) == 10  # one page of cards
    review_id = done_buttons[0].key.removeprefix("done_")
    done_buttons[0].click().run()
    assert not at.exception
    assert "1 of 240 replied" in at.get("progress")[0].proto.text
    assert at.session_state["replied"][review_id] == first  # the posted text is recorded

    at.toggle(key="q_show_done").set_value(True).run()
    at.button(key=f"undo_{review_id}").click().run()
    assert "0 of 240 replied" in at.get("progress")[0].proto.text


def test_edited_draft_is_what_gets_recorded():
    at = start()
    at.switch_page("app_pages/replies.py").run()
    review_id = next(b.key for b in at.button if b.key and b.key.startswith("done_")).removeprefix("done_")
    at.text_area(key=f"edit_{review_id}").input("Sorry about this. - Team PJ").run()
    at.button(key=f"save_{review_id}").click().run()
    assert at.code[0].value == "Sorry about this. - Team PJ"
    at.button(key=f"done_{review_id}").click().run()
    assert at.session_state["replied"][review_id] == "Sorry about this. - Team PJ"


def test_public_demo_without_api_key(monkeypatch):
    """Like Streamlit Cloud with no secret set: no uploader, a friendly note instead of an error."""
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = start()
    at.switch_page("app_pages/upload.py").run()
    assert not at.exception and not at.error
    assert len(at.get("file_uploader")) == 0
    assert any("pre-analysed week" in i.value for i in at.info)
    at.switch_page("app_pages/briefing.py").run()  # the demo itself still works fully
    assert not at.exception


def test_insights_outlet_filter():
    at = start()
    at.switch_page("app_pages/insights.py").run()
    at.segmented_control(key="insight_outlet").set_value("Peelamedu").run()
    assert not at.exception
    reviews = next(m.value for m in at.metric if m.label == "Reviews")
    assert reviews != "240"  # filtered to one outlet
