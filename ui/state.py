"""Shared app state, used by every page in app_pages/.

Streamlit re-runs the current page script on every click, so anything that must
survive (the analysed reviews, the report) lives in st.session_state. Pages call
these helpers instead of touching session_state keys directly, so the keys are
defined in one place.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from reviewlens.aggregate import WeeklyStats, compute_stats
from reviewlens.replies import build_queue
from reviewlens.report import WeeklyInsights

ROOT = Path(__file__).resolve().parents[1]
QUEUE_PAGE_SIZE = 10  # reply cards shown at a time
DEMO_DIR = ROOT / "data" / "demo"
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_week47.csv"


def load_demo() -> None:
    st.session_state.update(
        source="demo",
        replied={},  # review_id -> the reply text that was posted
        drafts={},  # review_id -> a draft the manager edited
        queue_limit=QUEUE_PAGE_SIZE,
        enriched=pd.read_csv(DEMO_DIR / "reviews_week47_enriched.csv", encoding="utf-8-sig",
                             dtype={"review_id": str}),
        report_md=(DEMO_DIR / "weekly_report.md").read_text(encoding="utf-8"),
        insights=WeeklyInsights.model_validate_json((DEMO_DIR / "weekly_insights.json").read_text(encoding="utf-8")),
    )


def ensure_data() -> None:
    """First visit: open on the demo week, so the app is never an empty screen."""
    if "enriched" not in st.session_state:
        load_demo()


def set_results(source: str, enriched: pd.DataFrame) -> None:
    """A newly analysed week replaces the old one, and its old report is cleared."""
    st.session_state.update(source=source, enriched=enriched, report_md=None, insights=None,
                            replied={}, drafts={}, queue_limit=QUEUE_PAGE_SIZE)


def set_report(markdown: str, insights: WeeklyInsights) -> None:
    st.session_state.update(report_md=markdown, insights=insights)


def enriched() -> pd.DataFrame:
    return st.session_state.enriched


def is_demo() -> bool:
    return st.session_state.get("source") == "demo"


def report_markdown() -> str | None:
    return st.session_state.get("report_md")


def insights() -> WeeklyInsights | None:
    return st.session_state.get("insights")


@st.cache_data(show_spinner=False)
def _stats_cached(df: pd.DataFrame) -> WeeklyStats:
    # st.cache_data hashes the DataFrame's contents, so each distinct table is computed once.
    return compute_stats(df)


def stats(df: pd.DataFrame | None = None) -> WeeklyStats:
    """Weekly numbers for the whole week (or for `df`, e.g. one outlet). Cached per table."""
    return _stats_cached(enriched() if df is None else df)


@st.cache_data(show_spinner=False)
def _queue_cached(df: pd.DataFrame) -> pd.DataFrame:
    return build_queue(df)


def reply_queue() -> pd.DataFrame:
    """All analysed reviews in reply order (see reviewlens/replies.py)."""
    return _queue_cached(enriched())


def replied() -> dict[str, str]:
    return st.session_state.setdefault("replied", {})


def drafts() -> dict[str, str]:
    return st.session_state.setdefault("drafts", {})


def waiting_count() -> int:
    return len(reply_queue()) - len(set(replied()) & set(reply_queue()["review_id"]))


def week_label(s: WeeklyStats) -> str:
    start, end = date.fromisoformat(s.period_start), date.fromisoformat(s.period_end)
    return f"Week {end.isocalendar().week} · {start.day} {start:%b} to {end.day} {end:%b %Y}"


def is_dark() -> bool:
    try:
        return st.context.theme.type == "dark"
    except Exception:  # older runtimes or tests without a browser
        return False


def has_outlets() -> bool:
    return "outlet" in enriched().columns


def to_excel_csv(df: pd.DataFrame) -> bytes:
    """utf-8-sig adds a BOM so Excel shows Tamil text correctly."""
    return df.to_csv(index=False).encode("utf-8-sig")
