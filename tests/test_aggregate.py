"""Offline tests: the report numbers must be exactly right."""

import pandas as pd
import pytest

from reviewlens.aggregate import compute_stats, percentages_that_sum_to_100


def make_enriched():
    rows = [
        # review_id, date, platform, rating, sentiment, category, urgency, language, status
        ("R-1", "2026-11-16", "Swiggy", 1, "negative", "delivery", "medium", "English", "ok"),
        ("R-2", "2026-11-17", "Swiggy", 2, "negative", "delivery", "medium", "Tanglish", "ok"),
        ("R-3", "2026-11-18", "Zomato", 1, "negative", "delivery", "high", "English", "ok"),
        ("R-4", "2026-11-18", "Google", 1, "negative", "staff", "high", "Tamil", "ok"),
        ("R-5", "2026-11-19", "Google", 5, "positive", "food_quality", "low", "English", "ok"),
        ("R-6", "2026-11-22", "Zomato", 3, "neutral", "food_quality", "low", "English", "ok"),
        ("R-7", "2026-11-22", "Zomato", 5, None, None, None, None, "needs_review"),
    ]
    df = pd.DataFrame(rows, columns=["review_id", "date", "platform", "rating", "sentiment",
                                     "category", "urgency", "language", "analysis_status"])
    df["summary"] = [f"summary {i}" for i in range(len(df))]
    return df


def test_counts_exclude_needs_review_rows():
    stats = compute_stats(make_enriched())
    assert stats.total_reviews == 6
    assert stats.needs_review == 1
    assert stats.sentiment_counts == {"positive": 1, "neutral": 1, "negative": 4}
    assert stats.category_counts["delivery"] == 3
    assert stats.category_counts["app"] == 0  # zero categories still appear


def test_period_rating_and_urgent_ids():
    stats = compute_stats(make_enriched())
    assert (stats.period_start, stats.period_end) == ("2026-11-16", "2026-11-22")
    assert stats.average_rating == pytest.approx(13 / 6, abs=0.01)
    assert [u.review_id for u in stats.urgent_reviews] == ["R-3", "R-4"]
    assert stats.urgent_reviews[1].category == "staff"
    assert stats.urgent_reviews[0].date == "2026-11-18"


def test_top_complaint_is_delivery_mostly_swiggy():
    top = compute_stats(make_enriched()).top_complaints[0]
    assert (top.category, top.count) == ("delivery", 3)
    assert top.top_platform == "Swiggy"
    assert top.top_platform_share_pct == pytest.approx(66.7)
    assert top.language_mix == {"English": 2, "Tanglish": 1}


@pytest.mark.parametrize("counts", [
    {"a": 1, "b": 1, "c": 1},          # 33.3 each -> naive rounding gives 99
    {"a": 151, "b": 34, "c": 63},
    {"a": 0, "b": 0, "c": 0},
])
def test_percentages_always_sum_to_100(counts):
    pct = percentages_that_sum_to_100(counts)
    assert sum(pct.values()) == (100 if sum(counts.values()) else 0)
