"""Offline tests for the reply-queue order."""

import pandas as pd
import pytest

from reviewlens.replies import build_queue, platform_tip, priority_tier


@pytest.mark.parametrize("rating, urgency, tier", [
    (1, "medium", 1),
    (2, "low", 1),
    (4, "high", 2),     # a 4-star review about a hair in the food jumps the queue
    (3, "medium", 3),
    (4, "low", 3),
    (5, "low", 4),
    (5, "high", 2),
    (None, "low", 3),   # missing rating: treated like a middle rating
])
def test_priority_tier(rating, urgency, tier):
    assert priority_tier(rating, urgency) == tier


def test_queue_order_and_failed_rows_excluded():
    df = pd.DataFrame({
        "review_id": ["A", "B", "C", "D", "E", "F"],
        "rating": [5, 4, 1, 2, 3, 1],
        "urgency": ["low", "high", "medium", "high", "medium", None],
        "sentiment": ["positive", "negative", "negative", "negative", "neutral", None],
        "date": ["2026-11-16", "2026-11-17", "2026-11-20", "2026-11-21", "2026-11-18", "2026-11-16"],
        "analysis_status": ["ok", "ok", "ok", "ok", "ok", "needs_review"],
    })
    queue = build_queue(df)
    # D (2 stars, high urgency) before C (1 star, medium): urgency decides within the 1-2 star tier
    assert list(queue["review_id"]) == ["D", "C", "B", "E", "A"]
    assert list(queue["priority_label"]) == ["1–2 stars", "1–2 stars", "Urgent issue", "3–4 stars", "5 stars"]


def test_platform_tips():
    assert "Google" in platform_tip("Google")
    assert "partner app" in platform_tip("Swiggy")
    assert platform_tip("Instagram").startswith("Reply on the platform")
