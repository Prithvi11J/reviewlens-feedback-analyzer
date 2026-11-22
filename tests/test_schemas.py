"""Offline tests: they check our schema rules without calling the LLM (free, fast)."""

import pytest
from pydantic import ValidationError

from reviewlens.schemas import CATEGORIES, ReviewAnalysis

VALID = {
    "sentiment": "negative",
    "category": "delivery",
    "urgency": "high",
    "language": "English",
    "summary": "Order arrived 70 minutes late and the food was cold.",
    "suggested_reply": "We are sorry your order reached late and cold.",
}


def test_valid_review_passes():
    result = ReviewAnalysis(**VALID)
    assert result.category == "delivery"


def test_category_list_matches_brief():
    assert set(CATEGORIES) == {"food_quality", "delivery", "price", "staff", "packaging", "app"}


@pytest.mark.parametrize(
    "field, bad_value",
    [
        ("category", "late_delivery"),  # not in the allowed list
        ("sentiment", "Negative"),  # wrong case still counts as invalid
        ("urgency", "urgent"),
        ("language", "Hindi"),
    ],
)
def test_invalid_values_are_rejected(field, bad_value):
    with pytest.raises(ValidationError):
        ReviewAnalysis(**{**VALID, field: bad_value})


def test_missing_field_is_rejected():
    incomplete = {k: v for k, v in VALID.items() if k != "suggested_reply"}
    with pytest.raises(ValidationError):
        ReviewAnalysis(**incomplete)
