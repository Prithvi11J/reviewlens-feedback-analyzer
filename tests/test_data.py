"""Offline tests for CSV validation."""

import pandas as pd
import pytest

from reviewlens.data import InvalidCSVError, validate_reviews


def make_df(**overrides):
    base = {
        "review_id": ["R-1", "R-2", "R-3", "R-2"],
        "date": ["2026-11-16", "2026-11-17", "2026-11-18", "2026-11-17"],
        "platform": ["Google", "Zomato", "Swiggy", "Zomato"],
        "rating": [5, 1, "abc", 1],
        "review_text": ["Great food", "Late delivery", "   ", "Late delivery"],
    }
    base.update(overrides)
    return pd.DataFrame(base)


def test_drops_empty_and_duplicate_rows():
    clean, report = validate_reviews(make_df())
    assert list(clean["review_id"]) == ["R-1", "R-2"]
    assert report.empty_text_dropped == 1
    assert report.duplicates_dropped == 1
    assert report.rows_kept == 2


def test_header_case_and_spaces_are_forgiven():
    df = make_df().rename(columns={"review_text": " Review_Text "})
    clean, _ = validate_reviews(df)
    assert "review_text" in clean.columns


def test_missing_column_fails_with_clear_message():
    df = make_df().drop(columns=["review_text"])
    with pytest.raises(InvalidCSVError, match="review_text"):
        validate_reviews(df)


def test_invalid_rating_becomes_nan_not_crash():
    df = make_df(review_text=["a", "b", "c", "d"], review_id=["1", "2", "3", "4"])
    clean, report = validate_reviews(df)
    assert clean["rating"].isna().sum() == 1
    assert report.notes
