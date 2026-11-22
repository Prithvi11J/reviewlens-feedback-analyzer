"""Offline tests for the batch step, using a fake chain instead of OpenAI."""

import pandas as pd
from langchain_core.runnables import RunnableLambda

from reviewlens.analyze import STATUS_NEEDS_REVIEW, STATUS_OK, analyze_reviews
from reviewlens.schemas import ReviewAnalysis


def fake_llm(inputs: dict) -> ReviewAnalysis:
    if "BROKEN" in inputs["review_text"]:
        raise ValueError("model returned garbage")
    return ReviewAnalysis(
        sentiment="negative", category="delivery", urgency="medium", language="English",
        summary="Late.", suggested_reply="Sorry.",
    )


def test_one_failure_does_not_stop_the_batch():
    df = pd.DataFrame({
        "review_id": ["R-1", "R-2", "R-3"],
        "platform": ["Google"] * 3,
        "rating": [1, 2, None],
        "review_text": ["late", "BROKEN review", "cold food"],
    })
    progress = []
    out = analyze_reviews(df, chain=RunnableLambda(fake_llm), chunk_size=2,
                          on_progress=lambda done, total: progress.append((done, total)))

    assert list(out["review_id"]) == ["R-1", "R-2", "R-3"]  # order and IDs preserved
    assert list(out["analysis_status"]) == [STATUS_OK, STATUS_NEEDS_REVIEW, STATUS_OK]
    assert pd.isna(out.loc[1, "category"])  # pandas stores missing values as NaN
    assert "garbage" in out.loc[1, "error"]
    assert progress == [(2, 3), (3, 3)]
