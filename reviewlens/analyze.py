"""The MAP step: run the review chain over every row of the table.

Key ideas:
- chain.batch(..., max_concurrency=N) sends N requests at a time: fast, but
  polite to the API's rate limits.
- .with_retry() retries a review that failed (timeout, bad output) before giving up.
- return_exceptions=True means one bad review never crashes the whole run;
  it is marked 'needs_review' so a human can look at it.
"""

from collections.abc import Callable

import pandas as pd
from langchain_core.runnables import Runnable

from reviewlens.chains import build_review_chain
from reviewlens.schemas import ReviewAnalysis

ANALYSIS_FIELDS = list(ReviewAnalysis.model_fields)  # sentiment, category, urgency, ...
STATUS_OK = "ok"
STATUS_NEEDS_REVIEW = "needs_review"

ProgressCallback = Callable[[int, int], None]  # (done, total)


def analyze_reviews(
    df: pd.DataFrame,
    chain: Runnable | None = None,
    max_concurrency: int = 8,
    chunk_size: int = 40,
    on_progress: ProgressCallback | None = None,
) -> pd.DataFrame:
    """Return `df` with the 6 analysis columns + an `analysis_status` column added.

    `chain` can be swapped for a fake in tests, so tests never call the real API.
    Work is split into chunks only so we can report progress between them.
    """
    if chain is None:
        chain = build_review_chain()
    chain = chain.with_retry(stop_after_attempt=3)

    has_outlet = "outlet" in df.columns
    inputs = [
        {"platform": row.platform, "rating": _fmt_rating(row.rating), "review_text": row.review_text,
         **({"outlet": row.outlet} if has_outlet else {})}
        for row in df.itertuples(index=False)
    ]

    results: list = []
    for start in range(0, len(inputs), chunk_size):
        chunk = inputs[start:start + chunk_size]
        results.extend(
            chain.batch(chunk, config={"max_concurrency": max_concurrency}, return_exceptions=True)
        )
        if on_progress:
            on_progress(len(results), len(inputs))

    rows = []
    for result in results:
        if isinstance(result, ReviewAnalysis):
            rows.append({**result.model_dump(), "analysis_status": STATUS_OK, "error": ""})
        else:  # an exception (or anything unexpected) after all retries
            rows.append({**dict.fromkeys(ANALYSIS_FIELDS), "analysis_status": STATUS_NEEDS_REVIEW,
                         "error": f"{type(result).__name__}: {result}"[:300]})

    enriched = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    return enriched


def _fmt_rating(rating) -> str:
    return "not given" if pd.isna(rating) else str(int(rating))
