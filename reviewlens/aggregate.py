"""REDUCE, part 1: turn the enriched table into exact numbers with pandas.

Nothing in this file calls an LLM. Every count and percentage the weekly
report shows is computed here, so the report can only describe these
numbers, never invent them.

The result is a Pydantic model, which gives us:
  - a clear, typed contract between this step and the report step
  - .model_dump_json() to hand the facts to the report LLM in Stage 4
"""

import pandas as pd
from pydantic import BaseModel

from reviewlens.analyze import STATUS_OK
from reviewlens.schemas import CATEGORIES, LANGUAGES, SENTIMENTS, URGENCIES

MAX_EXAMPLES = 5  # summaries per theme that the report LLM may quote from


class Theme(BaseModel):
    """One category seen through one sentiment, e.g. negative 'delivery'."""

    category: str
    count: int
    share_of_reviews_pct: float  # of all analysed reviews
    top_platform: str
    top_platform_share_pct: float  # of this theme's reviews
    language_mix: dict[str, int]
    example_summaries: list[str]


class UrgentReview(BaseModel):
    """A high-urgency review, with enough detail to spot clusters (same issue, same days)."""

    review_id: str
    date: str
    platform: str
    category: str
    summary: str


class WeeklyStats(BaseModel):
    period_start: str
    period_end: str
    total_reviews: int
    needs_review: int
    average_rating: float | None
    sentiment_counts: dict[str, int]
    sentiment_pct: dict[str, int]  # whole numbers that always add up to 100
    category_counts: dict[str, int]
    language_counts: dict[str, int]
    urgency_counts: dict[str, int]
    urgent_reviews: list[UrgentReview]  # every high-urgency review, oldest first
    top_complaints: list[Theme]
    top_praise: list[Theme]


def percentages_that_sum_to_100(counts: dict[str, int]) -> dict[str, int]:
    """Round to whole percentages using the largest-remainder method.

    Plain rounding can give 61% + 14% + 26% = 101%, which looks like a bug in a
    report. Here we round everything down, then give the leftover points to the
    values that lost the most in rounding.
    """
    total = sum(counts.values())
    if total == 0:
        return dict.fromkeys(counts, 0)
    exact = {k: v * 100 / total for k, v in counts.items()}
    result = {k: int(x) for k, x in exact.items()}
    leftover = 100 - sum(result.values())
    for k in sorted(exact, key=lambda k: exact[k] - result[k], reverse=True)[:leftover]:
        result[k] += 1
    return result


def _counts(series: pd.Series, allowed: tuple[str, ...]) -> dict[str, int]:
    """Count every allowed value, including the ones that appear 0 times."""
    vc = series.value_counts()
    return {value: int(vc.get(value, 0)) for value in allowed}


def _themes(df: pd.DataFrame, sentiment: str, top_n: int, total: int) -> list[Theme]:
    subset = df[df["sentiment"] == sentiment]
    themes = []
    for category, group in subset.groupby("category"):
        platforms = group["platform"].value_counts()
        # Sort so the same input always gives the same examples (reproducible).
        examples = group.sort_values("review_id")["summary"].drop_duplicates().head(MAX_EXAMPLES)
        themes.append(Theme(
            category=str(category),
            count=len(group),
            share_of_reviews_pct=round(len(group) * 100 / total, 1),
            top_platform=str(platforms.index[0]),
            top_platform_share_pct=round(platforms.iloc[0] * 100 / len(group), 1),
            language_mix={k: v for k, v in _counts(group["language"], LANGUAGES).items() if v},
            example_summaries=examples.tolist(),
        ))
    themes.sort(key=lambda t: (-t.count, t.category))  # biggest first, ties broken by name
    return themes[:top_n]


def _urgent(df: pd.DataFrame) -> list[UrgentReview]:
    """Summaries of the high-urgency reviews only.

    Category counts can't show a spike like '6 food-poisoning reports in 2 days':
    they are all just 'food_quality'. Listing the urgent summaries with dates
    gives the report step the detail it needs to notice a cluster.
    """
    urgent = df[df["urgency"] == "high"].copy()
    urgent["date"] = pd.to_datetime(urgent["date"], errors="coerce").dt.date.astype(str)
    urgent = urgent.sort_values(["date", "review_id"])
    return [
        UrgentReview(review_id=str(r.review_id), date=r.date, platform=str(r.platform),
                     category=str(r.category), summary=str(r.summary))
        for r in urgent.itertuples(index=False)
    ]


def compute_stats(enriched: pd.DataFrame, top_n: int = 3) -> WeeklyStats:
    """Summarise an enriched table (output of analyze_reviews) into exact numbers."""
    ok = enriched[enriched["analysis_status"] == STATUS_OK]
    if ok.empty:
        raise ValueError("No successfully analysed reviews to summarise.")
    total = len(ok)
    dates = pd.to_datetime(ok["date"], errors="coerce")
    rating = pd.to_numeric(ok["rating"], errors="coerce").mean()
    sentiment_counts = _counts(ok["sentiment"], SENTIMENTS)

    return WeeklyStats(
        period_start=dates.min().date().isoformat(),
        period_end=dates.max().date().isoformat(),
        total_reviews=total,
        needs_review=int((enriched["analysis_status"] != STATUS_OK).sum()),
        average_rating=None if pd.isna(rating) else round(float(rating), 2),
        sentiment_counts=sentiment_counts,
        sentiment_pct=percentages_that_sum_to_100(sentiment_counts),
        category_counts=_counts(ok["category"], CATEGORIES),
        language_counts=_counts(ok["language"], LANGUAGES),
        urgency_counts=_counts(ok["urgency"], URGENCIES),
        urgent_reviews=_urgent(ok),
        top_complaints=_themes(ok, "negative", top_n, total),
        top_praise=_themes(ok, "positive", top_n, total),
    )
