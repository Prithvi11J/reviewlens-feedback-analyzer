"""REDUCE, part 1: turn the enriched table into exact numbers with pandas.

Nothing in this file calls an LLM. Every count and percentage the weekly
report shows is computed here, so the report can only describe these
numbers, never invent them.

The result is a Pydantic model, which gives us:
  - a clear, typed contract between this step and the report step
  - .model_dump_json() to hand the facts to the report LLM in Stage 4
"""

from math import comb

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
    top_outlet: str | None = None  # only when the CSV has an `outlet` column
    top_outlet_share_pct: float | None = None
    language_mix: dict[str, int]
    example_summaries: list[str]


class UrgentReview(BaseModel):
    """A high-urgency review, with enough detail to spot clusters (same issue, same days)."""

    review_id: str
    date: str
    platform: str
    outlet: str | None = None
    category: str
    summary: str


class Hotspot(BaseModel):
    """A complaint type concentrated in one outlet, platform or language.

    e.g. packaging: 10 of 10 complaints at the Peelamedu outlet, which gets only 34% of reviews.
    """

    category: str
    dimension: str  # "outlet", "platform" or "language"
    value: str  # e.g. "Peelamedu", "Google", "Tanglish"
    count: int  # complaints of this category with this value
    total: int  # complaints of this category overall
    share_pct: float
    baseline_pct: float  # this value's share of ALL reviews, the yardstick
    chance: float  # how often pure luck would produce a concentration this strong


class OutletStats(BaseModel):
    """One branch's week, for comparing outlets side by side."""

    outlet: str
    reviews: int
    average_rating: float | None
    negative_pct: float
    urgent: int


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
    outlets: list[OutletStats] = []  # empty when the CSV has no `outlet` column
    hotspots: list[Hotspot] = []  # most surprising first


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


def _top(series: pd.Series) -> tuple[str, float]:
    """Most common value and its share in %. Ties are broken alphabetically, so results are stable."""
    counts = series.value_counts().sort_index().sort_values(ascending=False, kind="stable")
    return str(counts.index[0]), round(counts.iloc[0] * 100 / len(series), 1)


def _outlets(df: pd.DataFrame) -> list[OutletStats]:
    if "outlet" not in df:
        return []
    rows = []
    for outlet, group in df.groupby("outlet"):
        rating = pd.to_numeric(group["rating"], errors="coerce").mean()
        rows.append(OutletStats(
            outlet=str(outlet),
            reviews=len(group),
            average_rating=None if pd.isna(rating) else round(float(rating), 2),
            negative_pct=round((group["sentiment"] == "negative").mean() * 100, 1),
            urgent=int((group["urgency"] == "high").sum()),
        ))
    return sorted(rows, key=lambda o: o.outlet)


HOTSPOT_MIN_COMPLAINTS = 5  # fewer than this is too small to call a pattern
HOTSPOT_MAX_CHANCE = 0.01  # flag only if pure chance would produce it less than 1% of the time


def chance_of_at_least(k: int, n: int, p: float) -> float:
    """Probability of k or more 'hits' out of n if each one lands with probability p (binomial tail).

    Example: Peelamedu gets 34% of all reviews. If complaints landed there purely by chance,
    how likely is it that 10 of 10 packaging complaints come from Peelamedu? 0.34**10, about 0.002%.
    And 4 of 6? About 11%: easily luck, so not worth a manager's time.
    """
    return sum(comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


HOTSPOT_DIMENSIONS = ("outlet", "platform", "language")


def _hotspots(df: pd.DataFrame) -> list[Hotspot]:
    """Find complaint types concentrated in one outlet, platform or language.

    Checks ALL categories, not just the top 3, so a pattern is found whatever its rank.
    Each candidate is compared against that value's normal share of reviews: a platform
    that carries 40% of all reviews will naturally carry ~40% of each complaint type.
    Only concentrations that chance would rarely produce are reported.

    18 checks (6 categories x 3 dimensions) at a 1% threshold means roughly one false
    alarm every five or six weeks, an acceptable rate for a "worth a look" list.
    """
    negative = df[df["sentiment"] == "negative"]
    found = []
    for dimension in HOTSPOT_DIMENSIONS:
        if dimension not in df:
            continue
        baseline = df[dimension].value_counts(normalize=True)
        for category, group in negative.groupby("category"):
            if len(group) < HOTSPOT_MIN_COMPLAINTS:
                continue
            value, share = _top(group[dimension])
            count = int((group[dimension] == value).sum())
            chance = chance_of_at_least(count, len(group), float(baseline[value]))
            if share >= 50 and chance < HOTSPOT_MAX_CHANCE:
                found.append(Hotspot(
                    category=str(category), dimension=dimension, value=value, count=count,
                    total=len(group), share_pct=share,
                    baseline_pct=round(float(baseline[value]) * 100, 1), chance=chance,
                ))
    return sorted(found, key=lambda h: (h.chance, h.category))


def _themes(df: pd.DataFrame, sentiment: str, top_n: int, total: int) -> list[Theme]:
    subset = df[df["sentiment"] == sentiment]
    themes = []
    for category, group in subset.groupby("category"):
        top_platform, platform_share = _top(group["platform"])
        top_outlet, outlet_share = _top(group["outlet"]) if "outlet" in group else (None, None)
        # Sort so the same input always gives the same examples (reproducible).
        examples = group.sort_values("review_id")["summary"].drop_duplicates().head(MAX_EXAMPLES)
        themes.append(Theme(
            category=str(category),
            count=len(group),
            share_of_reviews_pct=round(len(group) * 100 / total, 1),
            top_platform=top_platform,
            top_platform_share_pct=platform_share,
            top_outlet=top_outlet,
            top_outlet_share_pct=outlet_share,
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
                     outlet=str(r.outlet) if "outlet" in urgent else None,
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
        outlets=_outlets(ok),
        hotspots=_hotspots(ok),
    )
