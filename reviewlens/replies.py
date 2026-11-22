"""The reply queue: which review to answer first.

Real review-management guides for Indian restaurants describe the same morning
routine: reply to 1- and 2-star reviews first, then anything about hygiene or
illness, then 3- and 4-star reviews, then 5-star reviews. We follow that order,
using our own `urgency` tag for the "urgent issue" tier (health and safety, money
taken wrongly, rude staff, wrong or missing orders).

Pure pandas, no Streamlit, so it can be tested on its own.
"""

import pandas as pd

TIERS = {
    1: "1–2 stars",
    2: "Urgent issue",
    3: "3–4 stars",
    4: "5 stars",
}

# Where each platform's reply actually happens. Swiggy ratings and comments stay inside
# the partner app, so the useful response there is fixing the order problem behind them.
PLATFORM_TIPS = {
    "Google": "Public reply on Google Maps, read by future guests. Reply here first.",
    "Zomato": "Reply from the Zomato restaurant partner app.",
    "Swiggy": "Swiggy feedback stays in the partner app: fix the order issue behind it there.",
}


def priority_tier(rating, urgency: str) -> int:
    """1 = answer first ... 4 = answer last."""
    stars = None if pd.isna(rating) else int(rating)
    if stars is not None and stars <= 2:
        return 1
    if urgency == "high":
        return 2  # e.g. a 4-star review that mentions a hair in the food
    if stars is None or stars <= 4:
        return 3
    return 4


def build_queue(enriched: pd.DataFrame) -> pd.DataFrame:
    """Every analysed review, in the order a manager should reply.

    Within a tier: high urgency first, then negative before neutral before positive,
    then the review that has waited longest (oldest date), then review_id for a stable order.
    """
    ok = enriched[enriched["analysis_status"] == "ok"].copy()
    ok["priority"] = [priority_tier(r, u) for r, u in zip(ok["rating"], ok["urgency"], strict=True)]
    ok["priority_label"] = ok["priority"].map(TIERS)
    ok["_urgency_rank"] = ok["urgency"].map({"high": 0, "medium": 1, "low": 2})
    ok["_sentiment_rank"] = ok["sentiment"].map({"negative": 0, "neutral": 1, "positive": 2})
    ok["_date"] = pd.to_datetime(ok["date"], errors="coerce")
    ordered = ok.sort_values(["priority", "_urgency_rank", "_sentiment_rank", "_date", "review_id"])
    return ordered.drop(columns=["_urgency_rank", "_sentiment_rank", "_date"]).reset_index(drop=True)


def platform_tip(platform: str) -> str:
    return PLATFORM_TIPS.get(platform, "Reply on the platform where the review was posted.")
