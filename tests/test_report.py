"""Offline tests for report assembly and the number check (no API calls)."""

from langchain_core.runnables import RunnableLambda

from reviewlens.aggregate import Theme, UrgentReview, WeeklyStats
from reviewlens.report import (
    ThemeInsight,
    UrgentCluster,
    WeeklyInsights,
    allowed_numbers,
    assemble_report,
    find_unsupported_numbers,
    generate_report,
)


def make_stats() -> WeeklyStats:
    def theme(cat, count, platform, share, langs):
        return Theme(category=cat, count=count, share_of_reviews_pct=10.0, top_platform=platform,
                     top_platform_share_pct=share, language_mix=langs,
                     example_summaries=["Order came 80 minutes late."])

    urgent = [UrgentReview(review_id=f"R-{i}", date=d, platform="Zomato", category="food_quality",
                           summary="Fish fry caused vomiting.") for i, d in
              [(1, "2026-11-20"), (2, "2026-11-20"), (3, "2026-11-21")]]
    return WeeklyStats(
        period_start="2026-11-16", period_end="2026-11-22", total_reviews=240, needs_review=0,
        average_rating=3.2, sentiment_counts={"positive": 122, "neutral": 5, "negative": 113},
        sentiment_pct={"positive": 51, "neutral": 2, "negative": 47},
        category_counts={}, language_counts={}, urgency_counts={"high": 3}, urgent_reviews=urgent,
        top_complaints=[theme("delivery", 31, "Swiggy", 74.2, {"English": 17, "Tanglish": 14}),
                        theme("price", 19, "Google", 84.2, {"English": 5, "Tanglish": 14})],
        top_praise=[theme("food_quality", 53, "Amazon", 36.0, {"English": 53})],
    )


def make_insights(**overrides) -> WeeklyInsights:
    base = dict(
        headline="Delivery delays drove most complaints.",
        top_complaints=[ThemeInsight(category="price", title="Hidden bill charges", insight="Extra fees."),
                        ThemeInsight(category="delivery", title="Late, cold deliveries", insight="Slow.")],
        top_praise=[ThemeInsight(category="food_quality", title="Biryani taste", insight="Loved.")],
        urgent_clusters=[UrgentCluster(title="Fish fry poisoning", review_ids=["R-1", "R-3", "R-99"],
                                       why_it_matters="Health risk.")],
        action_items=["Add a dinner rider.", "Show all charges on the menu.", "Reply to urgent reviews."],
    )
    base.update(overrides)
    return WeeklyInsights(**base)


def test_numbers_come_from_stats_and_order_follows_stats():
    md = assemble_report(make_stats(), make_insights()).markdown
    assert "**Total reviews: 240**" in md and "Negative 47%" in md
    # LLM listed price first, but the report keeps the stats ranking (delivery is #1)
    assert md.index("Late, cold deliveries") < md.index("Hidden bill charges")
    assert "(31 reviews; mostly Swiggy, 74%)" in md


def test_pattern_notes_only_above_50_percent():
    md = assemble_report(make_stats(), make_insights()).markdown
    assert "mostly Tanglish reviews, 74%" in md   # price: 14 of 19
    assert "mostly Amazon" not in md               # 36% is noise, so it's not mentioned


def test_cluster_ids_are_validated():
    report = assemble_report(make_stats(), make_insights())
    assert "**Fish fry poisoning**: 2 reports, on Fri 20 Nov and Sat 21 Nov" in report.markdown
    assert any("R-99" in w for w in report.warnings)


def test_clusters_ranked_by_intensity_not_ai_order():
    stats = make_stats()
    stats.urgent_reviews += [UrgentReview(review_id=f"H-{d}", date=f"2026-11-{d}", platform="Google",
                                          category="food_quality", summary="Hair in food.") for d in (16, 19, 22)]
    insights = make_insights(urgent_clusters=[
        UrgentCluster(title="Hair in food", review_ids=["H-16", "H-19", "H-22"], why_it_matters="Hygiene."),  # AI said first
        UrgentCluster(title="Fish fry poisoning", review_ids=["R-1", "R-2", "R-3"], why_it_matters="Health."),
    ])
    md = assemble_report(stats, insights).markdown
    # 3 reports in 2 days (1.5/day) outranks 3 reports over 7 days (0.4/day)
    assert md.index("Fish fry poisoning") < md.index("Hair in food")


def test_invented_number_is_flagged():
    insights = make_insights(action_items=["Hire 7 riders.", "Reply to all 3 urgent reviews.", "Fix 80 minute waits."])
    report = assemble_report(make_stats(), insights)
    flagged = [w for w in report.warnings if "number" in w]
    assert len(flagged) == 1 and "'7'" in flagged[0]  # 3 and 80 are in the data, 7 is not


def test_allowed_numbers_include_summary_numbers_and_rounded_values():
    allowed = allowed_numbers(make_stats())
    assert {"240", "47", "80", "74", "2026"} <= allowed
    assert find_unsupported_numbers(["Up 47% this week"], allowed) == set()


def test_wrong_themes_are_retried_then_flagged():
    calls = []

    def fake(_):
        calls.append(1)
        # always covers 'price' and 'food_quality' but never the real #1 complaint, 'delivery'
        return make_insights(top_complaints=[
            ThemeInsight(category="price", title="Hidden bill charges", insight="Extra fees."),
            ThemeInsight(category="food_quality", title="Wrong theme", insight="Off topic.")])

    report = generate_report(make_stats(), chain=RunnableLambda(fake))
    assert len(calls) == 3  # asked again twice
    assert "**Delivery** (31 reviews" in report.markdown  # readable fallback title
    assert any("'delivery'" in w for w in report.warnings)


def test_generate_report_with_fake_chain():
    fake = RunnableLambda(lambda _: make_insights())
    report = generate_report(make_stats(), chain=fake)
    assert report.markdown.startswith("# Weekly Feedback Report | 16 Nov to 22 Nov 2026")
