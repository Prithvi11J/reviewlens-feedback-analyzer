"""Offline tests for the optional `outlet` column and the restaurant context."""

import pandas as pd

from reviewlens.aggregate import chance_of_at_least, compute_stats
from reviewlens.chains import build_review_prompt
from reviewlens.data import validate_reviews
from reviewlens.report import ThemeInsight, UrgentCluster, WeeklyInsights, assemble_report


def make_enriched() -> pd.DataFrame:
    rows = [
        # id, outlet, platform, rating, sentiment, category, urgency, date
        ("R-1", "Peelamedu", "Swiggy", 1, "negative", "packaging", "medium", "2026-11-16"),
        ("R-2", "Peelamedu", "Zomato", 2, "negative", "packaging", "medium", "2026-11-17"),
        ("R-3", "RS Puram", "Google", 1, "negative", "packaging", "medium", "2026-11-18"),
        ("R-4", "Peelamedu", "Google", 1, "negative", "food_quality", "high", "2026-11-20"),
        ("R-5", "Peelamedu", "Zomato", 1, "negative", "food_quality", "high", "2026-11-21"),
        ("R-6", "RS Puram", "Google", 5, "positive", "food_quality", "low", "2026-11-19"),
    ]
    df = pd.DataFrame(rows, columns=["review_id", "outlet", "platform", "rating", "sentiment",
                                     "category", "urgency", "date"])
    df["language"] = "English"
    df["summary"] = "summary"
    df["analysis_status"] = "ok"
    return df


def test_validation_keeps_optional_outlet_column():
    raw = pd.DataFrame({"review_id": ["1"], "date": ["2026-11-16"], "platform": ["Google"],
                        "rating": [5], "review_text": ["Great biryani"], "outlet": [" RS Puram "]})
    clean, _ = validate_reviews(raw)
    assert clean.loc[0, "outlet"] == "RS Puram"


def test_outlet_stats_and_theme_top_outlet():
    stats = compute_stats(make_enriched())
    by_name = {o.outlet: o for o in stats.outlets}
    assert by_name["Peelamedu"].reviews == 4 and by_name["Peelamedu"].urgent == 2
    assert by_name["RS Puram"].negative_pct == 50.0
    packaging = next(t for t in stats.top_complaints if t.category == "packaging")
    assert (packaging.top_outlet, packaging.top_outlet_share_pct) == ("Peelamedu", 66.7)


def test_report_names_dominant_outlet_and_single_outlet_cluster():
    stats = compute_stats(make_enriched())
    insights = WeeklyInsights(
        headline="Peelamedu needs attention.",
        top_complaints=[ThemeInsight(category="packaging", title="Leaking parcels", insight="Spills.")],
        top_praise=[],
        urgent_clusters=[UrgentCluster(title="Fish fry illness", review_ids=["R-4", "R-5"], why_it_matters="Risk.")],
        action_items=["Check parcels.", "Check fish.", "Reply today."],
    )
    md = assemble_report(stats, insights).markdown
    assert "mostly Peelamedu outlet, 67%" in md
    assert "2 reports, on Fri 20 Nov and Sat 21 Nov, all at Peelamedu" in md


def week(packaging_outlets: list[str], other_reviews: dict[str, int]) -> pd.DataFrame:
    """Packaging complaints at the given outlets, plus positive reviews to set each outlet's baseline."""
    rows = [(o, "negative", "packaging") for o in packaging_outlets]
    rows += [(o, "positive", "food_quality") for o, n in other_reviews.items() for _ in range(n)]
    df = pd.DataFrame(rows, columns=["outlet", "sentiment", "category"])
    df["review_id"] = [f"R-{i}" for i in range(len(df))]
    df = df.assign(platform="Google", rating=3, urgency="low", date="2026-11-16",
                   language="English", summary="s", analysis_status="ok")
    return df


def test_chance_calculation():
    assert chance_of_at_least(10, 10, 0.34) < 0.0001  # 10 of 10: not luck
    assert 0.10 < chance_of_at_least(4, 6, 0.34) < 0.12  # 4 of 6: easily luck


def test_hotspot_flags_unlikely_concentration_only():
    strong = compute_stats(week(["Peelamedu"] * 8, {"RS Puram": 20, "Gandhipuram": 20})).hotspots
    assert [(h.category, h.dimension, h.value, h.count, h.total) for h in strong] == [
        ("packaging", "outlet", "Peelamedu", 8, 8)]

    weak = week(["Peelamedu"] * 4 + ["RS Puram"] * 2, {"RS Puram": 20, "Gandhipuram": 20, "Peelamedu": 16})
    assert compute_stats(weak).hotspots == []  # 4 of 6 with a 34% baseline: could be chance


def test_hotspots_also_cover_platform_and_language():
    df = week(["RS Puram"] * 8, {"RS Puram": 10, "Gandhipuram": 10, "Peelamedu": 10})
    complaints = df["sentiment"] == "negative"
    df.loc[complaints, "platform"] = "Swiggy"  # every packaging complaint on Swiggy...
    df.loc[complaints, "language"] = "Tanglish"  # ...and written in Tanglish
    df.loc[~complaints, "platform"] = ["Google", "Zomato", "Swiggy"] * 10
    found = {(h.dimension, h.value) for h in compute_stats(df).hotspots}
    assert ("platform", "Swiggy") in found and ("language", "Tanglish") in found


def test_report_does_not_repeat_hotspots_the_top_complaints_show():
    from reviewlens.report import new_hotspots
    df = week(["Peelamedu"] * 8, {"RS Puram": 20, "Gandhipuram": 20})
    stats = compute_stats(df)
    # packaging is the top complaint and its note already says "mostly Peelamedu outlet"
    assert stats.top_complaints[0].category == "packaging"
    assert new_hotspots(stats) == []


def test_noise_patterns_are_hidden_from_the_llm():
    from reviewlens.report import facts_for_llm
    facts = facts_for_llm(compute_stats(make_enriched()))
    assert '"top_outlet": "Peelamedu"' in facts  # packaging 67%: a real pattern, kept
    assert "no clear pattern" in facts  # platform leaders below 50% are masked


def test_prompt_has_restaurant_context_and_outlet_default():
    prompt = build_review_prompt()
    without = prompt.format(platform="Google", rating="5", review_text="Nice")
    assert "PJ's" in without and "Outlet: not given" in without
    with_outlet = prompt.format(platform="Google", rating="5", review_text="Nice", outlet="Peelamedu")
    assert "Outlet: Peelamedu" in with_outlet
