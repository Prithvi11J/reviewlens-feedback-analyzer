"""REDUCE, part 2: the weekly insights report.

Division of labour (the core design idea of this project):
  - pandas (aggregate.py) produced every NUMBER.
  - The LLM reads those facts and writes only WORDS: theme titles, insights,
    which urgent reviews belong to the same incident, and action items.
  - Our code assembles the final report, inserting the numbers itself.
  - A last check scans the LLM's words for any number that is not in the facts.

So the LLM has no opportunity to miscount, and if it slips a made-up number
into a sentence anyway, the report says so.
"""

import re
from dataclasses import dataclass, field
from datetime import date

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from reviewlens.aggregate import Theme, WeeklyStats
from reviewlens.config import get_llm
from reviewlens.schemas import Category

# ----------------------------------------------------------------------------
# 1. What the LLM must return
# ----------------------------------------------------------------------------


class ThemeInsight(BaseModel):
    category: Category = Field(description="Must be one of the categories given in the facts for this section.")
    title: str = Field(description="2 to 5 words naming WHAT customers talk about, e.g. 'Late, cold deliveries'. No numbers.")
    insight: str = Field(description="One sentence explaining the theme from the example summaries. No numbers.")


class UrgentCluster(BaseModel):
    title: str = Field(description="Short name of the incident, e.g. 'Food poisoning after fish fry'. No numbers.")
    review_ids: list[str] = Field(description="IDs of ALL urgent reviews describing this same incident or issue.")
    why_it_matters: str = Field(description="One sentence on the risk to the business. No numbers.")


class WeeklyInsights(BaseModel):
    headline: str = Field(description="One sentence: the single most important takeaway of the week.")
    top_complaints: list[ThemeInsight] = Field(description="One entry per complaint theme in the facts, same order.")
    top_praise: list[ThemeInsight] = Field(description="One entry per praise theme in the facts, same order.")
    urgent_clusters: list[UrgentCluster] = Field(
        description="Groups of 2 or more urgent reviews that describe the SAME specific incident or "
        "recurring issue (same dish, same staff behaviour, same payment bug). Most serious first. "
        "Leave empty if no such group exists."
    )
    action_items: list[str] = Field(
        description="Exactly 3 concrete actions for the manager, most important first. Each starts "
        "with a verb. Only use numbers that appear in the facts."
    )


# ----------------------------------------------------------------------------
# 2. The chain
# ----------------------------------------------------------------------------

REPORT_SYSTEM_PROMPT = """\
You are an operations analyst writing a weekly customer-feedback briefing for a restaurant manager.

You receive FACTS as JSON. They were calculated exactly from this week's reviews.
- All counts and percentages in the final report are inserted by the system from these facts.
  Do NOT write counts or percentages in titles or insights.
- In action items you may only use numbers that appear in the facts.
- Base every statement on the facts and the example summaries. Do not guess causes that the
  reviews don't mention.
- A platform or language is only a real pattern if its share is at least 50%.
- Urgent reviews are listed with dates: look for several reports of the same incident on the
  same or nearby days. A cluster of food-safety reports is the most serious finding possible.
- Write in plain, direct English that a busy manager can read in five minutes."""

REPORT_HUMAN_PROMPT = "FACTS:\n{facts}"


def build_report_chain() -> Runnable:
    prompt = ChatPromptTemplate.from_messages([("system", REPORT_SYSTEM_PROMPT), ("human", REPORT_HUMAN_PROMPT)])
    return prompt | get_llm().with_structured_output(WeeklyInsights, strict=True)


# ----------------------------------------------------------------------------
# 3. Assemble the report (code, not LLM, writes every number)
# ----------------------------------------------------------------------------

MIN_PATTERN_SHARE = 50.0  # below this, "mostly X" would be noise


@dataclass
class Report:
    markdown: str
    insights: WeeklyInsights
    warnings: list[str] = field(default_factory=list)


def generate_report(stats: WeeklyStats, chain: Runnable | None = None) -> Report:
    chain = chain or build_report_chain()
    insights = chain.with_retry(stop_after_attempt=3).invoke({"facts": stats.model_dump_json(indent=1)})
    return assemble_report(stats, insights)


def assemble_report(stats: WeeklyStats, insights: WeeklyInsights) -> Report:
    warnings: list[str] = []
    clusters = _valid_clusters(stats, insights.urgent_clusters, warnings)
    actions = insights.action_items[:3]
    if len(actions) < 3:
        warnings.append(f"The model returned {len(actions)} action item(s) instead of 3.")

    pct = stats.sentiment_pct
    lines = [
        f"# Weekly Feedback Report | {_fmt_period(stats.period_start, stats.period_end)}",
        "",
        f"**Total reviews: {stats.total_reviews}** &nbsp; Positive {pct['positive']}% · "
        f"Neutral {pct['neutral']}% · Negative {pct['negative']}%"
        + (f" &nbsp; (avg rating {stats.average_rating:.1f}/5)" if stats.average_rating else ""),
        "",
        f"> {insights.headline}",
        "",
        f"## Top {len(stats.top_complaints)} complaints",
    ]
    lines += _theme_lines(stats.top_complaints, insights.top_complaints, "reviews")
    lines += ["", "## Top praise"]
    lines += _theme_lines(stats.top_praise, insights.top_praise, "mentions")

    lines += ["", f"## ⚠️ Urgent: {len(stats.urgent_reviews)} reviews need a reply within 24 hours"]
    by_id = {u.review_id: u for u in stats.urgent_reviews}
    for c in clusters:
        days = sorted({by_id[i].date for i in c.review_ids})
        lines.append(f"- **{c.title}**: {len(c.review_ids)} reports, {_fmt_days(days)}. {c.why_it_matters}")
        lines.append(f"  <sub>{', '.join(c.review_ids)}</sub>")
    if not clusters:
        lines.append("- No repeated incident found among this week's urgent reviews.")

    lines += ["", "## Action items"]
    lines += [f"{i}. {a}" for i, a in enumerate(actions, 1)]

    if stats.needs_review:
        lines += ["", f"_{stats.needs_review} review(s) could not be analysed automatically and need a manual check._"]

    llm_text = [insights.headline, *actions, *(t.insight for t in insights.top_complaints + insights.top_praise),
                *(c.why_it_matters for c in clusters)]
    allowed = allowed_numbers(stats) | {str(len(c.review_ids)) for c in clusters}
    for bad in sorted(find_unsupported_numbers(llm_text, allowed)):
        warnings.append(f"The number '{bad}' in the AI-written text does not appear in the data.")

    if warnings:
        lines += ["", "---", "**Quality check:**"] + [f"- {w}" for w in warnings]

    return Report(markdown="\n".join(lines), insights=insights, warnings=warnings)


def _theme_lines(themes: list[Theme], insights: list[ThemeInsight], unit: str) -> list[str]:
    by_category = {i.category: i for i in insights}
    lines = []
    for n, theme in enumerate(themes, 1):
        ins = by_category.get(theme.category)
        title = ins.title if ins else theme.category.replace("_", " ").capitalize()
        # two trailing spaces = Markdown line break, so the insight starts on its own line
        lines.append(f"{n}. **{title}** ({theme.count} {unit}{_pattern_note(theme)})  ")
        if ins:
            lines.append(f"   {ins.insight}")
    return lines


def _pattern_note(theme: Theme) -> str:
    """Mention a platform or language only when it is a real pattern (>= 50%)."""
    notes = []
    if theme.top_platform_share_pct >= MIN_PATTERN_SHARE:
        notes.append(f"mostly {theme.top_platform}, {theme.top_platform_share_pct:.0f}%")
    lang, n = max(theme.language_mix.items(), key=lambda kv: kv[1])
    if lang != "English" and n * 100 / theme.count >= MIN_PATTERN_SHARE:
        notes.append(f"mostly {lang} reviews, {n * 100 / theme.count:.0f}%")
    return "; " + "; ".join(notes) if notes else ""


def _valid_clusters(stats: WeeklyStats, clusters: list[UrgentCluster], warnings: list[str]) -> list[UrgentCluster]:
    """Keep only review IDs that really are urgent this week; drop clusters smaller than 2."""
    urgent_ids = {u.review_id for u in stats.urgent_reviews}
    valid = []
    for c in clusters:
        ids = list(dict.fromkeys(i for i in c.review_ids if i in urgent_ids))  # dedupe, keep order
        unknown = set(c.review_ids) - urgent_ids
        if unknown:
            warnings.append(f"Ignored unknown review IDs in '{c.title}': {', '.join(sorted(unknown))}.")
        if len(ids) >= 2:
            valid.append(c.model_copy(update={"review_ids": ids}))
    return valid


# ----------------------------------------------------------------------------
# 4. The number check
# ----------------------------------------------------------------------------

NUMBER = re.compile(r"\d+(?:\.\d+)?")
ALWAYS_ALLOWED = {"24"}  # "reply within 24 hours" is our own policy, not a data claim


def allowed_numbers(stats: WeeklyStats) -> set[str]:
    """Every number that appears anywhere in the facts, including inside review summaries."""
    found = set(ALWAYS_ALLOWED)

    def walk(value):
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)):
            found.update({str(value), str(round(value))})
            if float(value).is_integer():
                found.add(str(int(value)))
        elif isinstance(value, str):
            found.update(NUMBER.findall(value))
        elif isinstance(value, dict):
            for k, v in value.items():
                walk(k)
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)

    walk(stats.model_dump())
    return found


def find_unsupported_numbers(texts: list[str], allowed: set[str]) -> set[str]:
    return {num for text in texts for num in NUMBER.findall(text) if num not in allowed}


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------


def _fmt_period(start: str, end: str) -> str:
    s, e = date.fromisoformat(start), date.fromisoformat(end)
    return f"{s.day} {s:%b} to {e.day} {e:%b %Y}"


def _fmt_days(days: list[str]) -> str:
    labels = [f"{date.fromisoformat(d):%a} {date.fromisoformat(d).day} {date.fromisoformat(d):%b}" for d in days]
    return "on " + (labels[0] if len(labels) == 1 else ", ".join(labels[:-1]) + " and " + labels[-1])
