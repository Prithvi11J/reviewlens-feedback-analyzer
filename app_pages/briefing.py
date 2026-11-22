"""Monday briefing: the first screen a manager sees.

Built "exception-first", like a good weekly operations meeting: the few things
that need a decision come first, stable numbers come after. Every number on this
page comes from the pandas stats; the AI only supplies titles and wording.
"""

import streamlit as st

from reviewlens.charts import CATEGORY_LABELS
from reviewlens.config import get_restaurant_name
from reviewlens.report import (cluster_facts, fmt_days, hotspot_title, new_hotspots, pattern_note,
                               valid_clusters)
from ui import state

s = state.stats()
ins = state.insights()
name = get_restaurant_name()

# ---------------------------------------------------------------- header
st.image(str(state.ROOT / "assets" / "pj_logo.svg"), width=300)
st.title("Monday briefing")
outlet_names = " · ".join(o.outlet for o in s.outlets)
st.caption(f"{name} · {state.week_label(s)}" + (f" · {outlet_names}" if outlet_names else ""))

with st.container(horizontal=True):
    st.metric("Reviews this week", s.total_reviews, border=True)
    st.metric("Average rating", f"{s.average_rating:.1f} ★" if s.average_rating else "-", border=True)
    st.metric("Negative", f"{s.sentiment_pct['negative']}%", border=True)
    st.metric("Need a reply today", s.urgency_counts["high"], border=True,
              help="High urgency: health or safety, money taken wrongly, rude staff, or a wrong or missing order.")

st.page_link("app_pages/replies.py", label=f"Open the reply queue · {state.waiting_count()} reviews waiting",
             icon=":material/inbox:")


# ---------------------------------------------------------------- fix first
def card(col, label: str, icon: str, color: str, title: str, facts: str, note: str) -> None:
    with col.container(border=True, height="stretch"):
        st.badge(label, icon=icon, color=color)
        st.markdown(f"### {title}")
        st.markdown(facts)
        st.caption(note)


st.subheader("Fix first")
cards = []

clusters = valid_clusters(s, ins) if ins else []
if clusters:
    c = clusters[0]  # the report ranks incidents most serious first
    days, outlet = cluster_facts(s, c)
    where = f" · all at **{outlet}**" if outlet else ""
    cards.append(("Urgent incident", ":material/warning:", "red", c.title,
                  f"**{len(c.review_ids)} reports** · {fmt_days(days).removeprefix('on ')}{where}", c.why_it_matters))

HOTSPOT_CARD = {  # badge label and icon per kind of concentration
    "outlet": ("Outlet hotspot", ":material/location_on:"),
    "platform": ("Platform hotspot", ":material/storefront:"),
    "language": ("Language pattern", ":material/translate:"),
}
for h in new_hotspots(s)[:1]:  # the most surprising one the top-complaints card doesn't already cover
    label, icon = HOTSPOT_CARD[h.dimension]
    cards.append((label, icon, "orange", hotspot_title(h),
                  f"**{h.count} of {h.total}** complaints · {h.value} has {h.baseline_pct:.0f}% of all reviews",
                  "Too concentrated to be chance. Start the fix there."))

if s.top_complaints:
    top = s.top_complaints[0]
    top_ins = next((t for t in ins.top_complaints if t.category == top.category), None) if ins else None
    note = pattern_note(top).removeprefix("; ")
    cards.append(("Biggest complaint", ":material/trending_up:", "blue",
                  top_ins.title if top_ins else CATEGORY_LABELS.get(top.category, top.category),
                  f"**{top.count} reviews**" + (f" · {note}" if note else ""),
                  top_ins.insight if top_ins else "Generate the weekly report for a written summary."))

if cards:
    for col, c in zip(st.columns(len(cards)), cards, strict=True):
        card(col, *c)
else:
    st.success("Nothing needs urgent attention this week.", icon=":material/check_circle:")

# ---------------------------------------------------------------- actions
st.subheader("This week's actions")
if ins:
    st.markdown("\n".join(f"{i}. {a}" for i, a in enumerate(ins.action_items[:3], 1)))
else:
    st.info("Actions come from the weekly report, which hasn't been written for this week yet.",
            icon=":material/description:")
    st.page_link("app_pages/report.py", label="Write the weekly report", icon=":material/arrow_forward:")

# ---------------------------------------------------------------- outlets
if s.outlets:
    st.subheader("Outlets")
    worst = max(s.outlets, key=lambda o: o.negative_pct)
    best = max(s.outlets, key=lambda o: o.average_rating or 0)
    for col, o in zip(st.columns(len(s.outlets)), s.outlets, strict=True):
        with col.container(border=True, height="stretch"):
            st.markdown(f"**{o.outlet}**")
            # Every card gets a status line, so the cards stay aligned and each one says something.
            if len(s.outlets) > 1 and o is worst:
                st.badge("Needs attention", icon=":material/priority_high:", color="red")
            elif len(s.outlets) > 1 and o is best:
                st.badge("Best rated", icon=":material/thumb_up:", color="green")
            else:
                st.badge("Steady", icon=":material/check:", color="gray")
            rating_delta = (o.average_rating or 0) - (s.average_rating or 0)
            st.metric("Average rating", f"{o.average_rating:.1f} ★" if o.average_rating else "-",
                      delta=f"{rating_delta:+.1f}", delta_description=f"vs all {name} outlets")
            neg_delta = o.negative_pct - s.sentiment_pct["negative"]
            st.metric("Negative reviews", f"{o.negative_pct:.0f}%", delta=f"{neg_delta:+.0f} pts",
                      delta_color="inverse", delta_description="vs all outlets")
            st.caption(f"{o.reviews} reviews · {o.urgent} need a reply today")

if state.is_demo():
    st.caption("Demo week: synthetic reviews for PJ's, a fictional restaurant, generated with known patterns "
               "so the analysis can be checked against an answer key.")
