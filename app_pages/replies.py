"""Reply queue: answer reviews in the order a restaurant manager should.

The AI already drafted a reply for every review. This page puts them in priority
order (see reviewlens/replies.py), lets the manager edit and copy each draft, and
tracks which ones are done. Posting still happens on Google / Zomato / Swiggy:
ReviewLens has no access to those accounts.
"""

from datetime import date

import pandas as pd
import streamlit as st

from reviewlens.charts import CATEGORY_LABELS
from reviewlens.replies import TIERS, platform_tip
from ui import state

TIER_STYLE = {  # badge colour and icon per priority tier
    1: ("red", ":material/sentiment_very_dissatisfied:"),
    2: ("orange", ":material/warning:"),
    3: ("blue", ":material/sentiment_neutral:"),
    4: ("green", ":material/sentiment_satisfied:"),
}


def mark_replied(review_id: str, text: str) -> None:
    state.replied()[review_id] = text


def undo_replied(review_id: str) -> None:
    state.replied().pop(review_id, None)


def save_draft(review_id: str) -> None:
    state.drafts()[review_id] = st.session_state[f"edit_{review_id}"]


def show_more() -> None:
    st.session_state.queue_limit += state.QUEUE_PAGE_SIZE


def stars(rating) -> str:
    return "" if pd.isna(rating) else "★" * int(rating) + "☆" * (5 - int(rating))


def reply_card(row) -> None:
    review_id = row.review_id
    done = review_id in state.replied()
    reply = state.replied().get(review_id) or state.drafts().get(review_id) or row.suggested_reply
    color, icon = TIER_STYLE[row.priority]

    with st.container(border=True):
        with st.container(horizontal=True, vertical_alignment="center", gap="small"):
            st.badge(row.priority_label, icon=icon, color=color)
            st.badge(row.platform, color="gray")
            if done:
                st.badge("Replied", icon=":material/check:", color="green")
            day = date.fromisoformat(str(row.date)[:10])
            where = f"{row.outlet} · " if "outlet" in row._fields else ""
            st.caption(f"{where}{day:%a} {day.day} {day:%b} · {stars(row.rating)} · "
                       f"{CATEGORY_LABELS.get(row.category, row.category)}")

        st.markdown(f"> {row.review_text}")
        if row.language != "English":
            st.caption(f"In English: {row.summary}")

        st.code(reply, language=None, wrap_lines=True)  # the code block has a built-in copy button
        st.caption(platform_tip(row.platform))

        with st.container(horizontal=True, gap="small"):
            if done:
                st.button("Undo", key=f"undo_{review_id}", icon=":material/undo:",
                          on_click=undo_replied, args=(review_id,))
            else:
                st.button("Mark as replied", key=f"done_{review_id}", icon=":material/check:",
                          on_click=mark_replied, args=(review_id, reply))
                with st.popover("Edit draft", icon=":material/edit:"):
                    # A fixed width makes the popover wide enough to edit a 3-sentence reply comfortably.
                    st.text_area("Your reply", reply, key=f"edit_{review_id}", height=160, width=560)
                    st.button("Save draft", key=f"save_{review_id}", on_click=save_draft, args=(review_id,))


# ---------------------------------------------------------------- page
st.title("Reply queue")
st.caption("In the order restaurant managers work through reviews: 1–2★ first, then urgent issues hidden in "
           "higher ratings, then 3–4★, then 5★. Each draft was written by AI: read it, edit it if needed, "
           "copy it (the copy icon appears at the draft's top right when you point at it), "
           "and post it on the platform.")

queue = state.reply_queue()
replied_ids = set(state.replied()) & set(queue["review_id"])
st.progress(len(replied_ids) / len(queue) if len(queue) else 1.0,
            text=f"{len(replied_ids)} of {len(queue)} replied")

counts = queue["priority"].value_counts()
tier_options = [t for t in TIERS if counts.get(t, 0)]
with st.container(horizontal=True, vertical_alignment="bottom"):
    chosen = st.pills("Priority", tier_options, selection_mode="multi", default=tier_options, key="q_tiers",
                      format_func=lambda t: f"{TIERS[t]} ({counts.get(t, 0)})")
    outlet = None
    if state.has_outlets():
        outlet = st.selectbox("Outlet", ["All outlets", *sorted(queue["outlet"].dropna().unique())], key="q_outlet")
    show_done = st.toggle("Show replied", key="q_show_done")

view = queue[queue["priority"].isin(chosen or [])]
if outlet and outlet != "All outlets":
    view = view[view["outlet"] == outlet]
if not show_done:
    view = view[~view["review_id"].isin(replied_ids)]

if view.empty:
    st.success("Nothing waiting here. Every review in this view has a reply.", icon=":material/task_alt:")
else:
    limit = st.session_state.get("queue_limit", state.QUEUE_PAGE_SIZE)
    for row in view.head(limit).itertuples(index=False):
        reply_card(row)
    if len(view) > limit:
        st.button(f"Show {min(state.QUEUE_PAGE_SIZE, len(view) - limit)} more ({len(view) - limit} left)",
                  key="more", on_click=show_more, icon=":material/expand_more:")

export = queue.assign(
    status=queue["review_id"].map(lambda i: "replied" if i in replied_ids else "waiting"),
    reply=queue["review_id"].map(lambda i: state.replied().get(i) or state.drafts().get(i))
    .fillna(queue["suggested_reply"]),
)
cols = [c for c in ["review_id", "date", "outlet", "platform", "rating", "priority_label", "status", "reply"]
        if c in export.columns]
st.download_button("Download reply log (CSV)", state.to_excel_csv(export[cols]), "reply_log.csv",
                   mime="text/csv", icon=":material/download:")
