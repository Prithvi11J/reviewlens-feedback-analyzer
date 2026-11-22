"""The fixed structure every analysed review must follow.

`Literal[...]` lists the ONLY values a field may hold. This schema is used twice:
  1. Sent to OpenAI as a JSON schema, so the model can only produce allowed values.
  2. Used by Pydantic to validate the reply on our side (a second safety net).

The `description=` text is also sent to the model. It acts as a per-field
instruction, so writing good descriptions is part of prompt engineering.
"""

from typing import Literal, get_args

from pydantic import BaseModel, Field

Sentiment = Literal["positive", "neutral", "negative"]
Category = Literal["food_quality", "delivery", "price", "staff", "packaging", "app"]
Urgency = Literal["low", "medium", "high"]
Language = Literal["English", "Tamil", "Tanglish", "Other"]

# Plain lists of the allowed values, handy for validation, charts and filters.
SENTIMENTS: tuple[str, ...] = get_args(Sentiment)
CATEGORIES: tuple[str, ...] = get_args(Category)
URGENCIES: tuple[str, ...] = get_args(Urgency)
LANGUAGES: tuple[str, ...] = get_args(Language)


class ReviewAnalysis(BaseModel):
    """Structured analysis of a single customer review."""

    sentiment: Sentiment = Field(
        description="Overall feeling of the customer about their experience."
    )
    category: Category = Field(
        description=(
            "The ONE main topic of the review: food_quality (taste, freshness, portion), "
            "delivery (late, cold on arrival, rider), price (bill, extra charges, value), "
            "staff (behaviour, service), packaging (spills, containers), "
            "app (ordering, payment or tracking problems)."
        )
    )
    urgency: Urgency = Field(
        description=(
            "high = needs a reply today. ONLY these situations: "
            "(1) health or safety: food poisoning, illness, spoiled/stale/rotten food, hair or foreign object; "
            "(2) money taken wrongly: refund demand, double charge, payment debited but order failed; "
            "(3) abusive, rude or hostile staff (shouting, insulting, arguing with the customer); "
            "(4) wrong or missing order (received someone else's food, or nothing at all). "
            "medium = any other clear complaint, however angry or emotional: late or cold delivery "
            "(even very late), slow service or being ignored, high prices or hidden charges, "
            "small portions, bad taste, packaging, app bugs. "
            "low = praise, neutral comments or minor suggestions."
        )
    )
    language: Language = Field(
        description=(
            "Language the review is written in. Tanglish = Tamil words written in "
            "English letters, often mixed with English words."
        )
    )
    summary: str = Field(
        description="ONE short sentence in English describing what the customer said, "
        "even when the review is in Tamil or Tanglish."
    )
    suggested_reply: str = Field(
        description="A polite, ready-to-post reply from the business, in English, "
        "2 to 3 sentences, that addresses the customer's specific point."
    )
