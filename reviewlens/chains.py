"""LangChain chains used by ReviewLens.

Stage 1: the per-review classification chain (the MAP step).
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

from reviewlens.config import get_llm
from reviewlens.schemas import ReviewAnalysis

REVIEW_SYSTEM_PROMPT = """\
You are a customer-feedback analyst for a restaurant and food delivery business in Tamil Nadu.
You read one customer review at a time and classify it for the operations team.

Rules:
- Reviews may be in English, Tamil, or Tanglish (Tamil written in English letters).
  Understand them fully; always write the summary and reply in English.
- Choose the single category that best matches the MAIN complaint or praise.
- The star rating is a hint, but the review text decides sentiment
  (a 4-star review that describes a serious problem is still negative).
- Urgency is about WHAT HAPPENED, not how upset the customer sounds.
  Strong emotion ("worst ever", "kids slept hungry") does not make a complaint high.
  Use "high" only when the review fits one of the four situations in the urgency
  definition; when in doubt between high and medium, choose medium.
  Examples (not from any real review):
    "Food came 90 minutes late, totally cold, I am furious" -> medium (late delivery)
    "Nobody attended our table for half an hour" -> medium (slow service)
    "The paneer tasted sour, clearly gone bad" -> high (spoiled food)
    "Waiter called us cheap and laughed at us" -> high (abusive staff)
- The reply must sound human and specific, never promise compensation,
  and must not include placeholders like [Name]."""

REVIEW_HUMAN_PROMPT = """\
Platform: {platform}
Star rating: {rating} / 5
Review: {review_text}"""


def build_review_chain() -> Runnable:
    """prompt | model-with-schema  ->  returns a ReviewAnalysis object."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", REVIEW_SYSTEM_PROMPT),
            ("human", REVIEW_HUMAN_PROMPT),
        ]
    )
    # strict=True: OpenAI constrains generation to our JSON schema,
    # so values outside the Literal lists cannot be produced.
    structured_llm = get_llm().with_structured_output(ReviewAnalysis, strict=True)
    return prompt | structured_llm
