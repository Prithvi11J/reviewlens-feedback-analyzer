"""Stage 1 demo: run the classification chain on the 3 sample reviews from the brief.

Run from the project root:
    python scripts/try_one_review.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reviewlens.chains import build_review_chain  # noqa: E402

SAMPLES = [
    {"review_id": "R-1041", "platform": "Zomato", "rating": 5,
     "review_text": "Biryani was amazing, packing also neat. Will order again!"},
    {"review_id": "R-1042", "platform": "Swiggy", "rating": 1,
     "review_text": "Ordered at 8pm, came at 9:10. Food fully cold. Very bad."},
    {"review_id": "R-1043", "platform": "Google", "rating": 2,
     "review_text": "Saapadu nalla irundhuchu but bill la extra charge pottanga"},
]


def main() -> None:
    chain = build_review_chain()
    for review in SAMPLES:
        analysis = chain.invoke(review)  # extra keys like review_id are simply ignored by the prompt
        # review_id comes from OUR data, not from the LLM, so it can never be mistyped.
        output = {"review_id": review["review_id"], **analysis.model_dump()}
        print(json.dumps(output, indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
