"""Stage 3 demo: compute the weekly numbers from an enriched CSV (no API calls, free).

Run from the project root, after scripts/run_batch.py:
    python scripts/show_stats.py
    python scripts/show_stats.py outputs/some_file_enriched.csv
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reviewlens.aggregate import compute_stats  # noqa: E402

DEFAULT = ROOT / "outputs" / "reviews_week47_enriched.csv"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    enriched = pd.read_csv(path, encoding="utf-8-sig", dtype={"review_id": str})
    s = compute_stats(enriched)

    pct = s.sentiment_pct
    print(f"PERIOD {s.period_start} to {s.period_end} | reviews: {s.total_reviews} "
          f"| needs_review: {s.needs_review} | avg rating: {s.average_rating}")
    print(f"Sentiment: positive {pct['positive']}%  neutral {pct['neutral']}%  negative {pct['negative']}%"
          f"   (sum = {sum(pct.values())}%)")
    print(f"Categories: {s.category_counts}")
    print(f"Languages : {s.language_counts}")
    print(f"Urgency   : {s.urgency_counts}")

    for title, themes in [("TOP COMPLAINTS", s.top_complaints), ("TOP PRAISE", s.top_praise)]:
        print(f"\n{title}")
        for i, t in enumerate(themes, 1):
            print(f"  {i}. {t.category:<13} {t.count:>3} reviews  (mostly {t.top_platform} "
                  f"{t.top_platform_share_pct:.0f}%, languages {t.language_mix})")
            for ex in t.example_summaries[:2]:
                print(f"       - {ex}")

    print(f"\nURGENT ({len(s.urgent_reviews)} reviews, by date)")
    for u in s.urgent_reviews:
        print(f"  {u.date}  {u.platform:<7} {u.category:<13} {u.summary[:80]}")

    print("\nThis is exactly what the report LLM will receive in Stage 4:")
    print(s.model_dump_json(indent=1)[:600] + " ...")


if __name__ == "__main__":
    main()
