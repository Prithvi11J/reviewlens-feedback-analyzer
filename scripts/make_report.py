"""Stage 4 demo: enriched CSV -> exact stats -> AI-written weekly report.

Run from the project root, after scripts/run_batch.py:
    python scripts/make_report.py
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reviewlens.aggregate import compute_stats  # noqa: E402
from reviewlens.report import generate_report  # noqa: E402

DEFAULT = ROOT / "outputs" / "reviews_week47_enriched.csv"

# The patterns planted by scripts/make_test_data.py, as simple text checks.
PLANTED_PATTERNS = {
    "1. Delivery complaints mostly Swiggy": lambda md: "mostly Swiggy" in md,
    "2. Price complaints mostly Tanglish / Google": lambda md: "mostly Tanglish" in md and "mostly Google" in md,
    "3. Fish-fry food-poisoning spike": lambda md: "fish" in md.lower() and ("poison" in md.lower() or "sick" in md.lower()),
}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    enriched = pd.read_csv(path, encoding="utf-8-sig", dtype={"review_id": str})
    report = generate_report(compute_stats(enriched))

    out = ROOT / "outputs" / "weekly_report.md"
    out.write_text(report.markdown, encoding="utf-8")
    print(report.markdown)
    print(f"\nSaved: {out}")

    if path == DEFAULT:
        print("\nPlanted-pattern check:")
        for name, check in PLANTED_PATTERNS.items():
            print(f"  [{'FOUND' if check(report.markdown) else 'MISSED'}] {name}")


if __name__ == "__main__":
    main()
