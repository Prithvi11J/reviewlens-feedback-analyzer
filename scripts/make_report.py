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
    # Found either in the top-complaints note ("mostly Google") or as a concentrated complaint ("on Google")
    "2. Price complaints mostly Tanglish / Google": lambda md: any(
        "Google" in line and "Tanglish" in md for line in md.splitlines()
        if ("Price" in line or "price" in line.lower()) and ("mostly Google" in line or "on Google" in line)),
    # Checks the code-written part of the cluster line ("all at Peelamedu"), not the AI's wording,
    # which varies between runs ("poisoning", "illness", "sick"...).
    "3. Fish-fry illness spike at Peelamedu": lambda md: any(
        "fish" in line.lower() and "all at Peelamedu" in line for line in md.splitlines()),
    "4. Packaging complaints mostly Peelamedu": lambda md: "Packaging complaints at the Peelamedu outlet" in md
        or "mostly Peelamedu outlet" in md,
}


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    enriched = pd.read_csv(path, encoding="utf-8-sig", dtype={"review_id": str})
    report = generate_report(compute_stats(enriched))

    out = ROOT / "outputs" / "weekly_report.md"
    out.write_text(report.markdown, encoding="utf-8")
    # The structured version (titles, clusters, actions) feeds the app's briefing cards.
    insights_out = ROOT / "outputs" / "weekly_insights.json"
    insights_out.write_text(report.insights.model_dump_json(indent=2), encoding="utf-8")
    print(report.markdown)
    print(f"\nSaved: {out} and {insights_out.name}")

    if path == DEFAULT:
        print("\nPlanted-pattern check:")
        for name, check in PLANTED_PATTERNS.items():
            print(f"  [{'FOUND' if check(report.markdown) else 'MISSED'}] {name}")


if __name__ == "__main__":
    main()
