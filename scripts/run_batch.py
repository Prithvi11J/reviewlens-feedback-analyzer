"""Stage 2 demo: validate a CSV, analyse every review, save the enriched table,
and (if an answer key exists) measure how accurate the tagging is.

Run from the project root:
    python scripts/run_batch.py                       # uses the week-47 sample
    python scripts/run_batch.py path/to/reviews.csv
    python scripts/run_batch.py --eval-only          # re-score the last run, no API calls
"""

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from reviewlens.analyze import STATUS_OK, analyze_reviews  # noqa: E402
from reviewlens.data import load_reviews  # noqa: E402

DEFAULT_CSV = ROOT / "data" / "sample" / "reviews_week47.csv"


def main() -> None:
    args = sys.argv[1:]
    if args and args[0] == "--eval-only":
        evaluate_saved(Path(args[1]) if len(args) > 1 else DEFAULT_CSV)
        return
    csv_path = Path(args[0]) if args else DEFAULT_CSV
    df, report = load_reviews(csv_path)
    print(f"File: {csv_path.name} | rows: {report.rows_in_file} | kept: {report.rows_kept} "
          f"| empty dropped: {report.empty_text_dropped} | duplicates dropped: {report.duplicates_dropped}")
    for note in report.notes:
        print(f"Note: {note}")

    start = time.perf_counter()
    enriched = analyze_reviews(df, on_progress=lambda d, t: print(f"  analysed {d}/{t}", flush=True))
    print(f"Done in {time.perf_counter() - start:.0f}s")

    ok = enriched["analysis_status"] == STATUS_OK
    print(f"OK: {ok.sum()} | needs_review: {(~ok).sum()}")

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    out_file = out_dir / f"{csv_path.stem}_enriched.csv"
    enriched.to_csv(out_file, index=False, encoding="utf-8-sig")  # BOM => Tamil shows correctly in Excel
    print(f"Saved: {out_file}")

    labels_file = csv_path.with_name(f"{csv_path.stem}_labels.csv")
    if labels_file.exists():
        evaluate(enriched, pd.read_csv(labels_file, encoding="utf-8-sig", dtype={"review_id": str}))


def evaluate_saved(csv_path: Path) -> None:
    """Re-score an earlier run against the answer key without calling the API (free)."""
    enriched = pd.read_csv(ROOT / "outputs" / f"{csv_path.stem}_enriched.csv",
                           encoding="utf-8-sig", dtype={"review_id": str})
    labels = pd.read_csv(csv_path.with_name(f"{csv_path.stem}_labels.csv"),
                         encoding="utf-8-sig", dtype={"review_id": str})
    evaluate(enriched, labels)


def evaluate(enriched: pd.DataFrame, labels: pd.DataFrame) -> None:
    """Compare the LLM's tags with the answer key."""
    merged = enriched.merge(labels, on="review_id", suffixes=("", "_expected"))
    print(f"\nAccuracy vs answer key ({len(merged)} reviews):")
    fields = [c for c in ["category", "sentiment", "language", "urgency"] if c in labels.columns]
    for col in fields:
        print(f"  {col:<10} {(merged[col] == merged[f'{col}_expected']).mean():.1%}")

    for col in ["category", "urgency"]:
        if col in fields:
            show_disagreements(merged, col)

    print(f"\nUrgency split: {enriched['urgency'].value_counts().to_dict()}")


def show_disagreements(merged: pd.DataFrame, col: str) -> None:
    wrong = merged[merged[col] != merged[f"{col}_expected"]]
    if wrong.empty:
        return
    print(f"\n{col.capitalize()} disagreements (expected -> got):")
    pairs = wrong.groupby([f"{col}_expected", col]).size().sort_values(ascending=False)
    for (exp, got), n in pairs.items():
        examples = wrong[(wrong[f"{col}_expected"] == exp) & (wrong[col] == got)]["review_text"]
        for text in examples.drop_duplicates().head(3):
            print(f"  {n:>3} x {exp} -> {got}   \"{text[:75]}\"")
            n = ""  # print the count only on the first line of each group


if __name__ == "__main__":
    main()
