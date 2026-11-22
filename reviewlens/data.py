"""Load and validate the reviews CSV (step 02 'Validate' in the brief).

Rule of thumb: never send dirty data to an LLM. Empty rows waste money,
duplicates double-count complaints, and a missing column should fail
loudly here instead of deep inside the pipeline.
"""

from dataclasses import dataclass, field

import pandas as pd

REQUIRED_COLUMNS = ["review_id", "date", "platform", "rating", "review_text"]
# Extra columns we use when present, but never require (the brief's format has only the 5 above).
# `outlet` lets a multi-branch restaurant compare its branches.
OPTIONAL_COLUMNS = ["outlet"]


class InvalidCSVError(ValueError):
    """The uploaded file cannot be analysed (missing columns, nothing usable...)."""


@dataclass
class ValidationReport:
    rows_in_file: int
    empty_text_dropped: int = 0
    duplicates_dropped: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def rows_kept(self) -> int:
        return self.rows_in_file - self.empty_text_dropped - self.duplicates_dropped


def validate_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
    """Return a clean copy of `df` plus a report of what was removed and why."""
    df = df.copy()
    # Forgive harmless header differences like " Review_Text ".
    df.columns = [str(c).strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise InvalidCSVError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    report = ValidationReport(rows_in_file=len(df))
    df = df[REQUIRED_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]]

    df["review_id"] = df["review_id"].astype(str).str.strip()
    df["review_text"] = df["review_text"].fillna("").astype(str).str.strip()
    for col in ["platform", *[c for c in OPTIONAL_COLUMNS if c in df.columns]]:
        df[col] = df[col].fillna("Unknown").astype(str).str.strip()

    empty = df["review_text"] == ""
    report.empty_text_dropped = int(empty.sum())
    df = df[~empty]

    dupes = df.duplicated(subset="review_id", keep="first")
    report.duplicates_dropped = int(dupes.sum())
    df = df[~dupes]

    # Bad ratings/dates become NaN/NaT instead of crashing; the text is what matters.
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    bad_ratings = int(df["rating"].isna().sum())
    if bad_ratings:
        report.notes.append(f"{bad_ratings} row(s) have a missing or invalid rating.")

    if df.empty:
        raise InvalidCSVError("No reviews with text were found in the file.")

    return df.reset_index(drop=True), report


def load_reviews(source) -> tuple[pd.DataFrame, ValidationReport]:
    """Read a CSV from a path or an uploaded file object, then validate it."""
    # utf-8-sig reads files with or without Excel's BOM.
    df = pd.read_csv(source, encoding="utf-8-sig", dtype={"review_id": str})
    return validate_reviews(df)
