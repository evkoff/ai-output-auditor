"""Load HaluEval and turn it into uniform test cases.

HaluEval ships one JSON object per line rather than a JSON array, and each
subset uses different field names: `knowledge`/`right_answer` in QA,
`document`/`right_summary` in summarization. This module hides both facts —
everything downstream works with `TestCase` and never touches the raw files.

Each source record holds a correct and a hallucinated version of the same
answer, so it expands into two test cases. That is what makes the sample
balanced 50/50 by construction, which in turn means a random baseline scores
50% and accuracy is directly interpretable.

The field mapping and the reason `dialogue_data.json` is excluded live in
docs/2026-08-31-ai-output-auditor-design.md, section 4.
"""

import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"



# Field names differ between subsets; this maps each one onto a common shape,
# so nothing downstream needs to know the original names.
SUBSETS = {
    "qa": {
        "file": "qa_data.json",
        "source": "knowledge",
        "grounded": "right_answer",
        "hallucinated": "hallucinated_answer",
        "context": "question",
    },
    "summarization": {
        "file": "summarization_data.json",
        "source": "document",
        "grounded": "right_summary",
        "hallucinated": "hallucinated_summary",
        "context": None,
    },
}


@dataclass
class TestCase:
    case_id: str       # "qa_00042_hallucinated"
    subset: str        # "qa" | "summarization"
    source: str        # the text the response must be grounded in
    response: str      # the answer being checked
    label: str         # ground truth: "grounded" | "hallucinated"
    context: str = ""  # the question, where the subset has one


def load_records(subset: str, limit: int | None = None) -> list[dict]:
    """Read up to `limit` raw records from one HaluEval file."""
    path = DATA_DIR / SUBSETS[subset]["file"]
    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            records.append(json.loads(line))
    return records



def to_test_cases(
    subset: str, 
    records: list[dict], 
    start_index: int = 0
) -> list[TestCase]:
    """Expand each record into two test cases: one grounded, one hallucinated.

    `start_index` keeps case_id unique when records have been sliced into
    separate splits — case_id is the cache key, and duplicates would silently
    serve the wrong cached verdict.
    """
    spec = SUBSETS[subset]
    cases = []

    for offset, record in enumerate(records):
        index = start_index + offset
        context = record[spec["context"]] if spec["context"] else ""

        for label in ("grounded", "hallucinated"):
            cases.append(
                TestCase(
                    case_id=f"{subset}_{index:05d}_{label}",
                    subset=subset,
                    source=record[spec["source"]],
                    response=record[spec[label]],
                    label=label,
                    context=context,
                )
            )

    return cases



def build_splits(
    dev_records: int = 20,
    test_records: int = 75,
) -> tuple[list[TestCase], list[TestCase]]:
    """Build dev and test splits, taking the same record counts from each subset.

    Splits by record, not by test case: the two cases from one record share a
    source text, so a case-level split would leak that text across the splits.
    """
    dev: list[TestCase] = []
    test: list[TestCase] = []

    for subset in SUBSETS:
        records = load_records(subset, limit=dev_records + test_records)

        dev.extend(
            to_test_cases(subset, records[:dev_records], start_index=0)
        )
        test.extend(
            to_test_cases(subset, records[dev_records:], start_index=dev_records)
        )

    return dev, test



def build_splits(
    dev_records: int = 20,
    test_records: int = 75,
) -> tuple[list[TestCase], list[TestCase]]:
    """Build dev and test splits, taking the same record counts from each subset.

    Splits by record, not by test case: the two cases from one record share a
    source text, so a case-level split would leak that text across the splits.
    """
    dev: list[TestCase] = []
    test: list[TestCase] = []

    for subset in SUBSETS:
        records = load_records(subset, limit=dev_records + test_records)

        dev.extend(
            to_test_cases(subset, records[:dev_records], start_index=0)
        )
        test.extend(
            to_test_cases(subset, records[dev_records:], start_index=dev_records)
        )

    return dev, test










# check:
if __name__ == "__main__":
    dev, test = build_splits()

    print(f"dev:  {len(dev)} cases")
    print(f"test: {len(test)} cases")

    # No record may appear in both splits: check that no source text is shared.
    dev_sources = {case.source for case in dev}
    test_sources = {case.source for case in test}
    print(f"overlapping sources: {len(dev_sources & test_sources)}")





