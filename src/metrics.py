"""Compute evaluation metrics by comparing detector verdicts against labels.

Every metric here is a fraction built from four counts, so the counts are
computed first and kept in the result. They are needed for the failure
analysis anyway: "misses a lot" and "cries wolf a lot" are different problems
with different consequences, and a single accuracy number hides which one a
detector has.

scikit-learn would produce the same figures, but it requires naming the
positive class through a parameter, and getting that wrong swaps precision
and recall silently — nothing fails, the numbers just answer other questions.
"""

from dataclasses import dataclass, replace # replace() makes a copy of a dataclass with some fields changed

from src.base import DetectorResult
from src.halueval import TestCase

# Which verdict counts as "positive" — the thing being detected. Kept in one
# named place because the choice decides what precision and recall mean, and
# an inconsistency between two parts of the code would not raise an error.
POSITIVE = "hallucinated"


@dataclass
class Metrics:
    total: int  # cases actually scored — excludes any the detector failed on

    # The four outcomes. Every scored case lands in exactly one of them.
    true_positives: int   # hallucination flagged — caught
    false_positives: int  # grounded answer flagged — a false alarm
    false_negatives: int  # hallucination passed — reached the user
    true_negatives: int   # grounded answer passed — correct

    accuracy: float   # right verdicts / all verdicts
    precision: float  # caught / everything flagged
    recall: float     # caught / all hallucinations there were
    f1: float         # one number balancing precision against recall


def _safe_divide(numerator: float, denominator: float) -> float:
    """Divide two numbers, returning 0.0 instead of crashing on a zero divisor.

    Every metric in this file is a fraction, and a zero denominator is a real
    outcome rather than a bug: precision divides by how many times the detector
    flagged anything, which is zero if a badly chosen threshold made it flag
    nothing. Returning 0.0 keeps the run alive, and the counts stored beside
    the metric make it obvious which situation produced it.
    """
    return numerator / denominator if denominator else 0.0


def compute(
    cases: list[TestCase],
    results: dict[str, DetectorResult],
) -> Metrics:
    """Score one detector against the known labels.

    Sorts every case into one of four outcomes — caught, missed, false alarm,
    correct pass — and derives accuracy, precision, recall and F1 from those
    counts. Both the counts and the metrics are returned, because the counts
    show *how* a detector is wrong and a single accuracy figure cannot.

    Cases the detector failed on are absent from `results` and are skipped
    rather than counted as either right or wrong.
    """
    tp = fp = fn = tn = 0

    # Iterating over cases rather than results, because cases are the complete
    # picture: a case the detector failed on is missing from results, and going
    # the other way would hide that it was ever attempted.
    for case in cases:
        result = results.get(case.case_id)

        # Absent means the detector failed on this case. Skipped rather than
        # counted, because calling it right or wrong would invent an outcome
        # that never happened. `total` below reflects the skip.
        if result is None:
            continue

        # `==` yields True or False, so these hold booleans rather than text.
        # Two named booleans read better than four string comparisons and are
        # harder to mistype.
        predicted_positive = result.verdict == POSITIVE
        actually_positive = case.label == POSITIVE

        if predicted_positive and actually_positive:
            tp += 1
        elif predicted_positive and not actually_positive:
            fp += 1
        elif not predicted_positive and actually_positive:
            fn += 1
        else:
            tn += 1

    total = tp + fp + fn + tn

    # Computed before the return because f1 is built from both of them.
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)

    return Metrics(
        total=total,
        true_positives=tp,
        false_positives=fp,
        false_negatives=fn,
        true_negatives=tn,
        accuracy=_safe_divide(tp + tn, total),
        precision=precision,
        recall=recall,
        # The harmonic mean, not the plain average: it stays low unless both
        # halves are decent. That is the point — it stops a detector scoring
        # well by flagging everything (perfect recall, hopeless precision) or
        # by flagging only the blatant cases (good precision, hopeless recall).
        f1=_safe_divide(2 * precision * recall, precision + recall),
    )

def _apply_threshold(
    results: dict[str, DetectorResult], threshold: float
) -> dict[str, DetectorResult]:
    """Answer "what would the verdicts be at this threshold?" without re-running.

    Returns a copy of every result with its verdict re-derived from its stored
    score. A result scoring 0.42 reads as "hallucinated" at a threshold of 0.5
    and as "grounded" at 0.3 — same measurement, different reading. The score
    is what the detector actually measured; the verdict is only an
    interpretation of it, so moving the line costs no computation.
    """
    rescored = {}

    # .items() walks a dictionary giving both key and value on each pass;
    # plain `for x in results` would give only the keys.
    for case_id, result in results.items():
        # replace() returns a *copy* with one field changed. Assigning to
        # result.verdict directly would alter the original, and the sweep
        # relies on every candidate threshold starting from the same data.
        rescored[case_id] = replace(
            result,
            verdict="grounded" if result.score >= threshold else POSITIVE,
        )

    return rescored


def tune_threshold(
    cases: list[TestCase],
    results: dict[str, DetectorResult],
) -> tuple[float, Metrics]:
    """Find the threshold that performs best, and return it with its metrics.

    Tries every hundredth from 0.00 to 1.00, re-reading the stored scores at
    each one, and keeps whichever scored highest.

    Accuracy decides the winner, not F1, and the reason is specific to this
    dataset being balanced 50/50. Both degenerate strategies — flag everything,
    flag nothing — score exactly 0.5 accuracy, so any result above that means
    real signal. F1 is not so clean here: flagging everything yields precision
    0.5 and recall 1.0, hence F1 of 0.667, which looks respectable while the
    detector does nothing at all.

    In production, where a missed hallucination and a false alarm cost
    different amounts, the target would instead be weighted toward whichever
    error is more expensive. That is a product decision, not a technical one.

    Run this on the dev split only. Tuning on the test split and then reporting
    on it inflates the result invisibly — nothing fails, the numbers simply
    come out better than they are.
    """

    best_threshold = 0.0 
    best_metrics = compute(cases, _apply_threshold(results, 0.0)) 

    # Stepping over whole numbers and dividing, rather than adding 0.01
    # repeatedly: accumulated floating-point error would make the last
    # candidates land slightly off their intended values.
    for step in range(101):
        threshold = step / 100
        candidate = compute(cases, _apply_threshold(results, threshold))

        if candidate.accuracy > best_metrics.accuracy:
            best_threshold = threshold
            best_metrics = candidate

    return best_threshold, best_metrics

def by_subset(
    cases: list[TestCase],
    results: dict[str, DetectorResult],
) -> dict[str, Metrics]:
    """Score each subset separately, returning one Metrics per subset name.

    A single averaged figure hides what matters here: the subsets differ in how
    answers are written — terse in QA, paragraph-length in summarization — and
    a detector can behave oppositely on them. Averaging would report neither
    behaviour.
    """
    subsets = sorted({case.subset for case in cases})

    per_subset = {} # subset name → metrics
    for subset in subsets:
        subset_cases = [case for case in cases if case.subset == subset] # 
        per_subset[subset] = compute(subset_cases, results) 

    return per_subset



# check:
if __name__ == "__main__":
    from src.embeddings import EmbeddingDetector
    from src.entailment import EntailmentDetector
    from src.halueval import build_splits
    from src.harness import run

    dev, _ = build_splits() 

    # The whole dev split this time — 80 cases, both subsets. The previous
    # run took only the first 40, which are all QA.
    results = run(EntailmentDetector(), dev)

    print("\n--- at the placeholder threshold of 0.5 ---")
    m = compute(dev, results)
    print(f"accuracy {m.accuracy:.3f}   f1 {m.f1:.3f}")
    print(f"caught {m.true_positives}, missed {m.false_negatives}, "
          f"false alarms {m.false_positives}, correct passes {m.true_negatives}")

    print("\n--- per subset, still at 0.5 ---")
    for subset, sm in by_subset(dev, results).items():
        print(f"{subset:16} accuracy {sm.accuracy:.3f}   f1 {sm.f1:.3f}   "
              f"(n={sm.total})")
        # Counts per subset, not just the ratios: an accuracy of exactly 0.500
        # can mean "half right" or "flagged nothing", and only the counts say which.
        print(f"{'':16} caught {sm.true_positives}, missed {sm.false_negatives}, "
              f"false alarms {sm.false_positives}, correct passes {sm.true_negatives}")

    print("\n--- after tuning on dev ---")
    threshold, tuned = tune_threshold(dev, results)
    print(f"best threshold {threshold:.2f}")
    print(f"accuracy {tuned.accuracy:.3f}   f1 {tuned.f1:.3f}")
    print(f"caught {tuned.true_positives}, missed {tuned.false_negatives}, "
          f"false alarms {tuned.false_positives}, correct passes {tuned.true_negatives}")