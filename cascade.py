"""Where on the entailment scale does the judge actually earn its cost?

A cascade runs the cheap detector on everything and escalates only the cases it
is unsure about. What counts as unsure has to be read off the data: "escalate
the middle of the scale" was a guess made before any of these numbers existed,
and the guess is worth nothing against a measurement.

This is the diagnostic, not the simulation. It answers one question — sorted by
entailment's own score, which stretch of the scale does entailment get wrong
and the judge get right? Everything comes from the cache, so nothing is spent.
"""

from run_eval import DETECTORS
from src.halueval import build_splits
from src.harness import run
from src.metrics import compute
from collections.abc import Callable

# The cheap stage and the expensive one, named as the harness names them.
# Text similarity is not a candidate: at 36% it has no signal to route on.
CHEAP = "entailment"
JUDGE = "llm_judge"

# Tenths of the scale. Ten buckets over 300 cases leaves about 30 per bucket —
# few enough to be noisy, which is why the count is printed beside every rate.
BUCKETS = 10

# ---- Reading the two detectors back out of the cache -----------------------

def results_for(name: str, cases) -> dict:
    """Cached verdicts for one detector, keyed by case id.

    The detector is taken from run_eval's list rather than constructed here:
    its threshold is part of the cache key, and a second copy of that number
    would eventually disagree with the first without anything failing.
    """
    detector = next(d for d in DETECTORS if d.name == name)

    # Zero budget: on a cache miss the harness stops instead of calling an API.
    return run(detector, cases, token_budget=0)

# ---- Sorting the cases by what the cheap stage thought ---------------------

def bucket_of(score: float) -> int:
    """Which tenth of the scale a score falls in; 0.00-0.10 is bucket 0."""
    # Clamped at both ends: a score of exactly 1.0 would land in bucket 10,
    # which does not exist, and cosine scores can dip below zero.
    return min(max(int(score * BUCKETS), 0), BUCKETS - 1)


def correct(result, case) -> bool:
    """Did this detector agree with the human label on this case?"""
    return result.verdict == case.label

# ---- What a judge call costs, measured by analyze.py on these same cases ---

# Dollars per 1,000 checks if every one of them reached the judge. At 752 input
# and 302 output tokens per check, priced at $0.15 and $0.60 per million.
JUDGE_COST_PER_1000 = 0.2943

# Median milliseconds per check. The cheap stage runs on everything, so its
# time is always paid; the judge's is paid only on what is escalated.
CHEAP_MS = 55
JUDGE_MS = 1091

# ---- Replaying the recorded verdicts under one routing rule ----------------

def route(cases, cheap_results, judge_results, escalate):
    """Build the verdicts a cascade would have produced, and its escalation rate.

    Nothing runs: every verdict already exists for every case, so a cascade is
    a choice between two recorded answers rather than a new computation.

    `escalate` takes the cheap stage's score and returns whether this case
    should go to the judge.
    """
    chosen, escalated = {}, 0

    for case in cases:
        cheap = cheap_results[case.case_id]

        if escalate(cheap.score):
            chosen[case.case_id] = judge_results[case.case_id]
            escalated += 1
        else:
            chosen[case.case_id] = cheap

    return chosen, escalated / len(cases)

def above(threshold: float) -> Callable[[float], bool]:
    """A rule escalating every score at or above `threshold`.

    A function that builds a function. Written this way rather than as a
    lambda with a default argument: each call gets its own `threshold`, so
    the eleven rules cannot end up sharing one value — the closure mistake
    that would have made every row of the table identical.
    """
    return lambda score: score >= threshold


# ---- The diagnostic: who is right, where -----------------------------------

if __name__ == "__main__":
    _, cases = build_splits()

    cheap_results = results_for(CHEAP, cases)
    judge_results = results_for(JUDGE, cases)

    # Every case lands in exactly one bucket, chosen by the cheap stage's score.
    buckets = [[] for _ in range(BUCKETS)]
    for case in cases:
        buckets[bucket_of(cheap_results[case.case_id].score)].append(case)

    print(f"\n{'entailment score':<18}{'n':>5}{'really wrong':>14}"
          f"{'entailment right':>18}{'judge right':>13}{'judge gains':>13}")

    for index, bucket in enumerate(buckets):
        low, high = index / BUCKETS, (index + 1) / BUCKETS
        if not bucket:
            print(f"{low:.1f}-{high:.1f}{'':<12}{0:>5}")
            continue

        # The base rate: a bucket holding no hallucinations needs no judge,
        # however often the judge happens to be right in it.
        hallucinated = sum(c.label == "hallucinated" for c in bucket) / len(bucket)
        cheap_acc = sum(correct(cheap_results[c.case_id], c) for c in bucket) / len(bucket)
        judge_acc = sum(correct(judge_results[c.case_id], c) for c in bucket) / len(bucket)

        print(f"{low:.1f}-{high:.1f}{'':<12}{len(bucket):>5}{hallucinated:>13.0%}"
              f"{cheap_acc:>17.0%}{judge_acc:>12.0%}{judge_acc - cheap_acc:>+13.0%}")

    # ---- The simulation: one row per rule ---------------------------------

    print(f"\n\n{'rule':<26}{'escalated':>11}{'acc':>8}{'prec':>8}{'rec':>8}"
          f"{'F1':>8}{'$/1000':>10}{'wait':>9}")

    # "Trust the cheap stage below T, escalate everything above it" — the shape
    # the diagnostic supports. T from 0.0 to 1.0 walks the whole range from
    # pure judge to pure entailment.
    rules: list[tuple[str, Callable[[float], bool]]] = [
        (f"escalate above {t / 10:.1f}", above(t / 10))
        for t in range(0, 11)
    ]


    # The guess the diagnostic rejected, kept so the report can show it was
    # tested rather than dismissed: escalate a band around the detector's own
    # cut-off and trust the cheap stage at both ends of the scale.
    rules.append(("escalate 0.26-0.66 only",
                  lambda s: 0.26 <= s <= 0.66))

    for label, escalate in rules:
        chosen, share = route(cases, cheap_results, judge_results, escalate)
        m = compute(cases, chosen)
        wait = CHEAP_MS + share * JUDGE_MS

        print(f"{label:<26}{share:>10.0%}{m.accuracy:>8.3f}{m.precision:>8.3f}"
              f"{m.recall:>8.3f}{m.f1:>8.3f}{share * JUDGE_COST_PER_1000:>10.4f}"
              f"{wait:>8.0f}ms")


