"""Run every detector over a whole split and report what each one cost.

Lives outside src/ because it is a tool for running the evaluation, not part
of the library — same category as check_groq.py and measure_tokens.py.

One command runs them all. Three of the four entries in DETECTORS are local
and free, finishing in seconds, so splitting them into separate commands would
only create a way to forget which detector was last run over which split.

Detector 1 runs at two thresholds, and the
reason is in the comment where DETECTORS is defined.
"""

from src.embeddings import EmbeddingDetector
from src.entailment import EntailmentDetector
from src.halueval import build_splits
from src.harness import run
from src.llm_judge import LLMJudgeDetector

# Which split to run. "dev" is the pilot — the only split whose results may be
# looked at and reacted to. "test" is the final measurement and must not run
# until the prompt is settled.
SPLIT = "test"

# What this run may spend. The harness resets its counter on every call, so a
# second run on the same day must be given what is actually left rather than
# the full daily allowance.
TOKEN_BUDGET = 180_000

# Thresholds are the values tuned on the dev split, applied to test unchanged.
#
# Detector 1 appears twice on purpose. Dev tuning selected 0.00 — "call
# everything grounded" — which on test yields exactly 0.500 by construction:
# correct by protocol, and a measurement of nothing. The naive 0.50 runs beside
# it as the figure anyone reaching for this approach would actually get. The
# threshold is part of the version string, so the two land under separate cache
# keys and cannot be confused.
#
# The judge is last so that the free detectors finish and are safely cached
# before anything spends quota.
DETECTORS = [
    EmbeddingDetector(threshold=0.00),
    EmbeddingDetector(threshold=0.50),
    EntailmentDetector(threshold=0.46),
    LLMJudgeDetector(),
]


if __name__ == "__main__":
    dev, test = build_splits()

    # Pick the split that SPLIT names. The dict is built and indexed in a
    # single expression: it exists only to turn a name into a list, so it
    # needs no variable of its own and no branching.
    #
    # Written this way rather than `dev if SPLIT == "dev" else test` because
    # that form treats every value except "dev" as "test" — a typo would run
    # the wrong split and spend the quota on it silently. Indexing a dict
    # raises KeyError on the spot instead.
    cases = {"dev": dev, "test": test}[SPLIT]

    print(f"running the judge over {len(cases)} {SPLIT} cases")
    print(f"budget for this run: {TOKEN_BUDGET:,} tokens\n")

    for detector in DETECTORS:
        print(f"\n--- {detector.name} {detector.version}")
        results = run(detector, cases, token_budget=TOKEN_BUDGET)
        # The harness prints its own summary; this line answers the one question it
        # does not — whether every case actually came back with a verdict.
        print(f"\n{len(results)} of {len(cases)} cases have a verdict")
