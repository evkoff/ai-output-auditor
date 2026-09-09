"""Run the LLM judge over a whole split and report what it cost.

Lives outside src/ because it is a tool for running the evaluation, not part
of the library — same category as check_groq.py and measure_tokens.py.

Used four times across the project: once over the dev split as a pilot, then
over the test split across two or three days, since the daily token cap forces
the run to stop partway and resume tomorrow.
"""

from src.halueval import build_splits
from src.harness import run
from src.llm_judge import LLMJudgeDetector

# Which split to run. "dev" is the pilot — the only split whose results may be
# looked at and reacted to. "test" is the final measurement and must not run
# until the prompt is settled.
SPLIT = "dev"

# What this run may spend. The harness resets its counter on every call, so a
# second run on the same day must be given what is actually left rather than
# the full daily allowance.
TOKEN_BUDGET = 93_000


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

    results = run(LLMJudgeDetector(), cases, token_budget=TOKEN_BUDGET) 

    # The harness prints its own summary; this line answers the one question it
    # does not — whether every case actually came back with a verdict.
    print(f"\n{len(results)} of {len(cases)} cases have a verdict")
