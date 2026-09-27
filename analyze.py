"""Score every detector against the labels, overall and per subset.

Reads what the runs already wrote. `run()` is called with a token budget of
zero, and harness.py checks the budget only on a cache miss, so nothing here
can reach an API: the judge's 300 test verdicts cost a day of quota to produce
and are never recomputed.

Step 1 of the analysis. Cost, latency, failure modes and the cascade each get
their own pass, so a wrong number is visible when it appears rather than buried
in one wall of output.
"""

from run_eval import DETECTORS
from src.halueval import build_splits
from src.harness import run
from src.metrics import Metrics, by_subset, compute
import statistics
from transformers import AutoTokenizer
from src.llm_judge import MODEL_NAME, SYSTEM_PROMPT, _build_user_message


# dev was used to tune the thresholds, so scoring on it would report numbers
# the tuning itself produced. test was never tuned against, and is the only
# honest measurement of these detectors.
SPLIT = "test"

# Zero on purpose. This is a safety belt, not a setting.
NO_SPENDING = 0

# ---- What a check costs, and what each method is called on screen ----------

# Groq's list price for openai/gpt-oss-120b, checked 2026-09-27. Output costs
# four times input, so the two are never blended into one rate. A third tier
# exists — cached input at half price — but only the system prompt repeats
# between calls, so full price is the honest assumption here.
PRICE_IN = 0.15 / 1_000_000    # dollars per input token
PRICE_OUT = 0.60 / 1_000_000   # dollars per output token

# The names the interface shows. Three naming systems are in play — the module
# name, the model name and the screen label — and only the screen label is one
# a reader has actually looked at.
LABELS = {
    "embeddings": "Text similarity",
    "entailment": "Entailment",
    "llm_judge": "AI judge",
}


HEADER = (f"{'':<14}{'n':>5}{'acc':>7}{'prec':>7}{'rec':>7}{'F1':>7}"
          f"{'caught':>9}{'alarms':>8}{'missed':>8}{'passed':>8}")


def row(name: str, m: Metrics) -> str:
    """One line of the table, widths matching HEADER."""
    return (f"{name:<14}{m.total:>5}{m.accuracy:>7.3f}{m.precision:>7.3f}"
            f"{m.recall:>7.3f}{m.f1:>7.3f}{m.true_positives:>9}"
            f"{m.false_positives:>8}{m.false_negatives:>8}"
            f"{m.true_negatives:>8}")

# ---- Statistics that do not depend on a convention -------------------------
def percentile(values: list[float], fraction: float) -> float:
    """The value below which `fraction` of the sample falls.

    Written out rather than taken from statistics.quantiles: that function
    interpolates between neighbouring points by one of several conventions,
    and a p90 whose value depends on which convention was picked is not a
    number worth publishing.
    """
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]

# ---- Recovering the input/output split the cache never recorded ------------

def judge_token_split(cases, results) -> tuple[list[int], list[int]]:
    """Input and output tokens per judge call, without calling anything.

    The cache records only the total, and the two halves are priced
    differently, so a blended rate would be wrong. The input side is rebuilt
    exactly — the same two messages the judge sends, serialized with the
    model's own chat template — and output is whatever is left of the total.
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    inputs, outputs, impossible = [], [], 0
    for case in cases:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_message(case)},
        ]
        # add_generation_prompt=True includes the tokens that open the reply,
        # which the API also counts on the input side.
        counted = len(tokenizer.apply_chat_template(
            messages, add_generation_prompt=True))

        total = results[case.case_id].tokens_used
        remainder = total - counted

        # A negative remainder means our count came out larger than what Groq
        # charged for in total, so the reconstruction is wrong somewhere.
        # Counted rather than clamped silently: a couple is rounding, many is
        # a bug, and the difference must be visible.
        if remainder < 0:
            impossible += 1
            remainder = 0

        inputs.append(counted)
        outputs.append(remainder)

    if impossible:
        print(f"  WARNING: {impossible} cases counted more input tokens than "
              f"the API charged in total — the split is unreliable")

    return inputs, outputs


# ---- Second table: how fast, and what 1,000 checks would cost --------------

def report_speed_and_cost(cases, results_by_detector) -> None:
    """Latency and list-price cost, one row per method.

    Median and p90 rather than the mean: one slow network call drags a mean
    upwards and misreports the wait a person actually sees.
    """
    print(f"\n\n{'':<18}{'median':>10}{'p90':>10}{'in':>8}{'out':>8}"
          f"{'$/1000':>10}")

    for name, results in results_by_detector.items():
        latencies = [r.latency_ms for r in results.values()]
        median = statistics.median(latencies)
        p90 = percentile(latencies, 0.90)

        if name == "llm_judge":
            inputs, outputs = judge_token_split(cases, results)
            mean_in = statistics.mean(inputs)
            mean_out = statistics.mean(outputs)
            price = (mean_in * PRICE_IN + mean_out * PRICE_OUT) * 1_000
        else:
            # Local models send nothing anywhere, so there is nothing to price.
            # What they cost is memory and CPU, which the length limit measures.
            mean_in = mean_out = price = 0

        print(f"{LABELS[name]:<18}{median:>8.0f}ms{p90:>8.0f}ms"
              f"{mean_in:>8.0f}{mean_out:>8.0f}{price:>10.4f}")



if __name__ == "__main__":
    dev, test = build_splits()
    cases = {"dev": dev, "test": test}[SPLIT]

    print(f"{SPLIT} split: {len(cases)} cases")

    # ---- Collected so the second table can be built after the first --------
    results_by_detector = {} # detector.name -> case_id -> DetectorResult


    for detector in DETECTORS:
        results = run(detector, cases, token_budget=NO_SPENDING)

        # Every figure below divides by this count. A short run would report
        # metrics for a different set of cases than the heading claims, and
        # nothing about the output would look wrong.
        if len(results) != len(cases):
            print(f"  INCOMPLETE: {len(results)} of {len(cases)} — skipped")
            continue

        print(f"\n=== {LABELS[detector.name]} — {detector.name} "
              f"{detector.version} ===")
        print(HEADER)
        print(row("overall", compute(cases, results)))
        for subset, metrics in by_subset(cases, results).items():
            print(row(subset, metrics))
        # Keyed by name, so the two embedding thresholds do not both claim a
        # row: latency and cost belong to the model, not to a threshold, and
        # reporting them twice would state the same fact twice.
        results_by_detector[detector.name] = results
    
    report_speed_and_cost(cases, results_by_detector)

