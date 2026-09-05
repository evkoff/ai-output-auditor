"""Detector 2: NLI entailment with HHEM-2.1-Open.

Reads source and response *together* and returns how strongly the source
supports the response. Unlike detector 1 this compares claims rather than
topics, so a swapped name or number moves the score even when the surrounding
wording is nearly identical.

Runs locally and free like detector 1, but without its 256-token ceiling:
HHEM-2.1 takes unlimited context, so it sees the whole source.

Requires transformers < 5. The model ships its own loading code through
trust_remote_code and that code targets the v4 API — see the execution plan.
"""

import time

from transformers import AutoModelForSequenceClassification # model loader for HuggingFace Transformers

from src.base import DetectorResult
from src.halueval import TestCase

# Hughes Hallucination Evaluation Model, Apache 2.0, ~110M parameters.
# Trained for exactly one question: is this text supported by that text.
MODEL_NAME = "vectara/hallucination_evaluation_model"


class EntailmentDetector:
    # Required by the Detector protocol. Bumping `version` invalidates the
    # cache, though only detector 3 actually spends anything on a re-run.
    name = "entailment"
    version = "v1"

    def __init__(self, threshold: float = 0.5):
        # PLACEHOLDER, same as detector 1 — the real value is tuned on dev.
        self.threshold = threshold

        # trust_remote_code=True permits the model to run Python it brings
        # with it, which is how it provides the .predict() method below.
        # Loading prints a warning about HHEMv2Config; it is harmless — the
        # scores match the values documented on the model card.
        self.model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
        )

    def check(self, case: TestCase) -> DetectorResult:
        start = time.perf_counter()

        # predict() takes a list of (premise, hypothesis) pairs and returns
        # one score per pair. Order matters: the premise — the text that must
        # do the supporting — comes first, the text under scrutiny - second.
        # We pass a single pair, so we unwrap the single result.
        scores = self.model.predict([(case.source, case.response)])
        score = scores[0].item()

        latency_ms = (time.perf_counter() - start) * 1000

        return DetectorResult(
            verdict="grounded" if score >= self.threshold else "hallucinated",
            score=score,
            latency_ms=latency_ms,
        )



# check:
if __name__ == "__main__":
    from src.halueval import build_splits

    dev, _ = build_splits()
    detector = EntailmentDetector()

    # Same four cases as detector 1, so the two outputs can be compared.
    for case in dev[:4]:
        result = detector.check(case)
        print(f"{case.case_id}  expected={case.label:13}  score={result.score:.3f}")
