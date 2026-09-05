"""Detector 1: embedding similarity — the baseline.

Encodes source and response separately and compares the two vectors, which
measures whether the texts are about the same thing rather than whether the
response is supported by the source. Expected to be the weakest of the three;
it is here as a point of reference, because "HHEM scores 0.85" means nothing
without something to compare it against.

Known limitation: the model truncates input at 256 word pieces, so on a long
source it sees only the beginning. See the design doc, detector 1.
"""

import time # needed for measuring latency

from sentence_transformers import SentenceTransformer, util # SentenceTransformer is a model calss, util has cosine similarity function

from src.base import DetectorResult
from src.halueval import TestCase

# The standard small sentence-embedding model: 22M parameters, runs on CPU.
# Chosen because it is the obvious default — a baseline should show what you
# get reaching for the first thing available, not an optimised pick.
MODEL_NAME = "all-MiniLM-L6-v2"


class EmbeddingDetector:
    # Required by the Detector protocol in base.py. `version` becomes part of
    # the cache key, so bumping it invalidates previously cached results.
    name = "embeddings"
    version = "v1"

    def __init__(self, threshold: float = 0.5):
        # PLACEHOLDER value. The real threshold is tuned on the dev split;
        # 0.5 is here only so the detector runs before that happens, and no
        # result produced with it means anything.
        self.threshold = threshold

        # Loading reads ~90 MB and takes seconds, so it happens once per
        # detector instance — never inside check(), which runs 380 times.
        self.model = SentenceTransformer(MODEL_NAME)

    def check(self, case: TestCase) -> DetectorResult:
        # perf_counter rather than time(): it only moves forward, so a system
        # clock adjustment mid-measurement cannot corrupt the reading.
        start = time.perf_counter()

        # Both texts in one call: the library processes them together, which
        # is faster than two separate calls. One vector comes back per input.
        source_vec, response_vec = self.model.encode([case.source, case.response])

        # cos_sim returns a PyTorch tensor wrapping a single number; .item()
        # unwraps it into a plain float, which is what DetectorResult expects.
        score = util.cos_sim(source_vec, response_vec).item()

        latency_ms = (time.perf_counter() - start) * 1000

        return DetectorResult(
            verdict="grounded" if score >= self.threshold else "hallucinated",
            score=score,
            latency_ms=latency_ms,
            # explanation and tokens_used keep their defaults: this detector
            # explains nothing and calls no API.
        )





# check:
if __name__ == "__main__":
    from src.halueval import build_splits

    # The underscore means "second return value not needed here".
    dev, _ = build_splits()

    detector = EmbeddingDetector()

    # Four cases is enough to see the pattern without waiting.
    for case in dev[:4]:
        result = detector.check(case)
        print(f"{case.case_id}  expected={case.label:13}  score={result.score:.3f}")