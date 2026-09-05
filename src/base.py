"""The shared interface every detector implements.

All three detectors return the same shape, so the evaluation harness and the
UI can treat them identically, and cost and latency get collected as a matter
of course rather than bolted on at the end. This is also what keeps the
comparison fair: one harness runs all three in exactly the same way.
"""

from dataclasses import dataclass
from typing import Protocol

from src.halueval import TestCase


@dataclass
class DetectorResult:
    verdict: str           # "grounded" | "hallucinated"
    score: float           # 0..1, higher means better grounded
    latency_ms: float      # every detector must measure this — no default on purpose
    explanation: str = ""  # only detector 3 fills this in
    tokens_used: int = 0   # stays 0 for the two local detectors


class Detector(Protocol):
    """What every detector must provide."""

    name: str     # "embeddings" | "entailment" | "llm_judge"
    version: str  # bump on any change that invalidates cached results

    def check(self, case: TestCase) -> DetectorResult: ...


#check
if __name__ == "__main__":
    result = DetectorResult(verdict="grounded", score=0.87, latency_ms=12.5)
    print(result)