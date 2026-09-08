"""Run one detector across a list of test cases.

Wraps every detector identically, which is what keeps the comparison fair:
one loop, one set of rules, no per-detector special cases.

Three things happen around each call:

1. The cache is consulted first. A cached result is returned without running
   the detector — free for the local ones, and the difference between one day
   and two for the judge.
2. Token spend is tracked against a budget, so the run stops short of the
   daily cap instead of colliding with it partway through.
3. Failures are recorded and skipped rather than aborting. A run spanning two
   days must not be lost to one malformed reply, and since failures are not
   cached, the next run retries them.
"""

from dataclasses import asdict # converts dataclass instances to dictionaries

from src.base import Detector, DetectorResult
from src.cache import Cache, make_key
from src.halueval import TestCase

# Groq's free tier allows 200,000 tokens a day. Stopping at 180,000 leaves
# about ten percent of headroom, so an undercount cannot run into the wall
# in the middle of a case.
DAILY_TOKEN_BUDGET = 180_000

# run() is the whole of this module's interface. It is the only place that
# knows about the cache, the token budget and failure handling — the
# detectors know nothing of any of it, which is what keeps them uniform.
def run(
    detector: Detector,
    cases: list[TestCase],
    cache: Cache | None = None,
    token_budget: int = DAILY_TOKEN_BUDGET,
) -> dict[str, DetectorResult]:
    """Run `detector` over `cases`, using and filling the cache as it goes."""
    # Created here rather than required from the caller: one cache is the
    # normal case, and passing a different one stays possible for tests.
    cache = cache or Cache()

    # Keyed by case_id rather than a list: failed cases are skipped, so a
    # list would silently fall out of step with the case list and metrics
    # would compare one case's verdict against another's label.
    results: dict[str, DetectorResult] = {} # keyed by case_id so lookups stay correct when cases are skipped
    tokens_spent = 0 # number of tokens spent on calls to the detector, not counting cached results
    hits = 0 # number of cached results taken from the cache rather than produced by a call
    calls = 0 # number of calls to the detector, not counting cached results
    failures = 0 # number of cases that failed to produce a result, not counting cached results

    # enumerate starts at 1 because the index is used only for progress output.
    for index, case in enumerate(cases, start=1):
        key = make_key(detector.name, detector.version, case.case_id)

        # Consulted before the call, never after — a cache that is written but
        # not read saves nothing.
        cached = cache.get(key)
        if cached is not None:
            results[case.case_id] = DetectorResult(**cached) # cached is a dict, but DetectorResult expects keyword arguments, so unpack it
            hits += 1
            continue

        # Checked only on a miss: cached results cost no tokens, so they must
        # not count against the budget.
        if tokens_spent >= token_budget:
            print(f"stopping: token budget reached ({tokens_spent:,})")
            break

        # The detector is called inside a try block so that any failure does not 
        # end the run. A failure is counted and skipped, and the next run retries it.
        try:
            result = detector.check(case)
        except Exception as error:
            # Caught broadly on purpose. Any failure at all — a malformed
            # reply, a network blip — must not end a run that may already
            # hold two days of work. The error is printed so it stays visible.
            print(f"  {case.case_id}: failed — {error}")
            failures += 1
            continue

        # Cached before being counted, so an interruption right here still
        # leaves the result on disk.
        cache.set(key, asdict(result))
        results[case.case_id] = result
        tokens_spent += result.tokens_used
        calls += 1

        # Progress report every ten cases, so the user sees that the run is
        # moving along without being spammed by a line for every case.
        if index % 10 == 0:
            print(
                f"  {index}/{len(cases)}  "
                f"cached={hits} called={calls} tokens={tokens_spent:,}"
            )

    print(
        f"{detector.name}: {len(results)} results, {hits} from cache, "
        f"{calls} calls, {tokens_spent:,} tokens, {failures} failures"
    )

    return results


# check:
if __name__ == "__main__":
    from src.embeddings import EmbeddingDetector
    from src.halueval import build_splits

    dev, _ = build_splits()

    # Detector 1 is free, so this can be repeated at no cost. Running twice
    # is the point: the second pass should be entirely cache hits.
    detector = EmbeddingDetector()

    print("first pass:")
    run(detector, dev[:10])

    print("\nsecond pass:")
    run(detector, dev[:10])
