"""Persistent store of detector results, shared by all three detectors.

Serves two purposes at once, which is why there is only one of it:

1. It preserves the experiment's data. Every verdict, score, latency and
   token count feeds the metrics and the failure analysis later.
2. It doubles as a cache. Keyed by detector, version and case, it lets a run
   skip work already done — trivial for the two local detectors, but the
   difference between one day and two for the judge, whose calls cost quota.

Stored as JSON Lines, one record appended per result. Appending is safer than
rewriting a whole file: a crash can damage at most the final line, and a
damaged final line is skipped on load.
"""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = PROJECT_ROOT / "cache" / "detector_results.jsonl"


def make_key(detector_name: str, detector_version: str, case_id: str) -> str:
    """Build the cache key.

    The version is part of the key on purpose: changing the judge's prompt
    must not reuse verdicts produced by the old one. Bumping the version
    invalidates them without deleting anything, and reverting the prompt
    makes the old results usable again for free.
    """
    return f"{detector_name}:{detector_version}:{case_id}"


class Cache:
    def __init__(self, path: Path = RESULTS_PATH):
        self.path = path

        # Everything ever written, held in memory for instant lookup.
        # A few hundred short records — nothing that strains memory.
        self.entries: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        """Read what previous runs wrote. Missing file simply means empty."""
        if not self.path.exists():
            return

        with open(self.path, encoding="utf-8") as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # A half-written final line from an interrupted run.
                    # Skipping it costs one API call; crashing would cost all.
                    continue
                self.entries[record["key"]] = record["value"]

    def get(self, key: str) -> dict | None:
        """Return the stored value, or None if this key was never cached."""
        return self.entries.get(key)

    def set(self, key: str, value: dict) -> None:
        """Store in memory and append to disk immediately.

        Written straight away rather than at the end of the run: the whole
        point is surviving an interruption, and anything held only in memory
        would be lost by exactly the crash we are guarding against.
        """
        self.entries[key] = value

        # Create the cache directory on first write.
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # "a" means append: write at the end, leave existing content alone.
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, "value": value}) + "\n")



# check:
if __name__ == "__main__":
    cache = Cache()

    # Deliberately not a real detector name or case id: a test must never
    # write something a real run could later mistake for its own result.
    key = make_key("__test__", "v0", "__example__")
    print("before:", cache.get(key))

    cache.set(key, {"verdict": "grounded", "explanation": "test"})
    print("after: ", cache.get(key))

    print("reloaded:", Cache().get(key))