# measure_tokens.py — measure real token counts in HaluEval records
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

SAMPLE_SIZE = 200

# What the judge will actually see: source, the answer being checked,
# and the question where the subset has one.
FIELDS = {
    "qa_data.json": ("knowledge", "hallucinated_answer", "question"),
    "summarization_data.json": ("document", "hallucinated_summary", None),
}

tokenizer = AutoTokenizer.from_pretrained("openai/gpt-oss-120b")

for name, (source_field, answer_field, context_field) in FIELDS.items():
    counts = []
    with open(Path("data") / name, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= SAMPLE_SIZE:
                break
            record = json.loads(line)
            parts = [record[source_field], record[answer_field]]
            if context_field:
                parts.append(record[context_field])
            counts.append(len(tokenizer.encode(" ".join(parts))))

    ordered = sorted(counts)
    print(f"===== {name} (n={len(counts)}) =====")
    print(f"  mean    {statistics.mean(counts):>6.0f}")
    print(f"  median  {statistics.median(counts):>6.0f}")
    print(f"  min     {min(counts):>6}")
    print(f"  p90     {ordered[int(len(ordered) * 0.9)]:>6}")
    print(f"  max     {max(counts):>6}")
    print()
