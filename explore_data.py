# explore_data.py — inspect one record from each HaluEval file we use
import json
from pathlib import Path

# dialogue_data.json is deliberately excluded — see the design doc
FILES = ["qa_data.json", "summarization_data.json"]

for name in FILES:
    path = Path("data") / name
    # readline() reads a single line without loading the whole file —
    # summarization_data.json is 45 MB
    with open(path, encoding="utf-8") as f:
        record = json.loads(f.readline())

    print(f"===== {name} =====")
    for field, value in record.items():
        text = str(value)
        preview = text[:300] + ("..." if len(text) > 300 else "")
        print(f"[{field}] ({len(text)} chars)")
        print(preview)
        print()
