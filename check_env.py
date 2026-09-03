# check_env.py — verify the stack resolves and HHEM actually runs
import torch
import transformers
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer, util

print(f"torch        {torch.__version__}")
print(f"transformers {transformers.__version__}")

# First pair is a contradiction: premise says Berlin, hypothesis says Paris.
# Almost identical wording, opposite meaning.
pairs = [
    ("The capital of France is Berlin.", "The capital of France is Paris."),
    ("I am in California", "I am in United States."),
]

hhem = AutoModelForSequenceClassification.from_pretrained(
    "vectara/hallucination_evaluation_model",
    trust_remote_code=True,
)
print("HHEM scores:", hhem.predict(pairs))

embedder = SentenceTransformer("all-MiniLM-L6-v2")
a, b = embedder.encode(pairs[0])
print("cosine:", util.cos_sim(a, b).item())
