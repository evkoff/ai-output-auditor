# check_groq.py — verify the API key works and the model answers
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["GROQ_API_KEY"]
print(f"key loaded, length {len(api_key)}")



from groq import Groq

client = Groq(api_key=api_key)
response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[{"role": "user", "content": "Reply with exactly one word: ok"}],
)
print("answer:", response.choices[0].message.content)

print("usage:", response.usage)