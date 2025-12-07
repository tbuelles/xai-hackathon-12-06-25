import os, requests, json
from dotenv import load_dotenv
load_dotenv()

api_key = os.environ["XAI_API_KEY"]

payload = {
    "model": "grok-4-latest",
    "messages": [
        {"role": "user", "content": "Write one sentence about attention mechanisms."}
    ],
    "temperature": 0,
    "logprobs": True,
    "top_logprobs": 5,   # optional; 0–20
    "stream": False,
}

r = requests.post(
    "https://api.x.ai/v1/chat/completions",
    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    json=payload,
)
r.raise_for_status()
out = r.json()

text = out["choices"][0]["message"]["content"]
logp = out["choices"][0].get("logprobs") or out["choices"][0]["message"].get("logprobs")

print(text)
print(json.dumps(logp, indent=2))
