import os, requests

API_KEY = os.environ["XAI_API_KEY"]
url = "https://api.x.ai/v1/chat/completions"

payload = {
    "model": "grok-4-latest",
    "messages": [
        {"role": "system", "content": "You are a test assistant."},
        {"role": "user", "content": "Testing. Just say hi and hello world and nothing else."}
    ],
    "temperature": 0,
    "stream": False,
}

resp = requests.post(
    url,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    },
    json=payload,
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
