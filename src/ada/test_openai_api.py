"""
This is a minimal example of testing whether the openapi compatible API is working.
change the API_KEY and HOST_URL respectively.
"""

import requests

HOST_URL = "http://localhost:8000/v1"
API_KEY = "token-test"
header = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}",
}

for i in range(100):
    payload = {
        "model": "NousResearch/Meta-Llama-3-8B-Instruct",
        "messages": [{"role": "user", "content": "Say this is a test!"}],
        "max_tokens": 100,
        "temperature": 0.0,
    }
    response = requests.post(HOST_URL + "/chat/completions", headers=header, json=payload)
    result = response.json()
    completions = result["choices"][0]["message"]["content"]
    usage = result["usage"]
    print(usage)