#!/usr/bin/env python3
"""Simple Groq API connectivity test script.

Reads `GROQ_API_KEY` from environment (load .env if present) and makes a
single test request to the specified model. Prints status and a short
response preview. This is a best-effort test; endpoint formats may vary.
"""
import os
import json
import sys
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GROQ_API_KEY")
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

if not API_KEY:
    print("GROQ_API_KEY not found in environment; set it in .env or env vars.")
    sys.exit(2)

import requests

# Best-effort endpoint guess; Groq's API may differ. This tries the common pattern.
endpoint = f"https://api.groq.ai/v1/models/{MODEL}/outputs"

headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

payload = {
    "input": "Provide a concise one-sentence test: 'This is a connectivity test.'",
}

try:
    resp = requests.post(endpoint, headers=headers, json=payload, timeout=30)
except Exception as e:
    print("Request failed:", str(e))
    sys.exit(3)

print("STATUS", resp.status_code)
ct = resp.headers.get("Content-Type", "")
if "application/json" in ct:
    try:
        data = resp.json()
        # print a truncated preview
        s = json.dumps(data)
        print("RESPONSE_PREVIEW", s[:1000])
    except Exception as e:
        print("Failed to decode JSON response:", e)
        print(resp.text[:1000])
else:
    print(resp.text[:1000])
