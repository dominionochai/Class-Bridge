import json
import os

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"


def translate_with_llm(segment: str) -> dict | None:
    key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not key:
        return None
    prompt = ("Map this classroom English to simple signable phrases. Return JSON: "
              "{\"translation\": \"simple english\", \"sign_ids\": [\"...\"]}. Input: " + segment)
    try:
        r = httpx.post(OPENROUTER_URL, headers={"Authorization": f"Bearer {key}"},
                       json={"model": "google/gemini-2.0-flash-001",
                             "messages": [{"role": "user", "content": prompt}]}, timeout=20)
        r.raise_for_status()
        return json.loads(r.json()["choices"][0]["message"]["content"])
    except Exception:
        return None


def smart_captions_with_gemini(text: str) -> str | None:
    key = os.getenv("GOOGLE_AI_STUDIO_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        return None
    prompt = ("Clean this raw lecture transcript into smart captions: fix punctuation, "
              "keep it verbatim, split by speaker if obvious. Output plain text captions.\n\n" + text)
    try:
        r = httpx.post(GEMINI_URL, params={"key": key},
                       json={"contents": [{"parts": [{"text": prompt}]}]}, timeout=20)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        return None
