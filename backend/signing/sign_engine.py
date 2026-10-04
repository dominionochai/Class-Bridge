"""Deterministic vocabulary-backed translation for classroom signing."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

_VOCAB_PATH = Path(__file__).with_name("vocab.json")
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.IGNORECASE)


def get_vocabulary() -> list[dict[str, Any]]:
    """Return the checked-in vocabulary in its JSON order."""
    with _VOCAB_PATH.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise ValueError("vocab.json must contain a list")
    return value


def _keyword_tokens(value: str) -> tuple[str, ...]:
    return tuple(token.lower() for token in _TOKEN_RE.findall(value))


_VOCAB = get_vocabulary()
_KEYWORDS: list[tuple[tuple[str, ...], str]] = []
for _entry in _VOCAB:
    for _keyword in _entry.get("keywords", []):
        _tokens = _keyword_tokens(str(_keyword))
        if _tokens:
            _KEYWORDS.append((_tokens, str(_entry["id"])))
_KEYWORDS.sort(key=lambda item: len(item[0]), reverse=True)


def translate_segment(text: str) -> dict[str, Any]:
    from llm_bridge import translate_with_llm
    llm = translate_with_llm(text)
    if llm is not None:
        return {
            "sign_ids": llm.get("sign_ids", []),
            "coverage_pct": 100.0 if llm.get("sign_ids") else 0.0,
            "fallback_reason": None,
            "captions_only": not llm.get("sign_ids"),
            "translation": llm.get("translation", text),
        }
    """Translate known words/phrases and explicitly report vocabulary gaps."""
    tokens = [token.lower() for token in _TOKEN_RE.findall(text or "")]
    if not tokens:
        return {
            "sign_ids": [],
            "coverage_pct": 0.0,
            "fallback_reason": "unknown_word",
            "captions_only": True,
        }

    covered = [False] * len(tokens)
    hits: list[tuple[int, str]] = []  # (start index, sign id)
    # Longest phrases claim tokens first, but we record WHERE they matched
    # so the final list can be returned in spoken order.
    for keyword_tokens, sign_id in _KEYWORDS:
        width = len(keyword_tokens)
        for start in range(0, len(tokens) - width + 1):
            end = start + width
            if covered[start:end] == [False] * width and tuple(tokens[start:end]) == keyword_tokens:
                covered[start:end] = [True] * width
                hits.append((start, sign_id))
    hits.sort(key=lambda item: item[0])
    sign_ids = [sign_id for _, sign_id in hits]

    covered_count = sum(covered)
    has_gap = covered_count != len(tokens)
    return {
        "sign_ids": sign_ids,
        "coverage_pct": round((covered_count / len(tokens)) * 100, 2),
        "fallback_reason": "vocab_gap" if has_gap else None,
        "captions_only": has_gap or not sign_ids,
    }
