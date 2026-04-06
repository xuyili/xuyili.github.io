#!/usr/bin/env python3
"""Validate bilingual AI digest JSON before publishing."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path


CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
LATIN_RE = re.compile(r"[A-Za-z]")

# Terms that are acceptable to keep in English inside Chinese copy.
ALLOWED_MIXED_TERMS = {
    "AI",
    "AGI",
    "API",
    "AutoModel",
    "Claude",
    "Copilot",
    "Gemini",
    "GitHub",
    "GPT",
    "Google",
    "Hugging Face",
    "JSON",
    "LLM",
    "MiniJinja",
    "NixOS",
    "OpenAI",
    "OpenClaw",
    "PyTorch",
    "Rust",
    "SQL",
    "Transformer",
    "Trainer",
    "pipeline",
}


def count_matches(regex: re.Pattern[str], text: str) -> int:
    return len(regex.findall(text))


def english_ratio(text: str) -> float:
    letters = count_matches(LATIN_RE, text)
    total = letters + count_matches(CJK_RE, text)
    return letters / total if total else 0.0


def chinese_ratio(text: str) -> float:
    chars = count_matches(CJK_RE, text)
    total = chars + count_matches(LATIN_RE, text)
    return chars / total if total else 0.0


def normalize_mixed_terms(text: str) -> str:
    normalized = text
    for term in sorted(ALLOWED_MIXED_TERMS, key=len, reverse=True):
        normalized = normalized.replace(term, "")
    return normalized


def is_probably_bad_zh(text: str) -> list[str]:
    issues: list[str] = []
    if not text.strip():
        return ["empty"]

    normalized = normalize_mixed_terms(text)
    zh_ratio = chinese_ratio(normalized)
    en_ratio = english_ratio(normalized)

    if count_matches(CJK_RE, normalized) < 4:
        issues.append("too_little_chinese")
    if zh_ratio < 0.35 and en_ratio > 0.45:
        issues.append("mostly_english")
    if text.strip() == normalized.strip() and count_matches(LATIN_RE, text) > 18 and count_matches(CJK_RE, text) == 0:
        issues.append("untranslated")
    return issues


def is_probably_bad_en(text: str) -> list[str]:
    issues: list[str] = []
    if not text.strip():
        return ["empty"]

    en_ratio = english_ratio(text)
    zh_ratio = chinese_ratio(text)
    if count_matches(LATIN_RE, text) < 8:
        issues.append("too_little_english")
    if en_ratio < 0.45 and zh_ratio > 0.2:
        issues.append("contains_too_much_chinese")
    return issues


def validate_item(item: dict, index: int) -> list[str]:
    issues: list[str] = []
    required = ["source", "date", "url", "title_en", "title_zh", "summary_en", "summary_zh"]
    for field in required:
        if not str(item.get(field, "")).strip():
            issues.append(f"missing:{field}")

    if item.get("title_en", "").strip() == item.get("title_zh", "").strip():
        issues.append("title_identical")
    if item.get("summary_en", "").strip() == item.get("summary_zh", "").strip():
        issues.append("summary_identical")

    for field in ("title_zh", "summary_zh"):
        for reason in is_probably_bad_zh(str(item.get(field, ""))):
            issues.append(f"{field}:{reason}")

    for field in ("title_en", "summary_en"):
        for reason in is_probably_bad_en(str(item.get(field, ""))):
            issues.append(f"{field}:{reason}")

    summary_zh = str(item.get("summary_zh", "")).strip()
    summary_zh_len = count_matches(CJK_RE, summary_zh)
    if summary_zh_len < 120:
        issues.append("summary_zh:too_short")
    if summary_zh_len > 320:
        issues.append("summary_zh:too_long")

    if not str(item.get("url", "")).startswith(("http://", "https://")):
        issues.append("invalid:url")

    return [f"item_{index}:{issue}" for issue in issues]


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs/ai-rss-digest.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    items = data.get("items", [])
    errors: list[str] = []

    if not isinstance(items, list):
        print("top_level:items_not_list")
        return 1

    for index, item in enumerate(items, start=1):
        errors.extend(validate_item(item, index))

    if errors:
        print("\n".join(errors))
        return 1

    print(f"OK: {len(items)} items validated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
