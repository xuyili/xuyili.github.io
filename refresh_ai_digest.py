#!/usr/bin/env python3
"""Refresh the bilingual AI RSS digest for GitHub Pages."""

from __future__ import annotations

import html
import json
import re
import socket
import time
import xml.etree.ElementTree as ET
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Iterable

import feedparser
from deep_translator import GoogleTranslator, MyMemoryTranslator


ROOT = Path(__file__).resolve().parent
FEEDS_OPML = ROOT / "feeds.opml"
OUTPUTS = [ROOT / "ai-rss-digest.json", ROOT / "docs" / "ai-rss-digest.json"]
TIMEZONE = timezone(timedelta(hours=8))
MAX_ITEMS = 50
FETCH_LIMIT = 40
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
)

BOOST_KEYWORDS = {
    "ai": 5.0,
    "llm": 5.0,
    "gpt": 5.0,
    "claude": 4.5,
    "gemini": 4.0,
    "openai": 4.5,
    "anthropic": 4.0,
    "model": 2.0,
    "models": 2.0,
    "training": 3.2,
    "inference": 3.2,
    "agent": 3.0,
    "agents": 3.0,
    "rag": 3.0,
    "prompt": 2.6,
    "prompts": 2.6,
    "reasoning": 2.8,
    "eval": 2.4,
    "benchmark": 2.4,
    "gpu": 2.0,
    "transformer": 3.0,
    "attention": 2.6,
    "neural": 2.4,
    "machine learning": 3.2,
    "deep learning": 3.0,
    "ml": 1.5,
    "automation": 2.0,
    "coding agent": 3.2,
    "codegen": 2.8,
    "copilot": 2.8,
    "developer tools": 2.2,
    "search": 1.3,
    "security": 1.3,
    "privacy": 1.1,
}

REQUIRED_AI_TERMS = [
    "ai",
    "llm",
    "gpt",
    "claude",
    "gemini",
    "openai",
    "anthropic",
    "model",
    "models",
    "training",
    "inference",
    "agent",
    "agents",
    "machine learning",
    "deep learning",
    "neural",
    "transformer",
    "copilot",
]

SOURCE_BOOSTS = {
    "simonwillison.net": 3.0,
    "gilesthomas.com": 3.2,
    "seangoedecke.com": 2.8,
    "minimaxir.com": 3.2,
    "gwern.net": 3.0,
    "garymarcus.substack.com": 2.4,
    "dwarkesh.com": 2.0,
    "grantslatton.com": 2.0,
    "martinalderson.com": 2.2,
    "joanwestenberg.com": 1.8,
    "xeiaso.net": 1.4,
}

PRIORITY_SOURCES = set(SOURCE_BOOSTS) | {
    "mitchellh.com",
    "joanwestenberg.com",
    "martinalderson.com",
    "wheresyoured.at",
    "grantslatton.com",
    "lucumr.pocoo.org",
    "xeiaso.net",
    "skyfall.dev",
    "geoffreylitt.com",
    "minimaxir.com",
    "dwarkesh.com",
}

NO_TRANSLATE = [
    "AI",
    "AGI",
    "API",
    "Claude",
    "Gemini",
    "GitHub",
    "Google",
    "GPT",
    "GPU",
    "JSON",
    "LLM",
    "LLMs",
    "ML",
    "NVIDIA",
    "OpenAI",
    "Python",
    "RAG",
    "RSS",
    "Rust",
    "SQL",
    "Transformer",
]

TOPIC_SENTENCES = {
    "training": "The piece is most useful for readers who care about how training choices affect stability, cost, and final model quality in real workflows.",
    "inference": "Its main value is in showing how inference latency, throughput, and product experience change when implementation details are adjusted.",
    "agent": "What makes it notable is the way it connects agent design decisions to practical reliability, tool use, and developer ergonomics.",
    "security": "The broader relevance comes from showing how security and model deployment concerns increasingly overlap in modern AI systems.",
    "policy": "Beyond the headline, the article helps frame how policy and market structure can shape the pace and direction of AI deployment.",
    "default": "The broader takeaway is that it turns a narrow update into something readers can reuse when thinking about tools, systems, and tradeoffs in AI work.",
}


def load_feeds() -> list[tuple[str, str]]:
    tree = ET.parse(FEEDS_OPML)
    feeds = []
    for outline in tree.findall(".//outline[@xmlUrl]"):
        name = outline.attrib.get("text") or outline.attrib.get("title") or outline.attrib["xmlUrl"]
        feeds.append((name, outline.attrib["xmlUrl"]))
    feeds.sort(key=lambda item: (item[0] not in PRIORITY_SOURCES, item[0]))
    return feeds[:FETCH_LIMIT]


def clean_html(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<(script|style)[^>]*>.*?</\\1>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]


def trim_words(text: str, limit: int) -> str:
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]).rstrip(" ,;:") + "..."


def parse_dt(entry: dict) -> datetime | None:
    for field in ("published", "updated", "pubDate"):
        value = entry.get(field)
        if not value:
            continue
        try:
            dt = parsedate_to_datetime(value)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=TIMEZONE)
            return dt.astimezone(TIMEZONE)
        except Exception:
            continue
    parsed = entry.get("published_parsed") or entry.get("updated_parsed")
    if parsed:
        try:
            return datetime(*parsed[:6], tzinfo=TIMEZONE)
        except Exception:
            return None
    return None


def keyword_hits(text: str) -> Counter:
    lowered = text.lower()
    hits: Counter[str] = Counter()
    for kw in BOOST_KEYWORDS:
        if kw in lowered:
            hits[kw] += lowered.count(kw)
    return hits


def classify_topic(text: str) -> str:
    lowered = text.lower()
    if any(k in lowered for k in ("training", "fine-tuning", "gradient", "weight decay", "optimizer")):
        return "training"
    if any(k in lowered for k in ("inference", "latency", "throughput", "serving", "quantization")):
        return "inference"
    if any(k in lowered for k in ("agent", "tool use", "workflow", "automation", "function calling")):
        return "agent"
    if any(k in lowered for k in ("security", "attack", "vulnerability", "privacy")):
        return "security"
    if any(k in lowered for k in ("policy", "regulation", "governance", "market")):
        return "policy"
    return "default"


def translate_text(translator, text: str) -> str:
    if not text.strip():
        return ""
    working = text
    replacements: dict[str, str] = {}
    for i, term in enumerate(NO_TRANSLATE):
        token = f"__NT{i}__"
        if term in working:
            working = working.replace(term, token)
            replacements[token] = term
    translated = translator.translate(working[:2400])
    for token, term in replacements.items():
        translated = translated.replace(token, term)
    translated = translated.replace("人工智能", "AI")
    translated = re.sub(r"\s+", " ", translated).strip()
    return translated


def score_entry(source: str, entry: dict) -> float:
    title = clean_html(entry.get("title", ""))
    summary = clean_html(entry.get("summary", "") or entry.get("description", ""))
    text = f"{title} {summary}".lower()
    score = SOURCE_BOOSTS.get(source, 0.0)
    hits = keyword_hits(text)
    for kw, count in hits.items():
        score += BOOST_KEYWORDS[kw] * min(count, 4)
    dt = parse_dt(entry)
    if dt:
        age_days = max(0.0, (datetime.now(TIMEZONE) - dt).total_seconds() / 86400)
        if age_days <= 2:
            score += 5.0
        elif age_days <= 7:
            score += 3.0
        elif age_days <= 14:
            score += 1.5
        else:
            score -= min(age_days / 10, 5)
    if len(summary) > 320:
        score += 1.5
    if len(title.split()) >= 5:
        score += 0.5
    return score


def is_ai_relevant(entry: dict) -> bool:
    text = f"{entry['title']} {entry['summary']}".lower()
    return any(term in text for term in REQUIRED_AI_TERMS)


def build_summary(title: str, excerpt: str, topic: str) -> str:
    sentences = split_sentences(excerpt)
    chosen = []
    for sentence in sentences:
        if len(sentence.split()) < 7:
            continue
        chosen.append(sentence)
        if len(" ".join(chosen).split()) >= 70:
            break
    if not chosen:
        chosen = [excerpt]

    lead = trim_words(" ".join(chosen), 80)
    support = TOPIC_SENTENCES[topic]
    impact = (
        f"For the digest, {title} stands out because it offers enough concrete detail to be useful, "
        "while also surfacing the tradeoffs that matter when people evaluate tools, models, and implementation decisions."
    )
    summary = f"{lead} {support} {impact}"
    return trim_words(summary, 165)


def fetch_entries(feeds: Iterable[tuple[str, str]]) -> list[dict]:
    all_entries: list[dict] = []
    feeds = list(feeds)

    def fetch_one(name: str, url: str) -> list[dict]:
        batch: list[dict] = []
        feed = feedparser.parse(url, agent=USER_AGENT)
        for entry in feed.entries[:12]:
            title = clean_html(entry.get("title", ""))
            summary = clean_html(
                entry.get("summary", "")
                or entry.get("description", "")
                or " ".join(
                    clean_html(part.get("value", ""))
                    for part in entry.get("content", [])
                    if isinstance(part, dict)
                )
            )
            if not title or not summary:
                continue
            batch.append(
                {
                    "source": name,
                    "title": title,
                    "summary": summary,
                    "url": entry.get("link", ""),
                    "date": parse_dt(entry),
                    "score": score_entry(name, entry),
                }
            )
        return batch

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_one, name, url): (name, url) for name, url in feeds}
        for idx, future in enumerate(as_completed(futures), start=1):
            name, _ = futures[future]
            try:
                all_entries.extend(future.result())
                print(f"Fetched {idx}/{len(feeds)}: {name}", flush=True)
            except Exception as exc:
                print(f"Skipped {name}: {exc}", flush=True)
    return all_entries


def dedupe(entries: list[dict]) -> list[dict]:
    seen: set[str] = set()
    result = []
    for entry in sorted(entries, key=lambda e: e["score"], reverse=True):
        key = entry["url"] or entry["title"].lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(entry)
    return result


def main() -> None:
    socket.setdefaulttimeout(18)
    feeds = load_feeds()
    raw_entries = fetch_entries(feeds)
    filtered = [e for e in raw_entries if e["score"] >= 6.0 and is_ai_relevant(e)]
    ranked = dedupe(filtered or raw_entries)[:MAX_ITEMS]

    translators = [
        MyMemoryTranslator(source="en-GB", target="zh-CN"),
        GoogleTranslator(source="en", target="zh-CN"),
    ]
    now = datetime.now(TIMEZONE)
    items = []

    for idx, entry in enumerate(ranked, start=1):
        topic = classify_topic(f"{entry['title']} {entry['summary']}")
        summary_en = build_summary(entry["title"], entry["summary"], topic)
        title_zh = ""
        summary_zh = ""
        last_error = None
        for translator in translators:
            try:
                title_zh = translate_text(translator, entry["title"])
                time.sleep(0.12)
                summary_zh = translate_text(translator, summary_en)
                break
            except Exception as exc:
                last_error = exc
                title_zh = ""
                summary_zh = ""
        if not title_zh or not summary_zh:
            raise RuntimeError(f"translation failed for {entry['title']}: {last_error}")
        time.sleep(0.12)

        items.append(
            {
                "source": entry["source"],
                "date": (entry["date"] or now).strftime("%Y-%m-%d"),
                "url": entry["url"],
                "title_en": entry["title"],
                "title_zh": title_zh,
                "summary_en": summary_en,
                "summary_zh": summary_zh,
            }
        )
        print(f"Translated {idx}/{len(ranked)}: {entry['title'][:60]}", flush=True)

    payload = {
        "generated_at": now.isoformat(timespec="seconds"),
        "items": items,
    }

    for output in OUTPUTS:
        output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
