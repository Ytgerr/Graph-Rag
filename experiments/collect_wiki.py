#!/usr/bin/env python3
"""
Сбор датасета из Wikipedia по заданной теме.

Использование:
    python collect_wiki.py Moscow
    python collect_wiki.py "Machine learning" --lang en --limit 300
    python collect_wiki.py Physics --output app/dataset/data_from_wiki.txt
    python collect_wiki.py Moscow --expand   # LLM-powered sub-topic expansion
"""

import argparse
import json
import re
import sys
import time
import random
import logging
import os
from pathlib import Path
from typing import Optional, List

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DEFAULT_OUTPUT = "app/dataset/data_from_wiki.txt"
DEFAULT_LIMIT = 500
DEFAULT_LANG = "en"
USER_AGENT = "GraphRAG-DataCollector/1.0 (educational project)"
MIN_TEXT_LENGTH = 300
API_URL_TPL = "https://{lang}.wikipedia.org/w/api.php"


def _api(lang: str, session: requests.Session, **params) -> dict:
    params.setdefault("format", "json")
    resp = session.get(API_URL_TPL.format(lang=lang), params=params, timeout=15)
    resp.raise_for_status()
    return resp.json()


def _get_page_text(title: str, lang: str, session: requests.Session) -> str:
    """Get plain text of a Wikipedia article via the TextExtracts API."""
    data = _api(
        lang, session,
        action="query",
        titles=title,
        prop="extracts",
        explaintext="1",
        exsectionformat="plain",
        redirects="1",
    )
    pages = data.get("query", {}).get("pages", {})
    for page in pages.values():
        text = page.get("extract", "")
        if text:
            text = re.sub(r'==+\s*(References|See also|External links|Notes|Further reading|Bibliography)\s*==+.*', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'==+\s*(Примечания|Литература|Ссылки|См\. также)\s*==+.*', '', text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r'\s+', ' ', text).strip()
            return text
    return ""


def _get_see_also_links(title: str, lang: str, session: requests.Session) -> List[str]:
    """Extract 'See also' section links from a Wikipedia article."""
    data = _api(
        lang, session,
        action="parse",
        page=title,
        prop="sections",
        redirects="1",
    )
    sections = data.get("parse", {}).get("sections", [])
    see_also_idx = None
    for sec in sections:
        if sec.get("line", "").lower() in ("see also", "см. также"):
            see_also_idx = sec.get("index")
            break

    if not see_also_idx:
        return []

    data = _api(
        lang, session,
        action="parse",
        page=title,
        prop="links",
        section=see_also_idx,
        redirects="1",
    )
    links = data.get("parse", {}).get("links", [])
    return [lnk["*"] for lnk in links if lnk.get("ns", -1) == 0 and lnk.get("exists") is not None]


def search_titles(topic: str, lang: str, limit: int, session: requests.Session) -> List[str]:
    """Use MediaWiki search API to find articles related to the topic."""
    titles = []
    sr_offset = 0
    batch = 50

    while len(titles) < limit:
        data = _api(
            lang, session,
            action="query",
            list="search",
            srsearch=topic,
            srnamespace="0",
            srlimit=str(min(batch, limit - len(titles))),
            sroffset=str(sr_offset),
        )
        results = data.get("query", {}).get("search", [])
        if not results:
            break

        for item in results:
            t = item["title"]
            if t not in titles:
                titles.append(t)

        sr_offset += batch
        if "continue" not in data or "sroffset" not in data.get("continue", {}):
            break
        time.sleep(0.3)

    return titles[:limit]


def _generate_subtopics(topic: str, lang: str, api_key: str, model: str = "openai/gpt-4o-mini") -> List[str]:
    """Use LLM to generate diverse sub-topics for comprehensive Wikipedia coverage."""
    lang_name = {"en": "English", "ru": "Russian"}.get(lang, lang)

    prompt = (
        f"I need to collect a comprehensive Wikipedia dataset about: \"{topic}\"\n\n"
        f"Generate 15-20 diverse search queries (in {lang_name}) that would help find "
        f"Wikipedia articles covering ALL important aspects of this topic.\n\n"
        f"The queries should cover:\n"
        f"- The main topic itself\n"
        f"- History and origins\n"
        f"- Geography/location (if applicable)\n"
        f"- Key people, organizations, institutions\n"
        f"- Culture, traditions, notable features\n"
        f"- Economy, industry, infrastructure\n"
        f"- Science, technology, education\n"
        f"- Related sub-topics and neighboring concepts\n"
        f"- Current events and modern developments\n\n"
        f"Return ONLY a JSON array of strings, no explanation. Example:\n"
        f'[\"query 1\", \"query 2\", \"query 3\"]'
    )

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a research assistant that generates comprehensive search queries for Wikipedia data collection. Always respond with valid JSON arrays only."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 1000,
            },
            timeout=30,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()

        # Extract JSON array from response (handle markdown code blocks)
        json_match = re.search(r'\[.*\]', content, re.DOTALL)
        if json_match:
            subtopics = json.loads(json_match.group())
            if isinstance(subtopics, list) and all(isinstance(s, str) for s in subtopics):
                log.info("LLM generated %d sub-topics for '%s'", len(subtopics), topic)
                return subtopics

        log.warning("LLM returned unexpected format, falling back to main topic only")
        return [topic]

    except Exception as e:
        log.warning("LLM sub-topic generation failed: %s, falling back to main topic only", e)
        return [topic]


class CollectionProgress:
    def __init__(self):
        self.status = "idle"
        self.total = 0
        self.current = 0
        self.collected = 0
        self.topic = ""
        self.error = ""
        self.subtopics: List[str] = []

    def to_dict(self):
        return {
            "status": self.status,
            "topic": self.topic,
            "total": self.total,
            "current": self.current,
            "collected": self.collected,
            "error": self.error,
            "subtopics": self.subtopics,
        }


def collect(
    topic: str,
    lang: str = DEFAULT_LANG,
    limit: int = DEFAULT_LIMIT,
    output: str = DEFAULT_OUTPUT,
    progress: Optional[CollectionProgress] = None,
    expand: bool = False,
    api_key: Optional[str] = None,
) -> dict:
    if progress:
        progress.status = "discovering"
        progress.topic = topic

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    all_titles: List[str] = []

    if expand and api_key:
        if progress:
            progress.status = "expanding"

        subtopics = _generate_subtopics(topic, lang, api_key)

        if progress:
            progress.subtopics = subtopics
            progress.status = "discovering"

        per_subtopic = max(limit // len(subtopics), 10)

        for i, sub in enumerate(subtopics):
            log.info("Searching sub-topic [%d/%d]: %s (limit %d)", i + 1, len(subtopics), sub, per_subtopic)
            titles = search_titles(sub, lang, per_subtopic, session)
            for t in titles:
                if t not in all_titles:
                    all_titles.append(t)
            if len(all_titles) >= limit:
                break
            time.sleep(0.3)

        # Also grab "See also" links from the main article for extra coverage
        try:
            see_also = _get_see_also_links(topic, lang, session)
            for t in see_also:
                if t not in all_titles:
                    all_titles.append(t)
            if see_also:
                log.info("Added %d 'See also' links from main article", len(see_also))
        except Exception:
            pass

        all_titles = all_titles[:limit]
        log.info("Total unique titles after expansion: %d (from %d sub-topics)", len(all_titles), len(subtopics))
    else:
        all_titles = search_titles(topic, lang, limit, session)

    if not all_titles:
        err = f"No articles found for '{topic}' on {lang}.wikipedia.org"
        if progress:
            progress.status = "error"
            progress.error = err
        session.close()
        raise ValueError(err)

    log.info("Found %d titles for '%s'", len(all_titles), topic)

    if progress:
        progress.total = len(all_titles)
        progress.status = "collecting"

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    success = 0
    empty = 0
    failed = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for i, title in enumerate(all_titles, 1):
            if progress:
                progress.current = i

            try:
                text = _get_page_text(title, lang, session)
                if text and len(text) >= MIN_TEXT_LENGTH:
                    f.write(text)
                    f.write("\n\n")
                    success += 1
                    if progress:
                        progress.collected = success
                    if success % 10 == 0:
                        log.info("[%d/%d] Collected: %d articles", i, len(all_titles), success)
                else:
                    empty += 1
            except Exception as e:
                failed += 1
                log.debug("Error %s: %s", title, e)

            if i < len(all_titles):
                time.sleep(random.uniform(0.3, 0.8))

    session.close()

    size_kb = output_path.stat().st_size / 1024
    log.info("Done: %d articles, %d empty, %d errors", success, empty, failed)
    log.info("File: %s (%.1f KB)", output_path, size_kb)

    if progress:
        progress.status = "done"

    return {
        "success": success,
        "empty": empty,
        "failed": failed,
        "file": str(output_path),
        "size_kb": round(size_kb, 1),
    }


def main():
    parser = argparse.ArgumentParser(description="Collect Wikipedia dataset by topic")
    parser.add_argument("topic", help="Topic to search (e.g.: Moscow, AI, Physics)")
    parser.add_argument("--lang", default=DEFAULT_LANG, help=f"Wikipedia language (default: {DEFAULT_LANG})")
    parser.add_argument("--limit", type=int, default=DEFAULT_LIMIT, help=f"Max documents (default: {DEFAULT_LIMIT})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output file path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--expand", action="store_true", help="Use LLM to generate sub-topics for broader coverage")
    args = parser.parse_args()

    api_key = os.getenv("OPENAI_API_KEY") if args.expand else None
    if args.expand and not api_key:
        log.error("--expand requires OPENAI_API_KEY environment variable")
        sys.exit(1)

    log.info("Topic: %s | Lang: %s | Limit: %d | Expand: %s | Output: %s",
             args.topic, args.lang, args.limit, args.expand, args.output)
    collect(args.topic, args.lang, args.limit, args.output, expand=args.expand, api_key=api_key)


if __name__ == "__main__":
    main()
