#!/usr/bin/env python3
"""Scrape missing eBL dictionary entries via API and merge into eBL_Dictionary.csv."""

import csv
import json
import os
import re
import sys
import time
from urllib.parse import quote

import pandas as pd
import requests

DATA_DIR = "/data/lsb/deep_past/data"
EBL_CSV = os.path.join(DATA_DIR, "eBL_Dictionary.csv")
LEXICON_CSV = os.path.join(DATA_DIR, "OA_Lexicon_eBL.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "eBL_Dictionary_extended.csv")
API_BASE = "https://www.ebl.lmu.de/api/words"
DELAY = 0.3  # seconds between requests (polite crawling)


def get_existing_ebl_words(ebl_path: str) -> set:
    """Get set of lowercase word stems already in eBL_Dictionary.csv."""
    df = pd.read_csv(ebl_path)
    words = set()
    for w in df["word"].dropna():
        clean = re.sub(r"\s+[IVX]+$", "", str(w).strip()).strip().lower()
        if clean:
            words.add(clean)
    return words


def get_missing_ebl_words(lex_path: str, existing: set) -> list:
    """Find eBL words referenced in OA Lexicon but not in eBL_Dictionary.csv."""
    lex = pd.read_csv(lex_path)
    missing = set()
    for url in lex["eBL"].dropna():
        url = str(url)
        if "word=" not in url:
            continue
        word = url.split("word=")[-1].strip()
        if word.lower() not in existing:
            missing.add(word)
    return sorted(missing)


def fetch_word(word: str, session: requests.Session) -> list:
    """Fetch word entries from eBL API. Returns list of (word_id, guideWord, meaning)."""
    url = f"{API_BASE}?word={quote(word)}"
    try:
        resp = session.get(url, timeout=15)
        if resp.status_code == 404:
            return []
        resp.raise_for_status()
        data = resp.json()
        if not isinstance(data, list):
            return []
        results = []
        for entry in data:
            word_id = entry.get("_id", "")
            guide = entry.get("guideWord", "")
            meaning = entry.get("meaning", "")
            # Extract quoted translations from meaning
            if meaning:
                translations = re.findall(r'"([^"]+)"', meaning)
                definition = meaning
            else:
                translations = []
                definition = ""
            # Use guideWord as primary, fall back to first quoted translation
            primary_eng = guide if guide else (translations[0] if translations else "")
            if word_id and (primary_eng or definition):
                results.append({
                    "word": word_id,
                    "definition": definition if definition else f'"{primary_eng}"',
                    "derived_from": "",
                    "guideWord": primary_eng,
                })
        return results
    except Exception as e:
        print(f"   ERROR fetching {word}: {e}")
        return []


def main():
    print("=" * 60)
    print("eBL Dictionary Scraper")
    print("=" * 60)

    existing = get_existing_ebl_words(EBL_CSV)
    print(f"Existing eBL entries: {len(existing)}")

    missing = get_missing_ebl_words(LEXICON_CSV, existing)
    print(f"Missing words to fetch: {len(missing)}")

    if not missing:
        print("Nothing to fetch!")
        return

    # Load existing CSV
    existing_df = pd.read_csv(EBL_CSV)
    new_rows = []
    fetched = 0
    errors = 0

    session = requests.Session()
    session.headers.update({"User-Agent": "eBL-Academic-Research/1.0 (Akkadian MT project)"})

    print(f"\nFetching {len(missing)} words from eBL API...")
    for i, word in enumerate(missing):
        if (i + 1) % 100 == 0 or i == 0:
            print(f"   [{i+1}/{len(missing)}] {word}...")
            sys.stdout.flush()

        entries = fetch_word(word, session)
        if entries:
            new_rows.extend(entries)
            fetched += 1
        else:
            errors += 1

        time.sleep(DELAY)

    print(f"\nResults:")
    print(f"   Words fetched: {fetched}")
    print(f"   New entries: {len(new_rows)}")
    print(f"   Errors/404s: {errors}")

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        # Ensure columns match existing CSV
        for col in existing_df.columns:
            if col not in new_df.columns:
                new_df[col] = ""
        new_df = new_df[existing_df.columns]

        merged = pd.concat([existing_df, new_df], ignore_index=True)
        merged.to_csv(OUTPUT_CSV, index=False)
        print(f"\n   Saved extended dictionary: {OUTPUT_CSV}")
        print(f"   Original: {len(existing_df)} rows")
        print(f"   Extended: {len(merged)} rows (+{len(new_rows)})")

        # Also show sample new entries
        print(f"\n   Sample new entries:")
        for row in new_rows[:15]:
            w = row["word"]
            g = row.get("guideWord", "")
            print(f"      {w:30s} -> {g}")
    else:
        print("\n   No new entries found.")


if __name__ == "__main__":
    main()
