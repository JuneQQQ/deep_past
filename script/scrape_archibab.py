#!/usr/bin/env python3
"""
Scrape archibab.fr API for Akkadian-French parallel corpus.

API: https://www.archibab.fr/api/v1/texte_p/?min_id=1&page=N&size=250
Auth: X-API-KEY (public, no login required)
Output: data/archibab/archibab_raw.jsonl  (all texts)
        data/archibab/archibab_parallel.csv (transliteration + French translation pairs)

Usage:
    python script/scrape_archibab.py [--resume] [--size 250] [--delay 2.0]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

# ── Constants ─────────────────────────────────────────────────────────────────

API_BASE = "https://www.archibab.fr/api/v1"
API_KEY_RAW = "wbcJ5TLV.EkrKl5vQbq6G8dNgfKK5tbGITj5Lu75g"
# Frontend scrambles the key: r[4:8] + r[0:4] + r[8:]
API_KEY = API_KEY_RAW[4:8] + API_KEY_RAW[0:4] + API_KEY_RAW[8:]

HEADERS = {
    "Authorization": "Bearer undefined",
    "X-API-KEY": API_KEY,
    "Accept": "application/json",
}

PROXY = "http://127.0.0.1:7890"

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "archibab"
RAW_JSONL = OUTPUT_DIR / "archibab_raw.jsonl"
PARALLEL_CSV = OUTPUT_DIR / "archibab_parallel.csv"
PROGRESS_FILE = OUTPUT_DIR / ".scrape_progress.json"

# ── HTML cleanup ──────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_LINE_NUM_RE = re.compile(r"^\d+['′']?\s*")


def clean_html(text: str) -> str:
    """Strip HTML tags and normalize whitespace."""
    if not text:
        return ""
    text = _HTML_TAG_RE.sub("", text)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&nbsp;", " ").replace("\xa0", " ")
    text = _MULTI_SPACE_RE.sub(" ", text)
    return text.strip()


def extract_transliteration(text_obj: dict) -> str:
    """
    Extract transliteration from a text object.
    Prefers transcriptions_lignes (line-by-line), falls back to transcr_extrait.
    """
    lines = text_obj.get("transcriptions_lignes", [])
    if lines:
        # Sort by position: column, then line number
        sorted_lines = sorted(lines, key=lambda l: (
            l.get("enveloppe", False),
            l.get("colonne_prime", ""),
            l.get("ligne", 0),
        ))
        parts = []
        for l in sorted_lines:
            raw = l.get("transcription", "")
            cleaned = clean_html(raw).strip()
            if cleaned:
                line_num = l.get("ligne_et_prime", str(l.get("ligne", "")))
                parts.append(f"{line_num} {cleaned}")
        if parts:
            return "\n".join(parts)

    # Fallback: transcr_extrait
    extrait = text_obj.get("transcr_extrait", "")
    if extrait:
        return clean_html(extrait)

    return ""


def extract_translation(text_obj: dict) -> str:
    """Extract French translation."""
    trad = text_obj.get("traduction", "")
    return clean_html(trad)


# ── Scraping ──────────────────────────────────────────────────────────────────

def load_progress() -> dict:
    """Load scraping progress from file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, "r") as f:
            return json.load(f)
    return {"last_page": 0, "total_count": 0, "scraped_ids": []}


def save_progress(progress: dict):
    """Save scraping progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def fetch_page(client: httpx.Client, page: int, size: int) -> dict:
    """Fetch a single page of text listings."""
    url = f"{API_BASE}/texte_p/"
    params = {
        "min_id": 1,
        "o": "ref,id",
        "page": page,
        "size": size,
    }
    resp = client.get(url, params=params, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()


def scrape_all(resume: bool = True, page_size: int = 250, delay: float = 2.0):
    """Scrape all texts from archibab.fr API."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress = load_progress() if resume else {"last_page": 0, "total_count": 0, "scraped_ids": set()}
    if isinstance(progress.get("scraped_ids"), list):
        progress["scraped_ids"] = set(progress["scraped_ids"])

    start_page = progress["last_page"] + 1 if resume and progress["last_page"] > 0 else 1

    # Open JSONL in append mode if resuming, write mode otherwise
    jsonl_mode = "a" if resume and progress["last_page"] > 0 else "w"

    transport = httpx.HTTPTransport(proxy=PROXY, verify=False)
    client = httpx.Client(transport=transport, follow_redirects=True)

    try:
        # First request to get total count
        print(f"📡 Fetching page 1 to get total count...")
        first_data = fetch_page(client, 1, page_size)
        total_count = first_data["count"]
        total_pages = (total_count + page_size - 1) // page_size
        print(f"📊 Total texts: {total_count}, pages: {total_pages} (size={page_size})")

        progress["total_count"] = total_count

        with open(RAW_JSONL, jsonl_mode, encoding="utf-8") as f_jsonl:
            for page in range(start_page, total_pages + 1):
                t0 = time.time()

                if page == 1 and start_page == 1:
                    data = first_data
                else:
                    try:
                        data = fetch_page(client, page, page_size)
                    except Exception as e:
                        print(f"   ❌ Page {page} failed: {e}")
                        # Retry once after delay
                        time.sleep(delay * 3)
                        try:
                            data = fetch_page(client, page, page_size)
                        except Exception as e2:
                            print(f"   ❌ Page {page} retry failed: {e2}, skipping")
                            continue

                results = data.get("results", [])
                new_count = 0
                for text_obj in results:
                    tid = text_obj.get("id")
                    if tid and tid not in progress["scraped_ids"]:
                        f_jsonl.write(json.dumps(text_obj, ensure_ascii=False) + "\n")
                        progress["scraped_ids"].add(tid)
                        new_count += 1

                elapsed = time.time() - t0
                with_trad = sum(1 for r in results if r.get("traduction"))
                with_trans = sum(1 for r in results if r.get("transcriptions_lignes") or r.get("transcr_extrait"))
                with_both = sum(1 for r in results if r.get("traduction") and (r.get("transcriptions_lignes") or r.get("transcr_extrait")))

                print(
                    f"   📄 Page {page}/{total_pages} | "
                    f"{len(results)} texts ({new_count} new) | "
                    f"trad={with_trad} trans={with_trans} both={with_both} | "
                    f"{elapsed:.1f}s"
                )

                # Save progress
                progress["last_page"] = page
                # Convert set to list for JSON serialization
                progress_save = {
                    "last_page": page,
                    "total_count": total_count,
                    "scraped_ids": list(progress["scraped_ids"]),
                }
                save_progress(progress_save)

                # Rate limiting
                if page < total_pages:
                    time.sleep(delay)

        print(f"\n✅ Scraping complete: {len(progress['scraped_ids'])} texts saved to {RAW_JSONL}")

    finally:
        client.close()


# ── Post-processing ───────────────────────────────────────────────────────────

def build_parallel_csv():
    """Build parallel corpus CSV from raw JSONL."""
    if not RAW_JSONL.exists():
        print(f"❌ {RAW_JSONL} not found. Run scraping first.")
        return

    print(f"📝 Building parallel CSV from {RAW_JSONL}...")
    total = 0
    with_pair = 0

    with open(RAW_JSONL, "r", encoding="utf-8") as f_in, \
         open(PARALLEL_CSV, "w", encoding="utf-8", newline="") as f_out:

        writer = csv.writer(f_out)
        writer.writerow([
            "id", "ref", "transliteration", "translation_fr",
            "n_lines", "has_line_detail", "edition_reservee",
        ])

        for line in f_in:
            total += 1
            text_obj = json.loads(line)

            transliteration = extract_transliteration(text_obj)
            translation = extract_translation(text_obj)

            if not transliteration or not translation:
                continue

            with_pair += 1
            writer.writerow([
                text_obj.get("id", ""),
                text_obj.get("ref", ""),
                transliteration,
                translation,
                len(text_obj.get("transcriptions_lignes", [])),
                1 if text_obj.get("transcriptions_lignes") else 0,
                1 if text_obj.get("edition_reservee") else 0,
            ])

    print(f"✅ Parallel CSV: {with_pair}/{total} texts with both transliteration + translation")
    print(f"   → {PARALLEL_CSV}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Scrape archibab.fr for Akkadian-French corpus")
    parser.add_argument("--resume", action="store_true", help="Resume from last progress")
    parser.add_argument("--size", type=int, default=250, help="Page size (default: 250)")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between pages in seconds (default: 2.0)")
    parser.add_argument("--csv-only", action="store_true", help="Only build CSV from existing JSONL")
    args = parser.parse_args()

    if args.csv_only:
        build_parallel_csv()
        return

    scrape_all(resume=args.resume, page_size=args.size, delay=args.delay)
    build_parallel_csv()


if __name__ == "__main__":
    main()
