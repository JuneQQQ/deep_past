#!/usr/bin/env python3
"""
Scrape cdli.earth for Akkadian parallel corpus (ATF transliteration + English translation).

Strategy
--------
Phase 1 (discover) : Paginate search results → collect artifact numeric IDs.
Phase 2 (download) : For each ID, fetch JSON API → extract genre + ATF → parse TL + TR.
                     Genre filter applied in-flight; blocked artifacts are discarded.
Phase 3 (build)    : Write genre-filtered parallel pairs to CSV.

Endpoints
---------
Search : https://cdli.earth/search?translation=Yes&translationLang=en&period=Old+Assyrian&limit=1000
JSON   : https://cdli.earth/artifacts/{id}/json   (genre + full ATF with translations)

Output
------
data/cdli/cdli_raw.jsonl      – all parsed ATF records that passed genre filter
data/cdli/cdli_parallel.csv   – quality-filtered parallel pairs

Usage
-----
    python script/scrape_cdli.py [--resume] [--delay 1.0]
    python script/scrape_cdli.py --no-genre          # skip genre filtering
    python script/scrape_cdli.py --csv-only           # rebuild CSV from JSONL
    python script/scrape_cdli.py --period "Old Babylonian"  # other period
"""

import argparse
import concurrent.futures
import csv
import json
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlencode

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

REQ_TIMEOUT = (10, 30)  # (connect, read) seconds

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:125.0) Gecko/20100101 Firefox/125.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
]

# ── Constants ──────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "data" / "cdli"
RAW_JSONL = OUTPUT_DIR / "cdli_raw.jsonl"
PARALLEL_CSV = OUTPUT_DIR / "cdli_parallel.csv"
PROGRESS_FILE = OUTPUT_DIR / ".scrape_progress.json"

SEARCH_BASE = "https://cdli.earth/search"
JSON_TEMPLATE = "https://cdli.earth/artifacts/{pid}/json"

PROXY = "http://127.0.0.1:7890"

# ── Genre filter ───────────────────────────────────────────────────────────────
# Substrings matched case-insensitively against the genre string from CDLI.
# If the genre is empty / unknown we keep the record (benefit of the doubt).

ALLOWED_GENRE_KEYWORDS = {
    "letter", "administrative", "legal", "account", "contract",
    "debt", "receipt", "loan", "memo", "agreement", "order",
    "note", "verdict", "regulation", "litigation", "purchase",
    "warrant", "protocol", "inventory",
}

BLOCKED_GENRE_KEYWORDS = {
    "hymn", "prayer", "incantation", "ritual", "lexical",
    "school", "mathematical", "omen", "literary", "votive",
    "building", "year name", "royal", "monumental", "seal",
    "inscription", "astronomical", "medical",
}


def is_genre_allowed(genre: str) -> bool:
    """Return True if genre passes the filter."""
    if not genre:
        return True  # unknown → keep
    gl = genre.lower()
    for kw in BLOCKED_GENRE_KEYWORDS:
        if kw in gl:
            return False
    # If we have a genre string and none of the allowed keywords match,
    # still allow it (we only block explicitly bad ones).
    return True


# ── ATF Parsing ────────────────────────────────────────────────────────────────

def normalize_atf(text: str) -> str:
    """Convert CDLI ATF ASCII notation to Unicode transliteration."""
    # Multi-char first
    text = text.replace("sz", "š").replace("SZ", "Š")
    text = re.sub(r"s,", "ṣ", text)
    text = re.sub(r"S,", "Ṣ", text)
    text = re.sub(r"t,", "ṭ", text)
    text = re.sub(r"T,", "Ṭ", text)

    # Sumerograms: _word-word_ → WORD.WORD
    def _sumer(m):
        inner = m.group(1).upper()
        return inner

    text = re.sub(r"_([^_]+)_", _sumer, text)
    return text


def parse_atf(raw: str) -> dict | None:
    """Parse raw ATF text into a structured record.

    Returns None if the text has no usable transliteration.
    """
    if not raw or not raw.strip():
        return None

    lines = raw.strip().split("\n")
    p_number = ""
    publication = ""
    language = ""
    tl_parts: list[str] = []
    tr_parts: list[str] = []

    # Header: &P{id} = {publication}
    if lines[0].startswith("&"):
        m = re.match(r"&(P\d+)\s*=\s*(.*)", lines[0])
        if m:
            p_number, publication = m.group(1), m.group(2).strip()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Language declaration
        m = re.match(r"#\s*atf:\s*lang\s+(.+)", line)
        if m:
            language = m.group(1).strip()
            continue

        # Section / surface / ruling markers
        if line.startswith("@") or line.startswith("$"):
            continue

        # English translation: # tr.en: text
        m = re.match(r"#\s*tr\.en:\s*(.*)", line)
        if m:
            t = m.group(1).strip()
            if t:
                tr_parts.append(t)
            continue

        # Skip other comments / notes
        if line.startswith("#"):
            continue

        # Transliteration line: N. text  (N may have prime ′ marks)
        m = re.match(r"^(\d+['\u2032]*)\.\s+(.*)", line)
        if m:
            t = m.group(2).strip()
            if t:
                tl_parts.append(t)
            continue

    if not tl_parts:
        return None

    tl_raw = " ".join(tl_parts)
    return {
        "p_number": p_number,
        "publication": publication,
        "language": language,
        "transliteration_raw": tl_raw,
        "transliteration": normalize_atf(tl_raw),
        "translation": " ".join(tr_parts),
        "n_tl_lines": len(tl_parts),
        "n_tr_lines": len(tr_parts),
    }


# ── JSON API helpers ───────────────────────────────────────────────────────────

def extract_from_json(data: list | dict) -> dict | None:
    """Extract genre, ATF, period, language from the CDLI JSON API response.

    The JSON endpoint returns a list with one object containing:
      - genres[].genre.genre  → e.g. "Legal", "Letter"
      - inscription.atf       → full ATF text with # tr.en: lines
      - period.name           → e.g. "Old Assyrian"
      - languages[].language.language → e.g. "Old Assyrian"
      - designation            → publication name
    """
    if isinstance(data, list):
        if not data:
            return None
        data = data[0]

    # Genre
    genres_raw = data.get("genres", [])
    genre_parts = []
    for g in genres_raw:
        gname = (g.get("genre") or {}).get("genre", "")
        if gname:
            genre_parts.append(gname)
    genre = "; ".join(genre_parts)

    # ATF
    inscription = data.get("inscription")
    atf_text = ""
    if inscription:
        atf_text = inscription.get("atf", "")

    # Period
    period_obj = data.get("period") or {}
    period_name = period_obj.get("name", "")

    # Language
    langs_raw = data.get("languages", [])
    lang_parts = []
    for l in langs_raw:
        lname = (l.get("language") or {}).get("language", "")
        if lname:
            lang_parts.append(lname)
    language = "; ".join(lang_parts)

    designation = data.get("designation", "")

    return {
        "genre": genre,
        "atf": atf_text,
        "period": period_name,
        "language_meta": language,
        "designation": designation,
    }


# ── Search result parsing ─────────────────────────────────────────────────────

def extract_artifact_ids(html: str) -> list[int]:
    """Extract artifact numeric IDs from search results HTML."""
    # P-numbers in text: (P123456)
    p_nums = re.findall(r"\(P(\d+)\)", html)
    # Also from href="/artifacts/123456"
    href_ids = re.findall(r'href="/artifacts/(\d+)"', html)
    # Combine, deduplicate, exclude common non-artifact IDs
    all_ids = set()
    for x in p_nums + href_ids:
        if x.isdigit():
            n = int(x)
            if n > 1000:  # real P-numbers are large
                all_ids.add(n)
    return sorted(all_ids)


# ── Phase 1: Discovery ────────────────────────────────────────────────────────

def discover_all_ids(
    session: requests.Session, period: str, delay: float, max_pages: int = 0,
) -> list[int]:
    """Paginate search results to collect all artifact IDs."""
    all_ids: set[int] = set()
    page = 1
    limit = 1000  # CDLI supports up to 10000

    print("🔍 Phase 1: Discovering artifacts with English translations...")
    print(f"   period={period!r}  limit={limit}")

    while True:
        params = {
            "translation": "Yes",
            "translationLang": "en",
            "period": period,
            "limit": limit,
            "page": page,
        }

        print(f"   📡 Search page {page}...", end="", flush=True)
        try:
            resp = session.get(SEARCH_BASE, params=params, timeout=(10, 120))
            resp.raise_for_status()
        except Exception as e:
            print(f" ❌ {e}")
            break

        ids = extract_artifact_ids(resp.text)
        if not ids:
            print(" (no results, stopping)")
            break

        new_ids = set(ids) - all_ids
        all_ids.update(ids)
        print(f"  → {len(ids)} artifacts ({len(new_ids)} new), cumulative: {len(all_ids)}")

        # Check if there's a next page
        next_page_marker = f"page={page + 1}"
        if next_page_marker not in resp.text:
            break
        if max_pages and page >= max_pages:
            print(f"   (reached --max-pages={max_pages})")
            break

        page += 1
        time.sleep(delay)

    print(f"   ✅ Discovery: {len(all_ids)} unique artifact IDs")
    return sorted(all_ids)


# ── Phase 2: Download + Parse + Filter ────────────────────────────────────────

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {
        "phase": "discover",
        "discovered_ids": [],
        "processed_ids": [],
        "stats": {},
    }


def save_progress(progress: dict):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f)


def scrape_all(
    resume: bool,
    period: str,
    delay: float,
    check_genre: bool,
    max_pages: int,
    use_proxy: bool = False,
):
    """Main scraping pipeline using requests with anti-detection."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    progress = (
        load_progress()
        if resume
        else {
            "phase": "discover",
            "discovered_ids": [],
            "processed_ids": [],
            "stats": {},
        }
    )
    processed_set = set(progress.get("processed_ids", []))

    session = requests.Session()
    if use_proxy:
        session.proxies = {"http": PROXY, "https": PROXY}
    session.verify = False
    session.headers.update({
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/json,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
    })
    retry_strategy = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503])
    session.mount("https://", HTTPAdapter(max_retries=retry_strategy))
    session.mount("http://", HTTPAdapter(max_retries=retry_strategy))

    try:
        # ── Phase 1: Discovery ──
        if progress["phase"] == "discover" or not progress.get("discovered_ids"):
            ids = discover_all_ids(session, period, delay, max_pages)
            progress["discovered_ids"] = ids
            progress["phase"] = "download"
            save_progress(progress)
        else:
            ids = progress["discovered_ids"]
            print(f"📋 Loaded {len(ids)} previously discovered artifact IDs")

        # ── Phase 2: Download JSON API → extract ATF + genre ──
        remaining = [pid for pid in ids if pid not in processed_set]
        print(
            f"\n📥 Phase 2: Downloading {len(remaining)} artifacts via JSON API "
            f"(already done: {len(processed_set)})"
        )
        print(f"   Genre filtering: {'ON' if check_genre else 'OFF (--no-genre)'}")

        jsonl_mode = "a" if resume and processed_set else "w"
        stats = progress.get("stats", {})
        for k in ("total", "saved", "no_json", "no_atf", "no_tl", "no_tr", "genre_blocked", "error"):
            stats.setdefault(k, 0)
        genre_counter: Counter = Counter()

        with open(RAW_JSONL, jsonl_mode, encoding="utf-8") as f_out:
            for i, pid in enumerate(remaining):
                t0 = time.time()
                stats["total"] += 1

                try:  # try/finally ensures sleep+progress always run

                    # 2a. Fetch JSON API
                    json_url = JSON_TEMPLATE.format(pid=pid)
                    try:
                        resp = session.get(json_url, timeout=REQ_TIMEOUT)
                        if resp.status_code == 404:
                            stats["no_json"] += 1
                            processed_set.add(pid)
                            continue
                        resp.raise_for_status()
                    except requests.exceptions.Timeout:
                        stats["error"] += 1
                        processed_set.add(pid)
                        continue
                    except Exception as e:
                        if (i + 1) <= 20 or (i + 1) % 200 == 0:
                            print(f"   ❌ P{pid}: {str(e)[:60]}", flush=True)
                        stats["error"] += 1
                        processed_set.add(pid)
                        continue

                    # 2b. Parse JSON response
                    try:
                        json_data = resp.json()
                    except Exception:
                        stats["error"] += 1
                        processed_set.add(pid)
                        continue

                    meta = extract_from_json(json_data)
                    if not meta or not meta["atf"]:
                        stats["no_atf"] += 1
                        processed_set.add(pid)
                        continue

                    # 2c. Quick check: skip ATFs without any English translation
                    atf_text = meta["atf"]
                    if "tr.en" not in atf_text:
                        stats["no_tr"] += 1
                        processed_set.add(pid)
                        continue

                    # 2d. Genre filter
                    genre = meta["genre"]
                    genre_counter[genre or "(empty)"] += 1
                    if check_genre and not is_genre_allowed(genre):
                        stats["genre_blocked"] += 1
                        processed_set.add(pid)
                        continue

                    # 2e. Parse ATF
                    record = parse_atf(atf_text)
                    if not record:
                        stats["no_tl"] += 1
                        processed_set.add(pid)
                        continue
                    if not record["translation"]:
                        stats["no_tr"] += 1
                        processed_set.add(pid)
                        continue

                    # Enrich with metadata
                    record["genre"] = genre
                    record["period"] = meta.get("period", "")

                    # 2f. Save
                    f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_out.flush()
                    stats["saved"] += 1
                    processed_set.add(pid)

                    elapsed = time.time() - t0
                    print(
                        f"   [{i+1}/{len(remaining)}] P{pid} "
                        f"| genre='{genre}' "
                        f"| tl={record['n_tl_lines']}L tr={record['n_tr_lines']}L "
                        f"| {elapsed:.1f}s "
                        f"| saved={stats['saved']}",
                        flush=True,
                    )

                finally:
                    # === ALWAYS runs: sleep + periodic progress ===
                    time.sleep(delay + random.uniform(0.1, 0.8))

                    if (i + 1) % 200 == 0:
                        progress["processed_ids"] = sorted(processed_set)
                        progress["stats"] = stats
                        save_progress(progress)
                        print(
                            f"   💾 [{i+1}/{len(remaining)}] "
                            f"saved={stats['saved']} err={stats['error']} "
                            f"no_tr={stats['no_tr']} no_atf={stats['no_atf']}",
                            flush=True,
                        )

        # Final save
        progress["processed_ids"] = sorted(processed_set)
        progress["stats"] = stats
        progress["phase"] = "done"
        save_progress(progress)

        print(f"\n{'='*60}")
        print(f"✅ Scraping complete!")
        print(f"   Total attempted : {stats['total']}")
        print(f"   Saved           : {stats['saved']}")
        print(f"   No JSON/404     : {stats['no_json']}")
        print(f"   No ATF in JSON  : {stats['no_atf']}")
        print(f"   No transliter.  : {stats['no_tl']}")
        print(f"   No translation  : {stats['no_tr']}")
        print(f"   Genre blocked   : {stats['genre_blocked']}")
        print(f"   Errors          : {stats['error']}")
        print(f"\n📊 Genre distribution (top 20):")
        for g, cnt in genre_counter.most_common(20):
            allowed = "✅" if is_genre_allowed(g if g != "(empty)" else "") else "🚫"
            print(f"   {allowed} {g}: {cnt}")
        print(f"\n   → {RAW_JSONL}")

    finally:
        session.close()


# ── Phase 3: Build CSV ────────────────────────────────────────────────────────

def build_csv():
    """Build quality-filtered parallel CSV from raw JSONL."""
    if not RAW_JSONL.exists():
        print(f"❌ {RAW_JSONL} not found. Run scraping first.")
        return

    print(f"📝 Building parallel CSV from {RAW_JSONL}...")

    records = []
    with open(RAW_JSONL, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Quality filter
    good = []
    for r in records:
        tl = r.get("transliteration", "")
        tr = r.get("translation", "")
        if not tl or not tr:
            continue
        if len(tr) < 20:
            continue
        if len(tl) < 10:
            continue
        # Ratio check
        ratio = len(tr) / max(len(tl), 1)
        if ratio < 0.1 or ratio > 8:
            continue
        good.append(r)

    with open(PARALLEL_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "p_number",
            "transliteration",
            "translation",
            "genre",
            "publication",
            "language",
            "n_tl_lines",
            "n_tr_lines",
        ])
        for r in good:
            w.writerow([
                r["p_number"],
                r["transliteration"],
                r["translation"],
                r.get("genre", ""),
                r.get("publication", ""),
                r.get("language", ""),
                r.get("n_tl_lines", 0),
                r.get("n_tr_lines", 0),
            ])

    genres = Counter(r.get("genre", "") or "(empty)" for r in good)
    print(f"✅ Parallel CSV: {len(good)}/{len(records)} records")
    print(f"   → {PARALLEL_CSV}")
    print(f"   Genre breakdown: {dict(genres.most_common(10))}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Scrape cdli.earth for Akkadian parallel corpus"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last progress"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between requests in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--period",
        type=str,
        default="Old Assyrian",
        help='CDLI period filter (default: "Old Assyrian")',
    )
    parser.add_argument(
        "--no-genre",
        action="store_true",
        help="Skip genre checking (faster, saves all records)",
    )
    parser.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="Max search pages to crawl (0=unlimited, default: 0)",
    )
    parser.add_argument(
        "--csv-only",
        action="store_true",
        help="Only rebuild CSV from existing JSONL",
    )
    parser.add_argument(
        "--proxy",
        action="store_true",
        help="Use HTTP proxy (default: direct connection)",
    )
    args = parser.parse_args()

    if args.csv_only:
        build_csv()
        return

    scrape_all(
        resume=args.resume,
        period=args.period,
        delay=args.delay,
        check_genre=not args.no_genre,
        max_pages=args.max_pages,
        use_proxy=args.proxy,
    )
    build_csv()


if __name__ == "__main__":
    main()
