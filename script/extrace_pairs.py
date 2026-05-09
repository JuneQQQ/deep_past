#!/usr/bin/env python3
"""
Akkadian-English pair extractor for PDF corpora.

Features:
- All runtime settings are centralized in Config.
- Recursively scans PDFs with optional excludes.
- Renders each PDF page independently in single-page extraction mode.
- Runs LLM requests concurrently (default: 10 workers).
- Writes outputs into a timestamped run directory.
- Exports train.csv with the same schema as data/train.csv:
  oare_id, transliteration, translation
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
import re
import sys
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests

DEFAULT_PROMPT = """Role & Objective:
You are an Assyriology-focused high-precision extraction engine. You will be provided with the content of a SINGLE academic PDF page.

Your job is to extract ONLY high-confidence, fully complete Old Assyrian transliteration-English translation pairs to build a machine-translation training dataset. Precision is absolutely more important than recall.

Strict Page Boundary Rule (CRITICAL):
Because you are reading a single page, the first sentence at the very top or the last sentence at the very bottom might be cut off or fragmented due to page breaks.

DO NOT GUESS OR COMPLETE:
You are strictly forbidden from guessing, hallucinating, or auto-completing missing text from another page.

DROP FRAGMENTS:
If an Akkadian sentence OR its English translation starts mid-sentence, ends abruptly, ends with a dangling hyphen or slash such as "-" or "-/", contains an obvious continuation fragment, or lacks a logical grammatical conclusion, you MUST completely discard the entire pair.

100% CONTAINMENT:
Only extract pairs where BOTH the Akkadian source and the English translation are 100% visibly complete and fully contained within this specific page. When in doubt about completeness, throw it out.

Completeness Rejection Checklist:
Discard the pair if ANY of the following is true:
- the source or translation clearly begins in the middle of an existing sentence or clause
- the source or translation is visibly cut off at the top or bottom page boundary
- the source or translation ends with a dangling hyphen, slash, ellipsis, or other incomplete-break marker
- the translation is only a continuation dependent on off-page context rather than a complete standalone rendering
- you are uncertain whether the pair is complete

Hard Domain Filter:

ACCEPT: Old Assyrian letters, contracts, loans, trade, taxation, goods, household, and administrative records.

REJECT: If this page is clearly dominated by royal inscriptions, hymns, omens, lexical lists, grammar discussion, bibliography, or philological commentary, you must return the fallback marker.

What Counts as a Valid Pair:

The Akkadian side must be a real transliteration in Latin script, not cuneiform Unicode, not a sign list, and not word-by-word glossing.

The English side must be the final fluent translation printed by the editor, not literal interlinear glosses.

Extract only pairs that are clearly explicitly aligned on this page.

Always Ignore:

Word-by-word interlinear glosses.

Dictionaries, sign lists, lexical tables.

Commentary, notes, bibliography, concordances.

Page headers, page numbers, footnotes, figure captions, line numbers, reference markers.

Normalization Rules (Akkadian):

Preserve determinatives (e.g., {d}, {m}, {ki}).

Preserve hyphens, Sumerograms (capitalized), diacritics, subscripts, numbers, fractions, and placeholders like PN and GN.

Remove line numbers and obvious OCR debris.

Remove editorial square brackets [ ] but KEEP the readable text inside them.

Convert damage/unknown markers such as [x], [x x], isolated x, ..., …, or text like "broken line(s)" to exactly <gap>.

Remove angle brackets < > while keeping their inner content.

For <<...>> erroneous-sign notation, keep only the inner content.

If the Akkadian text is too fragmentary to be useful, skip the pair.

Normalization Rules (English):

Extract the final fluent English translation only.

KEEP Contextual Parentheses: Keep meaningful words added by scholars to make the English readable, including the parentheses themselves. Examples to keep: (of), (to), (and), (their), (this), (it is written).

REMOVE Grammatical/Editorial Parentheses: You MUST completely delete parentheses containing metadata, and any double spaces left behind. Examples to remove: (fem.), (pl.), (plural), (sing.), (masc.), (fem. plur.), (fem. sing.), (?), (!), (uncertain), (lit. ...), (meaning ...), (var.), (sic), (reading ...).

Preserve placeholders like PN, GN, numbers, fractions, and <gap>.

CRITICAL: Never split a single grammatical English sentence into multiple JSON objects. If a sentence translates to 'that Assur-taklaku left with you', it MUST be one single JSON object. Check your output: if any English string ends with a preposition (e.g., 'for the silver') or lacks a main verb, you have failed the extraction.
CRITICAL ALIGNMENT RULE (The SOV Verb Problem):
Akkadian uses Subject-Object-Verb (SOV) word order, placing the verb at the very end of the sentence. Often, the final verb wraps to the beginning of the next line on the tablet.
YOU MUST NOT LEAVE THE VERB DANGLING. If the English translation in your current pair contains a verb, you MUST ensure the corresponding Akkadian verb (e.g., bi-il₅, tù-šé-bi-lam, al-qé, a-lá-qé) is included at the end of the Akkadian string, even if you have to pull it from the beginning of the next line. Never split an Akkadian verb from its object if they belong to the same English sentence.

Output Format (CRITICAL):
Return ONLY a valid JSON array of objects with the exact keys "akkadian" and "english".
Do not wrap the output in Markdown code blocks. Output the raw array starting with [ and ending with ].

Example:
[
  {
    "akkadian": "adi a-wa-at kārim ni-ša-me-ú",
    "english": "until we hear the verdict (of) the kārum"
  }
]

Fallback:
If there are no high-confidence, fully complete pairs on this page, return EXACTLY the following string and nothing else:
[NO_VALID_PAIRS_FOUND]"""


@dataclass
class Config:
    input_pdf_root: Path = Path("/data/lsb/deep_past/data/old-assyrian")

    base_url: str = os.environ.get(
        "EXTRACT_PAIRS_BASE_URL",
        "https://www.dmxapi.cn/v1",
    ).strip()
    api_key: str = os.environ.get(
        "EXTRACT_PAIRS_API_KEY",
        "sk-WuVg5VaB0iV7NxJNXiOFjc7nFH9ge3aNI8n6hChX4HQKQNVp",
    ).strip()
    model_name: str = os.environ.get(
        "EXTRACT_PAIRS_MODEL",
        "gemini-3.1-pro-preview",
    ).strip()

    prompt: str = field(default_factory=lambda: DEFAULT_PROMPT)
    fallback_marker: str = "[NO_VALID_PAIRS_FOUND]"

    dpi: int = 300
    max_retries: int = 3
    retry_delay: float = 5.0
    request_timeout: int = 300
    temperature: float = 0.0
    max_tokens: int = 10240

    max_workers: int = 10
    max_inflight_factor: int = 2
    include_page_text: bool = True
    max_page_text_chars: int = 4000

    output_root: Path = Path("/data/lsb/deep_past/output")
    output_prefix: str = "extract_pairs"

    excludes: list[str] = field(default_factory=list)
    page_range_specs: list[str] = field(
        default_factory=lambda: [
            # "Larsen 2002 - The Assur-nada Archive. PIHANS 96 2002.pdf:51-282"
            # "AKT 6a.pdf:52-436"
            "AKT 8 2015.pdf:49-517"
            # "CAD_05_G_open.pdf:15-172",
            # "CAD_06_H_open.pdf:15-280",
            # "CAD_01-1_A-AL_open.pdf:37-392",
            # "CAD_01-2_AM-AZ_open.pdf:21-551",
            # "CAD_02_B_open.pdf:19-384",
            # "CAD_03_D_open.pdf:15-217",
            # "CAD_04_E_open.pdf:15-435",
            # "CAD_07_I-J_open.pdf:17-347",
            # "CAD_08_K_open.pdf:21-638",
            # "CAD_09_L_open.pdf:21-279",
            # "CAD_10-1_MA_open.pdf:25-465",
            # "CAD_10-2_ME-MU_open.pdf:21-344",
            # "CAD_11-1_NA-NARU_open.pdf:25-406",
            # "CAD_11-2_NAS-NUZ_open.pdf:23-379",
            # "CAD_12_P_open.pdf:31-589",
            # "CAD_13_Q_open.pdf:25-356",
            # "CAD_14_R_open.pdf:31-472",
            # "CAD_15_S_open.pdf:25-452",
            # "CAD_16_Sx_open.pdf:17-278",
            # "CAD_17-1-SzA-SzAP_open.pdf:29-520",
            # "CAD_17-2_SzAQ-SzILU_open.pdf:29-482",
            # "CAD_17-3_SzIM-SzUZ_open.pdf:25-444",
            # "CAD_18_T_open.pdf:31-530",
            # "CAD_19_Tx_open.pdf:33-199",
            # "CAD_20_ U_W.pdf:33-444",
            # "CAD_21_Z_open.pdf:17-186",
        ]
    )

    page_results_filename: str = "page_results.jsonl"
    pairs_metadata_filename: str = "pairs_metadata.jsonl"
    train_csv_filename: str = "train.csv"
    summary_filename: str = "summary.json"
    run_config_filename: str = "run_config.json"

    @property
    def max_inflight(self) -> int:
        return max(1, self.max_workers * self.max_inflight_factor)

    def validate(self) -> None:
        missing = []
        if not self.base_url:
            missing.append("base_url")
        if not self.api_key:
            missing.append("api_key")
        if not self.model_name:
            missing.append("model_name")
        if missing:
            joined = ", ".join(missing)
            raise ValueError(
                f"Missing required config value(s): {joined}. "
                "Set them in Config defaults, environment variables, or CLI flags."
            )


@dataclass(frozen=True)
class PageJob:
    pdf_path: str
    pdf_name: str
    page_num: int
    total_pages_in_pdf: int
    page_text: str
    image_b64: str


@dataclass
class PageResult:
    pdf_path: str
    pdf_name: str
    page_num: int
    total_pages_in_pdf: int
    status: str
    parse_status: str
    raw_response: str
    pairs: list[dict[str, str]]
    latency_s: float
    filtered_pair_count: int = 0
    error: str = ""


@dataclass(frozen=True)
class PageRangeRule:
    raw_spec: str
    pdf_spec: str
    pages: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Akkadian-English translation pairs from PDFs."
    )
    parser.add_argument(
        "pdf_folder",
        nargs="?",
        default=None,
        help="Optional PDF root directory. Defaults to Config.input_pdf_root.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Config:
    cfg = Config()
    cfg.validate()
    return cfg


def serialize_config(cfg: Config) -> dict[str, Any]:
    data = asdict(cfg)
    data["input_pdf_root"] = str(cfg.input_pdf_root)
    data["output_root"] = str(cfg.output_root)
    data["max_inflight"] = cfg.max_inflight
    return data


def ensure_run_dir(cfg: Config) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = cfg.output_root / f"{cfg.output_prefix}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def is_excluded(path: Path, excludes: list[str]) -> bool:
    resolved = path.resolve()
    parent_names = {p.name for p in path.parents}
    for entry in excludes:
        excluded_path = Path(entry).resolve()
        if resolved == excluded_path or excluded_path in resolved.parents:
            return True
        if path.name == entry or entry in parent_names:
            return True
    return False


def discover_pdfs(pdf_folder: Path, excludes: list[str]) -> tuple[list[Path], int]:
    all_pdfs = sorted(pdf_folder.rglob("*.pdf"))
    pdf_files = [pdf for pdf in all_pdfs if not is_excluded(pdf, excludes)]
    return pdf_files, len(all_pdfs) - len(pdf_files)


def parse_page_range_rules(raw_specs: list[str]) -> list[PageRangeRule]:
    rules: list[PageRangeRule] = []
    for raw_spec in raw_specs:
        clean = raw_spec.strip()
        if not clean:
            continue
        if ":" not in clean:
            raise ValueError(
                f"Invalid page range spec: {raw_spec}. "
                'Expected format like "AKT 1 1990.pdf:1-3,5,8-10".'
            )
        pdf_spec, pages_spec = clean.split(":", 1)
        pdf_spec = pdf_spec.strip()
        pages_spec = pages_spec.strip()
        if not pdf_spec or not pages_spec:
            raise ValueError(
                f"Invalid page range spec: {raw_spec}. "
                'Expected format like "AKT 1 1990.pdf:1-3,5,8-10".'
            )

        pages: set[int] = set()
        for chunk in pages_spec.split(","):
            token = chunk.strip()
            if not token:
                continue
            if "-" in token:
                start_text, end_text = token.split("-", 1)
                start = int(start_text.strip())
                end = int(end_text.strip())
                if start <= 0 or end <= 0 or start > end:
                    raise ValueError(f"Invalid page interval in spec: {raw_spec}")
                pages.update(range(start, end + 1))
            else:
                page_num = int(token)
                if page_num <= 0:
                    raise ValueError(f"Invalid page number in spec: {raw_spec}")
                pages.add(page_num)

        if not pages:
            raise ValueError(f"No valid pages found in spec: {raw_spec}")
        rules.append(
            PageRangeRule(
                raw_spec=raw_spec,
                pdf_spec=pdf_spec,
                pages=tuple(sorted(pages)),
            )
        )
    return rules


def pdf_matches_page_rule(pdf_path: Path, pdf_spec: str) -> bool:
    path_str = str(pdf_path)
    resolved_str = str(pdf_path.resolve())
    return (
        pdf_path.name == pdf_spec
        or pdf_path.stem == pdf_spec
        or path_str == pdf_spec
        or resolved_str == pdf_spec
        or path_str.endswith(pdf_spec)
        or resolved_str.endswith(pdf_spec)
    )


def filter_pdfs_by_page_range_rules(
    pdf_files: list[Path],
    rules: list[PageRangeRule],
) -> list[Path]:
    if not rules:
        return pdf_files
    return [
        pdf for pdf in pdf_files
        if any(pdf_matches_page_rule(pdf, rule.pdf_spec) for rule in rules)
    ]


def resolve_selected_pages(
    pdf_path: Path,
    total_pages_in_pdf: int,
    rules: list[PageRangeRule],
) -> tuple[list[int] | None, list[str]]:
    matched_specs: list[str] = []
    selected_pages: set[int] = set()

    for rule in rules:
        if pdf_matches_page_rule(pdf_path, rule.pdf_spec):
            matched_specs.append(rule.raw_spec)
            for page_num in rule.pages:
                if 1 <= page_num <= total_pages_in_pdf:
                    selected_pages.add(page_num)

    if not matched_specs:
        return None, []
    return sorted(selected_pages), matched_specs


def page_to_base64(page: fitz.Page, dpi: int) -> str:
    matrix = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    pix = page.get_pixmap(matrix=matrix, colorspace=fitz.csRGB)
    png_bytes = pix.tobytes("png")
    return base64.b64encode(png_bytes).decode("utf-8")


def extract_page_text(page: fitz.Page, max_chars: int) -> str:
    text = page.get_text("text") or ""
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars > 0:
        text = text[:max_chars]
    return text


def build_payload(cfg: Config, job: PageJob) -> dict[str, Any]:
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": cfg.prompt,
        },
        {
            "type": "text",
            "text": (
                f"CURRENT PAGE: PDF page {job.page_num} of {job.total_pages_in_pdf}. "
                "Single-page mode is active. Extract only pairs that are fully visible and complete on this page. "
                "Do not infer missing text from any previous or next page."
            ),
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{job.image_b64}",
                "detail": "high",
            },
        },
    ]
    if cfg.include_page_text and job.page_text:
        content.append(
            {
                "type": "text",
                "text": (
                    "Supplemental text extracted from the PDF for this same page only. "
                    "This text may be noisy; the page image is authoritative.\n\n"
                    f"{job.page_text}"
                ),
            }
        )

    return {
        "model": cfg.model_name,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
    }


def call_vision_api(cfg: Config, job: PageJob) -> str:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
    }
    payload = build_payload(cfg, job)
    url = f"{cfg.base_url.rstrip('/')}/chat/completions"

    last_error = ""
    for attempt in range(1, cfg.max_retries + 1):
        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=cfg.request_timeout,
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as exc:  # pragma: no cover - network/runtime path
            last_error = str(exc)
            print(
                f"    [API error attempt {attempt}/{cfg.max_retries}] {last_error}",
                flush=True,
            )
            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_delay)
    return cfg.fallback_marker if not last_error else f"{cfg.fallback_marker}\n{last_error}"


def strip_markdown_fences(text: str) -> str:
    clean = text.strip()
    if not clean.startswith("```"):
        return clean
    lines = clean.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_response(raw: str, fallback_marker: str) -> tuple[list[dict[str, str]], str]:
    if not raw.strip():
        return [], "empty_response"

    stripped = raw.strip()
    if stripped == fallback_marker or fallback_marker in stripped:
        return [], "fallback"

    clean = strip_markdown_fences(stripped)

    candidates = [clean]
    left = clean.find("[")
    right = clean.rfind("]")
    if left != -1 and right != -1 and left < right:
        candidates.append(clean[left : right + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if not isinstance(parsed, list):
            continue

        pairs: list[dict[str, str]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            akkadian = str(item.get("akkadian", "")).strip()
            english = str(item.get("english", "")).strip()
            if akkadian and english:
                pairs.append({"akkadian": akkadian, "english": english})

        if pairs:
            return pairs, "ok"
        return [], "empty_array"

    return [], "invalid_json"


def has_unbalanced_delimiters(text: str) -> bool:
    pairs = [("(", ")"), ("[", "]"), ("{", "}"), ("<", ">")]
    for left, right in pairs:
        if text.count(left) != text.count(right):
            return True
    return False


def is_obviously_incomplete_text(text: str, *, is_english: bool) -> bool:
    clean = text.strip()
    if not clean:
        return True
    if clean.startswith(("-", "/", "-/", "/-")):
        return True
    if clean.endswith(("-", "/", ",", ":", "-/", "/-")):
        return True
    if re.search(r"(?:^|\s)[-–—/]+$", clean):
        return True
    if has_unbalanced_delimiters(clean):
        return True
    if is_english:
        lowered = clean.lower()
        if re.search(
            r"\b(?:and|or|but|of|to|for|with|from|in|on|at|by|as|than|if|when|while|then)$",
            lowered,
        ):
            return True
    return False


def filter_incomplete_pairs(
    pairs: list[dict[str, str]],
) -> tuple[list[dict[str, str]], int]:
    filtered: list[dict[str, str]] = []
    removed = 0
    for pair in pairs:
        if is_obviously_incomplete_text(pair["akkadian"], is_english=False):
            removed += 1
            continue
        if is_obviously_incomplete_text(pair["english"], is_english=True):
            removed += 1
            continue
        filtered.append(pair)
    return filtered, removed


def process_page_job(cfg: Config, job: PageJob) -> PageResult:
    started = time.time()
    try:
        raw = call_vision_api(cfg, job)
        pairs, parse_status = parse_response(raw, cfg.fallback_marker)
        pairs, filtered_pair_count = filter_incomplete_pairs(pairs)
        if filtered_pair_count and parse_status == "ok":
            parse_status = "ok_filtered" if pairs else "filtered_incomplete"
        status = "ok" if pairs else "no_pairs"
        return PageResult(
            pdf_path=job.pdf_path,
            pdf_name=job.pdf_name,
            page_num=job.page_num,
            total_pages_in_pdf=job.total_pages_in_pdf,
            status=status,
            parse_status=parse_status,
            raw_response=raw,
            pairs=pairs,
            latency_s=round(time.time() - started, 3),
            filtered_pair_count=filtered_pair_count,
        )
    except Exception as exc:  # pragma: no cover - runtime guard
        return PageResult(
            pdf_path=job.pdf_path,
            pdf_name=job.pdf_name,
            page_num=job.page_num,
            total_pages_in_pdf=job.total_pages_in_pdf,
            status="error",
            parse_status="exception",
            raw_response="",
            pairs=[],
            latency_s=round(time.time() - started, 3),
            filtered_pair_count=0,
            error=str(exc),
        )


def make_oare_id(
    pdf_path: str,
    page_num: int,
    pair_index: int,
    pair: dict[str, str],
) -> str:
    seed = (
        f"{pdf_path}::page={page_num}::pair={pair_index}"
        f"::{pair['akkadian']}::{pair['english']}"
    )
    return str(uuid.uuid5(uuid.NAMESPACE_URL, seed))


def write_jsonl_line(handle: Any, record: dict[str, Any]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def handle_completed_future(
    future: Future[PageResult],
    future_to_job: dict[Future[PageResult], PageJob],
    train_writer: csv.DictWriter,
    page_results_fh: Any,
    pairs_metadata_fh: Any,
    summary: dict[str, Any],
) -> None:
    job = future_to_job.pop(future)
    result = future.result()

    summary["completed_pages"] += 1
    summary["page_status_counts"][result.status] = (
        summary["page_status_counts"].get(result.status, 0) + 1
    )
    summary["parse_status_counts"][result.parse_status] = (
        summary["parse_status_counts"].get(result.parse_status, 0) + 1
    )
    if result.error:
        summary["error_count"] += 1

    page_record = {
        "source_file": result.pdf_name,
        "source_path": result.pdf_path,
        "page": result.page_num,
        "total_pages_in_pdf": result.total_pages_in_pdf,
        "status": result.status,
        "parse_status": result.parse_status,
        "pair_count": len(result.pairs),
        "filtered_pair_count": result.filtered_pair_count,
        "latency_s": result.latency_s,
        "error": result.error,
        "raw_response": result.raw_response,
    }
    write_jsonl_line(page_results_fh, page_record)

    if result.pairs:
        for pair_index, pair in enumerate(result.pairs, start=1):
            oare_id = make_oare_id(
                result.pdf_path,
                result.page_num,
                pair_index,
                pair,
            )
            train_writer.writerow(
                {
                    "oare_id": oare_id,
                    "transliteration": pair["akkadian"],
                    "translation": pair["english"],
                }
            )
            write_jsonl_line(
                pairs_metadata_fh,
                {
                    "oare_id": oare_id,
                    "source_file": result.pdf_name,
                    "source_path": result.pdf_path,
                    "page": result.page_num,
                    "pair_index": pair_index,
                    "transliteration": pair["akkadian"],
                    "translation": pair["english"],
                },
            )
        summary["total_pairs"] += len(result.pairs)
        filtered_note = (
            f", filtered {result.filtered_pair_count}"
            if result.filtered_pair_count
            else ""
        )
        print(
            f"  Page {job.page_num}/{job.total_pages_in_pdf} [{job.pdf_name}] -> "
            f"{len(result.pairs)} pairs{filtered_note} ({result.latency_s:.1f}s)",
            flush=True,
        )
    else:
        suffix = f", error={result.error}" if result.error else ""
        filtered_note = (
            f", filtered {result.filtered_pair_count}"
            if result.filtered_pair_count
            else ""
        )
        print(
            f"  Page {job.page_num}/{job.total_pages_in_pdf} [{job.pdf_name}] -> "
            f"{result.parse_status} ({result.latency_s:.1f}s{filtered_note}{suffix})",
            flush=True,
        )


def render_and_submit_jobs(
    cfg: Config,
    pdf_files: list[Path],
    page_range_rules: list[PageRangeRule],
    matched_page_range_specs: set[str],
    executor: ThreadPoolExecutor,
    future_to_job: dict[Future[PageResult], PageJob],
    summary: dict[str, Any],
    train_writer: csv.DictWriter,
    page_results_fh: Any,
    pairs_metadata_fh: Any,
) -> None:
    for pdf_path in pdf_files:
        print(f"📄 Processing: {pdf_path.name}", flush=True)
        try:
            doc = fitz.open(str(pdf_path))
        except Exception as exc:
            print(f"  [ERROR opening PDF] {exc}", flush=True)
            summary["open_error_count"] += 1
            continue

        with doc:
            total_pages_in_pdf = len(doc)
            page_asset_cache: dict[int, tuple[str, str]] = {}

            def get_page_assets(page_num: int) -> tuple[str, str]:
                cached = page_asset_cache.get(page_num)
                if cached is not None:
                    return cached
                page = doc[page_num - 1]
                image_b64 = page_to_base64(page, cfg.dpi)
                page_text = (
                    extract_page_text(page, cfg.max_page_text_chars)
                    if cfg.include_page_text
                    else ""
                )
                page_asset_cache[page_num] = (image_b64, page_text)
                return image_b64, page_text

            selected_pages, matched_specs = resolve_selected_pages(
                pdf_path=pdf_path,
                total_pages_in_pdf=total_pages_in_pdf,
                rules=page_range_rules,
            )
            if matched_specs:
                matched_page_range_specs.update(matched_specs)
                if selected_pages:
                    print(
                        f"  Restricted pages: {selected_pages}",
                        flush=True,
                    )
                else:
                    print(
                        f"  No valid in-range pages after applying: {matched_specs}",
                        flush=True,
                    )
                    continue

            page_numbers = selected_pages or list(range(1, total_pages_in_pdf + 1))

            for page_num in page_numbers:
                summary["discovered_pages"] += 1
                try:
                    image_b64, page_text = get_page_assets(page_num)
                except Exception as exc:
                    summary["render_error_count"] += 1
                    summary["page_status_counts"]["render_error"] = (
                        summary["page_status_counts"].get("render_error", 0) + 1
                    )
                    write_jsonl_line(
                        page_results_fh,
                        {
                            "source_file": pdf_path.name,
                            "source_path": str(pdf_path.resolve()),
                            "page": page_num,
                            "total_pages_in_pdf": total_pages_in_pdf,
                            "status": "render_error",
                            "parse_status": "not_called",
                            "pair_count": 0,
                            "filtered_pair_count": 0,
                            "latency_s": 0.0,
                            "error": str(exc),
                            "raw_response": "",
                        },
                    )
                    print(
                        f"  Page {page_num}/{total_pages_in_pdf} [{pdf_path.name}] -> "
                        f"render_error ({exc})",
                        flush=True,
                    )
                    continue

                job = PageJob(
                    pdf_path=str(pdf_path.resolve()),
                    pdf_name=pdf_path.name,
                    page_num=page_num,
                    total_pages_in_pdf=total_pages_in_pdf,
                    page_text=page_text,
                    image_b64=image_b64,
                )
                future = executor.submit(process_page_job, cfg, job)
                future_to_job[future] = job

                while len(future_to_job) >= cfg.max_inflight:
                    done, _ = wait(
                        future_to_job.keys(),
                        return_when=FIRST_COMPLETED,
                    )
                    for finished in done:
                        handle_completed_future(
                            finished,
                            future_to_job,
                            train_writer,
                            page_results_fh,
                            pairs_metadata_fh,
                            summary,
                        )


def drain_futures(
    future_to_job: dict[Future[PageResult], PageJob],
    train_writer: csv.DictWriter,
    page_results_fh: Any,
    pairs_metadata_fh: Any,
    summary: dict[str, Any],
) -> None:
    while future_to_job:
        done, _ = wait(future_to_job.keys(), return_when=FIRST_COMPLETED)
        for future in done:
            handle_completed_future(
                future,
                future_to_job,
                train_writer,
                page_results_fh,
                pairs_metadata_fh,
                summary,
            )


def process_folder(pdf_folder: Path, cfg: Config) -> Path:
    pdf_files, excluded_count = discover_pdfs(pdf_folder, cfg.excludes)
    page_range_rules = parse_page_range_rules(cfg.page_range_specs)
    pdf_files = filter_pdfs_by_page_range_rules(pdf_files, page_range_rules)
    if not pdf_files:
        if page_range_rules:
            raise FileNotFoundError(
                f"Did not find any PDF under {pdf_folder} matching page_range_specs: {cfg.page_range_specs}"
            )
        raise FileNotFoundError(f"No PDF files found under {pdf_folder}")

    run_dir = ensure_run_dir(cfg)
    train_csv_path = run_dir / cfg.train_csv_filename
    page_results_path = run_dir / cfg.page_results_filename
    pairs_metadata_path = run_dir / cfg.pairs_metadata_filename
    summary_path = run_dir / cfg.summary_filename
    run_config_path = run_dir / cfg.run_config_filename

    with open(run_config_path, "w", encoding="utf-8") as handle:
        json.dump(serialize_config(cfg), handle, ensure_ascii=False, indent=2)

    summary: dict[str, Any] = {
        "pdf_folder": str(pdf_folder.resolve()),
        "run_dir": str(run_dir.resolve()),
        "pdf_count": len(pdf_files),
        "excluded_pdf_count": excluded_count,
        "page_range_specs": list(cfg.page_range_specs),
        "discovered_pages": 0,
        "completed_pages": 0,
        "total_pairs": 0,
        "open_error_count": 0,
        "render_error_count": 0,
        "error_count": 0,
        "page_status_counts": {},
        "parse_status_counts": {},
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    print(f"Found {len(pdf_files) + excluded_count} PDF(s) in {pdf_folder}")
    if excluded_count:
        print(f"Excluded {excluded_count} PDF(s) matching: {cfg.excludes}")
    if cfg.page_range_specs:
        print(f"Page range specs: {cfg.page_range_specs}")
    print(f"Concurrent LLM requests: {cfg.max_workers}")
    print(f"Run output directory: {run_dir}\n")

    started = time.time()
    future_to_job: dict[Future[PageResult], PageJob] = {}
    matched_page_range_specs: set[str] = set()

    with (
        open(train_csv_path, "w", encoding="utf-8", newline="") as train_fh,
        open(page_results_path, "w", encoding="utf-8") as page_results_fh,
        open(pairs_metadata_path, "w", encoding="utf-8") as pairs_metadata_fh,
        ThreadPoolExecutor(max_workers=cfg.max_workers) as executor,
    ):
        train_writer = csv.DictWriter(
            train_fh,
            fieldnames=["oare_id", "transliteration", "translation"],
        )
        train_writer.writeheader()

        render_and_submit_jobs(
            cfg=cfg,
            pdf_files=pdf_files,
            page_range_rules=page_range_rules,
            matched_page_range_specs=matched_page_range_specs,
            executor=executor,
            future_to_job=future_to_job,
            summary=summary,
            train_writer=train_writer,
            page_results_fh=page_results_fh,
            pairs_metadata_fh=pairs_metadata_fh,
        )
        drain_futures(
            future_to_job=future_to_job,
            train_writer=train_writer,
            page_results_fh=page_results_fh,
            pairs_metadata_fh=pairs_metadata_fh,
            summary=summary,
        )

    unmatched_page_range_specs = [
        rule.raw_spec
        for rule in page_range_rules
        if rule.raw_spec not in matched_page_range_specs
    ]
    summary["matched_page_range_specs"] = sorted(matched_page_range_specs)
    summary["unmatched_page_range_specs"] = unmatched_page_range_specs
    summary["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["elapsed_s"] = round(time.time() - started, 3)

    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    if unmatched_page_range_specs:
        print(f"Warning: unmatched page range specs: {unmatched_page_range_specs}")

    print(f"\n{'=' * 60}")
    print("Done")
    print(f"  Pages discovered     : {summary['discovered_pages']}")
    print(f"  Pages completed      : {summary['completed_pages']}")
    print(f"  Pairs extracted      : {summary['total_pairs']}")
    print(f"  Train CSV            : {train_csv_path}")
    print(f"  Page results JSONL   : {page_results_path}")
    print(f"  Pairs metadata JSONL : {pairs_metadata_path}")
    print(f"  Summary JSON         : {summary_path}")
    return run_dir


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    pdf_folder = Path(args.pdf_folder) if args.pdf_folder else cfg.input_pdf_root

    if not pdf_folder.exists():
        raise FileNotFoundError(f"Input folder does not exist: {pdf_folder}")
    if not pdf_folder.is_dir():
        raise NotADirectoryError(f"Input path is not a directory: {pdf_folder}")

    process_folder(pdf_folder=pdf_folder, cfg=cfg)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
