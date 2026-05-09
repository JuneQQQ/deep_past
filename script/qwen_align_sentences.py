#!/usr/bin/env python3
"""
Qwen sentence alignment script — v2 (hardened)

Key changes vs v1:
  1. Prompt now has ABSOLUTE RULES forbidding transliteration edits, gap filling, number changes.
  2. cleaned_block removed from JSON schema — LLM only returns split points.
  3. Hard constraint checks: gap count, number preservation, diacritics signature, substring check.
  4. Validation compares pairs vs SOURCE directly (not vs LLM's cleaned_block).
  5. Thresholds tightened (translit sim 0.55→0.92, etc.).
  6. Optional force_source_transliteration post-processing to guarantee fidelity.
"""
from __future__ import annotations

import csv
import importlib.metadata as importlib_metadata
import json
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from json_repair import repair_json
from tqdm import tqdm
from transformers import AutoProcessor, AutoTokenizer

# Make the worker start method explicit before importing vLLM internals.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


# ═══════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════

class Config:
    # --- Model ---
    MODEL_PATH = "/data/lsb/deep_past/output/checkpoint-2400-merged"

    # --- Paths ---
    INPUT_FILE = "/data/lsb/deep_past/data/data_k/train_clean.csv"
    OUTPUT_FILE = "/data/lsb/deep_past/data/data_k/train_clean_qwen.csv"
    REJECT_FILE = "/data/lsb/deep_past/data/data_k/train_clean_qwen_reject.csv"
    ERROR_LOG = "/data/lsb/deep_past/data/data_k/qwen_sentence_align_error.log"
    DEBUG_TRACE_FILE = "/data/lsb/deep_past/data/data_k/qwen_sentence_align_debug.jsonl"

    # --- Input ---
    INPUT_FORMAT = "auto"
    MAX_INPUT_RECORDS = 0
    CSV_FALLBACK_FIELDNAMES = [
        "oare_id", "transliteration", "translation", "data_source", "tl_restored",
    ]
    OUTPUT_FIELDNAMES = [
        "oare_id", "transliteration", "translation", "data_source", "dialect",
    ]
    DEFAULT_DIALECT = "OA"

    # --- Field candidates ---
    AKKADIAN_FIELD_CANDIDATES = [
        "raw_akkadian", "transliteration", "akkadian",
        "source_transliteration", "block_transliteration", "text_transliteration",
    ]
    ENGLISH_FIELD_CANDIDATES = [
        "raw_english", "translation", "english",
        "source_translation", "block_translation", "text_translation",
    ]

    # --- Inference ---
    TENSOR_PARALLEL_SIZE = 1
    MAX_MODEL_LEN =  10240
    MAX_NEW_TOKENS = 5120
    MAX_NUM_SEQS = 8
    GPU_MEMORY_UTILIZATION = float(
        os.environ.get("QWEN_CLEAN_GPU_MEMORY_UTILIZATION", "0.8"),
    )
    VLLM_ATTENTION_BACKEND = os.environ.get("QWEN_CLEAN_ATTENTION_BACKEND", "FLASHINFER").upper()
    REQUEST_BATCH_SIZE = max(
        1, int(os.environ.get("QWEN_CLEAN_REQUEST_BATCH_SIZE", str(MAX_NUM_SEQS))),
    )
    TEMPERATURE = 0
    TOP_P = 0.8
    JSON_PARSE_RETRY_COUNT = 9

    # --- Validation thresholds (TIGHTENED in v2) ---
    MAX_SENTENCE_PAIRS = 48

    # v2: transliteration should be nearly identical (LLM only splits, doesn't edit)
    MIN_SOURCE_SIM_TRANSLIT = 0.92          # was 0.55
    MIN_SOURCE_SIM_TRANSLATION = 0.85       # was 0.70

    # v2: reassembled pairs must cover nearly all of source
    MIN_COVERAGE_SIM_TRANSLIT = 0.90        # was 0.78
    MIN_COVERAGE_SIM_TRANSLATION = 0.88     # was 0.82

    # v2: tighter length bounds — LLM shouldn't add content
    MIN_LENGTH_RATIO = 0.80                 # was 0.55
    MAX_LENGTH_RATIO = 1.10                 # was 1.45

    MAX_DUPLICATE_PAIR_RATIO = 0.25
    MAX_SHORT_TRANSLATION_PAIR_RATIO = 0.34
    MIN_TRANSLATION_WORDS_PER_PAIR = 2

    # --- v2: hard constraint thresholds ---
    HARD_DIACRITICS_MIN_SIM = 0.90
    HARD_TRANSLIT_MAX_GROWTH = 1.05         # max 5% token growth
    HARD_TRANSLIT_WORD_OVERLAP_MIN = 0.80
    HARD_NUMBER_MAX_LOSS_RATIO = 0.20       # lose at most 20% of numbers
    HARD_TRANSLATION_WORD_PRECISION_MIN = 0.96
    HARD_TRANSLATION_MAX_INSERT_RATIO = 0.05
    HARD_TRANSLATION_MAX_DELETE_RATIO = 0.05

    # --- v2: post-processing ---
    FORCE_SOURCE_TRANSLITERATION = True     # replace LLM translit with source substr
    DEDUP_KEEP_ROWS = True

    # --- 长度阈值（用于区分是否需要对齐）---
    MIN_ALIGN_LENGTH = 240  # 阿卡德语原文低于此值不需要对齐，只判断是否丢弃


# ═══════════════════════════════════════════════════════════
# System prompt (v2 — hardened)
# ═══════════════════════════════════════════════════════════

SYSTEM_PROMPT_SHORT = """\
You are an expert Assyriologist and bilingual alignment specialist.

Your job is to evaluate whether a short Akkadian–English parallel block should be kept or discarded.

## ABSOLUTE RULES — violations cause automatic rejection:

1. **NEVER modify the transliteration.** Copy it character-for-character.
   - Do NOT correct diacritics (do NOT change ú→u, š→s, á→a, etc.)
   - Do NOT change sign readings (do NOT change is-→iṣ-, dan→dān, etc.)
   - Do NOT fill in <gap> markers — they represent physical damage on the tablet
   - Do NOT add or remove hyphens between syllables
   - Do NOT reorder tokens

2. **NEVER modify numbers** in the translation. "4 minas" stays "4 minas", not "34 minas".

3. **NEVER fill <gap> markers.** Every <gap> in the source MUST appear in your output
   at exactly the same position. Do not replace <gap> with guessed text.

4. **NEVER add content** that is not in the original. Do not complete fragmentary
   sentences. Do not add explanatory text.

5. **Preserve quotes and punctuation** from the original translation faithfully.

## What you CAN do:
- Remove layout markers: obv., rev., l.e., u.e., le.e., re.e.
- Remove line-number artifacts (e.g. "1. " at start of lines)
- Fix split-word markers like "-/"
- Normalize whitespace
- Simply decide whether to keep or reject this block (no sentence splitting needed)

## Output format:
Return ONLY valid JSON in this exact schema:
{
  "decision": "keep" | "reject",
  "reject_reason": "",
  "sentence_pairs": []
}

If decision is "reject", explain why in reject_reason.
sentence_pairs must be an empty list.

Allowed reject_reason values:
too_noisy, too_fragmentary, not_parallel, alignment_uncertain, missing_content, malformed_input
"""

SYSTEM_PROMPT_LONG = """\
You are an expert Assyriologist and bilingual sentence alignment specialist.

Your job is to split an Akkadian–English parallel block into aligned sentence pairs.

## ABSOLUTE RULES — violations cause automatic rejection:

1. **NEVER modify the transliteration.** Copy it character-for-character.
   - Do NOT correct diacritics (do NOT change ú→u, š→s, á→a, etc.)
   - Do NOT change sign readings (do NOT change is-→iṣ-, dan→dān, etc.)
   - Do NOT fill in <gap> markers — they represent physical damage on the tablet
   - Do NOT add or remove hyphens between syllables
   - Do NOT reorder tokens

2. **NEVER modify numbers** in the translation. "4 minas" stays "4 minas", not "34 minas".

3. **NEVER fill <gap> markers.** Every <gap> in the source MUST appear in your output
   at exactly the same position. Do not replace <gap> with guessed text.

4. **NEVER add content** that is not in the original. Do not complete fragmentary
   sentences. Do not add explanatory text.

5. **Preserve quotes and punctuation** from the original translation faithfully.

## What you CAN do:
- Remove layout markers: obv., rev., l.e., u.e., le.e., re.e.
- Remove line-number artifacts (e.g. "1. " at start of lines)
- Fix split-word markers like "-/"
- Normalize whitespace
- Split the block into sentence-aligned pairs

## Output format:
Return ONLY valid JSON in this exact schema:
{
  "decision": "keep" | "reject",
  "reject_reason": "",
  "sentence_pairs": [
    {"transliteration": "...", "translation": "..."}
  ]
}

If decision is "keep", sentence_pairs must cover the entire block in order.
If decision is "reject", sentence_pairs must be an empty list.

Allowed reject_reason values:
too_noisy, too_fragmentary, not_parallel, alignment_uncertain, missing_content, malformed_input"""


# ═══════════════════════════════════════════════════════════
# Regex patterns
# ═══════════════════════════════════════════════════════════

LAYOUT_MARKER_RE = re.compile(
    r"(?:(?<=^)|(?<=\s))(?:obv\.|rev\.|l\.e\.|u\.e\.|le\.e\.|re\.e\.)(?=\s|$)",
    re.IGNORECASE,
)
LINE_NUMBER_PREFIX_RE = re.compile(r"^\s*\d+(?:-\d+)?\*?\s*")
MULTISPACE_RE = re.compile(r"\s+")
TOKEN_WORD_RE = re.compile(r"[A-Za-zÀ-ÿḫḪšŠṣṢṭṬ0-9]+")
NUMBER_RE = re.compile(r"\b\d+(?:\s*[½⅓⅔¼¾])?\b")
DIACRITICS_RE = re.compile(r"[àáâãäåèéêëìíîïòóôõöùúûüšŠḫḪṣṢṭṬ]")
ALIGNMENT_TOKEN_RE = re.compile(r"<gap>|[A-Za-zÀ-ÿḫḪšŠṣṢṭṬ0-9½⅓⅔¼¾]+")
NUMERIC_GAP_CONTEXT_RE = re.compile(
    r"(?:\b\d+(?:\s*[½⅓⅔¼¾])?\b|\bme-at\b|\bli-im\b|\bGÍN\b|\bma-na\b|\bGÚ\b|\bTÚG\b)"
    r"(?:\s+\S+){0,2}\s*<gap>"
    r"|<gap>\s*(?:\s+\S+){0,2}"
    r"(?:\b\d+(?:\s*[½⅓⅔¼¾])?\b|\bme-at\b|\bli-im\b|\bGÍN\b|\bma-na\b|\bGÚ\b|\bTÚG\b)",
    re.IGNORECASE,
)

ENGLISH_STOPWORDS = {
    "about", "after", "and", "because", "before", "between", "could",
    "during", "from", "have", "into", "should", "the", "that", "their",
    "there", "them", "then", "these", "this", "those", "through",
    "under", "upon", "were", "which", "while", "with", "without", "would",
}

HANGING_TRANSLATION_END_TOKENS = {
    "a", "an", "and", "as", "at", "because", "by", "for", "from", "if",
    "in", "into", "is", "my", "of", "on", "or", "our", "since", "that",
    "the", "their", "then", "these", "this", "those", "to", "we", "when",
    "while", "with", "without", "your",
}
MID_SENTENCE_START_TOKENS = HANGING_TRANSLATION_END_TOKENS | {
    "don't", "not", "there",
}
PAIR_ESCAPE_ARTIFACT_RE = re.compile(r'(^|[\s"\'([{])n(?=[A-Z0-9])')
TERMINAL_BOUNDARY_RE = re.compile(r'(?:[.!?;:]|<gap>)(?:["\')\]]+)?$')
SUSPICIOUS_DECIMAL_RE = re.compile(r"\b\d*\.(?:3333\d*|6666\d*)\b")
SUSPICIOUS_SUBSCRIPT_TOKEN_RE = re.compile(r"[^\s]*[₀₁₂₃₄₅₆₇₈₉][^\s]*")


# ═══════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════

@dataclass
class ValidationResult:
    keep: bool
    reject_reason: str
    cleaned_transliteration: str
    cleaned_translation: str
    pairs: list[dict[str, str]]
    metrics: dict[str, float | int | str]
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class ParseAttemptLog:
    attempt: int
    raw_model_output: str
    error: str = ""


class ParseRetryExhaustedError(RuntimeError):
    def __init__(self, last_error: Exception, attempt_logs: list[ParseAttemptLog]) -> None:
        super().__init__(str(last_error))
        self.last_error = last_error
        self.attempt_logs = attempt_logs


# ═══════════════════════════════════════════════════════════
# Text utilities
# ═══════════════════════════════════════════════════════════

def normalize_space(text: Any) -> str:
    return MULTISPACE_RE.sub(" ", str(text or "")).strip()


def decode_literal_whitespace_escapes(text: str) -> str:
    return (
        str(text or "")
        .replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", " ")
    )


def strip_line_number_prefixes(text: str) -> str:
    return "\n".join(
        LINE_NUMBER_PREFIX_RE.sub("", line)
        for line in str(text or "").splitlines()
    )


def repair_translation_escape_artifacts(text: str) -> str:
    """Recover escaped line breaks before flattening whitespace."""
    text = decode_literal_whitespace_escapes(text)
    text = re.sub(r'(^|[\s"\'([{])n(?=[A-Z0-9])', r'\1\n', text)
    text = strip_line_number_prefixes(text)
    text = re.sub(r'(?<![A-Za-z])n(?![A-Za-z])', ' ', text)
    return text


def light_clean_common(text: str) -> str:
    text = decode_literal_whitespace_escapes(text)
    text = text.replace("\u00ad", "")
    text = text.replace("\u00ab", '"').replace("\u00bb", '"')
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    text = LAYOUT_MARKER_RE.sub(" ", text)
    text = strip_line_number_prefixes(text)
    text = text.replace("\n", " ")
    text = text.replace("-/", "").replace("/-", "")
    return normalize_space(text)


def light_clean_transliteration(text: str) -> str:
    return light_clean_common(text)


def light_clean_translation(text: str) -> str:
    text = repair_translation_escape_artifacts(text)
    text = light_clean_common(text)
    text = re.sub(r"\s+([,.;:?!])", r"\1", text)
    text = re.sub(r"\s*}\s*$", "", text)
    return normalize_space(text)


def normalize_compare_text(text: str) -> str:
    text = light_clean_common(text).lower()
    text = re.sub(r"[^\w\sÀ-ÿḫḪšŠṣṢṭṬ]", " ", text, flags=re.UNICODE)
    return normalize_space(text)


def safe_similarity(a: str, b: str) -> float:
    a_norm = normalize_compare_text(a)
    b_norm = normalize_compare_text(b)
    if not a_norm and not b_norm:
        return 1.0
    if not a_norm or not b_norm:
        return 0.0
    return float(SequenceMatcher(None, a_norm, b_norm).ratio())


def token_count(text: str) -> int:
    return len(TOKEN_WORD_RE.findall(text or ""))


def first_alpha_token(text: str) -> str:
    tokens = re.findall(r"[A-Za-zÀ-ÿḫḪšŠṣṢṭṬ]+", str(text or ""))
    return tokens[0].lower() if tokens else ""


def last_alpha_token(text: str) -> str:
    tokens = re.findall(r"[A-Za-zÀ-ÿḫḪšŠṣṢṭṬ]+", str(text or ""))
    return tokens[-1].lower() if tokens else ""


def has_terminal_boundary(text: str) -> bool:
    return bool(TERMINAL_BOUNDARY_RE.search(normalize_space(text)))


def dedupe_pair_ratio(pairs: list[dict[str, str]]) -> float:
    if not pairs:
        return 0.0
    seen: set[tuple[str, str]] = set()
    dup = 0
    for pair in pairs:
        key = (
            normalize_compare_text(pair["transliteration"]),
            normalize_compare_text(pair["translation"]),
        )
        if key in seen:
            dup += 1
        seen.add(key)
    return dup / max(len(pairs), 1)


def has_english_in_akkadian(translit_text: str) -> bool:
    ascii_tokens = re.findall(r"[A-Za-z]{3,}", translit_text.lower().replace("-", ""))
    return any(token in ENGLISH_STOPWORDS for token in ascii_tokens)


def collect_suspicious_forms(text: str) -> dict[str, list[str]]:
    text = str(text or "")
    decimal_fractions = sorted(set(SUSPICIOUS_DECIMAL_RE.findall(text)))
    subscript_tokens = sorted(set(SUSPICIOUS_SUBSCRIPT_TOKEN_RE.findall(text)))
    result: dict[str, list[str]] = {}
    if decimal_fractions:
        result["decimal_fractions"] = decimal_fractions
    if subscript_tokens:
        result["subscript_tokens"] = subscript_tokens
    return result


def summarize_suspicious_pairs(pairs: list[dict[str, str]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for idx, pair in enumerate(pairs, start=1):
        translit_hits = collect_suspicious_forms(pair.get("transliteration", ""))
        translation_hits = collect_suspicious_forms(pair.get("translation", ""))
        if translit_hits or translation_hits:
            summary.append({
                "sentence_index": idx,
                "transliteration_hits": translit_hits,
                "translation_hits": translation_hits,
                "transliteration": pair.get("transliteration", ""),
                "translation": pair.get("translation", ""),
            })
    return summary


def truncate_text(text: str, limit: int = 8000) -> str:
    text = str(text or "")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...[truncated {len(text) - limit} chars]"


def write_debug_event(handle: Any, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    handle.flush()


def build_block_debug_event(
    *,
    source_row: dict[str, Any],
    raw_model_output: str,
    validation: ValidationResult | None,
    retries_used: int,
    alignment_status: str,
    reject_reason: str = "",
    attempt_logs: list[ParseAttemptLog] | None = None,
) -> dict[str, Any]:
    pre_force_pairs: list[dict[str, str]] = []
    final_pairs: list[dict[str, str]] = []
    force_source_changed_pairs: list[int] = []
    validation_metrics: dict[str, Any] = {}

    if validation is not None:
        pre_force_pairs = validation.debug.get("pre_force_pairs", [])
        final_pairs = validation.pairs
        force_source_changed_pairs = validation.debug.get("force_source_changed_pairs", [])
        validation_metrics = validation.metrics

    event: dict[str, Any] = {
        "event": "block_trace",
        "oare_id": source_row.get("oare_id", ""),
        "input_row_index": source_row.get("_input_row_index"),
        "input_file": Config.INPUT_FILE,
        "alignment_status": alignment_status,
        "reject_reason": reject_reason,
        "parse_retries_used": retries_used,
        "source_akkadian_field": source_row.get("_akkadian_field", ""),
        "source_english_field": source_row.get("_english_field", ""),
        "source_block_transliteration": truncate_text(
            source_row.get("_source_block_transliteration", "")
        ),
        "source_block_translation": truncate_text(
            source_row.get("_source_block_translation", "")
        ),
        "locally_cleaned_transliteration": truncate_text(
            source_row.get("_locally_cleaned_transliteration", "")
        ),
        "locally_cleaned_translation": truncate_text(
            source_row.get("_locally_cleaned_translation", "")
        ),
        "raw_model_output": truncate_text(raw_model_output),
        "raw_output_hits": collect_suspicious_forms(raw_model_output),
        "source_tl_hits": collect_suspicious_forms(
            source_row.get("_source_block_transliteration", "")
        ),
        "local_tl_hits": collect_suspicious_forms(
            source_row.get("_locally_cleaned_transliteration", "")
        ),
        "pre_force_pairs": pre_force_pairs,
        "final_pairs": final_pairs,
        "pre_force_suspicious_pairs": summarize_suspicious_pairs(pre_force_pairs),
        "final_suspicious_pairs": summarize_suspicious_pairs(final_pairs),
        "force_source_changed_pairs": force_source_changed_pairs,
        "validation_metrics": validation_metrics,
    }
    if attempt_logs is not None:
        event["parse_attempt_logs"] = [
            {
                "attempt": log.attempt,
                "error": log.error,
                "raw_model_output": truncate_text(log.raw_model_output),
            }
            for log in attempt_logs
        ]
    return event


# ═══════════════════════════════════════════════════════════
# v2: Hard constraint helpers
# ═══════════════════════════════════════════════════════════

def count_gaps(text: str) -> int:
    """Count <gap> markers in text."""
    return text.lower().count("<gap>")


def extract_numbers(text: str) -> list[str]:
    """Extract all numbers (including fractional markers) from text."""
    return NUMBER_RE.findall(text)


def extract_diacritics_signature(text: str) -> str:
    """Extract diacritics sequence — fingerprint for detecting silent edits."""
    return "".join(DIACRITICS_RE.findall(text))


def extract_alignment_tokens(text: str) -> list[str]:
    """Tokenize text for content-preservation checks while ignoring punctuation-only edits."""
    return ALIGNMENT_TOKEN_RE.findall(str(text or "").lower())


def token_overlap_stats(source_text: str, candidate_text: str) -> tuple[float, float, int]:
    """
    Return (recall, precision, shared_token_count) over multiset word tokens.
    Recall answers: how much of the source survived?
    Precision answers: how much of the candidate is grounded in the source?
    """
    src_tokens = extract_alignment_tokens(source_text)
    cand_tokens = extract_alignment_tokens(candidate_text)
    if not src_tokens and not cand_tokens:
        return 1.0, 1.0, 0
    if not src_tokens:
        return 1.0, 0.0, 0
    if not cand_tokens:
        return 0.0, 1.0, 0

    src_counter = Counter(src_tokens)
    cand_counter = Counter(cand_tokens)
    shared = sum((src_counter & cand_counter).values())
    recall = shared / max(len(src_tokens), 1)
    precision = shared / max(len(cand_tokens), 1)
    return recall, precision, shared


def token_edit_stats(source_text: str, candidate_text: str) -> dict[str, int]:
    """Count word-level inserts / deletes / replaces between source and candidate."""
    src_tokens = extract_alignment_tokens(source_text)
    cand_tokens = extract_alignment_tokens(candidate_text)
    matcher = SequenceMatcher(None, src_tokens, cand_tokens)
    inserted = 0
    deleted = 0
    replaced = 0

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "insert":
            inserted += j2 - j1
        elif tag == "delete":
            deleted += i2 - i1
        elif tag == "replace":
            deleted += i2 - i1
            inserted += j2 - j1
            replaced += max(i2 - i1, j2 - j1)

    return {
        "source_tokens": len(src_tokens),
        "candidate_tokens": len(cand_tokens),
        "inserted": inserted,
        "deleted": deleted,
        "replaced": replaced,
    }


def has_numeric_gap_context(text: str) -> bool:
    """Detect <gap> markers that appear inside a numeric / quantity expression."""
    return bool(NUMERIC_GAP_CONTEXT_RE.search(str(text or "")))


def validate_pair_boundaries(
    pairs: list[dict[str, str]],
) -> tuple[bool, str, dict[str, Any]]:
    """
    Reject obvious mid-sentence splits that global coverage checks miss.
    """
    debug: dict[str, Any] = {}
    for idx, pair in enumerate(pairs):
        translation = normalize_space(pair.get("translation", ""))
        if not translation:
            return False, f"pair_{idx}_empty_translation", debug
        if PAIR_ESCAPE_ARTIFACT_RE.search(translation):
            debug[f"pair_{idx}_translation"] = translation[:120]
            return False, f"pair_{idx}_escape_artifact", debug

        if idx >= len(pairs) - 1:
            continue

        last_token = last_alpha_token(translation)
        if last_token in HANGING_TRANSLATION_END_TOKENS:
            debug[f"pair_{idx}_last_token"] = last_token
            debug[f"pair_{idx}_tail"] = translation[-120:]
            return False, f"pair_{idx}_hanging_boundary:{last_token}", debug

        next_translation = normalize_space(pairs[idx + 1].get("translation", ""))
        next_first = first_alpha_token(next_translation)
        if (
            next_first in MID_SENTENCE_START_TOKENS
            and not has_terminal_boundary(translation)
        ):
            debug[f"pair_{idx}_tail"] = translation[-120:]
            debug[f"pair_{idx+1}_head"] = next_translation[:120]
            return False, f"pair_{idx}_mid_sentence_split", debug

    return True, "", debug


def validate_hard_constraints(
    source_akk: str,
    source_eng: str,
    pairs: list[dict[str, str]],
) -> tuple[bool, str, dict[str, Any]]:
    """
    Hard constraints checked BEFORE soft thresholds. Cannot be bypassed.

    Returns (passed, reject_reason, debug_metrics).
    """
    joined_translit = " ".join(p["transliteration"] for p in pairs)
    joined_translation = " ".join(p["translation"] for p in pairs)
    debug: dict[str, Any] = {}

    # --- 1. <gap> count must not decrease ---
    src_gaps_akk = count_gaps(source_akk)
    out_gaps_akk = count_gaps(joined_translit)
    src_gaps_eng = count_gaps(source_eng)
    out_gaps_eng = count_gaps(joined_translation)
    debug["src_gaps_akk"] = src_gaps_akk
    debug["out_gaps_akk"] = out_gaps_akk
    debug["src_gaps_eng"] = src_gaps_eng
    debug["out_gaps_eng"] = out_gaps_eng

    if out_gaps_akk < src_gaps_akk:
        return False, f"gap_filled_translit:src={src_gaps_akk},out={out_gaps_akk}", debug
    if out_gaps_eng < src_gaps_eng:
        return False, f"gap_filled_translation:src={src_gaps_eng},out={out_gaps_eng}", debug

    # --- 1b. If Akkadian has a numeric gap, English must preserve an explicit gap too ---
    numeric_gap_akk = has_numeric_gap_context(source_akk)
    debug["numeric_gap_akk"] = int(numeric_gap_akk)
    if numeric_gap_akk and out_gaps_eng == 0:
        return False, "numeric_gap_not_preserved_in_translation", debug

    # --- 2. Numbers must be preserved in translation ---
    src_nums = extract_numbers(source_eng)
    out_nums = extract_numbers(joined_translation)
    if src_nums:
        src_counter = Counter(src_nums)
        out_counter = Counter(out_nums)
        # reject if new numbers appear that weren't in source
        hallucinated = set(out_counter.keys()) - set(src_counter.keys())
        if hallucinated:
            debug["hallucinated_numbers"] = list(hallucinated)
            return False, f"number_hallucinated:{hallucinated}", debug
        # reject if too many numbers disappear
        missing = sum((src_counter - out_counter).values())
        threshold = max(1, int(len(src_nums) * Config.HARD_NUMBER_MAX_LOSS_RATIO))
        if missing > threshold:
            debug["numbers_missing"] = missing
            debug["numbers_total"] = len(src_nums)
            return False, f"numbers_lost:{missing}/{len(src_nums)}", debug

    # --- 2b. Translation must preserve source wording at token level ---
    eng_recall, eng_precision, shared_tokens = token_overlap_stats(
        source_eng, joined_translation
    )
    debug["translation_word_recall"] = round(eng_recall, 4)
    debug["translation_word_precision"] = round(eng_precision, 4)
    debug["translation_word_shared"] = shared_tokens

    edit_stats = token_edit_stats(source_eng, joined_translation)
    debug["translation_word_inserted"] = edit_stats["inserted"]
    debug["translation_word_deleted"] = edit_stats["deleted"]
    debug["translation_word_replaced"] = edit_stats["replaced"]

    src_eng_token_count = max(edit_stats["source_tokens"], 1)
    max_insertions = max(1, int(src_eng_token_count * Config.HARD_TRANSLATION_MAX_INSERT_RATIO))
    max_deletions = max(1, int(src_eng_token_count * Config.HARD_TRANSLATION_MAX_DELETE_RATIO))

    if eng_precision < Config.HARD_TRANSLATION_WORD_PRECISION_MIN:
        return False, (
            f"translation_content_added:precision={eng_precision:.3f}"
        ), debug
    if edit_stats["inserted"] > max_insertions:
        return False, (
            f"translation_words_inserted:{edit_stats['inserted']}/{src_eng_token_count}"
        ), debug
    if edit_stats["deleted"] > max_deletions:
        return False, (
            f"translation_words_deleted:{edit_stats['deleted']}/{src_eng_token_count}"
        ), debug

    # --- 3. Diacritics signature must be preserved in transliteration ---
    src_sig = extract_diacritics_signature(source_akk)
    out_sig = extract_diacritics_signature(joined_translit)
    if src_sig and out_sig:
        sig_sim = SequenceMatcher(None, src_sig, out_sig).ratio()
        debug["diacritics_sim"] = round(sig_sim, 4)
        if sig_sim < Config.HARD_DIACRITICS_MIN_SIM:
            return False, f"diacritics_modified:sim={sig_sim:.3f}", debug

    # --- 4. Transliteration must not grow (LLM must not add content) ---
    src_tokens = len(source_akk.split())
    out_tokens = len(joined_translit.split())
    debug["translit_src_tokens"] = src_tokens
    debug["translit_out_tokens"] = out_tokens
    if src_tokens > 0 and out_tokens > src_tokens * Config.HARD_TRANSLIT_MAX_GROWTH + 2:
        return False, f"translit_content_added:src={src_tokens},out={out_tokens}", debug

    # --- 5. Each pair's transliteration words should come from source ---
    norm_src_words = set(normalize_compare_text(source_akk).split())
    for i, pair in enumerate(pairs):
        norm_pair_text = normalize_compare_text(pair["transliteration"])
        if len(norm_pair_text) < 10:
            continue
        pair_words = set(norm_pair_text.split())
        if not pair_words:
            continue
        overlap = len(pair_words & norm_src_words) / len(pair_words)
        if overlap < Config.HARD_TRANSLIT_WORD_OVERLAP_MIN:
            debug[f"pair_{i}_word_overlap"] = round(overlap, 3)
            return False, f"translit_pair_{i}_not_in_source:overlap={overlap:.2f}", debug

    return True, "", debug


# ═══════════════════════════════════════════════════════════
# v2: Force source transliteration (post-processing)
# ═══════════════════════════════════════════════════════════

def force_source_transliteration(
    pairs: list[dict[str, str]],
    source_akk: str,
) -> list[dict[str, str]]:
    """
    Replace each pair's transliteration with the best-matching substring
    from the original source. Trust LLM for split-point locations only.
    """
    source_words = source_akk.split()
    if not source_words:
        return pairs

    result = []
    src_cursor = 0

    for pair in pairs:
        pair_words = pair["transliteration"].split()
        pair_len = len(pair_words)
        if pair_len == 0:
            result.append(pair)
            continue

        # Search for best-matching window around current cursor
        search_start = max(0, src_cursor - 5)
        search_end = min(len(source_words), src_cursor + pair_len + 15)

        best_start = src_cursor
        best_end = min(src_cursor + pair_len, len(source_words))
        best_score = -1.0

        for start in range(search_start, search_end):
            # Try windows of similar length (±30%)
            for end in range(
                max(start + 1, start + int(pair_len * 0.7)),
                min(len(source_words) + 1, start + int(pair_len * 1.3) + 1),
            ):
                candidate = " ".join(source_words[start:end])
                score = SequenceMatcher(
                    None, candidate.lower(), pair["transliteration"].lower()
                ).ratio()
                if score > best_score:
                    best_score = score
                    best_start = start
                    best_end = end

        original_segment = " ".join(source_words[best_start:best_end])
        result.append({
            "transliteration": original_segment,
            "translation": pair["translation"],
        })
        src_cursor = best_end

    return result


# ═══════════════════════════════════════════════════════════
# vLLM runtime setup
# ═══════════════════════════════════════════════════════════

def configure_vllm_runtime() -> None:
    rotary_spec = None
    if find_spec("flash_attn") is not None:
        try:
            rotary_spec = find_spec("flash_attn.ops.triton.rotary")
        except ModuleNotFoundError:
            rotary_spec = None
    if find_spec("flash_attn") is not None and rotary_spec is None:
        shim_dir = Path(__file__).resolve().parent / "_vllm_compat_shims"
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [e for e in current_pythonpath.split(os.pathsep) if e]
        if str(shim_dir) not in pythonpath_entries:
            pythonpath_entries.insert(0, str(shim_dir))
            os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
            print(
                f"⚠️ flash_attn.ops.triton.rotary missing; "
                f"using vLLM rotary compatibility shim via {shim_dir}."
            )
    if Config.VLLM_ATTENTION_BACKEND == "FLASHINFER":
        # FlashInfer does not depend on the external flash_attn package path.
        return
    try:
        from flash_attn.ops.triton.rotary import apply_rotary  # noqa: F401
        return
    except Exception as exc:
        shim_dir = Path(tempfile.gettempdir()) / "qwen_clean_vllm_shim"
        shim_dir.mkdir(parents=True, exist_ok=True)
        sitecustomize_path = shim_dir / "sitecustomize.py"
        sitecustomize_path.write_text(
            (
                "import importlib.util\n"
                "_original_find_spec = importlib.util.find_spec\n\n"
                "def _patched_find_spec(name, package=None):\n"
                "    if name == 'flash_attn' or name.startswith('flash_attn.'):\n"
                "        return None\n"
                "    return _original_find_spec(name, package)\n\n"
                "importlib.util.find_spec = _patched_find_spec\n"
            ),
            encoding="utf-8",
        )
        current_pythonpath = os.environ.get("PYTHONPATH", "")
        pythonpath_entries = [e for e in current_pythonpath.split(os.pathsep) if e]
        if str(shim_dir) not in pythonpath_entries:
            pythonpath_entries.insert(0, str(shim_dir))
            os.environ["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
        print(
            f"⚠️ Incompatible flash_attn detected ({exc.__class__.__name__}: {exc}). "
            f"Masking external flash_attn for spawned vLLM workers via {sitecustomize_path}."
        )


def import_vllm_runtime():
    """Import vLLM lazily so runtime shims and clearer diagnostics can apply first."""
    torch_version = None
    vllm_version = None
    try:
        torch_version = importlib_metadata.version("torch")
    except importlib_metadata.PackageNotFoundError:
        pass
    try:
        vllm_version = importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        pass

    if vllm_version and torch_version:
        try:
            vllm_dist = importlib_metadata.distribution("vllm")
            torch_req = None
            for req in (vllm_dist.requires or []):
                if req.startswith("torch=="):
                    torch_req = req.split("==", 1)[1]
                    break
            if torch_req and torch_req != torch_version:
                print(
                    "⚠️ vLLM / torch version mismatch detected before import: "
                    f"vllm {vllm_version} requires torch=={torch_req}, "
                    f"but current torch is {torch_version}."
                )
        except Exception:
            pass

    try:
        from vllm import LLM, SamplingParams
        return LLM, SamplingParams
    except Exception as exc:
        detail = (
            f"Failed to import vLLM runtime: {exc}\n"
            f"Detected torch={torch_version or 'unknown'}, vllm={vllm_version or 'unknown'}.\n"
            "If you see an undefined C++ symbol from vllm/_C.abi3.so, "
            "torch and vllm are ABI-incompatible and must be reinstalled as a matching set."
        )
        raise RuntimeError(detail) from exc


# ═══════════════════════════════════════════════════════════
# I/O helpers
# ═══════════════════════════════════════════════════════════

def detect_input_format(path: Path) -> str:
    if Config.INPUT_FORMAT != "auto":
        return Config.INPUT_FORMAT
    return "csv" if path.suffix.lower() == ".csv" else "jsonl"


def looks_like_csv_header(row: list[str]) -> bool:
    normalized = {normalize_space(cell).lower() for cell in row if normalize_space(cell)}
    if not normalized:
        return False
    translit_keys = {"raw_akkadian", "transliteration", "akkadian", "source_transliteration"}
    translation_keys = {"raw_english", "translation", "english", "source_translation"}
    return bool(normalized & translit_keys) and bool(normalized & translation_keys)


def iter_input_rows(path: Path) -> list[dict[str, Any]]:
    fmt = detect_input_format(path)
    rows: list[dict[str, Any]] = []
    if fmt == "csv":
        with path.open("r", encoding="utf-8", newline="") as fh:
            raw_rows = list(csv.reader(fh))
        if not raw_rows:
            return rows
        if looks_like_csv_header(raw_rows[0]):
            header = raw_rows[0]
            data_rows = raw_rows[1:]
        else:
            header = list(Config.CSV_FALLBACK_FIELDNAMES[: len(raw_rows[0])])
            if len(raw_rows[0]) > len(header):
                header.extend(f"extra_col_{idx}" for idx in range(len(header), len(raw_rows[0])))
            data_rows = raw_rows
            print(f"⚠️ CSV appears headerless; using fallback columns: {header}")
        for raw_row in data_rows:
            padded_row = list(raw_row[: len(header)])
            if len(padded_row) < len(header):
                padded_row.extend([""] * (len(header) - len(padded_row)))
            rows.append(dict(zip(header, padded_row, strict=False)))
        return rows

    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at line {line_no}: {exc}") from exc
    return rows


def find_first_text_field(
    row: dict[str, Any], candidates: list[str]
) -> tuple[str | None, str]:
    for field in candidates:
        if field in row:
            value = normalize_space(row.get(field, ""))
            if value:
                return field, value
    return None, ""


def write_csv_record(writer: csv.DictWriter, record: dict[str, Any]) -> None:
    writer.writerow({fn: record.get(fn, "") for fn in Config.OUTPUT_FIELDNAMES})


def resolve_output_dialect(row: dict[str, Any]) -> str:
    return normalize_space(row.get("dialect", "")) or Config.DEFAULT_DIALECT


def flush_outputs(*handles: Any) -> None:
    for handle in handles:
        handle.flush()


# ═══════════════════════════════════════════════════════════
# Prompt builder (v2 — simplified, no cleaned_block)
# ═══════════════════════════════════════════════════════════

def load_chat_tokenizer(model_path: str):
    try:
        return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    except Exception as tok_exc:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError(
                f"Failed to load tokenizer via AutoTokenizer ({tok_exc}) and "
                "AutoProcessor has no tokenizer attribute."
            ) from tok_exc
        print(
            f"⚠️ AutoTokenizer load failed ({tok_exc.__class__.__name__}: {tok_exc}). "
            "Falling back to AutoProcessor(...).tokenizer."
        )
        return tokenizer


def build_prompt(tokenizer: Any, akkadian: str, english: str, need_align: bool = True) -> str:
    system_prompt = SYSTEM_PROMPT_LONG if need_align else SYSTEM_PROMPT_SHORT
    if need_align:
        user_content = (
            f"Split into aligned sentence pairs.\n\n"
            f"Akkadian:\n{akkadian}\n\n"
            f"English:\n{english}"
        )
    else:
        user_content = (
            f"Evaluate whether to keep or reject this block.\n\n"
            f"Akkadian:\n{akkadian}\n\n"
            f"English:\n{english}"
        )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": user_content,
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


# ═══════════════════════════════════════════════════════════
# JSON parsing
# ═══════════════════════════════════════════════════════════

def _extract_json_blob(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    try:
        return repair_json(cleaned, return_objects=True)
    except Exception:
        pass
    object_match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if object_match:
        object_text = object_match.group(0)
        try:
            return json.loads(object_text)
        except Exception:
            return repair_json(object_text, return_objects=True)
    array_match = re.search(r"\[\s*\{.*\}\s*\]", cleaned, flags=re.DOTALL)
    if array_match:
        array_text = array_match.group(0)
        try:
            return json.loads(array_text)
        except Exception:
            return repair_json(array_text, return_objects=True)
    raise ValueError("No valid JSON object found in model output")


def parse_model_payload(text: str, source_akk: str, source_eng: str) -> dict[str, Any]:
    parsed = _extract_json_blob(text)

    # Bare array → treat as keep
    if isinstance(parsed, list):
        return {
            "decision": "keep",
            "reject_reason": "",
            "sentence_pairs": parsed,
        }

    if not isinstance(parsed, dict):
        raise ValueError("Model output JSON is neither an object nor an array")

    # Normalize key names
    if "sentence_pairs" not in parsed:
        for key in ("pairs", "alignments", "results", "data"):
            if isinstance(parsed.get(key), list):
                parsed["sentence_pairs"] = parsed[key]
                break

    parsed.setdefault("decision", "keep")
    parsed.setdefault("reject_reason", "")
    parsed.setdefault("sentence_pairs", [])
    return parsed


def parse_and_validate_model_output(
    raw_model_output: str,
    source_row: dict[str, Any],
) -> ValidationResult:
    payload = parse_model_payload(
        raw_model_output,
        source_row["_locally_cleaned_transliteration"],
        source_row["_locally_cleaned_translation"],
    )
    need_align = source_row.get("_need_align", True)
    return validate_payload(
        payload,
        source_row["_locally_cleaned_transliteration"],
        source_row["_locally_cleaned_translation"],
        need_align,
    )


def parse_with_retries(
    llm: LLM,
    prompt: str,
    sampling_params: SamplingParams,
    source_row: dict[str, Any],
    initial_raw_model_output: str,
) -> tuple[str, ValidationResult, int, list[ParseAttemptLog]]:
    attempt_logs: list[ParseAttemptLog] = []
    current_output = initial_raw_model_output
    max_attempts = Config.JSON_PARSE_RETRY_COUNT + 1

    for attempt in range(1, max_attempts + 1):
        try:
            validation = parse_and_validate_model_output(current_output, source_row)
            attempt_logs.append(ParseAttemptLog(attempt=attempt, raw_model_output=current_output))
            return current_output, validation, attempt - 1, attempt_logs
        except Exception as exc:
            attempt_logs.append(
                ParseAttemptLog(
                    attempt=attempt,
                    raw_model_output=current_output,
                    error=f"parse_error:{exc}",
                )
            )
            if attempt == max_attempts:
                raise ParseRetryExhaustedError(exc, attempt_logs) from exc

            try:
                retry_outputs = llm.generate([prompt], sampling_params)
                current_output = retry_outputs[0].outputs[0].text.strip()
            except Exception as retry_exc:
                attempt_logs.append(
                    ParseAttemptLog(
                        attempt=attempt + 1,
                        raw_model_output="",
                        error=f"retry_generate_error:{retry_exc}",
                    )
                )
                raise ParseRetryExhaustedError(retry_exc, attempt_logs) from retry_exc

    raise ParseRetryExhaustedError(
        ValueError("JSON parse retries exhausted"),
        attempt_logs,
    )


# ═══════════════════════════════════════════════════════════
# Validation (v2 — hardened)
# ═══════════════════════════════════════════════════════════

def validate_payload(
    payload: dict[str, Any],
    source_akk: str,
    source_eng: str,
    need_align: bool = True,
) -> ValidationResult:
    """
    v2 validation pipeline:
      1. Parse decision
      2. For short texts (need_align=False): only check decision
      3. For long texts (need_align=True): extract + clean pairs
      4. Hard constraints (gap, numbers, diacritics, substring)
      5. Soft constraints (similarity, coverage, length ratio, dedup)
      6. Optional: force_source_transliteration
    """
    decision = str(payload.get("decision", "keep")).strip().lower()
    reject_reason = normalize_space(payload.get("reject_reason", ""))

    # v2: use locally-cleaned source as ground truth, NOT LLM's cleaned_block
    cleaned_translit = light_clean_transliteration(source_akk)
    cleaned_translation = light_clean_translation(source_eng)

    if decision == "reject":
        return ValidationResult(
            keep=False,
            reject_reason=reject_reason or "model_reject",
            cleaned_transliteration=cleaned_translit,
            cleaned_translation=cleaned_translation,
            pairs=[],
            metrics={
                "decision": "reject",
                "model_reject_reason": reject_reason,
            },
        )

    # 短文本不需要对齐，只返回 keep
    if not need_align:
        return ValidationResult(
            keep=True,
            reject_reason="",
            cleaned_transliteration=cleaned_translit,
            cleaned_translation=cleaned_translation,
            pairs=[{
                "transliteration": cleaned_translit,
                "translation": cleaned_translation,
            }],
            metrics={
                "decision": "keep",
                "need_align": False,
                "source_length": len(source_akk),
            },
        )

    # --- Extract pairs ---
    raw_pairs = payload.get("sentence_pairs")
    if not isinstance(raw_pairs, list):
        return ValidationResult(
            keep=False, reject_reason="invalid_sentence_pairs",
            cleaned_transliteration=cleaned_translit,
            cleaned_translation=cleaned_translation,
            pairs=[], metrics={"decision": "invalid"},
        )

    pairs: list[dict[str, str]] = []
    for item in raw_pairs:
        if not isinstance(item, dict):
            continue
        pair_translit = light_clean_transliteration(
            item.get("transliteration") or item.get("akkadian") or item.get("source") or ""
        )
        pair_translation = light_clean_translation(
            item.get("translation") or item.get("english") or item.get("target") or ""
        )
        if pair_translit and pair_translation:
            pairs.append({
                "transliteration": pair_translit,
                "translation": pair_translation,
            })

    if not pairs:
        return ValidationResult(
            keep=False, reject_reason="empty_pairs",
            cleaned_transliteration=cleaned_translit,
            cleaned_translation=cleaned_translation,
            pairs=[], metrics={"decision": "invalid"},
        )

    if len(pairs) > Config.MAX_SENTENCE_PAIRS:
        return ValidationResult(
            keep=False, reject_reason="too_many_pairs",
            cleaned_transliteration=cleaned_translit,
            cleaned_translation=cleaned_translation,
            pairs=[], metrics={"pair_count": len(pairs)},
        )

    # ===== HARD CONSTRAINTS (v2) =====
    hard_pass, hard_reason, hard_debug = validate_hard_constraints(
        source_akk, source_eng, pairs
    )
    if not hard_pass:
        return ValidationResult(
            keep=False,
            reject_reason=hard_reason,
            cleaned_transliteration=cleaned_translit,
            cleaned_translation=cleaned_translation,
            pairs=[],
            metrics={"decision": "hard_reject", "hard_reason": hard_reason, **hard_debug},
        )

    boundary_pass, boundary_reason, boundary_debug = validate_pair_boundaries(pairs)
    if not boundary_pass:
        return ValidationResult(
            keep=False,
            reject_reason=boundary_reason,
            cleaned_transliteration=cleaned_translit,
            cleaned_translation=cleaned_translation,
            pairs=[],
            metrics={
                "decision": "boundary_reject",
                "boundary_reason": boundary_reason,
                **boundary_debug,
            },
        )

    # ===== SOFT CONSTRAINTS =====
    joined_translit = light_clean_transliteration(
        " ".join(pair["transliteration"] for pair in pairs)
    )
    joined_translation = light_clean_translation(
        " ".join(pair["translation"] for pair in pairs)
    )

    # v2: compare against SOURCE directly
    source_sim_translit = safe_similarity(cleaned_translit, joined_translit)
    source_sim_translation = safe_similarity(cleaned_translation, joined_translation)
    coverage_sim_translit, coverage_precision_translit, _ = token_overlap_stats(
        cleaned_translit, joined_translit
    )
    coverage_sim_translation, coverage_precision_translation, _ = token_overlap_stats(
        cleaned_translation, joined_translation
    )

    translit_len_ratio = len(normalize_compare_text(joined_translit)) / max(
        1, len(normalize_compare_text(cleaned_translit))
    )
    translation_len_ratio = len(normalize_compare_text(joined_translation)) / max(
        1, len(normalize_compare_text(cleaned_translation))
    )

    duplicate_ratio = dedupe_pair_ratio(pairs)
    short_translation_pairs = sum(
        1 for pair in pairs
        if token_count(pair["translation"]) < Config.MIN_TRANSLATION_WORDS_PER_PAIR
    )
    short_translation_ratio = short_translation_pairs / max(len(pairs), 1)
    english_in_akkadian = has_english_in_akkadian(joined_translit)

    metrics: dict[str, float | int | str] = {
        "decision": "keep",
        "pair_count": len(pairs),
        "source_sim_translit": round(source_sim_translit, 4),
        "source_sim_translation": round(source_sim_translation, 4),
        "coverage_sim_translit": round(coverage_sim_translit, 4),
        "coverage_precision_translit": round(coverage_precision_translit, 4),
        "coverage_sim_translation": round(coverage_sim_translation, 4),
        "coverage_precision_translation": round(coverage_precision_translation, 4),
        "translit_len_ratio": round(translit_len_ratio, 4),
        "translation_len_ratio": round(translation_len_ratio, 4),
        "duplicate_ratio": round(duplicate_ratio, 4),
        "short_translation_ratio": round(short_translation_ratio, 4),
        "english_in_akkadian": int(english_in_akkadian),
    }

    if source_sim_translit < Config.MIN_SOURCE_SIM_TRANSLIT:
        return ValidationResult(
            False, "overcleaned_translit",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if source_sim_translation < Config.MIN_SOURCE_SIM_TRANSLATION:
        return ValidationResult(
            False, "overcleaned_translation",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if coverage_sim_translit < Config.MIN_COVERAGE_SIM_TRANSLIT:
        return ValidationResult(
            False, "dropped_translit_content",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if coverage_sim_translation < Config.MIN_COVERAGE_SIM_TRANSLATION:
        return ValidationResult(
            False, "dropped_translation_content",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if english_in_akkadian:
        return ValidationResult(
            False, "language_confusion_english_in_akkadian",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if not (Config.MIN_LENGTH_RATIO <= translit_len_ratio <= Config.MAX_LENGTH_RATIO):
        return ValidationResult(
            False, "bad_translit_length_ratio",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if not (Config.MIN_LENGTH_RATIO <= translation_len_ratio <= Config.MAX_LENGTH_RATIO):
        return ValidationResult(
            False, "bad_translation_length_ratio",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if duplicate_ratio > Config.MAX_DUPLICATE_PAIR_RATIO:
        return ValidationResult(
            False, "too_many_duplicate_pairs",
            cleaned_translit, cleaned_translation, [], metrics,
        )
    if short_translation_ratio > Config.MAX_SHORT_TRANSLATION_PAIR_RATIO:
        return ValidationResult(
            False, "too_many_short_pairs",
            cleaned_translit, cleaned_translation, [], metrics,
        )

    pre_force_pairs = [dict(pair) for pair in pairs]

    # ===== POST-PROCESSING (v2): force source transliteration =====
    if Config.FORCE_SOURCE_TRANSLITERATION:
        pairs = force_source_transliteration(pairs, source_akk)

    force_changed_pairs = [
        idx
        for idx, (before, after) in enumerate(zip(pre_force_pairs, pairs, strict=False), start=1)
        if before.get("transliteration", "") != after.get("transliteration", "")
    ]

    return ValidationResult(
        True,
        "",
        cleaned_translit,
        cleaned_translation,
        pairs,
        metrics,
        debug={
            "pre_force_pairs": pre_force_pairs,
            "force_source_changed_pairs": force_changed_pairs,
        },
    )


# ═══════════════════════════════════════════════════════════
# Record builders
# ═══════════════════════════════════════════════════════════

def build_keep_record(
    source_row: dict[str, Any],
    validation: ValidationResult,
    pair: dict[str, str],
    sentence_index: int,
    raw_model_output: str,
) -> dict[str, Any]:
    record = {k: v for k, v in source_row.items() if not k.startswith("_")}
    record["source_block_transliteration"] = source_row["_source_block_transliteration"]
    record["source_block_translation"] = source_row["_source_block_translation"]
    record["locally_cleaned_transliteration"] = source_row["_locally_cleaned_transliteration"]
    record["locally_cleaned_translation"] = source_row["_locally_cleaned_translation"]
    record["model_cleaned_transliteration"] = validation.cleaned_transliteration
    record["model_cleaned_translation"] = validation.cleaned_translation
    record["source_akkadian_field"] = source_row["_akkadian_field"]
    record["source_english_field"] = source_row["_english_field"]
    record["alignment_status"] = "kept"
    record["sentence_index"] = sentence_index
    record["sentence_count"] = len(validation.pairs)
    record["alignment_model_path"] = Config.MODEL_PATH
    record["alignment_validation"] = validation.metrics
    record["model_raw_output"] = raw_model_output
    record["transliteration"] = pair["transliteration"]
    record["translation"] = pair["translation"]
    record["dialect"] = resolve_output_dialect(source_row)
    return record


def build_reject_record(
    source_row: dict[str, Any],
    reject_reason: str,
    raw_model_output: str,
    validation: ValidationResult | None = None,
) -> dict[str, Any]:
    record = {k: v for k, v in source_row.items() if not k.startswith("_")}
    record["source_block_transliteration"] = source_row.get("_source_block_transliteration", "")
    record["source_block_translation"] = source_row.get("_source_block_translation", "")
    record["locally_cleaned_transliteration"] = source_row.get("_locally_cleaned_transliteration", "")
    record["locally_cleaned_translation"] = source_row.get("_locally_cleaned_translation", "")
    record["source_akkadian_field"] = source_row.get("_akkadian_field", "")
    record["source_english_field"] = source_row.get("_english_field", "")
    record["alignment_status"] = "rejected"
    record["reject_reason"] = reject_reason
    record["alignment_model_path"] = Config.MODEL_PATH
    record["model_raw_output"] = raw_model_output
    record["dialect"] = resolve_output_dialect(source_row)
    if validation is not None:
        record["model_cleaned_transliteration"] = validation.cleaned_transliteration
        record["model_cleaned_translation"] = validation.cleaned_translation
        record["alignment_validation"] = validation.metrics
    return record


# ═══════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════

def main() -> None:
    input_path = Path(Config.INPUT_FILE)
    output_path = Path(Config.OUTPUT_FILE)
    reject_path = Path(Config.REJECT_FILE)
    error_path = Path(Config.ERROR_LOG)
    debug_path = Path(Config.DEBUG_TRACE_FILE)

    print(f"🚀 Starting Qwen sentence align v2 (hardened)")
    print(f"   Model: {Config.MODEL_PATH}")
    print(f"📄 Input:  {input_path}")
    print(f"📄 Keep:   {output_path}")
    print(f"📄 Reject: {reject_path}")
    print(f"🔒 Hard constraints: gap_check, number_check, diacritics_check, substr_check")
    print(f"🔒 Force source translit: {Config.FORCE_SOURCE_TRANSLITERATION}")
    print(f"⚙️ vLLM attention backend: {Config.VLLM_ATTENTION_BACKEND}")
    print(f"⚙️ vLLM worker mp method: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'auto')}")
    print(f"🔒 Thresholds: translit_sim≥{Config.MIN_SOURCE_SIM_TRANSLIT}, "
          f"translation_sim≥{Config.MIN_SOURCE_SIM_TRANSLATION}")

    if not input_path.exists():
        print(f"❌ Error: Input file not found at {input_path}")
        return

    configure_vllm_runtime()
    LLM, SamplingParams = import_vllm_runtime()

    print("📦 Loading tokenizer...")
    tokenizer = load_chat_tokenizer(Config.MODEL_PATH)

    print("📄 Reading input rows...")
    input_rows = iter_input_rows(input_path)
    if Config.MAX_INPUT_RECORDS > 0:
        input_rows = input_rows[: Config.MAX_INPUT_RECORDS]

    prompts: list[str] = []
    valid_rows: list[dict[str, Any]] = []
    skipped_missing_fields = 0

    for row_idx, row in enumerate(input_rows):
        akk_field, source_akk = find_first_text_field(row, Config.AKKADIAN_FIELD_CANDIDATES)
        eng_field, source_eng = find_first_text_field(row, Config.ENGLISH_FIELD_CANDIDATES)
        if not source_akk or not source_eng:
            skipped_missing_fields += 1
            continue

        local_akk = light_clean_transliteration(source_akk)
        local_eng = light_clean_translation(source_eng)
        if not local_akk or not local_eng:
            skipped_missing_fields += 1
            continue

        enriched_row = dict(row)
        enriched_row["_input_row_index"] = row_idx
        enriched_row["_akkadian_field"] = akk_field
        enriched_row["_english_field"] = eng_field
        enriched_row["_source_block_transliteration"] = source_akk
        enriched_row["_source_block_translation"] = source_eng
        enriched_row["_locally_cleaned_transliteration"] = local_akk
        enriched_row["_locally_cleaned_translation"] = local_eng

        # 根据阿卡德语原文长度决定是否需要对齐
        akk_len = len(local_akk)
        need_align = akk_len >= Config.MIN_ALIGN_LENGTH
        enriched_row["_need_align"] = need_align
        enriched_row["_akk_length"] = akk_len

        prompts.append(build_prompt(tokenizer, local_akk, local_eng, need_align))
        valid_rows.append(enriched_row)

    print(f"✅ Prepared {len(prompts)} prompts.")
    # 统计需要对齐和不需要对齐的数量
    need_align_count = sum(1 for r in valid_rows if r.get("_need_align", True))
    no_align_count = len(valid_rows) - need_align_count
    print(f"   ├─ 需要对齐 (≥{Config.MIN_ALIGN_LENGTH}字符): {need_align_count}")
    print(f"   └─ 只需判断丢弃 (<{Config.MIN_ALIGN_LENGTH}字符): {no_align_count}")
    if skipped_missing_fields:
        print(f"⚠️ Skipped {skipped_missing_fields} rows with missing fields.")
    if not prompts:
        print("❌ No valid prompts to process.")
        return

    print(
        f"🧠 Initializing vLLM on {Config.TENSOR_PARALLEL_SIZE} GPUs "
        f"(max_num_seqs={Config.MAX_NUM_SEQS})..."
    )
    llm = LLM(
        model=Config.MODEL_PATH,
        tensor_parallel_size=Config.TENSOR_PARALLEL_SIZE,
        max_model_len=Config.MAX_MODEL_LEN,
        max_num_seqs=Config.MAX_NUM_SEQS,
        gpu_memory_utilization=Config.GPU_MEMORY_UTILIZATION,
        trust_remote_code=True,
        attention_config={"backend": Config.VLLM_ATTENTION_BACKEND},
    )
    sampling_params = SamplingParams(
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        max_tokens=Config.MAX_NEW_TOKENS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    reject_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text("", encoding="utf-8")
    debug_path.write_text("", encoding="utf-8")

    block_success_count = 0
    kept_pair_count = 0
    rejected_block_count = 0
    parse_error_count = 0
    parse_retry_attempt_count = 0
    parse_retry_recovered_count = 0
    hard_reject_count = 0
    soft_reject_count = 0
    dedup_skipped_count = 0
    debug_event_count = 0
    seen_keep_pairs: set[tuple[str, str]] = set()

    print(f"⚡ Starting batched generation (batch_size={Config.REQUEST_BATCH_SIZE})...")
    start_time = time.time()

    with (
        output_path.open("w", encoding="utf-8", newline="") as keep_f,
        reject_path.open("w", encoding="utf-8", newline="") as reject_f,
        error_path.open("a", encoding="utf-8") as err_f,
        debug_path.open("a", encoding="utf-8") as debug_f,
    ):
        keep_writer = csv.DictWriter(keep_f, fieldnames=Config.OUTPUT_FIELDNAMES)
        reject_writer = csv.DictWriter(reject_f, fieldnames=Config.OUTPUT_FIELDNAMES)
        keep_writer.writeheader()
        reject_writer.writeheader()
        flush_outputs(keep_f, reject_f)
        write_debug_event(
            debug_f,
            {
                "event": "run_config",
                "input_file": str(input_path),
                "output_file": str(output_path),
                "reject_file": str(reject_path),
                "error_log": str(error_path),
                "model_path": Config.MODEL_PATH,
                "force_source_transliteration": Config.FORCE_SOURCE_TRANSLITERATION,
                "request_batch_size": Config.REQUEST_BATCH_SIZE,
                "max_input_records": Config.MAX_INPUT_RECORDS,
            },
        )
        debug_event_count += 1

        with tqdm(total=len(prompts), desc="Align v2") as pbar:
            for batch_start in range(0, len(prompts), Config.REQUEST_BATCH_SIZE):
                batch_end = min(batch_start + Config.REQUEST_BATCH_SIZE, len(prompts))
                batch_prompts = prompts[batch_start:batch_end]
                batch_rows = valid_rows[batch_start:batch_end]
                outputs = llm.generate(batch_prompts, sampling_params)

                for idx, output in enumerate(outputs):
                    source_row = batch_rows[idx]
                    raw_model_output = output.outputs[0].text.strip()
                    # --- Parse ---
                    try:
                        (
                            raw_model_output,
                            validation,
                            retries_used,
                            attempt_logs,
                        ) = parse_with_retries(
                            llm=llm,
                            prompt=batch_prompts[idx],
                            sampling_params=sampling_params,
                            source_row=source_row,
                            initial_raw_model_output=raw_model_output,
                        )
                        parse_retry_attempt_count += retries_used
                        if retries_used > 0:
                            parse_retry_recovered_count += 1
                    except ParseRetryExhaustedError as exc:
                        parse_error_count += 1
                        rejected_block_count += 1
                        parse_retry_attempt_count += max(0, len(exc.attempt_logs) - 1)
                        write_csv_record(
                            reject_writer,
                            build_reject_record(
                                source_row,
                                reject_reason=(
                                    f"parse_error_after_{len(exc.attempt_logs)}_attempts:"
                                    f"{exc.last_error}"
                                ),
                                raw_model_output=(
                                    exc.attempt_logs[-1].raw_model_output
                                    if exc.attempt_logs
                                    else raw_model_output
                                ),
                            ),
                        )
                        error_lines = [
                            "--- ERROR ---",
                            f"Type: {exc.last_error}",
                            f"Record: {source_row.get('oare_id') or source_row.get('_input_row_index')}",
                            f"Akkadian field: {source_row.get('_akkadian_field')}",
                            f"English field: {source_row.get('_english_field')}",
                            f"Parse attempts: {len(exc.attempt_logs)}",
                        ]
                        for attempt_log in exc.attempt_logs:
                            error_lines.extend([
                                f"Attempt {attempt_log.attempt}:",
                                (
                                    attempt_log.error
                                    if attempt_log.error
                                    else "ok"
                                ),
                                "Model Output:",
                                attempt_log.raw_model_output,
                                "",
                            ])
                        err_f.write("\n".join(error_lines))
                        flush_outputs(reject_f, err_f)
                        write_debug_event(
                            debug_f,
                            build_block_debug_event(
                                source_row=source_row,
                                raw_model_output=(
                                    exc.attempt_logs[-1].raw_model_output
                                    if exc.attempt_logs
                                    else raw_model_output
                                ),
                                validation=None,
                                retries_used=max(0, len(exc.attempt_logs) - 1),
                                alignment_status="parse_error",
                                reject_reason=str(exc.last_error),
                                attempt_logs=exc.attempt_logs,
                            ),
                        )
                        debug_event_count += 1
                        pbar.update(1)
                        continue

                    # --- Rejected ---
                    if not validation.keep:
                        rejected_block_count += 1
                        reason = validation.reject_reason or "validation_reject"
                        if "hard_reject" in str(validation.metrics.get("decision", "")):
                            hard_reject_count += 1
                        else:
                            soft_reject_count += 1
                        write_csv_record(
                            reject_writer,
                            build_reject_record(
                                source_row,
                                reject_reason=reason,
                                raw_model_output=raw_model_output,
                                validation=validation,
                            ),
                        )
                        flush_outputs(reject_f)
                        write_debug_event(
                            debug_f,
                            build_block_debug_event(
                                source_row=source_row,
                                raw_model_output=raw_model_output,
                                validation=validation,
                                retries_used=retries_used,
                                alignment_status="rejected",
                                reject_reason=reason,
                                attempt_logs=attempt_logs,
                            ),
                        )
                        debug_event_count += 1
                        pbar.update(1)
                        continue

                    # --- Kept ---
                    write_debug_event(
                        debug_f,
                        build_block_debug_event(
                            source_row=source_row,
                            raw_model_output=raw_model_output,
                            validation=validation,
                            retries_used=retries_used,
                            alignment_status="kept",
                            attempt_logs=attempt_logs,
                        ),
                    )
                    debug_event_count += 1
                    wrote_pair = False
                    for sentence_index, pair in enumerate(validation.pairs, start=1):
                        dedup_key = (pair["transliteration"], pair["translation"])
                        if Config.DEDUP_KEEP_ROWS and dedup_key in seen_keep_pairs:
                            dedup_skipped_count += 1
                            continue
                        seen_keep_pairs.add(dedup_key)
                        write_csv_record(
                            keep_writer,
                            build_keep_record(
                                source_row, validation, pair,
                                sentence_index, raw_model_output,
                            ),
                        )
                        kept_pair_count += 1
                        wrote_pair = True
                    if wrote_pair:
                        block_success_count += 1
                    flush_outputs(keep_f)
                    pbar.update(1)

                flush_outputs(keep_f, reject_f, err_f)

    elapsed = max(time.time() - start_time, 1e-9)

    print()
    print("=" * 60)
    print(f"✅ Completed in {elapsed:.2f}s ({len(prompts) / elapsed:.2f} blocks/sec)")
    print(f"📦 Blocks kept:        {block_success_count}")
    print(f"📄 Sentence pairs:     {kept_pair_count}")
    print(f"🗑️  Blocks rejected:    {rejected_block_count}")
    print(f"   ├─ hard constraint: {hard_reject_count}")
    print(f"   ├─ soft threshold:  {soft_reject_count}")
    print(f"   └─ parse errors:    {parse_error_count}")
    print(f"🔁 Parse retries used: {parse_retry_attempt_count}")
    print(f"✅ Retry recoveries:    {parse_retry_recovered_count}")
    print(f"♻️  Dedup skipped:      {dedup_skipped_count}")
    print(f"🪵 Debug events:       {debug_event_count}")
    print(f"📝 Keep:   {output_path}")
    print(f"📝 Reject: {reject_path}")
    print(f"🪵 Errors: {error_path}")
    print(f"🪵 Debug:  {debug_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
