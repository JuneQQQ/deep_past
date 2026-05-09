#!/usr/bin/env python3
"""
Build a stricter ArchiBab clean dataset on top of prepare_data.py.

Pipeline:
1. Call prepare_data.py --input <archibab csv> to get a baseline clean CSV.
2. Re-apply stronger ArchiBab-specific cleanup:
   - strip edition line-number prefixes from transliteration
   - remove scholarly meta notes in translation
   - remove French/German-style editorial residue and bibliography refs
3. Force a minimal 5-column output:
   oare_id, transliteration, translation, data_source, dialect
"""

from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import pandas as pd

try:
    from openai import AsyncOpenAI, APIError, RateLimitError
    _OPENAI_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    AsyncOpenAI = None
    APIError = Exception
    RateLimitError = Exception
    _OPENAI_AVAILABLE = False


SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DEFAULT_INPUT = ROOT_DIR / "data" / "archibab" / "archibab_akk_en.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "archibab" / "archibab_akk_en_clean.csv"
PREPARE_SCRIPT = SCRIPT_DIR / "prepare_data.py"
ARCHIBAB_ENABLE_LLM_REPAIR = os.environ.get("ARCHIBAB_ENABLE_LLM_REPAIR", "0") == "1"
ARCHIBAB_LLM_API_KEY = os.environ.get("ARCHIBAB_LLM_API_KEY", os.environ.get("OPENAI_API_KEY", "EMPTY"))
ARCHIBAB_LLM_BASE_URL = os.environ.get("ARCHIBAB_LLM_BASE_URL", os.environ.get("OPENAI_BASE_URL", ""))
ARCHIBAB_LLM_MODEL = os.environ.get("ARCHIBAB_LLM_MODEL", "gemini-3.1-pro-preview")
ARCHIBAB_LLM_CONCURRENCY = max(1, int(os.environ.get("ARCHIBAB_LLM_CONCURRENCY", "4")))
ARCHIBAB_LLM_MAX_RETRIES = max(1, int(os.environ.get("ARCHIBAB_LLM_MAX_RETRIES", "3")))

sys.path.insert(0, str(SCRIPT_DIR))
import prepare_data as base  # noqa: E402


_LINE_PREFIX_RE = re.compile(r"^\s*(\d{1,3})(?:['’])?(?=\s+(?:[\[(<{]|[^\W\d_]))")
_TRANSLATION_HEADER_RE = re.compile(r"(?i)\bTraduction(?:\s+[A-Za-z.]+)?\s*:")
_FOREIGN_META_RE = re.compile(
    r"Traduction(?:\s+[A-Za-z.]+)?\s*:|"
    r"T\s*moins\s*et\s*date|"
    r"T[ée]moins?\s+et\s+date|"
    r"Traduction de l['’]éditeur\s*:|"
    r"Traduction IA\s*:|"
    r"Traduction R\.\s*de Boer\s*:|"
    r"Traduction A\.\s*George\s*:",
    re.I,
)
_BIBLIO_REF_RE = re.compile(
    r"\b(?:AbB|ARM|YOS|TCL|UET|RA|NABU)\s*[A-Z]*\s*\d+(?:[ ,./-]+\d+)*\b",
    re.I,
)
_LINE_META_RE = re.compile(
    r"\(?\b(?:lines?|line)\s*\d+\s*[-–—]\s*\d+[^)\n.;:!?]*"
    r"(?:fragmentary|too broken to be translated|too damaged to be translated|too fragmentary)\)?",
    re.I,
)
_PRESERVATION_META_RE = re.compile(
    r"\(?\b(?:most of\s+)?(?:obv(?:erse)?|rev(?:erse)?|text|tablet|beginning|remainder|rest)\b"
    r"[^)\n.;:!?]*"
    r"(?:broken|lost|missing|fragmentary|not preserved|not inscribed|not deciphered|"
    r"too broken to be translated|too damaged to be translated)\)?",
    re.I,
)
_BROKEN_LINES_RE = re.compile(
    r"\(?\b(?:gap of\s+)?(?:one|two|three|four|five|\d+)\s+(?:largely\s+)?broken lines?\b[^)]*\)?",
    re.I,
)
_RAW_EDITORIAL_JUNK_RE = re.compile(
    r"Traduction(?:\s+[A-Za-z.]+)?\s*:|"
    r"T\s*moins\s*et\s*date|"
    r"T[ée]moins?\s+et\s+date|"
    r"\blines?\s*\d+\s*[-–—]\s*\d+|"
    r"\b(?:AbB|ARM|YOS|TCL|UET|RA|NABU)\s*[A-Z]*\s*\d+(?:[ ,./-]+\d+)*\b",
    re.I,
)
_SCHOLARLY_PLACEHOLDER_RE = re.compile(
    r"\b(?:[fm]?PN[0-9₀-₉]*|DN[0-9₀-₉]*|GN[0-9₀-₉]*|RN[0-9₀-₉]*|PB)\b"
)
_DANGLING_EDITORIAL_RE = re.compile(
    r"\(\s*(?:remainder|rest|beginning|obverse|reverse|fragmentary|too broken|too damaged)[^)]*\)",
    re.I,
)
_BROKEN_TAIL_RE = re.compile(r"<gap>\s*\.?$|[,;:-]\s*$")
_WITNESS_DATE_RE = re.compile(r"\b(?:witness(?:es)?|date|seal impressions?)\b", re.I)
_BROKEN_NOTE_RE = re.compile(
    r"\(\s*(?:broken|substantial break|seal impressions?|witness(?:es)?|date)\s*\)",
    re.I,
)
_META_ONLY_SENTENCE_RE = re.compile(
    r"^\s*(?:<gap>\s*)?(?:(?:one|two|three|four|five|\d+)\s+)?"
    r"(?:witness(?:es)?|date|seal impressions?)\b",
    re.I,
)
_SINGLE_LINE_REF_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_LINE_RANGE_RE = re.compile(r"^\s*(\d+)\s*-\s*(\d+)\s*$")
_ROYAL_CONTENT_RE = re.compile(
    r"\b(?:RIME|royal inscription|foundation deposit|statue of himself|offering prayer|"
    r"dedicated for his life|inscribed for his life|brought into the esagila|"
    r"raised the top of the emeslam|the king established)\b",
    re.I,
)
_CULTIC_CONTENT_RE = re.compile(
    r"\b(?:daily offerings?|food offerings?|offering of wool|morning and evening offering|"
    r"flour offering|aširtum offering|sirqum offering|funerary offering|"
    r"new moon celebration|full moon celebration|sacrifice(?: for)?|"
    r"food allocations? of the temple|food allocation of the temple|"
    r"property of \(god\)|property of god|large old temple|temple complex|"
    r"harp\(-songs\)|banquet\b|"
    r"festival(?:s)?|eššešum|full moon celebration|nešakkum(?:-officials?)?|"
    r"pašīšum(?:-priests?)?|chief lamentation priest|lamentation singers?|"
    r"priest(?:s)?\b|temple of|the temple of|guennakkum)\b",
    re.I,
)
_LITERARY_CONTENT_RE = re.compile(
    r"\b(?:hymn(?:s)?|incantation(?:s)?|proverb(?:s)?|literary|dialogue|balag|ershema)\b",
    re.I,
)
_DIVINATION_CONTENT_RE = re.compile(
    r"\b(?:omen(?:s)?|oracle(?:s)?|divination|extispic(?:y)?|omen-consultation|"
    r"purification rite)\b",
    re.I,
)
_SCHOOL_CONTENT_RE = re.compile(
    r"\b(?:school(?: exercise)?|copybook|student(?: exercise)?|lexical list|"
    r"school tablet|exercise tablet|model letter)\b",
    re.I,
)
_EDITORIAL_COMMENTARY_RE = re.compile(
    r"(?:Maybe a fragment of a model trial|This text is heavily damaged|"
    r"Only the beginning and the date can be read|less cursive handwriting|"
    r"Translation\s*\(composite\))",
    re.I,
)
_LEXICAL_GLOSS_CONTENT_RE = re.compile(
    r"\((?:cloth|clothes|garment|garments|wood|stone|metal)\)\s+[A-Za-zĀāĒēĪīŌōŪūṢṣŠšṬṭḪḫ'\-]+",
    re.I,
)
_FRENCH_STRONG_START_RE = re.compile(
    r"^\s*(?:Dis à|Ainsi\s*\(parle\)|Que\s+Šamaš|En outre\b|Pourquoi as-tu|"
    r"Je souhaite|Lib[eé]re?\b|Toi qui\b)|"
    r"correspondant\s+au|champs?\s+de\s+leur\s+choix|"
    r"Ann[ée]e\s+o[ùu]\s+le\s+roi",
    re.I,
)
_FRENCH_TOKEN_RE = re.compile(
    r"\b(?:ainsi|parle|pourquoi|arriv[ée]\w*|donn[ée]?\w*|serviteurs?|"
    r"p[ée]ch[ée]s?|contestes?|venue|toi|moi|libre|champ\b|cris-tu|fasse|"
    r"correspondant|choix|ann[ée]e|entrer|porte|temple)\b",
    re.I,
)
_ENGLISH_RECOVERY_RE = re.compile(
    r"\b(?:Say to|Speak to|To [A-ZĀĒĪŌŪṢŠṬḪ][^:]{0,120}\bspeak,\s*thus)\b"
)
_LIT_GLOSS_RE = re.compile(r"\s*\blit\.\s*[^;.]*(?=[;.]|,\s*[A-Z]|$)", re.I)
_OVERINFER_YEAR_RE = re.compile(
    r'("?\s*The year:\s*[^"(]{1,220}?)\s*\([^"]{20,800}'
    r'(?:source of abundance|uptake point|on the bank\?|royally established|'
    r'broad fields in the heart of the country)[^"]*\)("\.?)',
    re.I,
)
_OVERINFER_YEAR_TAIL_RE = re.compile(
    r'("?\s*The year:\s*[^"]*?\bking\b),[^"]*("\.?)',
    re.I,
)
_METRIC_LITER_GLOSS_RE = re.compile(
    r"\(\s*(?:<gap>\s*)?[0-9<>,.? ]+\s+lit(?:er|re)s?\s*\)",
    re.I,
)
_FRENCH_REPAIR_PROMPT = """You are cleaning a noisy Assyriological translation.
Return exactly one clean English translation and nothing else.

Rules:
1. Remove French or other non-English segments completely.
2. Remove scholarly residue such as "lit.", "i.e.", editorial comments, and over-inferred background.
3. Preserve names, numbers, units, dates, and <gap>.
4. Do not invent information not supported by the transliteration.
5. If the remaining translation is unusable, output exactly <DROP>.

Transliteration:
{transliteration}

Noisy translation:
{translation}

Clean English:"""


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _is_probably_french_translation(text: str) -> bool:
    text = str(text or "").strip()
    if not text:
        return False
    if _FRENCH_STRONG_START_RE.search(text):
        return True
    return len(_FRENCH_TOKEN_RE.findall(text)) >= 4


def _recover_english_tail_from_mixed_translation(text: str) -> str:
    text = str(text or "")
    match = _ENGLISH_RECOVERY_RE.search(text)
    if not match:
        return ""
    candidate = text[match.start():].strip(" \t\n\r-;:,")
    if not candidate:
        return ""
    return candidate


def _base_archibab_id(oare_id: str) -> str:
    return re.sub(r"^[0-9a-f]{8}-", "", str(oare_id or "").strip())


def _line_ref_sort_key(line_ref: str) -> tuple[int, int, str]:
    value = str(line_ref or "").strip()
    match = _LINE_RANGE_RE.match(value)
    if match:
        return (int(match.group(1)), int(match.group(2)), value)
    return (10**9, 10**9, value)


def strip_archibab_line_prefixes(text: str) -> str:
    """Aggressively remove edition line numbers from raw ArchiBab transliterations."""
    if not isinstance(text, str):
        return ""

    text = base.decode_literal_whitespace_escapes(text)
    if "\n" not in text and "\r" not in text:
        return text

    lines = text.splitlines()
    numbered = []
    nonempty_lines = 0
    for line in lines:
        if line.strip():
            nonempty_lines += 1
        match = _LINE_PREFIX_RE.match(line)
        numbered.append((line, match))

    nums = [int(match.group(1)) for _, match in numbered if match]
    if len(nums) < 3 or nonempty_lines == 0:
        return text

    forward_steps = [
        nums[idx] - nums[idx - 1]
        for idx in range(1, len(nums))
        if 0 < (nums[idx] - nums[idx - 1]) <= 4
    ]
    unmatched_nonempty = sum(1 for line, match in numbered if line.strip() and match is None)
    looks_like_line_numbers = (
        len(forward_steps) >= max(2, len(nums) - 2)
        and (
            unmatched_nonempty > 0
            or len(nums) >= max(4, nonempty_lines - 1)
        )
        and len(nums) >= max(3, nonempty_lines // 3)
    )
    if not looks_like_line_numbers:
        return text

    cleaned_lines = []
    for line, match in numbered:
        if match:
            line = line[match.end():].lstrip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


def aggressive_clean_translation(text: str) -> str:
    """Remove scholarly residue while preserving meaningful parenthetical content."""
    if not isinstance(text, str):
        return ""

    text = base.clean_archibab_translation_residue(text)
    text = base.decode_literal_whitespace_escapes(text)
    text = text.replace("”", '"').replace("“", '"').replace("’", "'").replace("‘", "'")
    text = text.replace("–", "-").replace("—", "-")

    if _TRANSLATION_HEADER_RE.search(text):
        segments = [
            seg.strip(" \n\r\t:;")
            for seg in _TRANSLATION_HEADER_RE.split(text)
            if seg and seg.strip(" \n\r\t:;")
        ]
        if segments:
            text = max(segments, key=lambda s: len(re.sub(r"\W+", "", s)))

    text = _SCHOLARLY_PLACEHOLDER_RE.sub(" <gap> ", text)
    text = _FOREIGN_META_RE.sub(" <gap> ", text)
    text = _BIBLIO_REF_RE.sub(" ", text)
    text = _LINE_META_RE.sub(" <gap> ", text)
    text = re.sub(
        r"\b(?:lines?|line)\s*\d+\s*[-–—]\s*\d+\s*(?:are|is)?\s*"
        r"(?:too broken|too damaged|too fragmentary|fragmentary)\s*(?:to be translated)?",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = _PRESERVATION_META_RE.sub(" <gap> ", text)
    text = _BROKEN_LINES_RE.sub(" <gap> ", text)
    text = re.sub(
        r"(?:(?:^)|(?:[.;]\s*))(?:(?:one|two|three|four|five|\d+)\s+)?"
        r"witness(?:es)?;?\s*date(?:\s*\([^)]*\))?(?:[.!?]|$)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?:(?:^)|(?:[.;]\s*))witness(?:es)?\s*:[^.!?]*(?:[.!?]|$)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?:(?:^)|(?:[.;]\s*))date\s*\([^)]*\)(?:[.!?]|$)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?:(?:^)|(?:[.;]\s*))\(\s*seal impressions?\s*\)(?:[.!?]|$)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"(?:(?:^)|(?:[.;]\s*))\(\s*substantial break\s*\)(?:[.!?]|$)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\(\s*(?:rev|obv)\s*\)", " <gap> ", text, flags=re.I)
    text = _BROKEN_NOTE_RE.sub(" <gap> ", text)
    text = re.sub(r"\bsubstantial break\b", " <gap> ", text, flags=re.I)
    text = re.sub(r"(?<![A-Za-z])(?:rev|obv)\.(?![A-Za-z])", " ", text, flags=re.I)
    text = re.sub(r"\(\s*\)", " ", text)
    text = text.replace("(remainder of too )", " <gap> ")
    text = text.replace("(remainder too )", " <gap> ")
    text = text.replace("T moins et date", " <gap> ")
    text = text.replace("Témoins et date", " <gap> ")
    text = re.sub(r"\s*<gap>\s*(?:[.,;:])\s*", " <gap> ", text)
    text = base.normalize_gaps(text)
    text = _normalize_space(text)
    return text.strip(" \t\n\r-;:,")


def repair_archibab_final_translation(text: str, raw_text: str = "") -> str:
    """Final targeted fixes for residue that survives the main cleanup pipeline."""
    if not isinstance(text, str):
        return ""

    raw_text = str(raw_text or "")
    text = text.replace('""', '"')
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    text = _LIT_GLOSS_RE.sub("", text)
    text = _METRIC_LITER_GLOSS_RE.sub("", text)
    text = re.sub(r"(?<=[A-Za-z])\(\s*[!?]\s*\)(?=[0-9A-Za-z⅓⅔¼½¾])", " ", text)
    text = re.sub(r"\(\s*gap\s*\)", " <gap> ", text, flags=re.I)
    text = re.sub(r"\(\s*[!?]\s*\)", "", text)

    if (
        raw_text.lstrip().startswith("Via ")
        or raw_text.lstrip().startswith("[Via]")
    ) and text.startswith("ia "):
        text = f"Via {text[3:]}"

    text = re.sub(r"\bia(?=\s+[A-ZĀĒĪŌŪṢŠṬḪḬ])", "via", text)

    mixed_english = _recover_english_tail_from_mixed_translation(text)
    if mixed_english and _is_probably_french_translation(text[:max(1, text.find(mixed_english))]):
        text = mixed_english

    text = re.sub(r"\b(Speak)\s+\d+\s+(to)\b", r"\1 \2", text, flags=re.I)
    text = re.sub(r":\s*\d+\s+(Thus\s+says)\b", r": \1", text, flags=re.I)
    text = re.sub(
        r"\b((?:Give|Bring|Send|Take|Tell|Let|Deliver|Dispatch))\s+"
        r"((?:them|him|her|me|us|you))\s+\d+\s+(the\b)",
        r"\1 \2 \3",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"([.!?])\s*\d+\s+"
        r"(?=(?:Be aware|Thus says|Speak to|Do not|May\b|Let\b|Take action\b|"
        r"Now\b|I have\b|You\b))",
        r"\1 ",
        text,
        flags=re.I,
    )

    text = re.sub(
        r"\[\s*DC\s*:?\s*corriger\s+en\s*:?\s*[^\]]+\]",
        " ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\[\s*part too broken for translation\s*\]",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\[\s*rest of [^\]]*(?:broken|lost|fragmentary|damaged|translation)[^\]]*\]",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\(\s*broken off;\s*uncertain,\s*whether inscribed or not\s*\)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\(\s*(?:rev\.?\s*)?(?:remainder\s+)?(?:too\s+)?(?:fragmentary|broken|damaged)\s+for translation\s*\)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\(\s*(?:too|broken|fragmentary)\s+for translation\s*\)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\(\s*(?:broken\s+)?for translation\s*\)", " <gap> ", text, flags=re.I)
    text = re.sub(
        r"\(\s*(?:rev\.?\s*)?(?:remainder\s+)?too\s+badly\s+preserved\s+to\s+translate\s*\)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(
        r"\(\s*(?:rev\.?\s*)?(?:remainder\s+)?too\s+fragmentary\s+for translation\s*\)",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = re.sub(r"\b(?:remainder\s+too\s+)?fragmentary\s+for translation\b", " <gap> ", text, flags=re.I)
    text = re.sub(r"\bpart too broken for translation\b", " <gap> ", text, flags=re.I)
    text = re.sub(r"\(\s*text\s+too\s+for translation\s*\)", " <gap> ", text, flags=re.I)
    text = re.sub(r"\btext\s+too\s+for translation\b", " <gap> ", text, flags=re.I)
    text = re.sub(r"<gap>\s+for translation\b", " <gap> ", text, flags=re.I)
    text = re.sub(r"\btoo\s+badly\s+preserved\s+to\s+translate\b", " <gap> ", text, flags=re.I)
    text = re.sub(r"\b(?:remainder\s+)?too\s+fragmentary\s+for translation\b", " <gap> ", text, flags=re.I)
    text = re.sub(r"\bDC\s*:?\s*corriger\s+en\s*:?\s*[^.;]+\.?", " ", text, flags=re.I)
    text = re.sub(r"\brest of\s*$", " ", text, flags=re.I)
    text = re.sub(r"\(\s*i\.\s*e\.?,?[^)]*\)", " ", text, flags=re.I)
    text = re.sub(r"\(\s*(?:therefore|consequent|animals?)\s*\)", " ", text, flags=re.I)
    text = re.sub(
        r"([A-Za-zĀāĒēĪīŌōŪūṢṣŠšṬṭḪḫḬḭ0-9-]+)\?(?=[:.,;)]|$)",
        r"\1",
        text,
    )
    text = re.sub(r",\s*-\s*(?=\b(?:he|she|they|it)\b)", ", ", text, flags=re.I)
    text = _OVERINFER_YEAR_RE.sub(r"\1\2", text)
    text = _OVERINFER_YEAR_TAIL_RE.sub(r"\1\2", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = text.replace("the overseer is]", "the overseer is ")
    text = text.replace("(the remainder is too )", " <gap> ")
    text = text.replace("] ", " ").replace(" [", " ")
    # 补全类括号剥壳（保留文字）：(and) → and, (for) → for
    text = re.sub(r'\(([^)]{1,80})\)', r'\1', text)
    # 双句点修复
    text = re.sub(r'\.\.+', '.', text)
    text = base.normalize_gaps(text)
    return _normalize_space(text)


def aggressive_clean_transliteration(text: str) -> str:
    """Drop scholarly placeholder tokens that are not Akkadian lexical material."""
    if not isinstance(text, str):
        return ""
    text = re.sub(
        r"(?<![A-Za-zÀ-ÿ₀-₉])x(?:[\s-]+x)*(?![A-Za-zÀ-ÿ₀-₉])",
        " <gap> ",
        text,
        flags=re.I,
    )
    text = _SCHOLARLY_PLACEHOLDER_RE.sub(" <gap> ", text)
    text = base.normalize_gaps(text)
    # Repair residual spaced hyphen shells left after x/x-x placeholders collapse to <gap>.
    text = re.sub(r"(?<=-)\s*-\s*(?:-\s*)*(?=\{)", "<gap>-", text)
    text = re.sub(r"(?<=-)\s*-\s*(?:-\s*)*(?=\w)", "<gap> ", text)
    text = re.sub(r"(?<=\S)\s+(?:-\s*){2,}-(?=\w)", " <gap>-", text)
    text = re.sub(r"(?<=\S)\s+-\s*(?:-\s*)+(?=\{)", " <gap>-", text)
    text = re.sub(r"(?<=\S)\s+-\s*(?:-\s*)+(?=\w)", " <gap> ", text)
    text = re.sub(r"<gap>\s+-\s+(?=\S)", "<gap> ", text)
    return _normalize_space(text)


def build_linefix_map(raw_df: pd.DataFrame) -> dict[str, str]:
    """Precompute stricter transliteration values from raw data."""
    result: dict[str, str] = {}
    if raw_df.empty:
        return result

    for _, row in raw_df.iterrows():
        oare_id = str(row.get("oare_id") or "").strip()
        raw_tl = str(row.get("transliteration") or "")
        if not oare_id or not raw_tl:
            continue
        stripped = strip_archibab_line_prefixes(raw_tl)
        if stripped != raw_tl:
            result[oare_id] = base.preprocess_transliteration(stripped)
    return result


def build_exclusion_reason_map(raw_df: pd.DataFrame) -> dict[str, str]:
    """Drop whole base texts for clearly undesired content domains."""
    result: dict[str, str] = {}
    if raw_df.empty or "_base_id" not in raw_df.columns:
        return result

    pattern_map = {
        "royal": _ROYAL_CONTENT_RE,
        "cultic": _CULTIC_CONTENT_RE,
        "literary": _LITERARY_CONTENT_RE,
        "divination": _DIVINATION_CONTENT_RE,
        "school": _SCHOOL_CONTENT_RE,
        "editorial_commentary": _EDITORIAL_COMMENTARY_RE,
        "lexical_gloss": _LEXICAL_GLOSS_CONTENT_RE,
    }

    for base_id, group in raw_df.groupby("_base_id", dropna=False, sort=False):
        blob_parts = []
        for col in ("titre", "archibab_ref", "archive", "translation", "transliteration"):
            if col not in group.columns:
                continue
            values = [str(v).strip() for v in group[col].tolist() if str(v).strip()]
            if values:
                blob_parts.append(" ".join(values))
        blob = " || ".join(blob_parts)
        if not blob:
            continue

        reasons = [name for name, pattern in pattern_map.items() if pattern.search(blob)]
        if reasons:
            result[str(base_id or "")] = ",".join(reasons)
    return result


async def _call_archibab_llm(prompt: str) -> str | None:
    if not (_OPENAI_AVAILABLE and ARCHIBAB_LLM_BASE_URL):
        return None
    client = AsyncOpenAI(api_key=ARCHIBAB_LLM_API_KEY, base_url=ARCHIBAB_LLM_BASE_URL)
    for attempt in range(1, ARCHIBAB_LLM_MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=ARCHIBAB_LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=512,
            )
            text = (response.choices[0].message.content or "").strip()
            text = re.sub(r"^```[a-zA-Z]*\n", "", text)
            text = re.sub(r"\n```$", "", text).strip()
            return text or None
        except (APIError, RateLimitError):
            if attempt >= ARCHIBAB_LLM_MAX_RETRIES:
                return None
            await asyncio.sleep(2 ** attempt)
        except Exception:
            return None
    return None


async def _repair_french_rows_async(rows: list[dict[str, str]]) -> dict[str, str]:
    semaphore = asyncio.Semaphore(ARCHIBAB_LLM_CONCURRENCY)
    repaired: dict[str, str] = {}

    async def _work(row: dict[str, str]) -> None:
        async with semaphore:
            prompt = _FRENCH_REPAIR_PROMPT.format(
                transliteration=row.get("transliteration", ""),
                translation=row.get("translation", ""),
            )
            result = await _call_archibab_llm(prompt)
            if result:
                repaired[str(row.get("oare_id") or "")] = result

    await asyncio.gather(*[_work(row) for row in rows])
    return repaired


def repair_or_drop_french_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    candidate_rows = []
    drop_ids: set[str] = set()
    recovered = 0
    for _, row in df.iterrows():
        oare_id = str(row.get("oare_id") or "")
        text = str(row.get("translation") or "")
        recovered_tail = _recover_english_tail_from_mixed_translation(text)
        if recovered_tail and _is_probably_french_translation(text[:max(1, text.find(recovered_tail))]):
            df.loc[df["oare_id"] == oare_id, "translation"] = recovered_tail
            recovered += 1
            continue
        if _is_probably_french_translation(text):
            candidate_rows.append(
                {
                    "oare_id": oare_id,
                    "transliteration": str(row.get("transliteration") or ""),
                    "translation": text,
                }
            )

    repaired: dict[str, str] = {}
    if candidate_rows and ARCHIBAB_ENABLE_LLM_REPAIR and _OPENAI_AVAILABLE and ARCHIBAB_LLM_BASE_URL:
        repaired = asyncio.run(_repair_french_rows_async(candidate_rows))

    for row in candidate_rows:
        oare_id = row["oare_id"]
        fixed = repaired.get(oare_id, "").strip()
        if fixed and fixed != "<DROP>":
            df.loc[df["oare_id"] == oare_id, "translation"] = fixed
        else:
            drop_ids.add(oare_id)

    if recovered:
        print(f"   🔧 ArchiBab 法英混杂修复: {recovered} 条")
    if drop_ids:
        print(f"   🪓 ArchiBab 法语污染删除: {len(drop_ids)} 条")
        df = df.loc[~df["oare_id"].astype(str).isin(drop_ids)].copy()
    return df


def drop_undesired_content_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "_drop_reason" not in df.columns:
        return df

    df = df.copy()
    drop_mask = df["_drop_reason"].fillna("").astype(str) != ""
    if drop_mask.any():
        counts = Counter()
        for value in df.loc[drop_mask, "_drop_reason"].astype(str):
            for reason in value.split(","):
                if reason:
                    counts[reason] += 1
        detail = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"   🪓 ArchiBab 体裁过滤: {int(drop_mask.sum())} 条 ({detail})")
        df = df.loc[~drop_mask].copy()
    return df


def drop_remaining_editorial_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    df = df.copy()
    tr = df["translation"].astype(str)
    tr_tokens = tr.str.split().str.len().clip(lower=1)
    gap_density = tr.str.count(r"<gap>") / tr_tokens
    strong_meta = tr.str.contains(_RAW_EDITORIAL_JUNK_RE, na=False)
    drop_mask = strong_meta & ((gap_density >= 0.18) | (tr_tokens <= 18))
    if drop_mask.any():
        print(f"   🪓 ArchiBab 终态元注释过滤: {int(drop_mask.sum())} 条")
        df = df.loc[~drop_mask].copy()
    return df


def drop_fragment_heavy_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Aggressively drop fragment-heavy rows that still look worse than sentence-aligned data."""
    if df.empty:
        return df

    df = df.copy()
    tr = df["translation"].astype(str)
    tl = df["transliteration"].astype(str)
    align_kind = (
        df["_align_type"].astype(str).str.lower()
        if "_align_type" in df.columns
        else pd.Series([""] * len(df), index=df.index)
    )
    grouped_doc = align_kind.eq("doc_grouped")
    tr_tokens = tr.str.split().str.len().clip(lower=1)
    tl_tokens = tl.str.split().str.len().clip(lower=1)
    tr_gap = tr.str.count(r"<gap>")
    tl_gap = tl.str.count(r"<gap>")
    gap_density = (tr_gap + tl_gap) / (tr_tokens + tl_tokens).clip(lower=1)

    dangling_editorial = tr.str.contains(_DANGLING_EDITORIAL_RE, na=False)
    quote_imbalance = (tr.str.count('"') % 2) != 0
    paren_imbalance = tr.str.count(r"\(") != tr.str.count(r"\)")
    broken_tail = tr.str.contains(_BROKEN_TAIL_RE, na=False)
    long_fragment = ((tr_tokens >= 60) & (tr_gap >= 1) & ~grouped_doc) | (
        (tr_tokens >= 100) & (tr_gap >= 3) & grouped_doc
    )
    many_gaps = ((tr_gap >= 3) & ~grouped_doc) | ((tr_gap >= 5) & grouped_doc)
    dense_fragment = gap_density >= 0.15

    drop_mask = (
        many_gaps
        | dense_fragment
        | dangling_editorial
        | quote_imbalance
        | paren_imbalance
        | long_fragment
        | (broken_tail & (tr_gap >= 1))
    )
    if drop_mask.any():
        print(
            "   🪓 ArchiBab 激进碎片过滤:"
            f" {int(drop_mask.sum())} 条"
            f" (many_gaps={int(many_gaps.sum())},"
            f" dense_fragment={int(dense_fragment.sum())},"
            f" dangling_editorial={int(dangling_editorial.sum())},"
            f" quote_imbalance={int(quote_imbalance.sum())},"
            f" paren_imbalance={int(paren_imbalance.sum())},"
            f" long_fragment={int(long_fragment.sum())})"
        )
        df = df.loc[~drop_mask].copy()
    return df


def drop_sentence_meta_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sentence-level fragments that are mostly witness/date/seal metadata."""
    if df.empty or "_align_type" not in df.columns:
        return df

    df = df.copy()
    tr = df["translation"].astype(str)
    tr_tokens = tr.str.split().str.len().clip(lower=1)
    align_type = df["_align_type"].astype(str).str.lower()
    meta_sentence = (
        align_type.eq("sentence")
        & (
            tr.str.contains(_WITNESS_DATE_RE, na=False)
            | tr.str.contains(_BROKEN_NOTE_RE, na=False)
            | tr.str.contains(_META_ONLY_SENTENCE_RE, na=False)
        )
        & ((tr_tokens <= 18) | tr.str.contains(r"<gap>", na=False))
    )
    if meta_sentence.any():
        print(f"   🪓 ArchiBab sentence 元数据碎片过滤: {int(meta_sentence.sum())} 条")
        df = df.loc[~meta_sentence].copy()
    return df


def drop_placeholder_contaminated_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows whose raw source still used scholarly placeholders like PN2/fPN/PB."""
    if df.empty or "_raw_has_placeholder" not in df.columns:
        return df

    df = df.copy()
    drop_mask = df["_raw_has_placeholder"].fillna(False).astype(bool)
    if drop_mask.any():
        print(f"   🪓 ArchiBab 学术占位符过滤: {int(drop_mask.sum())} 条")
        df = df.loc[~drop_mask].copy()
    return df


def drop_sentence_alignment_risk_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sentence-aligned rows whose English is much longer than the Akkadian slice."""
    if df.empty or "_align_type" not in df.columns:
        return df

    df = df.copy()
    align_type = df["_align_type"].astype(str).str.lower()
    tl_tokens = df["transliteration"].astype(str).str.split().str.len().clip(lower=1)
    tr_tokens = df["translation"].astype(str).str.split().str.len().clip(lower=1)
    ratio = tl_tokens / tr_tokens

    severe_short_vs_long = (tl_tokens <= 16) & (tr_tokens >= 25)
    severe_ratio = (ratio < 0.30) & (tr_tokens >= 18)
    ultra_short_sentence = (tl_tokens <= 2) & (tr_tokens >= 10)
    ultra_ratio = (ratio < 0.12) & (tr_tokens >= 10)
    drop_mask = align_type.eq("sentence") & (
        severe_short_vs_long | severe_ratio | ultra_short_sentence | ultra_ratio
    )
    if drop_mask.any():
        print(
            "   🪓 ArchiBab sentence 对齐风险过滤:"
            f" {int(drop_mask.sum())} 条"
            f" (short_vs_long={int(severe_short_vs_long.sum())},"
            f" severe_ratio={int(severe_ratio.sum())},"
            f" ultra_short={int(ultra_short_sentence.sum())},"
            f" ultra_ratio={int(ultra_ratio.sum())})"
        )
        df = df.loc[~drop_mask].copy()
    return df


def drop_single_line_sentence_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop sentence rows that correspond to a single source line but a full English clause."""
    if df.empty or "_align_type" not in df.columns or "_line_ref" not in df.columns:
        return df

    df = df.copy()
    align_type = df["_align_type"].astype(str).str.lower()
    line_ref = df["_line_ref"].astype(str)
    tl_tokens = df["transliteration"].astype(str).str.split().str.len().clip(lower=1)
    tr_tokens = df["translation"].astype(str).str.split().str.len().clip(lower=1)

    single_line = line_ref.apply(
        lambda s: bool((m := _SINGLE_LINE_REF_RE.match(s)) and m.group(1) == m.group(2))
    )
    drop_mask = align_type.eq("sentence") & single_line & (
        ((tl_tokens <= 1) & (tr_tokens >= 7))
        | ((tl_tokens <= 2) & (tr_tokens >= 12))
    )
    if drop_mask.any():
        print(f"   🪓 ArchiBab 单行 sentence 过碎过滤: {int(drop_mask.sum())} 条")
        df = df.loc[~drop_mask].copy()
    return df


def merge_sentence_rows_by_base_id(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse sentence-only ArchiBab fragments back into one record per base text."""
    if df.empty or "_align_type" not in df.columns or "_base_id" not in df.columns:
        return df

    df = df.copy()
    sentence_df = df[df["_align_type"].astype(str).str.lower().eq("sentence")].copy()
    other_df = df[~df["_align_type"].astype(str).str.lower().eq("sentence")].copy()
    if sentence_df.empty:
        return df

    merged_rows = []
    passthrough_rows = []
    grouped_count = 0
    grouped_source_rows = 0
    for base_id, group in sentence_df.groupby("_base_id", dropna=False, sort=False):
        if len(group) <= 1:
            passthrough_rows.append(group)
            continue
        group = group.sort_values(
            by="_line_ref",
            key=lambda s: s.map(_line_ref_sort_key),
            kind="stable",
        ).reset_index(drop=True)
        merged = group.iloc[0].copy()
        merged["oare_id"] = str(base_id or group.iloc[0]["oare_id"])
        merged["transliteration"] = _normalize_space(" ".join(group["transliteration"].astype(str)))
        merged["translation"] = _normalize_space(" ".join(group["translation"].astype(str)))
        merged["_align_type"] = "doc_grouped"
        merged["_line_ref"] = "grouped_sentence"
        raw_parts = [
            part for part in group.get("_raw_translation", pd.Series(dtype=str)).astype(str).tolist() if part
        ]
        merged["_raw_translation"] = " ".join(raw_parts)
        merged_rows.append(merged)
        grouped_count += 1
        grouped_source_rows += len(group)

    merged_df = pd.DataFrame(merged_rows)
    if grouped_count:
        print(
            "   🧵 ArchiBab sentence 分组回缝:"
            f" {grouped_source_rows} 条片段 -> {grouped_count} 条文档"
        )
    passthrough_df = pd.concat(passthrough_rows, ignore_index=True, sort=False) if passthrough_rows else pd.DataFrame(columns=df.columns)
    return pd.concat([other_df, passthrough_df, merged_df], ignore_index=True, sort=False)


def run_prepare_stage(input_path: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="archibab_prepare_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        staged_input = tmpdir_path / input_path.name
        shutil.copy2(input_path, staged_input)

        cmd = [sys.executable, str(PREPARE_SCRIPT), "--input", str(staged_input)]
        print(f"▶️ 运行 prepare_data.py: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)

        staged_output = staged_input.with_name(f"{staged_input.stem}_clean.csv")
        if not staged_output.exists():
            raise FileNotFoundError(f"prepare_data.py 未产出中间文件: {staged_output}")

        with tempfile.NamedTemporaryFile(prefix="archibab_stage_", suffix=".csv", delete=False) as fh:
            preserved = Path(fh.name)
        shutil.copy2(staged_output, preserved)
        return preserved


def build_archibab_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    raw_df = pd.read_csv(input_path)
    raw_df["_base_id"] = raw_df["oare_id"].astype(str).map(_base_archibab_id)
    linefix_map = build_linefix_map(raw_df)
    exclusion_reason_map = build_exclusion_reason_map(raw_df)
    raw_meta = raw_df[["oare_id", "align_type", "line_ref", "_base_id"]].copy()
    raw_meta = raw_meta.rename(columns={"align_type": "_align_type", "line_ref": "_line_ref"})
    raw_translation_meta = raw_df[["oare_id", "translation"]].copy()
    raw_translation_meta = raw_translation_meta.rename(columns={"translation": "_raw_translation"})
    raw_placeholder_mask = (
        raw_df["transliteration"].astype(str).str.contains(_SCHOLARLY_PLACEHOLDER_RE, na=False)
        | raw_df["translation"].astype(str).str.contains(_SCHOLARLY_PLACEHOLDER_RE, na=False)
    )
    raw_placeholder_meta = raw_df.loc[:, ["oare_id"]].copy()
    raw_placeholder_meta["_raw_has_placeholder"] = raw_placeholder_mask.astype(bool)
    print(f"   🔧 原始 TL 行号修正候选: {len(linefix_map)} 条")
    print(f"   🔎 原始体裁剔除候选: {len(exclusion_reason_map)} 条基底文本")

    staged_output = run_prepare_stage(input_path)
    try:
        df = pd.read_csv(staged_output)
    finally:
        if staged_output.exists():
            staged_output.unlink()

    df = df.merge(raw_meta, on="oare_id", how="left")
    df = df.merge(raw_translation_meta, on="oare_id", how="left")
    df = df.merge(raw_placeholder_meta, on="oare_id", how="left")
    if exclusion_reason_map:
        exclusion_meta = pd.DataFrame(
            {
                "_base_id": list(exclusion_reason_map.keys()),
                "_drop_reason": list(exclusion_reason_map.values()),
            }
        )
        df = df.merge(exclusion_meta, on="_base_id", how="left")

    if "data_source" not in df.columns:
        df["data_source"] = "official"
    df["data_source"] = df["data_source"].apply(base.normalize_data_source_label)

    if linefix_map:
        mask = df["oare_id"].astype(str).isin(linefix_map)
        if mask.any():
            df.loc[mask, "transliteration"] = df.loc[mask, "oare_id"].astype(str).map(linefix_map)
            print(f"   🔧 覆写 TL 行号修正: {int(mask.sum())} 条")

    df = drop_undesired_content_rows(df)
    df["transliteration"] = df["transliteration"].astype(str).apply(aggressive_clean_transliteration)
    df["translation"] = df["translation"].astype(str).apply(aggressive_clean_translation)
    df = base.filter_archibab_fragments(df)
    df = drop_sentence_meta_rows(df)
    df = drop_placeholder_contaminated_rows(df)
    df = merge_sentence_rows_by_base_id(df)
    df["transliteration"] = df["transliteration"].astype(str).apply(aggressive_clean_transliteration)
    df["translation"] = df["translation"].astype(str).apply(aggressive_clean_translation)
    df = drop_sentence_alignment_risk_rows(df)
    df = drop_single_line_sentence_rows(df)
    df["transliteration"] = df["transliteration"].astype(str).apply(aggressive_clean_transliteration)
    df["translation"] = df["translation"].astype(str).apply(aggressive_clean_translation)
    df = drop_remaining_editorial_rows(df)
    df = drop_fragment_heavy_rows(df)
    df = base.quality_filter(df)
    df["transliteration"] = df["transliteration"].astype(str).apply(aggressive_clean_transliteration)
    df["translation"] = df["translation"].astype(str).apply(aggressive_clean_translation)
    df["translation"] = [
        repair_archibab_final_translation(text, raw_text)
        for text, raw_text in zip(df["translation"].astype(str), df["_raw_translation"].astype(str))
    ]
    df = repair_or_drop_french_rows(df)
    df["dialect"] = "OB"
    if "_base_id" in df.columns:
        df["oare_id"] = df["_base_id"].fillna(df["oare_id"]).astype(str)

    keep = ["oare_id", "transliteration", "translation", "data_source", "dialect"]
    df = df[keep].copy()
    df["oare_id"] = df["oare_id"].astype(str)
    df["transliteration"] = df["transliteration"].astype(str).apply(_normalize_space)
    df["translation"] = df["translation"].astype(str).apply(_normalize_space)
    df["data_source"] = df["data_source"].astype(str).apply(base.normalize_data_source_label)
    df["dialect"] = "OB"
    df = df[(df["oare_id"] != "") & (df["transliteration"] != "") & (df["translation"] != "")]
    df = df.drop_duplicates(subset=["oare_id"], keep="first").reset_index(drop=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a stricter ArchiBab clean dataset.")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Path to archibab_akk_en.csv")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Path to output clean CSV")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    df = build_archibab_data(input_path, output_path)
    print(f"\n✅ ArchiBab clean 数据已保存: {output_path}")
    print(f"   行数: {len(df)}")
    print(f"   列: {', '.join(df.columns.tolist())}")


if __name__ == "__main__":
    main()
