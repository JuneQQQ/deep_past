#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import fitz

# ============================================================
# Inlined dependencies from extract_akt8_direct.py
# ============================================================

# Dataclasses
@dataclass
class PageLine:
    x0: float
    x1: float
    y0: float
    y1: float
    text: str
    width_ratio: float


@dataclass
class AlignedRow:
    page: int
    y0: float
    left: str
    right: str
    raw_left: str
    raw_right: str
    entry_index: int


@dataclass
class CandidateRecord:
    page: int
    row_start_index: int
    row_end_index: int
    raw_akkadian: str
    raw_english: str
    akkadian: str
    english: str
    clean_char_ratio: float
    translit_retention: float
    translation_retention: float


# Regex patterns
EDITORIAL_RE = re.compile(
    r'\b(?:cf\.|see now|improved readings|variant|variants|study of the original|'
    r'missing in|through damage|still visible|erased|erasure|damage|restored|'
    r'comment|comments|bibliography|line[s]?|translation "|translation "|meaning|'
    r'occurs|archive|register|registers)\b',
    re.IGNORECASE
)
ENGLISH_WORD_RE = re.compile(r'[A-Za-z]+')
_BASE_TABLET_MARKER_PATTERN = (
    r'(?:obv\.?|rev\.?|l\.e\.?|u\.e\.?|le\.e\.?|re\.e\.?|r\.e\.?|'
    r'lo\.e\.?|ri\.e\.?)'
)
_LEFT_TABLET_MARKER_PATTERN = _BASE_TABLET_MARKER_PATTERN
_TRANSLATION_TABLET_MARKER_PATTERN = (
    r'(?:' + _BASE_TABLET_MARKER_PATTERN[3:-1] + r'|e\.|r\.)'
)
LEFT_PREFIX_RE = re.compile(
    rf'^(?:(?:\d+\*?\s+)+)?{_LEFT_TABLET_MARKER_PATTERN}\s*',
    re.IGNORECASE
)
LEADING_LINE_NUM_RE = re.compile(r'^(?:\d+\*?\s+)+')
PAGE_HEADER_RE = re.compile(
    r'^(?:I+\.|[IVX]+\.)\s+[A-Z][A-Z \-,\'()]+$|^\d+\s*$|^\s*[IVX]+\.\s',
    re.IGNORECASE
)
ENTRY_HEADER_RE = re.compile(r'^\d+\.\s+Kt\s+', re.IGNORECASE)
SKIP_SECTION_RE = re.compile(
    r'^(Notes?|Comment|Bibliography|Index|Registers?|Concordance|Plates?|Variants?)\b',
    re.IGNORECASE
)
TRANSLIT_TOKEN_RE = re.compile(
    r'(?:\b[0-9]+\b|\b[A-Z]{2,}(?:\.[A-Z]+)*\b|\{[a-z]+\}|'
    r'[A-Za-záàéèíìúùšṣṭḫŠṢṬḪ"!#$%&(\'/]+-[A-Za-záàéèíìúùšṣṭḫŠṢṬḪ0-9"!#$%&(\'/]+)'
)
ENGLISH_QTY_UNIT_RE = re.compile(
    r'\b(?:\d+(?:[¼½¾⅓⅔])?|¼|½|¾|⅓|⅔|one|two|three|four|five|six|seven|eight|'
    r'nine|ten|eleven|twelve|thirteen|fourteen|fifteen|twenty|thirty|forty|'
    r'fifty|sixty)\s+(?:mina|minas|shekel|shekels|talent|talents)\b',
    re.IGNORECASE,
)
ENGLISH_NAME_RE = re.compile(r"\b[A-ZĀĒĪŪŠṢṬḪ][A-Za-zĀāĒēĪīŪūŠšṢṣṬṭḪḫÇçȘșȚț'’-]+\b")
AKKADIAN_COMMODITY_RE = re.compile(
    r'\b(?:ma-na|GÍN|GÚ|AN\.NA|URUDU|KÙ\.BABBAR|KÙ\.B\.?|TÚG(?:\.HI\.A)?|'
    r'ANŠE(?:\.HI\.A)?)\b'
)
SUPERSCRIPT_TRANS = str.maketrans({
    "⁰": "0",
    "¹": "1",
    "²": "2",
    "³": "3",
    "⁴": "4",
    "⁵": "5",
    "⁶": "6",
    "⁷": "7",
    "⁸": "8",
    "⁹": "9",
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
})
UNEXPECTED_OCR_SYMBOL_RE = re.compile(r"[γβµ]")

# Constant sets
STOPWORDS = {
    'the', 'have', 'for', 'of', 'from', 'these', 'in', 'with', 'to', 'has',
    'should', 'will', 'and', 'that', 'this', 'had', 'those', 'would'
}
TRANSLATION_SIGNAL_WORDS = {
    'say', 'owed', 'been', 'given', 'their', 'are', 'received', 'made', 'bought',
    'being', 'your', 'had', 'be', 'sealed', 'placed', 'entrusted', 'give', 'my',
    'took', 'refused', 'his', 'returned', 'arrived', 'settled', 'left', 'has',
    'claimed', 'gave', 'send', 'paid', 'if', 'take', 'deposited', 'owe', 'brought',
    'added', 'wrote', 'testified', 'ordered', 'would', 'was', 'declared', 'confirmed',
    'divided', 'demanded', 'sent', 'witnessed', 'says', 'owes', 'delivered', 'collected',
    'were', 'invested', 'should', 'her', 'came', 'asked', 'sold', 'entitled', 'answered',
    'promised', 'told', 'thus', 'went', 'bring', 'receive', 'have', 'agreed', 'shall',
    'pay', 'swore', 'died', 'our', 'assisted', 'appealed', 'will', 'is', 'contracted'
}


# --------------- OCR post-processing ---------------

def fix_ocr_artifacts(text: str) -> str:
    """Fix common MinerU OCR character-level errors in transliterations."""
    text = text.translate(SUPERSCRIPT_TRANS)
    # 1. Emphatic consonants: cedilla/comma-below variants → correct dots-below
    text = text.replace('\u0219', '\u1e63')   # ș → ṣ  (s-comma-below)
    text = text.replace('\u0218', '\u1e62')   # Ș → Ṣ
    text = text.replace('\u015f', '\u1e63')   # ş → ṣ  (s-cedilla)
    text = text.replace('\u015e', '\u1e62')   # Ş → Ṣ
    text = text.replace('\u021b', '\u1e6d')   # ț → ṭ  (t-comma-below)
    text = text.replace('\u021a', '\u1e6c')   # Ț → Ṭ
    # 2. đ (d-stroke) → {d}  (divine determinative)
    text = re.sub(r'đ(?=[A-Z])', '{d}', text)
    text = re.sub(r'dτ(?=[A-Z])', '{d}', text)
    # 2.5 Hard-coded OCR garbage tokens observed in AKT OCR output
    text = re.sub(r'\{(?:ší|bia)\}', ' ', text)
    # 2.55 Broken editorial brackets from damaged-text notation.
    # OCR often turns ``[a-ma-kam]`` into ``{a-ma -kam2`` and
    # ``[A-šur]-i-dí`` into ``{A-šur1-i-dí``.  In Assyriological
    # transliteration, Arabic digits 1/2/3 are not valid homophone
    # indices inside these syllabic chains, so strip them when they
    # are glued to letters or hyphens.
    text = re.sub(r'(?<!\w)\{(?=[A-Za-zÀ-ÿ]+(?:\s*-\s*|-))', '', text)
    text = re.sub(r'(?<=[A-Za-zÀ-ÿxX\-\]\}])[123](?=[A-Za-zÀ-ÿxX\-\[\{]|$)', '', text)
    text = re.sub(r'(\d/\d)[123](?=\s+[A-Za-zÀ-ÿ])', r'\1', text)
    text = re.sub(r'(?<=[A-Za-zÀ-ÿ0-9])\s*-\s*(?=[A-Za-zÀ-ÿ0-9])', '-', text)
    # 2.6 Stray single braces and apostrophes inside syllabic chains
    text = re.sub(r'(?<=[A-Za-zÀ-ÿ0-9]-)\{(?=[A-Za-zÀ-ÿ0-9])', '', text)
    text = re.sub(r'(?<=[A-Za-zÀ-ÿ0-9])\}(?=-[A-Za-zÀ-ÿ0-9])', '', text)
    text = re.sub(r"(?<=[A-Za-zÀ-ÿ])['’](?=-[A-Za-zÀ-ÿ])", '', text)
    # 2.7 Edge marker OCR variants
    text = re.sub(r'(?<!\w)(?:lo\.e\.?|ri\.e\.?)(?!\w)', ' ', text, flags=re.IGNORECASE)
    # 3. Dotless-i → regular i
    text = text.replace('\u0131', 'i')          # ı → i
    text = text.replace('\u0130', 'I')          # İ → I
    # 4. Line-break slashes: "ma-/na" → "ma-na", "li-wi-/tim" → "li-wi-tim"
    text = re.sub(r'-/\s*', '-', text)
    # 4.5 Broken/damage dots normalized to <gap>
    text = re.sub(r'(?<!\.)\.{2,}(?!\.)', ' <gap> ', text)
    # 5. Macron artifact
    text = text.replace('\u00af', '')            # ¯ → remove
    # 6. Prime → remove (damage marker artifact)
    text = text.replace('\u2032', '')            # ′ → remove
    text = text.replace('™', '')
    text = text.replace('ª', '')
    text = text.replace('°', '')
    text = text.replace('τ', '')
    text = text.replace('∪', 'U')
    # 7. Known name-specific OCR fix before generic ogonek normalization
    text = text.replace('Puzurų', 'Puzur4')
    text = re.sub(r'\bPuzuru(?=[-/])', 'Puzur4', text)
    # 8. Ogonek artifacts (OCR misread of subscript digits near vowels)
    text = text.replace('\u0105', 'a')           # ą → a
    text = text.replace('\u0173', 'u')           # ų → u
    # 9. Stray '7' OCR artifacts from misread line numbers / superscripts.
    # OCR reads faint line numbers or superscript markers as literal '7' embedded
    # in transliteration text.
    # 9a. Uppercase sign + 7 + separator: AN7.NA → AN.NA, DINGIR7-ba → DINGIR-ba
    text = re.sub(r'(?<=[A-Z])7(?=[.\-])', '', text)
    # 9b. Uppercase sign + 7 at word end: KB7 → KB, DÙG7 → DÙG, DUMU7 → DUMU
    text = re.sub(r'(?<=[A-ZÁÀÉÈÍÌÚÙŠṢṬḪ])7(?=[\s,;:\-]|$)', '', text)
    # 9c. Lowercase syllable + 7 + hyphen: ša7-ṣí → ša-ṣí, ma7-na → ma-na
    text = re.sub(r'(?<=[a-záàéèíìúùšṣṭḫ])7(?=-)', '', text)
    # 9d. Lowercase syllable + 7 at word end/before punctuation: x7 → x
    text = re.sub(r'(?<=[a-záàéèíìúùšṣṭḫ])7(?=[\s,;:\]\)»«\'"\.]|$)', '', text)
    # 9d2. Apostrophe + 7 before hyphen: ma'7-nim → ma'-nim
    text = re.sub(r"(?<=')7(?=-)", '', text)
    # 9e. Stray leading digits from line number bleed: "5e7-ri" → "e-ri"
    text = re.sub(r'(?<!\S)\d{1,2}(?=[a-záàéèíìúùšṣṭḫ]{1,3}7?-)', '', text)
    # 9e2. Stray digit before uppercase name: "7B-ap-šu" → "B-ap-šu"
    text = re.sub(r'(?<=\s)7(?=[A-ZÁÀÉÈÍÌÚÙŠṢṬḪ][a-záàéèíìúùšṣṭḫ\-])', '', text)
    # 9e3. Stray digit after hyphen at word end or before brackets: "Ú-7[]" → "Ú-[]", "Ú-7" → "Ú-"
    text = re.sub(r'(?<=-)7(?=\[|\s|$)', '', text)
    # 9e4. Stray digit after collation marks before hyphen: "is!?7-tim" → "is!?-tim"
    text = re.sub(r'(?<=[!?])7(?=-)', '', text)
    # 9f. Stray trailing 7 from line numbers merged with preceding digits
    text = re.sub(r'(\d{2})7\s', r'\1 ', text)
    # 9g. Fraction + stray 7: 2/37 → 2/3 (line number 7 merged after fraction)
    text = re.sub(r'(\d/\d)7(?=[a-záàéèíìúùšṣṭḫA-Z])', r'\1', text)
    # 9h. OCR reads subscript 5 as Latin ``s`` in common il5- forms.
    text = re.sub(r'\bils(?=-)', 'il5', text)
    # 9i. OCR sometimes reads a damaged closing bracket as lowercase ``l``.
    # Keep this narrow to the patterns observed in AKT 6a.
    text = re.sub(r'(?<=-)rul-(?=[A-Za-zÀ-ÿ])', 'ru-', text)
    text = re.sub(r'(?<=-)nul-(?=[A-Za-zÀ-ÿ])', 'nu-', text)
    text = re.sub(r'(?<=-)ril-(?=[A-Za-zÀ-ÿ])', 'ri-', text)
    text = re.sub(r'(?<=[A-ZÁÀÉÈÍÌÚÙŠṢṬḪ0-9])l\b', '', text)
    # 10. Sign/determinative normalization
    text = re.sub(r'\bSIGs\b', 'SIG5', text)
    text = re.sub(r'\bSIG,(?=-)', 'SIG5', text)
    text = re.sub(r'\bSIG,\s*', 'SIG5 ', text)
    text = re.sub(r'\bKÙ\.B\.?(?![A-Z])', 'KÙ.BABBAR', text)
    text = re.sub(r'(?<=\d)\s*F?KB(?:["\']*\d*)?(?![A-Za-z])', ' KÙ.BABBAR', text)
    text = re.sub(r'(?<![A-Za-z0-9])F?KB(?:["\']*\d*)?(?![A-Za-z])', 'KÙ.BABBAR', text)
    return text


def fix_translation_ocr_artifacts(text: str) -> str:
    """Remove OCR-only superscripts and editorial marker debris in English."""
    text = text.translate(str.maketrans({
        "⁰": "",
        "¹": "",
        "²": "",
        "³": "",
        "⁴": "",
        "⁵": "",
        "⁶": "",
        "⁷": "",
        "⁸": "",
        "⁹": "",
        "ª": "",
        "°": "",
    }))
    text = re.sub(
        rf'(?<!\w)\d+\s*{_TRANSLATION_TABLET_MARKER_PATTERN}(?!\w)\s*',
        ' ',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        rf'(?<!\w){_TRANSLATION_TABLET_MARKER_PATTERN}(?!\w)',
        ' ',
        text,
        flags=re.IGNORECASE,
    )
    # Stray line numbers that survived span filtering (e.g. "... rece- 20 e. ived")
    text = re.sub(
        r'(?<!\w)\d{1,3}\s*(?=' + _TRANSLATION_TABLET_MARKER_PATTERN + r')',
        ' ',
        text,
        flags=re.IGNORECASE,
    )
    # Restore words split by PDF line breaks: ``col- ony`` → ``colony``.
    # For proper names with an intentional hyphen, keep the hyphen but
    # remove the inserted whitespace: ``Puzur- Ti'amtum`` → ``Puzur-Ti'amtum``.
    text = re.sub(
        r'\b([A-ZĀĒĪŪŠṢṬḪ][A-Za-zĀāĒēĪīŪūŠšṢṣṬṭḪḫÇçȘșȚț\'’.-]{1,30})-\s+'
        r'([a-z][A-Za-zĀāĒēĪīŪūŠšṢṣṬṭḪḫÇçȘșȚț\'’.-]{1,30})\b',
        r'\1-\2',
        text,
    )
    text = re.sub(r'(?<=[a-z])-\s+(?=[a-z])', '', text)
    text = re.sub(
        r'\b([A-ZĀĒĪŪŠṢṬḪ][A-Za-zĀāĒēĪīŪūŠšṢṣṬṭḪḫÇçȘșȚț\'’.-]{1,30})-\s+'
        r'([A-ZĀĒĪŪŠṢṬḪ][A-Za-zĀāĒēĪīŪūŠšṢṣṬṭḪḫÇçȘșȚț\'’.-]{1,30})\b',
        r'\1-\2',
        text,
    )
    return text


# Helper functions
def normalize_whitespace(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\xad', '')
    return text.strip()


def normalize_visual_noise(text: str, *, is_translation: bool) -> str:
    text = text.replace('\xad', '')
    text = text.replace('–', '-').replace('—', '-')
    text = text.replace('…', ' ... ')
    text = re.sub(r'\*(?=[A-Za-zÀ-ÿ])', '', text)
    if not is_translation:
        text = text.replace('<<', '').replace('>>', '')
    return normalize_whitespace(text)


def strip_left_prefix(text: str) -> str:
    text = LEFT_PREFIX_RE.sub('', text)
    text = LEADING_LINE_NUM_RE.sub('', text)
    text = re.sub(r'^[\.\,\;\:\-]+\s*', '', text)
    return normalize_whitespace(text)


def group_lines_by_y(lines: list[PageLine], tolerance: float) -> list[list[PageLine]]:
    if not lines:
        return []
    grouped = [[lines[0]]]
    for line in lines[1:]:
        if abs(line.y0 - grouped[-1][-1].y0) <= tolerance:
            grouped[-1].append(line)
        else:
            grouped.append([line])
    return grouped


def nonspace_len(text: str) -> int:
    return len(re.sub(r'\s+', '', text))


def looks_like_transliteration(text: str) -> bool:
    text = normalize_whitespace(text)
    if not text or EDITORIAL_RE.search(text):
        return False
    tokens = TRANSLIT_TOKEN_RE.findall(text)
    if len(tokens) < 2:
        return False
    lowered = [t.lower() for t in tokens]
    stopword_count = sum(1 for w in lowered if w in STOPWORDS)
    return stopword_count < len(tokens) * 0.5


def looks_like_translation(text: str) -> bool:
    text = normalize_whitespace(text)
    if not text or EDITORIAL_RE.search(text):
        return False
    if re.search(r'\b\d{4}\b', text) or re.search(r'[A-Za-z][/\\][A-Za-z]', text):
        return False
    words = ENGLISH_WORD_RE.findall(text)
    if len(words) < 3:
        return False
    lowered = [w.lower() for w in words]
    if lowered[:1] == ['witnessed'] and len(words) >= 2:
        return True
    signal_count = sum(1 for w in lowered if w in TRANSLATION_SIGNAL_WORDS)
    if signal_count >= 1:
        return True
    starts_with_num_or_quote = bool(re.match(r'^[0-9"\']', text))
    if starts_with_num_or_quote and len(words) >= 6:
        return True
    # Relaxed: long enough English text with common Akkadian translation patterns
    if len(words) >= 5:
        # Common patterns in Akkadian translations
        akk_patterns = (
            'mina', 'minas', 'shekel', 'shekels', 'talent', 'talents',
            'silver', 'gold', 'tin', 'textile', 'textiles', 'kutānu',
            'colony', 'palace', 'tablet', 'seal', 'eponym', 'eponymy',
            'son', 'daughter', 'wife', 'father', 'mother', 'brother',
            'caravan', 'donkey', 'donkeys', 'merchandise', 'goods',
        )
        akk_count = sum(1 for w in lowered if w in akk_patterns)
        if akk_count >= 1:
            return True
    # Relaxed: starts with "Thus" (letter opening) or "Say to"
    if text.startswith('Thus ') or text.startswith('Say to '):
        return True
    # Relaxed: contains typical Assyrian proper name patterns (macron vowels)
    if any(c in text for c in 'āēīūṣṭ') and len(words) >= 4:
        return True
    return False


def translation_has_terminal_punctuation(text: str) -> bool:
    text = text.strip()
    return text.endswith(('.', '!', '?'))


def _extract_english_opening_names(text: str) -> list[str]:
    prefix = re.split(r'[.:;]', text, maxsplit=1)[0]
    prefix = ENGLISH_QTY_UNIT_RE.split(prefix, maxsplit=1)[0]
    stop = {'to', 'from', 'thus', 'say', 'and', 'as', 'the'}
    return [name for name in ENGLISH_NAME_RE.findall(prefix) if name.lower() not in stop]


def _extract_akkadian_opening_names(text: str) -> list[str]:
    prefix = AKKADIAN_COMMODITY_RE.split(text, maxsplit=1)[0]
    tokens = re.findall(r'\S+', prefix)
    names: list[str] = []
    skip = {
        'a-na', 'qí-bi-ma', 'qi-bi-ma', 'qí-bi4-ma', 'qi-bi4-ma',
        'um-ma', 'ù', 'u', 'ša', 'ša',
    }
    for token in tokens:
        token = token.strip(',:;[]()<>')
        if not token or token.lower() in skip:
            continue
        if '-' not in token:
            continue
        if any('A' <= ch <= 'Z' or ch in 'ŠṢṬḪÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝ' for ch in token):
            names.append(token)
    return names


def obvious_alignment_mismatch_reason(akk: str, eng: str) -> str | None:
    if not re.match(r'^(?:To|From)\b', eng):
        return None
    english_names = _extract_english_opening_names(eng)
    if len(english_names) < 3:
        return None

    akkadian_names = _extract_akkadian_opening_names(akk)
    if len(akkadian_names) <= 1:
        return 'letter_opening_name_mismatch'
    if re.search(r'\bum-ma\s+(?:ma-na|GÍN|GÚ|KÙ\.BABBAR|KÙ\.B\.?|AN\.NA|URUDU|TÚG)\b', akk):
        return 'letter_opening_name_mismatch'
    return None


def reject_reason_for_candidate(
    raw_akkadian: str,
    raw_english: str,
    clean_akkadian: str,
    clean_english: str,
    cfg,
) -> str | None:
    if not looks_like_transliteration(clean_akkadian):
        return 'weak_transliteration'
    if not looks_like_translation(clean_english):
        return 'weak_translation'
    english_words = ENGLISH_WORD_RE.findall(clean_english)
    if len(english_words) < 3:
        return 'translation_too_short_words'
    mismatch_reason = obvious_alignment_mismatch_reason(clean_akkadian, clean_english)
    if mismatch_reason is not None:
        return mismatch_reason
    if UNEXPECTED_OCR_SYMBOL_RE.search(clean_akkadian):
        return 'unexpected_ocr_symbols'
    return None


def make_oare_id(pdf_name: str, page: int, pair_index: int, akkadian: str, english: str) -> str:
    content = f'{pdf_name}|{page}|{pair_index}|{akkadian}|{english}'
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]


def write_jsonl_line(fh, data: dict) -> None:
    fh.write(json.dumps(data, ensure_ascii=False) + '\n')


def _merge_cross_page_records(
    all_page_records: list[tuple[int, list[tuple[int, int, int, str, str]]]]
) -> list[tuple[int, int, int, str, str]]:
    flat = []
    for page_num, records in all_page_records:
        for entry_index, row_start, row_end, left, right in records:
            flat.append((page_num, entry_index, row_start, row_end, left, right))
    if not flat:
        return []
    merged = [flat[0]]
    for page_num, entry_index, row_start, row_end, left, right in flat[1:]:
        last_page, last_entry_index, last_start, last_end, last_left, last_right = merged[-1]
        if (
            page_num == last_page + 1
            and entry_index == 0
        ):
            merged[-1] = (
                last_page,
                last_entry_index,
                last_start,
                row_end,
                normalize_whitespace(last_left + ' ' + left),
                normalize_whitespace(last_right + ' ' + right),
            )
        else:
            merged.append((page_num, entry_index, row_start, row_end, left, right))
    return [
        (page_num, row_start, row_end, left, right)
        for page_num, _, row_start, row_end, left, right in merged
    ]


def save_outputs(records: list[CandidateRecord], stats: dict[str, Any], cfg) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.output_csv, 'w', encoding='utf-8') as csv_fh, \
         open(cfg.metadata_jsonl, 'w', encoding='utf-8') as meta_fh:
        writer = csv.DictWriter(csv_fh, fieldnames=['oare_id', 'transliteration', 'translation'])
        writer.writeheader()
        page_pair_counters = Counter()
        for record in records:
            page_pair_counters[record.page] += 1
            pair_index = page_pair_counters[record.page]
            oare_id = make_oare_id(
                str(cfg.pdf_path.name), record.page, pair_index,
                record.akkadian, record.english
            )
            writer.writerow({
                'oare_id': oare_id,
                'transliteration': record.akkadian,
                'translation': record.english
            })
            write_jsonl_line(meta_fh, {
                'oare_id': oare_id,
                'page': record.page,
                'row_start_index': record.row_start_index,
                'row_end_index': record.row_end_index,
                'raw_akkadian': record.raw_akkadian,
                'raw_english': record.raw_english,
                'clean_char_ratio': round(record.clean_char_ratio, 4),
                'translit_retention': round(record.translit_retention, 4),
                'translation_retention': round(record.translation_retention, 4),
            })
    cfg.report_json.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )


# ============================================================
# End of inlined dependencies
# ============================================================


@dataclass
class ExtractConfig:
    pdf_path: Path = Path("/data/lsb/deep_past/data/old-assyrian/AKT 8 2015.pdf")
    middle_json_path: Path = Path(
        "/data/lsb/deep_past/output/akt8_compare/mineru_ocr_raw/AKT 8 2015/ocr/AKT 8 2015_middle.json"
    )
    output_dir: Path = Path("/data/lsb/deep_past/output/akt8_mineru_ocr_extract")
    source_start_page: int = 49
    source_end_page: int = 517
    column_split_ratio: float = 0.52
    column_margin: float = 20.0
    line_group_y_tolerance: float = 3.5
    full_width_ratio: float = 0.68
    min_row_pairs_per_page: int = 3
    preview_count: int = 12
    run_ocr: bool = True
    ocr_chunk_size: int = 200
    ocr_lang: str = "latin"
    ocr_lang_right: str = ""  # separate OCR lang for right column; empty = use ocr_lang for both
    ocr_workers: int = 4
    mineru_device: str = "cuda"
    middle_json_path_right: Path = Path("")  # cached en middle.json; empty = not provided

    @property
    def output_csv(self) -> Path:
        return self.output_dir / "train.csv"

    @property
    def metadata_jsonl(self) -> Path:
        return self.output_dir / "pairs_metadata.jsonl"

    @property
    def rejected_jsonl(self) -> Path:
        return self.output_dir / "rejected_candidates.jsonl"

    @property
    def report_json(self) -> Path:
        return self.output_dir / "report.json"


# Tablet orientation markers (not part of the transliteration content)
_TABLET_MARKER_RE = re.compile(rf"^{_LEFT_TABLET_MARKER_PATTERN}$", re.IGNORECASE)
# Tablet line numbers: pure digits optionally followed by a prime mark (e.g. "5", "10", "15'")
_LINE_NUMBER_RE = re.compile(r"^\d{1,3}'?$")
# Margin line numbers merged after fraction ½ by OCR (e.g. "5½4 GÍN" → "5½ GÍN", "½/3 ma-na" → "½ ma-na")
_FRAC_LINENUM_RE = re.compile(r"½/?\d{1,2}")
# Margin line numbers merged at the start of a span (e.g. "20 e-ra-áb" → "e-ra-áb")
# Margin x0 ∈ [80,91], main text x0 ∈ [101,178]; threshold sits in the 10px gap.
_MARGIN_X_THRESHOLD = 96.0
# Ratio-based margin threshold for variable page widths (96/595 ≈ 0.161)
_MARGIN_X_RATIO = 0.161
_LEADING_LINENUM_RE = re.compile(r"^\d{1,2}'?\s+")
# Combined line-number + tablet-marker in one span (e.g. "20 e.", "5 rev.", "10 lo.e.")
_LINENUM_MARKER_RE = re.compile(
    rf"^\d{{1,3}}'?\s*{_LEFT_TABLET_MARKER_PATTERN}\s*$",
    re.IGNORECASE,
)


def _mineru_config_path() -> Path:
    config_name = os.environ.get("MINERU_TOOLS_CONFIG_JSON", "mineru.json")
    config_path = Path(config_name)
    if not config_path.is_absolute():
        config_path = Path.home() / config_name
    return config_path


def _read_mineru_config() -> dict[str, Any]:
    config_path = _mineru_config_path()
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def _set_mineru_local_model_dir(repo_mode: str, root_path: Path) -> None:
    config_path = _mineru_config_path()
    config = _read_mineru_config()
    models_dir = config.setdefault("models-dir", {})
    root_str = str(root_path)
    if models_dir.get(repo_mode) == root_str:
        return
    models_dir[repo_mode] = root_str
    config_path.write_text(
        json.dumps(config, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def _normalize_paddle_lang(lang: str) -> str:
    from mineru.model.ocr.pytorch_paddle import (
        arabic_lang,
        cyrillic_lang,
        devanagari_lang,
        east_slavic_lang,
        latin_lang,
    )

    if lang in latin_lang:
        return "latin"
    if lang in east_slavic_lang:
        return "east_slavic"
    if lang in arabic_lang:
        return "arabic"
    if lang in cyrillic_lang:
        return "cyrillic"
    if lang in devanagari_lang:
        return "devanagari"
    return lang


def _required_mineru_pipeline_paths(*ocr_langs: str) -> list[str]:
    import yaml

    from mineru.model.ocr.pytorch_paddle import get_model_params, root_dir
    from mineru.utils.enum_class import ModelPath

    models_config_path = (
        Path(root_dir) / "pytorchocr" / "utils" / "resources" / "models_config.yml"
    )
    with open(models_config_path, "r", encoding="utf-8") as fh:
        models_config = yaml.safe_load(fh)

    required_paths = {ModelPath.doclayout_yolo, ModelPath.layout_reader}
    langs_to_prepare = {"ch"}
    langs_to_prepare.update(lang for lang in ocr_langs if lang)
    for lang in langs_to_prepare:
        canonical_lang = _normalize_paddle_lang(lang)
        det_model, rec_model, _ = get_model_params(canonical_lang, models_config)
        required_paths.add(f"{ModelPath.pytorch_paddle}/{det_model}")
        required_paths.add(f"{ModelPath.pytorch_paddle}/{rec_model}")

    return sorted(required_paths)


def _missing_pipeline_paths(root_path: Path | None, required_paths: list[str]) -> list[str]:
    if root_path is None:
        return list(required_paths)
    return [rel for rel in required_paths if not (root_path / rel).exists()]


def _ensure_local_mineru_pipeline_assets(ocr_lang: str) -> Path:
    try:
        from mineru.utils.models_download_utils import auto_download_and_get_model_root_path
    except ImportError as exc:
        raise RuntimeError(
            "MinerU Python package is not importable in this environment; "
            "cannot bootstrap local OCR models."
        ) from exc

    required_paths = _required_mineru_pipeline_paths(ocr_lang)
    config = _read_mineru_config()
    models_dir = config.get("models-dir") or {}
    current_root = Path(models_dir["pipeline"]) if models_dir.get("pipeline") else None
    missing = _missing_pipeline_paths(current_root, required_paths)
    if not missing and current_root is not None:
        return current_root

    if current_root is None:
        print("  ⚙️  MinerU local pipeline cache is not configured; bootstrapping models...",
              flush=True)
    else:
        print(
            f"  ⚙️  MinerU local pipeline cache is incomplete at {current_root}; "
            f"fetching {len(missing)} missing asset(s)...",
            flush=True,
        )

    previous_source = os.environ.get("MINERU_MODEL_SOURCE")
    errors: list[str] = []
    for source in ("modelscope", "huggingface"):
        try:
            os.environ["MINERU_MODEL_SOURCE"] = source
            downloaded_root: Path | None = None
            for rel_path in missing or required_paths:
                downloaded_root = Path(
                    auto_download_and_get_model_root_path(rel_path, repo_mode="pipeline")
                )
            if downloaded_root is None:
                raise RuntimeError("download helper returned no root path")
            remaining = _missing_pipeline_paths(downloaded_root, required_paths)
            if remaining:
                raise RuntimeError(
                    f"download completed but still missing: {', '.join(remaining)}"
                )
            _set_mineru_local_model_dir("pipeline", downloaded_root)
            if previous_source is None:
                os.environ.pop("MINERU_MODEL_SOURCE", None)
            else:
                os.environ["MINERU_MODEL_SOURCE"] = previous_source
            print(f"  ✅ MinerU local pipeline cache ready: {downloaded_root}", flush=True)
            return downloaded_root
        except Exception as exc:  # pragma: no cover - network/backend dependent
            errors.append(f"{source}: {exc}")
    if previous_source is None:
        os.environ.pop("MINERU_MODEL_SOURCE", None)
    else:
        os.environ["MINERU_MODEL_SOURCE"] = previous_source

    raise RuntimeError(
        "Failed to prepare MinerU local pipeline models for OCR. "
        + " | ".join(errors)
    )


def _find_middle_json(output_dir: Path) -> Path:
    """Locate the middle.json produced by `mineru` CLI under its output tree."""
    candidates = list(output_dir.rglob("*_middle.json"))
    if len(candidates) == 1:
        return candidates[0]
    # Fall back to any middle.json
    candidates = list(output_dir.rglob("middle.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No middle.json found under {output_dir}")


def _block_text(block: dict[str, Any]) -> str:
    """Extract full text from a MinerU block (joining all spans)."""
    parts: list[str] = []
    for line in block.get("lines") or []:
        for span in line.get("spans") or []:
            if isinstance(span, dict) and span.get("type") == "text":
                parts.append(span.get("content", ""))
    return normalize_whitespace(" ".join(parts))


def _extract_spans(
    page_entry: dict[str, Any],
    page_width: float,
    *,
    side_filter: str = "all",
    split_x: float = 0.0,
) -> list[Any]:
    """Extract PageLine objects from a single page_entry.

    *side_filter*: ``"all"`` (no filtering), ``"left"`` (x0 < split_x),
    or ``"right"`` (x0 >= split_x).
    """
    lines: list[Any] = []
    for block in page_entry.get("preproc_blocks") or []:
        btype = block.get("type", "text")
        if btype == "title":
            text = _block_text(block)
            if text:
                bbx = block.get("bbox") or [0, 0, 0, 0]
                bx0 = float(bbx[0])
                if side_filter == "left" and bx0 >= split_x:
                    continue
                if side_filter == "right" and bx0 < split_x:
                    continue
                lines.append(PageLine(
                    x0=bx0, x1=float(bbx[2]),
                    y0=float(bbx[1]), y1=float(bbx[3]),
                    text=text,
                    width_ratio=float((bbx[2] - bbx[0]) / page_width),
                ))
            continue
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                if not isinstance(span, dict) or span.get("type") != "text":
                    continue
                content = span.get("content", "").strip()
                if not content or _TABLET_MARKER_RE.match(content) or _LINE_NUMBER_RE.match(content) or _LINENUM_MARKER_RE.match(content):
                    continue
                sbbox = span.get("bbox") or []
                if len(sbbox) != 4:
                    continue
                x0, y0, x1, y1 = map(float, sbbox)
                if side_filter == "left" and x0 >= split_x:
                    continue
                if side_filter == "right" and x0 < split_x:
                    continue
                # Strip leading margin line numbers (proportional to page width)
                margin_threshold = page_width * _MARGIN_X_RATIO
                if x0 < margin_threshold:
                    content = _LEADING_LINENUM_RE.sub("", content).strip()
                    if not content or _LINE_NUMBER_RE.match(content):
                        continue
                lines.append(PageLine(
                    x0=x0, x1=x1, y0=y0, y1=y1,
                    text=content,
                    width_ratio=float((x1 - x0) / page_width),
                ))
    return lines


# Physical column boundary for dual-OCR span selection.
# Left text column x0 ∈ [80, 199], right text column x0 ∈ [200, 297].
# This is distinct from column_split_ratio (0.52 → 309px) used downstream
# for row-level left/right assignment (which accounts for varying span widths).
_DUAL_OCR_SPLIT_X = 200.0


def build_page_lines(
    page_entry: dict[str, Any],
    page_entry_right: dict[str, Any] | None = None,
    column_split_ratio: float = 0.52,
) -> tuple[list[Any], float]:
    """Build PageLine objects from MinerU spans (one PageLine per span).

    When *page_entry_right* is provided (dual-language OCR), left-column spans
    come from *page_entry* (latin) and right-column spans from
    *page_entry_right* (en).  Otherwise all spans come from *page_entry*.

    Title blocks and full-width description blocks are skipped here — they are
    handled at the block level in extract_aligned_rows.
    """
    page_width = float((page_entry.get("page_size") or [595, 842])[0])

    if page_entry_right is not None:
        lines = _extract_spans(
            page_entry, page_width, side_filter="left", split_x=_DUAL_OCR_SPLIT_X,
        )
        lines += _extract_spans(
            page_entry_right, page_width, side_filter="right", split_x=_DUAL_OCR_SPLIT_X,
        )
    else:
        lines = _extract_spans(page_entry, page_width)

    lines.sort(key=lambda item: (round(item.y0), item.x0))
    return lines, page_width


def extract_aligned_rows(
    page_entry: dict[str, Any],
    source_page_num: int,
    cfg: ExtractConfig,
    page_entry_right: dict[str, Any] | None = None,
) -> list[Any]:
    page_lines, page_width = build_page_lines(
        page_entry, page_entry_right=page_entry_right,
        column_split_ratio=cfg.column_split_ratio,
    )
    split_x = page_width * cfg.column_split_ratio
    grouped = group_lines_by_y(page_lines, cfg.line_group_y_tolerance)
    rows: list[Any] = []
    in_notes_section = False
    entry_index = 0

    for group in grouped:
        normalized_group = [
            normalize_visual_noise(line.text, is_translation=False) for line in group
        ]
        if any(ENTRY_HEADER_RE.match(text) for text in normalized_group):
            entry_index += 1
            in_notes_section = False
            continue
        # "Notes" / "Comment" headers start a notes section; skip until
        # the next entry title resets the flag.
        if any(SKIP_SECTION_RE.match(text) for text in normalized_group):
            in_notes_section = True
            continue
        if in_notes_section:
            continue
        if all(PAGE_HEADER_RE.match(text) for text in normalized_group):
            continue
        if any(line.width_ratio >= cfg.full_width_ratio for line in group):
            continue

        left_lines = [
            line
            for line in group
            if line.x1 <= split_x + cfg.column_margin and line.x0 < split_x
        ]
        right_lines = [
            line
            for line in group
            if line.x0 >= split_x - cfg.column_margin and line.x1 > split_x
        ]
        if not left_lines or not right_lines:
            # Append trailing right-only text to the last aligned row
            # (translation extends beyond transliteration).
            if right_lines and not left_lines and rows and rows[-1].entry_index == entry_index:
                extra_right = normalize_whitespace(
                    " ".join(
                        normalize_visual_noise(line.text, is_translation=True)
                        for line in right_lines
                    )
                )
                extra_right = normalize_whitespace(
                    re.sub(r"^[\.\,\;\:\-]+\s*", "", extra_right)
                )
                if extra_right and not EDITORIAL_RE.search(extra_right):
                    rows[-1] = AlignedRow(
                        page=rows[-1].page,
                        y0=rows[-1].y0,
                        left=rows[-1].left,
                        right=normalize_whitespace(rows[-1].right + " " + extra_right),
                        raw_left=rows[-1].raw_left,
                        raw_right=normalize_whitespace(rows[-1].raw_right + " " + extra_right),
                        entry_index=rows[-1].entry_index,
                    )
            continue

        raw_left = normalize_whitespace(
            " ".join(
                normalize_visual_noise(line.text, is_translation=False) for line in left_lines
            )
        )
        raw_right = normalize_whitespace(
            " ".join(
                normalize_visual_noise(line.text, is_translation=True) for line in right_lines
            )
        )
        left = strip_left_prefix(raw_left)
        # MinerU spans don't contain line numbers (they are separate left-column
        # spans already filtered), so don't strip leading digits from English.
        right = normalize_whitespace(
            re.sub(r"^[\.\,\;\:\-]+\s*", "", raw_right)
        )
        if not left or not right:
            continue
        if EDITORIAL_RE.search(left) or EDITORIAL_RE.search(right):
            continue

        rows.append(
            AlignedRow(
                page=source_page_num,
                y0=min(line.y0 for line in group),
                left=left,
                right=right,
                raw_left=raw_left,
                raw_right=raw_right,
                entry_index=entry_index,
            )
        )

    return rows


_TABLET_MARKER_INLINE_RE = re.compile(
    rf"(?<!\w){_LEFT_TABLET_MARKER_PATTERN}(?!\w)\s*",
    re.IGNORECASE,
)


def _strip_tablet_markers(text: str) -> str:
    """Remove tablet orientation markers from concatenated text."""
    return normalize_whitespace(_TABLET_MARKER_INLINE_RE.sub("", text))


def _merge_editorial_brackets(text: str) -> str:
    """Merge Assyriological editorial brackets into surrounding text.

    MinerU correctly preserves bracket notation like ``a-hu-u[r]`` and
    ``nu'-hu-u[m š]a``.  But prepare_data.preprocess_transliteration strips
    ``[]`` and ``'`` by replacing them with spaces, breaking words apart.

    This pre-processing removes brackets while keeping their content joined
    to the surrounding characters, so downstream cleaning won't fragment words.
    """
    # [content] → content  (e.g., a-hu-u[r] → a-hu-ur)
    text = re.sub(r"\[([^\]]*)\]", r"\1", text)
    return text


def assemble_records_split_entries(
    rows: list[Any], page_num: int
) -> list[tuple[int, int, int, str, str]]:
    """Assemble records by entry header boundaries, not line spacing."""
    if not rows:
        return []

    groups: list[list[Any]] = [[rows[0]]]
    for row in rows[1:]:
        if row.entry_index != groups[-1][-1].entry_index:
            groups.append([row])
        else:
            groups[-1].append(row)

    records: list[tuple[int, int, int, str, str]] = []
    for group in groups:
        raw_left = _merge_editorial_brackets(_strip_tablet_markers(
            normalize_whitespace(" ".join(r.left for r in group))
        ))
        raw_left = _FRAC_LINENUM_RE.sub("½", raw_left)
        raw_right = _merge_editorial_brackets(
            normalize_whitespace(" ".join(r.right for r in group))
        )
        raw_right = _FRAC_LINENUM_RE.sub("½", raw_right)
        if raw_left and raw_right:
            start_idx = rows.index(group[0])
            end_idx = rows.index(group[-1])
            records.append((group[0].entry_index, start_idx, end_idx, raw_left, raw_right))
    return records


def _run_mineru_cli_chunk(
    pdf_path: Path, chunk_start: int, chunk_end: int, cfg: ExtractConfig,
) -> list[dict[str, Any]]:
    """Run ``mineru`` CLI on a page range and return tagged pdf_info entries.

    ``chunk_start`` / ``chunk_end`` are **1-indexed** source page numbers.
    """
    import subprocess

    s0, e0 = chunk_start - 1, chunk_end - 1  # CLI uses 0-indexed
    n_pages = e0 - s0 + 1
    print(f"  🔬 pages {chunk_start}-{chunk_end} ({n_pages} pages)...", flush=True)
    t0 = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "out"
        cmd = [
            "mineru",
            "-p", str(pdf_path),
            "-o", str(tmp_out),
            "-m", "ocr",
            "-l", cfg.ocr_lang,
            "-b", "pipeline",
            "-f", "False",
            "-t", "False",
            "--source", "local",
            "-s", str(s0),
            "-e", str(e0),
        ]
        if cfg.mineru_device and cfg.mineru_device != "auto":
            cmd += ["-d", cfg.mineru_device]

        env = os.environ.copy()
        env["MINERU_MODEL_SOURCE"] = "local"

        result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"mineru CLI exited with code {result.returncode} "
                f"for pages {chunk_start}-{chunk_end}"
            )

        try:
            middle_path = _find_middle_json(tmp_out)
        except FileNotFoundError:
            print(f"  ⚠️  No middle.json produced for pages {chunk_start}-{chunk_end}",
                  flush=True)
            return []
        middle_json = json.loads(middle_path.read_text(encoding="utf-8"))

    pdf_info = middle_json.get("pdf_info") or []
    elapsed = time.time() - t0
    print(
        f"  ✅ pages {chunk_start}-{chunk_end}: {len(pdf_info)} pages in {elapsed:.1f}s "
        f"({elapsed / max(n_pages, 1):.2f}s/page)"
    )
    for entry in pdf_info:
        page_idx = int(entry.get("page_idx", 0))
        entry["_source_page"] = chunk_start + page_idx
    return pdf_info


def run_mineru_ocr(cfg: ExtractConfig) -> list[dict[str, Any]]:
    """Run MinerU OCR via the ``mineru`` CLI, processing in chunks.

    Uses ``-m ocr -l latin`` which ensures PaddleOCR correctly recognises
    diacritics like š, ṣ, ṭ.  Large page ranges are split into chunks
    of ``cfg.ocr_chunk_size`` pages to avoid OOM / silent crashes.
    """
    _ensure_local_mineru_pipeline_assets(cfg.ocr_lang)

    start = cfg.source_start_page
    end = cfg.source_end_page
    chunks = []
    s = start
    while s <= end:
        ce = min(s + cfg.ocr_chunk_size - 1, end)
        chunks.append((s, ce))
        s = ce + 1

    total_pages = end - start + 1
    print(
        f"📄 MinerU OCR: {len(chunks)} chunk(s), "
        f"pages {start}-{end} ({total_pages} pages)"
    )
    t_total = time.time()

    all_pdf_info: list[dict[str, Any]] = []
    for ci, (cs, ce) in enumerate(chunks):
        print(f"\n--- Chunk {ci + 1}/{len(chunks)} ---")
        chunk_info = _run_mineru_cli_chunk(cfg.pdf_path, cs, ce, cfg)
        all_pdf_info.extend(chunk_info)

    elapsed_total = time.time() - t_total
    print(
        f"\n✅ MinerU OCR complete: {len(all_pdf_info)} pages in {elapsed_total:.1f}s "
        f"({elapsed_total / max(len(all_pdf_info), 1):.2f}s/page)"
    )

    # Cache combined middle.json for reuse
    cache_path = cfg.output_dir / "middle.json"
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"pdf_info": all_pdf_info}, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"💾 Cached middle.json: {cache_path}")
    return all_pdf_info


def _process_one_page(
    page_entry: dict[str, Any],
    source_page_num: int,
    cfg: ExtractConfig,
    page_entry_right: dict[str, Any] | None = None,
) -> tuple[int, int, list[Any], list[tuple[int, int, int, str, str]]]:
    """Extract aligned rows and records from a single page (thread-safe)."""
    aligned_rows = extract_aligned_rows(page_entry, source_page_num, cfg,
                                        page_entry_right=page_entry_right)
    n_rows = len(aligned_rows)
    if n_rows < cfg.min_row_pairs_per_page:
        return source_page_num, n_rows, aligned_rows, []
    records = assemble_records_split_entries(aligned_rows, source_page_num)
    return source_page_num, n_rows, aligned_rows, records


def _run_ocr_for_lang(cfg: ExtractConfig, lang: str, label: str) -> list[dict[str, Any]]:
    """Run MinerU OCR with a specific language and return pdf_info."""
    cfg_copy = ExtractConfig(**{f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__.values()})
    cfg_copy.ocr_lang = lang
    # Use separate cache file per language
    print(f"\n{'='*60}")
    print(f"🔤 OCR pass: {label} (lang={lang})")
    print(f"{'='*60}")
    pdf_info = run_mineru_ocr(cfg_copy)
    # Rename cached file to include lang
    cache_src = cfg.output_dir / "middle.json"
    cache_dst = cfg.output_dir / f"middle_{lang}.json"
    if cache_src.exists():
        cache_src.rename(cache_dst)
        print(f"💾 Renamed cache: {cache_dst}")
    return pdf_info


def _page_index_by_source(pdf_info: list[dict[str, Any]], cfg: ExtractConfig) -> dict[int, dict[str, Any]]:
    """Build {source_page_num: page_entry} lookup from pdf_info."""
    index: dict[int, dict[str, Any]] = {}
    for entry in pdf_info:
        if "_source_page" in entry:
            sp = int(entry["_source_page"])
        else:
            sp = cfg.source_start_page + int(entry.get("page_idx", 0))
        index[sp] = entry
    return index


def extract_mineru(cfg: ExtractConfig) -> tuple[list[Any], dict[str, Any]]:
    dual_ocr = bool(cfg.ocr_lang_right and cfg.ocr_lang_right != cfg.ocr_lang)

    # --- Obtain pdf_info: either run MinerU OCR or load from middle.json ---
    if cfg.run_ocr:
        if dual_ocr:
            pdf_info = _run_ocr_for_lang(cfg, cfg.ocr_lang, "left/transliteration")
            pdf_info_right = _run_ocr_for_lang(cfg, cfg.ocr_lang_right, "right/English")
        else:
            pdf_info = run_mineru_ocr(cfg)
            pdf_info_right = []
        middle_json_path_str = str(cfg.output_dir / f"middle_{cfg.ocr_lang}.json"
                                   if dual_ocr else cfg.output_dir / "middle.json")
    else:
        middle = json.loads(cfg.middle_json_path.read_text(encoding="utf-8"))
        pdf_info = middle.get("pdf_info") or []
        middle_json_path_str = str(cfg.middle_json_path.resolve())
        # Load right-column OCR if available
        if dual_ocr and cfg.middle_json_path_right.name:
            middle_r = json.loads(cfg.middle_json_path_right.read_text(encoding="utf-8"))
            pdf_info_right = middle_r.get("pdf_info") or []
        else:
            pdf_info_right = []

    right_index = _page_index_by_source(pdf_info_right, cfg) if pdf_info_right else {}
    if dual_ocr:
        print(f"🔀 Dual-language OCR: left={cfg.ocr_lang} ({len(pdf_info)} pages), "
              f"right={cfg.ocr_lang_right} ({len(right_index)} pages)")

    stats: dict[str, Any] = {
        "engine": "mineru_ocr_preproc_blocks",
        "pdf_path": str(cfg.pdf_path.resolve()),
        "middle_json_path": middle_json_path_str,
        "source_start_page": cfg.source_start_page,
        "source_end_page": cfg.source_end_page,
        "run_ocr": cfg.run_ocr,
        "dual_ocr": dual_ocr,
        "ocr_lang_left": cfg.ocr_lang,
        "ocr_lang_right": cfg.ocr_lang_right if dual_ocr else "",
        "pages_in_middle_json": len(pdf_info),
        "pages_scanned": 0,
        "pages_with_row_pairs": 0,
        "pages_with_kept_records": 0,
        "aligned_rows": 0,
        "raw_records": 0,
        "kept_records": 0,
        "cross_page_merges": 0,
        "reject_reasons": Counter(),
        "kept_by_page": {},
    }
    kept_records: list[Any] = []
    rejected_rows: list[dict[str, Any]] = []
    all_page_records: list[tuple[int, list[tuple[int, int, str, str]]]] = []

    # Build (page_entry, source_page_num, page_entry_right) tuples
    page_tasks: list[tuple[dict[str, Any], int, dict[str, Any] | None]] = []
    for page_entry in pdf_info:
        if "_source_page" in page_entry:
            source_page_num = int(page_entry["_source_page"])
        else:
            page_idx = int(page_entry.get("page_idx", 0))
            source_page_num = cfg.source_start_page + page_idx
        if source_page_num < cfg.source_start_page or source_page_num > cfg.source_end_page:
            continue
        pe_right = right_index.get(source_page_num)
        page_tasks.append((page_entry, source_page_num, pe_right))

    # Parallel extraction across pages
    n_workers = min(cfg.ocr_workers, len(page_tasks)) if page_tasks else 1
    t_extract = time.time()
    with ThreadPoolExecutor(max_workers=max(n_workers, 1)) as pool:
        futures = [
            pool.submit(_process_one_page, pe, spn, cfg, pe_right)
            for pe, spn, pe_right in page_tasks
        ]
        results = [f.result() for f in futures]

    for source_page_num, n_rows, aligned_rows, page_records in results:
        stats["pages_scanned"] += 1
        if n_rows < cfg.min_row_pairs_per_page:
            continue
        stats["pages_with_row_pairs"] += 1
        stats["aligned_rows"] += n_rows
        if page_records:
            all_page_records.append((source_page_num, page_records))

    print(f"  ⚡ Extraction: {len(page_tasks)} pages in {time.time() - t_extract:.1f}s ({n_workers} workers)")

    merged_records = _merge_cross_page_records(all_page_records)
    stats["cross_page_merges"] = sum(len(records) for _, records in all_page_records) - len(
        merged_records
    )

    kept_for_pages: Counter[int] = Counter()
    for page_num, row_start, row_end, akkadian, english in merged_records:
        stats["raw_records"] += 1

        # MinerU OCR post-processing: fix diacritics, line-break slashes,
        # and other common OCR artifacts before quality filtering.
        akkadian = fix_ocr_artifacts(normalize_whitespace(akkadian).strip())
        english = fix_translation_ocr_artifacts(normalize_whitespace(english).strip())

        reject_reason = reject_reason_for_candidate(
            raw_akkadian=akkadian,
            raw_english=english,
            clean_akkadian=akkadian,
            clean_english=english,
            cfg=cfg,
        )
        if reject_reason is not None:
            stats["reject_reasons"][reject_reason] += 1
            rejected_rows.append(
                {
                    "page": page_num,
                    "row_start_index": row_start,
                    "row_end_index": row_end,
                    "akkadian": akkadian,
                    "english": english,
                    "reason": reject_reason,
                }
            )
            continue

        char_ratio = nonspace_len(akkadian) / max(
            1, nonspace_len(english)
        )
        kept_records.append(
            CandidateRecord(
                page=page_num,
                row_start_index=row_start,
                row_end_index=row_end,
                raw_akkadian=akkadian,
                raw_english=english,
                akkadian=akkadian,
                english=english,
                translit_retention=1.0,
                translation_retention=1.0,
                clean_char_ratio=char_ratio,
            )
        )
        kept_for_pages[page_num] += 1

    stats["pages_with_kept_records"] = len(kept_for_pages)
    stats["kept_by_page"] = dict(kept_for_pages)
    stats["kept_records"] = len(kept_records)
    stats["reject_reasons"] = dict(stats["reject_reasons"])
    stats["rejected_preview_count"] = min(len(rejected_rows), cfg.preview_count)
    stats["rejected_preview"] = rejected_rows[: cfg.preview_count]
    return kept_records, stats




DEFAULT_AKT6_PAGE_RANGE_SPECS = [
    "AKT 6a.pdf:52-436",
    "AKT 6b.pdf:59-353",
    "AKT 6c.pdf:46-298",
    "AKT 6d.pdf:26-147",
    "AKT 6e.pdf:97-266",
]


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def ensure_mineru_runtime() -> None:
    try:
        importlib.import_module("mineru")
        return
    except Exception as exc:
        candidate = Path("/root/miniforge3/envs/mineru/bin/python")
        already_reexeced = os.environ.get("AKT6_MINERU_REEXECED") == "1"
        current_python = Path(sys.executable).resolve()
        if (
            candidate.exists()
            and current_python != candidate.resolve()
            and not already_reexeced
        ):
            log(
                "MinerU package missing in current interpreter; "
                f"re-executing with {candidate}"
            )
            env = os.environ.copy()
            env["AKT6_MINERU_REEXECED"] = "1"
            os.execvpe(
                str(candidate),
                [str(candidate), str(Path(__file__).resolve()), *sys.argv[1:]],
                env,
            )
        raise RuntimeError(
            "MinerU Python package is not importable in this environment. "
            f"Current interpreter: {current_python}. "
            f"Please run with {candidate}."
        ) from exc


@dataclass(frozen=True)
class PageRangeRule:
    pattern: str
    start_page: int
    end_page: int
    raw_spec: str


@dataclass(frozen=True)
class RenderedPage:
    source_page: int
    image_path: Path


@dataclass
class RunConfig:
    input_dir: Path = Path("/data/lsb/deep_past/data/old-assyrian")
    pdf_glob: str = "AKT 6a.pdf"
    output_root: Path = Path("/data/lsb/deep_past/output/akt6_mineru_ocr")
    page_range_specs: list[str] = field(
        default_factory=lambda: list(DEFAULT_AKT6_PAGE_RANGE_SPECS)
    )
    render_dpi: int = 300
    image_batch_size: int = 4
    ocr_lang: str = "latin"
    extract_workers: int = 4
    extract_line_group_y_tolerance: float = 30.0
    extract_column_split_ratio: float = 0.44
    extract_column_margin: float = 100.0
    extract_full_width_ratio: float = 0.75
    mineru_device: str = "auto"
    keep_rendered_pages: bool = False
    limit_pages_per_pdf: int = 0
    formula_enable: bool = False
    table_enable: bool = False


def parse_page_range_spec(spec: str) -> PageRangeRule:
    try:
        pattern, page_range = spec.rsplit(":", 1)
        start_str, end_str = page_range.split("-", 1)
        start_page = int(start_str)
        end_page = int(end_str)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --page-range-spec {spec!r}. Expected format: 'AKT 6a.pdf:52-436'."
        ) from exc
    if start_page < 1 or end_page < start_page:
        raise ValueError(f"Invalid page range in spec {spec!r}.")
    return PageRangeRule(
        pattern=pattern.strip(),
        start_page=start_page,
        end_page=end_page,
        raw_spec=spec,
    )


def resolve_selected_pages(
    pdf_path: Path,
    total_pages: int,
    rules: list[PageRangeRule],
    limit_pages_per_pdf: int,
) -> tuple[list[int], list[str]]:
    matched_specs: list[str] = []
    selected: set[int] = set()
    for rule in rules:
        if fnmatch(pdf_path.name, rule.pattern):
            matched_specs.append(rule.raw_spec)
            start_page = max(1, rule.start_page)
            end_page = min(total_pages, rule.end_page)
            selected.update(range(start_page, end_page + 1))
    if not selected:
        selected = set(range(1, total_pages + 1))
    page_numbers = sorted(selected)
    if limit_pages_per_pdf > 0:
        page_numbers = page_numbers[:limit_pages_per_pdf]
    return page_numbers, matched_specs


def render_pdf_pages(
    pdf_path: Path,
    page_numbers: list[int],
    render_dir: Path,
    dpi: int,
) -> list[RenderedPage]:
    render_dir.mkdir(parents=True, exist_ok=True)
    rendered_pages: list[RenderedPage] = []
    scale = dpi / 72.0
    total_pages = len(page_numbers)
    log(
        f"{pdf_path.name}: rendering {total_pages} pages to {render_dir} "
        f"(dpi={dpi})"
    )
    with fitz.open(str(pdf_path)) as doc:
        for idx, page_num in enumerate(page_numbers, start=1):
            page = doc[page_num - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), colorspace=fitz.csRGB)
            image_path = render_dir / f"page_{page_num:04d}.png"
            image_path.write_bytes(pix.tobytes("png"))
            rendered_pages.append(RenderedPage(source_page=page_num, image_path=image_path))
            if idx == 1 or idx == total_pages or idx % 10 == 0:
                log(
                    f"{pdf_path.name}: rendered {idx}/{total_pages} pages "
                    f"(source page {page_num})"
                )
    return rendered_pages


def resolve_mineru_device(requested_device: str) -> str:
    if not requested_device or requested_device == "auto":
        requested_device = "cuda"

    if not requested_device.startswith("cuda"):
        return requested_device

    try:
        import torch

        device_index = 0
        if ":" in requested_device:
            _, index_part = requested_device.split(":", 1)
            if index_part:
                device_index = int(index_part)
        if not torch.cuda.is_available():
            return "cpu"
        torch.cuda.get_device_properties(device_index)
        return requested_device
    except Exception:
        return "cpu"


def run_mineru_image_ocr(
    rendered_pages: list[RenderedPage],
    output_dir: Path,
    cfg: RunConfig,
) -> tuple[Path, dict[str, Any]]:
    if not rendered_pages:
        raise ValueError("No rendered pages to OCR.")

    resolved_device = resolve_mineru_device(cfg.mineru_device)
    os.environ["MINERU_MODEL_SOURCE"] = "local"
    os.environ["MINERU_DEVICE_MODE"] = resolved_device
    log(
        f"{output_dir.name}: starting MinerU image OCR "
        f"(pages={len(rendered_pages)}, batch_size={cfg.image_batch_size}, "
        f"device={resolved_device}, requested={cfg.mineru_device})"
    )

    log(f"{output_dir.name}: ensuring local MinerU assets for lang={cfg.ocr_lang}")
    model_root = _ensure_local_mineru_pipeline_assets(cfg.ocr_lang)
    log(f"{output_dir.name}: MinerU assets ready at {model_root}")

    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json
    from mineru.backend.pipeline.pipeline_analyze import batch_image_analyze
    from mineru.data.data_reader_writer import FileBasedDataWriter
    from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes, load_images_from_pdf_core
    import pypdfium2 as pdfium

    middle_json_path = output_dir / "middle.json"
    image_writer = FileBasedDataWriter(str(output_dir / "ocr_assets" / "images"))
    pdf_info: list[dict[str, Any]] = []
    started_at = time.time()

    for chunk_start in range(0, len(rendered_pages), cfg.image_batch_size):
        chunk = rendered_pages[chunk_start: chunk_start + cfg.image_batch_size]
        prepared: list[tuple[RenderedPage, Any, list[dict[str, Any]]]] = []
        images_with_extra_info = []
        chunk_pages = [page.source_page for page in chunk]
        log(
            f"{output_dir.name}: OCR chunk {chunk_start // cfg.image_batch_size + 1} "
            f"with source pages {chunk_pages}"
        )

        for rendered_page in chunk:
            image_bytes = rendered_page.image_path.read_bytes()
            pdf_bytes = images_bytes_to_pdf_bytes(image_bytes)
            images_list = load_images_from_pdf_core(pdf_bytes)
            if not images_list:
                continue
            pdf_doc = pdfium.PdfDocument(pdf_bytes)
            prepared.append((rendered_page, pdf_doc, images_list))
            images_with_extra_info.append((images_list[0]["img_pil"], True, cfg.ocr_lang))

        if not prepared:
            continue

        batch_results = batch_image_analyze(
            images_with_extra_info,
            formula_enable=cfg.formula_enable,
            table_enable=cfg.table_enable,
        )

        for (rendered_page, pdf_doc, images_list), result in zip(prepared, batch_results):
            page_image = images_list[0]["img_pil"]
            model_list = [{
                "layout_dets": result,
                "page_info": {
                    "page_no": 0,
                    "width": page_image.width,
                    "height": page_image.height,
                },
            }]
            middle_json = result_to_middle_json(
                model_list,
                images_list,
                pdf_doc,
                image_writer,
                lang=cfg.ocr_lang,
                ocr_enable=True,
                formula_enabled=cfg.formula_enable,
            )
            page_entry = middle_json["pdf_info"][0]
            page_entry["_source_page"] = rendered_page.source_page
            page_entry["_rendered_image"] = rendered_page.image_path.name
            pdf_info.append(page_entry)
            pdf_doc.close()

        processed = min(chunk_start + len(chunk), len(rendered_pages))
        log(
            f"{output_dir.name}: image OCR processed {processed}/{len(rendered_pages)} "
            f"rendered pages"
        )

    pdf_info.sort(key=lambda item: int(item.get("_source_page", 0)))
    middle_json = {
        "pdf_info": pdf_info,
        "_backend": "pipeline",
        "_input_mode": "rendered_page_images",
    }
    middle_json_path.write_text(
        json.dumps(middle_json, ensure_ascii=False),
        encoding="utf-8",
    )
    stats = {
        "input_mode": "rendered_page_images",
        "ocr_lang": cfg.ocr_lang,
        "render_dpi": cfg.render_dpi,
        "image_batch_size": cfg.image_batch_size,
        "rendered_pages": len(rendered_pages),
        "middle_json_pages": len(pdf_info),
        "image_ocr_elapsed_s": round(time.time() - started_at, 2),
        "mineru_model_root": str(model_root),
        "mineru_requested_device": cfg.mineru_device,
        "mineru_resolved_device": resolved_device,
    }
    return middle_json_path, stats


def write_merged_outputs(summaries: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    merged_csv_path = output_root / "train.csv"
    merged_meta_path = output_root / "pairs_metadata.jsonl"
    merged_rows = 0
    merged_meta = 0
    source_pdfs: list[str] = []

    with open(merged_csv_path, "w", encoding="utf-8", newline="") as csv_fh, \
         open(merged_meta_path, "w", encoding="utf-8") as meta_fh:
        writer = csv.DictWriter(
            csv_fh,
            fieldnames=["oare_id", "transliteration", "translation"],
        )
        writer.writeheader()

        for summary in summaries:
            pdf_name = summary["pdf_name"]
            train_csv_path = Path(summary["train_csv"])
            meta_path = Path(summary["output_dir"]) / "pairs_metadata.jsonl"
            if not train_csv_path.exists():
                log(f"{pdf_name}: skipping merged CSV append, missing {train_csv_path}")
                continue

            with open(train_csv_path, encoding="utf-8", newline="") as part_fh:
                reader = csv.DictReader(part_fh)
                for row in reader:
                    writer.writerow(
                        {
                            "oare_id": row.get("oare_id", ""),
                            "transliteration": row.get("transliteration", ""),
                            "translation": row.get("translation", ""),
                        }
                    )
                    merged_rows += 1

            if meta_path.exists():
                with open(meta_path, encoding="utf-8") as part_meta_fh:
                    for line in part_meta_fh:
                        if not line.strip():
                            continue
                        meta_fh.write(line)
                        merged_meta += 1

            source_pdfs.append(pdf_name)

    log(
        f"Merged {len(source_pdfs)} PDFs into {merged_csv_path} "
        f"({merged_rows} rows, {merged_meta} metadata rows)"
    )
    return {
        "merged_train_csv": str(merged_csv_path.resolve()),
        "merged_pairs_metadata_jsonl": str(merged_meta_path.resolve()),
        "merged_pdf_count": len(source_pdfs),
        "merged_rows": merged_rows,
        "merged_metadata_rows": merged_meta,
        "merged_source_pdfs": source_pdfs,
    }


def process_one_pdf(pdf_path: Path, cfg: RunConfig, rules: list[PageRangeRule]) -> dict[str, Any]:
    with fitz.open(str(pdf_path)) as doc:
        total_pages = len(doc)

    selected_pages, matched_specs = resolve_selected_pages(
        pdf_path=pdf_path,
        total_pages=total_pages,
        rules=rules,
        limit_pages_per_pdf=cfg.limit_pages_per_pdf,
    )
    if not selected_pages:
        raise ValueError(f"No pages selected for {pdf_path.name}.")

    output_dir = cfg.output_root / pdf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    page_min = min(selected_pages)
    page_max = max(selected_pages)
    log(
        f"Starting {pdf_path.name}: total_pages={total_pages}, "
        f"selected_pages={len(selected_pages)} ({page_min}-{page_max})"
    )
    if matched_specs:
        log(f"{pdf_path.name}: matched page specs {matched_specs}")
    else:
        log(f"{pdf_path.name}: no page spec matched, using all pages")

    render_started_at = time.time()
    if cfg.keep_rendered_pages:
        render_dir = output_dir / "rendered_pages"
        rendered_pages = render_pdf_pages(pdf_path, selected_pages, render_dir, cfg.render_dpi)
    else:
        with tempfile.TemporaryDirectory(prefix=f"{pdf_path.stem}_render_") as tmpdir:
            render_dir = Path(tmpdir)
            rendered_pages = render_pdf_pages(pdf_path, selected_pages, render_dir, cfg.render_dpi)
            middle_json_path, image_stats = run_mineru_image_ocr(rendered_pages, output_dir, cfg)
    if cfg.keep_rendered_pages:
        middle_json_path, image_stats = run_mineru_image_ocr(rendered_pages, output_dir, cfg)

    render_elapsed_s = round(time.time() - render_started_at, 2)
    log(
        f"{pdf_path.name}: rendering + image OCR finished in {render_elapsed_s}s, "
        f"starting pair extraction"
    )
    extract_cfg = ExtractConfig(
        pdf_path=pdf_path,
        middle_json_path=middle_json_path,
        output_dir=output_dir,
        source_start_page=min(selected_pages),
        source_end_page=max(selected_pages),
        run_ocr=False,
        ocr_lang=cfg.ocr_lang,
        ocr_workers=cfg.extract_workers,
        line_group_y_tolerance=cfg.extract_line_group_y_tolerance,
        column_split_ratio=cfg.extract_column_split_ratio,
        column_margin=cfg.extract_column_margin,
        full_width_ratio=cfg.extract_full_width_ratio,
        mineru_device=cfg.mineru_device,
    )
    records, stats = extract_mineru(extract_cfg)
    log(f"{pdf_path.name}: pair extraction finished, writing outputs")
    stats["engine"] = "mineru_rendered_page_images_preproc_blocks"
    stats["render_dpi"] = cfg.render_dpi
    stats["image_batch_size"] = cfg.image_batch_size
    stats["extract_line_group_y_tolerance"] = cfg.extract_line_group_y_tolerance
    stats["render_elapsed_s"] = render_elapsed_s
    stats["page_range_specs"] = matched_specs
    stats["selected_page_count"] = len(selected_pages)
    stats["selected_page_min"] = min(selected_pages)
    stats["selected_page_max"] = max(selected_pages)
    stats["keep_rendered_pages"] = cfg.keep_rendered_pages
    stats.update(image_stats)
    save_outputs(records, stats, extract_cfg)

    summary = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path.resolve()),
        "output_dir": str(output_dir.resolve()),
        "selected_page_count": len(selected_pages),
        "selected_page_min": min(selected_pages),
        "selected_page_max": max(selected_pages),
        "matched_page_range_specs": matched_specs,
        "kept_records": stats["kept_records"],
        "raw_records": stats["raw_records"],
        "cross_page_merges": stats["cross_page_merges"],
        "middle_json_path": str(middle_json_path.resolve()),
        "train_csv": str((output_dir / "train.csv").resolve()),
        "report_json": str((output_dir / "report.json").resolve()),
    }
    log(
        f"Completed {pdf_path.name}: kept {stats['kept_records']} / raw "
        f"{stats['raw_records']} (pages {min(selected_pages)}-{max(selected_pages)})"
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render AKT 6*.pdf pages to images, run MinerU OCR on the rendered images, "
            "then extract aligned Akkadian-English training pairs."
        )
    )
    parser.add_argument("--input-dir", type=Path, default=RunConfig.input_dir)
    parser.add_argument("--pdf-glob", type=str, default=RunConfig.pdf_glob)
    parser.add_argument("--output-root", type=Path, default=RunConfig.output_root)
    parser.add_argument(
        "--page-range-spec",
        action="append",
        default=None,
        help="Optional page restriction, e.g. 'AKT 6a.pdf:52-436'. Can be repeated.",
    )
    parser.add_argument("--render-dpi", type=int, default=RunConfig.render_dpi)
    parser.add_argument("--image-batch-size", type=int, default=RunConfig.image_batch_size)
    parser.add_argument("--ocr-lang", type=str, default=RunConfig.ocr_lang)
    parser.add_argument("--extract-workers", type=int, default=RunConfig.extract_workers)
    parser.add_argument(
        "--extract-line-group-y-tolerance",
        type=float,
        default=RunConfig.extract_line_group_y_tolerance,
        help="Y-axis tolerance for pairing left/right OCR lines during extraction.",
    )
    parser.add_argument(
        "--extract-column-split-ratio",
        type=float,
        default=RunConfig.extract_column_split_ratio,
        help="Ratio of page width for left/right column split (default 0.44 for 300dpi rendered images).",
    )
    parser.add_argument(
        "--extract-column-margin",
        type=float,
        default=RunConfig.extract_column_margin,
        help="Column margin in pixels for left/right assignment (default 100 for 300dpi).",
    )
    parser.add_argument(
        "--extract-full-width-ratio",
        type=float,
        default=RunConfig.extract_full_width_ratio,
        help="Width ratio threshold for skipping full-width lines (default 0.75).",
    )
    parser.add_argument("--mineru-device", type=str, default=RunConfig.mineru_device)
    parser.add_argument("--keep-rendered-pages", action="store_true")
    parser.add_argument(
        "--limit-pages-per-pdf",
        type=int,
        default=RunConfig.limit_pages_per_pdf,
        help="For smoke tests / debugging. 0 means no limit.",
    )
    return parser.parse_args()


def build_config(args: argparse.Namespace) -> RunConfig:
    page_range_specs = list(args.page_range_spec) if args.page_range_spec else list(DEFAULT_AKT6_PAGE_RANGE_SPECS)
    return RunConfig(
        input_dir=args.input_dir,
        pdf_glob=args.pdf_glob,
        output_root=args.output_root,
        page_range_specs=page_range_specs,
        render_dpi=args.render_dpi,
        image_batch_size=args.image_batch_size,
        ocr_lang=args.ocr_lang,
        extract_workers=args.extract_workers,
        extract_line_group_y_tolerance=args.extract_line_group_y_tolerance,
        extract_column_split_ratio=args.extract_column_split_ratio,
        extract_column_margin=args.extract_column_margin,
        extract_full_width_ratio=args.extract_full_width_ratio,
        mineru_device=args.mineru_device,
        keep_rendered_pages=args.keep_rendered_pages,
        limit_pages_per_pdf=args.limit_pages_per_pdf,
    )


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    rules = [parse_page_range_spec(spec) for spec in cfg.page_range_specs]
    ensure_mineru_runtime()

    pdf_files = sorted(cfg.input_dir.glob(cfg.pdf_glob))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDFs matched {cfg.pdf_glob!r} under {cfg.input_dir}"
        )

    cfg.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    matched_specs: set[str] = set()
    log(
        f"AKT6 image OCR start: input_dir={cfg.input_dir}, pdf_glob={cfg.pdf_glob}, "
        f"output_root={cfg.output_root}"
    )
    log(
        f"Found {len(pdf_files)} PDFs, render_dpi={cfg.render_dpi}, "
        f"image_batch_size={cfg.image_batch_size}, extract_workers={cfg.extract_workers}, "
        f"extract_line_group_y_tolerance={cfg.extract_line_group_y_tolerance}"
    )
    log(f"Using page range specs: {cfg.page_range_specs}")
    log(f"MinerU runtime OK: python={Path(sys.executable).resolve()}")
    log(f"Preflight MinerU assets for lang={cfg.ocr_lang}")
    model_root = _ensure_local_mineru_pipeline_assets(cfg.ocr_lang)
    log(f"Preflight MinerU assets ready at {model_root}")

    started_at = time.time()
    for idx, pdf_path in enumerate(pdf_files, start=1):
        log(f"PDF {idx}/{len(pdf_files)}: {pdf_path.name}")
        summary = process_one_pdf(pdf_path, cfg, rules)
        summaries.append(summary)
        matched_specs.update(summary["matched_page_range_specs"])

    merged_summary = write_merged_outputs(summaries, cfg.output_root)

    unmatched_specs = sorted(set(cfg.page_range_specs) - matched_specs)
    aggregate = {
        "input_dir": str(cfg.input_dir.resolve()),
        "pdf_glob": cfg.pdf_glob,
        "pdf_count": len(pdf_files),
        "elapsed_s": round(time.time() - started_at, 2),
        "page_range_specs": cfg.page_range_specs,
        "unmatched_page_range_specs": unmatched_specs,
        "pdf_summaries": summaries,
    }
    aggregate.update(merged_summary)
    summary_path = cfg.output_root / "summary.json"
    summary_path.write_text(
        json.dumps(aggregate, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Summary written to {summary_path}")


if __name__ == "__main__":
    main()
