#!/usr/bin/env python3
"""
Build a strict Sumerogram prompt glossary from OA_Lexicon_eBL.csv.

Output format:
    {
      "É.GAL": ["ekallu", "palace"],
      "KÙ.AN": ["amuttu", "meteoric iron"]
    }

Field meanings:
- item[0]: normalized Akkadian reading / lemma (metadata only)
- item[1]: English gloss used for prompt injection

Important:
- Only tokens with an explicit English gloss are exported.
- We do not fall back to the Akkadian reading as a fake English hint.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path


_SUMEROGRAM_TOKEN_RE = re.compile(
    r"(?<![a-záàéèíìúù])"
    r"([A-ZÁÀÉÈÍÌÚÙŠṢṬḪ][A-ZÁÀÉÈÍÌÚÙŠṢṬḪ₀-₉0-9.]+)"
    r"(?![a-záàéèíìúù])"
)
_SUBSCRIPT_TO_ASCII = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")

# Conservative hand-authored glosses for prompt injection.
# If a token is not here, it should not be injected.
_GLOSS_OVERRIDES = {
    "AN.NA": "tin",
    "ANŠE": "donkey",
    "ANŠE.ḪI": "donkeys",
    "DAM.GÀR": "merchant",
    "DUB": "tablet",
    "DUB.SAR": "scribe",
    "DUMU": "son",
    "É.GAL": "palace",
    "GAL": "great/chief",
    "ILLAT": "caravan",
    "IR": "slave",
    "KIŠIB": "seal",
    "KÙ.AN": "meteoric iron",
    "KÙ.BABBAR": "silver",
    "KÙ.GI": "gold",
    "LUGAL": "king",
    "SIG5": "good",
    "TÚG": "textile",
    "TÚG.ḪI": "textiles",
    "URUDU": "copper",
    "ḪA.LÁ": "share",
    "ÌR": "slave",
}


def _normalize_token(token: str) -> str:
    return str(token or "").strip().translate(_SUBSCRIPT_TO_ASCII)


def _pick_best(counter: Counter[str]) -> str:
    if not counter:
        return ""
    return counter.most_common(1)[0][0]


def build_sumerogram_glossary(
    lexicon_csv: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, list[str]]:
    lexicon_csv = Path(lexicon_csv)
    if not lexicon_csv.exists():
        raise FileNotFoundError(f"Lexicon CSV not found: {lexicon_csv}")

    stats: dict[str, dict[str, Counter[str]]] = {}
    with lexicon_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            row_type = str(row.get("type") or "").strip()
            if row_type in {"PN", "GN"}:
                continue

            form = str(row.get("form") or "").strip()
            if not form:
                continue

            tokens = {
                _normalize_token(tok)
                for tok in _SUMEROGRAM_TOKEN_RE.findall(form)
                if tok
            }
            if not tokens:
                continue

            lexeme = str(row.get("lexeme") or "").strip()
            norm = str(row.get("norm") or "").strip()
            for token in tokens:
                if len(token) < 2:
                    continue
                token_stats = stats.setdefault(
                    token,
                    {"lexeme": Counter(), "norm": Counter()},
                )
                if lexeme:
                    token_stats["lexeme"][lexeme] += 1
                if norm:
                    token_stats["norm"][norm] += 1

    glossary: dict[str, list[str]] = {}
    for token in sorted(stats):
        base_token = re.sub(r"[0-9]+$", "", token)
        gloss = _GLOSS_OVERRIDES.get(token) or _GLOSS_OVERRIDES.get(base_token)
        if not gloss:
            continue
        reading = _pick_best(stats[token]["lexeme"]) or _pick_best(stats[token]["norm"])
        if not reading:
            continue
        glossary[token] = [reading, gloss]

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as fh:
            json.dump(glossary, fh, ensure_ascii=False, indent=2, sort_keys=True)

    return glossary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Sumerogram prompt glossary.")
    parser.add_argument(
        "--lexicon-csv",
        default="/data/lsb/deep_past/data/OA_Lexicon_eBL.csv",
        help="Path to OA_Lexicon_eBL.csv",
    )
    parser.add_argument(
        "--output",
        default="/data/lsb/deep_past/data/sumerogram_glossary.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    glossary = build_sumerogram_glossary(args.lexicon_csv, args.output)
    print(f"Built {len(glossary)} Sumerogram glossary entries -> {args.output}")


if __name__ == "__main__":
    main()
