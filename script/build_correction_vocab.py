#!/usr/bin/env python3
"""
Build a spelling correction vocabulary from training data + eBL lexicon.

Combines:
1. Surface forms from training data (with frequency counts)
2. Canonical forms from OA_Lexicon_eBL.csv (proper names, words, geographical names)

Provides:
- fix_encoding_corruption(): reverse deterministic character encoding damage
- suggest_corrections(): vocabulary-validated edit-distance corrections for remaining unknowns

Usage:
    python build_correction_vocab.py                    # build vocab + correction dict
    python build_correction_vocab.py --test-file test.csv  # also test corrections on file
"""

import csv
import json
import re
import sys
import collections
from pathlib import Path
from typing import Optional


# ============================================================================
# Deterministic encoding corruption patterns (reverse mapping)
# ============================================================================
# These are systematic character substitutions found in corrupted test data:
#   š (U+0161) → a   (caron stripped, then s→a byte damage)
#   ṭ (U+1E6D) → m   (underdot stripped, t→m byte damage)
#   - (hyphen)  → „   (U+201E double low-9 quotation mark)
#   h           → +   (in transliteration context)
#   … (U+2026 ellipsis) inserted spuriously


def is_corrupted_row(text: str) -> bool:
    """Detect if a transliteration row has encoding corruption markers."""
    markers = [
        '\u201e',   # „ (corrupted hyphen)
        '\u2026',   # … (spurious ellipsis)
        'mup-p',    # corrupted ṭup-p
        'me-+e',    # corrupted me-he
    ]
    return any(m in text for m in markers)


def fix_encoding_corruption(text: str) -> str:
    """
    Reverse deterministic encoding corruption in transliteration text.

    Only applies fixes that are 100% safe (deterministic byte-level damage).
    For ambiguous š→a reversals, use suggest_corrections() with vocabulary.
    """
    if not isinstance(text, str):
        return ""

    # 1. Fix corrupted hyphens: „ (U+201E) → -
    text = text.replace('\u201e', '-')
    # Clean up double hyphens created by „ adjacent to existing hyphens
    text = re.sub(r'-{2,}', '-', text)

    # 2. Fix spurious ellipsis: … (U+2026) → remove
    text = text.replace('\u2026', '')

    # 3. Fix ṭ→m corruption: mup-p → ṭup-p (all morphological variants)
    # This is safe because "mup-p" never occurs in legitimate Akkadian
    text = re.sub(r'\bmup-p', 'ṭup-p', text)

    # 4. Fix h→+ corruption: me-+e-er → me-he-er
    text = re.sub(r'(?<=[a-záàéèíìúù])-\+(?=[a-záàéèíìúù])', '-h', text)

    # 5. Fix š→a corruption at obvious positions (standalone known words)
    # „aa” (corrupted „ša”) when standalone or before common suffixes
    text = re.sub(r'(?<![a-záàéèíìúùA-ZÁÀÉÈÍÌÚÙ])aa(?![a-záàéèíìúùA-Z])', 'ša', text)

    # 5. Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


# ============================================================================
# Vocabulary-based š→a correction
# ============================================================================

# Positions where a→š substitution is linguistically valid:
# š always precedes a vowel in Akkadian syllabic writing
_VOWELS = set('aáàeéèiíìuúùAÁÀEÉÈIÍÌUÚÙ')

# Patterns that strongly suggest š→a corruption (a followed by vowel in
# positions where the resulting syllable matches known š-syllables)
_SHA_CORRUPTION_RE = re.compile(
    r'(?<![a-záàéèíìúùA-ZÁÀÉÈÍÌÚÙ])'  # not preceded by a letter (word/syllable start)
    r'a([aáàeéèiíìuúù])'               # 'a' + vowel
)


def _generate_sha_candidates(token: str) -> list[str]:
    """
    Generate candidate corrections by trying a→š at all positions.

    The corruption replaces every š with a regardless of position.
    We rely on vocabulary validation to prevent false positives.

    Returns list of candidate strings.
    """
    candidates = []

    # Find ALL positions where 'a' could be a corrupted 'š'
    positions = [i for i, ch in enumerate(token) if ch == 'a']

    if not positions:
        return []

    # Generate candidates for single substitutions
    for pos in positions:
        candidate = token[:pos] + 'š' + token[pos + 1:]
        candidates.append(candidate)

    # Generate candidates for double substitutions (e.g., au-um-au → šu-um-šu)
    if len(positions) >= 2 and len(positions) <= 8:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                chars = list(token)
                chars[positions[i]] = 'š'
                chars[positions[j]] = 'š'
                candidates.append(''.join(chars))

    # Triple substitutions for heavily corrupted tokens (cap at 6 positions)
    if 3 <= len(positions) <= 6:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                for k in range(j + 1, len(positions)):
                    chars = list(token)
                    chars[positions[i]] = 'š'
                    chars[positions[j]] = 'š'
                    chars[positions[k]] = 'š'
                    candidates.append(''.join(chars))

    return candidates


def build_combined_vocabulary(
    train_csv: str,
    lexicon_csv: str,
) -> dict[str, int]:
    """
    Build combined vocabulary from training data and eBL lexicon.

    Returns dict mapping token → frequency (eBL tokens get base frequency 1).
    """
    vocab: dict[str, int] = collections.Counter()

    # 1. Training data tokens (with frequency)
    for csv_entry in train_csv.split(","):
        csv_path = csv_entry.split(":")[0]  # Remove optional weight suffix
        if not csv_path.strip():
            continue
        try:
            with open(csv_path, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    src = row.get('transliteration', row.get('input_text', ''))
                    for tok in re.split(r'[\s:]+', src):
                        tok = tok.strip('.,;!?()[]{}"\' ')
                        if tok and not re.match(r'^\d+[.,/]?\d*$', tok):
                            vocab[tok] += 1
        except Exception as e:
            print(f"Warning: Failed to read {csv_path} for vocab: {e}")

    # 2. eBL lexicon forms (base frequency 1)
    with open(lexicon_csv, encoding='utf-8') as f:
        for row in csv.DictReader(f):
            form = row.get('form', '').strip()
            if form:
                for tok in re.split(r'[\s]+', form):
                    tok = tok.strip()
                    if tok and tok not in vocab:
                        vocab[tok] = 1

    return dict(vocab)


def suggest_corrections(
    text: str,
    vocab: dict[str, int],
    *,
    max_edit_distance: int = 2,
    min_token_len: int = 4,
) -> tuple[str, list[dict]]:
    """
    Suggest vocabulary-validated corrections for unknown tokens.

    Returns (corrected_text, list_of_corrections).
    Each correction is {'original': str, 'corrected': str, 'method': str}.
    """
    corrections = []
    tokens = re.split(r'(\s+)', text)  # preserve whitespace
    result = []

    for token in tokens:
        if not token.strip() or token.isspace():
            result.append(token)
            continue

        # Strip punctuation for lookup, but preserve it for output
        stripped = token.strip('.,;!?()[]{}"\' ')
        prefix = token[:len(token) - len(token.lstrip('.,;!?()[]{}"\' '))]
        suffix = token[len(token.rstrip('.,;!?()[]{}"\' ')):]

        if not stripped or stripped in vocab or re.match(r'^\d+[.,/]?\d*$', stripped):
            result.append(token)
            continue

        # Skip tokens with parentheses (handled by preprocess_transliteration later)
        if '(' in stripped or ')' in stripped:
            result.append(token)
            continue

        # Try š→a reversal candidates (no min length - even 2-char aa→ša is valid)
        best_candidate = None
        best_freq = 0

        candidates = _generate_sha_candidates(stripped)
        for candidate in candidates:
            if candidate in vocab and vocab[candidate] > best_freq:
                best_candidate = candidate
                best_freq = vocab[candidate]

        if best_candidate:
            corrections.append({
                'original': stripped,
                'corrected': best_candidate,
                'method': 'sha_reversal',
                'frequency': best_freq,
            })
            result.append(prefix + best_candidate + suffix)
            continue

        # Try simple edit distance (conservative: only for longer tokens)
        if max_edit_distance > 0 and len(stripped) >= min_token_len:
            # Skip tokens with parentheses or special chars
            if re.search(r'[(){}\[\]]', stripped):
                result.append(token)
                continue
            close_match = _find_closest_vocab(stripped, vocab, max_edit_distance)
            if close_match:
                corrections.append({
                    'original': stripped,
                    'corrected': close_match,
                    'method': 'edit_distance',
                    'frequency': vocab[close_match],
                })
                result.append(prefix + close_match + suffix)
                continue

        result.append(token)

    return ''.join(result), corrections


def _edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)

    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def _find_closest_vocab(
    token: str,
    vocab: dict[str, int],
    max_distance: int,
) -> Optional[str]:
    """Find closest vocabulary word within max_distance edits."""
    best = None
    best_dist = max_distance + 1
    best_freq = 0

    # Quick length filter
    for word, freq in vocab.items():
        if abs(len(word) - len(token)) > max_distance:
            continue
        dist = _edit_distance(token, word)
        if dist <= max_distance and (dist < best_dist or
                                      (dist == best_dist and freq > best_freq)):
            best = word
            best_dist = dist
            best_freq = freq

    return best


def correct_transliteration(
    text: str,
    vocab: dict[str, int],
    *,
    max_edit_distance: int = 1,
) -> tuple[str, list[dict]]:
    """
    Full correction pipeline for a single transliteration.

    1. Apply deterministic encoding corruption fixes
    2. Apply vocabulary-validated š→a reversal
    3. Apply conservative edit-distance corrections

    Returns (corrected_text, list_of_corrections).
    """
    all_corrections = []

    # Step 1: Deterministic fixes
    if is_corrupted_row(text):
        original = text
        text = fix_encoding_corruption(text)
        if text != original:
            all_corrections.append({
                'original': original,
                'corrected': text,
                'method': 'encoding_fix',
            })

    # Step 2: Vocabulary-validated corrections
    text, vocab_corrections = suggest_corrections(
        text, vocab,
        max_edit_distance=max_edit_distance,
        min_token_len=4,
    )
    all_corrections.extend(vocab_corrections)

    return text, all_corrections


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--train-csv',
        default='/data/lsb/deep_past/train.csv',
        help='Training data CSV',
    )
    parser.add_argument(
        '--lexicon-csv',
        default='/data/lsb/deep_past/data/OA_Lexicon_eBL.csv',
        help='eBL lexicon CSV',
    )
    parser.add_argument(
        '--output-vocab',
        default='/data/lsb/deep_past/data/correction_vocab.json',
        help='Output vocabulary JSON',
    )
    parser.add_argument(
        '--test-file',
        default=None,
        help='Test CSV to apply corrections to',
    )
    parser.add_argument(
        '--max-edit-distance',
        type=int,
        default=1,
        help='Max edit distance for vocabulary-based corrections',
    )
    args = parser.parse_args()

    # Build vocabulary
    print("Building combined vocabulary...")
    vocab = build_combined_vocabulary(args.train_csv, args.lexicon_csv)
    print(f"  Combined vocabulary: {len(vocab)} unique tokens")
    print(f"  Top-10 by frequency:")
    for tok, freq in sorted(vocab.items(), key=lambda x: -x[1])[:10]:
        print(f"    {freq:5d}x  {tok}")

    # Save vocabulary
    output_path = Path(args.output_vocab)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab, f, ensure_ascii=False, indent=0)
    print(f"\n  Saved to {output_path} ({output_path.stat().st_size // 1024}KB)")

    # Test corrections
    if args.test_file:
        print(f"\nTesting corrections on {args.test_file}...")
        with open(args.test_file, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                text = row.get('transliteration', '')
                row_id = row.get('id', '?')

                corrected, corrections = correct_transliteration(
                    text, vocab,
                    max_edit_distance=args.max_edit_distance,
                )

                print(f"\n  --- Row {row_id} ---")
                print(f"  Original:  {text[:120]}")
                print(f"  Corrected: {corrected[:120]}")
                if corrections:
                    print(f"  Fixes ({len(corrections)}):")
                    for c in corrections:
                        if c['method'] == 'encoding_fix':
                            print(f"    [encoding] applied deterministic fixes")
                        else:
                            print(f"    [{c['method']}] {c['original']} → {c['corrected']} (freq={c.get('frequency', '?')})")
    else:
        # Auto-detect test file
        test_path = Path(args.train_csv).parent / 'test.csv'
        if not test_path.exists():
            test_path = Path('/data/lsb/deep_past/data/test.csv')
        if test_path.exists():
            print(f"\nAuto-testing on {test_path}...")
            with open(test_path, encoding='utf-8') as f:
                for row in csv.DictReader(f):
                    text = row.get('transliteration', '')
                    row_id = row.get('id', '?')
                    corrected, corrections = correct_transliteration(
                        text, vocab, max_edit_distance=args.max_edit_distance,
                    )
                    print(f"\n  --- Row {row_id} ---")
                    print(f"  Original:  {text}")
                    print(f"  Corrected: {corrected}")
                    if corrections:
                        for c in corrections:
                            if c['method'] == 'encoding_fix':
                                print(f"    [encoding] deterministic fixes applied")
                            else:
                                print(f"    [{c['method']}] {c['original']} → {c['corrected']} (freq={c.get('frequency', '?')})")


if __name__ == '__main__':
    main()
