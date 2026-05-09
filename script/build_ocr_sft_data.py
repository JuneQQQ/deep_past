#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


TRANSLATION_SYSTEM_PROMPT = (
    "You are an expert Assyriologist. Translate the following Akkadian text into English."
)
OCR_SYSTEM_PROMPT = (
    "You are an OCR correction expert. Fix the spelling and OCR errors in the following "
    "noisy Akkadian text. Output ONLY the clean Akkadian text."
)
SEGMENT_SYSTEM_PROMPT = (
    "You are an expert linguist. Segment the following long Akkadian text into logical, "
    "independent sentences. Output a JSON list of strings."
)


@dataclass
class Config:
    train_clean_csv: Path = Path("/data/lsb/deep_past/data/train_clean.csv")
    train_csv: Path = Path("/data/lsb/deep_past/data/train.csv")
    sentence_alignment_csv: Path = Path("/data/lsb/deep_past/data/Sentences_Oare_FirstWord_LinNum.csv")
    output_dir: Path = Path("/data/lsb/deep_past/data/sft_ocr_multitask")
    random_seed: int = 42

    @property
    def output_jsonl(self) -> Path:
        return self.output_dir / "sft_ocr_multitask_standard.jsonl"

    @property
    def summary_json(self) -> Path:
        return self.output_dir / "sft_ocr_multitask_summary.json"


OCR_CHAR_SUBS: list[tuple[str, str]] = [
    ("ṭ", "m"),
    ("ṣ", "s"),
    ("š", "s"),
    ("ḫ", "h"),
    ("ú", "u"),
    ("ù", "u"),
    ("í", "i"),
    ("ì", "i"),
    ("á", "a"),
    ("à", "a"),
    ("é", "e"),
    ("è", "e"),
    ("₄", "4"),
]
TOKEN_SUBS: list[tuple[str, str]] = [
    ("ṭup-pì", "mup-pì"),
    ("ṭup-pu", "mup-pu"),
    ("KÙ.BABBAR", "KÙ.B."),
    ("šu-ma", "su-ma"),
    ("a-na", "ana"),
]


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def stable_rng(seed_text: str, base_seed: int) -> random.Random:
    digest = hashlib.sha256(f"{base_seed}:{seed_text}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


def load_base_rows(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    clean_df = pd.read_csv(cfg.train_clean_csv)
    raw_df = pd.read_csv(cfg.train_csv)

    clean_df = clean_df.copy()
    raw_df = raw_df.copy()

    clean_df["transliteration"] = clean_df["transliteration"].astype(str).apply(normalize_space)
    clean_df["translation"] = clean_df["translation"].astype(str).apply(normalize_space)
    raw_df["transliteration"] = raw_df["transliteration"].astype(str).apply(normalize_space)
    raw_df["translation"] = raw_df["translation"].astype(str).apply(normalize_space)

    clean_ids = set(clean_df["oare_id"].astype(str))
    missing_raw = raw_df[~raw_df["oare_id"].astype(str).isin(clean_ids)].copy()
    if "data_source" not in missing_raw.columns:
        missing_raw["data_source"] = "official_raw"
    if "uncertainty" not in missing_raw.columns:
        missing_raw["uncertainty"] = ""

    base_df = pd.concat(
        [
            clean_df[["oare_id", "transliteration", "translation", "data_source", "uncertainty"]],
            missing_raw[["oare_id", "transliteration", "translation", "data_source", "uncertainty"]],
        ],
        ignore_index=True,
    )
    return clean_df, base_df


def make_translation_examples(source_df: pd.DataFrame, target_count: int, seed: int) -> list[dict]:
    mask = (
        (source_df["transliteration"].astype(str) != "")
        & (source_df["translation"].astype(str) != "")
        & (~source_df["transliteration"].astype(str).str.contains("<gap>", na=False))
        & (~source_df["translation"].astype(str).str.contains("<gap>", na=False))
    )
    if "uncertainty" in source_df.columns:
        mask &= source_df["uncertainty"].fillna("").astype(str).eq("")

    rows = source_df[
        mask
    ].to_dict(orient="records")
    rng = random.Random(seed)
    examples: list[dict] = []
    for idx in range(target_count):
        row = rows[idx] if idx < len(rows) else rng.choice(rows)
        examples.append(
            {
                "task": "translation",
                "source_oare_ids": [str(row["oare_id"])],
                "system_prompt": TRANSLATION_SYSTEM_PROMPT,
                "input_text": f"Input:\n{row['transliteration']}",
                "target_text": str(row["translation"]),
            }
        )
    return examples


def mutate_token(token: str, rng: random.Random) -> str:
    original = token
    for clean, noisy in TOKEN_SUBS:
        if clean in token and rng.random() < 0.4:
            token = token.replace(clean, noisy)

    chars = list(token)
    for idx, ch in enumerate(chars):
        if rng.random() < 0.14:
            for src, dst in OCR_CHAR_SUBS:
                if ch == src:
                    chars[idx] = dst
                    break
    token = "".join(chars)

    if "-" in token and rng.random() < 0.75:
        parts = token.split("-")
        if len(parts) >= 2:
            join_style = rng.choice(["space", "drop"])
            token = (" " if join_style == "space" else "").join(parts)

    if token == original and rng.random() < 0.25 and len(token) >= 4:
        cut = rng.randrange(1, len(token) - 1)
        token = token[:cut] + " " + token[cut:]

    return token


def inject_ocr_noise(clean_text: str, row_id: str, base_seed: int) -> str:
    rng = stable_rng(row_id, base_seed)
    clean_text = normalize_space(clean_text)
    tokens = clean_text.split()
    noisy_tokens: list[str] = []
    changed = False

    for token in tokens:
        new_token = mutate_token(token, rng)
        if new_token != token:
            changed = True
        noisy_tokens.append(new_token)

    if len(noisy_tokens) >= 3 and rng.random() < 0.3:
        insert_at = rng.randrange(1, len(noisy_tokens))
        noisy_tokens.insert(insert_at, "<gap>")
        changed = True
    elif len(noisy_tokens) >= 2 and rng.random() < 0.18:
        replace_at = rng.randrange(0, len(noisy_tokens))
        noisy_tokens[replace_at] = "<gap>"
        changed = True

    noisy_text = normalize_space(" ".join(noisy_tokens))
    if not changed:
        if "-" in noisy_text:
            noisy_text = noisy_text.replace("-", " ", 1)
        else:
            noisy_text = noisy_text.replace("ṭ", "m", 1)
        noisy_text = normalize_space(noisy_text)
    return noisy_text


def make_ocr_examples(clean_df: pd.DataFrame, base_seed: int) -> list[dict]:
    examples: list[dict] = []
    for row in clean_df.to_dict(orient="records"):
        clean_text = str(row["transliteration"])
        noisy_text = inject_ocr_noise(clean_text, str(row["oare_id"]), base_seed)
        examples.append(
            {
                "task": "ocr_correction",
                "source_oare_ids": [str(row["oare_id"])],
                "system_prompt": OCR_SYSTEM_PROMPT,
                "input_text": f"Input (Noisy):\n{noisy_text}",
                "target_text": clean_text,
            }
        )
    return examples


def reconstruct_aligned_sentences(clean_df: pd.DataFrame, cfg: Config) -> list[dict]:
    align_df = pd.read_csv(cfg.sentence_alignment_csv)
    text_map = (
        clean_df[["oare_id", "transliteration"]]
        .copy()
        .assign(oare_id=lambda x: x["oare_id"].astype(str))
        .set_index("oare_id")["transliteration"]
        .to_dict()
    )
    align_df = align_df[align_df["text_uuid"].astype(str).isin(text_map.keys())].copy()
    align_df["text_uuid"] = align_df["text_uuid"].astype(str)
    align_df["sentence_uuid"] = align_df["sentence_uuid"].astype(str)
    align_df["sentence_obj_in_text"] = pd.to_numeric(
        align_df["sentence_obj_in_text"], errors="coerce"
    ).fillna(0).astype(int)
    align_df["first_word_number"] = pd.to_numeric(
        align_df["first_word_number"], errors="coerce"
    ).fillna(0).astype(int)
    align_df = align_df[align_df["first_word_number"] > 0].copy()
    align_df = align_df.sort_values(
        ["text_uuid", "sentence_obj_in_text", "first_word_number", "sentence_uuid"]
    )

    reconstructed: list[dict] = []
    for text_uuid, group in align_df.groupby("text_uuid", sort=False):
        tokens = normalize_space(text_map[text_uuid]).split()
        if len(tokens) < 2:
            continue
        rows = group.to_dict(orient="records")
        starts = [int(row["first_word_number"]) - 1 for row in rows]
        if not starts:
            continue
        if any(start < 0 for start in starts):
            continue
        if starts != sorted(starts):
            continue
        for idx, row in enumerate(rows):
            start = starts[idx]
            if start >= len(tokens):
                continue
            end = starts[idx + 1] if idx + 1 < len(starts) else len(tokens)
            if end <= start:
                continue
            sent_tokens = tokens[start:end]
            sentence_tl = normalize_space(" ".join(sent_tokens))
            if not sentence_tl:
                continue
            reconstructed.append(
                {
                    "text_uuid": text_uuid,
                    "sentence_uuid": row["sentence_uuid"],
                    "sentence_obj_in_text": int(row["sentence_obj_in_text"]),
                    "translation": normalize_space(str(row.get("translation", ""))),
                    "transliteration": sentence_tl,
                }
            )
    return reconstructed


def collect_segmentation_candidates(clean_df: pd.DataFrame, cfg: Config) -> tuple[list[list[dict]], dict]:
    reconstructed = reconstruct_aligned_sentences(clean_df, cfg)
    sent_df = pd.DataFrame(reconstructed)
    if sent_df.empty:
        return [], {
            "segmentation_aligned_texts": 0,
            "segmentation_aligned_sentences": 0,
            "segmentation_candidate_windows": 0,
        }

    wc = sent_df["transliteration"].astype(str).str.split().str.len()
    char_len = sent_df["transliteration"].astype(str).str.len()
    eligible = (
        (wc >= 2)
        & (wc <= 24)
        & (char_len <= 160)
        & (~sent_df["transliteration"].astype(str).str.contains("<gap>", na=False))
        & (~sent_df["translation"].astype(str).str.contains("<gap>", na=False))
        & (sent_df["translation"].astype(str) != "")
    )
    sent_df = sent_df.loc[eligible].copy()
    sent_df = sent_df.sort_values(["text_uuid", "sentence_obj_in_text", "sentence_uuid"])
    windows: list[list[dict]] = []
    for _, group in sent_df.groupby("text_uuid", sort=False):
        rows = group.to_dict(orient="records")
        for window_size in range(2, 5):
            if len(rows) < window_size:
                continue
            for start in range(0, len(rows) - window_size + 1):
                window_rows = rows[start : start + window_size]
                total_chars = sum(len(str(row["transliteration"])) for row in window_rows)
                if total_chars > 260:
                    continue
                windows.append(window_rows)
    stats = {
        "segmentation_aligned_texts": int(sent_df["text_uuid"].nunique()),
        "segmentation_aligned_sentences": int(len(sent_df)),
        "segmentation_candidate_windows": int(len(windows)),
    }
    return windows, stats


def make_segmentation_examples(clean_df: pd.DataFrame, cfg: Config, target_count: int, seed: int) -> tuple[list[dict], dict]:
    windows, stats = collect_segmentation_candidates(clean_df, cfg)
    if not windows:
        raise RuntimeError("No eligible short-sentence windows found for segmentation synthesis.")
    rng = random.Random(seed)
    examples: list[dict] = []
    for idx in range(target_count):
        window = windows[idx] if idx < len(windows) else rng.choice(windows)
        sentences = [normalize_space(str(row["transliteration"])) for row in window]
        concatenated = normalize_space(" ".join(sentences))
        examples.append(
            {
                "task": "segmentation",
                "source_oare_ids": [str(window[0]["text_uuid"])],
                "system_prompt": SEGMENT_SYSTEM_PROMPT,
                "input_text": f"Input (Concatenated):\n{concatenated}",
                "target_text": json.dumps(sentences, ensure_ascii=False, indent=2),
            }
        )
    stats["segmentation_examples"] = len(examples)
    return examples, stats


def to_message_record(example: dict) -> dict:
    return {
        "messages": [
            {"role": "system", "content": example["system_prompt"]},
            {"role": "user", "content": example["input_text"]},
            {"role": "assistant", "content": example["target_text"]},
        ],
    }


def to_flat_record(example: dict, idx: int) -> dict:
    return {
        "id": f"{example['task']}-{idx:06d}",
        "task": example["task"],
        "source_oare_ids": "|".join(example["source_oare_ids"]),
        "system_prompt": example["system_prompt"],
        "input_text": example["input_text"],
        "target_text": example["target_text"],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

def build_dataset(cfg: Config) -> dict:
    clean_df, base_df = load_base_rows(cfg)
    translation_pool_mask = (
        (~clean_df["transliteration"].astype(str).str.contains("<gap>", na=False))
        & (~clean_df["translation"].astype(str).str.contains("<gap>", na=False))
        & (clean_df["uncertainty"].fillna("").astype(str) == "")
    )

    correction_examples = make_ocr_examples(clean_df, base_seed=cfg.random_seed)
    correction_count = len(correction_examples)
    total_target = int(round(correction_count / 0.30))
    translation_count = int(round(total_target * 0.40))
    segmentation_count = total_target - translation_count - correction_count

    translation_examples = make_translation_examples(
        source_df=clean_df,
        target_count=translation_count,
        seed=cfg.random_seed + 101,
    )
    segmentation_examples, segmentation_stats = make_segmentation_examples(
        clean_df=clean_df,
        cfg=cfg,
        target_count=segmentation_count,
        seed=cfg.random_seed + 202,
    )

    all_examples = translation_examples + correction_examples + segmentation_examples
    rng = random.Random(cfg.random_seed + 303)
    rng.shuffle(all_examples)

    message_rows = [to_message_record(example) for example in all_examples]
    flat_rows = [to_flat_record(example, idx + 1) for idx, example in enumerate(all_examples)]

    task_counts: dict[str, int] = {}
    for row in flat_rows:
        task_counts[row["task"]] = task_counts.get(row["task"], 0) + 1

    return {
        "message_rows": message_rows,
        "flat_rows": flat_rows,
        "summary": {
            "train_clean_rows": int(len(clean_df)),
            "base_translation_rows": int(len(base_df)),
            "translation_pool_rows": int(translation_pool_mask.sum()),
            "total_rows": int(len(flat_rows)),
            "task_counts": task_counts,
            "task_ratios": {
                task: round(count / len(flat_rows), 4) for task, count in task_counts.items()
            },
            **segmentation_stats,
        },
    }


def save_dataset(cfg: Config, dataset: dict) -> None:
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(cfg.output_jsonl, dataset["message_rows"])

    cfg.summary_json.write_text(
        json.dumps(dataset["summary"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    cfg = Config()
    dataset = build_dataset(cfg)
    save_dataset(cfg, dataset)

    print(f"Saved standard SFT JSONL: {cfg.output_jsonl}")
    print(f"Saved summary           : {cfg.summary_json}")
    print(json.dumps(dataset["summary"], ensure_ascii=False, indent=2))

    for row in dataset["message_rows"][:3]:
        print("\n---")
        print(json.dumps(row, ensure_ascii=False)[:600])


if __name__ == "__main__":
    main()
