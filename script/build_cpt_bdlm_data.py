#!/usr/bin/env python3
"""
Build a precomputed BDLM CPT dataset for train_cpt_bdlm.py.

This script moves all expensive and stochastic preprocessing out of the
training loop:
- Load transliterations from source CSVs.
- Build Akkadian -> English lookup from OA Lexicon + eBL Dictionary.
- Tokenize texts in batches.
- Precompute deterministic BDLM span-corruption source/target pairs.
- Save a DatasetDict to disk for fast training-time loading.

Acceleration strategies:
- Stream CSV parsing with the stdlib csv module.
- Batch tokenization.
- Multi-process example construction.
- Direct token-id corruption (no source-side re-encode).
- Worker-local cache for annotated masked spans.
"""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import inspect
import json
import os
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from urllib.parse import unquote_plus

import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, set_seed


parser = argparse.ArgumentParser()
parser.add_argument("--output-dir", type=str, default=None, help="Output dataset dir")
parser.add_argument("--timestamp", type=str, default=None, help="Optional timestamp tag for metadata")
parser.add_argument("--num-proc", type=int, default=None, help="Worker processes for offline build")
parser.add_argument("--train-variants", type=int, default=None, help="Offline variants per train text")
parser.add_argument("--eval-variants", type=int, default=None, help="Offline variants per eval text")
parser.add_argument("--overwrite", action="store_true", help="Overwrite existing dataset dir")
args, _ = parser.parse_known_args()


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def _safe_cpu_count() -> int:
    try:
        return max(int(os.cpu_count() or 1), 1)
    except Exception:
        return 1


class Config:
    def __init__(self):
        self.root_dir = "/data/lsb/deep_past"
        self.data_dir = os.path.join(self.root_dir, "data")

        timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timestamp = timestamp

        self.output_dir = args.output_dir or os.path.join(self.data_dir, "cpt_bdlm_dataset")
        self.overwrite_output = bool(args.overwrite)

        self.transliteration_csvs = [
            os.path.join(self.data_dir, "qwen_sentence_aligned_clean.csv"),
            os.path.join(self.data_dir, "archibab/archibab_akk_en_clean.csv"),
            os.path.join(self.data_dir, "cdli/cdli_clean.csv"),
        ]
        self.transliteration_col = "transliteration"

        # SFT mixing: add translation pairs to CPT data so encoder stays
        # compatible with translation task (prevents distribution shift)
        self.sft_mix_ratio = 0.5       # 50% of final dataset = SFT pairs
        self.sft_csv = os.path.join(self.data_dir, "qwen_sentence_aligned_clean.csv")
        self.sft_src_col = "transliteration"
        self.sft_tgt_col = "translation"
        self.min_text_chars = 10
        self.deduplicate_lines = True
        self.eval_ratio = 0.02
        self.random_seed = 42

        self.oa_lexicon_path = os.path.join(self.data_dir, "OA_Lexicon_eBL.csv")
        self.ebl_dict_path = os.path.join(self.data_dir, "eBL_Dictionary.csv")
        self.ebl_guidewords_path = os.path.join(self.data_dir, "ebl_api_guidewords.json")

        self.model_name = "google/byt5-large"
        self.tokenize_batch_size = 1024
        self.max_source_length = 1024
        self.max_target_length = 1024
        self.sliding_window_overlap = 128

        self.noise_density = 0.20
        self.mean_noise_span_length = 20
        self.max_sentinel_tokens = 100

        self.use_dict_augmented = True
        self.dict_replace_prob = 0.5
        self.dict_annotation_format = "[={eng}]"

        self.train_variants_per_text = args.train_variants if args.train_variants is not None else 4
        self.eval_variants_per_text = args.eval_variants if args.eval_variants is not None else 1

        default_num_proc = min(8, _safe_cpu_count())
        self.num_proc = max(1, args.num_proc if args.num_proc is not None else default_num_proc)

    def to_dict(self) -> Dict[str, object]:
        return {
            k: (str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v)
            for k, v in self.__dict__.items()
        }


config = Config()


def sliding_window_split(text: str, max_bytes: int, overlap_bytes: int) -> List[str]:
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        if end < len(encoded):
            search_start = max(start + max_bytes // 2, start)
            space_pos = encoded.rfind(b" ", search_start, end)
            if space_pos > start:
                end = space_pos
        chunk_text = encoded[start:end].decode("utf-8", errors="ignore").strip()
        if chunk_text:
            chunks.append(chunk_text)
        start = end - overlap_bytes if end < len(encoded) else len(encoded)
        if start <= (end - max_bytes):
            start = end
    return chunks if chunks else [text]


def load_transliterations_from_csvs(
    csv_paths: List[str],
    tl_col: str,
    min_chars: int,
    deduplicate: bool,
    max_bytes: int,
    overlap_bytes: int,
) -> List[str]:
    all_texts: List[str] = []
    for path in csv_paths:
        if not os.path.exists(path):
            print(f"   ⚠️ CSV not found, skipping: {path}")
            continue

        local_count = 0
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if tl_col not in (reader.fieldnames or []):
                print(f"   ⚠️ Column '{tl_col}' not in {path}, skipping")
                continue
            for row in reader:
                text = str(row.get(tl_col) or "").strip()
                if text:
                    all_texts.append(text)
                    local_count += 1
        print(f"   📂 {os.path.basename(path)}: {local_count} transliterations")

    filtered = [text for text in all_texts if len(text) >= min_chars]

    if deduplicate:
        seen = set()
        unique: List[str] = []
        for text in filtered:
            if text not in seen:
                seen.add(text)
                unique.append(text)
        filtered = unique

    long_count = sum(1 for text in filtered if len(text.encode("utf-8")) > max_bytes)
    if long_count:
        expanded: List[str] = []
        for text in filtered:
            expanded.extend(sliding_window_split(text, max_bytes=max_bytes, overlap_bytes=overlap_bytes))
        print(
            f"   ✂️ Sliding window: {long_count} long texts -> "
            f"{len(expanded) - len(filtered) + long_count} extra chunks"
        )
        filtered = expanded

    if len(filtered) < 2:
        raise ValueError(f"Need >=2 texts after filtering, got {len(filtered)}")

    print("\n📊 Corpus stats:")
    print(f"   Total raw: {len(all_texts)}")
    print(f"   After filter + dedup + split: {len(filtered)}")
    return filtered


def resolve_model_path(model_name: str) -> Tuple[str, bool]:
    if os.path.isdir(model_name):
        return model_name, True
    cache_roots = [
        "/root/huggingface_cache/hub",
        os.path.expanduser("~/.cache/huggingface/hub"),
    ]
    model_pattern = model_name.replace("/", "--")
    for cache_root in cache_roots:
        pattern = os.path.join(cache_root, f"models--{model_pattern}", "snapshots", "*", "config.json")
        hits = glob.glob(pattern)
        if hits:
            hits.sort()
            return os.path.dirname(hits[-1]), True
    return model_name, False


def collect_sentinel_token_ids(tokenizer, max_sentinels: int) -> List[int]:
    sentinel_ids: List[int] = []
    for i in range(max_sentinels):
        token = f"<extra_id_{i}>"
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None:
            break
        token_id = int(token_id)
        roundtrip = tokenizer.convert_ids_to_tokens(token_id)
        if roundtrip != token:
            unk_id = tokenizer.unk_token_id
            if unk_id is not None and token_id == int(unk_id):
                break
        sentinel_ids.append(token_id)
    if not sentinel_ids:
        raise ValueError("Tokenizer does not expose <extra_id_*> sentinel tokens.")
    return sentinel_ids


def build_akkadian_english_dict(lex_path: str, ebl_path: str, guidewords_path: str | None = None) -> Dict[str, str]:
    # Step 1: eBL Dictionary CSV — word -> first quoted English translation
    ebl_trans: Dict[str, str] = {}
    with open(ebl_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            word = re.sub(r"\s+[IVX]+$", "", str(row.get("word") or "").strip()).strip().lower()
            if not word or word in ebl_trans:
                continue
            defn = str(row.get("definition") or "").strip()
            translations = re.findall(r'"([^"]+)"', defn)
            if translations:
                ebl_trans[word] = translations[0].strip()

    # Step 1b: Scraped API guidewords (url_word -> guideWord) — fills gaps in CSV
    api_guidewords: Dict[str, str] = {}
    if guidewords_path and os.path.exists(guidewords_path):
        with open(guidewords_path, encoding="utf-8") as fh:
            api_guidewords = json.load(fh)
        print(f"   📖 Loaded {len(api_guidewords)} API guidewords from {os.path.basename(guidewords_path)}")

    # Step 2: OA Lexicon form -> eBL URL word -> lookup (CSV first, then API fallback)
    form_to_eng: Dict[str, str] = {}
    with open(lex_path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            form = str(row.get("form") or "").strip().lower()
            if not form or form in form_to_eng:
                continue
            url = str(row.get("eBL") or "")
            if "word=" not in url:
                continue
            url_word_raw = url.split("word=")[-1].strip()
            url_word_lower = unquote_plus(url_word_raw).lower()
            eng = ebl_trans.get(url_word_lower)
            if not eng:
                eng = api_guidewords.get(url_word_raw)
            if eng:
                form_to_eng[form] = eng
    return form_to_eng


def split_corpus(lines: List[str], eval_ratio: float, seed: int) -> Tuple[List[str], List[str]]:
    if len(lines) < 2:
        raise ValueError("Need at least two lines for train/eval split")
    eval_size = max(1, int(len(lines) * eval_ratio))
    eval_size = min(eval_size, len(lines) - 1)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(lines))
    eval_idx = set(int(x) for x in order[:eval_size])
    train_lines = [line for idx, line in enumerate(lines) if idx not in eval_idx]
    eval_lines = [line for idx, line in enumerate(lines) if idx in eval_idx]
    return train_lines, eval_lines


def batch_tokenize_texts(texts: List[str], tokenizer, batch_size: int, max_length: int) -> List[List[int]]:
    all_ids: List[List[int]] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=False,
            truncation=True,
            max_length=max(2, max_length - 1),
            padding=False,
        )
        all_ids.extend([list(map(int, ids)) for ids in enc["input_ids"]])
    return all_ids


def _seed_for_variant(base_seed: int, split_name: str, example_idx: int, variant_idx: int) -> int:
    payload = f"{base_seed}:{split_name}:{example_idx}:{variant_idx}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFF_FFFF_FFFF_FFFF


def _build_tasks(
    split_name: str,
    texts: List[str],
    token_ids_list: List[List[int]],
    variants_per_text: int,
    base_seed: int,
) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    for example_idx, (text, token_ids) in enumerate(zip(texts, token_ids_list)):
        for variant_idx in range(max(1, variants_per_text)):
            tasks.append(
                {
                    "split": split_name,
                    "example_idx": example_idx,
                    "variant_idx": variant_idx,
                    "seed": _seed_for_variant(base_seed, split_name, example_idx, variant_idx),
                    "text": text,
                    "token_ids": token_ids,
                }
            )
    return tasks


_WORKER_STATE: Dict[str, object] = {}


def _worker_init(
    model_path: str,
    local_files_only: bool,
    sentinel_token_ids: List[int],
    eos_token_id: int,
    noise_density: float,
    mean_noise_span_length: float,
    max_source_length: int,
    max_target_length: int,
    akk_eng_dict: Dict[str, str],
    use_dict_augmented: bool,
    dict_replace_prob: float,
    dict_annotation_format: str,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local_files_only)
    _WORKER_STATE.clear()
    _WORKER_STATE.update(
        {
            "tokenizer": tokenizer,
            "sentinel_token_ids": list(map(int, sentinel_token_ids)),
            "eos_token_id": int(eos_token_id),
            "noise_density": float(noise_density),
            "mean_noise_span_length": float(mean_noise_span_length),
            "max_source_length": int(max_source_length),
            "max_target_length": int(max_target_length),
            "akk_eng_dict": akk_eng_dict,
            "use_dict_augmented": bool(use_dict_augmented),
            "dict_replace_prob": float(dict_replace_prob),
            "dict_annotation_format": str(dict_annotation_format),
            "annotated_cache": {},
        }
    )


def _random_segmentation(num_items: int, num_segments: int, rng: np.random.Generator) -> np.ndarray:
    if num_segments <= 0:
        raise ValueError(f"num_segments must be > 0, got {num_segments}")
    if num_items < num_segments:
        raise ValueError(f"num_items ({num_items}) < num_segments ({num_segments})")
    if num_segments == 1:
        return np.array([num_items], dtype=np.int32)
    breakpoints = np.sort(rng.choice(np.arange(1, num_items), size=num_segments - 1, replace=False))
    points = np.concatenate(([0], breakpoints, [num_items]))
    return np.diff(points).astype(np.int32)


def _random_spans_noise_mask(
    length: int,
    noise_density: float,
    mean_noise_span_length: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if length < 2:
        return np.array([True] * length, dtype=bool)
    num_noise_tokens = int(np.round(length * noise_density))
    num_noise_tokens = min(max(num_noise_tokens, 1), length - 1)
    num_noise_spans = int(np.round(num_noise_tokens / mean_noise_span_length))
    num_noise_spans = max(num_noise_spans, 1)
    num_noise_spans = min(num_noise_spans, num_noise_tokens)
    num_nonnoise_tokens = length - num_noise_tokens
    nonnoise_span_lengths = _random_segmentation(num_nonnoise_tokens, num_noise_spans, rng)
    noise_span_lengths = _random_segmentation(num_noise_tokens, num_noise_spans, rng)
    noise_mask = np.zeros(length, dtype=bool)
    cursor = 0
    for nonnoise_len, noise_len in zip(nonnoise_span_lengths, noise_span_lengths):
        cursor += int(nonnoise_len)
        noise_mask[cursor: cursor + int(noise_len)] = True
        cursor += int(noise_len)
    return noise_mask


# Function word translations that the model can learn on its own — skip annotation
_DICT_STOPWORDS = {
    "to, for", "and, but, also", "and", "but", "also", "or",
    "who(m), which; (s)he who, that which; of", "not, no; without, un-",
    "not", "no", "like; when, as, that", "saying:", "if",
    "from, out of; since, after", "with", "may, let",
    "until, as far as", "you", "I; me", "we", "this", "that",
    ", on; by; from", "there", "here", "as much as", "because (of)",
    "not, no", "before, in front of",
}

def _annotate_span_with_dict(span_text: str) -> Tuple[List[int] | None, bool]:
    cache: Dict[str, Tuple[List[int] | None, bool]] = _WORKER_STATE["annotated_cache"]  # type: ignore[index]
    cached = cache.get(span_text)
    if cached is not None:
        return cached

    tokenizer = _WORKER_STATE["tokenizer"]  # type: ignore[index]
    akk_eng_dict: Dict[str, str] = _WORKER_STATE["akk_eng_dict"]  # type: ignore[index]
    dict_annotation_format = _WORKER_STATE["dict_annotation_format"]  # type: ignore[index]
    max_target_length = _WORKER_STATE["max_target_length"]  # type: ignore[index]

    words = span_text.split()
    annotated_words: List[str] = []
    changed = False
    for word in words:
        annotated_words.append(word)
        eng = akk_eng_dict.get(word.lower())
        if eng and eng not in _DICT_STOPWORDS:
            annotated_words.append(dict_annotation_format.replace("{eng}", eng))
            changed = True

    if changed:
        annotated_text = " ".join(annotated_words)
        annotated_ids = tokenizer(
            annotated_text,
            add_special_tokens=False,
            truncation=True,
            max_length=max_target_length,
            padding=False,
        )["input_ids"]
        result: Tuple[List[int] | None, bool] = ([int(t) for t in annotated_ids], True)
    else:
        result = (None, False)

    if len(cache) < 50000:
        cache[span_text] = result
    return result


def _build_example(task: Dict[str, object]) -> Dict[str, object]:
    tokenizer = _WORKER_STATE["tokenizer"]  # type: ignore[index]
    sentinel_token_ids: List[int] = _WORKER_STATE["sentinel_token_ids"]  # type: ignore[index]
    eos_token_id: int = _WORKER_STATE["eos_token_id"]  # type: ignore[index]
    noise_density: float = _WORKER_STATE["noise_density"]  # type: ignore[index]
    mean_noise_span_length: float = _WORKER_STATE["mean_noise_span_length"]  # type: ignore[index]
    max_source_length: int = _WORKER_STATE["max_source_length"]  # type: ignore[index]
    max_target_length: int = _WORKER_STATE["max_target_length"]  # type: ignore[index]
    use_dict_augmented: bool = _WORKER_STATE["use_dict_augmented"]  # type: ignore[index]
    dict_replace_prob: float = _WORKER_STATE["dict_replace_prob"]  # type: ignore[index]
    akk_eng_dict: Dict[str, str] = _WORKER_STATE["akk_eng_dict"]  # type: ignore[index]

    token_ids = [int(t) for t in task["token_ids"]]  # type: ignore[index]
    if len(token_ids) == 0:
        token_ids = [eos_token_id, eos_token_id]
    elif len(token_ids) == 1:
        token_ids = token_ids + [token_ids[0]]

    rng = np.random.default_rng(int(task["seed"]))  # type: ignore[index]
    noise_mask = _random_spans_noise_mask(len(token_ids), noise_density, mean_noise_span_length, rng)

    source_ids: List[int] = []
    target_ids: List[int] = []
    masked_spans = 0
    annotated_spans = 0
    sentinel_idx = 0
    i = 0

    while i < len(token_ids):
        if noise_mask[i]:
            if sentinel_idx >= len(sentinel_token_ids):
                break

            sentinel_id = int(sentinel_token_ids[sentinel_idx])
            source_ids.append(sentinel_id)
            target_ids.append(sentinel_id)
            masked_spans += 1

            span_start = i
            while i < len(token_ids) and noise_mask[i]:
                i += 1
            span_ids = token_ids[span_start:i]

            use_aug = (
                use_dict_augmented
                and bool(akk_eng_dict)
                and rng.random() < dict_replace_prob
            )
            if use_aug:
                span_text = tokenizer.decode(span_ids, skip_special_tokens=True)
                annotated_ids, changed = _annotate_span_with_dict(span_text)
                if changed and annotated_ids is not None:
                    target_ids.extend(annotated_ids)
                    annotated_spans += 1
                else:
                    target_ids.extend(span_ids)
            else:
                target_ids.extend(span_ids)
            sentinel_idx += 1
        else:
            source_ids.append(int(token_ids[i]))
            i += 1

    source_ids.append(eos_token_id)
    target_ids.append(eos_token_id)

    return {
        "input_ids": source_ids[:max_source_length],
        "labels": target_ids[:max_target_length],
        "input_len": min(len(source_ids), max_source_length),
        "label_len": min(len(target_ids), max_target_length),
        "masked_spans": masked_spans,
        "annotated_spans": annotated_spans,
        "seed": int(task["seed"]),
        "variant_idx": int(task["variant_idx"]),
        "example_idx": int(task["example_idx"]),
        "source_text": str(task["text"]),
    }


def build_split_records(
    split_name: str,
    texts: List[str],
    token_ids_list: List[List[int]],
    variants_per_text: int,
    cfg: Config,
    model_path: str,
    local_files_only: bool,
    sentinel_token_ids: List[int],
    eos_token_id: int,
    akk_eng_dict: Dict[str, str],
) -> List[Dict[str, object]]:
    tasks = _build_tasks(
        split_name=split_name,
        texts=texts,
        token_ids_list=token_ids_list,
        variants_per_text=variants_per_text,
        base_seed=cfg.random_seed,
    )

    print(
        f"   ⚙️ {split_name}: {len(texts)} base texts × {max(1, variants_per_text)} variant(s) "
        f"-> {len(tasks)} examples"
    )

    initargs = (
        model_path,
        local_files_only,
        sentinel_token_ids,
        eos_token_id,
        cfg.noise_density,
        cfg.mean_noise_span_length,
        cfg.max_source_length,
        cfg.max_target_length,
        akk_eng_dict,
        cfg.use_dict_augmented,
        cfg.dict_replace_prob,
        cfg.dict_annotation_format,
    )

    if cfg.num_proc <= 1:
        _worker_init(*initargs)
        return [_build_example(task) for task in tqdm(tasks, desc=f"   {split_name}", unit="ex")]

    chunksize = max(8, len(tasks) // max(cfg.num_proc * 8, 1))
    with ProcessPoolExecutor(
        max_workers=cfg.num_proc,
        initializer=_worker_init,
        initargs=initargs,
    ) as executor:
        return list(tqdm(
            executor.map(_build_example, tasks, chunksize=chunksize),
            total=len(tasks), desc=f"   {split_name}", unit="ex",
        ))


def summarize_records(name: str, records: List[Dict[str, object]]) -> Dict[str, object]:
    if not records:
        return {"split": name, "examples": 0}
    annotated_example_count = sum(1 for row in records if int(row["annotated_spans"]) > 0)
    total_masked_spans = sum(int(row["masked_spans"]) for row in records)
    total_annotated_spans = sum(int(row["annotated_spans"]) for row in records)
    avg_input_len = sum(int(row["input_len"]) for row in records) / len(records)
    avg_label_len = sum(int(row["label_len"]) for row in records) / len(records)
    return {
        "split": name,
        "examples": len(records),
        "annotated_example_count": annotated_example_count,
        "annotated_example_ratio": round(annotated_example_count / len(records), 4),
        "total_masked_spans": total_masked_spans,
        "total_annotated_spans": total_annotated_spans,
        "annotated_span_ratio": round(total_annotated_spans / total_masked_spans, 4) if total_masked_spans else 0.0,
        "avg_input_len": round(avg_input_len, 2),
        "avg_label_len": round(avg_label_len, 2),
    }


def main() -> None:
    print("=" * 60)
    print("BUILD BDLM CPT DATASET")
    print("=" * 60)

    if os.path.exists(config.output_dir):
        if not config.overwrite_output:
            raise FileExistsError(
                f"Output dir already exists: {config.output_dir}\n"
                "Pass --overwrite to rebuild."
            )
        print(f"   🗑️ Overwriting existing dataset dir: {config.output_dir}")
        shutil.rmtree(config.output_dir)

    set_seed(config.random_seed)
    os.makedirs(config.output_dir, exist_ok=True)

    print("\n📚 Building Akkadian→English dictionary...")
    if config.use_dict_augmented:
        if not (os.path.exists(config.oa_lexicon_path) and os.path.exists(config.ebl_dict_path)):
            raise FileNotFoundError("OA Lexicon or eBL Dictionary not found; cannot build BDLM dataset.")
        akk_eng_dict = build_akkadian_english_dict(config.oa_lexicon_path, config.ebl_dict_path, config.ebl_guidewords_path)
        print(f"   ✅ Dictionary size: {len(akk_eng_dict)}")
    else:
        akk_eng_dict = {}
        print("   ⬜ Dictionary augmentation disabled")

    print("\n📂 Loading transliterations...")
    corpus_lines = load_transliterations_from_csvs(
        csv_paths=config.transliteration_csvs,
        tl_col=config.transliteration_col,
        min_chars=config.min_text_chars,
        deduplicate=config.deduplicate_lines,
        max_bytes=config.max_source_length,
        overlap_bytes=config.sliding_window_overlap,
    )

    train_lines, eval_lines = split_corpus(corpus_lines, config.eval_ratio, config.random_seed)
    print("\n📊 Split stats:")
    print(f"   Train base texts: {len(train_lines)}")
    print(f"   Eval base texts: {len(eval_lines)}")

    print("\n🔤 Loading tokenizer...")
    resolved_model_path, local_files_only = resolve_model_path(config.model_name)
    print(f"   Model source: {resolved_model_path}")
    print(f"   local_files_only={local_files_only}")

    if not local_files_only and config.num_proc > 1:
        print("   ⚠️ Model path is not local; forcing num_proc=1 to avoid parallel remote tokenizer loads")
        config.num_proc = 1

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
    if eos_token_id is None:
        raise ValueError("Tokenizer must expose either eos_token_id or pad_token_id.")

    sentinel_token_ids = collect_sentinel_token_ids(tokenizer, config.max_sentinel_tokens)
    print(f"   Sentinel tokens: {len(sentinel_token_ids)}")

    # Prepend meta prefix (OAOI/OAOD) so CPT encoder sees the same prefix format as SFT
    def _add_meta_prefix(text: str) -> str:
        if '<gap>' in text or '[' in text or '...' in text:
            return f"OAOD {text}"
        return f"OAOI {text}"

    train_lines = [_add_meta_prefix(t) for t in train_lines]
    eval_lines = [_add_meta_prefix(t) for t in eval_lines]
    print(f"   ✅ Added OAOI/OAOD meta prefix to all {len(train_lines)+len(eval_lines)} texts")

    print("\n⚡ Batch tokenizing base texts...")
    train_token_ids = batch_tokenize_texts(train_lines, tokenizer, config.tokenize_batch_size, config.max_source_length)
    eval_token_ids = batch_tokenize_texts(eval_lines, tokenizer, config.tokenize_batch_size, config.max_source_length)

    print("\n🏗️ Precomputing offline BDLM examples...")
    train_records = build_split_records(
        split_name="train",
        texts=train_lines,
        token_ids_list=train_token_ids,
        variants_per_text=config.train_variants_per_text,
        cfg=config,
        model_path=resolved_model_path,
        local_files_only=local_files_only,
        sentinel_token_ids=sentinel_token_ids,
        eos_token_id=int(eos_token_id),
        akk_eng_dict=akk_eng_dict,
    )
    eval_records = build_split_records(
        split_name="eval",
        texts=eval_lines,
        token_ids_list=eval_token_ids,
        variants_per_text=config.eval_variants_per_text,
        cfg=config,
        model_path=resolved_model_path,
        local_files_only=local_files_only,
        sentinel_token_ids=sentinel_token_ids,
        eos_token_id=int(eos_token_id),
        akk_eng_dict=akk_eng_dict,
    )

    # ── SFT mixing: add translation pairs ──
    sft_train_records = []
    sft_eval_records = []
    if config.sft_mix_ratio > 0 and os.path.exists(config.sft_csv):
        print(f"\n📝 Building SFT translation pairs (mix_ratio={config.sft_mix_ratio})...")
        sft_df = pd.read_csv(config.sft_csv)
        sft_df = sft_df.dropna(subset=[config.sft_src_col, config.sft_tgt_col])
        print(f"   SFT source: {len(sft_df)} pairs from {os.path.basename(config.sft_csv)}")

        # Tokenize SFT pairs
        sft_sources = sft_df[config.sft_src_col].astype(str).tolist()
        sft_targets = sft_df[config.sft_tgt_col].astype(str).tolist()

        # Add task prefix
        sft_sources = [f"OAOI {s}" if '<gap>' not in s else f"OAOD {s}" for s in sft_sources]

        sft_src_ids = batch_tokenize_texts(sft_sources, tokenizer, config.tokenize_batch_size, config.max_source_length)
        sft_tgt_ids = batch_tokenize_texts(sft_targets, tokenizer, config.tokenize_batch_size, config.max_target_length)

        # Split SFT data same ratio as CPT
        n_sft_eval = max(1, int(len(sft_src_ids) * config.eval_ratio))
        rng_sft = np.random.default_rng(config.random_seed + 999)
        sft_order = rng_sft.permutation(len(sft_src_ids))
        sft_eval_idx = set(int(x) for x in sft_order[:n_sft_eval])

        for i in range(len(sft_src_ids)):
            # Add EOS
            src = sft_src_ids[i] + [int(eos_token_id)]
            tgt = sft_tgt_ids[i] + [int(eos_token_id)]
            rec = {
                "input_ids": src[:config.max_source_length],
                "labels": tgt[:config.max_target_length],
                "input_len": min(len(src), config.max_source_length),
                "label_len": min(len(tgt), config.max_target_length),
                "masked_spans": 0,
                "annotated_spans": 0,
                "seed": 0,
                "variant_idx": 0,
                "example_idx": i,
                "source_text": sft_sources[i][:200],
            }
            if i in sft_eval_idx:
                sft_eval_records.append(rec)
            else:
                sft_train_records.append(rec)

        # Calculate how many SFT records to keep based on ratio
        target_sft_count = int(len(train_records) * config.sft_mix_ratio / (1 - config.sft_mix_ratio))
        if len(sft_train_records) > target_sft_count:
            rng_sft.shuffle(sft_train_records)
            sft_train_records = sft_train_records[:target_sft_count]

        print(f"   SFT train: {len(sft_train_records)}, SFT eval: {len(sft_eval_records)}")

    all_train = train_records + sft_train_records
    all_eval = eval_records + sft_eval_records
    # Shuffle train
    rng_final = np.random.default_rng(config.random_seed + 42)
    rng_final.shuffle(all_train)

    dataset = DatasetDict(
        {
            "train": Dataset.from_list(all_train),
            "eval": Dataset.from_list(all_eval),
        }
    )

    print(f"\n💾 Saving dataset (CPT: {len(train_records)} + SFT: {len(sft_train_records)} = {len(all_train)} train)...")
    dataset.save_to_disk(config.output_dir)

    summary = {
        "builder_config": config.to_dict(),
        "resolved_model_path": resolved_model_path,
        "local_files_only": local_files_only,
        "dict_size": len(akk_eng_dict),
        "raw_corpus_lines": len(corpus_lines),
        "train_base_texts": len(train_lines),
        "eval_base_texts": len(eval_lines),
        "splits": {
            "train": summarize_records("train", train_records),
            "eval": summarize_records("eval", eval_records),
        },
    }
    with open(os.path.join(config.output_dir, "builder_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)

    print("\n✅ BDLM dataset build finished.")
    print(f"   Output: {config.output_dir}")
    print(f"   Train examples: {len(train_records)}")
    print(f"   Eval examples: {len(eval_records)}")


if __name__ == "__main__":
    main()
