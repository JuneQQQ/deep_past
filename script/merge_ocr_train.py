#!/usr/bin/env python3
"""
Deduplicate OCR-extracted train.csv against official train.csv.

Scope:
- Only removes rows from OCR data.
- Never edits or deletes official train.csv.
- Only deduplicates by normalized (transliteration, translation) pair.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Config:
    root_dir: Path = Path("/data/lsb/deep_past")
    official_train_csv: Path = Path("/data/lsb/deep_past/data/train.csv")
    ocr_train_csv: Path = Path(
        "/data/lsb/deep_past/output/extract_pairs_20260311_144517/train.csv"
    )

    @property
    def data_dir(self) -> Path:
        return self.root_dir / "data"

    @property
    def ocr_run_dir(self) -> Path:
        return self.ocr_train_csv.resolve().parent

    @property
    def output_csv(self) -> Path:
        return self.ocr_run_dir / "train_dedup.csv"

    @property
    def merged_output_csv(self) -> Path:
        return self.ocr_run_dir / "train_ocr_merged.csv"

    @property
    def data_merged_output_csv(self) -> Path:
        return self.data_dir / "train_ocr_merged.csv"

    @property
    def report_json(self) -> Path:
        return self.ocr_run_dir / "train_dedup_report.json"

    @property
    def prepare_data_py(self) -> Path:
        return Path(__file__).with_name("prepare_data.py")


REQUIRED_COLUMNS = ("oare_id", "transliteration", "translation")
DATA_SOURCE_COLUMN = "data_source"


def load_prepare_data_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("prepare_data_local", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load prepare_data module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing = [col for col in REQUIRED_COLUMNS if col not in reader.fieldnames]
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {missing}")
        rows = []
        for row in reader:
            rows.append({key: (value if value is not None else "") for key, value in row.items()})
        return rows, list(reader.fieldnames)


def normalize_pair(row: dict[str, str], prepare_data: Any) -> tuple[str, str]:
    src = prepare_data.preprocess_transliteration(row["transliteration"])
    src = prepare_data.normalize_punctuation(src).strip()
    tgt = prepare_data.preprocess_translation(row["translation"], light_mode=True)
    tgt = prepare_data.normalize_punctuation(tgt).strip()
    return src, tgt


def with_data_source(rows: list[dict[str, str]], source_name: str) -> list[dict[str, str]]:
    enriched = []
    for row in rows:
        new_row = dict(row)
        new_row[DATA_SOURCE_COLUMN] = source_name
        enriched.append(new_row)
    return enriched


def build_output_fieldnames(base_fieldnames: list[str]) -> list[str]:
    fieldnames = list(base_fieldnames)
    if DATA_SOURCE_COLUMN not in fieldnames:
        fieldnames.append(DATA_SOURCE_COLUMN)
    return fieldnames


def ensure_safe_paths(cfg: Config) -> None:
    if cfg.official_train_csv.resolve() == cfg.ocr_train_csv.resolve():
        raise ValueError("official_train_csv and ocr_train_csv must be different files.")
    if cfg.output_csv.resolve() == cfg.official_train_csv.resolve():
        raise ValueError("output_csv must not overwrite official train.csv.")
    if cfg.merged_output_csv.resolve() == cfg.official_train_csv.resolve():
        raise ValueError("merged_output_csv must not overwrite official train.csv.")
    if cfg.data_merged_output_csv.resolve() == cfg.official_train_csv.resolve():
        raise ValueError("data_merged_output_csv must not overwrite official train.csv.")


def main() -> None:
    cfg = Config()
    ensure_safe_paths(cfg)

    prepare_data = load_prepare_data_module(cfg.prepare_data_py)
    official_rows, official_fieldnames = read_csv_rows(cfg.official_train_csv)
    ocr_rows, ocr_fieldnames = read_csv_rows(cfg.ocr_train_csv)
    output_fieldnames = build_output_fieldnames(
        official_fieldnames if len(official_fieldnames) >= len(ocr_fieldnames) else ocr_fieldnames
    )

    official_keys = {normalize_pair(row, prepare_data) for row in official_rows}

    kept_rows: list[dict[str, str]] = []
    seen_ocr_keys: set[tuple[str, str]] = set()
    removed_against_official = 0
    removed_internal_duplicates = 0

    for row in ocr_rows:
        pair_key = normalize_pair(row, prepare_data)
        if pair_key in official_keys:
            removed_against_official += 1
            continue
        if pair_key in seen_ocr_keys:
            removed_internal_duplicates += 1
            continue
        seen_ocr_keys.add(pair_key)
        kept_rows.append(row)

    tagged_official_rows = with_data_source(official_rows, "official")
    tagged_kept_rows = with_data_source(kept_rows, "ocr")

    cfg.output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.merged_output_csv.parent.mkdir(parents=True, exist_ok=True)
    cfg.data_merged_output_csv.parent.mkdir(parents=True, exist_ok=True)
    with cfg.output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(tagged_kept_rows)

    merged_rows = tagged_official_rows + tagged_kept_rows
    with cfg.merged_output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)
    with cfg.data_merged_output_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=output_fieldnames)
        writer.writeheader()
        writer.writerows(merged_rows)

    report = {
        "official_train_csv": str(cfg.official_train_csv),
        "ocr_train_csv": str(cfg.ocr_train_csv),
        "output_csv": str(cfg.output_csv),
        "merged_output_csv": str(cfg.merged_output_csv),
        "data_merged_output_csv": str(cfg.data_merged_output_csv),
        "official_rows": len(official_rows),
        "ocr_rows_before": len(ocr_rows),
        "removed_against_official": removed_against_official,
        "removed_internal_duplicates": removed_internal_duplicates,
        "ocr_rows_after": len(kept_rows),
        "merged_rows": len(merged_rows),
    }
    cfg.report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Official train.csv : {cfg.official_train_csv}")
    print(f"OCR train.csv      : {cfg.ocr_train_csv}")
    print(f"Output CSV         : {cfg.output_csv}")
    print(f"Merged output CSV  : {cfg.merged_output_csv}")
    print(f"Data merged CSV    : {cfg.data_merged_output_csv}")
    print(f"Report JSON        : {cfg.report_json}")
    print(f"Official rows      : {len(official_rows)}")
    print(f"OCR rows before    : {len(ocr_rows)}")
    print(f"Removed vs official: {removed_against_official}")
    print(f"Removed OCR dupes  : {removed_internal_duplicates}")
    print(f"OCR rows after     : {len(kept_rows)}")
    print(f"Merged rows        : {len(merged_rows)}")


if __name__ == "__main__":
    main()
