#!/usr/bin/env python3
"""
Build CPT (Continual Pre-Training) corpus from Old Assyrian PDF library.

Pipeline:
  1. Recursively discover PDFs under data/old-assyrian/
  2. MinerU OCR per PDF, chunked by page range with resumable caches
  2. Light cleaning (normalize whitespace, strip page headers/footers)
  3. Sliding-window split to <512 ByT5 bytes
  4. Output: data/cpt/cpt.jsonl (one JSON object per chunk)

Usage:
  conda activate mineru
  python script/build_cpt_data.py [--skip-ocr]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

# ============================================================================
# Config
# ============================================================================
PDF_DIR = Path("/data/lsb/deep_past/data/old-assyrian")
OUTPUT_DIR = Path("/data/lsb/deep_past/data/cpt")
OUTPUT_JSONL = OUTPUT_DIR / "cpt.jsonl"
OUTPUT_TXT = OUTPUT_DIR / "all_akkadian_corpus.txt"
EXPORTED_DOC_IDS = OUTPUT_DIR / "cpt_exported_doc_ids.txt"

OCR_LANG = "latin"
OCR_CHUNK_SIZE = 96           # 默认按较小页块处理，避免 PDF 渲染阶段占用过高内存
MINERU_DEVICE = "cuda"
MAX_BYTE_LEN = 510            # ByT5 max_length=512 减去 EOS/padding 余量
SLIDING_OVERLAP = 128         # 滑动窗口重叠字节数
MIN_LINE_CHARS = 10           # 最短行字符数
MIN_CHUNK_BYTES = 20          # 最短 chunk 字节数

# ============================================================================
# MinerU OCR helpers (adapted from extract_akt8_mineru_ocr.py)
# ============================================================================

def _find_middle_json(output_dir: Path) -> Path:
    candidates = list(output_dir.rglob("*_middle.json"))
    if len(candidates) == 1:
        return candidates[0]
    candidates = list(output_dir.rglob("middle.json"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No middle.json found under {output_dir}")


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


def discover_pdf_files(pdf_dir: Path) -> list[Path]:
    return sorted(
        (
            path for path in pdf_dir.rglob("*")
            if path.is_file() and path.suffix.lower() == ".pdf"
        ),
        key=lambda path: path.relative_to(pdf_dir).as_posix().lower(),
    )


def make_doc_id(pdf_root: Path, pdf_path: Path) -> str:
    rel = pdf_path.relative_to(pdf_root).with_suffix("")
    slug = "__".join(rel.parts)
    slug = re.sub(r"[^0-9A-Za-z._-]+", "_", slug).strip("._")
    if not slug:
        slug = "document"
    digest = hashlib.sha1(str(rel).encode("utf-8")).hexdigest()[:10]
    return f"{slug}__{digest}"


def _doc_relative_path(pdf_root: Path, pdf_path: Path) -> str:
    return pdf_path.relative_to(pdf_root).as_posix()


def _pdf_page_count(pdf_path: Path) -> int:
    try:
        import fitz
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF (fitz) is required to count PDF pages. "
            "Run this script from the `mineru` conda environment."
        ) from exc

    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def detect_gpu_vram_gb(device: str) -> int | None:
    if not device or device == "auto" or device.startswith("cpu"):
        return None
    try:
        from mineru.utils.model_utils import get_vram
    except ImportError:
        return None
    try:
        return int(get_vram(device))
    except Exception:
        return None


def recommend_ocr_workers(device: str, vram_gb: int | None) -> int:
    if not device or device == "auto" or device.startswith("cpu"):
        return 1
    if vram_gb is None:
        return 1
    if vram_gb >= 40:
        return 2
    return 1


def recommend_ocr_chunk_size(vram_gb: int | None, workers: int) -> int:
    if workers >= 2:
        return 64
    if vram_gb is not None and vram_gb >= 24:
        return 96
    return 64


def recommend_min_batch_inference_size(vram_gb: int | None, workers: int) -> int:
    if vram_gb is None:
        return 384
    if vram_gb >= 24:
        return 512 if workers == 1 else 384
    if vram_gb >= 16:
        return 384
    return 384


def recommend_render_threads(workers: int) -> int:
    return 1 if workers >= 2 else 2


def _run_mineru_cli_chunk(
    pdf_path: Path,
    chunk_start: int,
    chunk_end: int,
    *,
    device: str,
    lang: str,
    virtual_vram: int | None = None,
    min_batch_inference_size: int | None = None,
    render_threads: int | None = None,
    log_path: Path | None = None,
) -> list[dict[str, Any]]:
    """Run MinerU OCR on one inclusive 1-indexed page range."""
    s0, e0 = chunk_start - 1, chunk_end - 1
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_out = Path(tmpdir) / "out"
        cmd = [
            "mineru",
            "-p", str(pdf_path),
            "-o", str(tmp_out),
            "-m", "ocr",
            "-l", lang,
            "-b", "pipeline",
            "-f", "False",
            "-t", "False",
            "--source", "local",
            "-s", str(s0),
            "-e", str(e0),
        ]
        if device and device != "auto":
            cmd += ["-d", device]
        if virtual_vram is not None:
            cmd += ["--vram", str(virtual_vram)]

        env = os.environ.copy()
        env["MINERU_MODEL_SOURCE"] = "local"
        if virtual_vram is not None:
            env["MINERU_VIRTUAL_VRAM_SIZE"] = str(virtual_vram)
        if min_batch_inference_size is not None:
            env["MINERU_MIN_BATCH_INFERENCE_SIZE"] = str(min_batch_inference_size)
        if render_threads is not None:
            env["MINERU_PDF_RENDER_THREADS"] = str(render_threads)

        if log_path is not None:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as log_fh:
                log_fh.write(
                    f"\n=== chunk pages {chunk_start}-{chunk_end} @ {time.strftime('%F %T')} ===\n"
                )
                log_fh.write("CMD: " + " ".join(cmd) + "\n")
                if render_threads is not None:
                    log_fh.write(f"MINERU_PDF_RENDER_THREADS={render_threads}\n")
                if min_batch_inference_size is not None:
                    log_fh.write(
                        f"MINERU_MIN_BATCH_INFERENCE_SIZE={min_batch_inference_size}\n"
                    )
                result = subprocess.run(
                    cmd,
                    env=env,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                )
        else:
            result = subprocess.run(cmd, env=env)
        if result.returncode != 0:
            raise RuntimeError(
                f"mineru CLI exited with code {result.returncode} "
                f"for pages {chunk_start}-{chunk_end}"
            )

        middle_path = _find_middle_json(tmp_out)
        middle_json = json.loads(middle_path.read_text(encoding="utf-8"))
        return middle_json.get("pdf_info") or []


def run_mineru_pdf(
    pdf_root: Path,
    pdf_path: Path,
    output_dir: Path,
    *,
    device: str = "cuda",
    lang: str = "latin",
    chunk_size: int = OCR_CHUNK_SIZE,
    virtual_vram: int | None = None,
    min_batch_inference_size: int | None = None,
    render_threads: int | None = None,
    log_dir: Path | None = None,
) -> Path:
    doc_id = make_doc_id(pdf_root, pdf_path)
    rel_path = _doc_relative_path(pdf_root, pdf_path)
    ocr_dir = output_dir / doc_id / "ocr"
    ocr_dir.mkdir(parents=True, exist_ok=True)
    log_path = (log_dir / f"{doc_id}.log") if log_dir is not None else None

    final_path = ocr_dir / f"{doc_id}_middle.json"
    partial_path = ocr_dir / f"{doc_id}_middle.partial.json"
    page_count = _pdf_page_count(pdf_path)

    if final_path.exists():
        try:
            cached = json.loads(final_path.read_text(encoding="utf-8"))
        except Exception:
            cached = {}
        if (
            cached.get("source_rel_path") == rel_path
            and int(cached.get("page_count") or 0) == page_count
            and cached.get("ocr_lang", lang) == lang
        ):
            print(f"   ♻️  跳过已完成 OCR: {rel_path} ({page_count} 页)", flush=True)
            return final_path
        print(f"   🔁 重新生成不兼容缓存: {rel_path}", flush=True)

    total_chunks = max((page_count + chunk_size - 1) // chunk_size, 1)
    all_pdf_info: list[dict[str, Any]] = []
    next_page = 1

    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
        except Exception:
            partial = {}
        if (
            partial.get("source_rel_path") == rel_path
            and int(partial.get("page_count") or 0) == page_count
            and partial.get("ocr_lang", lang) == lang
        ):
            all_pdf_info = partial.get("pdf_info") or []
            next_page = int(partial.get("completed_upto_page") or 0) + 1
            if next_page <= page_count:
                print(
                    f"   ↩️  断点续跑: {rel_path} 从第 {next_page} 页继续 "
                    f"({page_count} 页总计)",
                    flush=True,
                )

    if next_page > page_count:
        payload = {
            "doc_id": doc_id,
            "source_rel_path": rel_path,
            "page_count": page_count,
            "ocr_lang": lang,
            "pdf_info": all_pdf_info,
        }
        final_path.write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
        partial_path.unlink(missing_ok=True)
        return final_path

    print(
        f"   📘 OCR {rel_path}: {page_count} 页, {total_chunks} chunk(s), "
        f"输出到 {ocr_dir}"
        + (f", 日志 {log_path}" if log_path is not None else ""),
        flush=True,
    )
    t0 = time.time()

    chunk_start = next_page
    while chunk_start <= page_count:
        chunk_end = min(chunk_start + chunk_size - 1, page_count)
        chunk_idx = (chunk_start - 1) // chunk_size + 1
        print(
            f"      🔬 chunk {chunk_idx}/{total_chunks}: 页 {chunk_start}-{chunk_end}",
            flush=True,
        )
        chunk_info = _run_mineru_cli_chunk(
            pdf_path,
            chunk_start,
            chunk_end,
            device=device,
            lang=lang,
            virtual_vram=virtual_vram,
            min_batch_inference_size=min_batch_inference_size,
            render_threads=render_threads,
            log_path=log_path,
        )
        for entry in chunk_info:
            local_page_idx = int(entry.get("page_idx", 0))
            source_page = chunk_start + local_page_idx
            entry["_source_page"] = source_page
            entry["page_idx"] = source_page - 1
        all_pdf_info.extend(chunk_info)

        partial_payload = {
            "doc_id": doc_id,
            "source_rel_path": rel_path,
            "page_count": page_count,
            "ocr_lang": lang,
            "virtual_vram": virtual_vram,
            "mineru_min_batch_inference_size": min_batch_inference_size,
            "mineru_pdf_render_threads": render_threads,
            "completed_upto_page": chunk_end,
            "pdf_info": all_pdf_info,
        }
        partial_path.write_text(
            json.dumps(partial_payload, ensure_ascii=False),
            encoding="utf-8",
        )
        chunk_start = chunk_end + 1

    all_pdf_info.sort(key=lambda entry: int(entry.get("_source_page", entry.get("page_idx", 0) + 1)))
    final_payload = {
        "doc_id": doc_id,
        "source_rel_path": rel_path,
        "page_count": page_count,
        "ocr_lang": lang,
        "virtual_vram": virtual_vram,
        "mineru_min_batch_inference_size": min_batch_inference_size,
        "mineru_pdf_render_threads": render_threads,
        "pdf_info": all_pdf_info,
    }
    final_path.write_text(
        json.dumps(final_payload, ensure_ascii=False),
        encoding="utf-8",
    )
    partial_path.unlink(missing_ok=True)

    elapsed = time.time() - t0
    print(
        f"   ✅ OCR 完成: {rel_path} ({len(all_pdf_info)} 页, {elapsed:.1f}s)",
        flush=True,
    )
    return final_path


def collect_middle_json_index(mineru_output_dir: Path) -> list[dict[str, Any]]:
    """Collect metadata for final middle.json files without loading all pdf_info into memory."""
    results: list[dict[str, Any]] = []
    for mj in sorted(mineru_output_dir.rglob("*_middle.json")):
        try:
            entry = load_middle_json_index_entry(mj)
            results.append(entry)
            print(f"   📄 {entry['source_rel_path']}: {entry['page_count']} 页")
        except Exception as e:
            print(f"   ⚠️ 解析失败 {mj.name}: {e}")
    return results


def load_middle_json_index_entry(middle_json_path: Path) -> dict[str, Any]:
    data = json.loads(middle_json_path.read_text(encoding="utf-8"))
    pdf_info = data.get("pdf_info") or []
    if not pdf_info:
        raise ValueError(f"{middle_json_path} does not contain pdf_info")
    source_rel_path = (
        data.get("source_rel_path")
        or data.get("source_pdf_name")
        or middle_json_path.parent.parent.name
    )
    return {
        "path": middle_json_path,
        "doc_id": data.get("doc_id") or middle_json_path.stem.replace("_middle", ""),
        "source_rel_path": source_rel_path,
        "page_count": int(data.get("page_count") or len(pdf_info)),
    }


def load_pdf_info_from_middle_json(middle_json_path: Path) -> list[dict[str, Any]]:
    data = json.loads(middle_json_path.read_text(encoding="utf-8"))
    pdf_info = data.get("pdf_info") or []
    return sorted(
        pdf_info,
        key=lambda entry: int(entry.get("_source_page", entry.get("page_idx", 0) + 1)),
    )


def fingerprint_text(text: str) -> bytes:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).digest()


class IncrementalCorpusWriter:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_jsonl = output_dir / "cpt.jsonl"
        self.output_txt = output_dir / "all_akkadian_corpus.txt"
        self.exported_doc_ids_path = output_dir / "cpt_exported_doc_ids.txt"

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.output_jsonl.touch(exist_ok=True)
        if not self.output_txt.exists():
            self.output_txt.write_text("", encoding="utf-8")
        if self.output_jsonl.stat().st_size == 0:
            self.exported_doc_ids_path.write_text("", encoding="utf-8")
            self.output_txt.write_text("", encoding="utf-8")
        else:
            self.exported_doc_ids_path.touch(exist_ok=True)

        self.exported_doc_ids = {
            line.strip()
            for line in self.exported_doc_ids_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.seen_chunks: set[bytes] = set()
        self.seen_page_keys: set[str] = set()

        self.total_chunks = 0
        self.total_text_pages = 0
        self.total_jsonl_bytes_written = 0
        self.total_txt_bytes_written = self.output_txt.stat().st_size
        self.docs_exported_this_run = 0
        self.chunks_added_this_run = 0
        self.pages_added_this_run = 0
        self.jsonl_bytes_added_this_run = 0
        self.txt_bytes_added_this_run = 0

        with self.output_jsonl.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                self.total_chunks += 1
                self.total_jsonl_bytes_written += len(line.encode("utf-8"))
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = str(record.get("text", "")).strip()
                if text:
                    self.seen_chunks.add(fingerprint_text(text))
                doc_id = str(record.get("doc_id", "")).strip()
                source_page = record.get("source_page")
                if doc_id and source_page is not None:
                    self.seen_page_keys.add(f"{doc_id}:{int(source_page)}")
        self.total_text_pages = len(self.seen_page_keys)

        self._jsonl_fh = self.output_jsonl.open("a", encoding="utf-8")
        self._txt_fh = self.output_txt.open("a", encoding="utf-8")
        self._exported_docs_fh = self.exported_doc_ids_path.open("a", encoding="utf-8")

    def close(self) -> None:
        self._jsonl_fh.close()
        self._txt_fh.close()
        self._exported_docs_fh.close()

    def append_doc(self, doc: dict[str, Any]) -> dict[str, int]:
        doc_id = doc["doc_id"]
        source_rel_path = doc["source_rel_path"]
        if doc_id in self.exported_doc_ids:
            print(f"   ♻️  跳过已导出的语料: {source_rel_path}", flush=True)
            return {"pages": 0, "chunks": 0}

        pdf_info = load_pdf_info_from_middle_json(doc["path"])
        doc_chunks = 0
        doc_pages = 0
        for page_entry in pdf_info:
            page_text = extract_page_text(page_entry)
            cleaned = light_clean(page_text)
            if not cleaned:
                continue

            source_page = int(page_entry.get("_source_page", page_entry.get("page_idx", 0) + 1))
            page_key = f"{doc_id}:{source_page}"
            page_added = False

            for chunk_index, chunk in enumerate(sliding_window_split(cleaned), start=1):
                chunk_fp = fingerprint_text(chunk)
                if chunk_fp in self.seen_chunks:
                    continue
                self.seen_chunks.add(chunk_fp)

                record = {
                    "text": chunk,
                    "doc_id": doc_id,
                    "source_rel_path": source_rel_path,
                    "source_page": source_page,
                    "chunk_index": chunk_index,
                }
                jsonl_line = json.dumps(record, ensure_ascii=False) + "\n"
                self._jsonl_fh.write(jsonl_line)
                self._txt_fh.write(chunk + "\n")

                doc_chunks += 1
                self.total_chunks += 1
                self.total_jsonl_bytes_written += len(jsonl_line.encode("utf-8"))
                self.total_txt_bytes_written += len(chunk.encode("utf-8")) + 1
                self.chunks_added_this_run += 1
                self.jsonl_bytes_added_this_run += len(jsonl_line.encode("utf-8"))
                self.txt_bytes_added_this_run += len(chunk.encode("utf-8")) + 1
                page_added = True

            if page_added and page_key not in self.seen_page_keys:
                self.seen_page_keys.add(page_key)
                doc_pages += 1
                self.total_text_pages += 1
                self.pages_added_this_run += 1

        self._jsonl_fh.flush()
        self._txt_fh.flush()
        self._exported_docs_fh.write(doc_id + "\n")
        self._exported_docs_fh.flush()
        self.exported_doc_ids.add(doc_id)
        self.docs_exported_this_run += 1

        if doc_chunks:
            print(f"   ✏️ {source_rel_path}: {doc_chunks} chunks appended", flush=True)
        else:
            print(f"   📭 {source_rel_path}: no new chunks", flush=True)
        return {"pages": doc_pages, "chunks": doc_chunks}


# ============================================================================
# Text extraction from MinerU middle.json
# ============================================================================

def extract_page_text(page_entry: dict[str, Any]) -> str:
    """从单页 MinerU 输出提取所有文本（joining all blocks/lines/spans）。"""
    parts: list[str] = []
    for block in page_entry.get("preproc_blocks") or []:
        for line in block.get("lines") or []:
            for span in line.get("spans") or []:
                if isinstance(span, dict) and span.get("type") == "text":
                    content = span.get("content", "").strip()
                    if content:
                        parts.append(content)
    return " ".join(parts)


# ============================================================================
# Light cleaning
# ============================================================================

# 页眉/页脚常见模式
_PAGE_HEADER_RE = re.compile(
    r"^(?:\d{1,4}\s*$)|"                          # 纯页码 "123"
    r"^(?:PLATE|FIGURE|TABLE|INDEX|BIBLIOGRAPHY|"
    r"INTRODUCTION|PREFACE|CONTENTS|GLOSSARY|"
    r"APPENDIX|NOTES)\b",
    re.IGNORECASE,
)
# 控制字符
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


def light_clean(text: str) -> str:
    """轻量清洗：规范空白、去控制字符、去纯页码行。"""
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"\s+", " ", text).strip()
    # 去掉过短行
    if len(text) < MIN_LINE_CHARS:
        return ""
    # 去页眉/页脚
    if _PAGE_HEADER_RE.match(text):
        return ""
    return text


# ============================================================================
# Sliding window split (byte-level, for ByT5)
# ============================================================================

def sliding_window_split(text: str, max_bytes: int = MAX_BYTE_LEN,
                         overlap: int = SLIDING_OVERLAP) -> list[str]:
    """将文本按字节边界切分为重叠片段，在空格处切分避免断词。"""
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return [text] if len(encoded) >= MIN_CHUNK_BYTES else []

    chunks: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + max_bytes, len(encoded))
        # 在词边界回退
        if end < len(encoded):
            space_pos = encoded.rfind(b" ", start + max_bytes // 2, end)
            if space_pos > start:
                end = space_pos
        chunk = encoded[start:end].decode("utf-8", errors="ignore").strip()
        if len(chunk.encode("utf-8")) >= MIN_CHUNK_BYTES:
            chunks.append(chunk)
        # 下一段
        start = end - overlap if end < len(encoded) else len(encoded)
        if start <= (end - max_bytes):
            start = end
    return chunks


# ============================================================================
# Main pipeline
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Build CPT corpus from Old Assyrian PDFs via MinerU OCR")
    parser.add_argument("--skip-ocr", action="store_true", help="跳过 OCR，仅使用已有的 MinerU 输出")
    parser.add_argument("--pdf-dir", type=Path, default=PDF_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--device", type=str, default=MINERU_DEVICE)
    parser.add_argument("--lang", type=str, default=OCR_LANG)
    parser.add_argument("--ocr-chunk-size", type=int, default=0, help="每次 OCR 的页块大小；0 表示自动")
    parser.add_argument("--ocr-workers", type=int, default=0, help="并行 OCR worker 数；0 表示自动")
    parser.add_argument("--mineru-vram", type=int, default=0, help="传给 MinerU 的单进程显存上限；0 表示自动")
    parser.add_argument(
        "--mineru-min-batch-inference-size",
        type=int,
        default=0,
        help="设置环境变量 MINERU_MIN_BATCH_INFERENCE_SIZE；0 表示自动",
    )
    parser.add_argument(
        "--render-threads",
        type=int,
        default=0,
        help="MinerU PDF 渲染进程数；0 表示自动",
    )
    args = parser.parse_args()

    pdf_dir = args.pdf_dir
    output_dir = args.output_dir
    mineru_out = output_dir / "mineru_output"   # MinerU 直接输出目录
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.ocr_chunk_size < 0:
        raise ValueError("--ocr-chunk-size 不能为负数")
    if args.ocr_workers < 0:
        raise ValueError("--ocr-workers 不能为负数")
    if args.mineru_vram < 0:
        raise ValueError("--mineru-vram 不能为负数")
    if args.mineru_min_batch_inference_size < 0:
        raise ValueError("--mineru-min-batch-inference-size 不能为负数")
    if args.render_threads < 0:
        raise ValueError("--render-threads 不能为负数")

    # 1. 递归统计 PDF
    pdf_files = discover_pdf_files(pdf_dir)
    total_size_gb = sum(f.stat().st_size for f in pdf_files) / 1024**3
    print(f"📂 PDF 根目录: {pdf_dir}")
    print(f"   递归检索到 {len(pdf_files)} 个 PDF, 总计 {total_size_gb:.1f} GB")
    if not pdf_files:
        print("   ⚠️ 无 PDF 文件，退出")
        return

    manifest_path = output_dir / "pdf_manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for pdf_path in pdf_files:
            manifest.write(json.dumps({
                "doc_id": make_doc_id(pdf_dir, pdf_path),
                "source_rel_path": _doc_relative_path(pdf_dir, pdf_path),
                "size_bytes": pdf_path.stat().st_size,
            }, ensure_ascii=False) + "\n")
    print(f"   🧾 清单已写入: {manifest_path}")

    # 2. MinerU OCR（逐 PDF、分页分块，可恢复）
    t_total = time.time()
    failed_pdfs: list[tuple[str, str]] = []
    detected_vram_gb = detect_gpu_vram_gb(args.device)
    effective_vram = args.mineru_vram or detected_vram_gb
    effective_workers = args.ocr_workers or recommend_ocr_workers(args.device, effective_vram)
    effective_chunk_size = args.ocr_chunk_size or recommend_ocr_chunk_size(
        effective_vram,
        effective_workers,
    )
    effective_min_batch = (
        args.mineru_min_batch_inference_size
        or recommend_min_batch_inference_size(effective_vram, effective_workers)
    )
    effective_render_threads = args.render_threads or recommend_render_threads(
        effective_workers
    )
    ocr_log_dir = output_dir / "ocr_logs"
    ocr_log_dir.mkdir(parents=True, exist_ok=True)
    corpus_writer = IncrementalCorpusWriter(output_dir)
    try:
        if not args.skip_ocr:
            print(
                f"\n🚀 MinerU OCR 启动 (device={args.device}, lang={args.lang}, "
                f"ocr_chunk_size={effective_chunk_size}, workers={effective_workers}, "
                f"render_threads={effective_render_threads}, vram={effective_vram or 'auto'}, "
                f"min_batch={effective_min_batch})"
            )
            if detected_vram_gb is not None:
                print(f"   🖥️  检测到 GPU 显存: {detected_vram_gb} GB")
            print(f"   📝 语料动态追加到: {corpus_writer.output_jsonl}")
            _ensure_local_mineru_pipeline_assets(args.lang)
            if effective_workers == 1:
                for idx, pdf_path in enumerate(pdf_files, start=1):
                    rel_path = _doc_relative_path(pdf_dir, pdf_path)
                    print(f"\n[{idx}/{len(pdf_files)}] {rel_path}", flush=True)
                    try:
                        middle_json_path = run_mineru_pdf(
                            pdf_dir,
                            pdf_path,
                            mineru_out,
                            device=args.device,
                            lang=args.lang,
                            chunk_size=effective_chunk_size,
                            virtual_vram=effective_vram,
                            min_batch_inference_size=effective_min_batch,
                            render_threads=effective_render_threads,
                            log_dir=ocr_log_dir,
                        )
                        corpus_writer.append_doc(load_middle_json_index_entry(middle_json_path))
                    except Exception as exc:
                        failed_pdfs.append((rel_path, str(exc)))
                        print(f"   ❌ OCR 失败: {rel_path}: {exc}", flush=True)
            else:
                print(f"   🧵 启用并行 OCR worker: {effective_workers}")
                future_to_pdf: dict[Any, Path] = {}
                with ThreadPoolExecutor(max_workers=effective_workers) as executor:
                    for pdf_path in pdf_files:
                        future = executor.submit(
                            run_mineru_pdf,
                            pdf_dir,
                            pdf_path,
                            mineru_out,
                            device=args.device,
                            lang=args.lang,
                            chunk_size=effective_chunk_size,
                            virtual_vram=effective_vram,
                            min_batch_inference_size=effective_min_batch,
                            render_threads=effective_render_threads,
                            log_dir=ocr_log_dir,
                        )
                        future_to_pdf[future] = pdf_path

                    completed = 0
                    for future in as_completed(future_to_pdf):
                        pdf_path = future_to_pdf[future]
                        rel_path = _doc_relative_path(pdf_dir, pdf_path)
                        completed += 1
                        try:
                            middle_json_path = future.result()
                            corpus_writer.append_doc(load_middle_json_index_entry(middle_json_path))
                            print(f"   ✅ worker 完成 {completed}/{len(pdf_files)}: {rel_path}", flush=True)
                        except Exception as exc:
                            failed_pdfs.append((rel_path, str(exc)))
                            print(f"   ❌ OCR 失败: {rel_path}: {exc}", flush=True)
        else:
            print(f"\n⏭️ 跳过 OCR，使用已有输出: {mineru_out}")
            print(f"   📝 语料动态追加到: {corpus_writer.output_jsonl}")
            for doc in collect_middle_json_index(mineru_out):
                corpus_writer.append_doc(doc)

        # 3. 收集所有 middle.json
        print(f"\n📦 收集 MinerU 输出...")
        all_pdfs = collect_middle_json_index(mineru_out)
        total_pages = sum(doc["page_count"] for doc in all_pdfs)

        elapsed_total = time.time() - t_total
        print(f"\n{'='*60}")
        print(f"📄 OCR 完成: {len(all_pdfs)} 个 PDF, {total_pages} 页, {elapsed_total:.1f}s")
        if failed_pdfs:
            print(f"⚠️  OCR 失败 {len(failed_pdfs)} 个 PDF:")
            for rel_path, err in failed_pdfs:
                print(f"   - {rel_path}: {err}")

        print(f"\n{'='*60}")
        print(f"🎉 CPT 语料库构建完成！")
        print(f"   本轮新增文档: {corpus_writer.docs_exported_this_run}")
        print(f"   本轮新增有效页: {corpus_writer.pages_added_this_run}")
        print(f"   本轮新增 chunks: {corpus_writer.chunks_added_this_run}")
        print(f"   累计有效页:   {corpus_writer.total_text_pages}")
        print(f"   累计 chunks:  {corpus_writer.total_chunks}")
        print(f"   JSONL 大小:   {corpus_writer.total_jsonl_bytes_written / 1024 / 1024:.1f} MB")
        print(f"   TXT 大小:     {corpus_writer.total_txt_bytes_written / 1024 / 1024:.1f} MB")
        print(f"   JSONL 路径:   {corpus_writer.output_jsonl}")
        print(f"   TXT 路径:     {corpus_writer.output_txt}")
        print(f"{'='*60}")
    finally:
        corpus_writer.close()


if __name__ == "__main__":
    main()
