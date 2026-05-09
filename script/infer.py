#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
阿卡德语-英语翻译推理脚本
================================================================================
与 train.py 和 prepare_data.py 保持一致的预处理/后处理逻辑

主要特性：
1. 从模型目录加载 generation_config
2. 与训练一致的预处理
3. 符合官方要求的后处理
4. 支持模型融合（weight averaging）
5. 支持伪标签生成（Pseudo-Labeling）和过滤
================================================================================
"""

import json
import math
import os
import re
import statistics
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Dict, Tuple
from difflib import SequenceMatcher

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm

warnings.filterwarnings(
    "ignore",
    message=r"`num_beams` is set to 1\..*early_stopping.*",
    category=UserWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r"`num_beams` is set to 1\..*length_penalty.*",
    category=UserWarning,
)

# 从 prepare_data 导入核心函数，确保与 train.py 逻辑一致
try:
    from prepare_data import (
        load_lexicon,
        build_onomasticon,
        preprocess_transliteration,
    )
except ImportError:
    # 如果在 notebook 中运行且 prepare_data 内容已粘贴，则忽略导入错误
    pass

# ============================================================================
# 编码损坏修复（内嵌实现，不依赖 build_correction_vocab.py）
# ============================================================================

_CORR_VOWELS = set('aáàeéèiíìuúùAÁÀEÉÈIÍÌUÚÙ')


def is_corrupted_row(text: str) -> bool:
    """Detect transliteration rows with characteristic encoding corruption."""
    if not isinstance(text, str):
        return False
    markers = [
        '\u201e',   # „ corrupted hyphen
        '\u2026',   # … spurious ellipsis
        'mup-p',    # corrupted ṭup-p
        'me-+e',    # corrupted me-he
    ]
    return any(marker in text for marker in markers)


def _fix_encoding_corruption(text: str) -> str:
    """Reverse deterministic byte-level corruption before vocab lookup."""
    if not isinstance(text, str):
        return ""

    text = text.replace('\u201e', '-')
    text = re.sub(r'-{2,}', '-', text)
    text = text.replace('\u2026', '')
    text = re.sub(r'\bmup-p', 'ṭup-p', text)
    text = re.sub(r'(?<=[a-záàéèíìúù])-\+(?=[a-záàéèíìúù])', '-h', text)
    text = re.sub(r'(?<![a-záàéèíìúùA-ZÁÀÉÈÍÌÚÙ])aa(?![a-záàéèíìúùA-Z])', 'ša', text)
    return re.sub(r'\s+', ' ', text).strip()


def _generate_sha_candidates(token: str) -> list[str]:
    """Generate plausible š<-a reversal candidates and keep vocab to arbitrate."""
    positions = [i for i, ch in enumerate(token) if ch == 'a']
    if not positions:
        return []

    candidates: list[str] = []
    for pos in positions:
        candidates.append(token[:pos] + 'š' + token[pos + 1:])

    if 2 <= len(positions) <= 8:
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                chars = list(token)
                chars[positions[i]] = 'š'
                chars[positions[j]] = 'š'
                candidates.append(''.join(chars))

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


def _edit_distance(s1: str, s2: str) -> int:
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
    best = None
    best_dist = max_distance + 1
    best_freq = 0

    for word, freq in vocab.items():
        if abs(len(word) - len(token)) > max_distance:
            continue
        dist = _edit_distance(token, word)
        if dist <= max_distance and (dist < best_dist or (dist == best_dist and freq > best_freq)):
            best = word
            best_dist = dist
            best_freq = freq

    return best


def _suggest_vocab_corrections(
    text: str,
    vocab: dict[str, int],
    *,
    max_edit_distance: int = 0,
    min_token_len: int = 4,
) -> tuple[str, list[dict]]:
    corrections = []
    tokens = re.split(r'(\s+)', text)
    result = []

    for token in tokens:
        if not token.strip() or token.isspace():
            result.append(token)
            continue

        stripped = token.strip('.,;!?()[]{}"\' ')
        prefix = token[:len(token) - len(token.lstrip('.,;!?()[]{}"\' '))]
        suffix = token[len(token.rstrip('.,;!?()[]{}"\' ')):]

        if not stripped or stripped in vocab or re.match(r'^\d+[.,/]?\d*$', stripped):
            result.append(token)
            continue

        if '(' in stripped or ')' in stripped:
            result.append(token)
            continue

        best_candidate = None
        best_freq = 0
        for candidate in _generate_sha_candidates(stripped):
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

        if max_edit_distance > 0 and len(stripped) >= min_token_len:
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


def _correct_transliteration(
    text: str,
    vocab: dict[str, int],
    *,
    max_edit_distance: int = 0,
) -> tuple[str, list[dict]]:
    """Correction pipeline used at inference time when correction_vocab.json exists."""
    all_corrections = []

    if is_corrupted_row(text):
        original = text
        text = _fix_encoding_corruption(text)
        if text != original:
            all_corrections.append({
                'original': original,
                'corrected': text,
                'method': 'encoding_fix',
            })

    text, vocab_corrections = _suggest_vocab_corrections(
        text,
        vocab,
        max_edit_distance=max_edit_distance,
        min_token_len=4,
    )
    all_corrections.extend(vocab_corrections)
    return text, all_corrections

# ============================================================================
# Sumerogram 词义注入（内嵌实现，不依赖外部模块）
# ============================================================================

_SUMEROGRAM_TOKEN_RE = re.compile(
    r'(?<![a-záàéèíìúù])'
    r'([A-ZÁÀÉÈÍÌÚÙŠṢṬḪ][A-ZÁÀÉÈÍÌÚÙŠṢṬḪ₀-₉0-9.]+)'
    r'(?![a-záàéèíìúù])'
)
_SUMEROGRAM_HIGH_FREQ_SKIP = {'KÙ.BABBAR', 'DUMU', 'GÍN', 'GÍN.TA'}


def _load_sumerogram_glossary_json(path: str) -> dict:
    """Load glossary from JSON: {sumerogram: [akkadian, english]}."""
    import json as _json_sg
    with open(path, 'r', encoding='utf-8') as f:
        data = _json_sg.load(f)
    return {k: tuple(v) for k, v in data.items()}


def _inject_sumerogram_glossary_hints(text: str, glossary: dict, max_hints: int = 6) -> str:
    """Inject [GAL=chief; É.GAL=palace] prefix based on detected Sumerograms."""
    if not isinstance(text, str) or not text.strip() or not glossary:
        return text
    hints = []
    seen = set()
    for m in _SUMEROGRAM_TOKEN_RE.finditer(text):
        if len(hints) >= max_hints:
            break
        tok = m.group(1)
        if tok in seen or tok in _SUMEROGRAM_HIGH_FREQ_SKIP:
            continue
        seen.add(tok)
        entry = glossary.get(tok)
        if not entry:
            base = re.sub(r'[₀-₉0-9]+$', '', tok)
            entry = glossary.get(base)
        if entry:
            hints.append(f"{tok}={entry[1]}")
    if not hints:
        return text
    return "[" + "; ".join(hints) + "] " + text


# ============================================================================
# Silver 数据专用清洗逻辑
# ============================================================================

def clean_silver_specific_patterns(text: str) -> str:
    """
    清洗 silver_unlabeled.csv 中特有的脏数据模式
    
    这些模式在 train.csv 中不存在，需要定制化处理：
    1. 破碎的限定词标记：{d-<gap> → <gap>
    2. 其他类似的 {X-<gap> 模式
    """
    if not isinstance(text, str):
        return ""
    
    # 1. 破碎的限定词：{d-<gap> → <gap>
    # 这些是神名限定词后接破损，表示神名已不可读
    text = re.sub(r'\{[a-zA-Z]+-\s*<gap>', '<gap>', text, flags=re.I)
    
    # 2. 孤立的花括号开头（未闭合的限定词）：{X 后面没有 }
    # 例如：{d 后面直接是空格或结束
    # 这种情况较少见，保守处理：只处理明显的破损
    text = re.sub(r'\{[a-zA-Z]+-(?=\s|$)', '<gap>', text)
    
    # 3. 清理可能残留的空花括号
    text = re.sub(r'\{\s*\}', '', text)
    
    return text


# ============================================================================
# Notebook-style postprocessing
# ============================================================================

_ALLOWED_FRACS = [
    (1 / 6, "0.16666"),
    (1 / 4, "0.25"),
    (1 / 3, "0.33333"),
    (1 / 2, "0.5"),
    (2 / 3, "0.66666"),
    (3 / 4, "0.75"),
    (5 / 6, "0.83333"),
]
_FRAC_TOL = 2e-3
_FLOAT_RE = re.compile(r"(?<![\w/])(\d+\.\d{4,})(?![\w/])")


def _canon_decimal(x: float) -> str:
    ip = int(math.floor(x + 1e-12))
    frac = x - ip
    best = min(_ALLOWED_FRACS, key=lambda t: abs(frac - t[0]))

    if abs(frac - best[0]) <= _FRAC_TOL:
        dec = best[1]
        if ip == 0:
            return dec
        return f"{ip}{dec[1:]}" if dec.startswith("0.") else f"{ip}+{dec}"

    return f"{x:.5f}".rstrip("0").rstrip(".")


_WS_RE = re.compile(r"\s+")
_GAP_UNIFIED_RE = re.compile(
    r"<\s*big[\s_\-]*gap\s*>"
    r"|<\s*gap\s*>"
    r"|\bbig[\s_\-]*gap\b"
    r"|\bx(?:\s+x)+\b"
    r"|\.{3,}|…+|\[\.+\]"
    r"|\[\s*x\s*\]|\(\s*x\s*\)"
    r"|(?<!\w)x{2,}(?!\w)"
    r"|(?<!\w)x(?!\w)"
    r"|\(\s*large\s+break\s*\)"
    r"|\(\s*break\s*\)"
    r"|\(\s*\d+\s+broken\s+lines?\s*\)",
    re.I,
)


def _normalize_gaps_vec(ser: pd.Series) -> pd.Series:
    return ser.str.replace(_GAP_UNIFIED_RE, "<gap>", regex=True)


_EXACT_FRAC_RE = re.compile(r"0\.8333|0\.6666|0\.3333|0\.1666|0\.625|0\.75|0\.25|0\.5")
_EXACT_FRAC_MAP = {
    "0.8333": "⅚",
    "0.6666": "⅔",
    "0.3333": "⅓",
    "0.1666": "⅙",
    "0.625": "⅝",
    "0.75": "¾",
    "0.25": "¼",
    "0.5": "½",
}


def _frac_repl(m: re.Match) -> str:
    return _EXACT_FRAC_MAP[m.group(0)]


_SOFT_GRAM_RE = re.compile(
    r"\(\s*(?:fem|plur|pl|sing|singular|plural|\?|\!)"
    r"(?:\.\s*(?:plur|plural|sing|singular))?"
    r"\.?\s*[^)]*\)",
    re.I,
)
_BARE_GRAM_RE = re.compile(r"(?<!\w)(?:fem|sing|pl|plural)\.?(?!\w)\s*", re.I)
_UNCERTAIN_RE = re.compile(r"\(\?\)")
_CURLY_QUOTES_RE = re.compile("[\u201c\u201d\u2018\u2019]")

_MONTH_RE = re.compile(r"\bMonth\s+(XII|XI|X|IX|VIII|VII|VI|V|IV|III|II|I)\b", re.I)
_ROMAN2INT = {"I": 1, "II": 2, "III": 3, "IV": 4, "V": 5, "VI": 6, "VII": 7, "VIII": 8, "IX": 9, "X": 10, "XI": 11, "XII": 12}

_REPEAT_WORD_RE = re.compile(r"\b(\w+)(?:\s+\1\b)+")
_REPEAT_PUNCT_RE = re.compile(r"([.,])\1+")
_PUNCT_SPACE_RE = re.compile(r"\s+([.,:])")

_FORBIDDEN_TRANS = str.maketrans("", "", '()——<>⌈⌋⌊[]+ʾ;')

_COMMODITY_RE = re.compile(r'-(gold|tax|textiles)\b')
_COMMODITY_REPL = {
    "gold": "pašallum gold",
    "tax": "šadduātum tax",
    "textiles": "kutānum textiles",
}


def _commodity_repl(m: re.Match) -> str:
    return _COMMODITY_REPL[m.group(1)]


_SHEKEL_REPLS = [
    (re.compile(r'5\s+11\s*/\s*12\s+shekels?', re.I), '6 shekels less 15 grains'),
    (re.compile(r'5\s*/\s*12\s+shekels?', re.I), '⅔ shekel 15 grains'),
    (re.compile(r'7\s*/\s*12\s+shekels?', re.I), '½ shekel 15 grains'),
    (re.compile(r'1\s*/\s*12\s*(?:\(shekel\)|\bshekel)?', re.I), '15 grains'),
]

_SLASH_ALT_RE = re.compile(r'(?<!\d)\s*/\s*(?!\d)\S+')
_STRAY_MARKS_RE = re.compile(r'<<[^>]*>>|<(?!gap\b)[^>]*>')
_MULTI_GAP_RE = re.compile(r'(?:<gap>\s*){2,}')
_PN_RE = re.compile(r"\bPN\b")


def _month_repl(m: re.Match) -> str:
    return f"Month {_ROMAN2INT.get(m.group(1).upper(), m.group(1))}"


class VectorizedPostprocessor:
    """Notebook-aligned output normalizer applied before candidate selection."""

    def postprocess_batch(self, translations: List[str]) -> List[str]:
        s = pd.Series(translations).fillna("").astype(str)

        s = _normalize_gaps_vec(s)
        s = s.str.replace(_PN_RE, "<gap>", regex=True)
        s = s.str.replace(_COMMODITY_RE, _commodity_repl, regex=True)

        for pat, repl in _SHEKEL_REPLS:
            s = s.str.replace(pat, repl, regex=True)

        s = s.str.replace(_EXACT_FRAC_RE, _frac_repl, regex=True)
        s = s.str.replace(_FLOAT_RE, lambda m: _canon_decimal(float(m.group(1))), regex=True)

        s = s.str.replace(_SOFT_GRAM_RE, " ", regex=True)
        s = s.str.replace(_BARE_GRAM_RE, " ", regex=True)
        s = s.str.replace(_UNCERTAIN_RE, "", regex=True)

        s = s.str.replace(_STRAY_MARKS_RE, "", regex=True)
        s = s.str.replace(_SLASH_ALT_RE, "", regex=True)
        s = s.str.replace(_CURLY_QUOTES_RE, "", regex=True)

        s = s.str.replace(_MONTH_RE, _month_repl, regex=True)
        s = s.str.replace(_MULTI_GAP_RE, "<gap>", regex=True)

        s = s.str.replace("<gap>", "\x00GAP\x00", regex=False)
        s = s.str.translate(_FORBIDDEN_TRANS)
        s = s.str.replace("\x00GAP\x00", " <gap> ", regex=False)

        s = s.str.replace(_REPEAT_WORD_RE, r"\1", regex=True)
        for n in range(4, 1, -1):
            pat = r"\b((?:\w+\s+){" + str(n - 1) + r"}\w+)(?:\s+\1\b)+"
            s = s.str.replace(pat, r"\1", regex=True)

        s = s.str.replace(_PUNCT_SPACE_RE, r"\1", regex=True)
        s = s.str.replace(_REPEAT_PUNCT_RE, r"\1", regex=True)
        s = s.str.replace(_WS_RE, " ", regex=True).str.strip()

        return s.tolist()

    def postprocess_one(self, translation: str) -> str:
        return self.postprocess_batch([translation])[0]


def fast_byt5_batch_decode(
    sequences,
    *,
    tokenizer,
    chunk_size: int = 32,
) -> List[str]:
    """Fast local decode for ByT5 byte IDs."""
    if torch.is_tensor(sequences):
        sequences = sequences.detach().cpu()
        if sequences.ndim == 1:
            rows = [sequences.numpy()]
        else:
            rows = [row.numpy() for row in sequences]
    elif isinstance(sequences, np.ndarray):
        if sequences.dtype != object:
            if sequences.ndim == 1:
                rows = [sequences]
            else:
                rows = [row for row in sequences]
        else:
            rows = [np.asarray(row) for row in sequences.tolist()]
    else:
        try:
            rows = [np.asarray(row) for row in sequences]
        except TypeError:
            rows = [np.asarray(sequences)]
        if rows and rows[0].ndim == 0:
            rows = [np.asarray(sequences)]

    offset = int(getattr(tokenizer, "offset", 3))
    utf_vocab_size = int(getattr(tokenizer, "_utf_vocab_size", 256))
    low = offset
    high = offset + utf_vocab_size
    decoded: List[str] = []
    total = len(rows)

    for start in range(0, total, chunk_size):
        batch = rows[start:start + chunk_size]
        for row in batch:
            row = np.asarray(row).reshape(-1)
            valid = row[(row >= low) & (row < high)]
            if valid.size == 0:
                decoded.append("")
                continue
            b = (valid - offset).astype(np.uint8, copy=False).tobytes()
            decoded.append(b.decode("utf-8", errors="ignore"))

    return decoded


def decode_sequences(sequences, *, tokenizer) -> List[str]:
    if tokenizer.__class__.__name__ == "ByT5Tokenizer":
        return fast_byt5_batch_decode(sequences, tokenizer=tokenizer)
    return tokenizer.batch_decode(
        sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

# ============================================================================
# 依赖函数
# ============================================================================
# 注意：本脚本假设 prepare_data.py 中的函数已经在当前上下文（Global Scope）中定义。
# 这通常是因为在 Jupyter Notebook 中，将 prepare_data.py 的代码与本脚本放在了同一个 Cell 或之前的 Cell 中运行。

# ============================================================================
# 配置
# ============================================================================

class InferenceConfig:
    """推理配置"""
    
    def __init__(self):
        self._config_sources = {}
        self._training_config_path = None

        # 自动检测运行环境
        self.is_kaggle = os.path.exists("/kaggle/input")
        
        # 路径配置（自动适配 Kaggle 和本地环境）
        if self.is_kaggle:
            # Kaggle 云端环境
            self.data_dir = "/kaggle/input/competitions/deep-past-initiative-machine-translation"
            self.test_csv = os.path.join(self.data_dir, "test.csv")
            self.output_csv = "/kaggle/working/submission.csv"
            # Fallback: hardcoded kagglehub cache paths
            if not os.path.exists(self.test_csv):
                for _p in [
                    "/root/.cache/kagglehub/datasets/bcdefga/test1236666/versions/2/all_sources_test_format.csv",
                    "/root/.cache/kagglehub/datasets/bcdefga/test1236666/versions/2/test.csv",
                ]:
                    if os.path.exists(_p):
                        self.test_csv = _p
                        self.data_dir = os.path.dirname(_p)
                        break
            # 自动查找模型目录（与 input 同级）
            self.model_paths = self._find_model_paths_kaggle()
            # 默认不开启伪标签模式
            self.pseudo_label_mode = False
            self.mrt_data_mode = False
        else:
            # 本地环境
            self.data_dir = "/data/lsb/deep_past/data"
            self.test_csv = os.path.join(self.data_dir, "test.csv") # 默认测试集，伪标签模式下会被覆盖
            self.output_csv = "submission.csv"
            self.local_mode = "standard"  # "standard" / "pseudo_label" / "mrt_data" / "score"

            # === 本地模式配置 (用户指定) ===
            self.model_paths = ["/data/lsb/deep_past/output/model_20260313_161037/checkpoint-9400"]

            self.pseudo_label_mode = self.local_mode == "pseudo_label"
            self.mrt_data_mode = self.local_mode == "mrt_data"
            self.score_mode = self.local_mode == "score"

            # Score 模式：对有 ground truth 的数据进行推理并计算指标
            # 输入 CSV 需要 transliteration + translation 列
            self.score_input_csv = ""  # 设置为非空路径以启用
            self.score_output_csv = ""  # 评分结果输出路径（留空则自动生成）

            self.input_silver_csv = os.path.join(self.data_dir, "train_clean_ocr_merged.csv")
            self.output_silver_raw = os.path.join(self.data_dir, "silver_pseudo_labels_raw.csv")
            self.output_silver_filtered = os.path.join(self.data_dir, "silver_pseudo_labels_filtered.csv")

            self.input_mrt_csv = os.path.join(self.data_dir, "train_clean_ocr_merged.csv")
            self.output_mrt_raw = os.path.join(self.data_dir, "mrt_candidates_raw.jsonl")
            self.output_mrt_filtered = os.path.join(self.data_dir, "mrt_candidates_filtered.jsonl")
            self.mrt_min_unique_candidates = 2

            # 过滤阈值
            self.filter_min_confidence = -1.5 # Beam 模式: Log probability 阈值 (负数，越接近 0 越好)
            self.filter_min_mbr_score = 55.0  # MBR 模式: 共识分数阈值 (weighted 模式可能略高于 100)
            self.filter_min_len_ratio = 0.6
            self.filter_max_len_ratio = 2.0
            
        
        
        # 自动扩展模型路径：如果是一个父目录且包含 fold0, fold1...，则自动扫描加入
        self.model_paths = self._expand_model_paths(self.model_paths)

        self.model_weights = [1.0] * len(self.model_paths)

        # 融合权重导出：将 ensemble/merged 权重保存到 output 目录（默认关闭，仅 Kaggle 云端可用）
        self.export_merged_model = False

        # 解码策略
        # "beam":            标准 beam search（快，稳定）
        # "sampling_mbr":    单模型采样 + MBR 共识
        # "hybrid_mbr":      单模型混合候选 MBR（beam + sampling，先做 Model Soup）
        # "multi_model_mbr": 多模型顺序采样 + 联合 MBR 共识（默认，精度优先）
        self.decode_strategy = "hybrid_mbr"   # "beam" / "sampling_mbr" / "hybrid_mbr" / "multi_model_mbr"

        # MBR / hybrid 高频参数
        self.mbr_num_candidates = 20    # 候选池大小
        self.mbr_temperature = 0.8      # sampling_mbr: 采样温度
        self.mbr_top_p = 0.92           # sampling_mbr: nucleus sampling 阈值
        self.multi_model_beam_cands = 4 # multi_model_mbr: 每个模型的标准 beam 候选数
        self.multi_model_sample_cands = 6
        self.sample_temperatures = [0.8]
        self.num_sample_per_temp = 6
        self.hybrid_beam_cands = 4
        self.hybrid_sample_cands = 6
        self.hybrid_temperature = 0.8
        self.hybrid_top_p = 0.92
        self.use_adaptive_beams = True
        self.adaptive_beam_len_threshold = 100
        self.mbr_pool_cap = 32
        self.mbr_w_chrf = 0.55
        self.mbr_w_bleu = 0.25
        self.mbr_w_jaccard = 0.20
        self.mbr_w_length = 0.10
        self.mbr_metric = "geom"    # "geom" / "chrf" / "weighted" / "bleu"
        
        # 生成参数 (设为 None 时将自动从 model_dir/training_config.json 中加载)
        self.num_beams = None             
        self.max_new_tokens = 512         
        self.length_penalty = 1.3     
        self.repetition_penalty = 1.2
        self.no_repeat_ngram_size = None
        self.early_stopping = None        
        self.use_generation_config = True  # 从模型目录加载
        
        # 批处理（Kaggle T4×2: batch_size=4 安全；本地 RTX 5090: batch_size=10）
        self.batch_size = 16 if not self.is_kaggle else 6
        self.max_input_length = 512   # 必须与 train.py 的 max_input_length 一致
        self.max_target_length = 512  # 必须与 train.py 的 max_target_length 一致
        
        # 任务前缀
        self.task_prefix = "translate Old Assyrian to English: "

        # Meta prefix: 极简 2 字母前缀 [source][completeness]
        # None = 从 training_config.json 自动继承；True/False = 手动覆盖
        self.use_meta_prefix = None
        
        # 硬件
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.num_workers = 2
        self.use_better_transformer = True  # 启用 BetterTransformer 加速推理
        
        # 词典配置
        self.lexicon_csv = os.path.join(self.data_dir, "OA_Lexicon_eBL.csv")
        
        # 专名处理模式: "hint" / "placeholder" / "none"（将从 training_config 自动继承）
        self.name_handling_mode = None
        self.max_hints_per_sample = None
        self.use_month_hints = None  # 月份 hint 子开关（从 training_config 继承）
        
        # 推理精度 ("fp32" / "fp16" / "bf16")
        self.dtype = "fp32"
        
        # 文档级上下文（从 training_config.json 自动继承，默认关闭）
        self.use_doc_context = None

        # RAG 示例注入（从 training_config.json 自动继承）
        self.use_rag = None
        self.rag_max_examples = None
        self.rag_min_translation_len = None
        self.rag_max_example_bytes = None
        self.rag_budget_ratio = None
        self.rag_train_csv = None  # 推理时用于构建 RAG 索引的训练数据路径
        
        # 编码损坏修复（仅从模型目录加载 correction_vocab.json；找不到即关闭）
        self.use_correction_vocab = True
        self.correction_vocab_filename = "correction_vocab.json"

        # Sumerogram 词义注入（默认由 training_config.json 决定；缺失时视为关闭）
        self.use_sumerogram_glossary = None
        self.sumerogram_glossary_filename = "sumerogram_glossary.json"
        self.max_sumerogram_hints = None

        # 动态 Batching 排序方向
        # True  = 长序列先推理（早暴露显存峰值，避免后期 OOM）
        # False = 短序列先推理（默认，Padding 效率更高）
        self.sort_longest_first = False
        self.progress_log_every = 20
        self.enable_stage_timing_logs = True
        self.stage_timing_log_every = 20

    def get_multi_model_sampling_plan(self) -> Tuple[List[float], int]:
        """返回 multi_model_mbr 的采样计划：(temperatures, samples_per_temp)。"""
        temps = []
        for value in getattr(self, "sample_temperatures", []) or []:
            try:
                temps.append(float(value))
            except (TypeError, ValueError):
                continue

        per_temp = int(getattr(self, "num_sample_per_temp", 0) or 0)
        if temps and per_temp > 0:
            return temps, per_temp

        fallback = int(getattr(self, "multi_model_sample_cands", 0) or 0)
        if fallback > 0:
            return [float(self.mbr_temperature)], fallback

        return [], 0
        

    def _expand_model_paths(self, paths):
        expanded = []
        for p in paths:
            if os.path.exists(p):
                # 检查是否有 fold 子目录
                folds = [os.path.join(p, d) for d in os.listdir(p) if d.startswith("fold") and os.path.isdir(os.path.join(p, d))]
                if folds:
                    folds.sort() # fold0, fold1...
                    expanded.extend(folds)
                    print(f"🔍 扫描到 {p} 包含 {len(folds)} 个 fold 子模型，已自动加入推理列表。")
                else:
                    expanded.append(p)
            else:
                expanded.append(p)
        return expanded

    def load_training_config(self):
        """尝试从当前模型包加载 training_config.json 并覆盖推理配置。"""
        import json
        if not hasattr(self, 'model_paths') or not self.model_paths:
            return
            
        # 以第一个模型的配置为准
        model_dir = self.model_paths[0]
        config_path = _find_first_model_asset(
            [model_dir],
            "training_config.json",
            feature_name="training_config",
        )

        if config_path:
            try:
                self._training_config_path = config_path
                with open(config_path, "r", encoding="utf-8") as f:
                    train_cfg = json.load(f)
                
                print(f"\n📥 找到 training_config.json: {config_path}")
                print(f"   正在同步配置 (优先保留代码中非 None 的设定)...")
                
                # 覆盖关键数据流参数映射 {推断属性: 训练属性}
                keys_to_sync = {
                    "max_input_length": "max_input_length",
                    "max_new_tokens": "max_target_length",
                    "name_handling_mode": "name_handling_mode",
                    "max_hints_per_sample": "max_hints_per_sample",
                    "use_month_hints": "use_month_hints",
                    "use_sumerogram_glossary": "use_sumerogram_glossary",
                    "max_sumerogram_hints": "max_sumerogram_hints",
                    "num_beams": "num_beams",
                    "length_penalty": "length_penalty",
                    "repetition_penalty": "repetition_penalty",
                    "no_repeat_ngram_size": "no_repeat_ngram_size",
                    "use_rag": "use_rag",
                    "rag_max_examples": "rag_max_examples",
                    "rag_min_translation_len": "rag_min_translation_len",
                    "rag_max_example_bytes": "rag_max_example_bytes",
                    "rag_budget_ratio": "rag_budget_ratio",
                    "early_stopping": "generate_early_stopping",
                    "dtype": "dtype",
                    "use_meta_prefix": "use_meta_prefix",
                    "use_doc_context": "use_doc_context",
                }
                default_false_if_missing = {
                    "use_month_hints",
                    "use_sumerogram_glossary",
                    "use_rag",
                    "use_meta_prefix",
                    "use_doc_context",
                }
                default_values_if_missing = {
                    "max_sumerogram_hints": 6,
                }
                
                for infer_k, train_k in keys_to_sync.items():
                    if train_k in train_cfg:
                        current_val = getattr(self, infer_k, None)
                        if current_val is None:
                            setattr(self, infer_k, train_cfg[train_k])
                            self._config_sources[infer_k] = f"training_config:{train_k}"
                            print(f"   ✅ [加载训练参数] {infer_k}: None -> {train_cfg[train_k]}")
                        else:
                            self._config_sources[infer_k] = f"code_override(train={train_cfg[train_k]})"
                            print(f"   📌 [保留代码设定] {infer_k}: {current_val} (覆盖训练参数 {train_cfg[train_k]})")
                    else:
                        current_val = getattr(self, infer_k, None)
                        if infer_k in default_false_if_missing and current_val is None:
                            setattr(self, infer_k, False)
                            self._config_sources[infer_k] = f"default_off(training_config missing: {train_k})"
                            print(f"   ℹ️ [训练配置缺失] {infer_k}: None -> False (默认关闭)")
                        elif infer_k in default_values_if_missing and current_val is None:
                            default_value = default_values_if_missing[infer_k]
                            setattr(self, infer_k, default_value)
                            self._config_sources[infer_k] = f"default_value({default_value}; training_config missing: {train_k})"
                            print(f"   ℹ️ [训练配置缺失] {infer_k}: None -> {default_value} (使用安全默认值)")
                        else:
                            self._config_sources.setdefault(infer_k, "code_default")

                print(f"   开关解析:")
                for key in [
                    "name_handling_mode",
                    "use_month_hints",
                    "use_sumerogram_glossary",
                    "max_sumerogram_hints",
                    "use_rag",
                ]:
                    print(
                        f"   · {key} = {getattr(self, key, None)} "
                        f"| 来源: {self._config_sources.get(key, 'code_default')}"
                    )
            except Exception as e:
                print(f"\n⚠️ 加载 training_config.json 失败: {e}")
        else:
            print(f"\n⚠️ 未在模型包内找到 training_config.json，将完全依赖代码硬编码。")
            for key, value in {
                "use_month_hints": False,
                "use_sumerogram_glossary": False,
                "use_rag": False,
                "max_sumerogram_hints": 6,
            }.items():
                if getattr(self, key, None) is None:
                    setattr(self, key, value)
                    self._config_sources[key] = "default_without_training_config"
            self._config_sources.setdefault("name_handling_mode", "code_default")

    def _find_model_paths_kaggle(self) -> list:
        """
        在 Kaggle 环境自动查找模型目录
        查找规则：以 byt5-akkadian- 开头的目录，递归搜索 config.json
        支持 Kaggle Models 的版本目录结构 (e.g., /kaggle/input/{slug}/pytorch/default/1/)
        """
        model_paths = []
        kaggle_input = "/kaggle/input"
        model_prefixes = ("byt5-akkadian-", "deep-past-akkadian-", "akkadian-")
        
        if not os.path.exists(kaggle_input):
             return []
        
        def find_config_recursive(path, depth=0, max_depth=5):
            """递归查找包含模型文件的目录"""
            if depth > max_depth:
                return
            if not os.path.isdir(path):
                return
            
            # 支持 config.json 或 model.safetensors 作为模型目录标识
            model_indicators = ["config.json", "model.safetensors", "pytorch_model.bin"]
            if any(os.path.exists(os.path.join(path, f)) for f in model_indicators):
                model_paths.append(path)
                print(f"   ✓ 发现模型: {path}")
                return
            
            # 递归搜索子目录
            try:
                for item in os.listdir(path):
                    item_path = os.path.join(path, item)
                    if os.path.isdir(item_path):
                        find_config_recursive(item_path, depth + 1, max_depth)
            except PermissionError:
                pass

        # Search /kaggle/input (prefix-filtered) + kagglehub cache (unfiltered)
        _search_dirs = [
            ("/kaggle/input", True),
            ("/kaggle/input/datasets", False),
            (os.path.expanduser("~/.cache/kagglehub/datasets"), False),
        ]
        for base_dir, use_prefix in _search_dirs:
            if not os.path.exists(base_dir):
                continue
            try:
                for item in os.listdir(base_dir):
                    if use_prefix and not any(item.startswith(p) for p in model_prefixes):
                        continue
                    find_config_recursive(os.path.join(base_dir, item))
            except PermissionError:
                pass
        
        if not model_paths:
            print("   ⚠ 未找到模型目录")
            # 列出可用目录帮助调试
            try:
                for base_dir in ["/kaggle/input", "/kaggle/input/datasets"]:
                    if not os.path.exists(base_dir):
                        continue
                    items = os.listdir(base_dir)
                    print(f"   [{base_dir}] 可用目录: {items}")
                    for item in items:
                        if base_dir == "/kaggle/input" and not any(item.startswith(p) for p in model_prefixes):
                            continue
                        item_path = os.path.join(base_dir, item)
                        sub_items = os.listdir(item_path)
                        print(f"   📁 {item}/ 内容: {sub_items}")
                        # 继续打印更深层
                        for sub in sub_items:
                            sub_path = os.path.join(item_path, sub)
                            if os.path.isdir(sub_path):
                                sub2_items = os.listdir(sub_path)
                                print(f"      📁 {sub}/ 内容: {sub2_items}")
                                # 再深一层
                                for sub2 in sub2_items:
                                    sub2_path = os.path.join(sub_path, sub2)
                                    if os.path.isdir(sub2_path):
                                        print(f"         📁 {sub2}/ 内容: {os.listdir(sub2_path)}")
            except Exception as e:
                print(f"   调试信息获取失败: {e}")
        
        return model_paths


# ============================================================================
# 预处理函数（复用 prepare_data.py）
# ============================================================================
# HINTS GENERATION (O(1) Hash Lookup with Morphological Fallback)
# ============================================================================

_HINT_MIN_FORM_LEN = 4  # 过短的 key 极易误匹配，跳过

# 已知阿卡德语语法后缀（只允许剥离这些，不做任意音节剥离）
_AKKADIAN_SUFFIXES = {
    'ma',   # 强调词
    'ni',   # 从句标记
    'šu',   # “他的”（物主代词）
    'ša',   # “她的”
    'a',    # 属格 / 主格变体
    'im',   # 属格
    'um',   # 主格
    'am',   # 宾格
    'kà',   # “你的”
    'ia',   # “我的”
    'ka',   # “你的”（变体）
    'at',   # 阴性标记
    'ti',   # “我的”（变体）
}

# 阿卡德语高频功能词黑名单（这些词即使在词典里也不应作为 HINTS）
_HINT_STOPFORMS = {
    # 介词 / 连词
    "a-na", "um-ma", "i-na", "ù", "ú", "u",
    "ša", "ša-ma", "ki-ma", "ki-i", "ki",
    "la", "ul", "ul-ma",
    # 代词
    "a-ta", "a-ti",  # you (m/f)
    "a-tù-nu",       # you (pl)
    "a-hu-a",        # my brothers
    "a-ma", "i-ma",
    "ma-ma", "ma-ma-an",
    "a-ba", "a-bi",
    "šu-ma", "šu-ut",
    "i-li", "i-la",
    # 动词形式
    "a-dí-in",       # I gave
    "i-dí-in",       # he gave
    # 所有格（my brother / my father）
    "a-hi-a",        # my brother
    "a-bi-a",        # my father
    # 计量单位（极高频且词典里有同形专名）
    "ma-na",         # mina (重量单位)
    "ma-lá",         # 常见副词，非专名
    # 楔形文字限定词/质量词
    "sig5",          # good quality (限定词)
    # 常见商业词汇（布料/月份/法律用语）
    "ku-ta-nim",     # kutānum (布料单位，非专名)
    "sà-ra-tim",     # sarrātim (月份名，非专名)
    "ba-a-nim",      # bā'ānim (律师/代理人，非专名)
    "a-le-e",        # alê (疢问词"where is"，非专名)
    "am-ra-ma",      # amrāma (祝愿词"look!"，非专名)
    "a-lá-ni",      # alānē (城市复数，非专名)
}

# 月份音译形式 → 月份编号（官方月份表）
_MONTH_TRANSLITERATION_FORMS = {
    'bé-el-tí-é.gal-lim': 1, 'be-el-té-é.gal-lim': 1,
    'ša sá-ra-tim': 2, 'ke-na-tim': 3,
    'ma-hu-ur-dingir': 4, 'ma-ḫu-ur-i-lí': 4,
    'áb-ša-ra-ni': 5, 'áb ša-ra-ni': 5, 'áb-ša-ra-nu': 5,
    'hu-bu-ur': 6, 'ṣí-ip-im': 7,
    'qá-ra-a-tí': 8, 'qá-ra-a-tim': 8,
    'kán-wa-ar-ta': 9, 'kán-wa-ar-ta-an': 9, 'kán-wár-ta': 9, 'kán-mar-ta': 9,
    'kán-bar-ta': 9, 'kà-an-ma-ar-ta': 9,
    'té-i-na-tim': 10, 'ku-zal-li': 11, 'ku-zal-lu': 11,
    'a-lá-na-tum': 12, 'a-lá-na-tim': 12,
}

def _apply_month_hints(text: str) -> str:
    """检测音译中的阿卡德语月名形式，添加月份 hint。"""
    if not isinstance(text, str) or not text.strip():
        return text
    text_lower = text.lower()
    month_hints = []
    for form, month_num in _MONTH_TRANSLITERATION_FORMS.items():
        if form.lower() in text_lower:
            month_hints.append(f"{form}=month {month_num}")
    if month_hints and '## MONTH:' not in text:
        text = f"{text} ## MONTH: {'; '.join(month_hints)}"
    return text

def _apply_hints(text: str, lexicon: dict, max_hints: int = 10,
                  enable_place_names: bool = False) -> str:
    """基于限定词（Determinatives）的精准 Hint 系统
    
    只提取被限定词明确标记的专名，避免把动词词根误认为人名。
    支持的限定词：{d}(神名)、DUMU(父名)、KIŠIB(印章名)
    {ki}(地名) 默认关闭，由 enable_place_names 控制。
    
    策略："宁漏掉一万，不看错一个"
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if "## HINTS:" in text or not lexicon:
        return text
        
    hints = []
    matched_forms = set()
    words = text.split()
    n = len(words)
    
    def _try_match(word_str):
        """Suffix-Aware 查词典：只对限定词相邻的词做后缀剥离"""
        clean = word_str.strip('.,;:?!()[]"\'>').lower()
        if not clean or len(clean) < _HINT_MIN_FORM_LEN or clean in _HINT_STOPFORMS:
            return None
        
        candidates = [clean]
        if '-' in clean:
            parts = clean.split('-')
            stem_parts = parts[:]
            for _ in range(min(2, len(parts) - 1)):
                if stem_parts[-1] in _AKKADIAN_SUFFIXES:
                    stem_parts = stem_parts[:-1]
                    stem = '-'.join(stem_parts)
                    if stem != clean:
                        candidates.append(stem)
                else:
                    break
        
        for cand in candidates:
            if len(cand) >= _HINT_MIN_FORM_LEN and cand not in _HINT_STOPFORMS:
                if cand in lexicon and cand not in matched_forms:
                    return cand
        return None
    
    for i, word in enumerate(words):
        if len(hints) >= max_hints:
            break
        
        # --- 限定词规则 ---
        
        # 1. {d}+专名：神名限定词，后面的词是神名
        if word == '{d}' and i + 1 < n:
            cand = _try_match(words[i + 1])
            if cand:
                hints.append(f"{cand}={lexicon[cand]}")
                matched_forms.add(cand)
            continue
        
        # 2. 词+{ki}：地名限定词（默认关闭，由 enable_place_names 控制）
        if '{ki}' in word:
            if enable_place_names:
                place = word.replace('{ki}', '').strip('.,;:?!()[]"\'>').lower()
                if place and len(place) >= _HINT_MIN_FORM_LEN and place not in _HINT_STOPFORMS:
                    if place in lexicon and place not in matched_forms:
                        hints.append(f"{place}={lexicon[place]}")
                        matched_forms.add(place)
            continue
        
        # 3. DUMU+专名：“son of”，后面的词很可能是人名
        if word == 'DUMU' and i + 1 < n:
            cand = _try_match(words[i + 1])
            if cand:
                hints.append(f"{cand}={lexicon[cand]}")
                matched_forms.add(cand)
            continue
        
        # 4. KIŠIB+专名：“seal of”，后面的词很可能是人名
        if word == 'KIŠIB' and i + 1 < n:
            cand = _try_match(words[i + 1])
            if cand:
                hints.append(f"{cand}={lexicon[cand]}")
                matched_forms.add(cand)
            continue

    if hints:
        return f"{text} ## HINTS: {'; '.join(hints)}"
    return text


def strip_input_artifacts(text: str) -> str:
    """剥离模型输出中可能回显的输入端标记（HINTS / RAG 前缀 / task prefix）。"""
    if not isinstance(text, str):
        return text
    # 1. 剥离 ## HINTS: / ## MONTH: 及其后面的所有内容
    if '## HINTS:' in text:
        text = text[:text.index('## HINTS:')]
    if '## MONTH:' in text:
        text = text[:text.index('## MONTH:')]
    # 2. 剥离 RAG 前缀回显（格式: "source | target || source | target || 实际翻译"）
    if ' || ' in text:
        # 取最后一个 || 之后的内容
        text = text.rsplit(' || ', 1)[-1]
    if text.startswith('Examples:\n') or text.startswith('Examples:'):
        lines = text.strip().split('\n')
        text = lines[-1] if lines else text
    # 3. 剥离 task prefix（如 "translate Akkadian to English: "）
    prefix_marker = 'to English:'
    if prefix_marker in text:
        text = text[text.index(prefix_marker) + len(prefix_marker):]
    return text.strip()


def _is_damaged_infer(text: str) -> bool:
    """判断阿卡德语转写是否含有破损标记（推理端，与 train.py _is_damaged 逻辑一致）。"""
    if '<gap>' in text:
        return True
    if '[' in text or ']' in text:
        return True
    if re.search(r'\bx\b', text, re.IGNORECASE):
        return True
    if '⸢' in text or '⸣' in text:
        return True
    if '...' in text:
        return True
    return False


def _build_meta_prefix_infer(text: str, dialect: str = "OA") -> str:
    """构建推理端 meta prefix: [dialect][source][completeness]。
    推理时来源统一为 O (official)，完整度自动判断。
    dialect: OA/OB/AK/SX (默认 OA，测试集全是 OA)
    """
    _DIAL_MAP = {"OA": "OA", "OB": "OB", "OAkk": "XX", "Sumerian": "XX"}
    dial = _DIAL_MAP.get(str(dialect).strip(), "OA")
    comp = "D" if _is_damaged_infer(text) else "I"
    return f"{dial}O{comp} "


def _get_infer_prefix(text: str, config: InferenceConfig) -> str:
    """根据配置返回推理端前缀。"""
    if config.use_meta_prefix:
        return _build_meta_prefix_infer(text)
    return config.task_prefix


def preprocess_input(text: str, config: InferenceConfig, lexicon: dict = None) -> str:
    """
    完整的输入预处理
    复用 prepare_data.py 的 preprocess_transliteration，并添加 HINTS 和前缀
    """
    if not isinstance(text, str) or pd.isna(text):
        return _get_infer_prefix("", config)
    
    # 1. 基础清洗 (使用 prepare_data 中的核心逻辑)
    text = preprocess_transliteration(text)
    
    # 2. Silver 数据专用清洗（处理 train.csv 中不存在的脏模式）
    if config.pseudo_label_mode:
        text = clean_silver_specific_patterns(text)
    
    # 3. hint 模式：添加词典提示（推理时始终应用，不做随机跳过）
    if config.name_handling_mode == 'hint' and lexicon:
        _enable_pn = getattr(config, 'hint_enable_place_names', False)
        text = _apply_hints(text, lexicon, config.max_hints_per_sample, enable_place_names=_enable_pn)
    
    # 3.5 月份 hint（独立子开关）
    if config.use_month_hints:
        text = _apply_month_hints(text)
    
    # 4. 添加任务前缀（meta_prefix 模式下自动判断完整度）
    return _get_infer_prefix(text, config) + text


# ============================================================================
# 模型加载
# ============================================================================

def load_single_model(model_path: str, device: torch.device, config: InferenceConfig):
    """加载单个模型，多卡时复制到第二张卡实现数据并行"""
    print(f"加载模型：{model_path} (精度: {config.dtype})")
    
    torch_dtype = torch.float32
    if config.dtype == "fp16":
        torch_dtype = torch.float16
    elif config.dtype == "bf16":
        torch_dtype = torch.bfloat16
    
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch_dtype
    ).to(device).eval()
    
    # 应用 BetterTransformer 加速
    if config.use_better_transformer and torch.cuda.is_available():
        try:
            from optimum.bettertransformer import BetterTransformer
            model = BetterTransformer.transform(model)
            print(f"   ✓ BetterTransformer 已启用")
        except ImportError:
            print(f"   ⚠ optimum 未安装，跳过 BetterTransformer")
        except Exception as e:
            print(f"   ⚠ BetterTransformer 失败: {e}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # 加载 generation_config（如果存在）
    gen_config_path = os.path.join(model_path, "generation_config.json")
    if config.use_generation_config and os.path.exists(gen_config_path):
        from transformers import GenerationConfig
        gen_config = GenerationConfig.from_pretrained(model_path)
        model.generation_config = gen_config
        model.generation_config.max_length = None
        print(f"   ✓ 加载 generation_config: num_beams={gen_config.num_beams}, length_penalty={gen_config.length_penalty}")
        for attr in ("num_beams", "length_penalty", "repetition_penalty", "early_stopping"):
            if getattr(config, attr, None) is None:
                gen_val = getattr(gen_config, attr, None)
                if gen_val is not None:
                    setattr(config, attr, gen_val)
                    print(f"   ✓ 同步 generation_config.{attr} -> {gen_val}")
    else:
        if getattr(model, "generation_config", None) is not None:
            model.generation_config.max_length = None
        print(f"   ⚠ 使用默认生成参数")
    
    return model, tokenizer


def _resolve_generation_value(model, config_value, attr_name: str, default=None):
    if config_value is not None:
        return config_value
    gen_cfg = getattr(model, "generation_config", None)
    if gen_cfg is not None:
        gen_val = getattr(gen_cfg, attr_name, None)
        if gen_val is not None:
            return gen_val
    return default


def load_model_ensemble(
    model_paths: List[str],
    weights: List[float],
    device: torch.device
):
    """
    加载模型融合（model soup）
    
    参考：https://www.kaggle.com/code/qifeihhh666/dpc-weight-averaging-clean-repetition
    """
    print(f"\n创建模型融合（{len(model_paths)}个模型）")
    import gc
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 归一化权重
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    
    print("模型权重：")
    for path, weight in zip(model_paths, normalized_weights):
        print(f"  {Path(path).name}: {weight:.3f}")
    
    # 1. 加载基础模型（CPU 上安全运算）
    print(f"   加载 Base 模型 (CPU): {Path(model_paths[0]).name}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[0], local_files_only=True)
    state_dict = base_model.state_dict()
    
    # 预乘第一个模型的权重
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].float() * normalized_weights[0]
    
    # 2. 流式累加后续模型（O(1) 内存：加载→累加→销毁）
    for i, model_path in enumerate(model_paths[1:]):
        print(f"   累加模型 {i+2}: {Path(model_path).name}")
        temp_model = AutoModelForSeq2SeqLM.from_pretrained(model_path, local_files_only=True)
        temp_sd = temp_model.state_dict()
        
        skipped = 0
        for key in state_dict.keys():
            if key in temp_sd and temp_sd[key].shape == state_dict[key].shape:
                state_dict[key] += temp_sd[key].float() * normalized_weights[i + 1]
            elif key in temp_sd:
                skipped += 1
        
        if skipped > 0:
            print(f"      ⚠️ 跳过 {skipped} 个形状不匹配的参数")
        
        # 阅后即焚：销毁临时模型释放内存
        del temp_model, temp_sd
        gc.collect()
    
    # 3. 加载融合权重并推送到 GPU
    base_model.load_state_dict(state_dict)
    del state_dict
    gc.collect()
    
    print("   ✅ 融合计算完毕，推送至 GPU...")
    model = base_model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_paths[0], local_files_only=True)
    
    print("✓ 模型融合完成")
    
    return model, tokenizer


# ============================================================================
# MBR 解码 (Minimum Bayes Risk)
# ============================================================================

def mbr_select(
    candidates: List[str],
    metric: str = "weighted",
    pool_cap: Optional[int] = None,
    w_chrf: float = 0.55,
    w_bleu: float = 0.25,
    w_jaccard: float = 0.20,
    w_length: float = 0.10,
):
    """
    MBR 解码：从候选列表中选择与其他所有候选平均相似度最高的翻译
    
    原理：如果一个候选与大多数其他候选相似，说明它大概率是正确的翻译
    
    Args:
        candidates: 所有模型生成的候选翻译列表
        metric: 评估指标 ("weighted" / "chrf" / "bleu")
    
    Returns:
        (best_text, best_score): 最佳候选翻译 + MBR 共识分数（weighted 模式可能略高于 100）
    """
    if not candidates:
        return "", 0.0
    if len(candidates) == 1:
        return candidates[0], 100.0
    
    # 去重（保留顺序）
    seen = set()
    unique = []
    for c in candidates:
        c_stripped = c.strip()
        if c_stripped and c_stripped not in seen:
            seen.add(c_stripped)
            unique.append(c_stripped)

    if pool_cap:
        unique = unique[:pool_cap]
    
    if len(unique) <= 1:
        return (unique[0] if unique else candidates[0]), 100.0
    
    # 使用 sacrebleu 内置评分（无需 evaluate 库，Kaggle 兼容）
    import sacrebleu

    metric = (metric or "weighted").lower()
    use_weighted = metric in ("weighted", "ensemble", "hybrid")

    if use_weighted:
        chrf_metric = sacrebleu.metrics.CHRF(word_order=2)
        bleu_metric = sacrebleu.metrics.BLEU(effective_order=True)
        pairwise_weight_sum = max(w_chrf + w_bleu + w_jaccard, 1e-9)
        lengths = [len(text.split()) for text in unique]
        median_len = float(statistics.median(lengths)) if lengths else 0.0
        sigma = max(median_len * 0.4, 5.0)

        def _jaccard(a: str, b: str) -> float:
            ta = set(a.lower().split())
            tb = set(b.lower().split())
            if not ta and not tb:
                return 100.0
            if not ta or not tb:
                return 0.0
            return 100.0 * len(ta & tb) / len(ta | tb)

        def _safe_bleu(a: str, b: str) -> float:
            try:
                return float(bleu_metric.sentence_score(a, [b]).score)
            except Exception:
                return 0.0

        scores = []
        for i, hyp in enumerate(unique):
            total = 0.0
            count = 0
            for j, ref in enumerate(unique):
                if i == j:
                    continue
                pairwise = (
                    w_chrf * float(chrf_metric.sentence_score(hyp, [ref]).score)
                    + w_bleu * _safe_bleu(hyp, ref)
                    + w_jaccard * _jaccard(hyp, ref)
                ) / pairwise_weight_sum
                total += pairwise
                count += 1

            avg_pairwise = total / max(count, 1)
            z = (lengths[i] - median_len) / sigma if sigma > 0 else 0.0
            length_bonus = 100.0 * math.exp(-0.5 * z * z)
            scores.append(avg_pairwise + w_length * length_bonus)
    else:
        # 计算每个候选与所有其他候选的平均分
        use_geom = metric in ("geom", "geometric", "official")
        if use_geom:
            chrf_scorer = sacrebleu.metrics.CHRF(word_order=2)
            bleu_scorer = sacrebleu.metrics.BLEU(effective_order=True)
        scores = []
        for i, hyp in enumerate(unique):
            total = 0.0
            count = 0
            for j, ref in enumerate(unique):
                if i == j:
                    continue
                if use_geom:
                    c = float(chrf_scorer.sentence_score(hyp, [ref]).score)
                    try:
                        b = float(bleu_scorer.sentence_score(hyp, [ref]).score)
                    except Exception:
                        b = 0.0
                    total += math.sqrt(max(c, 0.0) * max(b, 0.0))
                elif metric == "chrf":
                    result = sacrebleu.sentence_chrf(hyp, [ref], char_order=6, word_order=2)
                    total += result.score
                else:
                    result = sacrebleu.sentence_bleu(hyp, [ref])
                    total += result.score
                count += 1
            avg = total / max(count, 1)
            scores.append(avg)

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return unique[best_idx], scores[best_idx]


def finalize_hybrid_candidate_batch(
    out_beam_local,
    out_sample_local,
    *,
    tokenizer,
    beam_cands: int,
    sample_cands: int,
    postprocess_batch: Optional[Callable[[List[str]], List[str]]] = None,
    restore_mappings: Optional[List[Any]] = None,
    restore_fn: Optional[Callable[[str, Any], str]] = None,
    mbr_selector: Optional[Callable[[List[str]], Tuple[str, float]]] = None,
):
    """共享的 hybrid-MBR finalize helper，供 infer/train 复用。"""
    timings = {
        "decode": 0.0,
        "postprocess": 0.0,
        "restore": 0.0,
        "mbr": 0.0,
        "total": 0.0,
    }
    total_t0 = time.perf_counter()

    t_decode = time.perf_counter()
    beam_raw = decode_sequences(out_beam_local, tokenizer=tokenizer)
    sample_raw = decode_sequences(out_sample_local, tokenizer=tokenizer)
    timings["decode"] += time.perf_counter() - t_decode

    if postprocess_batch is None:
        beam_decoded = beam_raw
        sample_decoded = sample_raw
    else:
        t_post = time.perf_counter()
        beam_decoded = postprocess_batch(beam_raw)
        sample_decoded = postprocess_batch(sample_raw)
        timings["postprocess"] += time.perf_counter() - t_post

    if restore_mappings is not None:
        item_count = len(restore_mappings)
    else:
        item_count = min(
            len(beam_decoded) // max(int(beam_cands), 1),
            len(sample_decoded) // max(int(sample_cands), 1),
        )

    grouped_candidates = []
    restore_mappings = restore_mappings or [None] * item_count
    for i in range(item_count):
        beam_start = i * beam_cands
        beam_end = beam_start + beam_cands
        sample_start = i * sample_cands
        sample_end = sample_start + sample_cands
        cand_set = beam_decoded[beam_start:beam_end] + sample_decoded[sample_start:sample_end]
        mapping = restore_mappings[i] if i < len(restore_mappings) else None
        if restore_fn and mapping:
            t_restore = time.perf_counter()
            cand_set = [restore_fn(c, mapping) for c in cand_set]
            timings["restore"] += time.perf_counter() - t_restore
        grouped_candidates.append(cand_set)

    if mbr_selector is None:
        def mbr_selector(candidates: List[str]) -> Tuple[str, float]:
            return mbr_select(candidates, metric="chrf")

    t_mbr = time.perf_counter()
    decoded, scores = [], []
    for cand_set in grouped_candidates:
        best_text, best_score = mbr_selector(cand_set)
        decoded.append(best_text)
        scores.append(best_score)
    timings["mbr"] += time.perf_counter() - t_mbr
    timings["total"] = time.perf_counter() - total_t0

    return decoded, scores, grouped_candidates, timings


def _maybe_log_generation_progress(
    label: str,
    batch_idx: int,
    processed: int,
    total_segments: int,
    elapsed: float,
    gpu_mem: float,
    every: int,
) -> None:
    if every <= 0:
        return
    if batch_idx == 0 or (batch_idx + 1) % every == 0 or processed >= total_segments:
        pct = 100.0 * processed / max(1, total_segments)
        speed = processed / elapsed if elapsed > 0 else 0.0
        eta = (total_segments - processed) / speed if speed > 0 else 0.0
        print(
            f"   [{label}] {processed}/{total_segments} "
            f"({pct:.1f}%) | speed={speed:.1f} seg/s | ETA={eta:.0f}s | GPU={gpu_mem:.1f}G"
        )


def _format_stage_timing_summary(stats: Dict[str, float]) -> str:
    ordered_keys = [
        "model_load_main",
        "model_load_replica",
        "h2d",
        "beam_generate",
        "beam_cpu_copy",
        "sample_generate",
        "sample_cpu_copy",
        "decode",
        "postprocess",
        "placeholder_restore",
        "mbr_finalize_wait",
        "mbr_collect_wait",
        "mbr_collect_sync",
    ]
    parts = []
    for key in ordered_keys:
        value = float(stats.get(key, 0.0))
        if value > 1e-4:
            parts.append(f"{key}={value:.2f}s")
    return " | ".join(parts) if parts else "no timing data"


def _should_log_stage_timing(
    batch_idx: int,
    processed: int,
    total_segments: int,
    every: int,
) -> bool:
    if every <= 0:
        return False
    return batch_idx == 0 or (batch_idx + 1) % every == 0 or processed >= total_segments


_MODEL_PACKAGE_STOP_BASENAMES = {"", "input", "output", "working"}


def _iter_model_package_dirs(model_path: str, *, max_depth: int = 6) -> List[str]:
    """
    Search only within the same model package.

    Cloud layouts often store weights in a nested directory such as
    `.../checkpoint-9400` or `.../pytorch/default/1`, while auxiliary assets
    like `correction_vocab.json` live one or more levels above inside the same
    mounted model package.
    """
    if not model_path:
        return []

    current = os.path.abspath(model_path)
    candidates: List[str] = []
    seen = set()

    for _ in range(max_depth):
        if not current or current in seen:
            break

        base = os.path.basename(current)
        parent = os.path.dirname(current)
        parent_base = os.path.basename(parent)

        # Stop before climbing above the package boundary.
        if base in _MODEL_PACKAGE_STOP_BASENAMES or parent_base == "datasets":
            break

        candidates.append(current)
        seen.add(current)

        if parent == current or parent_base in _MODEL_PACKAGE_STOP_BASENAMES:
            break

        current = parent

    return candidates


def _find_first_model_asset(
    model_paths: List[str],
    filename: str,
    *,
    feature_name: str,
) -> Optional[str]:
    """Search within each model package (model dir + safe ancestor dirs)."""
    print(f"   🔎 {feature_name}: model-package mode, searching for {filename}")
    if not model_paths:
        print(f"   ⚠️ {feature_name}: no model paths configured, disabled")
        return None

    for idx, model_path in enumerate(model_paths):
        for asset_dir in _iter_model_package_dirs(model_path):
            candidate = os.path.join(asset_dir, filename)
            if os.path.exists(candidate):
                try:
                    size_bytes = os.path.getsize(candidate)
                    size_info = f", {size_bytes} bytes"
                except OSError:
                    size_info = ""
                print(
                    f"   ✅ {feature_name}: found in model[{idx}] "
                    f"{candidate}{size_info}"
                )
                return candidate
            print(f"   · {feature_name}: missing in model[{idx}] {candidate}")

    print(f"   ⚠️ {feature_name}: {filename} not found in any model package, disabled")
    return None


def _load_sumerogram_glossary_json(path: str) -> Dict[str, Tuple[str, str]]:
    """Load serialized glossary JSON as {token: (reading, gloss)}."""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"expected object at top level, got {type(raw).__name__}")

    glossary: Dict[str, Tuple[str, str]] = {}
    for token, value in raw.items():
        if isinstance(value, dict):
            reading = (
                value.get("akkadian")
                or value.get("reading")
                or value.get("akkadian_reading")
                or ""
            )
            gloss = (
                value.get("english")
                or value.get("gloss")
                or value.get("meaning")
                or value.get("english_gloss")
                or ""
            )
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            reading, gloss = value[0], value[1]
        else:
            raise ValueError(
                f"token {token!r} has unsupported value type {type(value).__name__}"
            )
        glossary[str(token)] = (str(reading), str(gloss))

    return glossary


# ============================================================================
# 数据集
# ============================================================================

class AkkadianDataset(Dataset):
    """阿卡德语数据集"""
    
    def __init__(self, df: pd.DataFrame, config: InferenceConfig, lexicon: dict = None):
        if 'id' in df.columns:
            self.ids = df['id'].tolist()
        elif 'oare_id' in df.columns:
             self.ids = df['oare_id'].tolist()
        else:
             self.ids = [f"unknown_{i}" for i in range(len(df))]

        self.texts = [
            preprocess_input(text, config, lexicon)
            for text in df['transliteration']
        ]
    
    def __len__(self):
        return len(self.ids)
    
    def __getitem__(self, idx):
        return self.ids[idx], self.texts[idx]


# ============================================================================
# 推理
# ============================================================================

def run_inference(config: InferenceConfig):
    """运行推理"""
    
    import sys
    import os
    import threading
    
    start_time = time.time()
    stage_timing = defaultdict(float)
    stage_timing_lock = threading.Lock()

    def _record_stage_timing(name: str, delta: float) -> None:
        if delta <= 0:
            return
        with stage_timing_lock:
            stage_timing[name] += float(delta)

    def _snapshot_stage_timing() -> Dict[str, float]:
        with stage_timing_lock:
            return dict(stage_timing)

    from contextlib import contextmanager
    @contextmanager
    def _timed(name: str):
        """Context manager for stage timing: with _timed('decode'): ..."""
        t0 = time.perf_counter()
        yield
        _record_stage_timing(name, time.perf_counter() - t0)
    
    print("=" * 80)
    print("阿卡德语-英语翻译推理（优化版）")
    print("=" * 80)
    print(f"设备: {config.device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 加载测试数据 (判断是否为伪标签模式)
    if config.mrt_data_mode:
        print(f"\n{'='*80}")
        print(f"📂 [模式] MRT 候选数据生成")
        print(f"{'='*80}")
        print(f"   输入文件: {config.input_mrt_csv}")
        df = pd.read_csv(config.input_mrt_csv)
        print(f"   输出文件 (Raw): {config.output_mrt_raw}")
        print(f"   输出文件 (Filtered): {config.output_mrt_filtered}")
        print(f"   最小唯一候选数: {config.mrt_min_unique_candidates}")
    elif getattr(config, 'score_mode', False) and config.score_input_csv:
        print(f"\n{'='*80}")
        print(f"📂 [模式] Ground-Truth 评分 (Score Mode)")
        print(f"{'='*80}")
        print(f"   输入文件: {config.score_input_csv}")
        df = pd.read_csv(config.score_input_csv)
        # 验证必需列
        assert 'transliteration' in df.columns, "score_input_csv 缺少 transliteration 列"
        assert 'translation' in df.columns, "score_input_csv 缺少 translation 列 (ground truth)"
        _score_ground_truth = df['translation'].fillna("").astype(str).tolist()
        if not config.score_output_csv:
            config.score_output_csv = config.score_input_csv.replace('.csv', '_scored.csv')
        print(f"   输出文件: {config.score_output_csv}")
        print(f"   Ground truth 列: translation ({sum(1 for t in _score_ground_truth if t)} 条非空)")
    elif config.pseudo_label_mode:
        print(f"\n{'='*80}")
        print(f"📂 [模式] 伪标签生成 (Pseudo-Labeling)")
        print(f"{'='*80}")
        print(f"   输入文件: {config.input_silver_csv}")
        df = pd.read_csv(config.input_silver_csv)
        print(f"   输出文件 (Raw): {config.output_silver_raw}")
        print(f"   输出文件 (Filtered): {config.output_silver_filtered}")
        if config.decode_strategy == "beam":
            print(f"   过滤阈值: log_prob>={config.filter_min_confidence}, len_ratio=[{config.filter_min_len_ratio}, {config.filter_max_len_ratio}]")
        else:
            print(f"   过滤阈值: mbr_score>={config.filter_min_mbr_score}, len_ratio=[{config.filter_min_len_ratio}, {config.filter_max_len_ratio}]")
    else:
        print(f"\n📂 [模式] 标准测试集推理")
        print(f"   输入文件: {config.test_csv}")
        df = pd.read_csv(config.test_csv)
    
    print(f"   样本数：{len(df)} 条")
    sys.stdout.flush()  # 实时刷新
    
    # 加载词典并构建 Onomasticon
    lexicon = load_lexicon(config.lexicon_csv)
    onomasticon = build_onomasticon(lexicon)
    print(f"   📚 Onomasticon: {len(onomasticon)} 个专有名词")
    
    # 加载校准词典（优先从当前模型包读取，fallback 到 data_dir）— 仍用于 placeholder/onomasticon
    calibrated_lexicon = {}
    calibrated_lexicon_source = "none"
    _cal_model_path = _find_first_model_asset(
        config.model_paths,
        "calibrated_lexicon.json",
        feature_name="Calibrated lexicon",
    )
    _cal_candidates2 = [
        _cal_model_path,
        os.path.join(config.data_dir, "calibrated_lexicon.json"),
    ]
    for cal_path in _cal_candidates2:
        if cal_path and os.path.exists(cal_path):
            import json as _json_loader2
            with open(cal_path, 'r', encoding='utf-8') as f:
                calibrated_lexicon = _json_loader2.load(f)
            print(f"   📚 Calibrated lexicon: {len(calibrated_lexicon)} entries (from {cal_path})")
            calibrated_lexicon_source = cal_path
            break

    # 加载高精度 hint 词典（train/eval/infer 统一使用，无 raw fallback）
    hint_lexicon = {}
    hint_lexicon_source = "none"
    _HP_HINT_FILENAME = "high_precision_hint_lexicon.json"
    _hp_model_path = _find_first_model_asset(
        config.model_paths,
        _HP_HINT_FILENAME,
        feature_name="High-precision hint lexicon",
    )
    _hp_candidates = [
        _hp_model_path,
        os.path.join(config.data_dir, _HP_HINT_FILENAME),
    ]
    for hp_path in _hp_candidates:
        if hp_path and os.path.exists(hp_path):
            import json as _json_hp
            with open(hp_path, 'r', encoding='utf-8') as f:
                hint_lexicon = _json_hp.load(f)
            print(f"   📚 High-precision hint lexicon: {len(hint_lexicon)} entries (from {hp_path})")
            hint_lexicon_source = hp_path
            break
    if config.name_handling_mode == 'hint' and not hint_lexicon:
        print(f"   ⚠️ Hint mode ON but no high-precision hint lexicon found, raw_fallback=OFF → hints disabled")

    _feature_status = {
        "calibrated_lexicon_loaded": bool(calibrated_lexicon),
        "calibrated_lexicon_source": calibrated_lexicon_source,
        "hint_lexicon_loaded": bool(hint_lexicon),
        "hint_lexicon_source": hint_lexicon_source,
        "correction_vocab_enabled": False,
        "correction_vocab_source": "none",
        "sumerogram_glossary_enabled": False,
        "sumerogram_glossary_source": "none",
    }
    
    # 加载编码损坏修复词表（仅从当前模型包读取；找不到即关闭）
    _correction_vocab = None
    if config.use_correction_vocab:
        _cv_path = _find_first_model_asset(
            config.model_paths,
            config.correction_vocab_filename,
            feature_name="Correction vocab",
        )
        if _cv_path:
            try:
                import json as _json_cv
                with open(_cv_path, 'r', encoding='utf-8') as f:
                    _correction_vocab = _json_cv.load(f)
                print(f"   📚 Correction vocab: enabled with {len(_correction_vocab)} tokens")
                _feature_status["correction_vocab_enabled"] = True
                _feature_status["correction_vocab_source"] = _cv_path
            except Exception as _e:
                _correction_vocab = None
                print(f"   ⚠️ Correction vocab: failed to load {_cv_path}, disabled ({_e})")
        else:
            print(f"   ⚠️ Correction vocab: not found in model package, disabled")
    else:
        print(f"   ⏭️ Correction vocab: disabled by config")

    # 加载 Sumerogram 词义注入词表（仅从当前模型包读取；找不到即关闭）
    _sumerogram_glossary = None
    if config.use_sumerogram_glossary:
        _sg_path = _find_first_model_asset(
            config.model_paths,
            config.sumerogram_glossary_filename,
            feature_name="Sumerogram glossary",
        )
        if _sg_path:
            try:
                _sumerogram_glossary = _load_sumerogram_glossary_json(_sg_path)
                print(f"   📚 Sumerogram glossary: enabled with {len(_sumerogram_glossary)} entries")
                _feature_status["sumerogram_glossary_enabled"] = True
                _feature_status["sumerogram_glossary_source"] = _sg_path
            except Exception as _e:
                _sumerogram_glossary = None
                print(f"   ⚠️ Sumerogram glossary: failed to load {_sg_path}, disabled ({_e})")
        else:
            print(f"   ⚠️ Sumerogram glossary: not found in model dirs, disabled")

    # 初始化专名处理
    placeholder_svc = None
    if config.name_handling_mode == 'placeholder':
        from prepare_data import get_placeholder_service
        placeholder_svc = get_placeholder_service(onomasticon, lexicon)
        print(f"   📌 Placeholder 模式")
    elif config.name_handling_mode == 'hint':
        if hint_lexicon:
            print(f"   📌 Hint 模式 (max_hints={config.max_hints_per_sample}, hp_lexicon={len(hint_lexicon)}, raw_fallback=OFF)")
        else:
            print(f"   📌 Hint 模式: DISABLED (no high-precision lexicon, raw_fallback=OFF)")
    else:
        print(f"   📌 专名处理: 关闭")

    print(f"\n🧩 预处理功能总览")
    training_cfg_label = config._training_config_path or "none"
    print(f"   training_config: {training_cfg_label}")
    print(
        f"   编码修复开关: {'已启用' if config.use_correction_vocab else '未启用'}"
        f" | 来源: {config._config_sources.get('use_correction_vocab', 'code_default')}"
    )
    print(
        f"   编码修复词表: "
        f"{'已加载' if _feature_status['correction_vocab_enabled'] else '未加载'}"
        f" | 来源: {_feature_status['correction_vocab_source']}"
    )
    print(
        f"   校准词典: "
        f"{'已启用' if _feature_status['calibrated_lexicon_loaded'] else '未启用'}"
        f" | 来源: {_feature_status['calibrated_lexicon_source']}"
    )
    if config.name_handling_mode == 'none':
        name_mode_label = "关闭"
    elif config.name_handling_mode == 'hint':
        name_mode_label = (
            f"Hint 模式 | hp_lexicon={len(hint_lexicon)} | max_hints={config.max_hints_per_sample} | "
            f"raw_fallback=OFF | place_names={'ON' if getattr(config, 'hint_enable_place_names', False) else 'OFF'}"
        )
    elif config.name_handling_mode == 'placeholder':
        name_mode_label = "Placeholder 模式"
    else:
        name_mode_label = str(config.name_handling_mode)
    print(
        f"   专名处理开关: {name_mode_label}"
        f" | 来源: {config._config_sources.get('name_handling_mode', 'code_default')}"
    )
    print(
        f"   月份提示开关: {'已启用' if config.use_month_hints else '未启用'}"
        f" | 来源: {config._config_sources.get('use_month_hints', 'code_default')}"
    )
    print(
        f"   Sumerogram 开关: {'已启用' if config.use_sumerogram_glossary else '未启用'}"
        f" | 来源: {config._config_sources.get('use_sumerogram_glossary', 'code_default')}"
    )
    print(
        f"   Sumerogram 词义词表: "
        f"{'已加载' if _feature_status['sumerogram_glossary_enabled'] else '未加载'}"
        f" | 来源: {_feature_status['sumerogram_glossary_source']}"
    )
    print(
        f"   Sumerogram 提示上限: {config.max_sumerogram_hints}"
        f" | 来源: {config._config_sources.get('max_sumerogram_hints', 'code_default')}"
    )
    print(
        f"   Silver 专用清洗: "
        f"{'已启用' if config.pseudo_label_mode else '未启用（仅伪标签模式）'}"
    )
    print(
        f"   RAG 开关: {'已启用' if config.use_rag else '未启用'}"
        f" | 来源: {config._config_sources.get('use_rag', 'code_default')}"
    )
    
    # ====================================================================
    # 模型加载策略
    # - 非 multi_model_mbr：保持原逻辑（可选 Model Soup，hybrid_mbr 也走这里）
    # - multi_model_mbr：跳过融合，后续进入逐模型顺序加载采样
    # ====================================================================
    model = None
    tokenizer = None
    model_replicas = []
    n_gpus = torch.cuda.device_count()

    if config.decode_strategy == "multi_model_mbr":
        print(f"\n🧠 multi_model_mbr 模式：跳过 Model Soup，采用逐模型顺序采样。")
        tokenizer = None
    else:
        with _timed("model_load_main"):
            if len(config.model_paths) > 1:
                model, tokenizer = load_model_ensemble(
                    config.model_paths,
                    config.model_weights,
                    config.device
                )
            else:
                model, tokenizer = load_single_model(config.model_paths[0], config.device, config)

        # ====================================================================
        # 融合权重导出（仅 Kaggle 云端，默认关闭）
        # ====================================================================
        if config.export_merged_model and len(config.model_paths) > 1:
            _export_dir = os.path.join(os.path.dirname(config.output_csv), "merged_model")
            os.makedirs(_export_dir, exist_ok=True)
            print(f"\n💾 Exporting merged model to {_export_dir}...")
            model.save_pretrained(_export_dir)
            tokenizer.save_pretrained(_export_dir)
            print(f"   ✅ Merged model exported ({len(config.model_paths)} models → {_export_dir})")

        # ====================================================================
        # 多卡数据并行 (Data Parallelism) — 失败时安全降级到单卡
        # ====================================================================
        model_replicas = [model]
        if n_gpus > 1:
            import gc
            dev2 = torch.device("cuda:1")
            try:
                print(f"\n   🚀 检测到 {n_gpus} 张 GPU，从磁盘重新加载模型到 cuda:1...")
                # 最稳定方案：从磁盘重新走完整加载流程（与 cuda:0 完全一致）
                with _timed("model_load_replica"):
                    if len(config.model_paths) > 1:
                        model2, _ = load_model_ensemble(config.model_paths, config.model_weights, dev2)
                    else:
                        model2, _ = load_single_model(config.model_paths[0], dev2, config)
                gc.collect()
                model_replicas.append(model2)
                print(f"   ✓ 双卡数据并行就绪（当前模式: {config.decode_strategy}）")
            except Exception as e:
                print(f"   ⚠️ 双卡加载失败 ({e})，降级到单卡推理")
                if 'model2' in dir():
                    del model2
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    # ====================================================================
    # RAG 索引构建（如果训练时启用了 RAG，推理时也必须注入）
    # ====================================================================
    rag_index = None
    if config.use_rag:
        print(f"\n🔍 构建 RAG 检索索引...")
        # 查找训练数据（优先从模型目录，fallback 到 data_dir）
        rag_csv = config.rag_train_csv
        if not rag_csv:
            _rag_candidates = [
                os.path.join(config.model_paths[0], "train.csv") if config.model_paths else "",
                os.path.join(config.data_dir, "train.csv"),
                os.path.join(config.data_dir, "train_clean.csv"),
            ]
            for _rc in _rag_candidates:
                if _rc and os.path.exists(_rc):
                    rag_csv = _rc
                    break
        
        if rag_csv and os.path.exists(rag_csv):
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from rank_bm25 import BM25Okapi
                import numpy as np
                
                rag_df = pd.read_csv(rag_csv)
                # 确定列名（兼容 input_text/transliteration 和 target_text/translation）
                src_col = 'input_text' if 'input_text' in rag_df.columns else 'transliteration'
                tgt_col = 'target_text' if 'target_text' in rag_df.columns else 'translation'
                
                # 过滤干净示例
                clean_mask = (
                    (rag_df[tgt_col].str.len() >= config.rag_min_translation_len) &
                    (~rag_df[tgt_col].str.contains('<gap>', case=False, na=False)) &
                    (rag_df[tgt_col].str.strip().str.len() > 0) &
                    (rag_df[src_col].str.strip().str.len() > 5)
                )
                clean_df = rag_df[clean_mask].reset_index(drop=True)
                
                # 提取纯文本
                rag_transliterations = []
                rag_translations = []
                for _, row in clean_df.iterrows():
                    inp = str(row[src_col])
                    # Strip any known prefix (meta or dialect-based)
                    if config.use_meta_prefix and re.match(r'^(?:OA|OB|XX)[ON][ID] ', inp):
                        inp = inp[5:]
                    elif inp.startswith(config.task_prefix):
                        inp = inp[len(config.task_prefix):]
                    tgt = str(row[tgt_col]).strip()
                    words = tgt.split()
                    if len(words) > 3 and len(set(words)) / len(words) < 0.3:
                        continue
                    rag_transliterations.append(inp.strip())
                    rag_translations.append(tgt)
                
                # 构建 BM25 + TF-IDF 索引
                tokenized_corpus = [doc.split() for doc in rag_transliterations]
                rag_bm25 = BM25Okapi(tokenized_corpus)
                rag_tfidf = TfidfVectorizer(max_features=8000, ngram_range=(3, 6), analyzer='char_wb', sublinear_tf=True)
                rag_tfidf_matrix = rag_tfidf.fit_transform(rag_transliterations)
                
                rag_index = {
                    'bm25': rag_bm25,
                    'tfidf': rag_tfidf,
                    'tfidf_matrix': rag_tfidf_matrix,
                    'transliterations': rag_transliterations,
                    'translations': rag_translations,
                }
                print(f"   ✅ RAG 索引就绪: {len(rag_transliterations)}/{len(rag_df)} 条干净示例 (from {rag_csv})")
            except ImportError as e:
                print(f"   ⚠️ RAG 依赖缺失 ({e})，跳过 RAG 注入")
                config.use_rag = False
            except Exception as e:
                print(f"   ⚠️ RAG 索引构建失败 ({e})，跳过 RAG 注入")
                config.use_rag = False
        else:
            print(f"   ⚠️ 未找到训练数据用于 RAG 索引，跳过 RAG 注入")
            config.use_rag = False
    
    def _rag_retrieve(query_source, top_k=2):
        """实时检索 RAG 示例"""
        if not rag_index or not query_source or not query_source.strip():
            return []
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        candidate_pool = top_k * 10
        # BM25
        bm25_scores = rag_index['bm25'].get_scores(query_source.split())
        bm25_ranking = np.argsort(bm25_scores)[::-1][:candidate_pool].tolist()
        # TF-IDF
        query_vec = rag_index['tfidf'].transform([query_source])
        tfidf_scores = cosine_similarity(query_vec, rag_index['tfidf_matrix']).flatten()
        tfidf_ranking = np.argsort(tfidf_scores)[::-1][:candidate_pool].tolist()
        # RRF 融合
        fused = {}
        for ranking in [bm25_ranking, tfidf_ranking]:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (60 + rank + 1)
        sorted_cands = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        budget = int(config.max_input_length * config.rag_budget_ratio)
        used_bytes = 0
        for idx, _ in sorted_cands:
            if len(results) >= top_k:
                break
            translit = rag_index['transliterations'][idx]
            trans = rag_index['translations'][idx]
            if translit.strip() == query_source.strip():
                continue
            if not (bm25_scores[idx] >= 0.5 or tfidf_scores[idx] >= 0.15):
                continue
            example_str = f"{translit} | {trans}"
            example_bytes = len(example_str.encode('utf-8'))
            if example_bytes > config.rag_max_example_bytes:
                continue  # 宁缺毋滥：超长示例直接跳过，不截断加 ...（防止模型学到省略输出）
            if used_bytes + example_bytes > budget:
                break
            results.append({'source': translit, 'target': trans})
            used_bytes += example_bytes
        return results
    
    # ====================================================================
    # 预处理
    # ====================================================================
    print(f"\n📐 预处理...")
    sys.stdout.flush()
    
    flat_segments = []  # (original_id, processed_text, placeholder_mapping, original_text)
    id_order = []       # 保持原始 ID 顺序
    cleaned_transliterations = {}  # 存储清洗后的原文（用于输出）
    preprocess_debug_by_id = {}    # 前几个样本的预处理轨迹
    
    # 统计计数器
    stats = {
        'total': len(df),
        'empty': 0,
        'silver_cleaned': 0,  # silver 专用清洗计数
        'total_segments': 0,
        'rag_injected': 0,    # RAG 实际成功注入的片段数
    }
    
    # 统一 ID 列名（用于最终输出）— 自动从实际列名检测
    if (config.pseudo_label_mode or config.mrt_data_mode or getattr(config, 'score_mode', False)) and 'oare_id' in df.columns:
        id_col = 'oare_id'
    elif 'id' in df.columns:
        id_col = 'id'
    elif 'oare_id' in df.columns:
        id_col = 'oare_id'
    else:
        id_col = df.columns[0]
        print(f"   ⚠️ 未找到 id/oare_id 列，使用 '{id_col}' 作为 ID 列")
    
    # 为每行分配唯一行级 ID（防止同一 oare_id 多行被合并）
    df = df.reset_index(drop=True)
    df['_row_id'] = range(len(df))
    
    preprocess_start = time.time()
    for idx, row in df.iterrows():
        sample_id = row['_row_id']  # 用行级 ID 确保唯一
        text = row['transliteration']
        id_order.append(sample_id)
        preprocess_trace = None
        if len(preprocess_debug_by_id) < 5:
            preprocess_trace = {
                'raw_input': text,
                'correction_enabled': bool(_correction_vocab),
                'correction_triggered': False,
                'name_handling_mode': config.name_handling_mode,
                'placeholder_applied': False,
                'hint_applied': False,
                'month_hints_enabled': bool(config.use_month_hints),
                'month_hints_applied': False,
                'sumerogram_enabled': bool(_sumerogram_glossary),
                'sumerogram_applied': False,
                'silver_enabled': bool(config.pseudo_label_mode),
                'rag_enabled': bool(config.use_rag and rag_index),
                'rag_applied': False,
            }
        
        # 实时进度（每 500 条打印一次）
        if (idx + 1) % 500 == 0:
            elapsed = time.time() - preprocess_start
            speed = (idx + 1) / elapsed
            eta = (len(df) - idx - 1) / speed if speed > 0 else 0
            print(f"   预处理进度: {idx+1}/{len(df)} ({(idx+1)/len(df)*100:.1f}%) | {speed:.1f} it/s | ETA {eta:.0f}s")
            sys.stdout.flush()
        
        
        if not isinstance(text, str) or pd.isna(text):
            _empty_prefix = _get_infer_prefix("", config)
            flat_segments.append((sample_id, _empty_prefix, {}, text))
            stats['empty'] += 1
            cleaned_transliterations[sample_id] = ""
            if preprocess_trace is not None:
                preprocess_trace.update({
                    'after_correction': '',
                    'correction_changed': False,
                    'correction_fixes': [],
                    'after_preprocess': '',
                    'after_silver_clean': '',
                    'silver_changed': False,
                    'after_injections': "",
                    'rag_prefix': "",
                    'final_model_input': _empty_prefix,
                })
                preprocess_debug_by_id[sample_id] = preprocess_trace
            continue
        
        # 0. 编码损坏修复（š→a, ṭ→m, „→-, +→h 逆向修复）
        original_text = text
        correction_fixes = []
        if _correction_vocab and is_corrupted_row(text):
            if preprocess_trace is not None:
                preprocess_trace['correction_triggered'] = True
            text, correction_fixes = _correct_transliteration(text, _correction_vocab, max_edit_distance=0)
            if correction_fixes:
                stats.setdefault('correction_vocab_fixed', 0)
                stats['correction_vocab_fixed'] += 1
        corrected_text = text

        # 1. 基础预处理
        cleaned = preprocess_transliteration(text)
        after_preprocess = cleaned
        
        # 2. Silver 数据专用清洗
        silver_changed = False
        if config.pseudo_label_mode:
            original_cleaned = cleaned
            cleaned = clean_silver_specific_patterns(cleaned)
            if cleaned != original_cleaned:
                stats['silver_cleaned'] += 1
                silver_changed = True
        
        # 存储清洗后的原文（_row_id 唯一，每行一条）
        cleaned_transliterations[sample_id] = cleaned
        
        # 不再使用 structural split：每行只保留一个 segment
        seg = cleaned
        stats['total_segments'] += 1

        mapping = {}
        if config.name_handling_mode == 'placeholder' and placeholder_svc:
            before_placeholder = seg
            seg, _, mapping = placeholder_svc.apply_placeholders(seg)
            if preprocess_trace is not None:
                preprocess_trace['placeholder_applied'] = (seg != before_placeholder)
        elif config.name_handling_mode == 'hint' and hint_lexicon:
            before_hints = seg
            seg = _apply_hints(seg, hint_lexicon, config.max_hints_per_sample)
            if preprocess_trace is not None:
                preprocess_trace['hint_applied'] = (seg != before_hints)
        if config.use_month_hints:
            before_month_hints = seg
            seg = _apply_month_hints(seg)
            if preprocess_trace is not None:
                preprocess_trace['month_hints_applied'] = (seg != before_month_hints)

        # Sumerogram 词义注入
        if _sumerogram_glossary:
            before_sumerogram = seg
            seg = _inject_sumerogram_glossary_hints(seg, _sumerogram_glossary, max_hints=config.max_sumerogram_hints)
            if preprocess_trace is not None:
                preprocess_trace['sumerogram_applied'] = (seg != before_sumerogram)

        # RAG 示例注入（推理时 100% 注入，格式与 train.py 一致）
        rag_prefix = ""
        if config.use_rag and rag_index:
            examples = _rag_retrieve(seg, top_k=config.rag_max_examples)
            if examples:
                parts = [f"{ex['source']} | {ex['target']}" for ex in examples]
                rag_prefix = " || ".join(parts) + " || "
                stats['rag_injected'] += 1
                if preprocess_trace is not None:
                    preprocess_trace['rag_applied'] = True

        final_model_input = rag_prefix + _get_infer_prefix(seg, config) + seg
        flat_segments.append((sample_id, final_model_input, mapping, text))
        if preprocess_trace is not None:
            preprocess_trace.update({
                'after_correction': corrected_text,
                'correction_changed': corrected_text != original_text,
                'correction_fixes': correction_fixes,
                'after_preprocess': after_preprocess,
                'after_silver_clean': cleaned,
                'silver_changed': silver_changed,
                'after_injections': seg,
                'rag_prefix': rag_prefix,
                'final_model_input': final_model_input,
            })
            preprocess_debug_by_id[sample_id] = preprocess_trace
    
    
    preprocess_time = time.time() - preprocess_start
    total_segments = len(flat_segments)
    
    print(f"\n   ✅ 预处理完成 (耗时: {preprocess_time:.1f}s)")
    print(f"   ─────────────────────────────────────")
    print(f"   原始样本:     {stats['total']}")
    print(f"   空样本:       {stats['empty']}")
    if config.pseudo_label_mode:
        print(f"   Silver清洗:   {stats['silver_cleaned']} (破损限定词修复)")
    if stats.get('correction_vocab_fixed', 0) > 0:
        print(f"   编码修复:     {stats['correction_vocab_fixed']} (š→a, ṭ→m 逆向修复)")
    print(f"   总片段数:     {stats['total_segments']}")
    if config.use_rag:
        rag_rate = (stats['rag_injected'] / stats['total_segments']) * 100 if stats['total_segments'] > 0 else 0
        print(f"   RAG 注入:     {stats['rag_injected']} 片段 ({rag_rate:.1f}%)")
    print(f"   ─────────────────────────────────────")
    sys.stdout.flush()
    
    # 动态 Batching：按输入文本长度排序，极大减少 Padding 浪费
    # （最终重组时由于记录了原始的 sample_id，顺序会被自动恢复，因此这里排序不影响结果）
    flat_segments.sort(key=lambda x: len(x[1]), reverse=config.sort_longest_first)
    _sort_dir = "长序列优先（早暴露显存峰值）" if config.sort_longest_first else "短序列优先（Padding 效率优化）"
    print(f"   🔄 动态 Batching: {_sort_dir}")
    sys.stdout.flush()
    
    # 创建简单数据集
    class SegmentDataset(Dataset):
        def __init__(self, items):
            self.items = items
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            return self.items[idx]
    
    dataset = SegmentDataset(flat_segments)
    
    def collate_fn(batch):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        mappings = [item[2] for item in batch]
        return ids, texts, mappings
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # ====================================================================
    # Tokenization Prefetch 工具
    # ====================================================================
    import threading as _th
    from queue import Queue as _Queue

    def _prefetch_tokenized(dl, tok, max_len):
        """后台线程提前 tokenize 下一批，消除 GPU 等待 CPU 的气泡。"""
        _sentinel = object()
        q = _Queue(maxsize=2)

        def _producer():
            for ids, texts, mappings in dl:
                enc = tok(texts, max_length=max_len, padding=True,
                          truncation=True, return_tensors="pt")
                q.put((ids, texts, mappings, enc.input_ids, enc.attention_mask))
            q.put(_sentinel)

        t = _th.Thread(target=_producer, daemon=True)
        t.start()
        while True:
            item = q.get()
            if item is _sentinel:
                break
            yield item

    # ====================================================================
    # 生成翻译
    # ====================================================================
    print(f"\n🔮 生成翻译 ({total_segments} 个片段)...")
    print(f"   Batch Size: {config.batch_size}, Num Beams: {config.num_beams}")
    sys.stdout.flush()
    
    segment_results = []
    debug_samples = []

    max_new = config.max_new_tokens if config.max_new_tokens is not None else config.max_target_length
    
    notebook_postprocessor = VectorizedPostprocessor()

    def _postprocess_batch(texts):
        stripped = [strip_input_artifacts(t) for t in texts]
        return notebook_postprocessor.postprocess_batch(stripped)

    def _postprocess_one(text):
        return _postprocess_batch([text])[0]

    def _mbr_select_with_config(candidates):
        return mbr_select(
            candidates,
            metric=config.mbr_metric,
            pool_cap=config.mbr_pool_cap,
            w_chrf=config.mbr_w_chrf,
            w_bleu=config.mbr_w_bleu,
            w_jaccard=config.mbr_w_jaccard,
            w_length=config.mbr_w_length,
        )

    def _hybrid_mbr_select(candidates):
        return mbr_select(
            candidates,
            metric=config.mbr_metric,
            pool_cap=config.mbr_pool_cap,
        )

    gen_start = time.time()

    def _maybe_log_stage_timing(label: str, batch_idx: int, processed: int) -> None:
        if not config.enable_stage_timing_logs:
            return
        every = int(getattr(config, "stage_timing_log_every", 0) or 0)
        if not _should_log_stage_timing(batch_idx, processed, total_segments, every):
            return
        print(f"   ⏱ [{label} timing] {_format_stage_timing_summary(_snapshot_stage_timing())}")

    if config.decode_strategy == "multi_model_mbr":
        import gc
        from concurrent.futures import ThreadPoolExecutor

        sample_temperatures, num_sample_per_temp = config.get_multi_model_sampling_plan()
        total_sample_cands = len(sample_temperatures) * num_sample_per_temp

        print(
            f"   multi_model_mbr: {len(config.model_paths)} 个模型 × "
            f"(Beam候选 {config.multi_model_beam_cands} + 采样候选 {total_sample_cands})"
        )
        if total_sample_cands > 0:
            print(
                f"   采样计划: temps={sample_temperatures}, "
                f"{num_sample_per_temp}/temp, top_p={config.mbr_top_p}"
            )
        print(
            f"   Weighted MBR: metric={config.mbr_metric}, "
            f"weights=({config.mbr_w_chrf:.2f}, {config.mbr_w_bleu:.2f}, "
            f"{config.mbr_w_jaccard:.2f}, {config.mbr_w_length:.2f}), cap={config.mbr_pool_cap}"
        )

        row_id_to_seg_idx = {item[0]: idx for idx, item in enumerate(flat_segments)}
        all_segment_candidates = [[] for _ in range(total_segments)]
        _cand_lock = threading.Lock()
        mbr_pool = ThreadPoolExecutor(max_workers=4)
        mbr_futures = {}  # seg_idx -> Future

        def _append_candidates(sp_ids_list, sp_candidate_sets):
            with _cand_lock:
                for s_id, cand_set in zip(sp_ids_list, sp_candidate_sets):
                    seg_idx = row_id_to_seg_idx[s_id]
                    all_segment_candidates[seg_idx].extend(cand_set)

        for model_idx, model_path in enumerate(config.model_paths):
            is_last_model = (model_idx == len(config.model_paths) - 1)
            print(f"\n🤖 [{model_idx+1}/{len(config.model_paths)}] 加载模型: {model_path}")

            with _timed("model_load_main"):
                model_main, tokenizer_current = load_single_model(model_path, config.device, config)
            current_replicas = [model_main]
            model2 = None

            if n_gpus > 1:
                dev2 = torch.device("cuda:1")
                try:
                    with _timed("model_load_replica"):
                        model2, _ = load_single_model(model_path, dev2, config)
                    current_replicas.append(model2)
                    print(f"   ✓ 双卡数据并行就绪（当前模型）")
                except Exception as e:
                    print(f"   ⚠️ 第二张卡加载失败 ({e})，降级单卡")
                    if model2 is not None:
                        del model2
                        model2 = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

            use_dual_gpu = len(current_replicas) > 1
            _worker_errors = []
            pending_candidate_jobs = []
            _pending_lock = threading.Lock()

            def _decode_postprocess_split(out_cpu, tokenizer_ref, cands_per_item, batch_n, sp_mappings):
                """Decode → postprocess → placeholder restore → split into per-item candidate lists."""
                with _timed("decode"):
                    raw = decode_sequences(out_cpu, tokenizer=tokenizer_ref)
                with _timed("postprocess"):
                    decoded = _postprocess_batch(raw)
                result = [[] for _ in range(batch_n)]
                for i in range(batch_n):
                    cand_set = decoded[i * cands_per_item:(i + 1) * cands_per_item]
                    if placeholder_svc and sp_mappings[i]:
                        with _timed("placeholder_restore"):
                            cand_set = [placeholder_svc.restore(c, sp_mappings[i]) for c in cand_set]
                    result[i] = cand_set
                return result

            def _finalize_candidate_batch(batch_n, out_beam_cpu, out_sample_cpus, sp_mappings):
                local_candidate_sets = [[] for _ in range(batch_n)]
                if out_beam_cpu is not None:
                    for i, cs in enumerate(_decode_postprocess_split(
                        out_beam_cpu, tokenizer_current, config.multi_model_beam_cands, batch_n, sp_mappings
                    )):
                        local_candidate_sets[i].extend(cs)
                for out_sample_cpu in out_sample_cpus or []:
                    for i, cs in enumerate(_decode_postprocess_split(
                        out_sample_cpu, tokenizer_current, num_sample_per_temp, batch_n, sp_mappings
                    )):
                        local_candidate_sets[i].extend(cs)
                return local_candidate_sets

            def _drain_candidate_jobs(block: bool = False):
                remaining = []
                with _pending_lock:
                    jobs = pending_candidate_jobs[:]
                    pending_candidate_jobs.clear()

                for sp_ids_list, fut in jobs:
                    if not block and not fut.done():
                        remaining.append((sp_ids_list, fut))
                        continue

                    with _timed("mbr_finalize_wait"):
                        local_candidate_sets = fut.result()
                    _append_candidates(sp_ids_list, local_candidate_sets)

                    if is_last_model:
                        for s_id in sp_ids_list:
                            si = row_id_to_seg_idx[s_id]
                            if si not in mbr_futures and all_segment_candidates[si]:
                                mbr_futures[si] = mbr_pool.submit(
                                    _mbr_select_with_config, list(all_segment_candidates[si])
                                )

                if remaining:
                    with _pending_lock:
                        pending_candidate_jobs.extend(remaining)

            def _sample_on_device(sp_model, sp_ids, sp_ids_t, sp_mask_t, sp_mappings):
                try:
                    sp_dev = next(sp_model.parameters()).device
                    with _timed("h2d"):
                        sp_ids_t = sp_ids_t.to(sp_dev)
                        sp_mask_t = sp_mask_t.to(sp_dev)
                    batch_n = sp_ids_t.shape[0]

                    base_beam_width = max(
                        config.multi_model_beam_cands,
                        _resolve_generation_value(sp_model, config.num_beams, "num_beams", default=1),
                    )
                    beam_width = base_beam_width
                    length_penalty = _resolve_generation_value(sp_model, config.length_penalty, "length_penalty")
                    repetition_penalty = _resolve_generation_value(sp_model, config.repetition_penalty, "repetition_penalty")
                    early_stopping = _resolve_generation_value(sp_model, config.early_stopping, "early_stopping")
                    out_beam_cpu = None
                    out_sample_cpus = []

                    if config.use_adaptive_beams and base_beam_width > config.multi_model_beam_cands:
                        median_input_len = float(sp_mask_t.sum(dim=1).float().median().item())
                        short_beam_width = max(config.multi_model_beam_cands, max(1, base_beam_width // 2))
                        if median_input_len < config.adaptive_beam_len_threshold:
                            beam_width = short_beam_width

                    # 1. Beam Search candidates (batched)
                    if config.multi_model_beam_cands > 0:
                        beam_kwargs = {
                            "input_ids": sp_ids_t,
                            "attention_mask": sp_mask_t,
                            "max_new_tokens": max_new,
                            "do_sample": False,
                            "num_beams": beam_width,
                            "num_return_sequences": config.multi_model_beam_cands,
                        }
                        if length_penalty is not None:
                            beam_kwargs["length_penalty"] = length_penalty
                        if repetition_penalty is not None:
                            beam_kwargs["repetition_penalty"] = repetition_penalty
                        if early_stopping is not None:
                            beam_kwargs["early_stopping"] = early_stopping

                        with _timed("beam_generate"):
                            out_beam = sp_model.generate(**beam_kwargs)
                        with _timed("beam_cpu_copy"):
                            out_beam_cpu = out_beam.cpu()

                    # 2. Sampling candidates (batched)
                    if sample_temperatures and num_sample_per_temp > 0:
                        for temp in sample_temperatures:
                            sample_kwargs = {
                                "input_ids": sp_ids_t,
                                "attention_mask": sp_mask_t,
                                "max_new_tokens": max_new,
                                "do_sample": True,
                                "top_p": config.mbr_top_p,
                                "temperature": temp,
                                "num_return_sequences": num_sample_per_temp,
                                "num_beams": 1,
                                "early_stopping": False,
                                "length_penalty": 1.0,
                            }
                            if repetition_penalty is not None:
                                sample_kwargs["repetition_penalty"] = repetition_penalty

                            with _timed("sample_generate"):
                                out_sample = sp_model.generate(**sample_kwargs)
                            with _timed("sample_cpu_copy"):
                                out_sample_cpus.append(out_sample.cpu())

                    finalize_future = mbr_pool.submit(
                        _finalize_candidate_batch,
                        batch_n,
                        out_beam_cpu,
                        out_sample_cpus,
                        sp_mappings,
                    )
                    with _pending_lock:
                        pending_candidate_jobs.append((list(sp_ids), finalize_future))
                except Exception as e:
                    _worker_errors.append(e)
                    import traceback
                    traceback.print_exc()

            pbar = tqdm(
                _prefetch_tokenized(dataloader, tokenizer_current, config.max_input_length),
                desc=f"模型{model_idx+1}采样中",
                total=len(dataloader),
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
            )

            with torch.no_grad():
                for batch_idx, (ids, input_texts, mappings, input_ids, attention_mask) in enumerate(pbar):

                    if use_dual_gpu and len(ids) > 1:
                        mid = len(ids) // 2
                        t0 = threading.Thread(
                            target=_sample_on_device,
                            args=(
                                current_replicas[0],
                                ids[:mid],
                                input_ids[:mid],
                                attention_mask[:mid],
                                mappings[:mid],
                            ),
                        )
                        t1 = threading.Thread(
                            target=_sample_on_device,
                            args=(
                                current_replicas[1],
                                ids[mid:],
                                input_ids[mid:],
                                attention_mask[mid:],
                                mappings[mid:],
                            ),
                        )
                        t0.start(); t1.start()
                        t0.join(); t1.join()
                    else:
                        _sample_on_device(
                            current_replicas[0],
                            ids,
                            input_ids,
                            attention_mask,
                            mappings,
                        )

                    if _worker_errors:
                        raise _worker_errors[0]

                    _drain_candidate_jobs(block=False)

                    processed = min((batch_idx + 1) * config.batch_size, total_segments)
                    elapsed = time.time() - gen_start
                    speed = processed / elapsed if elapsed > 0 else 0
                    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                    pbar.set_postfix({
                        'seg': f'{processed}/{total_segments}',
                        'speed': f'{speed:.1f}/s',
                        'GPU': f'{gpu_mem:.1f}G'
                    })
                    _maybe_log_generation_progress(
                        label=f"Model {model_idx + 1}",
                        batch_idx=batch_idx,
                        processed=processed,
                        total_segments=total_segments,
                        elapsed=elapsed,
                        gpu_mem=gpu_mem,
                        every=config.progress_log_every,
                    )
                    _maybe_log_stage_timing(f"Model {model_idx + 1}", batch_idx, processed)

            pbar.close()
            _drain_candidate_jobs(block=True)

            # 阅后即焚：释放当前模型显存
            for m in current_replicas:
                del m
            del current_replicas
            if model2 is not None:
                del model2
                model2 = None
            del model_main
            del tokenizer_current
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"   ✓ 模型 {model_idx+1} 完成，已释放显存")

        # 收集异步 MBR 结果（大部分 future 在最后一个模型的 GPU 推理期间已完成）
        print(f"\n🎯 MBR 共识选择 ({config.mbr_metric})... (异步已提交 {len(mbr_futures)}/{total_segments})")
        for seg_idx, (sample_id, input_text, _, _) in enumerate(tqdm(flat_segments, desc="MBR收集中")):
            if seg_idx in mbr_futures:
                with _timed("mbr_collect_wait"):
                    best_text, best_score = mbr_futures[seg_idx].result()
            else:
                candidates = all_segment_candidates[seg_idx]
                if candidates:
                    with _timed("mbr_collect_sync"):
                        best_text, best_score = _mbr_select_with_config(candidates)
                else:
                    best_text, best_score = "", 0.0
            raw_example = all_segment_candidates[seg_idx][0] if all_segment_candidates[seg_idx] else ""

            segment_results.append({
                'id': sample_id,
                'text': best_text,
                'confidence': best_score,
                'input_len': len(input_text),
                'candidates': list(all_segment_candidates[seg_idx]) if config.mrt_data_mode else None,
            })

            if len(debug_samples) < 5:
                debug_samples.append({
                    'id': sample_id,
                    'input': input_text,
                    'raw_output': raw_example,
                    'cleaned_output': best_text,
                    'preprocess_trace': preprocess_debug_by_id.get(sample_id, {}),
                })

        mbr_pool.shutdown(wait=False)
    else:
        import threading
        from concurrent.futures import ThreadPoolExecutor

        _res_lock = threading.Lock()
        _worker_errors = []  # 捕获线程内异常
        if config.decode_strategy == "hybrid_mbr":
            print(
                f"   hybrid_mbr: beam候选 {config.hybrid_beam_cands} + "
                f"采样候选 {config.hybrid_sample_cands} "
                f"(temp={config.hybrid_temperature}, top_p={config.hybrid_top_p})"
            )
            print(f"   Hybrid MBR: metric={config.mbr_metric}, cap={config.mbr_pool_cap}")

        mbr_pool = ThreadPoolExecutor(max_workers=4) if config.decode_strategy in ("sampling_mbr", "hybrid_mbr") else None
        pending_result_jobs = []
        _pending_result_lock = threading.Lock()

        def _append_results(sp_ids_list, sp_texts, sp_raw, sp_clean, sp_scores, sp_candidates=None):
            """线程安全的收集结果"""
            with _res_lock:
                for idx, (s_id, i_txt, r_out, c_out, sc) in enumerate(zip(sp_ids_list, sp_texts, sp_raw, sp_clean, sp_scores)):
                    segment_results.append({
                        'id': s_id,
                        'text': c_out,
                        'confidence': sc,
                        'input_len': len(i_txt),
                        'candidates': (
                            list(sp_candidates[idx])
                            if (config.mrt_data_mode and sp_candidates is not None)
                            else ([c_out] if config.mrt_data_mode else None)
                        ),
                    })
                    if len(debug_samples) < 5:
                        debug_samples.append({
                            'id': s_id,
                            'input': i_txt,
                            'raw_output': r_out,
                            'cleaned_output': c_out,
                            'preprocess_trace': preprocess_debug_by_id.get(s_id, {}),
                        })

        def _drain_result_jobs(block: bool = False):
            remaining = []
            with _pending_result_lock:
                jobs = pending_result_jobs[:]
                pending_result_jobs.clear()

            for sp_ids_list, sp_texts_list, fut in jobs:
                if not block and not fut.done():
                    remaining.append((sp_ids_list, sp_texts_list, fut))
                    continue
                with _timed("mbr_finalize_wait"):
                    fut_value = fut.result()
                    if len(fut_value) == 3:
                        sp_decoded, sp_scores, sp_candidates = fut_value
                    else:
                        sp_decoded, sp_scores = fut_value
                        sp_candidates = None
                _append_results(
                    sp_ids_list,
                    sp_texts_list,
                    sp_decoded,
                    sp_decoded,
                    sp_scores,
                    sp_candidates=sp_candidates,
                )

            if remaining:
                with _pending_result_lock:
                    pending_result_jobs.extend(remaining)

        # 预先解析解码参数
        gen_kwargs = {}
        n_cand = 0
        sampling_temperatures = []
        num_sample_per_temp = 0
        if config.decode_strategy == "beam":
            gen_kwargs = {
                'max_new_tokens': max_new,
                'return_dict_in_generate': True,
                'output_scores': True,
            }
            if config.num_beams is not None: gen_kwargs['num_beams'] = config.num_beams
            if config.length_penalty is not None: gen_kwargs['length_penalty'] = config.length_penalty
            if config.repetition_penalty is not None: gen_kwargs['repetition_penalty'] = config.repetition_penalty
            if config.no_repeat_ngram_size is not None: gen_kwargs['no_repeat_ngram_size'] = config.no_repeat_ngram_size
            if config.early_stopping is not None: gen_kwargs['early_stopping'] = config.early_stopping
        elif config.decode_strategy == "sampling_mbr":
            sampling_temperatures = [float(t) for t in (config.sample_temperatures or [])]
            num_sample_per_temp = int(config.num_sample_per_temp or 0)
            if sampling_temperatures and num_sample_per_temp > 0:
                n_cand = len(sampling_temperatures) * num_sample_per_temp
            else:
                sampling_temperatures = [config.mbr_temperature]
                num_sample_per_temp = config.mbr_num_candidates
                n_cand = config.mbr_num_candidates
        elif config.decode_strategy == "hybrid_mbr":
            gen_kwargs = {
                'max_new_tokens': max_new,
            }
        else:
            raise ValueError(f"Unsupported decode_strategy: {config.decode_strategy}")

        def _run_on_device(sp_model, sp_ids, sp_texts, sp_ids_t, sp_mask_t, sp_mappings):
            """在指定模型（设备）上运行一个子 batch，线程安全"""
            try:
                sp_dev = next(sp_model.parameters()).device
                with _timed("h2d"):
                    sp_ids_t = sp_ids_t.to(sp_dev)
                    sp_mask_t = sp_mask_t.to(sp_dev)

                if config.decode_strategy == "sampling_mbr":
                    out_cpus = []
                    for temp in sampling_temperatures:
                        sample_kwargs = {
                            "input_ids": sp_ids_t,
                            "attention_mask": sp_mask_t,
                            "max_new_tokens": max_new,
                            "do_sample": True,
                            "top_p": config.mbr_top_p,
                            "temperature": temp,
                            "num_return_sequences": num_sample_per_temp,
                            "num_beams": 1,
                            "early_stopping": False,
                            "length_penalty": 1.0,
                        }
                        if config.repetition_penalty is not None:
                            sample_kwargs["repetition_penalty"] = config.repetition_penalty
                        with _timed("sample_generate"):
                            out = sp_model.generate(**sample_kwargs)
                        with _timed("sample_cpu_copy"):
                            out_cpus.append(out.cpu())

                    def _finalize_sampling_batch(out_cpu_list, mappings_local):
                        decoded_by_temp = []
                        for out_cpu_local in out_cpu_list:
                            with _timed("decode"):
                                raw_decoded = decode_sequences(out_cpu_local, tokenizer=tokenizer)
                            with _timed("postprocess"):
                                decoded_by_temp.append(_postprocess_batch(raw_decoded))
                        grouped_candidates = []
                        for i in range(len(mappings_local)):
                            cand_set = []
                            for all_cand in decoded_by_temp:
                                start = i * num_sample_per_temp
                                end = start + num_sample_per_temp
                                cand_set.extend(all_cand[start:end])
                            if placeholder_svc and mappings_local[i]:
                                with _timed("placeholder_restore"):
                                    cand_set = [placeholder_svc.restore(c, mappings_local[i]) for c in cand_set]
                            grouped_candidates.append(cand_set)

                        decoded, scores = [], []
                        for cand_set in grouped_candidates:
                            best_text, best_score = _mbr_select_with_config(cand_set)
                            decoded.append(best_text)
                            scores.append(best_score)
                        return decoded, scores, grouped_candidates

                    finalize_future = mbr_pool.submit(_finalize_sampling_batch, out_cpus, sp_mappings)
                    with _pending_result_lock:
                        pending_result_jobs.append((list(sp_ids), list(sp_texts), finalize_future))
                elif config.decode_strategy == "hybrid_mbr":
                    length_penalty = _resolve_generation_value(sp_model, config.length_penalty, "length_penalty")
                    repetition_penalty = _resolve_generation_value(sp_model, config.repetition_penalty, "repetition_penalty")
                    early_stopping = _resolve_generation_value(sp_model, config.early_stopping, "early_stopping")
                    base_beam_width = max(
                        config.hybrid_beam_cands,
                        _resolve_generation_value(sp_model, config.num_beams, "num_beams", default=1),
                    )
                    beam_width = base_beam_width
                    if config.use_adaptive_beams and base_beam_width > config.hybrid_beam_cands:
                        median_input_len = float(sp_mask_t.sum(dim=1).float().median().item())
                        short_beam_width = max(config.hybrid_beam_cands, max(1, base_beam_width // 2))
                        if median_input_len < config.adaptive_beam_len_threshold:
                            beam_width = short_beam_width

                    beam_kwargs = {
                        "input_ids": sp_ids_t,
                        "attention_mask": sp_mask_t,
                        "max_new_tokens": max_new,
                        "do_sample": False,
                        "num_beams": beam_width,
                        "num_return_sequences": config.hybrid_beam_cands,
                    }
                    if length_penalty is not None:
                        beam_kwargs["length_penalty"] = length_penalty
                    if repetition_penalty is not None:
                        beam_kwargs["repetition_penalty"] = repetition_penalty
                    if early_stopping is not None:
                        beam_kwargs["early_stopping"] = early_stopping

                    sample_kwargs = {
                        "input_ids": sp_ids_t,
                        "attention_mask": sp_mask_t,
                        "max_new_tokens": max_new,
                        "do_sample": True,
                        "top_p": config.hybrid_top_p,
                        "temperature": config.hybrid_temperature,
                        "num_return_sequences": config.hybrid_sample_cands,
                        "num_beams": 1,
                        "early_stopping": False,
                        "length_penalty": 1.0,
                    }
                    if repetition_penalty is not None:
                        sample_kwargs["repetition_penalty"] = repetition_penalty

                    with _timed("beam_generate"):
                        out_beam = sp_model.generate(**beam_kwargs)
                    with _timed("beam_cpu_copy"):
                        out_beam_cpu = out_beam.cpu()
                    with _timed("sample_generate"):
                        out_sample = sp_model.generate(**sample_kwargs)
                    with _timed("sample_cpu_copy"):
                        out_sample_cpu = out_sample.cpu()

                    def _finalize_hybrid_batch(out_beam_local, out_sample_local, mappings_local):
                        decoded, scores, grouped_candidates, finalize_timings = finalize_hybrid_candidate_batch(
                            out_beam_local,
                            out_sample_local,
                            tokenizer=tokenizer,
                            beam_cands=config.hybrid_beam_cands,
                            sample_cands=config.hybrid_sample_cands,
                            postprocess_batch=_postprocess_batch,
                            restore_mappings=mappings_local,
                            restore_fn=(
                                (lambda text, mapping: placeholder_svc.restore(text, mapping))
                                if placeholder_svc else None
                            ),
                            mbr_selector=_hybrid_mbr_select,
                        )
                        _record_stage_timing("decode", finalize_timings["decode"])
                        _record_stage_timing("postprocess", finalize_timings["postprocess"])
                        if finalize_timings["restore"] > 0:
                            _record_stage_timing("placeholder_restore", finalize_timings["restore"])
                        return decoded, scores, grouped_candidates

                    finalize_future = mbr_pool.submit(_finalize_hybrid_batch, out_beam_cpu, out_sample_cpu, sp_mappings)
                    with _pending_result_lock:
                        pending_result_jobs.append((list(sp_ids), list(sp_texts), finalize_future))
                else:
                    with _timed("beam_generate"):
                        sp_out = sp_model.generate(
                            input_ids=sp_ids_t,
                            attention_mask=sp_mask_t,
                            **gen_kwargs,
                        )
                    with _timed("decode"):
                        raw_dec = decode_sequences(sp_out.sequences, tokenizer=tokenizer)
                    if hasattr(sp_out, 'sequences_scores') and sp_out.sequences_scores is not None:
                        raw_sc = sp_out.sequences_scores.cpu().float().numpy()
                        num_ret = gen_kwargs.get('num_return_sequences', 1)
                        if len(raw_sc) == sp_ids_t.shape[0] * num_ret:
                            sc = raw_sc[::num_ret].tolist()
                        else:
                            sc = raw_sc[:sp_ids_t.shape[0]].tolist()
                    else:
                        sc = [0.0] * sp_ids_t.shape[0]
                    with _timed("postprocess"):
                        cln = _postprocess_batch(raw_dec)
                    if placeholder_svc:
                        for k, (t, m) in enumerate(zip(cln, sp_mappings)):
                            if m:
                                with _timed("placeholder_restore"):
                                    cln[k] = placeholder_svc.restore(t, m)
                    _append_results(sp_ids, sp_texts, raw_dec, cln, sc)
            except Exception as e:
                _worker_errors.append(e)
                import traceback
                traceback.print_exc()

        pbar = tqdm(
            _prefetch_tokenized(dataloader, tokenizer, config.max_input_length),
            desc="翻译中",
            total=len(dataloader),
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

        use_dual_gpu = len(model_replicas) > 1

        with torch.no_grad():
            for batch_idx, (ids, input_texts, mappings, input_ids, attention_mask) in enumerate(pbar):

                if use_dual_gpu and len(ids) > 1:
                    mid = len(ids) // 2
                    t0 = threading.Thread(target=_run_on_device, args=(
                        model_replicas[0], ids[:mid], input_texts[:mid],
                        input_ids[:mid], attention_mask[:mid], mappings[:mid]))
                    t1 = threading.Thread(target=_run_on_device, args=(
                        model_replicas[1], ids[mid:], input_texts[mid:],
                        input_ids[mid:], attention_mask[mid:], mappings[mid:]))
                    t0.start(); t1.start()
                    t0.join(); t1.join()
                    if _worker_errors:
                        raise _worker_errors[0]
                else:
                    _run_on_device(model_replicas[0], ids, input_texts,
                                   input_ids, attention_mask, mappings)
                    if _worker_errors:
                        raise _worker_errors[0]

                if config.decode_strategy in ("sampling_mbr", "hybrid_mbr"):
                    _drain_result_jobs(block=False)

                processed = min((batch_idx + 1) * config.batch_size, total_segments)
                elapsed = time.time() - gen_start
                speed = processed / elapsed if elapsed > 0 else 0
                gpu_mem = torch.cuda.memory_allocated(0) / 1024**3 if torch.cuda.is_available() else 0
                pbar.set_postfix({
                    'seg': f'{processed}/{total_segments}',
                    'speed': f'{speed:.1f}/s',
                    'GPU': f'{gpu_mem:.1f}G'
                })
                _maybe_log_generation_progress(
                    label=config.decode_strategy,
                    batch_idx=batch_idx,
                    processed=processed,
                    total_segments=total_segments,
                    elapsed=elapsed,
                    gpu_mem=gpu_mem,
                    every=config.progress_log_every,
                )
                _maybe_log_stage_timing(config.decode_strategy, batch_idx, processed)

        pbar.close()
        if config.decode_strategy in ("sampling_mbr", "hybrid_mbr"):
            _drain_result_jobs(block=True)
        if mbr_pool is not None:
            mbr_pool.shutdown(wait=True)

    gen_time = time.time() - gen_start
    print(f"\n   ✅ 生成完成 (耗时: {gen_time:.1f}s, 平均速度: {total_segments/gen_time:.2f} seg/s)")
    if config.enable_stage_timing_logs:
        print(f"   ⏱ stage timing total: {_format_stage_timing_summary(_snapshot_stage_timing())}")

    # 关键校验：结果数量必须等于片段数量
    if len(segment_results) != total_segments:
        print(f"\n   ⚠️ 警告: 结果数量 ({len(segment_results)}) != 片段数量 ({total_segments})!")
        print(f"   缺失 {total_segments - len(segment_results)} 个片段的结果")
    sys.stdout.flush()
    
    # ====================================================================
    # 重组切分片段
    # ====================================================================
    from collections import OrderedDict
    
    # 聚合每个样本的片段
    final_data = [] # (id, full_translation, avg_confidence, full_transliteration)
    
    # 构建 ID -> [segments] 的映射
    id_map = OrderedDict()
    for res in segment_results:
        sid = res['id']
        if sid not in id_map:
            id_map[sid] = []
        id_map[sid].append(res)
    
    # 按行级 ID 顺序重组（每个 row_id 当前仅 1 段，保留该逻辑以便后续兼容）
    seen_ids = set()
    unique_id_order = []
    for sid in id_order:
        if sid not in seen_ids:
            seen_ids.add(sid)
            unique_id_order.append(sid)
    for row_id in unique_id_order:
        segments = id_map.get(row_id, [])
        if not segments:
            final_data.append({
                '_row_id': row_id,
                'translation': "",
                'confidence': -999.0,
                'candidates': [],
            })
            continue
        
        # 拼接翻译（过滤空段，避免多余空格）
        full_translation = ' '.join(s['text'] for s in segments if s['text'].strip())
        
        # 平均置信度 (按片段长度加权可能更好，这里简单平均)
        avg_confidence = sum(s['confidence'] for s in segments) / len(segments)
        merged_candidates = []
        seen_candidates = set()
        for seg in segments:
            for cand in seg.get('candidates') or []:
                cand_text = str(cand).strip()
                if cand_text and cand_text not in seen_candidates:
                    seen_candidates.add(cand_text)
                    merged_candidates.append(cand_text)
        
        final_data.append({
            '_row_id': row_id,
            'translation': full_translation,
            'confidence': avg_confidence,
            'candidates': merged_candidates,
        })

    # 合并原始信息（通过 _row_id 关联回 df）
    result_df = pd.DataFrame(final_data)
    
    # 使用清洗后的 transliteration
    result_df['transliteration'] = result_df['_row_id'].map(cleaned_transliterations)
    
    # 关联原始 ID 列
    row_id_to_orig_id = df.set_index('_row_id')[id_col].to_dict()
    result_df[id_col] = result_df['_row_id'].map(row_id_to_orig_id)
    
    # 合并 genre 信息 (如果存在)
    if 'genre' in df.columns:
        row_id_to_genre = df.set_index('_row_id')['genre'].to_dict()
        result_df['genre'] = result_df['_row_id'].map(row_id_to_genre)
    else:
        result_df['genre'] = 'unknown'
    if config.mrt_data_mode:
        if 'translation' in df.columns:
            row_id_to_ref = df.set_index('_row_id')['translation'].to_dict()
        elif 'target_text' in df.columns:
            row_id_to_ref = df.set_index('_row_id')['target_text'].to_dict()
        else:
            row_id_to_ref = {}
        result_df['reference'] = result_df['_row_id'].map(row_id_to_ref).fillna("")
        result_df['candidate_count'] = result_df['candidates'].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
    
    # 删除临时列
    result_df = result_df.drop(columns=['_row_id'])
    
    # ====================================================================
    # 伪标签过滤 (仅在伪标签模式下)
    # ====================================================================
    if config.mrt_data_mode:
        print(f"\n🧪 导出 MRT 候选训练数据...")
        raw_records = []
        for _, row in result_df.iterrows():
            raw_records.append({
                id_col: row[id_col],
                "oare_id": row[id_col],
                "transliteration": row["transliteration"],
                "reference": row.get("reference", ""),
                "translation": row["translation"],
                "genre": row.get("genre", "unknown"),
                "confidence": float(row["confidence"]),
                "candidate_count": int(row.get("candidate_count", 0)),
                "candidates": row["candidates"] if isinstance(row["candidates"], list) else [],
            })

        with open(config.output_mrt_raw, "w", encoding="utf-8") as f:
            for rec in raw_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"   💾 保存 MRT 原始候选: {config.output_mrt_raw} ({len(raw_records)} 条)")

        filtered_records = [
            rec for rec in raw_records
            if str(rec.get("reference", "")).strip()
            and len([c for c in rec.get("candidates", []) if str(c).strip()]) >= config.mrt_min_unique_candidates
        ]
        with open(config.output_mrt_filtered, "w", encoding="utf-8") as f:
            for rec in filtered_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"   ✅ 保存 MRT 训练候选: {config.output_mrt_filtered} ({len(filtered_records)} 条)")
        sys.stdout.flush()

    elif config.pseudo_label_mode:
        print(f"\n🔍 执行伪标签过滤...")
        
        # 计算长度比率
        result_df['src_len'] = result_df['transliteration'].astype(str).apply(len)
        result_df['tgt_len'] = result_df['translation'].astype(str).apply(len)
        result_df['len_ratio'] = result_df['tgt_len'] / (result_df['src_len'] + 1e-9)
        
        # 保存原始结果 (Raw)
        raw_cols = [id_col, 'transliteration', 'translation', 'genre', 'confidence', 'len_ratio']
        result_df[raw_cols].to_csv(config.output_silver_raw, index=False)
        print(f"   💾 保存原始伪标签 (Raw): {config.output_silver_raw} ({len(result_df)} 条)")
        
        # 应用过滤
        # 1. 置信度过滤（根据解码策略动态选择阈值）
        if config.decode_strategy == "beam":
            # Beam 模式: confidence 是 log_prob（负数，越接近 0 越好）
            cond_conf = result_df['confidence'] >= config.filter_min_confidence
            print(f"   置信度阈值 (log_prob): >= {config.filter_min_confidence}")
        else:
            # MBR 模式: confidence 是重排后的共识分数
            cond_conf = result_df['confidence'] >= config.filter_min_mbr_score
            print(f"   置信度阈值 (MBR {config.mbr_metric}): >= {config.filter_min_mbr_score}")
        # 2. 长度比率过滤
        cond_ratio = (result_df['len_ratio'] >= config.filter_min_len_ratio) & \
                     (result_df['len_ratio'] <= config.filter_max_len_ratio)
        
        filtered_df = result_df[cond_conf & cond_ratio].copy()
        
        # 统计
        print(f"   过滤前: {len(result_df)}")
        print(f"   过滤后: {len(filtered_df)} (保留率: {len(filtered_df)/len(result_df)*100:.1f}%)")
        print(f"   - 置信度不达标: {len(result_df) - cond_conf.sum()}")
        print(f"   - 长度比率异常: {len(result_df) - cond_ratio.sum()}")
        
        # 保存标准格式 (用于训练)
        clean_cols = [id_col, 'transliteration', 'translation']
        filtered_df[clean_cols].to_csv(config.output_silver_filtered, index=False)
        print(f"   ✅ 保存最终伪标签 (Filtered): {config.output_silver_filtered}")
        sys.stdout.flush()
        
    elif getattr(config, 'score_mode', False) and config.score_input_csv:
        # Score 模式：计算 BLEU / chrF++ / CER 并保存带分数的 CSV
        import sacrebleu
        predictions = result_df['translation'].fillna("").astype(str).tolist()
        references = _score_ground_truth

        # 对齐长度
        n = min(len(predictions), len(references))
        predictions = predictions[:n]
        references = references[:n]

        # Corpus-level metrics
        bleu = sacrebleu.corpus_bleu(predictions, [references])
        chrf = sacrebleu.corpus_chrf(predictions, [references], word_order=2)

        # Sentence-level scores
        bleu_scorer = sacrebleu.metrics.BLEU(effective_order=True)
        chrf_scorer = sacrebleu.metrics.CHRF(word_order=2)
        sent_bleu = []
        sent_chrf = []
        sent_cer = []
        for pred, ref in zip(predictions, references):
            try:
                sent_bleu.append(bleu_scorer.sentence_score(pred, [ref]).score)
            except Exception:
                sent_bleu.append(0.0)
            sent_chrf.append(chrf_scorer.sentence_score(pred, [ref]).score)
            # Simple CER (character error rate)
            if ref:
                import difflib
                s = difflib.SequenceMatcher(None, ref, pred)
                edits = sum(max(i2-i1, j2-j1) for tag, i1, i2, j1, j2 in s.get_opcodes() if tag != 'equal')
                sent_cer.append(edits / max(len(ref), 1))
            else:
                sent_cer.append(0.0 if not pred else 1.0)

        import math
        geom = math.sqrt(max(bleu.score, 0.0) * max(chrf.score, 0.0))
        avg_cer = sum(sent_cer) / max(len(sent_cer), 1)

        print(f"\n{'='*60}")
        print(f"📊 Ground-Truth 评分结果 ({n} 条)")
        print(f"{'='*60}")
        print(f"   BLEU      : {bleu.score:.2f}")
        print(f"   chrF++    : {chrf.score:.2f}")
        print(f"   Geom Mean : {geom:.2f}")
        print(f"   Avg CER   : {avg_cer:.4f}")
        print(f"{'='*60}")

        # 保存带分数的 CSV
        score_df = result_df.head(n).copy()
        score_df['reference'] = references
        score_df['sent_bleu'] = sent_bleu
        score_df['sent_chrf'] = sent_chrf
        score_df['sent_cer'] = sent_cer
        score_df.to_csv(config.score_output_csv, index=False)
        print(f"   ✅ 评分结果保存至: {config.score_output_csv}")

    else:
        # 标准模式：只保存 ID 和 Translation
        submission = result_df[[id_col, 'translation']].rename(columns={id_col: 'id'})
        
        # ====== 关键校验：防止 Submission Scoring Error ======
        # 1. translation 列不能有 NaN（空翻译用空字符串代替）
        nan_count = submission['translation'].isna().sum()
        if nan_count > 0:
            print(f"   ⚠️ 修复 {nan_count} 个 NaN translation → 空字符串")
            submission['translation'] = submission['translation'].fillna("")
        # 2. 行数必须与输入一致
        if len(submission) != len(df):
            print(f"   🚨 行数不匹配！submission={len(submission)}, test={len(df)}")
        # 3. ID 必须完整覆盖
        expected_ids = set(df[id_col].tolist())
        actual_ids = set(submission['id'].tolist())
        missing_ids = expected_ids - actual_ids
        if missing_ids:
            print(f"   🚨 缺失 {len(missing_ids)} 个 ID: {list(missing_ids)[:10]}...")
            # 补全缺失 ID（空翻译）
            missing_rows = pd.DataFrame({'id': list(missing_ids), 'translation': ''})
            submission = pd.concat([submission, missing_rows], ignore_index=True)
        # 4. 按 ID 排序（与 test.csv 顺序一致）
        submission = submission.sort_values('id').reset_index(drop=True)
        
        submission.to_csv(config.output_csv, index=False)
        print(f"\n✅ 预测结果保存至：{config.output_csv} ({len(submission)} 行)")
    
    # 显示详细示例
    print(f"\n📝 预测示例（详细）：")
    for i, sample in enumerate(debug_samples[:5]):
        trace = sample.get('preprocess_trace') or {}

        def _print_stage(label, key, prev_key=None):
            value = trace.get(key, "")
            suffix = ""
            if prev_key is not None and trace.get(prev_key, "") == value:
                suffix = " (无变化)"
            print(f"{label}")
            print(f"   {value}{suffix}")

        print(f"\n{'─'*60}")
        print(f"【样本 {i+1}】ID: {sample['id']}")
        if trace:
            _print_stage("📝 原始输入：", 'raw_input')
            _print_stage("🔧 编码修复后：", 'after_correction', 'raw_input')
            if config.use_correction_vocab and _correction_vocab is None:
                print("   说明: correction vocab 未加载，当前编码修复阶段实际未启用")
            if trace.get('correction_fixes'):
                print(f"   修复明细: {trace['correction_fixes']}")
            _print_stage("🧼 preprocess_transliteration 后：", 'after_preprocess', 'after_correction')
            if config.pseudo_label_mode:
                _print_stage("🪙 Silver 清洗后：", 'after_silver_clean', 'after_preprocess')
            else:
                print("🪙 Silver 清洗：")
                print("   未启用（仅 pseudo_label_mode 下生效）")
            _print_stage("💡 注入 HINTS / MONTH / Sumerogram 后：", 'after_injections', 'after_silver_clean')
            if trace.get('rag_prefix'):
                print(f"🧠 RAG 前缀：")
                print(f"   {trace['rag_prefix']}")
            _print_stage("📥 最终模型输入：", 'final_model_input', 'after_injections')
        else:
            print(f"📥 预处理输入：")
            print(f"   {sample['input']}")
        print(f"📤 模型原始输出：")
        print(f"   {sample['raw_output']}")
        print(f"✨ 后处理输出：")
        print(f"   {sample['cleaned_output']}")
    print(f"{'─'*60}\n")
    
    # 最终统计
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"📊 推理完成统计")
    print(f"{'='*80}")
    print(f"   总耗时: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"   ├─ 预处理: {preprocess_time:.1f}s")
    print(f"   ├─ 生成:   {gen_time:.1f}s")
    print(f"   └─ 后处理: {total_time - preprocess_time - gen_time:.1f}s")
    print(f"   样本数: {len(df)}")
    print(f"   片段数: {total_segments}")
    print(f"   平均速度: {total_segments/gen_time:.2f} seg/s (生成阶段)")
    print(f"{'='*80}\n")
    sys.stdout.flush()
    
    return result_df


# ============================================================================
# Main
# ============================================================================

def main():
    """主入口：所有行为由 InferenceConfig 类控制，无需命令行参数"""
    # 运行前核弹级清理，杀死所有僵尸残留
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    config = InferenceConfig()
    
    # 检查模型路径是否有效
    if not config.model_paths:
        raise ValueError("未找到模型路径！请在 InferenceConfig.model_paths 中配置。")
    
    # 同步训练配置
    config.load_training_config()
    
    # 打印配置摘要
    print(f"\n📋 推理配置:")
    if hasattr(config, "local_mode"):
        print(f"   local_mode: {getattr(config, 'local_mode', 'standard')}")
    print(f"   pseudo_label_mode: {config.pseudo_label_mode}")
    print(f"   mrt_data_mode: {getattr(config, 'mrt_data_mode', False)}")
    print(f"   model_paths: {config.model_paths}")
    print(f"   decode_strategy: {config.decode_strategy}")
    print(f"   batch_size: {config.batch_size}")
    print(f"   num_beams: {config.num_beams}")
    if config.decode_strategy == "sampling_mbr":
        sampling_temperatures = [float(t) for t in (config.sample_temperatures or [])]
        num_sample_per_temp = int(config.num_sample_per_temp or 0)
        if sampling_temperatures and num_sample_per_temp > 0:
            print(
                f"   mbr_num_candidates: {len(sampling_temperatures) * num_sample_per_temp} "
                f"({num_sample_per_temp}/temp × {sampling_temperatures})"
            )
        else:
            print(f"   mbr_num_candidates: {config.mbr_num_candidates}")
    elif config.decode_strategy == "hybrid_mbr":
        print(f"   hybrid_beam_cands: {config.hybrid_beam_cands}")
        print(
            f"   hybrid_sample_cands: {config.hybrid_sample_cands} "
            f"(temp={config.hybrid_temperature}, top_p={config.hybrid_top_p})"
        )
    elif config.decode_strategy == "multi_model_mbr":
        sample_temperatures, num_sample_per_temp = config.get_multi_model_sampling_plan()
        print(f"   multi_model_beam_cands: {config.multi_model_beam_cands}")
        print(
            f"   multi_model_sample_cands: "
            f"{len(sample_temperatures) * num_sample_per_temp} "
            f"({num_sample_per_temp}/temp × {sample_temperatures})"
        )
        print(f"   use_adaptive_beams: {config.use_adaptive_beams}")
    if config.decode_strategy == "hybrid_mbr":
        print(f"   mbr_metric: {config.mbr_metric}")
        print(f"   mbr_pool_cap: {config.mbr_pool_cap}")
    elif config.decode_strategy in ("sampling_mbr", "multi_model_mbr"):
        print(f"   mbr_metric: {config.mbr_metric}")
        print(
            f"   mbr_weights: chrf={config.mbr_w_chrf}, bleu={config.mbr_w_bleu}, "
            f"jaccard={config.mbr_w_jaccard}, length={config.mbr_w_length}, cap={config.mbr_pool_cap}"
        )
    
    # 统一入口：run_inference 内部根据 decode_strategy 选择推理路径
    submission = run_inference(config)
    
    print(f"\n{'=' * 80}")
    print("推理完成！")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
