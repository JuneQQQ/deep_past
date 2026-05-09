#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Rescue GRPO for Encoder-Decoder ByT5
================================================================================

针对当前翻译 RL 失败模式的修正版：

- Sequence-level PPO ratio / clipping
- Token-level exact KL
- Decayed CE anchor (CHORD-like simplified global weighting)
- Mixed reward: 0.7 * chrF + 0.3 * BLEU
- Under-length / over-length penalty
- Repetition penalty
- Dynamic sampling with soft weighting
- Early stopping only counts on epoch-end eval

适用场景：
- 已经有较强 SFT / merged checkpoint
- 但纯 RL 出现明显欠翻、短前缀拿高分、eval chrF 不升反降
================================================================================
"""

import argparse
import csv
import gc
import glob
import json
import math
import os
import random
import re
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import time as _time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from infer import VectorizedPostprocessor, strip_input_artifacts, decode_sequences

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=-1)
parser.add_argument("--timestamp", type=str, default=None)
parser.add_argument("--train_csv", type=str, default=None)
parser.add_argument("--smoke_test", action="store_true", help="Minimal run to verify pipeline")
parser.add_argument(
    "--resume_from_checkpoint",
    type=str,
    default=None,
    help="Resume source checkpoint dir, or 'auto'/'latest' to pick the newest resumable checkpoint in output_dir",
)
parser.add_argument(
    "--skip_initial_eval",
    action="store_true",
    help="Skip the evaluation that runs before the first resumed/new optimizer step",
)
args, _ = parser.parse_known_args()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
_bootstrap_timestamp = os.environ.get("TRAINING_TIMESTAMP") or args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["TRAINING_TIMESTAMP"] = _bootstrap_timestamp
_bootstrap_output_dir = os.path.join("/data/lsb/deep_past", "output", f"grpo_rescue_{_bootstrap_timestamp}")
os.makedirs(_bootstrap_output_dir, exist_ok=True)
_bootstrap_log_file = os.path.join(_bootstrap_output_dir, "training_log.txt")


class TeeLogger:
    def __init__(self, terminal, log_handle):
        self.terminal = terminal
        self.log_handle = log_handle

    def write(self, message):
        self.terminal.write(message)
        self.log_handle.write(message)
        self.log_handle.flush()

    def flush(self):
        self.terminal.flush()
        self.log_handle.flush()

    def isatty(self):
        return self.terminal.isatty() if hasattr(self.terminal, "isatty") else False


_shared_log_handle = open(_bootstrap_log_file, "a", encoding="utf-8")
sys.stdout = TeeLogger(sys.stdout, _shared_log_handle)
sys.stderr = TeeLogger(sys.stderr, _shared_log_handle)
print(f"Logging to: {_bootstrap_log_file}")

# Enable TF32 for Ampere/Ada/Blackwell GPUs (huge speedup)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

gc.collect()
torch.cuda.empty_cache()

print("=" * 80)
print("RESCUE GRPO FOR BYT5 TRANSLATION")
print("=" * 80)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _checkpoint_step(path: str) -> int:
    name = os.path.basename(path.rstrip("/"))
    try:
        return int(name.split("checkpoint-")[-1])
    except Exception:
        return -1


def _is_resumable_grpo_checkpoint(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    required_files = (
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
        "rng_state.pth",
    )
    return all(os.path.exists(os.path.join(path, name)) for name in required_files)


def _find_latest_resumable_grpo_checkpoint(search_dir: str) -> str | None:
    if not search_dir or not os.path.isdir(search_dir):
        return None
    candidates = [
        path for path in glob.glob(os.path.join(search_dir, "checkpoint-*"))
        if _is_resumable_grpo_checkpoint(path)
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda path: (_checkpoint_step(path), os.path.getmtime(path)))
    return candidates[-1]


def resolve_grpo_resume_checkpoint(output_dir: str, requested: str | None = None) -> str | None:
    if requested and requested.lower() not in {"auto", "latest", "last"}:
        candidate = requested
        if not os.path.isabs(candidate):
            candidate = os.path.abspath(candidate)
        if _is_resumable_grpo_checkpoint(candidate):
            return candidate
        latest = _find_latest_resumable_grpo_checkpoint(candidate)
        if latest:
            return latest
        raise FileNotFoundError(f"Requested GRPO resume checkpoint is not resumable: {requested}")
    return _find_latest_resumable_grpo_checkpoint(output_dir)


class GRPOConfig:
    def __init__(self) -> None:
        local_root = "/data/lsb/deep_past"
        self.root_dir = local_root
        self.data_dir = os.path.join(local_root, "data")
        self.train_csv = args.train_csv or os.path.join(self.data_dir, "qwen_sentence_aligned_clean.csv")

        timestamp = os.environ.get("TRAINING_TIMESTAMP") or args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timestamp = timestamp
        self.output_dir = os.path.join(self.root_dir, "output", f"grpo_rescue_{timestamp}")
        self.resume_from_checkpoint = args.resume_from_checkpoint
        self.run_initial_eval = not args.skip_initial_eval

        # 建议用你目前最强、最稳定的 merged / SFT / DFT checkpoint
        # self.model_name = "/root/projects/deep_past/output/merged_20260304_143126"
        self.model_name = "/data/lsb/deep_past/output/model_20260321_171137/checkpoint-20800"
        self.task_prefix = "translate Old Assyrian to English: "
        self.use_meta_prefix = True  # must match train.py setting

        self.max_input_length = 512
        self.max_target_length = 512

        # ---------------------------
        # Rollout / Optimization
        # ---------------------------
        self.grpo_epochs = 100
        self.grpo_inner_epochs = 2
        self.grpo_batch_size = 4
        self.gradient_accumulation = 1
        self.rollout_logprob_batch_size = 64
        self.ppo_candidate_batch_size = 32
        self.ce_batch_size = 32

        # ---------------------------
        # PPO / KL
        # ---------------------------
        self.clip_eps = 0.2
        self.clip_eps_higher = 0.30
        self.kl_coeff = 0.005
        self.learning_rate = 5e-5
        self.warmup_ratio = 0.01
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        # ---------------------------
        # Decayed CE Anchor
        # ---------------------------
        self.use_ce_anchor = False
        self.ce_alpha_init = 0.1
        self.ce_alpha_final = 0.01
        self.ce_alpha_hold_ratio = 0.1      # 前 10% 保持
        self.ce_alpha_decay_end_ratio = 0.4  # 10%~40% 平滑衰减

        # ---------------------------
        # Rollout generation (notebook-aligned hybrid pool)
        # ---------------------------
        self.rollout_num_beams = 1       # 纯 sample，不用 beam（减少 1 次 generate 调用）
        self.rollout_beam_cands = 0
        self.rollout_sample_cands = 4      # 纯 sample 4 candidates = group_size
        self.temperature = 0.75
        self.top_k = 0
        self.top_p = 0.92
        self.do_sample = True
        self.generation_repetition_penalty = 1.2
        self.no_repeat_ngram_size = 0
        self.rollout_length_penalty = 1.3
        self.rollout_early_stopping = True
        self.rollout_pool_cap = 32

        # ---------------------------
        # Eval
        # ---------------------------
        self.eval_batch_size = 64          # boundary eval (beam=4 generate)
        self.eval_boundary_num_beams = 4   # boundary eval (initial/epoch-end) beam 数

        # ---------------------------
        # Reward
        # ---------------------------
        self.reward_type = "geom_mean"  # chrf / bleu / geom_mean / mixed
        self.reward_baseline = 0.0
        self.reward_mix_chrf = 0.7
        self.reward_mix_bleu = 0.3

        # repetition penalty
        self.repetition_uni_ratio_floor = 0.70
        self.repetition_bigram_uni_ratio_floor = 0.75
        self.repetition_uni_scale = 20.0
        self.repetition_bigram_scale = 15.0

        # length penalties
        self.under_length_target_ratio = 0.90
        self.under_length_scale = 24.0
        self.over_length_target_ratio = 1.35
        self.over_length_scale = 6.0

        # absolute quality threshold
        self.reward_floor = 20.0
        self.poor_group_weight = 0.05

        # ---------------------------
        # Targeted Bonus
        # ---------------------------
        self.bonus_number_scale = 3.0
        self.bonus_entity_scale = 1.5
        # ---------------------------
        # Dynamic sampling
        # ---------------------------
        self.hard_drop_identical_groups = True
        self.hard_drop_low_info_groups = False
        self.min_reward_std = 0.15
        self.info_tau = 2.0
        self.min_group_weight = 0.05

        # ---------------------------
        # Misc
        # ---------------------------
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.gradient_checkpointing = False
        self.eval_every_n_steps = 200
        self.save_total_limit = 2
        self.early_stopping_patience = 10
        self.early_stopping_metric = "eval_loss"  # teacher-forced loss（无 boundary eval 时唯一可用指标）
        self.early_stopping_greater_is_better = False   # loss 越低越好
        self.eval_loss_batch_size = 96  # teacher-forced loss（无 generate，显存低）

        self.test_size = 0.1
        self.random_seed = 42
        self.fold_group_col = "oare_id"

        self.logging_steps = 10
        self.save_eval_predictions = True
        self.max_train_steps = 0

        self.group_size = self.rollout_beam_cands + self.rollout_sample_cands


config = GRPOConfig()

if args.smoke_test:
    config.grpo_epochs = 1
    config.rollout_beam_cands = 2
    config.rollout_sample_cands = 2
    config.group_size = 4
    config.grpo_batch_size = 2
    config.grpo_inner_epochs = 2
    config.gradient_accumulation = 2
    config.eval_every_n_steps = 5
    config.logging_steps = 1
    config.eval_batch_size = 8
    config.max_train_steps = 10
    print("[SMOKE TEST MODE] Using real data, limited steps")

set_seed(config.random_seed)
os.makedirs(config.output_dir, exist_ok=True)


_NOTEBOOK_POSTPROCESSOR = VectorizedPostprocessor()


def normalize_predictions_batch(texts: List[str]) -> List[str]:
    stripped = [strip_input_artifacts(str(t or "")) for t in texts]
    return _NOTEBOOK_POSTPROCESSOR.postprocess_batch(stripped)


def normalize_prediction_text(text: str) -> str:
    return normalize_predictions_batch([text])[0]


def mbr_pick(candidates: List[str], pool_cap: int = 32) -> str:
    import sacrebleu

    seen = set()
    deduped: List[str] = []
    for cand in candidates:
        cand = str(cand).strip()
        if cand and cand not in seen:
            deduped.append(cand)
            seen.add(cand)

    if pool_cap:
        deduped = deduped[:pool_cap]

    if not deduped:
        return ""
    if len(deduped) == 1:
        return deduped[0]

    metric = sacrebleu.metrics.CHRF(word_order=2)
    best_text = deduped[0]
    best_score = -1e9
    for i, cand in enumerate(deduped):
        score = 0.0
        for j, other in enumerate(deduped):
            if i == j:
                continue
            score += float(metric.sentence_score(cand, [other]).score)
        score /= max(1, len(deduped) - 1)
        if score > best_score:
            best_score = score
            best_text = cand
    return best_text


# ------------------------------------------------------------------------------
# Reward helpers
# ------------------------------------------------------------------------------

def compute_chrf(prediction: str, reference: str) -> float:
    import sacrebleu
    score = sacrebleu.sentence_chrf(prediction, [reference], word_order=2)
    return float(score.score)


def compute_bleu(prediction: str, reference: str) -> float:
    import sacrebleu
    score = sacrebleu.sentence_bleu(prediction, [reference])
    return float(score.score)


def repetition_penalty(text: str, cfg: GRPOConfig) -> float:
    toks = text.split()
    if len(toks) < 2:
        return 0.0

    uniq_ratio = len(set(toks)) / max(1, len(toks))
    penalty_uni = max(0.0, cfg.repetition_uni_ratio_floor - uniq_ratio) * cfg.repetition_uni_scale

    bigrams = list(zip(toks[:-1], toks[1:]))
    if len(bigrams) > 0:
        bigram_uniq_ratio = len(set(bigrams)) / len(bigrams)
        penalty_bi = max(
            0.0,
            cfg.repetition_bigram_uni_ratio_floor - bigram_uniq_ratio
        ) * cfg.repetition_bigram_scale
    else:
        penalty_bi = 0.0

    return penalty_uni + penalty_bi


def length_penalty(prediction: str, reference: str, cfg: GRPOConfig) -> float:
    pred_len = max(1, len(prediction.split()))
    ref_len = max(1, len(reference.split()))
    ratio = pred_len / ref_len

    under = max(0.0, cfg.under_length_target_ratio - ratio)
    over = max(0.0, ratio - cfg.over_length_target_ratio)

    return (
        cfg.under_length_scale * (under ** 2)
        + cfg.over_length_scale * (over ** 2)
    )


def extract_numeric_units(text: str) -> List[str]:
    import re

    # 先抓 mixed fraction，如 2 ½ / 1 ⅔
    mixed = re.findall(r'\d+\s*[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]', text)

    # 为避免重复，把 mixed 先替换掉
    text_wo_mixed = text
    for m in mixed:
        text_wo_mixed = text_wo_mixed.replace(m, " ")

    # 再抓普通分数和整数
    simple = re.findall(r'\d+/\d+|\d+|[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞]', text_wo_mixed)

    nums = mixed + simple
    nums = [re.sub(r"\s+", "", x) for x in nums]
    return nums

def compute_number_match_reward(prediction: str, reference: str, cfg: GRPOConfig) -> float:
    from collections import Counter

    ref_nums = extract_numeric_units(reference)
    if not ref_nums:
        return 0.0

    pred_nums = extract_numeric_units(prediction)

    ref_counter = Counter(ref_nums)
    pred_counter = Counter(pred_nums)

    matches = sum((ref_counter & pred_counter).values())
    total_ref = sum(ref_counter.values())

    return cfg.bonus_number_scale * (matches / total_ref)


def compute_entity_match_reward(prediction: str, reference: str, cfg: GRPOConfig) -> float:
    """
    奖励实体词一致性。
    支持带特殊字符（如 š, ā, ṭ）及连字符的阿卡德语/古代近东专有名词，
    并过滤掉常见的英文句首停用词。
    """
    import re
    
    # 匹配大写字母开头的单词，支持连字符和常见阿卡德语转写特殊字符
    # 包含：A-Z, Š, Ṭ, Ṣ, Ḫ, Ā, Ē, Ī, Ū 及其小写，和单引号
    entity_pattern = r"\b[A-ZŠṬṢḪĀĒĪŪ][a-zA-ZšṭṣḫāēīūŠṬṢḪĀĒĪŪ'\-’]*\b"
    
    reference = reference.replace("’", "'")
    prediction = prediction.replace("’", "'")
    
    ref_caps = re.findall(entity_pattern, reference)
    pred_caps = re.findall(entity_pattern, prediction)
    
    stop_words = {
        "The", "A", "An", "To", "From", "He", "She", "It", "They", 
        "In", "On", "At", "When", "If", "But", "And", "Or", "Then", 
        "As", "For", "By", "With", "This", "That", "These", "Those", 
        "I", "We", "You", "Your", "My", "His", "Her", "Their", "Our",
        "Witnessed", "Sealed", "Month", "Year", "Day", "Is", "Are", "Was", "Were"
    }
    
    ref_caps = [w for w in ref_caps if w not in stop_words]
    pred_caps = [w for w in pred_caps if w not in stop_words]
    
    keywords = {
        'shekel', 'shekels', 'mina', 'minas', 'talent', 'talents',
        'son', 'daughter', 'wife', 'brother', 'father', 'mother',
        'witness', 'seal', 'silver', 'gold', 'barley', 'grain', 'oil',
        'copper', 'lead', 'tin', 'bronze', 'slave', 'slaves', 'servant'
    }
    
    ref_lower = reference.lower()
    pred_lower = prediction.lower()
    
    ref_keywords = [w for w in re.findall(r'\b[a-z]+\b', ref_lower) if w in keywords]
    pred_keywords = [w for w in re.findall(r'\b[a-z]+\b', pred_lower) if w in keywords]
    
    ref_entities = [x.lower() for x in (ref_caps + ref_keywords)]
    pred_entities = [x.lower() for x in (pred_caps + pred_keywords)]
    
    if not ref_entities:
        return 0.0
        
    from collections import Counter
    ref_counter = Counter(ref_entities)
    pred_counter = Counter(pred_entities)
    
    matches = sum((ref_counter & pred_counter).values())
    total_ref = sum(ref_counter.values())
    
    return cfg.bonus_entity_scale * (matches / total_ref)


def coverage_penalty(prediction: str, reference: str, cfg: GRPOConfig) -> float:
    pred_len = max(1, len(prediction.split()))
    ref_len = max(1, len(reference.split()))
    ratio = pred_len / ref_len

    if ratio >= 0.8:
        return 0.0
    return (0.8 - ratio) * 50.0


def compute_reward(prediction: str, reference: str, cfg: GRPOConfig) -> float:
    pred = str(prediction).strip()
    ref = reference.strip()

    if not pred:
        return 0.0

    chrf = compute_chrf(pred, ref)
    bleu = compute_bleu(pred, ref)

    if cfg.reward_type == "chrf":
        base_reward = chrf
    elif cfg.reward_type == "bleu":
        base_reward = bleu
    elif cfg.reward_type == "geom_mean":
        base_reward = math.sqrt(max(chrf, 0.01) * max(bleu, 0.01))
    elif cfg.reward_type == "mixed":
        base_reward = cfg.reward_mix_chrf * chrf + cfg.reward_mix_bleu * bleu
    else:
        base_reward = chrf

    # 惩罚
    penalty = (
        repetition_penalty(pred, cfg)
        + length_penalty(pred, ref, cfg)
        + coverage_penalty(pred, ref, cfg)
    )
    
    # 定向奖励
    raw_bonus = compute_number_match_reward(pred, ref, cfg) + compute_entity_match_reward(pred, ref, cfg)
    
    # 限制短句的 bonus 影响（如果参考长度<10，按比例衰减 bonus，避免短句收益过大）
    ref_len = max(1, len(ref.split()))
    length_factor = max(0.5, min(1.0, ref_len / 10.0))
    bonus = raw_bonus * length_factor
    
    reward = base_reward - penalty + bonus - cfg.reward_baseline
    return float(reward)


def compute_group_advantages(rewards: List[float], eps: float = 1e-8) -> List[float]:
    arr = np.array(rewards, dtype=np.float64)
    mean = arr.mean()
    std = arr.std()
    if std < eps:
        return [0.0] * len(rewards)
    advs = (arr - mean) / (std + eps)
    advs = np.clip(advs, -3.0, 3.0)
    return advs.tolist()


# ------------------------------------------------------------------------------
# Dataset
# ------------------------------------------------------------------------------

# --- Meta prefix helpers (mirrored from train.py) ---
_DIALECT_CODE_MAP = {"OA": "OA", "OB": "OB", "OAkk": "XX", "Sumerian": "XX"}
_NOISY_DATA_SOURCES = {"qwen", "silver", "ocr", "pseudo", "synthetic", "noisy"}

def _normalize_data_source(value) -> str:
    text = str(value).strip().lower()
    return "ocr" if text in _NOISY_DATA_SOURCES else "official"

def _is_damaged(text: str) -> bool:
    if '<gap>' in text or '[' in text or ']' in text:
        return True
    if re.search(r'\bx\b', text, re.IGNORECASE):
        return True
    if '\u2E22' in text or '\u2E23' in text or '...' in text:
        return True
    return False

def _build_meta_prefix(text: str, data_source: str = "official", dialect: str = "OA") -> str:
    dial = _DIALECT_CODE_MAP.get(str(dialect).strip(), "OA")
    src = "O" if _normalize_data_source(data_source) == "official" else "N"
    comp = "D" if _is_damaged(text) else "I"
    return f"{dial}{src}{comp} "


class TranslationDataset(Dataset):
    def __init__(self, df: pd.DataFrame, task_prefix: str, use_meta_prefix: bool = False) -> None:
        self.prompts = []
        self.references = []
        self.oare_ids = []

        for _, row in df.iterrows():
            tl = str(row.get("transliteration", row.get("input_text", "")))
            tr = str(row.get("translation", row.get("target_text", "")))
            oid = str(row.get("oare_id", ""))

            if tl.strip() and tr.strip():
                if use_meta_prefix:
                    ds = str(row.get("data_source", "official"))
                    dialect = str(row.get("dialect", "OA"))
                    prefix = _build_meta_prefix(tl.strip(), ds, dialect)
                else:
                    prefix = task_prefix
                self.prompts.append(prefix + tl.strip())
                self.references.append(tr.strip())
                self.oare_ids.append(oid)

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return {
            "prompt": self.prompts[idx],
            "reference": self.references[idx],
            "oare_id": self.oare_ids[idx],
        }


# ------------------------------------------------------------------------------
# Trainer
# ------------------------------------------------------------------------------

class GRPOTrainer:
    def __init__(self, config: GRPOConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_file = os.path.join(config.output_dir, "training_log.txt")
        self.train_history_path = os.path.join(config.output_dir, "train_history.csv")
        self.eval_history_path = os.path.join(config.output_dir, "eval_history.csv")
        self.plots_dir = os.path.join(config.output_dir, "plots")

        self.global_step = 0
        self.total_steps_nominal = 1
        self.best_metric = -float("inf") if config.early_stopping_greater_is_better else float("inf")
        self.patience_counter = 0
        self.train_history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []
        self.resume_checkpoint_path = resolve_grpo_resume_checkpoint(
            config.output_dir,
            config.resume_from_checkpoint,
        )
        self.resume_state: Dict[str, Any] = {}
        self.start_epoch = 0

        if self.resume_checkpoint_path:
            print(f"\n[Resume] Auto-detected resumable checkpoint: {self.resume_checkpoint_path}")
            self._load_existing_histories()
        else:
            print("\n[Resume] No resumable checkpoint detected. Starting fresh run.")

        self._load_models()
        self._load_data()

    # -----------------------
    # setup
    # -----------------------
    def _load_existing_histories(self) -> None:
        if os.path.exists(self.train_history_path):
            try:
                train_df = pd.read_csv(self.train_history_path)
                self.train_history = train_df.to_dict("records")
            except Exception as exc:
                print(f"   [Resume] Failed to load train history: {exc}")
        if os.path.exists(self.eval_history_path):
            try:
                eval_df = pd.read_csv(self.eval_history_path)
                self.eval_history = eval_df.to_dict("records")
            except Exception as exc:
                print(f"   [Resume] Failed to load eval history: {exc}")

    def _load_models(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        cfg = self.config
        base_model_id = cfg.model_name
        policy_model_id = self.resume_checkpoint_path if self.resume_checkpoint_path else base_model_id
        tokenizer_id = policy_model_id if os.path.exists(os.path.join(policy_model_id, "tokenizer_config.json")) else base_model_id
        policy_local = os.path.isdir(policy_model_id)
        ref_local = os.path.isdir(base_model_id)
        load_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32

        print(f"\n[Model] Loading policy from {policy_model_id}")
        print(f"   Reference model: {base_model_id}")
        print(f"   dtype: {load_dtype}, device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=os.path.isdir(tokenizer_id))
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForSeq2SeqLM.from_pretrained(
            policy_model_id,
            torch_dtype=load_dtype,
            local_files_only=policy_local,
            low_cpu_mem_usage=True,
            dropout_rate=0.0,
        ).to(self.device)

        self.ref_model = AutoModelForSeq2SeqLM.from_pretrained(
            base_model_id,
            torch_dtype=load_dtype,
            local_files_only=ref_local,
            low_cpu_mem_usage=True,
            dropout_rate=0.0,
        ).to(self.device)

        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        if cfg.gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()
            self.policy_model.config.use_cache = False

        total = sum(p.numel() for p in self.policy_model.parameters())
        trainable = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        print(f"   Params: {total:,} total, {trainable:,} trainable")

    def _load_data(self) -> None:
        from sklearn.model_selection import GroupShuffleSplit

        cfg = self.config
        print(f"\n[Data] Loading from {cfg.train_csv}")
        df = pd.read_csv(cfg.train_csv)
        print(f"   Total samples: {len(df)}")

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
        )
        groups = df[cfg.fold_group_col] if cfg.fold_group_col in df.columns else None
        train_idx, eval_idx = next(gss.split(df, groups=groups))

        train_df = df.iloc[train_idx].reset_index(drop=True)
        eval_df = df.iloc[eval_idx].reset_index(drop=True)

        if args.smoke_test:
            train_df = train_df.head(6)
            eval_df = eval_df.head(4)

        self.train_dataset = TranslationDataset(train_df, cfg.task_prefix, cfg.use_meta_prefix)
        self.eval_dataset = TranslationDataset(eval_df, cfg.task_prefix, cfg.use_meta_prefix)
        print(f"   Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")

    # -----------------------
    # helper functions
    # -----------------------
    def _current_ce_alpha(self) -> float:
        cfg = self.config
        if not cfg.use_ce_anchor:
            return 0.0
        progress = min(1.0, self.global_step / max(1, self.total_steps_nominal - 1))
        hold_ratio = min(max(cfg.ce_alpha_hold_ratio, 0.0), 1.0)
        decay_end_ratio = min(max(cfg.ce_alpha_decay_end_ratio, hold_ratio), 1.0)

        if progress <= hold_ratio:
            return cfg.ce_alpha_init
        if progress >= decay_end_ratio:
            return cfg.ce_alpha_final

        decay_progress = (progress - hold_ratio) / max(decay_end_ratio - hold_ratio, 1e-8)
        return cfg.ce_alpha_init + decay_progress * (cfg.ce_alpha_final - cfg.ce_alpha_init)

    def _append_csv_row(self, path: str, row: Dict[str, float]) -> None:
        file_exists = os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(row)
            f.flush()

    def _record_train_history(self, row: Dict[str, float]) -> None:
        self.train_history.append(row)
        self._append_csv_row(self.train_history_path, row)

    def _record_eval_history(self, row: Dict[str, float]) -> None:
        self.eval_history.append(row)
        self._append_csv_row(self.eval_history_path, row)

    def _capture_rng_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            state["cuda"] = torch.cuda.get_rng_state_all()
        return state

    def _restore_rng_state(self, checkpoint_dir: str) -> None:
        rng_path = os.path.join(checkpoint_dir, "rng_state.pth")
        if not os.path.exists(rng_path):
            return
        state = torch.load(rng_path, map_location="cpu")
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if torch.cuda.is_available() and "cuda" in state:
            torch.cuda.set_rng_state_all(state["cuda"])

    def _save_resume_checkpoint(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> str:
        cfg = self.config
        checkpoint_dir = os.path.join(cfg.output_dir, f"checkpoint-{self.global_step}")

        os.makedirs(checkpoint_dir, exist_ok=True)

        self.policy_model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
        torch.save(scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
        torch.save(self._capture_rng_state(), os.path.join(checkpoint_dir, "rng_state.pth"))

        trainer_state = {
            "global_step": int(self.global_step),
            "best_metric": float(self.best_metric),
            "patience_counter": int(self.patience_counter),
            "completed_epochs": int(epoch + 1),
            "next_epoch": int(epoch + 1),
            "total_steps_nominal": int(self.total_steps_nominal),
        }
        with open(os.path.join(checkpoint_dir, "trainer_state.json"), "w", encoding="utf-8") as f:
            json.dump(trainer_state, f, indent=2)

        with open(os.path.join(checkpoint_dir, "grpo_config.json"), "w", encoding="utf-8") as f:
            json.dump(
                {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
                f,
                indent=2,
                default=str,
            )

        checkpoint_dirs = sorted(
            [
                path for path in glob.glob(os.path.join(cfg.output_dir, "checkpoint-*"))
                if _is_resumable_grpo_checkpoint(path)
            ],
            key=lambda path: (_checkpoint_step(path), os.path.getmtime(path)),
        )
        while len(checkpoint_dirs) > cfg.save_total_limit:
            old = checkpoint_dirs.pop(0)
            shutil.rmtree(old, ignore_errors=True)
            print(f"   Removed old resume checkpoint: {old}")

        return checkpoint_dir

    def _restore_training_state(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
    ) -> None:
        if not self.resume_checkpoint_path:
            return

        state_path = os.path.join(self.resume_checkpoint_path, "trainer_state.json")
        optimizer_path = os.path.join(self.resume_checkpoint_path, "optimizer.pt")
        scheduler_path = os.path.join(self.resume_checkpoint_path, "scheduler.pt")

        with open(state_path, "r", encoding="utf-8") as f:
            self.resume_state = json.load(f)

        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
        self._restore_rng_state(self.resume_checkpoint_path)

        self.global_step = int(self.resume_state.get("global_step", 0))
        self.best_metric = float(self.resume_state.get("best_metric", self.best_metric))
        self.patience_counter = int(self.resume_state.get("patience_counter", 0))
        self.start_epoch = int(self.resume_state.get("next_epoch", 0))
        self.total_steps_nominal = int(self.resume_state.get("total_steps_nominal", self.total_steps_nominal))

        print(
            f"[Resume] Restored optimizer/scheduler/state from {self.resume_checkpoint_path} "
            f"(step={self.global_step}, next_epoch={self.start_epoch})"
        )

    def _render_plots(self) -> None:
        import matplotlib.pyplot as plt

        os.makedirs(self.plots_dir, exist_ok=True)

        def _finalize_axis(ax, title: str, xlabel: str = "Step", add_unity: bool = False) -> None:
            if add_unity:
                ax.axhline(1.0, color="r", linestyle="--", alpha=0.5)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.grid(True, alpha=0.3)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend()

        if self.train_history:
            train_df = pd.DataFrame(self.train_history)

            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            ax = axes[0, 0]
            for col, label in [("rl_loss", "RL"), ("ce_loss", "CE"), ("kl_div", "KL")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            _finalize_axis(ax, "Train Loss Components")

            ax = axes[0, 1]
            for col, label in [("mean_reward", "Reward"), ("mean_len_ratio", "Len Ratio")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            _finalize_axis(ax, "Reward and Length Ratio", add_unity=True)

            ax = axes[1, 0]
            for col, label in [("keep_ratio", "Keep Ratio"), ("mean_keep_weight", "Keep Weight"), ("duplicate_group_ratio", "Dup Ratio")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            _finalize_axis(ax, "Dynamic Sampling")

            ax = axes[1, 1]
            for col, label in [("grad_norm", "Grad Norm"), ("lr", "LR")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            _finalize_axis(ax, "Optimization")

            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir, "train_metrics.png"), dpi=150)
            plt.close(fig)

        if self.eval_history:
            eval_df = pd.DataFrame(self.eval_history)

            fig, axes = plt.subplots(3, 1, figsize=(12, 13))
            ax = axes[0]
            for col, label in [("bleu", "BLEU"), ("chrf", "chrF++"), ("geom_mean", "Geom Mean")]:
                if col in eval_df:
                    ax.plot(eval_df["step"], eval_df[col], marker="o", label=label)
            _finalize_axis(ax, "Eval Metrics (generate)")

            ax = axes[1]
            ax2 = ax.twinx()
            if "eval_loss" in eval_df:
                ax.plot(eval_df["step"], eval_df["eval_loss"], marker="s", color="red", label="eval_loss")
            if "eval_token_acc" in eval_df:
                ax2.plot(eval_df["step"], eval_df["eval_token_acc"], marker="^", color="blue", label="token_acc")
                ax2.set_ylabel("token_acc")
            _finalize_axis(ax, "Eval Loss & Token Accuracy")
            handles1, labels1 = ax.get_legend_handles_labels()
            handles2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(handles1 + handles2, labels1 + labels2)

            ax = axes[2]
            for col, label in [("mean_len_ratio", "Len Ratio"), ("mean_pred_len", "Pred Len"), ("mean_ref_len", "Ref Len")]:
                if col in eval_df:
                    ax.plot(eval_df["step"], eval_df[col], marker="o", label=label)
            _finalize_axis(ax, "Eval Length Diagnostics", add_unity=True)

            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir, "eval_metrics.png"), dpi=150)
            plt.close(fig)

    def _train_pbar_desc(self, epoch: int) -> str:
        total_steps = max(1, self.total_steps_nominal)
        return f"Train e{epoch + 1}/{self.config.grpo_epochs} u{self.global_step}/{total_steps}"

    def _train_pbar_postfix(self, epoch_stats: Dict[str, List[float]]) -> str:
        if not epoch_stats.get("rl_loss"):
            return ""
        return (
            f"rl={np.mean(epoch_stats['rl_loss'][-5:]):.1e} "
            f"ce/rl={np.mean(epoch_stats['ce_to_rl_abs_ratio'][-5:]):.1e} "
            f"R={np.mean(epoch_stats['mean_reward'][-5:]):.1f} "
            f"keep={np.mean(epoch_stats['keep_ratio'][-5:]):.2f} "
            f"len={np.mean(epoch_stats['mean_len_ratio'][-5:]):.2f}"
        )

    def _prepare_prompt_and_labels(
        self,
        prompt: str,
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self._prepare_batch_prompt_and_labels([prompt] * len(target_texts), target_texts)

    def _prepare_batch_prompt_and_labels(
        self,
        prompts: List[str],
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if len(prompts) != len(target_texts):
            raise ValueError(f"prompts={len(prompts)} != target_texts={len(target_texts)}")

        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=cfg.max_input_length,
            return_tensors="pt",
        ).to(self.device)

        labels_batch = self.tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=cfg.max_target_length,
            return_tensors="pt",
            add_special_tokens=True,
        ).to(self.device)

        label_ids = labels_batch["input_ids"]
        label_ids = label_ids.masked_fill(label_ids == self.tokenizer.pad_token_id, -100)

        return enc["input_ids"], enc["attention_mask"], label_ids

    def _generate_raw_candidate_groups(
        self,
        prompts: List[str],
        beam_cands: int,
        sample_cands: int,
        num_beams: int,
        length_penalty: float,
        repetition_penalty: float,
        top_p: float,
        temperature: float,
        early_stopping: bool,
    ) -> List[List[str]]:
        cfg = self.config

        enc = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=cfg.max_input_length,
            return_tensors="pt",
        ).to(self.device)

        batch_size = len(prompts)
        raw_groups: List[List[str]] = [[] for _ in range(batch_size)]
        self.policy_model.eval()
        # Suppress max_length conflict warning
        gen_model = self.policy_model
        if hasattr(gen_model.config, 'max_length'):
            gen_model.config.max_length = cfg.max_target_length + cfg.max_input_length
        with torch.inference_mode():
            if beam_cands > 0:
                beam_kwargs = {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "max_new_tokens": cfg.max_target_length,
                    "do_sample": False,
                    "num_beams": max(num_beams, beam_cands),
                    "num_return_sequences": beam_cands,
                    "repetition_penalty": repetition_penalty,
                    "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
                    "use_cache": True,
                }
                if length_penalty is not None:
                    beam_kwargs["length_penalty"] = length_penalty
                if early_stopping is not None:
                    beam_kwargs["early_stopping"] = early_stopping
                beam_outputs = self.policy_model.generate(**beam_kwargs)
                beam_decoded = decode_sequences(beam_outputs, tokenizer=self.tokenizer)
                for b in range(batch_size):
                    raw_groups[b].extend(beam_decoded[b * beam_cands:(b + 1) * beam_cands])

            if sample_cands > 0:
                sample_kwargs = {
                    "input_ids": enc["input_ids"],
                    "attention_mask": enc["attention_mask"],
                    "max_new_tokens": cfg.max_target_length,
                    "do_sample": True,
                    "top_p": top_p,
                    "temperature": temperature,
                    "num_return_sequences": sample_cands,
                    "num_beams": 1,
                    "repetition_penalty": repetition_penalty,
                    "no_repeat_ngram_size": cfg.no_repeat_ngram_size,
                    "use_cache": True,
                }
                if cfg.top_k and cfg.top_k > 0:
                    sample_kwargs["top_k"] = cfg.top_k
                sample_outputs = self.policy_model.generate(**sample_kwargs)
                sample_decoded = decode_sequences(sample_outputs, tokenizer=self.tokenizer)
                for b in range(batch_size):
                    raw_groups[b].extend(sample_decoded[b * sample_cands:(b + 1) * sample_cands])

        return raw_groups

    def _postprocess_candidate_groups(
        self,
        raw_groups: List[List[str]],
    ) -> List[List[str]]:
        flat_raw = [text for group in raw_groups for text in group]
        flat_clean = normalize_predictions_batch(flat_raw) if flat_raw else []

        clean_groups: List[List[str]] = []
        offset = 0
        for group in raw_groups:
            group_size = len(group)
            clean_groups.append(flat_clean[offset:offset + group_size])
            offset += group_size

        return clean_groups

    def _generate_candidate_groups(
        self,
        prompts: List[str],
        beam_cands: int,
        sample_cands: int,
        num_beams: int,
        length_penalty: float,
        repetition_penalty: float,
        top_p: float,
        temperature: float,
        early_stopping: bool,
    ) -> Tuple[List[List[str]], List[List[str]]]:
        raw_groups = self._generate_raw_candidate_groups(
            prompts=prompts,
            beam_cands=beam_cands,
            sample_cands=sample_cands,
            num_beams=num_beams,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_p=top_p,
            temperature=temperature,
            early_stopping=early_stopping,
        )
        clean_groups = self._postprocess_candidate_groups(raw_groups)
        return raw_groups, clean_groups

    def _finalize_eval_hybrid_batch(
        self,
        raw_groups: List[List[str]],
        refs: List[str],
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        cfg = self.config
        clean_groups = self._postprocess_candidate_groups(raw_groups)
        post_decoded = [mbr_pick(group, pool_cap=cfg.eval_pool_cap) for group in clean_groups]
        pred_lens = [len(x.split()) for x in post_decoded]
        ref_lens = [len(x.split()) for x in refs]
        return post_decoded, refs, pred_lens, ref_lens

    def _finalize_eval_beam_batch(
        self,
        decoded: List[str],
        refs: List[str],
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        post_decoded = normalize_predictions_batch(decoded)
        pred_lens = [len(x.split()) for x in post_decoded]
        ref_lens = [len(x.split()) for x in refs]
        return post_decoded, refs, pred_lens, ref_lens

    def _sample_group(self, prompts: List[str], K: int) -> Tuple[List[List[str]], List[List[str]]]:
        cfg = self.config
        expected = cfg.rollout_beam_cands + cfg.rollout_sample_cands
        if K != expected:
            raise ValueError(f"group_size={K} but rollout candidates={expected}")
        return self._generate_candidate_groups(
            prompts=prompts,
            beam_cands=cfg.rollout_beam_cands,
            sample_cands=cfg.rollout_sample_cands,
            num_beams=cfg.rollout_num_beams,
            length_penalty=cfg.rollout_length_penalty,
            repetition_penalty=cfg.generation_repetition_penalty,
            top_p=cfg.top_p,
            temperature=cfg.temperature,
            early_stopping=cfg.rollout_early_stopping,
        )

    def _forward_token_stats(
        self,
        model: Any,
        prompt: str,
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
          logits: [B, T, V]
          token_log_probs: [B, T]  only selected labels
          valid_mask: [B, T]
        """
        return self._forward_token_stats_batch(model, [prompt] * len(target_texts), target_texts)

    def _forward_token_stats_batch(
        self,
        model: Any,
        prompts: List[str],
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_ids, enc_mask, labels = self._prepare_batch_prompt_and_labels(prompts, target_texts)

        outputs = model(
            input_ids=enc_ids,
            attention_mask=enc_mask,
            labels=labels,
        )
        logits = outputs.logits.float()
        valid_mask = (labels != -100).float()
        labels_for_gather = labels.masked_fill(labels == -100, 0)

        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels_for_gather.unsqueeze(-1)).squeeze(-1)
        token_log_probs = token_log_probs * valid_mask
        return logits, token_log_probs, valid_mask

    def _reduce_seq_mean_log_prob(
        self,
        token_log_probs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        seq_len = valid_mask.sum(dim=1).clamp(min=1.0)
        return token_log_probs.sum(dim=1) / seq_len

    def _compute_seq_nll(
        self,
        token_log_probs: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        return -self._reduce_seq_mean_log_prob(token_log_probs, valid_mask)

    def _compute_seq_mean_log_probs_batch(
        self,
        model: Any,
        prompts: List[str],
        target_texts: List[str],
        batch_size: int,
    ) -> torch.Tensor:
        if not prompts:
            return torch.empty(0, device=self.device)

        seq_logps_parts: List[torch.Tensor] = []
        for start in range(0, len(prompts), max(1, batch_size)):
            end = start + max(1, batch_size)
            _, token_logps, valid_mask = self._forward_token_stats_batch(
                model,
                prompts[start:end],
                target_texts[start:end],
            )
            seq_logps_parts.append(self._reduce_seq_mean_log_prob(token_logps, valid_mask))
        return torch.cat(seq_logps_parts, dim=0)

    def _compute_old_seq_log_probs_batch(
        self,
        group_records: List[Dict[str, Any]],
    ) -> List[List[float]]:
        cfg = self.config
        flat_prompts: List[str] = []
        flat_targets: List[str] = []
        group_sizes: List[int] = []

        for rec in group_records:
            size = len(rec["raw_texts"])
            group_sizes.append(size)
            flat_prompts.extend([rec["prompt"]] * size)
            flat_targets.extend(rec["raw_texts"])

        was_training = self.policy_model.training
        self.policy_model.eval()
        with torch.inference_mode():
            flat_seq_logps = self._compute_seq_mean_log_probs_batch(
                self.policy_model,
                flat_prompts,
                flat_targets,
                cfg.rollout_logprob_batch_size,
            )
        if was_training:
            self.policy_model.train()

        seq_logps_cpu = flat_seq_logps.detach().float().cpu().tolist()
        out: List[List[float]] = []
        offset = 0
        for size in group_sizes:
            out.append(seq_logps_cpu[offset:offset + size])
            offset += size
        return out

    def _compute_candidate_rl_inputs_batch(
        self,
        prompts: List[str],
        sampled_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
        if not prompts:
            empty = torch.empty(0, device=self.device)
            return empty, empty, empty

        seq_logps_parts: List[torch.Tensor] = []
        seq_kl_sum_parts: List[torch.Tensor] = []
        seq_token_count_parts: List[torch.Tensor] = []

        self.policy_model.train()
        self.ref_model.eval()

        batch_size = max(1, cfg.ppo_candidate_batch_size)
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            policy_logits, policy_token_logps, valid_mask = self._forward_token_stats_batch(
                self.policy_model,
                prompts[start:end],
                sampled_texts[start:end],
            )
            with torch.inference_mode():
                ref_logits, _, _ = self._forward_token_stats_batch(
                    self.ref_model,
                    prompts[start:end],
                    sampled_texts[start:end],
                )

            policy_log_probs_full = F.log_softmax(policy_logits, dim=-1)
            ref_log_probs_full = F.log_softmax(ref_logits.float(), dim=-1)
            policy_probs_full = policy_log_probs_full.exp()

            token_kl = (policy_probs_full * (policy_log_probs_full - ref_log_probs_full)).sum(dim=-1)
            seq_logps_parts.append(self._reduce_seq_mean_log_prob(policy_token_logps, valid_mask))
            seq_kl_sum_parts.append((token_kl * valid_mask).sum(dim=1))
            seq_token_count_parts.append(valid_mask.sum(dim=1).clamp(min=1.0))

        return (
            torch.cat(seq_logps_parts, dim=0),
            torch.cat(seq_kl_sum_parts, dim=0),
            torch.cat(seq_token_count_parts, dim=0),
        )

    def _compute_ce_losses_batch(
        self,
        prompts: List[str],
        references: List[str],
    ) -> torch.Tensor:
        cfg = self.config
        if not prompts:
            return torch.empty(0, device=self.device)

        ce_parts: List[torch.Tensor] = []
        batch_size = max(1, cfg.ce_batch_size)
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            _, token_logps, valid_mask = self._forward_token_stats_batch(
                self.policy_model,
                prompts[start:end],
                references[start:end],
            )
            ce_parts.append(self._compute_seq_nll(token_logps, valid_mask))
        return torch.cat(ce_parts, dim=0)

    def _compute_loss_for_groups(
        self,
        group_records: List[Dict[str, Any]],
        all_old_seq_log_probs: List[List[float]],
    ) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        cfg = self.config
        if not group_records:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {}

        ce_alpha = self._current_ce_alpha()
        group_prompts = [rec["prompt"] for rec in group_records]
        group_refs = [rec["reference"] for rec in group_records]
        group_sizes = [len(rec["raw_texts"]) for rec in group_records]

        flat_prompts: List[str] = []
        flat_targets: List[str] = []
        flat_advantages: List[float] = []
        flat_old_logps: List[float] = []

        for rec, old_seq_log_probs in zip(group_records, all_old_seq_log_probs):
            size = len(rec["raw_texts"])
            if size != len(old_seq_log_probs):
                raise ValueError(f"group candidate mismatch: {size} vs {len(old_seq_log_probs)}")
            flat_prompts.extend([rec["prompt"]] * size)
            flat_targets.extend(rec["raw_texts"])
            flat_advantages.extend(rec["advantages"])
            flat_old_logps.extend(old_seq_log_probs)

        policy_seq_logps, seq_kl_sums, seq_token_counts = self._compute_candidate_rl_inputs_batch(
            flat_prompts,
            flat_targets,
        )

        old_lp_tensor = torch.tensor(flat_old_logps, dtype=policy_seq_logps.dtype, device=self.device)
        adv_tensor = torch.tensor(flat_advantages, dtype=policy_seq_logps.dtype, device=self.device)

        if ce_alpha > 0.0:
            ce_losses = self._compute_ce_losses_batch(group_prompts, group_refs)
        else:
            ce_losses = torch.zeros(len(group_records), device=self.device, dtype=policy_seq_logps.dtype)

        weighted_rl_sum = torch.tensor(0.0, device=self.device, dtype=policy_seq_logps.dtype)
        ce_sum = torch.tensor(0.0, device=self.device, dtype=policy_seq_logps.dtype)
        weight_sum = 0.0
        batch_stats_agg: Dict[str, List[float]] = {}

        offset = 0
        for gi, rec in enumerate(group_records):
            size = group_sizes[gi]
            sl = slice(offset, offset + size)
            offset += size

            group_policy_seq_logps = policy_seq_logps[sl]
            group_old_lp = old_lp_tensor[sl]
            group_adv = adv_tensor[sl]
            group_kl_sums = seq_kl_sums[sl]
            group_token_counts = seq_token_counts[sl]
            group_weight = rec["keep_weight"]

            log_ratio = group_policy_seq_logps - group_old_lp
            ratio = torch.exp(log_ratio.clamp(-10, 10))

            ratio_clipped = ratio.clone()
            pos_mask = group_adv >= 0
            neg_mask = ~pos_mask

            if pos_mask.any():
                ratio_clipped[pos_mask] = torch.clamp(
                    ratio[pos_mask],
                    1.0 - cfg.clip_eps,
                    1.0 + cfg.clip_eps_higher,
                )
            if neg_mask.any():
                ratio_clipped[neg_mask] = torch.clamp(
                    ratio[neg_mask],
                    1.0 - cfg.clip_eps,
                    1.0 + cfg.clip_eps,
                )

            surr1 = ratio * group_adv
            surr2 = ratio_clipped * group_adv
            policy_loss = -torch.min(surr1, surr2).mean()
            kl_div = group_kl_sums.sum() / group_token_counts.sum().clamp(min=1.0)
            rl_loss = policy_loss + cfg.kl_coeff * kl_div
            ce_loss = ce_losses[gi]

            rl_term = group_weight * rl_loss
            ce_term = group_weight * ce_alpha * ce_loss
            total_loss = rl_term + ce_term
            approx_rl_term = abs(float(rl_term.item()))
            approx_ce_term = abs(float(ce_term.item()))

            weighted_rl_sum = weighted_rl_sum + rl_term
            ce_sum = ce_sum + ce_term
            weight_sum += group_weight

            stats = {
                "policy_loss": float(policy_loss.item()),
                "kl_div": float(kl_div.item()),
                "rl_loss": float(rl_loss.item()),
                "ce_loss": float(ce_loss.item()),
                "ce_alpha": float(ce_alpha),
                "total_loss": float(total_loss.item()),
                "weighted_total_loss": float(total_loss.item()),
                "approx_rl_term": approx_rl_term,
                "approx_ce_term": approx_ce_term,
                "ce_to_rl_abs_ratio": approx_ce_term / max(approx_rl_term, 1e-12),
                "mean_ratio": float(ratio.mean().item()),
                "mean_abs_ratio_delta": float((ratio - 1.0).abs().mean().item()),
                "mean_advantage": float(group_adv.mean().item()),
                "mean_abs_advantage": float(group_adv.abs().mean().item()),
                "group_weight": float(group_weight),
            }
            for key, val in stats.items():
                batch_stats_agg.setdefault(key, []).append(val)

        batch_loss = (weighted_rl_sum + ce_sum) / max(weight_sum, 1e-8)
        return batch_loss, batch_stats_agg

    # -----------------------
    # train
    # -----------------------
    def train_epoch(
        self,
        epoch: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_pbar: tqdm,
    ) -> Dict[str, float]:
        cfg = self.config
        K = cfg.group_size
        dataset = self.train_dataset

        indices = list(range(len(dataset)))
        np.random.shuffle(indices)

        epoch_stats: Dict[str, List[float]] = {
            "total_loss": [],
            "weighted_total_loss": [],
            "policy_loss": [],
            "rl_loss": [],
            "ce_loss": [],
            "ce_alpha": [],
            "kl_div": [],
            "approx_rl_term": [],
            "approx_ce_term": [],
            "ce_to_rl_abs_ratio": [],
            "mean_reward": [],
            "reward_std": [],
            "keep_ratio": [],
            "mean_keep_weight": [],
            "mean_pred_len": [],
            "mean_len_ratio": [],
            "duplicate_group_ratio": [],
            "mean_abs_ratio_delta": [],
            "mean_abs_advantage": [],
            "grad_norm": [],
            "mean_ratio": [],
            "mean_advantage": [],
            "group_weight": [],
        }

        batch_prompts: List[str] = []
        batch_refs: List[str] = []
        accumulated = 0

        optimizer.zero_grad()
        train_pbar.set_description_str(self._train_pbar_desc(epoch))

        for i, idx in enumerate(indices):
            item = dataset[idx]
            batch_prompts.append(item["prompt"])
            batch_refs.append(item["reference"])

            if len(batch_prompts) < cfg.grpo_batch_size and i < len(indices) - 1:
                continue

            raw_group_texts, clean_group_texts = self._sample_group(batch_prompts, K)

            group_records = []
            duplicate_groups = 0

            should_log_rollout = (self.global_step % cfg.logging_steps == 0 and accumulated == 0)

            _t0 = _time.perf_counter()
            # 并行计算 reward 提升 CPU 阶段速度
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                for b in range(len(batch_prompts)):
                    for k in range(K):
                        futures.append(
                            executor.submit(compute_reward, clean_group_texts[b][k], batch_refs[b], cfg)
                        )
                all_rewards = [f.result() for f in futures]
            
            idx = 0
            for b in range(len(batch_prompts)):
                raw_texts = raw_group_texts[b]
                clean_texts = clean_group_texts[b]

                rewards = all_rewards[idx:idx + K]
                idx += K
                advantages = compute_group_advantages(rewards)

                reward_std = float(np.std(rewards))
                uniq_ratio = len(set(clean_texts)) / max(1, len(clean_texts))
                pred_lens = [len(t.split()) for t in clean_texts]
                mean_pred_len = float(np.mean(pred_lens)) if pred_lens else 0.0
                ref_len = max(1, len(batch_refs[b].split()))
                mean_len_ratio = mean_pred_len / ref_len

                identical_group = len(set(clean_texts)) == 1
                low_info_group = reward_std < cfg.min_reward_std
                poor_quality_group = bool(max(rewards) < cfg.reward_floor)

                if identical_group:
                    duplicate_groups += 1

                if cfg.hard_drop_identical_groups and identical_group:
                    if should_log_rollout and b == 0:
                        print(f"\n⚠️ [Dynamic Sampling] Drop identical group at step={self.global_step}")
                        print(f"Prompt: {batch_prompts[b]}")
                        print(f"Reference: {batch_refs[b]}")
                        print(f"Clean texts: {clean_texts}")
                    continue

                if cfg.hard_drop_low_info_groups and low_info_group:
                    continue

                raw_weight = (reward_std / max(cfg.info_tau, 1e-8)) * max(uniq_ratio, 1e-3)
                keep_weight = float(np.clip(raw_weight, cfg.min_group_weight, 1.0))
                
                if poor_quality_group:
                    keep_weight = cfg.poor_group_weight

                group_records.append(
                    {
                        "prompt": batch_prompts[b],
                        "reference": batch_refs[b],
                        "raw_texts": raw_texts,
                        "clean_texts": clean_texts,
                        "rewards": rewards,
                        "advantages": advantages,
                        "reward_std": reward_std,
                        "uniq_ratio": uniq_ratio,
                        "keep_weight": keep_weight,
                        "mean_pred_len": mean_pred_len,
                        "mean_len_ratio": mean_len_ratio,
                    }
                )

                if should_log_rollout and len(group_records) == 1:
                    print(f"\n{'=' * 100}")
                    print(f"🔍 [Rollout Detail] Step={self.global_step} | Epoch={epoch + 1}")
                    print(f"Prompt:    {batch_prompts[b]}")
                    print(f"Reference: {batch_refs[b]}")
                    print(f"reward_std={reward_std:.4f} | uniq_ratio={uniq_ratio:.4f} | keep_weight={keep_weight:.4f}")
                    print(f"poor_quality={poor_quality_group} | low_info={low_info_group}")
                    print(f"{'-' * 100}")
                    for k in range(K):
                        print(f"[Rollout {k + 1}/{K}]")
                        print(f"Raw:    {raw_texts[k]}")
                        print(f"Clean:  {clean_texts[k]}")
                        print(f"Reward: {rewards[k]:.4f} | Advantage: {advantages[k]:.4f}")
                        print(f"{'-' * 100}")
                    print(f"{'=' * 100}\n")

            keep_ratio = len(group_records) / max(1, len(batch_prompts))
            duplicate_group_ratio = duplicate_groups / max(1, len(batch_prompts))

            if len(group_records) == 0:
                epoch_stats["keep_ratio"].append(float(keep_ratio))
                epoch_stats["duplicate_group_ratio"].append(float(duplicate_group_ratio))
                train_pbar.set_description_str(self._train_pbar_desc(epoch))
                train_pbar.set_postfix_str(
                    f"keep={keep_ratio:.2f} dup={duplicate_group_ratio:.2f}"
                )
                train_pbar.refresh()
                batch_prompts = []
                batch_refs = []
                continue

            all_old_seq_logps = self._compute_old_seq_log_probs_batch(group_records)

            for _mu in range(cfg.grpo_inner_epochs):
                batch_loss, batch_stats_agg = self._compute_loss_for_groups(
                    group_records,
                    all_old_seq_logps,
                )
                batch_loss = batch_loss / cfg.gradient_accumulation

                batch_loss.backward()
                accumulated += 1

                if accumulated >= cfg.gradient_accumulation:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy_model.parameters(),
                        cfg.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accumulated = 0
                    self.global_step += 1

                    for key, values in batch_stats_agg.items():
                        epoch_stats.setdefault(key, []).append(float(np.mean(values)))

                    flat_rewards = [r for rec in group_records for r in rec["rewards"]]
                    epoch_stats["mean_reward"].append(float(np.mean(flat_rewards)))
                    epoch_stats["reward_std"].append(float(np.std(flat_rewards)))
                    epoch_stats["keep_ratio"].append(float(keep_ratio))
                    epoch_stats["mean_keep_weight"].append(float(np.mean([rec["keep_weight"] for rec in group_records])))
                    epoch_stats["mean_pred_len"].append(float(np.mean([rec["mean_pred_len"] for rec in group_records])))
                    epoch_stats["mean_len_ratio"].append(float(np.mean([rec["mean_len_ratio"] for rec in group_records])))
                    epoch_stats["duplicate_group_ratio"].append(float(duplicate_group_ratio))
                    epoch_stats["grad_norm"].append(float(grad_norm))
                    current_lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else cfg.learning_rate

                    train_row = {
                        "step": int(self.global_step),
                        "epoch": float(epoch + 1),
                        "rl_loss": float(np.mean(epoch_stats["rl_loss"][-5:])),
                        "ce_loss": float(np.mean(epoch_stats["ce_loss"][-5:])),
                        "ce_alpha": float(np.mean(epoch_stats["ce_alpha"][-5:])),
                        "kl_div": float(np.mean(epoch_stats["kl_div"][-5:])),
                        "approx_rl_term": float(np.mean(epoch_stats["approx_rl_term"][-5:])),
                        "approx_ce_term": float(np.mean(epoch_stats["approx_ce_term"][-5:])),
                        "ce_to_rl_abs_ratio": float(np.mean(epoch_stats["ce_to_rl_abs_ratio"][-5:])),
                        "mean_abs_advantage": float(np.mean(epoch_stats["mean_abs_advantage"][-5:])),
                        "mean_abs_ratio_delta": float(np.mean(epoch_stats["mean_abs_ratio_delta"][-5:])),
                        "grad_norm": float(np.mean(epoch_stats["grad_norm"][-5:])),
                        "mean_reward": float(np.mean(epoch_stats["mean_reward"][-5:])),
                        "keep_ratio": float(np.mean(epoch_stats["keep_ratio"][-5:])),
                        "mean_keep_weight": float(np.mean(epoch_stats["mean_keep_weight"][-5:])),
                        "mean_len_ratio": float(np.mean(epoch_stats["mean_len_ratio"][-5:])),
                        "duplicate_group_ratio": float(np.mean(epoch_stats["duplicate_group_ratio"][-5:])),
                        "lr": current_lr,
                    }
                    self._record_train_history(train_row)

                    log_str = (
                        f"upd={self.global_step} "
                        f"rl={np.mean(epoch_stats['rl_loss'][-5:]):.3e} "
                        f"ce={np.mean(epoch_stats['ce_loss'][-5:]):.3e} "
                        f"a={np.mean(epoch_stats['ce_alpha'][-5:]):.3f} "
                        f"kl={np.mean(epoch_stats['kl_div'][-5:]):.3e} "
                        f"rlT={np.mean(epoch_stats['approx_rl_term'][-5:]):.3e} "
                        f"ceT={np.mean(epoch_stats['approx_ce_term'][-5:]):.3e} "
                        f"ce/rl={np.mean(epoch_stats['ce_to_rl_abs_ratio'][-5:]):.1e} "
                        f"|adv|={np.mean(epoch_stats['mean_abs_advantage'][-5:]):.3f} "
                        f"|r-1|={np.mean(epoch_stats['mean_abs_ratio_delta'][-5:]):.3e} "
                        f"g={np.mean(epoch_stats['grad_norm'][-5:]):.3e} "
                        f"reward={np.mean(epoch_stats['mean_reward'][-5:]):.2f} "
                        f"keep={np.mean(epoch_stats['keep_ratio'][-5:]):.2f} "
                        f"w={np.mean(epoch_stats['mean_keep_weight'][-5:]):.2f} "
                        f"lenr={np.mean(epoch_stats['mean_len_ratio'][-5:]):.2f}"
                    )
                    train_pbar.n = min(self.global_step, train_pbar.total)
                    train_pbar.set_description_str(self._train_pbar_desc(epoch))
                    train_pbar.set_postfix_str(self._train_pbar_postfix(epoch_stats))
                    train_pbar.refresh()
                    if self.global_step % cfg.logging_steps == 0:
                        print(f"[Train] epoch={epoch + 1} {log_str}")

                    # step eval: 快速 teacher-forced loss+token_acc，不跑 generate
                    if cfg.eval_every_n_steps > 0 and self.global_step % cfg.eval_every_n_steps == 0:
                        eval_metrics = self.evaluate_fast(
                            epoch=epoch,
                            eval_tag="step",
                        )
                        self._maybe_save(eval_metrics, epoch, count_patience=False)

                    if cfg.max_train_steps > 0 and self.global_step >= cfg.max_train_steps:
                        print(f"\n   [max_train_steps={cfg.max_train_steps} reached]")
                        return {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_stats.items()}

            batch_prompts = []
            batch_refs = []

        if accumulated > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.policy_model.parameters(),
                cfg.max_grad_norm,
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            self.global_step += 1
            epoch_stats["grad_norm"].append(float(grad_norm))
            train_pbar.n = min(self.global_step, train_pbar.total)
            train_pbar.set_description_str(self._train_pbar_desc(epoch))
            train_pbar.set_postfix_str(self._train_pbar_postfix(epoch_stats))
            train_pbar.refresh()

        return {k: float(np.mean(v)) if v else 0.0 for k, v in epoch_stats.items()}

    # -----------------------
    # eval: fast (teacher-forced loss + token_acc only, no generate)
    # -----------------------
    @torch.no_grad()
    def evaluate_fast(
        self,
        epoch: int = -1,
        eval_tag: str = "step",
    ) -> Dict[str, float]:
        """Fast eval: only teacher-forced CE loss + token accuracy (no generate).
        Comparable with train.py's eval_loss and token_acc."""
        cfg = self.config
        self.policy_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_valid = 0
        total_examples = 0

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=cfg.eval_loss_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

        for batch in tqdm(eval_loader, desc="Eval(fast)", leave=False):
            prompts = [item["prompt"] for item in batch]
            refs = [item["reference"] for item in batch]

            enc_ids, enc_mask, labels = self._prepare_batch_prompt_and_labels(prompts, refs)
            outputs = self.policy_model(
                input_ids=enc_ids,
                attention_mask=enc_mask,
                labels=labels,
            )
            bs = enc_ids.size(0)
            total_loss += float(outputs.loss.item()) * bs
            total_examples += bs

            logits = outputs.logits
            preds_tok = logits.argmax(dim=-1)
            valid_mask = (labels != -100)
            correct = ((preds_tok == labels) & valid_mask).sum().item()
            total_correct += correct
            total_valid += valid_mask.sum().item()

        eval_loss = total_loss / max(total_examples, 1)
        token_acc = total_correct / max(total_valid, 1)

        metrics = {
            "eval_loss": eval_loss,
            "eval_token_acc": token_acc,
        }

        print(f"\n   ┌─── Step {self.global_step} ({eval_tag}) ───┐")
        print(f"   │ loss: val={eval_loss:.4f}  token_acc={token_acc:.4f}     │")
        print(f"   └───────────────────────────────────────────┘")

        eval_row = {
            "step": int(self.global_step),
            "epoch": float(epoch + 1) if epoch >= 0 else float(epoch),
            "eval_tag": eval_tag,
            "decode_strategy": "teacher_forced",
            **metrics,
        }
        self._record_eval_history(eval_row)
        self._render_plots()

        self.policy_model.train()
        return metrics

    # -----------------------
    # eval: greedy generate + teacher-forced loss/token_acc
    # -----------------------
    @torch.no_grad()
    def evaluate(
        self,
        epoch: int = -1,
        save_predictions: bool | None = None,
        eval_tag: str = "eval",
    ) -> Dict[str, float]:
        import sacrebleu

        cfg = self.config
        self.policy_model.eval()
        if save_predictions is None:
            save_predictions = cfg.save_eval_predictions

        all_preds: List[str] = []
        all_refs: List[str] = []

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

        # 1. Beam generate (boundary eval)
        _num_beams = cfg.eval_boundary_num_beams
        for batch in tqdm(eval_loader, desc=f"Eval(beam={_num_beams})", leave=False):
            prompts = [item["prompt"] for item in batch]
            refs = [item["reference"] for item in batch]

            enc = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=cfg.max_input_length,
                return_tensors="pt",
            ).to(self.device)

            outputs = self.policy_model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=cfg.max_target_length,
                num_beams=cfg.eval_boundary_num_beams,
                do_sample=False,
                use_cache=True,
            )
            decoded = decode_sequences(outputs, tokenizer=self.tokenizer)
            post_decoded = normalize_predictions_batch(decoded)
            all_preds.extend(post_decoded)
            all_refs.extend(refs)

        # 2. Corpus metrics
        bleu = float(sacrebleu.corpus_bleu(all_preds, [all_refs]).score)
        chrf = float(sacrebleu.corpus_chrf(all_preds, [all_refs], word_order=2).score)
        geom = float(math.sqrt(max(bleu, 0.01) * max(chrf, 0.01)))
        pred_lens = [len(p.split()) for p in all_preds]
        ref_lens = [len(r.split()) for r in all_refs]
        mean_pred_len = float(np.mean(pred_lens)) if pred_lens else 0.0
        mean_ref_len = float(np.mean(ref_lens)) if ref_lens else 0.0
        mean_len_ratio = mean_pred_len / max(1e-8, mean_ref_len)

        # 3. Teacher-forced loss + token_acc
        loss_metrics = self.evaluate_fast(epoch=epoch, eval_tag=f"{eval_tag}_tf")
        eval_loss = loss_metrics["eval_loss"]
        token_acc = loss_metrics["eval_token_acc"]

        metrics = {
            "eval_loss": eval_loss,
            "eval_token_acc": token_acc,
            "bleu": bleu,
            "chrf": chrf,
            "geom_mean": geom,
            "mean_pred_len": mean_pred_len,
            "mean_ref_len": mean_ref_len,
            "mean_len_ratio": mean_len_ratio,
        }

        # train.py-style box logging
        print(f"\n   ┌─── Step {self.global_step} ({eval_tag}) ───┐")
        print(f"   │ BLEU={bleu:6.2f}  chrF++={chrf:6.2f}  geom={geom:6.2f} │")
        print(f"   │ loss: val={eval_loss:.4f}  token_acc={token_acc:.4f}     │")
        print(f"   │ len_ratio={mean_len_ratio:.2f}  decode=beam{_num_beams}          │")
        print(f"   └───────────────────────────────────────────┘")

        if save_predictions:
            pred_path = os.path.join(cfg.output_dir, f"predictions_{eval_tag}_step{self.global_step}.csv")
            per_sample_chrf = [compute_chrf(p, r) for p, r in zip(all_preds, all_refs)]
            pd.DataFrame({
                "reference": all_refs,
                "prediction": all_preds,
                "chrf": per_sample_chrf,
                "pred_len": pred_lens,
                "ref_len": ref_lens,
            }).to_csv(pred_path, index=False)

        eval_row = {
            "step": int(self.global_step),
            "epoch": float(epoch + 1) if epoch >= 0 else float(epoch),
            "eval_tag": eval_tag,
            "decode_strategy": "greedy",
            **metrics,
        }
        self._record_eval_history(eval_row)
        self._render_plots()

        self.policy_model.train()
        return metrics

    # -----------------------
    # save / stop
    # -----------------------
    def _maybe_save(self, metrics: Dict[str, float], epoch: int, count_patience: bool) -> None:
        cfg = self.config
        metric_val = metrics.get(cfg.early_stopping_metric, 0.0)

        # 支持 loss（越低越好）和 chrf（越高越好）两种模式
        if cfg.early_stopping_greater_is_better:
            is_better = metric_val > self.best_metric
        else:
            is_better = metric_val < self.best_metric

        if is_better:
            self.best_metric = metric_val
            self.patience_counter = 0

            save_dir = os.path.join(cfg.output_dir, f"best_step{self.global_step}")
            os.makedirs(save_dir, exist_ok=True)
            self.policy_model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

            with open(os.path.join(save_dir, "metrics.json"), "w") as f:
                json.dump(
                    {
                        **metrics,
                        "global_step": self.global_step,
                        "epoch": epoch,
                    },
                    f,
                    indent=2,
                )
            print(f"   -> New best! {cfg.early_stopping_metric}={metric_val:.2f}  Saved to {save_dir}")

            ckpts = sorted(
                glob.glob(os.path.join(cfg.output_dir, "best_step*")),
                key=os.path.getmtime,
            )
            while len(ckpts) > cfg.save_total_limit:
                old = ckpts.pop(0)
                shutil.rmtree(old, ignore_errors=True)
                print(f"   Removed old checkpoint: {old}")
        else:
            if count_patience:
                self.patience_counter += 1
                print(f"   No improvement. Patience: {self.patience_counter}/{cfg.early_stopping_patience}")
            else:
                print("   No improvement.")

    def should_stop(self) -> bool:
        return self.patience_counter >= self.config.early_stopping_patience

    # -----------------------
    # main loop
    # -----------------------
    def train(self) -> None:
        from transformers import get_cosine_schedule_with_warmup

        cfg = self.config

        rollout_batches_per_epoch = math.ceil(len(self.train_dataset) / cfg.grpo_batch_size)
        updates_per_epoch = math.ceil(
            rollout_batches_per_epoch * cfg.grpo_inner_epochs / cfg.gradient_accumulation
        )
        total_steps = updates_per_epoch * cfg.grpo_epochs
        warmup_steps = int(total_steps * cfg.warmup_ratio)
        self.total_steps_nominal = total_steps

        optimizer = torch.optim.AdamW(
            self.policy_model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps,
            total_steps,
        )
        self._restore_training_state(optimizer, scheduler)

        print(f"\n{'=' * 80}")
        print("Training Configuration:")
        print(f"  Group size (K):          {cfg.group_size}")
        print(f"  Batch size:              {cfg.grpo_batch_size}")
        print(f"  Logprob batch size:      {cfg.rollout_logprob_batch_size}")
        print(f"  PPO cand batch size:     {cfg.ppo_candidate_batch_size}")
        print(f"  CE batch size:           {cfg.ce_batch_size}")
        print(f"  Rollout batches/epoch:   ~{rollout_batches_per_epoch}")
        print(f"  Updates/epoch:           ~{updates_per_epoch}")
        print(f"  Total optimizer steps:   ~{total_steps}")
        print(f"  Warmup steps:            {warmup_steps}")
        print(f"  LR:                      {cfg.learning_rate}")
        print(f"  KL coeff:                {cfg.kl_coeff}")
        print(f"  CE alpha init/final:     {cfg.ce_alpha_init:.3f} / {cfg.ce_alpha_final:.3f}")
        print(f"  CE hold ratio:           {cfg.ce_alpha_hold_ratio:.2f}")
        print(f"  CE decay end ratio:      {cfg.ce_alpha_decay_end_ratio:.2f}")
        print(f"  Clip eps lower:          {cfg.clip_eps}")
        print(f"  Clip eps higher:         {cfg.clip_eps_higher}")
        print(f"  Rollout beams/sample:    {cfg.rollout_beam_cands} / {cfg.rollout_sample_cands}")
        print(f"  Rollout num_beams:       {cfg.rollout_num_beams}")
        print(f"  Rollout pool cap:        {cfg.rollout_pool_cap}")
        print(f"  Temperature:             {cfg.temperature}")
        print(f"  Top-k / Top-p:           {cfg.top_k} / {cfg.top_p}")
        print(f"  Eval decode:             step=greedy, boundary=beam{cfg.eval_boundary_num_beams}")
        print(f"  Eval batch size:         {cfg.eval_batch_size}")
        print(f"  Early stop metric:       {cfg.early_stopping_metric} (patience={cfg.early_stopping_patience})")
        print(f"  Reward type:             {cfg.reward_type}")
        print(f"  Under len target:        {cfg.under_length_target_ratio}")
        print(f"  Over len target:         {cfg.over_length_target_ratio}")
        print(f"  Dynamic info tau:        {cfg.info_tau}")
        print(f"  Min reward std:          {cfg.min_reward_std}")
        print(f"  Min group weight:        {cfg.min_group_weight}")
        print(f"  Use bf16:                {cfg.use_bf16}")
        print(f"{'=' * 80}\n")

        with open(os.path.join(cfg.output_dir, "grpo_config.json"), "w") as f:
            json.dump(
                {k: v for k, v in vars(cfg).items() if not k.startswith("_")},
                f,
                indent=2,
                default=str,
            )
        print(f"  Full terminal log:       {self.log_file}")
        print(f"  Train history CSV:       {self.train_history_path}")
        print(f"  Eval history CSV:        {self.eval_history_path}")
        print(f"  Plots dir:               {self.plots_dir}")

        if not self.resume_checkpoint_path and self.global_step == 0:
            bootstrap_checkpoint = self._save_resume_checkpoint(epoch=-1, optimizer=optimizer, scheduler=scheduler)
            print(f"  Bootstrap checkpoint:    {bootstrap_checkpoint}")

        if cfg.run_initial_eval:
            print("[Initial Evaluation]")
            init_metrics = self.evaluate_fast(
                epoch=-1,
                eval_tag="initial",
            )
            self._maybe_save(init_metrics, epoch=-1, count_patience=False)
        else:
            print("[Initial Evaluation] skipped")

        train_pbar = tqdm(total=max(1, total_steps), dynamic_ncols=True)
        try:
            train_pbar.n = min(self.global_step, train_pbar.total)
            train_pbar.refresh()

            for epoch in range(self.start_epoch, cfg.grpo_epochs):
                print(f"\n{'=' * 80}")
                print(f"Epoch {epoch + 1}/{cfg.grpo_epochs}")
                print(f"{'=' * 80}")

                epoch_stats = self.train_epoch(epoch, optimizer, scheduler, train_pbar)

                print(f"\n[Epoch {epoch + 1} Summary]")
                for k, v in epoch_stats.items():
                    print(f"  {k}: {v:.4f}")

                # epoch-end eval: 只有这里累计 patience
                eval_metrics = self.evaluate_fast(
                    epoch=epoch,
                    eval_tag="epoch_end",
                )
                self._maybe_save(eval_metrics, epoch, count_patience=True)
                saved_checkpoint = self._save_resume_checkpoint(epoch, optimizer, scheduler)
                print(f"   Resume checkpoint saved to {saved_checkpoint}")

                if self.should_stop():
                    print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                    break
        finally:
            train_pbar.n = min(self.global_step, train_pbar.total)
            train_pbar.refresh()
            train_pbar.close()

        final_dir = os.path.join(cfg.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        self.policy_model.save_pretrained(final_dir)
        self.tokenizer.save_pretrained(final_dir)
        self._render_plots()
        print(f"\nTraining completed. Final model saved to {final_dir}")
        print(f"Best {cfg.early_stopping_metric}: {self.best_metric:.2f}")
        print(f"Logs: {self.log_file}")
        print(f"Plots: {self.plots_dir}")


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    trainer = GRPOTrainer(config)
    trainer.train()
