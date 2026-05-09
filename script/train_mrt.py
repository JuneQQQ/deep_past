#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Offline MRT for Encoder-Decoder ByT5
================================================================================

使用预生成候选集合做 Minimum Risk Training：

- 输入：infer.py 的 MRT 候选 JSONL
- 目标：最小化候选集合上的期望风险（expected risk）
- 可选：附带轻量 CE anchor，避免模型漂移
- 评估：beam / hybrid_mbr
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
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from infer import VectorizedPostprocessor, strip_input_artifacts, decode_sequences


parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", type=str, default=None)
parser.add_argument("--train_jsonl", type=str, default=None)
parser.add_argument("--smoke_test", action="store_true", help="Minimal run to verify pipeline")
parser.add_argument(
    "--skip_initial_eval",
    action="store_true",
    help="Skip the evaluation that runs before the first optimizer step",
)
args, _ = parser.parse_known_args()

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
_bootstrap_timestamp = os.environ.get("TRAINING_TIMESTAMP") or args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
os.environ["TRAINING_TIMESTAMP"] = _bootstrap_timestamp
_bootstrap_output_dir = os.path.join("/root/projects/deep_past", "output", f"mrt_train_{_bootstrap_timestamp}")
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

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("=" * 80)
print("OFFLINE MRT FOR BYT5 TRANSLATION")
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


class MRTConfig:
    def __init__(self) -> None:
        local_root = "/root/projects/deep_past"
        self.root_dir = local_root
        self.data_dir = os.path.join(local_root, "data")
        self.train_jsonl = args.train_jsonl or os.path.join(self.data_dir, "mrt_candidates_filtered.jsonl")

        timestamp = os.environ.get("TRAINING_TIMESTAMP") or args.timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.root_dir, "output", f"mrt_train_{timestamp}")
        self.run_initial_eval = not args.skip_initial_eval

        self.model_name = "/root/projects/deep_past/example/model/byt5-akkadian-mbr-v2-pytorch-default-v1"
        self.task_prefix = "translate Akkadian to English: "

        self.max_input_length = 512
        self.max_target_length = 512

        # optimization
        self.num_epochs = 4
        self.mrt_batch_size = 8
        self.gradient_accumulation = 1
        self.candidate_batch_size = 24
        self.ce_batch_size = 16
        self.learning_rate = 1e-5
        self.warmup_ratio = 0.06
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0

        # mrt
        self.mrt_alpha = 2.5
        self.ce_alpha = 0.05
        self.min_candidates = 2
        self.inject_reference_candidate = True

        # eval
        self.eval_decode_strategy = "hybrid_mbr"  # beam / hybrid_mbr
        self.eval_batch_size = 32
        self.eval_num_beams = 8
        self.eval_length_penalty = 1.3
        self.eval_beam_cands = 4
        self.eval_sample_cands = 2
        self.eval_temperature = 0.75
        self.eval_top_p = 0.92
        self.eval_pool_cap = 32
        self.eval_early_stopping = True
        self.eval_async_postprocess = True

        # reward
        self.reward_mix_chrf = 0.7
        self.reward_mix_bleu = 0.3
        self.under_length_target_ratio = 0.90
        self.under_length_scale = 40.0
        self.over_length_target_ratio = 1.35
        self.over_length_scale = 6.0

        # misc
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        self.gradient_checkpointing = True
        self.eval_every_n_steps = 200
        self.logging_steps = 10
        self.save_total_limit = 2
        self.early_stopping_patience = 3
        self.early_stopping_metric = "chrf"
        self.save_eval_predictions = True
        self.max_train_steps = 0

        self.test_size = 0.1
        self.random_seed = 42
        self.fold_group_col = "oare_id"


config = MRTConfig()

if args.smoke_test:
    config.num_epochs = 1
    config.mrt_batch_size = 2
    config.eval_batch_size = 8
    config.eval_every_n_steps = 5
    config.logging_steps = 1
    config.eval_num_beams = 4
    config.eval_beam_cands = 2
    config.eval_sample_cands = 1
    config.eval_pool_cap = 8
    config.max_train_steps = 10
    print("[SMOKE TEST MODE] Using real data, limited steps")

set_seed(config.random_seed)
os.makedirs(config.output_dir, exist_ok=True)

_NOTEBOOK_POSTPROCESSOR = VectorizedPostprocessor()


def normalize_predictions_batch(texts: List[str]) -> List[str]:
    stripped = [strip_input_artifacts(str(t or "")) for t in texts]
    return _NOTEBOOK_POSTPROCESSOR.postprocess_batch(stripped)


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


def compute_chrf(prediction: str, reference: str) -> float:
    import sacrebleu
    return float(sacrebleu.sentence_chrf(prediction, [reference], word_order=2).score)


def compute_bleu(prediction: str, reference: str) -> float:
    import sacrebleu
    return float(sacrebleu.sentence_bleu(prediction, [reference]).score)


def length_penalty(prediction: str, reference: str, cfg: MRTConfig) -> float:
    pred_len = max(1, len(prediction.split()))
    ref_len = max(1, len(reference.split()))
    ratio = pred_len / ref_len
    under = max(0.0, cfg.under_length_target_ratio - ratio)
    over = max(0.0, ratio - cfg.over_length_target_ratio)
    return (
        cfg.under_length_scale * (under ** 2)
        + cfg.over_length_scale * (over ** 2)
    )


def compute_reward(prediction: str, reference: str, cfg: MRTConfig) -> float:
    pred = str(prediction).strip()
    ref = str(reference).strip()
    if not pred or not ref:
        return 0.0
    chrf = compute_chrf(pred, ref)
    bleu = compute_bleu(pred, ref)
    base_reward = cfg.reward_mix_chrf * chrf + cfg.reward_mix_bleu * bleu
    return float(base_reward - length_penalty(pred, ref, cfg))


class MRTRecordDataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]], task_prefix: str) -> None:
        self.records = []
        for rec in records:
            tl = str(rec.get("transliteration", "")).strip()
            ref = str(rec.get("reference", rec.get("translation", ""))).strip()
            candidates = [str(x).strip() for x in rec.get("candidates", []) if str(x).strip()]
            if not tl or not ref or len(candidates) < 2:
                continue
            self.records.append(
                {
                    "prompt": task_prefix + tl,
                    "reference": ref,
                    "transliteration": tl,
                    "oare_id": str(rec.get("oare_id", "")),
                    "genre": str(rec.get("genre", "unknown")),
                    "candidates": candidates,
                    "candidate_rewards": rec.get("candidate_rewards", []),
                }
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.records[idx]


class MRTTrainer:
    def __init__(self, config: MRTConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.train_history_path = os.path.join(config.output_dir, "train_history.csv")
        self.eval_history_path = os.path.join(config.output_dir, "eval_history.csv")
        self.plots_dir = os.path.join(config.output_dir, "plots")

        self.global_step = 0
        self.total_steps_nominal = 1
        self.best_metric = -float("inf")
        self.patience_counter = 0
        self.train_history: List[Dict[str, float]] = []
        self.eval_history: List[Dict[str, float]] = []

        self._load_model()
        self._load_data()

    def _record_train_history(self, row: Dict[str, Any]) -> None:
        self.train_history.append(row)
        pd.DataFrame(self.train_history).to_csv(self.train_history_path, index=False)

    def _record_eval_history(self, row: Dict[str, Any]) -> None:
        self.eval_history.append(row)
        pd.DataFrame(self.eval_history).to_csv(self.eval_history_path, index=False)

    def _render_plots(self) -> None:
        import matplotlib.pyplot as plt

        os.makedirs(self.plots_dir, exist_ok=True)

        if self.train_history:
            train_df = pd.DataFrame(self.train_history)
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))

            ax = axes[0, 0]
            for col, label in [("mrt_loss", "MRT"), ("ce_loss", "CE"), ("total_loss", "Total")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            ax.set_title("Train Loss Components")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax = axes[0, 1]
            for col, label in [("mean_reward", "Reward"), ("chosen_reward", "Chosen"), ("mean_len_ratio", "Len Ratio")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            ax.axhline(1.0, color="r", linestyle="--", alpha=0.5)
            ax.set_title("Reward / Length Ratio")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax = axes[1, 0]
            for col, label in [("candidate_entropy", "Candidate Entropy"), ("mean_group_size", "Group Size")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            ax.set_title("Candidate Distribution")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax = axes[1, 1]
            for col, label in [("grad_norm", "Grad Norm"), ("lr", "LR")]:
                if col in train_df:
                    ax.plot(train_df["step"], train_df[col], label=label)
            ax.set_title("Optimization")
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir, "train_metrics.png"), dpi=150)
            plt.close(fig)

        if self.eval_history:
            eval_df = pd.DataFrame(self.eval_history)
            fig, axes = plt.subplots(2, 1, figsize=(12, 9))

            ax = axes[0]
            for col, label in [("bleu", "BLEU"), ("chrf", "chrF++"), ("geom_mean", "Geom Mean")]:
                if col in eval_df:
                    ax.plot(eval_df["step"], eval_df[col], marker="o", label=label)
            ax.set_title("Eval Metrics")
            ax.grid(True, alpha=0.3)
            ax.legend()

            ax = axes[1]
            for col, label in [("mean_len_ratio", "Len Ratio"), ("mean_pred_len", "Pred Len"), ("mean_ref_len", "Ref Len")]:
                if col in eval_df:
                    ax.plot(eval_df["step"], eval_df[col], marker="o", label=label)
            ax.axhline(1.0, color="r", linestyle="--", alpha=0.5)
            ax.set_title("Eval Length Diagnostics")
            ax.grid(True, alpha=0.3)
            ax.legend()

            fig.tight_layout()
            fig.savefig(os.path.join(self.plots_dir, "eval_metrics.png"), dpi=150)
            plt.close(fig)

    def _load_model(self) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        cfg = self.config
        load_dtype = torch.bfloat16 if cfg.use_bf16 else torch.float32

        print(f"\n[Model] Loading from {cfg.model_name}")
        print(f"   dtype: {load_dtype}, device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_name,
            local_files_only=os.path.isdir(cfg.model_name),
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.policy_model = AutoModelForSeq2SeqLM.from_pretrained(
            cfg.model_name,
            torch_dtype=load_dtype,
            local_files_only=os.path.isdir(cfg.model_name),
            low_cpu_mem_usage=True,
            dropout_rate=0.0,
        ).to(self.device)

        if cfg.gradient_checkpointing:
            self.policy_model.gradient_checkpointing_enable()
            self.policy_model.config.use_cache = False

        total = sum(p.numel() for p in self.policy_model.parameters())
        trainable = sum(p.numel() for p in self.policy_model.parameters() if p.requires_grad)
        print(f"   Params: {total:,} total, {trainable:,} trainable")

    def _load_data(self) -> None:
        from sklearn.model_selection import GroupShuffleSplit

        cfg = self.config
        print(f"\n[Data] Loading MRT candidates from {cfg.train_jsonl}")
        raw_records: List[Dict[str, Any]] = []
        with open(cfg.train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw_records.append(json.loads(line))

        injected_reference_count = 0
        for rec in tqdm(raw_records, desc="Scoring MRT candidates", leave=False):
            ref = str(rec.get("reference", rec.get("translation", ""))).strip()
            cands = []
            seen = set()
            for cand in rec.get("candidates", []):
                norm = normalize_predictions_batch([cand])[0].strip()
                if norm and norm not in seen:
                    seen.add(norm)
                    cands.append(norm)
            ref_norm = normalize_predictions_batch([ref])[0].strip() if ref else ""
            if cfg.inject_reference_candidate and ref_norm and ref_norm not in seen:
                cands.insert(0, ref_norm)
                seen.add(ref_norm)
                injected_reference_count += 1
            rec["candidates"] = cands
            rec["candidate_rewards"] = [compute_reward(c, ref, cfg) for c in cands]

        df = pd.DataFrame(raw_records)
        df["oare_id"] = df.get("oare_id", "").astype(str)
        df["candidate_count"] = df["candidates"].apply(lambda xs: len(xs) if isinstance(xs, list) else 0)
        df = df[df["candidate_count"] >= cfg.min_candidates].reset_index(drop=True)

        print(f"   Total records after filtering: {len(df)}")
        print(
            f"   Reference injection:      "
            f"{'ON' if cfg.inject_reference_candidate else 'OFF'} "
            f"(inserted into {injected_reference_count} record(s))"
        )

        groups = df[cfg.fold_group_col] if cfg.fold_group_col in df.columns else None
        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=cfg.test_size,
            random_state=cfg.random_seed,
        )
        train_idx, eval_idx = next(gss.split(df, groups=groups))
        train_records = df.iloc[train_idx].to_dict("records")
        eval_records = df.iloc[eval_idx].to_dict("records")

        if args.smoke_test:
            train_records = train_records[:6]
            eval_records = eval_records[:4]

        self.train_dataset = MRTRecordDataset(train_records, cfg.task_prefix)
        self.eval_dataset = MRTRecordDataset(eval_records, cfg.task_prefix)
        print(f"   Train: {len(self.train_dataset)}, Eval: {len(self.eval_dataset)}")

    def _prepare_batch_prompt_and_labels(
        self,
        prompts: List[str],
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg = self.config
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
        label_ids = labels_batch["input_ids"].masked_fill(labels_batch["input_ids"] == self.tokenizer.pad_token_id, -100)
        return enc["input_ids"], enc["attention_mask"], label_ids

    def _forward_token_stats_batch(
        self,
        prompts: List[str],
        target_texts: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_ids, enc_mask, labels = self._prepare_batch_prompt_and_labels(prompts, target_texts)
        outputs = self.policy_model(input_ids=enc_ids, attention_mask=enc_mask, labels=labels)
        logits = outputs.logits.float()
        valid_mask = (labels != -100).float()
        labels_for_gather = labels.masked_fill(labels == -100, 0)
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels_for_gather.unsqueeze(-1)).squeeze(-1) * valid_mask
        seq_len = valid_mask.sum(dim=1).clamp(min=1.0)
        seq_mean_log_probs = token_log_probs.sum(dim=1) / seq_len
        return seq_mean_log_probs, valid_mask

    def _compute_seq_mean_log_probs_batch(
        self,
        prompts: List[str],
        target_texts: List[str],
    ) -> torch.Tensor:
        if not prompts:
            return torch.empty(0, device=self.device)
        parts: List[torch.Tensor] = []
        batch_size = max(1, self.config.candidate_batch_size)
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            seq_logps, _ = self._forward_token_stats_batch(prompts[start:end], target_texts[start:end])
            parts.append(seq_logps)
        return torch.cat(parts, dim=0)

    def _compute_ce_losses_batch(
        self,
        prompts: List[str],
        references: List[str],
    ) -> torch.Tensor:
        if not prompts:
            return torch.empty(0, device=self.device)
        parts: List[torch.Tensor] = []
        batch_size = max(1, self.config.ce_batch_size)
        for start in range(0, len(prompts), batch_size):
            end = start + batch_size
            seq_logps, valid_mask = self._forward_token_stats_batch(prompts[start:end], references[start:end])
            parts.append(-seq_logps)
        return torch.cat(parts, dim=0)

    def _compute_mrt_loss_for_records(
        self,
        records: List[Dict[str, Any]],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        cfg = self.config
        if not records:
            zero = torch.tensor(0.0, device=self.device)
            return zero, {}

        flat_prompts: List[str] = []
        flat_candidates: List[str] = []
        flat_rewards: List[float] = []
        group_sizes: List[int] = []
        group_refs: List[str] = []
        group_prompts: List[str] = []

        for rec in records:
            cands = rec["candidates"]
            rewards = rec["candidate_rewards"]
            size = len(cands)
            group_sizes.append(size)
            group_refs.append(rec["reference"])
            group_prompts.append(rec["prompt"])
            flat_prompts.extend([rec["prompt"]] * size)
            flat_candidates.extend(cands)
            flat_rewards.extend(rewards)

        seq_logps = self._compute_seq_mean_log_probs_batch(flat_prompts, flat_candidates)
        ce_losses = self._compute_ce_losses_batch(group_prompts, group_refs) if cfg.ce_alpha > 0 else torch.zeros(len(records), device=self.device, dtype=seq_logps.dtype)

        losses = []
        chosen_rewards = []
        expected_rewards = []
        entropies = []
        len_ratios = []
        offset = 0
        for gi, rec in enumerate(records):
            size = group_sizes[gi]
            sl = slice(offset, offset + size)
            offset += size

            group_logps = seq_logps[sl]
            rewards = torch.tensor(flat_rewards[sl], dtype=group_logps.dtype, device=self.device)
            probs = torch.softmax(cfg.mrt_alpha * group_logps, dim=0)
            risk = rewards.max() - rewards
            mrt_loss = (probs * risk).sum()
            total_loss = mrt_loss + cfg.ce_alpha * ce_losses[gi]
            losses.append(total_loss)

            best_idx = int(torch.argmax(probs).item())
            chosen_rewards.append(float(rewards[best_idx].item()))
            expected_rewards.append(float((probs * rewards).sum().item()))
            entropies.append(float((-(probs * torch.log(probs.clamp_min(1e-8)))).sum().item()))

            pred_len = len(rec["candidates"][best_idx].split())
            ref_len = max(1, len(rec["reference"].split()))
            len_ratios.append(pred_len / ref_len)

        metrics = {
            "mrt_loss": float(torch.stack(losses).mean().item()),
            "ce_loss": float(ce_losses.mean().item()) if len(ce_losses) > 0 else 0.0,
            "total_loss": float(torch.stack(losses).mean().item()),
            "mean_reward": float(np.mean(expected_rewards)) if expected_rewards else 0.0,
            "chosen_reward": float(np.mean(chosen_rewards)) if chosen_rewards else 0.0,
            "candidate_entropy": float(np.mean(entropies)) if entropies else 0.0,
            "mean_group_size": float(np.mean(group_sizes)) if group_sizes else 0.0,
            "mean_len_ratio": float(np.mean(len_ratios)) if len_ratios else 0.0,
        }
        return torch.stack(losses).mean(), metrics

    def _generate_raw_candidate_groups(
        self,
        prompts: List[str],
        beam_cands: int,
        sample_cands: int,
        num_beams: int,
        length_penalty: float,
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
        with torch.no_grad():
            if beam_cands > 0:
                beam_outputs = self.policy_model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=cfg.max_target_length,
                    do_sample=False,
                    num_beams=max(num_beams, beam_cands),
                    num_return_sequences=beam_cands,
                    length_penalty=length_penalty,
                    early_stopping=early_stopping,
                    use_cache=True,
                )
                beam_decoded = decode_sequences(beam_outputs, tokenizer=self.tokenizer)
                for b in range(batch_size):
                    raw_groups[b].extend(beam_decoded[b * beam_cands:(b + 1) * beam_cands])

            if sample_cands > 0:
                sample_outputs = self.policy_model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=cfg.max_target_length,
                    do_sample=True,
                    top_p=top_p,
                    temperature=temperature,
                    num_return_sequences=sample_cands,
                    num_beams=1,
                    early_stopping=False,
                    use_cache=True,
                )
                sample_decoded = decode_sequences(sample_outputs, tokenizer=self.tokenizer)
                for b in range(batch_size):
                    raw_groups[b].extend(sample_decoded[b * sample_cands:(b + 1) * sample_cands])
        return raw_groups

    def _postprocess_candidate_groups(self, raw_groups: List[List[str]]) -> List[List[str]]:
        flat_raw = [text for group in raw_groups for text in group]
        flat_clean = normalize_predictions_batch(flat_raw) if flat_raw else []
        clean_groups: List[List[str]] = []
        offset = 0
        for group in raw_groups:
            size = len(group)
            clean_groups.append(flat_clean[offset:offset + size])
            offset += size
        return clean_groups

    @torch.no_grad()
    def evaluate(
        self,
        epoch: int = -1,
        decode_strategy: str = "",
        save_predictions: bool | None = None,
        eval_tag: str = "eval",
    ) -> Dict[str, float]:
        import sacrebleu

        cfg = self.config
        self.policy_model.eval()
        active_decode_strategy = decode_strategy or cfg.eval_decode_strategy
        if save_predictions is None:
            save_predictions = cfg.save_eval_predictions

        all_preds: List[str] = []
        all_refs: List[str] = []
        all_pred_lens: List[int] = []
        all_ref_lens: List[int] = []

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=cfg.eval_batch_size,
            shuffle=False,
            collate_fn=lambda batch: batch,
        )

        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            prompts = [item["prompt"] for item in batch]
            refs = [item["reference"] for item in batch]

            if active_decode_strategy == "hybrid_mbr":
                raw_groups = self._generate_raw_candidate_groups(
                    prompts=prompts,
                    beam_cands=cfg.eval_beam_cands,
                    sample_cands=cfg.eval_sample_cands,
                    num_beams=cfg.eval_num_beams,
                    length_penalty=cfg.eval_length_penalty,
                    top_p=cfg.eval_top_p,
                    temperature=cfg.eval_temperature,
                    early_stopping=cfg.eval_early_stopping,
                )
                clean_groups = self._postprocess_candidate_groups(raw_groups)
                decoded = [mbr_pick(group, pool_cap=cfg.eval_pool_cap) for group in clean_groups]
            else:
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
                    num_beams=cfg.eval_num_beams,
                    length_penalty=cfg.eval_length_penalty,
                    use_cache=True,
                )
                decoded = normalize_predictions_batch(decode_sequences(outputs, tokenizer=self.tokenizer))

            all_preds.extend(decoded)
            all_refs.extend(refs)
            all_pred_lens.extend([len(x.split()) for x in decoded])
            all_ref_lens.extend([len(x.split()) for x in refs])

        bleu = float(sacrebleu.corpus_bleu(all_preds, [all_refs]).score)
        chrf = float(sacrebleu.corpus_chrf(all_preds, [all_refs], word_order=2).score)
        geom = float(math.sqrt(max(bleu, 0.01) * max(chrf, 0.01)))
        mean_pred_len = float(np.mean(all_pred_lens)) if all_pred_lens else 0.0
        mean_ref_len = float(np.mean(all_ref_lens)) if all_ref_lens else 0.0
        mean_len_ratio = mean_pred_len / max(1e-8, mean_ref_len)

        metrics = {
            "bleu": bleu,
            "chrf": chrf,
            "geom_mean": geom,
            "mean_pred_len": mean_pred_len,
            "mean_ref_len": mean_ref_len,
            "mean_len_ratio": mean_len_ratio,
        }
        print(
            f"\n   [Eval:{eval_tag}] step={self.global_step}  "
            f"decode={active_decode_strategy}  "
            f"BLEU={bleu:.2f}  chrF++={chrf:.2f}  geom={geom:.2f}  "
            f"len_ratio={mean_len_ratio:.2f}"
        )

        if save_predictions:
            pred_path = os.path.join(cfg.output_dir, f"predictions_{eval_tag}_step{self.global_step}.csv")
            per_sample_chrf = [compute_chrf(p, r) for p, r in zip(all_preds, all_refs)]
            pd.DataFrame(
                {
                    "reference": all_refs,
                    "prediction": all_preds,
                    "chrf": per_sample_chrf,
                    "pred_len": all_pred_lens,
                    "ref_len": all_ref_lens,
                }
            ).to_csv(pred_path, index=False)

        eval_row = {
            "step": int(self.global_step),
            "epoch": float(epoch + 1) if epoch >= 0 else float(epoch),
            "eval_tag": eval_tag,
            "decode_strategy": active_decode_strategy,
            **metrics,
        }
        self._record_eval_history(eval_row)
        self._render_plots()
        self.policy_model.train()
        return metrics

    def _maybe_save(self, metrics: Dict[str, float], epoch: int) -> None:
        cfg = self.config
        metric_val = metrics.get(cfg.early_stopping_metric, 0.0)
        if metric_val > self.best_metric:
            self.best_metric = metric_val
            self.patience_counter = 0

            save_dir = os.path.join(cfg.output_dir, f"best_step{self.global_step}")
            os.makedirs(save_dir, exist_ok=True)
            self.policy_model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

            with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
                json.dump({**metrics, "global_step": self.global_step, "epoch": epoch}, f, indent=2)
            print(f"   -> New best! {cfg.early_stopping_metric}={metric_val:.2f}  Saved to {save_dir}")

            ckpts = sorted(glob.glob(os.path.join(cfg.output_dir, "best_step*")), key=os.path.getmtime)
            while len(ckpts) > cfg.save_total_limit:
                old = ckpts.pop(0)
                shutil.rmtree(old, ignore_errors=True)
                print(f"   Removed old checkpoint: {old}")
        else:
            self.patience_counter += 1
            print(f"   No improvement. Patience: {self.patience_counter}/{cfg.early_stopping_patience}")

    def should_stop(self) -> bool:
        return self.patience_counter >= self.config.early_stopping_patience

    def train(self) -> None:
        from transformers import get_cosine_schedule_with_warmup

        cfg = self.config
        steps_per_epoch = math.ceil(len(self.train_dataset) / cfg.mrt_batch_size / max(1, cfg.gradient_accumulation))
        total_steps = max(1, steps_per_epoch * cfg.num_epochs)
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

        print(f"\n{'=' * 80}")
        print("Training Configuration:")
        print(f"  Train JSONL:             {cfg.train_jsonl}")
        print(f"  MRT batch size:          {cfg.mrt_batch_size}")
        print(f"  Candidate batch size:    {cfg.candidate_batch_size}")
        print(f"  CE batch size:           {cfg.ce_batch_size}")
        print(f"  Epochs:                  {cfg.num_epochs}")
        print(f"  Total optimizer steps:   ~{total_steps}")
        print(f"  Warmup steps:            {warmup_steps}")
        print(f"  LR:                      {cfg.learning_rate}")
        print(f"  MRT alpha:               {cfg.mrt_alpha}")
        print(f"  CE alpha:                {cfg.ce_alpha}")
        print(f"  Reference injection:     {cfg.inject_reference_candidate}")
        print(f"  Eval decode:             {cfg.eval_decode_strategy}")
        print(f"  Eval beams/sample:       {cfg.eval_beam_cands} / {cfg.eval_sample_cands}")
        print(f"  Reward mix chrf/bleu:    {cfg.reward_mix_chrf} / {cfg.reward_mix_bleu}")
        print(f"  Under length scale:      {cfg.under_length_scale}")
        print(f"  Use bf16:                {cfg.use_bf16}")
        print(f"{'=' * 80}\n")

        with open(os.path.join(cfg.output_dir, "mrt_config.json"), "w", encoding="utf-8") as f:
            json.dump({k: v for k, v in vars(cfg).items() if not k.startswith("_")}, f, indent=2)

        if cfg.run_initial_eval:
            print("\n🔎 INITIAL EVALUATION (before any new optimizer step)")
            metrics = self.evaluate(epoch=-1, decode_strategy=cfg.eval_decode_strategy, save_predictions=cfg.save_eval_predictions, eval_tag="initial")
            self._maybe_save(metrics, epoch=-1)

        self.policy_model.train()

        for epoch in range(cfg.num_epochs):
            indices = list(range(len(self.train_dataset)))
            random.shuffle(indices)
            epoch_stats: Dict[str, List[float]] = {
                "mrt_loss": [],
                "ce_loss": [],
                "total_loss": [],
                "mean_reward": [],
                "chosen_reward": [],
                "candidate_entropy": [],
                "mean_group_size": [],
                "mean_len_ratio": [],
                "grad_norm": [],
                "lr": [],
            }

            batch_records: List[Dict[str, Any]] = []
            accumulated = 0
            optimizer.zero_grad()
            train_pbar = tqdm(total=total_steps, desc=f"Train e{epoch + 1}/{cfg.num_epochs}", leave=False)

            for i, idx in enumerate(indices):
                batch_records.append(self.train_dataset[idx])
                if len(batch_records) < cfg.mrt_batch_size and i < len(indices) - 1:
                    continue

                batch_loss, batch_metrics = self._compute_mrt_loss_for_records(batch_records)
                (batch_loss / cfg.gradient_accumulation).backward()
                accumulated += 1

                if accumulated >= cfg.gradient_accumulation:
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), cfg.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    accumulated = 0
                    self.global_step += 1

                    for key, value in batch_metrics.items():
                        epoch_stats.setdefault(key, []).append(float(value))
                    epoch_stats["grad_norm"].append(float(grad_norm))
                    current_lr = float(scheduler.get_last_lr()[0]) if scheduler is not None else cfg.learning_rate
                    epoch_stats["lr"].append(current_lr)

                    train_row = {
                        "step": int(self.global_step),
                        "epoch": float(epoch + 1),
                        "mrt_loss": float(np.mean(epoch_stats["mrt_loss"][-5:])),
                        "ce_loss": float(np.mean(epoch_stats["ce_loss"][-5:])),
                        "total_loss": float(np.mean(epoch_stats["total_loss"][-5:])),
                        "mean_reward": float(np.mean(epoch_stats["mean_reward"][-5:])),
                        "chosen_reward": float(np.mean(epoch_stats["chosen_reward"][-5:])),
                        "candidate_entropy": float(np.mean(epoch_stats["candidate_entropy"][-5:])),
                        "mean_group_size": float(np.mean(epoch_stats["mean_group_size"][-5:])),
                        "mean_len_ratio": float(np.mean(epoch_stats["mean_len_ratio"][-5:])),
                        "grad_norm": float(np.mean(epoch_stats["grad_norm"][-5:])),
                        "lr": current_lr,
                    }
                    self._record_train_history(train_row)

                    train_pbar.n = min(self.global_step, train_pbar.total)
                    train_pbar.set_postfix_str(
                        f"mrt={train_row['mrt_loss']:.3f} ce={train_row['ce_loss']:.3f} "
                        f"R={train_row['mean_reward']:.2f} len={train_row['mean_len_ratio']:.2f}"
                    )
                    train_pbar.refresh()

                    if self.global_step % cfg.logging_steps == 0:
                        print(
                            f"[Train] epoch={epoch + 1} step={self.global_step} "
                            f"mrt={train_row['mrt_loss']:.3f} ce={train_row['ce_loss']:.3f} "
                            f"reward={train_row['mean_reward']:.2f} "
                            f"chosen={train_row['chosen_reward']:.2f} "
                            f"entropy={train_row['candidate_entropy']:.2f} "
                            f"lenr={train_row['mean_len_ratio']:.2f}"
                        )

                    if cfg.eval_every_n_steps > 0 and self.global_step % cfg.eval_every_n_steps == 0:
                        metrics = self.evaluate(epoch=epoch, decode_strategy=cfg.eval_decode_strategy, save_predictions=cfg.save_eval_predictions, eval_tag="step")
                        self._maybe_save(metrics, epoch=epoch)
                        if self.should_stop():
                            train_pbar.close()
                            print("\nEarly stopping triggered.")
                            return

                    if cfg.max_train_steps > 0 and self.global_step >= cfg.max_train_steps:
                        train_pbar.close()
                        print(f"\n[max_train_steps={cfg.max_train_steps} reached]")
                        return

                batch_records = []

            if accumulated > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                self.global_step += 1
                epoch_stats["grad_norm"].append(float(grad_norm))

            train_pbar.close()

            print(
                f"\n[Epoch {epoch + 1}] "
                f"mrt={np.mean(epoch_stats['mrt_loss']) if epoch_stats['mrt_loss'] else 0.0:.3f} "
                f"ce={np.mean(epoch_stats['ce_loss']) if epoch_stats['ce_loss'] else 0.0:.3f} "
                f"reward={np.mean(epoch_stats['mean_reward']) if epoch_stats['mean_reward'] else 0.0:.2f} "
                f"lenr={np.mean(epoch_stats['mean_len_ratio']) if epoch_stats['mean_len_ratio'] else 0.0:.2f}"
            )

            metrics = self.evaluate(epoch=epoch, decode_strategy=cfg.eval_decode_strategy, save_predictions=cfg.save_eval_predictions, eval_tag="epoch")
            self._maybe_save(metrics, epoch=epoch)
            if self.should_stop():
                print("\nEarly stopping triggered.")
                break

        print("\nTraining finished.")


def main() -> None:
    trainer = MRTTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
