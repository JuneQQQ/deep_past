#!/usr/bin/env python3
"""
================================================================================
Unified Preference Optimization Training: DPO / GRPO / MRT
================================================================================
Multi-GPU support via `accelerate`. Launch with:
  accelerate launch --num_processes=4 script/train_preference.py --algo dpo
  accelerate launch --num_processes=4 script/train_preference.py --algo grpo
  accelerate launch --num_processes=4 script/train_preference.py --algo mrt

Single GPU:
  python script/train_preference.py --algo dpo
================================================================================
"""
from __future__ import annotations

# ── CUDA_HOME workaround ──
import os as _os, tempfile as _tf, stat as _st
if not _os.environ.get("CUDA_HOME"):
    _fake = _tf.mkdtemp(prefix="fake_cuda_")
    _bin = _os.path.join(_fake, "bin"); _os.makedirs(_bin, exist_ok=True)
    _nvcc = _os.path.join(_bin, "nvcc")
    if not _os.path.exists(_nvcc):
        with open(_nvcc, "w") as _f:
            _f.write("#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n")
        _os.chmod(_nvcc, _st.S_IRWXU)
    _os.environ["CUDA_HOME"] = _fake

import argparse
import gc
import json
import math
import os
import re
import sys
import shutil
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

from accelerate import Accelerator
from accelerate.utils import set_seed
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
)

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from prepare_data import load_lexicon, build_onomasticon, postprocess_output


# ============================================================================
# CLI
# ============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--algo", type=str, default="dpo", choices=["dpo", "grpo", "mrt"],
                    help="Preference optimization algorithm")
parser.add_argument("--config", type=str, default=None, help="JSON config override file")
args, _ = parser.parse_known_args()


# ============================================================================
# Config
# ============================================================================
class Config:
    def __init__(self):
        self.root_dir = "/data/lsb/deep_past"
        self.data_dir = os.path.join(self.root_dir, "data")
        self.algo = args.algo

        self._timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = os.path.join(self.root_dir, "output", f"{self.algo}_{self._timestamp}")

        # Model
        self.model_name = "/data/lsb/deep_past/output/model_20260321_171137/checkpoint-20800"
        self.max_input_length = 512
        self.max_target_length = 512

        # Data
        self.train_csv = os.path.join(self.data_dir, "qwen_sentence_aligned_clean.csv")
        self.lexicon_csv = os.path.join(self.data_dir, "OA_Lexicon_eBL.csv")
        self.test_size = 0.1
        self.random_seed = 42
        self.fold_group_col = "oare_id"

        # Training
        self.num_epochs = 10
        self.batch_size = 4
        self.gradient_accumulation = 4  # effective batch = batch_size * grad_accum * num_gpus
        self.learning_rate = 5e-6
        self.warmup_steps = 200
        self.weight_decay = 0.01
        self.max_grad_norm = 1.0
        self.dtype = "bf16"
        self.gradient_checkpointing = True

        # DPO-specific
        self.dpo_beta = 0.3

        # GRPO-specific
        self.grpo_num_candidates = 8
        self.grpo_temperature = 0.8
        self.grpo_kl_coef = 0.1

        # MRT-specific
        self.mrt_num_samples = 8
        self.mrt_temperature = 1.0
        self.mrt_alpha = 0.005  # smoothing

        # Eval / Save
        self.eval_steps = 100
        self.save_total_limit = 2
        self.logging_steps = 10
        self.early_stopping_patience = 5

        # Generation (for eval)
        self.num_beams = 4
        self.length_penalty = 1.3

    @property
    def use_bf16(self):
        return self.dtype == "bf16"

    def to_dict(self):
        return {k: (str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v)
                for k, v in self.__dict__.items()}


config = Config()
if args.config and os.path.exists(args.config):
    with open(args.config) as f:
        overrides = json.load(f)
    for k, v in overrides.items():
        if hasattr(config, k):
            setattr(config, k, v)
    print(f"📋 Config overrides from {args.config}: {list(overrides.keys())}")


# ============================================================================
# Accelerator
# ============================================================================
accelerator = Accelerator(
    mixed_precision="bf16" if config.use_bf16 else "no",
    gradient_accumulation_steps=config.gradient_accumulation,
    log_with=None,
)
set_seed(config.random_seed)
is_main = accelerator.is_main_process

if is_main:
    os.makedirs(config.output_dir, exist_ok=True)
    print("=" * 60)
    print(f"🚀 {config.algo.upper()} PREFERENCE OPTIMIZATION")
    print("=" * 60)
    print(f"Devices: {accelerator.num_processes}")
    print(f"Mixed precision: {accelerator.mixed_precision}")
    for k, v in sorted(config.to_dict().items()):
        print(f"   {k}: {v}")


# ============================================================================
# Data
# ============================================================================
def load_preference_data(config):
    df = pd.read_csv(config.train_csv)

    # Normalize columns
    if "prompt" not in df.columns:
        if "input_text" in df.columns:
            df["prompt"] = df["input_text"]
        elif "transliteration" in df.columns:
            df["prompt"] = df["transliteration"].apply(lambda x: f"OAOI {x}" if pd.notna(x) else "")
        else:
            raise ValueError(f"No prompt column found. Columns: {df.columns.tolist()}")

    if "chosen" not in df.columns:
        if "translation" in df.columns:
            df["chosen"] = df["translation"]
        elif "target_text" in df.columns:
            df["chosen"] = df["target_text"]

    if "rejected" not in df.columns:
        # Fallback: cyclic shift
        chosen = df["chosen"].tolist()
        df["rejected"] = chosen[1:] + chosen[:1]

    # Filter
    df = df.dropna(subset=["prompt", "chosen", "rejected"])
    df = df[df["chosen"].str.strip().str.len() > 0].reset_index(drop=True)

    # Remove chosen == rejected
    mask = df["chosen"].str.strip() != df["rejected"].str.strip()
    df = df[mask].reset_index(drop=True)

    # Split
    if config.fold_group_col in df.columns:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
        train_idx, eval_idx = next(gss.split(df, groups=df[config.fold_group_col]))
        train_df = df.iloc[train_idx].reset_index(drop=True)
        eval_df = df.iloc[eval_idx].reset_index(drop=True)
    else:
        from sklearn.model_selection import train_test_split
        train_df, eval_df = train_test_split(df, test_size=config.test_size, random_state=config.random_seed)
        train_df = train_df.reset_index(drop=True)
        eval_df = eval_df.reset_index(drop=True)

    return train_df, eval_df


class PreferenceDataset(TorchDataset):
    def __init__(self, df: pd.DataFrame):
        self.prompts = df["prompt"].tolist()
        self.chosen = df["chosen"].tolist()
        self.rejected = df["rejected"].tolist()

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "chosen": self.chosen[idx], "rejected": self.rejected[idx]}


def collate_fn(batch, tokenizer, max_src_len, max_tgt_len):
    prompts = [item["prompt"] for item in batch]
    chosen = [item["chosen"] for item in batch]
    rejected = [item["rejected"] for item in batch]

    enc = tokenizer(prompts, padding=True, truncation=True, max_length=max_src_len, return_tensors="pt")
    cho = tokenizer(chosen, padding=True, truncation=True, max_length=max_tgt_len, return_tensors="pt")
    rej = tokenizer(rejected, padding=True, truncation=True, max_length=max_tgt_len, return_tensors="pt")

    cho_ids = cho["input_ids"].clone()
    cho_ids[cho_ids == tokenizer.pad_token_id] = -100
    rej_ids = rej["input_ids"].clone()
    rej_ids[rej_ids == tokenizer.pad_token_id] = -100

    return {
        "input_ids": enc["input_ids"],
        "attention_mask": enc["attention_mask"],
        "chosen_labels": cho_ids,
        "rejected_labels": rej_ids,
    }


# ============================================================================
# Core functions
# ============================================================================
def get_seq2seq_log_probs(model, input_ids, attention_mask, labels):
    """Compute per-sample sum of log probs for encoder-decoder."""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
    logits = outputs.logits
    valid_mask = (labels != -100).float()
    labels_safe = labels.clamp(min=0)
    log_probs = F.log_softmax(logits, dim=-1)
    token_lp = log_probs.gather(2, labels_safe.unsqueeze(-1)).squeeze(-1)
    return (token_lp * valid_mask).sum(dim=1)


def dpo_loss(policy_chosen_lp, policy_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta):
    logits = beta * ((policy_chosen_lp - policy_rejected_lp) - (ref_chosen_lp - ref_rejected_lp))
    loss = -F.logsigmoid(logits).mean()
    acc = (logits > 0).float().mean()
    margin = (policy_chosen_lp - policy_rejected_lp).mean()
    return loss, acc, margin


# ============================================================================
# Training
# ============================================================================
def main():
    train_df, eval_df = load_preference_data(config)
    if is_main:
        print(f"\n📂 Data: train={len(train_df)}, eval={len(eval_df)}")

    # Model + tokenizer
    model_path = config.model_name
    local = os.path.isdir(model_path)
    load_dtype = torch.bfloat16 if config.use_bf16 else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=local)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    policy_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=load_dtype, local_files_only=local, low_cpu_mem_usage=True,
    )
    if config.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()

    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_path, torch_dtype=load_dtype, local_files_only=local, low_cpu_mem_usage=True,
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    if is_main:
        total = sum(p.numel() for p in policy_model.parameters())
        trainable = sum(p.numel() for p in policy_model.parameters() if p.requires_grad)
        print(f"   Policy model: {trainable:,}/{total:,} trainable params")

    # Datasets + loaders
    train_ds = PreferenceDataset(train_df)
    eval_ds = PreferenceDataset(eval_df)

    _collate = lambda b: collate_fn(b, tokenizer, config.max_input_length, config.max_target_length)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, collate_fn=_collate, num_workers=2)
    eval_loader = DataLoader(eval_ds, batch_size=config.batch_size * 2, shuffle=False, collate_fn=_collate, num_workers=2)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    num_steps = len(train_loader) * config.num_epochs // config.gradient_accumulation
    warmup = min(config.warmup_steps, num_steps // 2)  # Cap warmup at 50% of total steps
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup, num_steps)

    # Prepare with accelerate
    policy_model, ref_model, optimizer, train_loader, eval_loader, scheduler = accelerator.prepare(
        policy_model, ref_model, optimizer, train_loader, eval_loader, scheduler
    )

    # Training loop
    global_step = 0
    best_metric = -float("inf")
    patience_counter = 0

    if is_main:
        print(f"\n{'='*60}")
        print(f"🚀 STARTING {config.algo.upper()} TRAINING")
        print(f"   Steps/epoch: {len(train_loader)}, Total epochs: {config.num_epochs}")
        print(f"   Effective batch: {config.batch_size * config.gradient_accumulation * accelerator.num_processes}")
        print(f"{'='*60}\n")

    for epoch in range(config.num_epochs):
        policy_model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", disable=not is_main)
        for batch in pbar:
            with accelerator.accumulate(policy_model):
                input_ids = batch["input_ids"]
                attn_mask = batch["attention_mask"]
                cho_labels = batch["chosen_labels"]
                rej_labels = batch["rejected_labels"]

                # Policy log probs
                p_cho = get_seq2seq_log_probs(policy_model, input_ids, attn_mask, cho_labels)
                p_rej = get_seq2seq_log_probs(policy_model, input_ids, attn_mask, rej_labels)

                # Reference log probs
                with torch.no_grad():
                    r_cho = get_seq2seq_log_probs(ref_model, input_ids, attn_mask, cho_labels)
                    r_rej = get_seq2seq_log_probs(ref_model, input_ids, attn_mask, rej_labels)

                if config.algo == "dpo":
                    loss, acc, margin = dpo_loss(p_cho, p_rej, r_cho, r_rej, config.dpo_beta)
                elif config.algo == "grpo":
                    # GRPO: group relative policy optimization
                    # reward = chosen_logp - rejected_logp (normalized within group)
                    reward_diff = (p_cho - p_rej) - (r_cho - r_rej)
                    loss = -reward_diff.mean()
                    kl = (p_cho - r_cho).mean()
                    loss = loss + config.grpo_kl_coef * kl.abs()
                    acc = (reward_diff > 0).float().mean()
                    margin = reward_diff.mean()
                elif config.algo == "mrt":
                    # MRT: minimum risk training (simplified)
                    # Use rejection as negative, chosen as positive
                    risk = F.softmax(-config.mrt_alpha * p_cho, dim=0)
                    loss = (risk * (-p_cho)).sum()
                    acc = (p_cho > p_rej).float().mean()
                    margin = (p_cho - p_rej).mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(policy_model.parameters(), config.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item()
            epoch_acc += acc.item()
            n_batches += 1
            global_step += 1

            if is_main:
                pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc.item():.2%}", margin=f"{margin.item():.3f}")

            if global_step % config.logging_steps == 0 and is_main:
                print(f"   Step {global_step}: loss={loss.item():.4f}, acc={acc.item():.2%}, lr={scheduler.get_last_lr()[0]:.2e}")

            # Eval
            if global_step % config.eval_steps == 0:
                policy_model.eval()
                eval_loss_sum = 0.0
                eval_acc_sum = 0.0
                eval_n = 0
                with torch.no_grad():
                    for eb in eval_loader:
                        ep_cho = get_seq2seq_log_probs(policy_model, eb["input_ids"], eb["attention_mask"], eb["chosen_labels"])
                        ep_rej = get_seq2seq_log_probs(policy_model, eb["input_ids"], eb["attention_mask"], eb["rejected_labels"])
                        er_cho = get_seq2seq_log_probs(ref_model, eb["input_ids"], eb["attention_mask"], eb["chosen_labels"])
                        er_rej = get_seq2seq_log_probs(ref_model, eb["input_ids"], eb["attention_mask"], eb["rejected_labels"])
                        el, ea, _ = dpo_loss(ep_cho, ep_rej, er_cho, er_rej, config.dpo_beta)
                        eval_loss_sum += el.item()
                        eval_acc_sum += ea.item()
                        eval_n += 1

                avg_eval_acc = eval_acc_sum / max(eval_n, 1)
                if is_main:
                    print(f"   [Eval] step={global_step} loss={eval_loss_sum/max(eval_n,1):.4f} acc={avg_eval_acc:.2%}")

                # Save best
                if avg_eval_acc > best_metric:
                    best_metric = avg_eval_acc
                    patience_counter = 0
                    if is_main:
                        save_dir = os.path.join(config.output_dir, f"checkpoint-{global_step}")
                        os.makedirs(save_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(policy_model)
                        unwrapped.save_pretrained(save_dir)
                        tokenizer.save_pretrained(save_dir)
                        # best_checkpoint symlink
                        link = os.path.join(config.output_dir, "best_checkpoint")
                        if os.path.islink(link):
                            os.unlink(link)
                        elif os.path.exists(link):
                            shutil.rmtree(link)
                        os.symlink(save_dir, link)
                        print(f"   ⭐ New best (acc={avg_eval_acc:.2%}) → {save_dir}")

                        # Limit checkpoints
                        import glob
                        ckpts = sorted(glob.glob(os.path.join(config.output_dir, "checkpoint-*")), key=os.path.getmtime)
                        for old in ckpts[:-config.save_total_limit]:
                            shutil.rmtree(old, ignore_errors=True)
                else:
                    patience_counter += 1
                    if patience_counter >= config.early_stopping_patience:
                        if is_main:
                            print(f"   🛑 Early stopping (patience={patience_counter})")
                        break

                policy_model.train()

        if is_main:
            avg_loss = epoch_loss / max(n_batches, 1)
            avg_acc = epoch_acc / max(n_batches, 1)
            print(f"\n   Epoch {epoch+1} done: avg_loss={avg_loss:.4f}, avg_acc={avg_acc:.2%}\n")

        if patience_counter >= config.early_stopping_patience:
            break

    # Save config
    if is_main:
        with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
            json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n✅ {config.algo.upper()} training completed. Output: {config.output_dir}")


if __name__ == "__main__":
    main()
