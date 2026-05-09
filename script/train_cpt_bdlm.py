#!/usr/bin/env python3
"""
Train on a precomputed BDLM CPT dataset.

This script intentionally does not build data on the fly.
Run build_cpt_bdlm_data.py first, then point this script at the saved dataset.
"""

from __future__ import annotations

# ── CUDA_HOME workaround: deepspeed import fails without nvcc ──
import os as _os, subprocess as _sp, tempfile as _tf, stat as _st
if not _os.environ.get("CUDA_HOME"):
    _fake = _tf.mkdtemp(prefix="fake_cuda_")
    _bin = _os.path.join(_fake, "bin")
    _os.makedirs(_bin, exist_ok=True)
    _nvcc = _os.path.join(_bin, "nvcc")
    if not _os.path.exists(_nvcc):
        with open(_nvcc, "w") as _f:
            _f.write("#!/bin/sh\necho 'Cuda compilation tools, release 12.8, V12.8.93'\n")
        _os.chmod(_nvcc, _st.S_IRWXU)
    _os.environ["CUDA_HOME"] = _fake

import argparse
import gc
import glob
import inspect
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import torch
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)


parser = argparse.ArgumentParser()
parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for output dir")
parser.add_argument("--dataset-dir", type=str, default=None, help="Path to prebuilt BDLM DatasetDict")
args, _ = parser.parse_known_args()


os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

gc.collect()
torch.cuda.empty_cache()


def _get_env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


RANK = _get_env_int("RANK", 0)
LOCAL_RANK = _get_env_int("LOCAL_RANK", 0)
WORLD_SIZE = _get_env_int("WORLD_SIZE", 1)
IS_DISTRIBUTED = WORLD_SIZE > 1

if torch.cuda.is_available() and IS_DISTRIBUTED:
    torch.cuda.set_device(LOCAL_RANK)


class Config:
    def __init__(self):
        self.root_dir = "/data/lsb/deep_past"
        self.data_dir = os.path.join(self.root_dir, "data")

        timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timestamp = timestamp
        self.output_dir = os.path.join(self.root_dir, "output", f"cpt_bdlm_{timestamp}")

        self.prebuilt_dataset_dir = args.dataset_dir or os.path.join(self.data_dir, "cpt_bdlm_dataset")
        self.max_train_samples = None
        self.max_eval_samples = None
        self.random_seed = 42

        # self.model_name = "google/byt5-large"
        self.model_name = "/data/lsb/deep_past/output/cpt_bdlm_20260319_180315/best_checkpoint"

        self.max_source_length = 512
        self.max_target_length = 512

        self.num_epochs = 50
        self.batch_size = 12
        self.eval_batch_size = 32
        self.gradient_accumulation = 1
        self.learning_rate = 1e-4  # 提升学习率：2e-7 太小导致 loss 降不下去
        self.warmup_steps = 1000
        self.weight_decay = 0.01
        self.lr_scheduler_type = "cosine"
        # self.optim = "adamw_bnb_8bit" if torch.cuda.is_available() else "adamw_torch"
        self.optim = "adamw_torch"
        self.gradient_checkpointing = False
        self.dataloader_num_workers = 4
        self.report_to = "none"
        self.ddp_find_unused_parameters = False
        self.save_on_each_node = False
        self.freeze_decoder = False  # Keep decoder trainable (successful old CPT did not freeze)

        self.dtype = "bf16"

        self.logging_steps = 10
        self.eval_steps = 200
        self.save_steps = 200
        self.save_total_limit = 5

        self.fsdp = ""
        self.fsdp_config = {
            "fsdp_transformer_layer_cls_to_wrap": ["T5Block"],
            "fsdp_backward_prefetch": "backward_pre",
            "fsdp_forward_prefetch": False,
            "fsdp_use_orig_params": True,
            "fsdp_cpu_ram_efficient_loading": True,
            "fsdp_sync_module_states": True,
            "fsdp_state_dict_type": "FULL_STATE_DICT",
            "fsdp_offload_params": False,
            "fsdp_sharding_strategy": "FULL_SHARD",
            "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
            "activation_checkpointing": True,
        }
        if self.fsdp:
            self.gradient_checkpointing = False

    @property
    def use_bf16(self) -> bool:
        return self.dtype == "bf16"

    def to_dict(self) -> Dict:
        return {
            k: (str(v) if not isinstance(v, (int, float, bool, str, type(None), list, dict)) else v)
            for k, v in self.__dict__.items()
        }


config = Config()


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


def load_model_for_cpt(model_id: str, local_files_only: bool, use_bf16: bool):
    load_dtype = torch.bfloat16 if (use_bf16 and torch.cuda.is_available()) else torch.float32
    from_pretrained_sig = inspect.signature(AutoModelForSeq2SeqLM.from_pretrained).parameters
    dtype_param_name = "dtype" if "dtype" in from_pretrained_sig else "torch_dtype"
    model_load_kwargs = {
        dtype_param_name: load_dtype,
        "local_files_only": local_files_only,
    }
    if "low_cpu_mem_usage" in from_pretrained_sig:
        model_load_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id, **model_load_kwargs)
    model.train()
    return model


def maybe_distributed_barrier() -> None:
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


def load_prebuilt_datasets(dataset_dir: str, cfg: Config):
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(
            f"Prebuilt dataset not found: {dataset_dir}\n"
            "Run script/build_cpt_bdlm_data.py first."
        )

    loaded = load_from_disk(dataset_dir)
    if not isinstance(loaded, DatasetDict):
        raise TypeError(f"Expected DatasetDict at {dataset_dir}, got {type(loaded).__name__}")
    if "train" not in loaded or "eval" not in loaded:
        raise KeyError(f"DatasetDict at {dataset_dir} must contain 'train' and 'eval' splits")

    train_dataset = loaded["train"]
    eval_dataset = loaded["eval"]

    if cfg.max_train_samples is not None:
        train_dataset = train_dataset.select(range(min(len(train_dataset), cfg.max_train_samples)))
    if cfg.max_eval_samples is not None:
        eval_dataset = eval_dataset.select(range(min(len(eval_dataset), cfg.max_eval_samples)))

    keep_columns = ["input_ids", "labels"]
    train_dataset = train_dataset.select_columns(keep_columns)
    eval_dataset = eval_dataset.select_columns(keep_columns)

    summary_path = os.path.join(dataset_dir, "builder_summary.json")
    summary = None
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            summary = json.load(fh)

    return train_dataset, eval_dataset, summary


@dataclass
class PrecomputedSeq2SeqCollator:
    tokenizer: AutoTokenizer
    max_source_length: int
    max_target_length: int

    def __post_init__(self):
        self.pad_token_id = self.tokenizer.pad_token_id
        if self.pad_token_id is None:
            raise ValueError("Tokenizer pad_token_id is required for batching.")

    def _pad(self, sequences: List[List[int]], pad_id: int, max_len_cap: int) -> Tuple[List[List[int]], List[List[int]]]:
        max_len = min(max_len_cap, max(len(seq) for seq in sequences))
        padded: List[List[int]] = []
        masks: List[List[int]] = []
        for seq in sequences:
            seq = seq[:max_len]
            pad_len = max_len - len(seq)
            padded.append(seq + [pad_id] * pad_len)
            masks.append([1] * len(seq) + [0] * pad_len)
        return padded, masks

    def __call__(self, features: List[Dict]) -> Dict[str, torch.Tensor]:
        input_ids = [[int(t) for t in feature["input_ids"]] for feature in features]
        labels = [[int(t) for t in feature["labels"]] for feature in features]

        batch_input_ids, attention_mask = self._pad(input_ids, int(self.pad_token_id), self.max_source_length)
        batch_labels, _ = self._pad(labels, -100, self.max_target_length)

        return {
            "input_ids": torch.tensor(batch_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(batch_labels, dtype=torch.long),
        }


def main() -> None:
    print("=" * 60)
    print("BDLM CPT TRAINING (Precomputed Dataset)")
    print("=" * 60)
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"GPU[{idx}]: {torch.cuda.get_device_name(idx)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(idx).total_memory / 1e9:.1f} GB")
    if IS_DISTRIBUTED:
        print(f"Distributed: rank={RANK} local_rank={LOCAL_RANK} world_size={WORLD_SIZE}")

    set_seed(config.random_seed + (RANK if IS_DISTRIBUTED else 0))
    os.makedirs(config.output_dir, exist_ok=True)

    print("\n📦 Loading prebuilt dataset...")
    train_dataset, eval_dataset, dataset_summary = load_prebuilt_datasets(config.prebuilt_dataset_dir, config)
    print(f"   Dataset dir: {config.prebuilt_dataset_dir}")
    print(f"   Train examples: {len(train_dataset)}")
    print(f"   Eval examples: {len(eval_dataset)}")
    if dataset_summary is not None:
        print(f"   Builder dict_size: {dataset_summary.get('dict_size')}")
        train_meta = dataset_summary.get("splits", {}).get("train", {})
        eval_meta = dataset_summary.get("splits", {}).get("eval", {})
        if train_meta:
            print(f"   Train annotated ratio: {train_meta.get('annotated_example_ratio')}")
        if eval_meta:
            print(f"   Eval annotated ratio: {eval_meta.get('annotated_example_ratio')}")

    print("\n🔤 Loading tokenizer...")
    resolved_model_path, local_files_only = resolve_model_path(config.model_name)
    print(f"   Model source: {resolved_model_path}")
    print(f"   local_files_only={local_files_only}")

    tokenizer = AutoTokenizer.from_pretrained(resolved_model_path, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "<pad>"})

    print("\n🤖 Loading model...")
    model = load_model_for_cpt(resolved_model_path, local_files_only, config.use_bf16)

    # Fix tied weights (old checkpoints save 4 independent copies)
    if getattr(model.config, "tie_word_embeddings", True):
        _shared = model.shared
        _needs = (
            model.encoder.embed_tokens.weight.data_ptr() != _shared.weight.data_ptr()
            or model.decoder.embed_tokens.weight.data_ptr() != _shared.weight.data_ptr()
            or model.lm_head.weight.data_ptr() != _shared.weight.data_ptr()
        )
        if _needs:
            model.encoder.embed_tokens = _shared
            model.decoder.embed_tokens = _shared
            model.lm_head.weight = _shared.weight
            print("   🔧 Re-tied shared weights")

    model_vocab_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > model_vocab_size:
        model.resize_token_embeddings(len(tokenizer))
        print(f"   Resized embeddings: {model_vocab_size} -> {len(tokenizer)}")
    print(f"   Model dtype: {next(model.parameters()).dtype}")

    # Freeze decoder to prevent span corruption format from polluting translation output
    if config.freeze_decoder:
        _frozen = 0
        for p in model.decoder.parameters():
            p.requires_grad = False
            _frozen += p.numel()
        # Also freeze lm_head (decoder output projection)
        for p in model.lm_head.parameters():
            p.requires_grad = False
            _frozen += p.numel()
        _total = sum(p.numel() for p in model.parameters())
        _trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   🧊 Decoder frozen: {_frozen:,} params frozen, {_trainable:,}/{_total:,} trainable")

    print("\n🧩 Preparing padding-only collator...")
    data_collator = PrecomputedSeq2SeqCollator(
        tokenizer=tokenizer,
        max_source_length=config.max_source_length,
        max_target_length=config.max_target_length,
    )

    print("\n⚙️ Building training arguments...")
    effective_batch = config.batch_size * config.gradient_accumulation * max(1, WORLD_SIZE)
    print(f"   Effective batch: {effective_batch}")

    _max_steps = int(os.environ.get("CPT_MAX_STEPS", "0"))
    _args_kwargs = dict(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs if _max_steps <= 0 else 100,
        max_steps=_max_steps if _max_steps > 0 else -1,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        weight_decay=config.weight_decay,
        optim=config.optim,
        bf16=config.use_bf16,
        fp16=False,
        gradient_checkpointing=config.gradient_checkpointing,
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=config.logging_steps,
        logging_dir=os.path.join(config.output_dir, "logs"),
        report_to=config.report_to,
        remove_unused_columns=False,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_prefetch_factor=2 if config.dataloader_num_workers > 0 else None,
        dataloader_persistent_workers=True if config.dataloader_num_workers > 0 else False,
        ddp_find_unused_parameters=config.ddp_find_unused_parameters,
        save_on_each_node=config.save_on_each_node,
        predict_with_generate=False,
    )

    if config.fsdp:
        _args_kwargs["fsdp"] = config.fsdp
        _args_kwargs["fsdp_config"] = config.fsdp_config
        _args_kwargs["gradient_checkpointing"] = False

    _seq2seq_params = inspect.signature(Seq2SeqTrainingArguments.__init__).parameters
    for key in [key for key in _args_kwargs if key not in _seq2seq_params]:
        _args_kwargs.pop(key)

    training_args = Seq2SeqTrainingArguments(**_args_kwargs)

    from transformers import EarlyStoppingCallback
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )
    trainer_params = inspect.signature(Seq2SeqTrainer.__init__).parameters
    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Seq2SeqTrainer(**trainer_kwargs)

    # Sanity probes: translation + dictionary alignment monitoring
    _TRANSLATION_PROBES = [
        # Short seal/witness
        ("OAOI KIŠIB a-šur-na-da DUMU i-dí-a-bu-um", "Seal of Aššur-nādā son of Iddin-abum"),
        # Letter opening
        ("OAOI um-ma šu-IŠTAR-ma a-na en-nam-a-šur qí-bi4-ma", "From Šu-Ištar to Ennam-Aššur:"),
        # Silver amount
        ("OAOI 2 ma-na KÙ.BABBAR ṣa-ru-pá-am", "2 minas of refined silver"),
        # Longer with gap
        ("OAOD a-na a-lá-hi-im qí-bi4-ma um-ma i-dí-a-bu-um-ma <gap> KÙ.BABBAR",
         "To Ali-ahum from Iddin-abum: <gap> silver"),
        # Textile/donkey
        ("OAOI 5 TÚG.HI.A ù 1 ANŠE", "5 textiles and 1 donkey"),
        # Witness list
        ("OAOI IGI a-šur-ma-lik IGI ì-lí-dan", "Witnessed by Aššur-malik, by Ilī-dan"),
        # Legal
        ("OAOI šu-ma lá iš-qú-ul ṣí-ib-tám ú-ṣa-áb", "If he does not pay, he will add interest"),
        # Date
        ("OAOI ITU.KAM ša ke-na-tim li-mu-um e-na-sú-in", "Month Kenātim, eponymy Enna-Suen"),
    ]
    # Dictionary probes: words that should map to specific English via CPT dict
    _DICT_PROBES = [
        ("kù.babbar", "silver"),
        ("dumu", "son"),
        ("gín", "shekel"),
        ("ma-na", "mina"),
        ("tùg", "textile"),
        ("anše", "donkey"),
        ("é.gal", "palace"),
        ("kišib", "seal"),
    ]
    from transformers import TrainerCallback
    class TranslationSanityCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, **kwargs):
            if not state.is_world_process_zero:
                return
            _model = kwargs.get("model", trainer.model)
            _model.eval()
            step = state.global_step
            print(f"\n   🔍 Sanity probes (step {step}):")

            # 1. Translation probes
            n_partial = 0
            for src, ref in _TRANSLATION_PROBES:
                try:
                    _inp = tokenizer(src, return_tensors="pt", truncation=True, max_length=config.max_source_length)
                    _inp = {k: v.to(_model.device) for k, v in _inp.items()}
                    with torch.no_grad():
                        _out = _model.generate(**_inp, max_new_tokens=config.max_target_length, max_length=None, num_beams=1)
                    _dec = tokenizer.decode(_out[0], skip_special_tokens=True)
                    # Check overlap with reference
                    ref_words = set(ref.lower().split())
                    out_words = set(_dec.lower().split())
                    overlap = len(ref_words & out_words) / max(len(ref_words), 1)
                    marker = "✅" if overlap > 0.5 else "⚠️" if overlap > 0.2 else "❌"
                    if overlap > 0.2:
                        n_partial += 1
                    print(f"     {marker} [{overlap:.0%}] {src[:50]}")
                    print(f"        OUT: {_dec[:70]}")
                    print(f"        REF: {ref[:70]}")
                except Exception as _e:
                    print(f"     ⚠️ Probe failed: {_e}")

            print(f"   📊 Translation probe score: {n_partial}/{len(_TRANSLATION_PROBES)} partial+ matches")

            # 2. Dictionary probes (check if span corruption learned word alignments)
            n_dict_hit = 0
            for akk_form, eng_expected in _DICT_PROBES:
                try:
                    # Formulate realistic dictionary input with prefix
                    probe_src = f"OAOI {akk_form}"
                    _inp = tokenizer(probe_src, return_tensors="pt", truncation=True, max_length=config.max_source_length)
                    _inp = {k: v.to(_model.device) for k, v in _inp.items()}
                    with torch.no_grad():
                        _out = _model.generate(**_inp, max_new_tokens=config.max_target_length, max_length=None, num_beams=1)
                    _dec = tokenizer.decode(_out[0], skip_special_tokens=True).lower()
                    hit = eng_expected.lower() in _dec
                    if hit:
                        n_dict_hit += 1
                        print(f"     ✅ [HIT] {probe_src} -> {_dec[:50]} (expected: {eng_expected})")
                    else:
                        print(f"     ❌ [MISS] {probe_src} -> {_dec[:50]} (expected: {eng_expected})")
                except Exception as _e:
                    print(f"     ⚠️ Dict Probe failed for {akk_form}: {_e}")
            print(f"   📚 Dictionary probe score: {n_dict_hit}/{len(_DICT_PROBES)} hits")

    trainer.add_callback(TranslationSanityCallback())

    print("\n" + "=" * 60)
    print("🚀 STARTING BDLM CPT TRAINING")
    print("=" * 60)

    train_result = trainer.train()
    trainer.log_metrics("train", train_result.metrics)
    if trainer.is_world_process_zero():
        trainer.save_metrics("train", train_result.metrics)
    trainer.save_state()

    print("\n📈 Final evaluation...")
    eval_metrics = trainer.evaluate()
    trainer.log_metrics("eval", eval_metrics)
    if trainer.is_world_process_zero():
        trainer.save_metrics("eval", eval_metrics)

    if trainer.is_world_process_zero():
        with open(os.path.join(config.output_dir, "training_config.json"), "w", encoding="utf-8") as fh:
            json.dump(config.to_dict(), fh, indent=4, ensure_ascii=False)

        run_summary = {
            "resolved_model_path": resolved_model_path,
            "local_files_only": local_files_only,
            "dataset_dir": config.prebuilt_dataset_dir,
            "dataset_summary": dataset_summary,
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "train_metrics": train_result.metrics,
            "eval_metrics": eval_metrics,
            "distributed": {
                "world_size": WORLD_SIZE,
                "rank": RANK,
                "local_rank": LOCAL_RANK,
            },
        }
        with open(os.path.join(config.output_dir, "cpt_bdlm_run_summary.json"), "w", encoding="utf-8") as fh:
            json.dump(run_summary, fh, indent=4, ensure_ascii=False)

    # Save best checkpoint as FP32 + create symlink
    if trainer.is_world_process_zero():
        best_ckpt = getattr(trainer.state, "best_model_checkpoint", None)
        if best_ckpt and os.path.isdir(best_ckpt):
            # Create best_checkpoint symlink
            link_path = os.path.join(config.output_dir, "best_checkpoint")
            if os.path.exists(link_path):
                if os.path.islink(link_path):
                    os.unlink(link_path)
                else:
                    import shutil
                    shutil.rmtree(link_path)
            os.symlink(best_ckpt, link_path)
            print(f"   🔗 best_checkpoint -> {best_ckpt}")

            # Convert best checkpoint to FP32
            _fp32_dir = best_ckpt + "_fp32"
            if not os.path.exists(_fp32_dir):
                print(f"   💾 Converting best checkpoint to FP32...")
                from transformers import AutoModelForSeq2SeqLM as _M
                _m = _M.from_pretrained(best_ckpt, local_files_only=True)
                _m.float()
                _m.save_pretrained(_fp32_dir)
                # Copy tokenizer files too
                for _f in os.listdir(best_ckpt):
                    if _f.startswith("tokenizer") or _f in ("special_tokens_map.json", "added_tokens.json"):
                        import shutil
                        shutil.copy2(os.path.join(best_ckpt, _f), os.path.join(_fp32_dir, _f))
                del _m
                print(f"   ✅ FP32 checkpoint: {_fp32_dir}")

    maybe_distributed_barrier()
    print("\n✅ BDLM CPT training finished.")
    print(f"   Output: {config.output_dir}")


if __name__ == "__main__":
    main()
