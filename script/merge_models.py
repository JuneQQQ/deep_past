#!/usr/bin/env python3
"""
手动融合多个模型权重（Model Soup / Weight Averaging）

用法:
    python merge_models.py                          # 使用默认模型路径
    python merge_models.py --models path1 path2     # 指定模型路径
    python merge_models.py --weights 0.6 0.4        # 指定权重（默认等权）
    python merge_models.py --output /path/to/save   # 指定输出目录

原理:
    对 N 个同架构模型的参数做加权平均：
    θ_merged = Σ(w_i * θ_i) / Σ(w_i)
    
    参考: Model Soups (Wortsman et al., 2022) https://arxiv.org/abs/2203.05482
"""

import os
import sys
import argparse
import torch
import json
import shutil
from pathlib import Path
from datetime import datetime, timezone, timedelta

_tz_shanghai = timezone(timedelta(hours=8))


def merge_models(model_paths, weights=None, output_dir=None, copy_tokenizer_from=0):
    """
    融合多个模型权重并保存到输出目录
    
    Args:
        model_paths: 模型目录列表（每个目录包含 model.safetensors + config.json）
        weights: 权重列表（默认等权），会自动归一化
        output_dir: 输出目录（默认自动生成带时间戳的目录名）
        copy_tokenizer_from: 从哪个模型复制 tokenizer 文件（索引，默认第一个）
    """
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    
    n_models = len(model_paths)
    if n_models < 2:
        print("❌ 至少需要 2 个模型才能融合")
        sys.exit(1)
    
    # 权重归一化
    if weights is None:
        weights = [1.0] * n_models
    assert len(weights) == n_models, f"权重数({len(weights)}) != 模型数({n_models})"
    total_w = sum(weights)
    norm_weights = [w / total_w for w in weights]
    
    # 输出目录
    if output_dir is None:
        ts = datetime.now(_tz_shanghai).strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(os.path.dirname(model_paths[0]), f"merged_{ts}")
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("🔀 Model Soup — 权重融合")
    print("=" * 60)
    print(f"   模型数: {n_models}")
    print(f"   输出目录: {output_dir}")
    print()
    for i, (path, w, nw) in enumerate(zip(model_paths, weights, norm_weights)):
        exists = os.path.exists(path)
        print(f"   [{i+1}] {path}")
        print(f"       权重: {w:.4f} (归一化: {nw:.4f}) {'✓' if exists else '❌ 不存在'}")
    print()
    
    # 验证所有路径存在
    for p in model_paths:
        if not os.path.exists(p):
            print(f"❌ 模型路径不存在: {p}")
            sys.exit(1)
    
    # 加载第一个模型
    print(f"📥 加载模型 [1]: {model_paths[0]}")
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_paths[0])
    state_dict = base_model.state_dict()
    
    # 转为 float32 进行精确平均
    for key in state_dict:
        state_dict[key] = state_dict[key].float() * norm_weights[0]
    
    # 加载并累加其他模型
    for i, model_path in enumerate(model_paths[1:], start=2):
        print(f"📥 加载模型 [{i}]: {model_path}")
        other_model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        other_sd = other_model.state_dict()
        
        # 架构兼容性检查
        for key in state_dict:
            if key not in other_sd:
                print(f"   ⚠️ 模型 [{i}] 缺少参数: {key}")
                continue
            if state_dict[key].shape != other_sd[key].shape:
                print(f"   ❌ 参数形状不匹配: {key}")
                print(f"      模型 [1]: {state_dict[key].shape}")
                print(f"      模型 [{i}]: {other_sd[key].shape}")
                print(f"   Model Soup 要求所有模型架构完全相同！")
                sys.exit(1)
            state_dict[key] += other_sd[key].float() * norm_weights[i - 1]
        
        # 释放显存
        del other_model, other_sd
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 转回原始精度
    orig_dtype = next(iter(base_model.state_dict().values())).dtype
    for key in state_dict:
        state_dict[key] = state_dict[key].to(orig_dtype)
    
    # 加载融合权重
    base_model.load_state_dict(state_dict)
    
    # 保存模型
    print(f"\n💾 保存融合模型到: {output_dir}")
    base_model.save_pretrained(output_dir)
    
    # 复制 tokenizer
    src_model = model_paths[copy_tokenizer_from]
    tokenizer = AutoTokenizer.from_pretrained(src_model)
    tokenizer.save_pretrained(output_dir)
    print(f"   ✓ Tokenizer 复制自: {src_model}")
    
    # 复制 generation_config（如果有）
    gen_cfg = os.path.join(src_model, "generation_config.json")
    if os.path.exists(gen_cfg):
        shutil.copy2(gen_cfg, output_dir)
        print(f"   ✓ generation_config.json 复制自: {src_model}")
    
    # 复制 training_config.json（如果有）
    train_cfg = os.path.join(src_model, "training_config.json")
    if os.path.exists(train_cfg):
        shutil.copy2(train_cfg, output_dir)
        print(f"   ✓ training_config.json 复制自: {src_model}")
    
    # 写入融合元信息
    merge_info = {
        "merge_timestamp": datetime.now(_tz_shanghai).strftime("%Y-%m-%d %H:%M:%S"),
        "models": model_paths,
        "weights": weights,
        "normalized_weights": norm_weights,
        "tokenizer_from": src_model,
    }
    with open(os.path.join(output_dir, "merge_info.json"), "w", encoding="utf-8") as f:
        json.dump(merge_info, f, indent=2, ensure_ascii=False)
    print(f"   ✓ merge_info.json 已保存")
    
    # 验证输出
    n_files = len(os.listdir(output_dir))
    total_size = sum(os.path.getsize(os.path.join(output_dir, f)) for f in os.listdir(output_dir))
    print(f"\n✅ 融合完成！")
    print(f"   文件数: {n_files}")
    print(f"   总大小: {total_size / 1024**3:.2f} GB")
    print(f"   输出目录: {output_dir}")
    print("=" * 60)
    
    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="融合多个模型权重（Model Soup）")
    parser.add_argument(
        "--models", nargs="+",
        default=[
            "/data/lsb/deep_past/output/model_20260304_093454/fold0",
            "/data/lsb/deep_past/output/model_20260304_093454/fold1",
            "/data/lsb/deep_past/output/model_20260304_093454/fold2",
            "/data/lsb/deep_past/output/model_20260304_093454/fold3",
        ],
        help="模型目录列表"
    )
    parser.add_argument(
        "--weights", nargs="+", type=float, default=None,
        help="权重列表（默认等权）"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="输出目录（默认自动生成）"
    )
    
    args = parser.parse_args()
    
    # 自动查找 checkpoint 子目录
    resolved_paths = []
    for p in args.models:
        if os.path.exists(p):
            # 检查是否直接包含模型文件
            has_model = any(f.endswith(('.safetensors', '.bin')) and 'model' in f
                          for f in os.listdir(p))
            if has_model:
                resolved_paths.append(p)
            else:
                # 查找最新的 checkpoint 子目录
                checkpoints = sorted([
                    d for d in os.listdir(p)
                    if d.startswith("checkpoint-") and os.path.isdir(os.path.join(p, d))
                ])
                if checkpoints:
                    cp_path = os.path.join(p, checkpoints[-1])
                    print(f"   📂 {p} → 使用 {checkpoints[-1]}")
                    resolved_paths.append(cp_path)
                else:
                    # 可能模型文件直接在根目录
                    resolved_paths.append(p)
        else:
            print(f"❌ 路径不存在: {p}")
            sys.exit(1)
    
    merge_models(resolved_paths, args.weights, args.output)
