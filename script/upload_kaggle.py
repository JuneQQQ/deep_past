#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Kaggle 数据集上传脚本
用于将本地目录（如模型权重或处理好的数据）上传为 Kaggle Dataset

使用方法:
    1. 修改下方 UploadConfig 类中的配置
    2. 运行: python script/upload_kaggle.py
    3. 或命令行覆盖上传目录:
       python script/upload_kaggle.py --dataset_path /path/to/dataset
"""

import argparse
import os
import sys
import json
import subprocess
import shutil
import re
import uuid
import tempfile
from pathlib import Path

# SSL 代理兼容：禁用 SSL 验证（代理 MITM 证书不被信任时需要）
# import requests
# _orig_send = requests.Session.send
# def _patched_send(self, request, **kwargs):
#     kwargs['verify'] = False
#     return _orig_send(self, request, **kwargs)
# requests.Session.send = _patched_send
# import urllib3
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


# ============================================================================
# 配置类（在此修改所有参数）
# ============================================================================

class UploadConfig:
    """上传配置 - 修改此处控制脚本行为"""
    
    # === 必填：要上传的目录路径 ===
    # dataset_path = "/data/lsb/deep_past/output/model_20260220_115256/checkpoint-3600"
    # hint train
    dataset_path = "/data/lsb/deep_past/output/dpo_20260321_221813/best_checkpoint"
    
    # === 数据集信息 ===
    dataset_title = "Deep Past Akkadian Translation"  # 数据集名称（None = 使用目录名）
    slug = None              # URL slug（None = 自动生成）
    
    # === 上传模式 ===
    force_new = True        # True = 强制创建新数据集；False = 自动检测
    
    # === Kaggle 认证（可选，优先使用 ~/.kaggle/kaggle.json）===
    kaggle_username = "bcdefga"   # 你的 Kaggle 用户名
    kaggle_key = None        # 你的 Kaggle API Key


# ============================================================================
# 核心逻辑（无需修改）
# ============================================================================

def get_kaggle_username(config: UploadConfig):
    """尝试获取 Kaggle 用户名"""
    if config.kaggle_username:
        return config.kaggle_username
    if "KAGGLE_USERNAME" in os.environ:
        return os.environ["KAGGLE_USERNAME"]
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if os.path.exists(kaggle_json_path):
        try:
            with open(kaggle_json_path, 'r') as f:
                return json.load(f).get('username')
        except:
            pass
    return None


def generate_slug(title):
    """生成合法的 Kaggle URL slug"""
    slug = re.sub(r'[^a-z0-9]+', '-', title.lower())
    slug = re.sub(r'-+', '-', slug).strip('-')
    return slug or f"dataset-{uuid.uuid4().hex[:8]}"


RESUME_ONLY_FILENAMES = {
    "dataset-metadata.json",
    "optimizer.pt",
    "scheduler.pt",
    "scaler.pt",
    "trainer_state.json",
}


def should_exclude_upload_file(path: Path) -> bool:
    name = path.name
    if name in RESUME_ONLY_FILENAMES:
        return True
    if re.fullmatch(r"rng_state(?:_[0-9]+)?\.(?:pth|pt)", name):
        return True
    return False


def stage_dataset_dir(source_dir: Path) -> tuple[Path, int, int]:
    staged_dir = Path(tempfile.mkdtemp(prefix="kaggle_upload_"))
    copied_files = 0
    skipped_files = 0

    for root, dirs, files in os.walk(source_dir):
        root_path = Path(root)
        rel_root = root_path.relative_to(source_dir)

        # 上传整个输出目录时，自动跳过中间 checkpoint，避免重复上传模型和续训状态。
        dirs[:] = [d for d in dirs if not d.startswith("checkpoint-")]

        dest_root = staged_dir / rel_root
        dest_root.mkdir(parents=True, exist_ok=True)

        for filename in files:
            src_file = root_path / filename
            if should_exclude_upload_file(src_file):
                skipped_files += 1
                continue
            shutil.copy2(src_file, dest_root / filename)
            copied_files += 1

    return staged_dir, copied_files, skipped_files


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upload a local directory as a Kaggle Dataset",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Override UploadConfig.dataset_path for this run",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = UploadConfig()
    if args.dataset_path:
        config.dataset_path = args.dataset_path
    
    if config.kaggle_username:
        os.environ["KAGGLE_USERNAME"] = config.kaggle_username
    if config.kaggle_key:
        os.environ["KAGGLE_KEY"] = config.kaggle_key
    
    project_root = Path(__file__).parent.parent
    target_dir = Path(config.dataset_path)
    if not target_dir.is_absolute():
        target_dir = project_root / target_dir
    target_dir = str(target_dir.resolve())
    
    if not os.path.exists(target_dir):
        print(f"❌ Error: Directory not found: {target_dir}")
        sys.exit(1)
        
    print(f"📦 Preparing to upload DATASET: {target_dir}")
    
    staged_dir, copied_files, skipped_files = stage_dataset_dir(Path(target_dir))
    print(f"   Staged files: {copied_files}, skipped resume-only files: {skipped_files}")
    print(f"   Upload staging dir: {staged_dir}")

    # Kaggle Dataset 使用 dataset-metadata.json
    metadata_path = os.path.join(staged_dir, "dataset-metadata.json")
    
    # 每次都创建新 dataset（随机 title/slug）
    is_new = True
    
    # 如果强制新建，必须清理旧的 metadata 否则 kaggle 命令会报错
    if is_new and os.path.exists(metadata_path):
        os.remove(metadata_path)
    
    if shutil.which("kaggle") is None:
        print("❌ Error: 'kaggle' command not found. Please install via: pip install kaggle")
        sys.exit(1)
        
    username = get_kaggle_username(config)
    if not username:
        print("❌ Error: Could not detect Kaggle username.")
        sys.exit(1)
            
    dir_name = os.path.basename(target_dir.rstrip('/'))
    # 强制拼接 UUID 以保证每次生成独立的名称和链接
    dataset_title = f"{config.dataset_title or dir_name} {uuid.uuid4().hex[:4]}"
    slug = f"{config.slug or generate_slug(dataset_title)}-{uuid.uuid4().hex[:6]}"
    if len(slug) < 5:
        slug = f"{slug}-{uuid.uuid4().hex[:4]}"

    dataset_id = f"{username}/{slug}"
    print(f"🆕 Creating new Kaggle Dataset: {dataset_id}")

    # Kaggle Dataset metadata 格式
    metadata = {
        "title": dataset_title,
        "id": dataset_id,
        "licenses": [
            {
                "name": "CC0-1.0"
            }
        ]
    }

    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)

        print(f"   Saved metadata to {metadata_path}")

        # 每次都创建全新 dataset
        cmd = ["kaggle", "datasets", "create", "-p", str(staged_dir)]
        
        print(f"🚀 Executing: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print("\n✅ Upload successful!")
        print(f"   Dataset URL: https://www.kaggle.com/datasets/{username}/{slug}")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Upload failed with exit code {e.returncode}")
        sys.exit(e.returncode)
    finally:
        shutil.rmtree(staged_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
