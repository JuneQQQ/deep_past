#!/usr/bin/env python3
"""
构建 DPO 偏好数据集：
- chosen: train_sliding.csv 的 translation
- rejected: silver_pseudo_labels_raw.csv 的 translation
- 按 (oare_id, transliteration) 做唯一匹配
- chosen/rejected 完全一致的样本直接丢弃
"""

import os
import re
import pandas as pd


class Config:
    # 本地路径
    local_root = "/data/lsb/deep_past"
    root_dir = local_root
    data_dir = os.path.join(root_dir, "data")

    # 输入文件
    chosen_csv = os.path.join(data_dir, "train_sliding.csv")
    rejected_csv = os.path.join(data_dir, "silver_pseudo_labels_raw.csv")

    # 输出文件（DPO 可直接使用）
    output_csv = os.path.join(data_dir, "dpo_pairs_from_train_silver.csv")

    # 任务前缀（与训练脚本保持一致）
    task_prefix = "translate Akkadian to English: "


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_with_checks(path: str, required_cols: list[str], name: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"{name} 不存在: {path}")

    df = pd.read_csv(path)
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列: {missing}，实际列: {df.columns.tolist()}")
    return df


def keep_unique_keys(df: pd.DataFrame, key_cols: list[str], text_col: str, name: str):
    """
    对给定 key 保留“唯一可匹配”样本：
    1) 先按 (key + text) 去重
    2) 再只保留 key 出现次数为 1 的样本（去掉歧义 key）
    """
    before = len(df)
    df = df.drop_duplicates(subset=key_cols + [text_col]).copy()
    after_exact_dedup = len(df)

    key_counts = df.groupby(key_cols, dropna=False).size().reset_index(name="count")
    unique_keys = key_counts[key_counts["count"] == 1][key_cols]
    ambiguous_keys = key_counts[key_counts["count"] > 1][key_cols]

    kept = df.merge(unique_keys, on=key_cols, how="inner")

    print(f"\n[{name}]")
    print(f"  原始行数: {before}")
    print(f"  去重(key+text)后: {after_exact_dedup}")
    print(f"  唯一 key 数: {len(unique_keys)}")
    print(f"  歧义 key 数(已丢弃): {len(ambiguous_keys)}")
    print(f"  保留行数: {len(kept)}")

    return kept, ambiguous_keys


def main():
    cfg = Config()

    key_cols = ["oare_id", "transliteration"]

    chosen_df = load_with_checks(
        cfg.chosen_csv,
        required_cols=["oare_id", "transliteration", "translation"],
        name="chosen(train_sliding)",
    )[["oare_id", "transliteration", "translation"]].rename(columns={"translation": "chosen"})

    rejected_df = load_with_checks(
        cfg.rejected_csv,
        required_cols=["oare_id", "transliteration", "translation"],
        name="rejected(silver_pseudo_labels_raw)",
    )[["oare_id", "transliteration", "translation"]].rename(columns={"translation": "rejected"})

    # 基础清洗
    for col in ["oare_id", "transliteration", "chosen"]:
        chosen_df[col] = chosen_df[col].astype(str).map(lambda x: x.strip())
    for col in ["oare_id", "transliteration", "rejected"]:
        rejected_df[col] = rejected_df[col].astype(str).map(lambda x: x.strip())

    chosen_df = chosen_df[chosen_df["chosen"].str.len() > 0].copy()
    rejected_df = rejected_df[rejected_df["rejected"].str.len() > 0].copy()

    # 保留唯一 key 的记录，避免一对多/多对一
    chosen_unique, _ = keep_unique_keys(chosen_df, key_cols, "chosen", "chosen(train_sliding)")
    rejected_unique, _ = keep_unique_keys(rejected_df, key_cols, "rejected", "rejected(silver_raw)")

    # 严格按 (oare_id, transliteration) 匹配
    merged = chosen_unique.merge(rejected_unique, on=key_cols, how="inner")
    print(f"\n[merge]")
    print(f"  匹配后总行数: {len(merged)}")

    # 去掉 chosen/rejected 完全一致（按规范化文本比较）
    merged["chosen_norm"] = merged["chosen"].map(normalize_text)
    merged["rejected_norm"] = merged["rejected"].map(normalize_text)

    same_mask = merged["chosen_norm"] == merged["rejected_norm"]
    same_count = int(same_mask.sum())
    final_df = merged[~same_mask].copy()

    # 生成 DPO 常用字段
    final_df["prompt"] = cfg.task_prefix + final_df["transliteration"].astype(str)
    final_df["input_text"] = final_df["prompt"]
    final_df["target_text"] = final_df["chosen"]

    # 输出列：含 DPO 训练核心字段 + 追踪字段
    out_cols = [
        "oare_id",
        "transliteration",
        "prompt",
        "chosen",
        "rejected",
        "input_text",
        "target_text",
    ]
    final_df = final_df[out_cols].reset_index(drop=True)

    os.makedirs(os.path.dirname(cfg.output_csv), exist_ok=True)
    final_df.to_csv(cfg.output_csv, index=False, encoding="utf-8")

    print(f"\n[final]")
    print(f"  去掉译文完全一致: {same_count}")
    print(f"  最终样本数: {len(final_df)}")
    print(f"  输出文件: {cfg.output_csv}")


if __name__ == "__main__":
    main()
