#!/usr/bin/env python3
"""
Build a clean dataset from akkadian_corpus.csv following qwen_sentence_aligned_clean.csv format.

This script calls prepare_data.py for proper cleaning.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import re

# 添加脚本目录到路径，以便导入 prepare_data
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))

# 导入 prepare_data 中的清洗函数
import prepare_data as base_clean


# ══════════════════════════════════════════════════════════════
# 路径配置
# ══════════════════════════════════════════════════════════════
DEFAULT_INPUT = ROOT_DIR / "data" / "data_t" / "akkadian_corpus.csv"
DEFAULT_OUTPUT = ROOT_DIR / "data" / "data_t" / "akkadian_corpus_clean.csv"


# ══════════════════════════════════════════════════════════════
# 辅助清洗函数（prepare_data 未覆盖的部分）
# ══════════════════════════════════════════════════════════════
def clean_transliteration_akd(text: str) -> str:
    """针对阿卡德语的音译清洗 - 使用 prepare_data + 额外标准化"""
    if not isinstance(text, str):
        return ""
    
    # 先调用 prepare_data 的基础清洗
    text = base_clean.preprocess_transliteration(text)
    
    # 标准化数字编码
    text = text.replace("₄", "4")
    text = text.replace("₅", "5")
    text = text.replace("₆", "6")
    text = text.replace("₇", "7")
    text = text.replace("₈", "8")
    text = text.replace("₉", "9")
    text = text.replace("₀", "0")
    
    # 统一分数格式
    text = text.replace("0.33333", "⅓")
    text = text.replace("0.66666", "⅔")
    text = text.replace("0.5", "½")
    text = text.replace("0.25", "¼")
    text = text.replace("0.75", "¾")
    text = text.replace("1.5", "1½")
    text = text.replace("2.5", "2½")
    text = text.replace("3.5", "3½")
    
    return text


def clean_translation_akd(text: str) -> str:
    """针对阿卡德语的翻译清洗 - 使用 prepare_data 的完整预处理流程"""
    if not isinstance(text, str):
        return ""
    
    # 使用 prepare_data 的完整预处理流程（包括 normalize_gaps 将 ... 转为 <gap>）
    text = base_clean.preprocess_translation(text)
    
    # 额外处理：移除方括号 []（泥板破损标记）
    # 保留方括号内的内容（学者补全）
    text = re.sub(r'\[([^\]]*)\]', r'\1', text)
    
    # 处理尖括号 <> 纠正标记
    # <xxx> 漏字补全：保留内容，但排除 <gap> 标签
    text = re.sub(r'<(?!gap\b)([^<>]+)>', r'\1', text)
    # <<xxx>> 多字删除：删除内容
    text = re.sub(r'<<[^<>]+>>', '', text)
    
    # 断裂变音符修复 (CDLI/学术翻译惯例)
    text = re.sub(r'S,(?=[a-záàéèíìúùāēīū])', 'Ṣ', text)
    text = re.sub(r'T,(?=[a-záàéèíìúùāēīū])', 'Ṭ', text)
    text = re.sub(r's,(?=[a-záàéèíìúùāēīū])', 'ṣ', text)
    text = re.sub(r't,(?=[a-záàéèíìúùāēīū])', 'ṭ', text)
    
    # 旁白类括号删除
    text = re.sub(r'\([^)]*\?[^)]*\)', '', text)
    text = re.sub(r'\((?:i\.e\.|seal impression|lines? \d|broken|colophon|erasure|uninscribed)[^)]*\)', '', text, flags=re.IGNORECASE)
    # 补全类括号剥壳（保留文字）
    text = re.sub(r'\(([^)]{1,80})\)', r'\1', text)
    
    # 双句点修复
    text = re.sub(r'\.\.+', '.', text)
    
    # 规范化空白
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """质量过滤"""
    if df.empty:
        return df

    df = df.copy()

    # 去除空值
    initial_count = len(df)
    df = df[(df["oare_id"] != "") & (df["transliteration"] != "") & (df["translation"] != "")]
    df = df.dropna(subset=["transliteration", "translation"])
    removed = initial_count - len(df)
    if removed > 0:
        print(f"   🗑️  去除空值: {removed} 条")

    # 去除过短的翻译（少于3个词）
    short_mask = df["translation"].str.split().str.len() < 3
    if short_mask.any():
        print(f"   🗑️  去除过短翻译: {int(short_mask.sum())} 条")
        df = df[~short_mask]

    return df.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════
# 主函数
# ══════════════════════════════════════════════════════════════
def build_data_t(input_path: Path, output_path: Path) -> pd.DataFrame:
    """构建清洗后的数据集"""

    print(f"📖 加载数据: {input_path}")
    df = pd.read_csv(input_path)
    print(f"   原始记录数: {len(df)}")

    # 只保留平行语料（有翻译的）
    df = df[df["has_translation"] == True].copy()
    print(f"   平行语料数: {len(df)}")

    # 清洗音译
    print("   🔧 清洗音译...")
    df["transliteration"] = df["transliteration"].astype(str).apply(clean_transliteration_akd)

    # 清洗翻译 - 调用 prepare_data 的核心逻辑
    print("   🔧 清洗翻译 (使用 prepare_data)...")
    df["translation"] = df["translation"].astype(str).apply(clean_translation_akd)

    # 质量过滤
    print("   🔧 质量过滤...")
    df = quality_filter(df)

    # 设置 data_source
    df["data_source"] = "AICC"

    # 选择并重排列（只保留指定列）
    df = df[["oare_id", "transliteration", "translation", "data_source", "dialect"]].copy()

    # 规范化空白
    df["transliteration"] = df["transliteration"].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())
    df["translation"] = df["translation"].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

    # 去除重复
    df = df.drop_duplicates(subset=["oare_id"], keep="first").reset_index(drop=True)

    # 保存
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean dataset from akkadian_corpus.csv")
    parser.add_argument("--input", default=str(DEFAULT_INPUT), help="Input CSV path")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    df = build_data_t(input_path, output_path)

    print(f"\n✅ 数据清洗完成: {output_path}")
    print(f"   记录数: {len(df)}")
    print(f"   列: {', '.join(df.columns.tolist())}")


if __name__ == "__main__":
    main()
