#!/usr/bin/env python3
"""
使用 vLLM 模型批量翻译阿卡德语文本
"""

import pandas as pd
import requests
import json
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# 配置
# ══════════════════════════════════════════════════════════════
VLLM_URL = "http://localhost:8006/v1/chat/completions"
MODEL_NAME = "/data/lsb/deep_past/output/checkpoint-2400-merged"

INPUT_FILE = Path("/data/lsb/deep_past/data/data_t/akkadian_corpus_no_translation.csv")
OUTPUT_FILE = Path("/data/lsb/deep_past/data/data_t/akkadian_corpus_translated.csv")

BATCH_SIZE = 50  # 每批处理数量
MAX_WORKERS = 4  # 并发数


def call_model(prompt: str, system_prompt: str = None) -> str:
    """调用 vLLM 模型"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.1,
        "max_tokens": 512,
    }

    try:
        response = requests.post(VLLM_URL, json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        content = result["choices"][0]["message"]["content"].strip()
        return extract_answer(content)
    except Exception as e:
        return f"[ERROR: {e}]"


def extract_answer(content: str) -> str:
    """从模型输出中提取实际答案"""
    if not content:
        return ""

    # 查找包含 "Translation:" 的行
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('Translation:'):
            txt = line.split('Translation:', 1)[1].strip()
            if txt:
                return txt

    # 尝试找最后一个完整的句子
    lines = [l.strip() for l in content.split('\n') if l.strip()]
    for line in reversed(lines):
        # 过滤掉思考过程
        if len(line) > 5 and not line.startswith('1.') and not line.startswith('*'):
            # 清理标记
            line = re.sub(r'^\d+\.\s*', '', line)
            line = re.sub(r'^\*\s*', '', line)
            return line[:200]

    # 最后手段：返回最后几行
    return '\n'.join(lines[-2:]) if lines else content[:100]


def translate_batch(texts: list) -> list:
    """批量翻译"""
    system_prompt = """You are an expert in Akkadian cuneiform texts.
Translate the Akkadian transliteration into clear, modern English.
Be accurate and scholarly.
Output ONLY the translation, no explanation."""

    results = []
    for text in texts:
        if pd.isna(text) or not str(text).strip():
            results.append("")
            continue
        
        result = call_model(str(text), system_prompt)
        results.append(result)
        time.sleep(0.5)  # 避免过快
    
    return results


def main():
    print("🚀 批量翻译阿卡德语文本")
    print(f"   输入: {INPUT_FILE}")
    print(f"   输出: {OUTPUT_FILE}")
    print(f"   模型: {MODEL_NAME}")

    # 读取数据
    print("\n📖 加载数据...")
    df = pd.read_csv(INPUT_FILE)
    print(f"   总数: {len(df)}")

    # 翻译
    print("\n🔄 开始翻译...")
    translations = []
    
    # 分批处理
    total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for i in tqdm(range(total_batches), desc="翻译进度"):
        start_idx = i * BATCH_SIZE
        end_idx = min((i + 1) * BATCH_SIZE, len(df))
        batch = df['transliteration'].iloc[start_idx:end_idx].tolist()
        
        batch_results = translate_batch(batch)
        translations.extend(batch_results)
        
        # 每批保存一次（防止中断丢失）
        if (i + 1) % 5 == 0:
            temp_df = df.copy()
            temp_df['translation'] = translations
            temp_df.to_csv(OUTPUT_FILE.parent / "temp_translation.csv", index=False)
    
    # 添加翻译列
    df['translation'] = translations
    
    # 清洗翻译
    print("\n🧹 清洗翻译...")
    def clean_translation(text):
        if pd.isna(text) or not text:
            return ""
        # 移除编辑标记
        text = re.sub(r'\[([^\]]*)\]', r'\1', str(text))
        text = re.sub(r'<(?!gap\b)([^<>]+)>', r'\1', text)
        text = re.sub(r'<<[^<>]+>>', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    df['translation'] = df['translation'].apply(clean_translation)
    
    # 保存
    print(f"\n💾 保存结果...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")
    
    print(f"\n✅ 完成!")
    print(f"   翻译数量: {len(df)}")
    print(f"   输出: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
