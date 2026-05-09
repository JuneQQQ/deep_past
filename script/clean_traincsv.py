#!/usr/bin/env python3
"""
利用大语言模型 (LLM) 进行双向语义对齐裁剪
- 高 Ratio (>= 1.2): 阿卡德语过长，英语截断 -> 裁剪阿卡德语
- 低 Ratio (<= 0.6): 阿卡德语截断，英语过长 -> 裁剪英语
"""

import csv
import pandas as pd
import asyncio
import logging
import re
import os
from tqdm.auto import tqdm
from openai import AsyncOpenAI, APIError, RateLimitError

# ============================================================
# 配置参数 (Configuration)
# ============================================================
INPUT_CSV = "/data/lsb/deep_past/data/train_clean.csv"
OUTPUT_CSV = "/data/lsb/deep_past/data/train_clean_llm_cleaned.csv"

# API 配置
API_KEY = "sk-WuVg5VaB0iV7NxJNXiOFjc7nFH9ge3aNI8n6hChX4HQKQNVp"
BASE_URL = "https://www.dmxapi.cn/v1"
MODEL_NAME = "gemini-3.1-pro-preview"

# 过滤阈值 (字符比例)
RATIO_LOW = 0.6
RATIO_HIGH = 1.2

# 并发与重试
CONCURRENCY = 10
MAX_RETRIES = 3

# ============================================================
# 日志设置 (Logging)
# ============================================================
logger = logging.getLogger("LLM_Cleaner")
logger.setLevel(logging.INFO)

# 终端输出处理器
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%H:%M:%S')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# 文件输出处理器 (保存详细日志供复盘)
log_file_path = os.path.join(os.path.dirname(INPUT_CSV), "llm_cleaner.log")
file_handler = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ============================================================
# Prompts 定义
# ============================================================
PROMPT_TRIM_AKKADIAN = """You are a meticulous Assyriologist. 
The English text below is a PARTIAL translation of the longer Akkadian text.
Your mission: Trim the Akkadian text so it matches the English, but DO NOT over-trim.

RULES:
1. Every piece of information in the English text (names, amounts, goods, verbs) MUST have its corresponding original words retained in your Akkadian output.
2. If the English translation is "I gave 10 minas of silver to Zuzu", your Akkadian output MUST include the words for "silver", "10 minas", "Zuzu", and the verb "gave". 
3. Do NOT just keep the first few words of the greeting/address.
4. Add "<gap>" at the end of the trimmed Akkadian.
5. Output ONLY the trimmed Akkadian text. No explanation.

[IMPORTANT] The resulting Akkadian-to-English word ratio should be roughly 1:1. If you output only 2-5 words for a long English sentence, you have FAILED.

Akkadian Input: {akkadian}
English Input: {english}
Output:"""

PROMPT_TRIM_ENGLISH = """You are a meticulous Assyriologist.
The Akkadian text below is broken (truncated), but the English translation is a full version.
Your mission: Trim the English text to match ONLY what is actually written in the Akkadian fragment.

RULES:
1. If a name or a verb (like "paid", "received") is NOT present in the Akkadian text, you MUST remove it from the English.
2. If the Akkadian ends abruptly, the English must end with a "<gap>".
3. Accuracy is more important than English grammar. If the Akkadian is a fragment, the English should be a fragment.
4. Output ONLY the trimmed English. No explanation.

Akkadian Input: {akkadian}
English Input: {english}
Output:"""

# ============================================================
# LLM 异步处理核心
# ============================================================
client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)


def is_likely_non_parallel(akk: str, eng: str) -> tuple[bool, str]:
    """Detect metadata, index tables, and other clearly non-parallel content."""
    akk = str(akk or "")
    eng = str(eng or "")

    page_refs = re.findall(r'\d+:\d+', akk)
    if len(page_refs) >= 3:
        return True, "page_references_in_source"

    tabular_keywords = {
        'debtor', 'creditor', 'interets', 'interest', 'varia', 'cf.', 'unopened case'
    }
    combined = f"{akk} {eng}".lower()
    if sum(1 for kw in tabular_keywords if kw in combined) >= 2:
        return True, "tabular_data"

    akk_words = akk.split()
    if len(akk_words) > 10:
        short_abbrev = sum(1 for w in akk_words if re.match(r'^[a-z]{1,2}$', w))
        if short_abbrev / len(akk_words) > 0.3:
            return True, "abbreviation_heavy"

    return False, ""


def compute_char_ratio(tl: str, tr: str) -> float:
    tl_chars = len(re.sub(r'\s+', '', str(tl)))
    tr_chars = len(re.sub(r'\s+', '', str(tr)))
    return tl_chars / max(tr_chars, 1)


def preview_text(text: str, limit: int = 100) -> str:
    compact = re.sub(r'\s+', ' ', str(text)).strip()
    if len(compact) <= limit:
        return compact
    return compact[:limit - 3] + "..."


def log_written_row(row: dict, idx: int, total: int) -> None:
    ratio = compute_char_ratio(row.get('transliteration', ''), row.get('translation', ''))
    logger.info(
        f"[WRITE {idx + 1}/{total}] {row.get('oare_id', '')} "
        f"source={row.get('data_source', '')} status={row.get('llm_fixed', '') or 'unchanged'} "
        f"ratio={ratio:.2f} "
        f"TL={preview_text(row.get('transliteration', ''))} "
        f"| TR={preview_text(row.get('translation', ''))}"
    )

async def call_llm_with_retry(prompt: str, row_id: str) -> str:
    """带重试机制的 LLM 调用"""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,  # 极低温度保证输出稳定性
                max_tokens=512
            )
            result = response.choices[0].message.content.strip()
            # 清理可能存在的 markdown 代码块包裹
            result = re.sub(r'^```[a-zA-Z]*\n', '', result)
            result = re.sub(r'\n```$', '', result).strip()
            return result
            
        except (APIError, RateLimitError) as e:
            logger.warning(f"[{row_id}] API Error on attempt {attempt}: {str(e)}")
            if attempt == MAX_RETRIES:
                logger.error(f"[{row_id}] Failed after {MAX_RETRIES} attempts.")
                return None
            await asyncio.sleep(2 ** attempt)  # 指数退避
        except Exception as e:
            logger.error(f"[{row_id}] Unexpected error: {str(e)}")
            return None

async def process_row(row_index: int, row: dict, semaphore: asyncio.Semaphore) -> tuple[int, dict]:
    """处理单行数据"""
    async with semaphore:
        oare_id = row['oare_id']
        tl = str(row['transliteration']).strip()
        tr = str(row['translation']).strip()
        data_source = str(row.get('data_source', '')).strip().lower()
        
        if data_source == 'ocr':
            row['llm_fixed'] = 'skipped_ocr'
            return row_index, row

        # 1. 计算无空格字符比例
        tl_chars = len(re.sub(r'\s+', '', tl))
        tr_chars = len(re.sub(r'\s+', '', tr))
        
        if tr_chars == 0:
            row['llm_fixed'] = row.get('llm_fixed', '') or 'unchanged'
            return row_index, row  # 保护空数据
            
        char_ratio = tl_chars / max(tr_chars, 1)
        tr_wc = len(tr.split())
        
        # 2. 短句保护 (低 Ratio 且英语单词数极少的情况，通常是语言学膨胀，不修)
        if char_ratio <= RATIO_LOW and tr_wc <= 10:
            logger.debug(f"[{oare_id}] Skipped (Short sentence protection): ratio={char_ratio:.2f}")
            row['llm_fixed'] = row.get('llm_fixed', '') or 'unchanged'
            return row_index, row

        # 3. 分发到不同的 Prompt 引擎
        prompt = None
        action_type = ""
        
        if char_ratio >= RATIO_HIGH:
            prompt = PROMPT_TRIM_AKKADIAN.format(akkadian=tl, english=tr)
            action_type = "Trim Akkadian"
        elif char_ratio <= RATIO_LOW:
            prompt = PROMPT_TRIM_ENGLISH.format(akkadian=tl, english=tr)
            action_type = "Trim English"
            
        # 如果不需要处理，直接返回
        if not prompt:
            row['llm_fixed'] = row.get('llm_fixed', '') or 'unchanged'
            return row_index, row
            
        # 4. 调用 LLM
        logger.info(f"[{oare_id}] Started {action_type} (Ratio: {char_ratio:.2f})")
        llm_result = await call_llm_with_retry(prompt, oare_id)
        
        if llm_result:
            if action_type == "Trim Akkadian":
                logger.info(f"[{oare_id}] Success! TL: {len(tl)} -> {len(llm_result)} chars")
                row['transliteration'] = llm_result
                row['llm_fixed'] = 'trimmed_akkadian'
            elif action_type == "Trim English":
                logger.info(f"[{oare_id}] Success! TR: {len(tr)} -> {len(llm_result)} chars")
                row['translation'] = llm_result
                row['llm_fixed'] = 'trimmed_english'
        else:
            row['llm_fixed'] = 'failed'
            
        return row_index, row

async def main():
    logger.info("=" * 60)
    logger.info("Starting LLM Semantic Alignment Pipeline")
    logger.info(f"Input: {INPUT_CSV}")
    logger.info(f"Model: {MODEL_NAME} | Concurrency: {CONCURRENCY}")
    logger.info("=" * 60)
    
    # 1. 读取数据
    try:
        df = pd.read_csv(INPUT_CSV)
        df = df.fillna('')
        logger.info(f"Loaded {len(df)} rows from dataset.")
    except Exception as e:
        logger.error(f"Failed to read CSV: {e}")
        return
        
    if 'llm_fixed' not in df.columns:
        df['llm_fixed'] = ''

    non_parallel_results = [
        is_likely_non_parallel(tl, tr)
        for tl, tr in zip(df['transliteration'], df['translation'])
    ]
    non_parallel_mask = pd.Series([flag for flag, _ in non_parallel_results], index=df.index)
    dropped_non_parallel = int(non_parallel_mask.sum())
    if dropped_non_parallel > 0:
        reason_counts = {}
        for flag, reason in non_parallel_results:
            if flag and reason:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
        reason_text = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
        logger.info(
            "Dropping non-parallel rows before LLM: %s%s",
            dropped_non_parallel,
            f" ({reason_text})" if reason_text else "",
        )
        df = df.loc[~non_parallel_mask].copy()
        df = df.reset_index(drop=True)

    # 2. 为所有行创建任务；OCR 样本会在 worker 中直接跳过
    tasks = []
    semaphore = asyncio.Semaphore(CONCURRENCY)
    
    rows = df.to_dict('records')
    fieldnames = list(df.columns)
    if 'llm_fixed' not in fieldnames:
        fieldnames.append('llm_fixed')

    official_rows = 0
    llm_candidates = 0
    high_ratio_candidates = 0
    low_ratio_candidates = 0
    short_sentence_protected = 0
    official_unchanged = 0
    skipped_ocr = 0
    
    for idx, row in enumerate(rows):
        tl = str(row['transliteration']).strip()
        tr = str(row['translation']).strip()
        data_source = str(row.get('data_source', '')).strip().lower()
        if data_source == 'ocr':
            skipped_ocr += 1
        else:
            official_rows += 1

        ratio = compute_char_ratio(tl, tr)
        tr_wc = len(tr.split())

        if data_source != 'ocr':
            if ratio >= RATIO_HIGH:
                llm_candidates += 1
                high_ratio_candidates += 1
            elif ratio <= RATIO_LOW and tr_wc > 10:
                llm_candidates += 1
                low_ratio_candidates += 1
            elif ratio <= RATIO_LOW and tr_wc <= 10:
                short_sentence_protected += 1
            else:
                official_unchanged += 1
        tasks.append(asyncio.create_task(process_row(idx, row, semaphore)))

    logger.info(
        "Startup stats | total=%s official=%s ocr=%s llm_candidates=%s",
        len(rows), official_rows, skipped_ocr, llm_candidates
    )
    logger.info(
        "LLM candidate breakdown | high_ratio=%s low_ratio=%s short_sentence_protected=%s official_unchanged=%s",
        high_ratio_candidates, low_ratio_candidates, short_sentence_protected, official_unchanged
    )
    
    if not tasks:
        logger.info("No rows need fixing based on the current thresholds.")
        return

    # 3. 异步处理 + 按原始顺序流式写出，方便实时查看输出文件
    pending_rows: dict[int, dict] = {}
    next_write_idx = 0
    total_rows = len(rows)

    with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as out_f:
        writer = csv.DictWriter(out_f, fieldnames=fieldnames)
        writer.writeheader()
        out_f.flush()

        for task in tqdm(asyncio.as_completed(tasks), total=total_rows, desc="Processing with LLM"):
            row_index, processed_row = await task
            pending_rows[row_index] = processed_row

            while next_write_idx in pending_rows:
                row_to_write = pending_rows.pop(next_write_idx)
                writer.writerow({name: row_to_write.get(name, '') for name in fieldnames})
                out_f.flush()
                log_written_row(row_to_write, next_write_idx, total_rows)
                next_write_idx += 1

    # 4. 重新读取流式输出结果，生成统计
    new_df = pd.read_csv(OUTPUT_CSV)
    
    # 5. 统计报告
    trimmed_akk = len(new_df[new_df['llm_fixed'] == 'trimmed_akkadian'])
    trimmed_eng = len(new_df[new_df['llm_fixed'] == 'trimmed_english'])
    failed = len(new_df[new_df['llm_fixed'] == 'failed'])
    skipped_ocr_out = len(new_df[new_df['llm_fixed'] == 'skipped_ocr'])
    
    logger.info("=" * 60)
    logger.info("Pipeline Completed!")
    logger.info(f"Trimmed Akkadian (High Ratio): {trimmed_akk}")
    logger.info(f"Trimmed English (Low Ratio): {trimmed_eng}")
    logger.info(f"Skipped OCR rows: {skipped_ocr_out}")
    logger.info(f"Failed to process: {failed}")
    logger.info(f"Output saved to: {OUTPUT_CSV}")
    logger.info(f"Detailed logs saved to: {log_file_path}")
    logger.info("=" * 60)

if __name__ == "__main__":
    # 解决 asyncio 在某些环境下(如 Jupyter/Windows) 的运行时错误
    import sys
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())
