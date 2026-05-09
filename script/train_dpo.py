#!/usr/bin/env python3
# !pip install evaluate 
# !pip install -U bitsandbytes sacrebleu
"""
================================================================================
STUDENT MODEL TRAINING - DPO (TRL, 本地版)
================================================================================
适配本地环境：
- 使用本地数据路径
- 支持 TRL DPO 偏好优化训练
- 优先加载本地缓存模型，必要时回退到 HuggingFace Hub
================================================================================
"""

# ============================================================================
# 离线模式配置（必须在 import transformers 之前）
# ============================================================================
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--fold", type=int, default=-1, help="Fold index to train (-1 means run all folds)")
parser.add_argument("--timestamp", type=str, default=None, help="Timestamp for output dir")
parser.add_argument("--train_csv", type=str, default=None, help="Override train_csv path (used for snapshot)")
args, _ = parser.parse_known_args()



# os.environ["HF_HUB_OFFLINE"] = "1"  # Disabled to download new model
# os.environ["TRANSFORMERS_OFFLINE"] = "1"  # Disabled to download new model
# os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["KAGGLE_API_TOKEN"] = "KGAT_fa65bd883abd49b2516d5c2fccae7b8f"
import re
import gc
import math
import torch
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Set, Dict
from difflib import SequenceMatcher

# 从 prepare_data 导入词典/后处理函数（脚本模式必须）
from prepare_data import load_lexicon, build_onomasticon, preprocess_transliteration, postprocess_output

# 全局 Onomasticon（专有名词词典）
_onomasticon: Set[str] = set()
_lexicon: Dict[str, str] = {}  # 全局词典，用于动态约束


def fast_byt5_batch_decode(sequences, *, tokenizer, chunk_size: int = 32):
    """Fast local decode for ByT5 byte IDs."""
    if torch.is_tensor(sequences):
        sequences = sequences.detach().cpu()
        if sequences.ndim == 1:
            rows = [sequences.numpy()]
        else:
            rows = [row.numpy() for row in sequences]
    elif isinstance(sequences, np.ndarray):
        if sequences.dtype != object:
            if sequences.ndim == 1:
                rows = [sequences]
            else:
                rows = [row for row in sequences]
        else:
            rows = [np.asarray(row) for row in sequences.tolist()]
    else:
        try:
            rows = [np.asarray(row) for row in sequences]
        except TypeError:
            rows = [np.asarray(sequences)]
        if rows and rows[0].ndim == 0:
            rows = [np.asarray(sequences)]

    offset = int(getattr(tokenizer, "offset", 3))
    utf_vocab_size = int(getattr(tokenizer, "_utf_vocab_size", 256))
    low = offset
    high = offset + utf_vocab_size
    decoded = []
    total = len(rows)

    for start in range(0, total, chunk_size):
        batch = rows[start:start + chunk_size]
        for row in batch:
            row = np.asarray(row).reshape(-1)
            valid = row[(row >= low) & (row < high)]
            if valid.size == 0:
                decoded.append("")
                continue
            byte_seq = (valid - offset).astype(np.uint8, copy=False).tobytes()
            decoded.append(byte_seq.decode("utf-8", errors="ignore"))

    return decoded


def decode_sequences(sequences, *, tokenizer):
    if tokenizer.__class__.__name__ == "ByT5Tokenizer":
        return fast_byt5_batch_decode(sequences, tokenizer=tokenizer)
    return tokenizer.batch_decode(
        sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


# Clear memory
gc.collect()
torch.cuda.empty_cache()

print("="*60)
print("STUDENT MODEL TRAINING")
print("="*60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ============================================================================
# CONFIGURATION (在文件头部，方便快速修改)
# ============================================================================

class Config:
    """消融实验配置类 - 所有参数集中管理"""
    
    def __init__(self):
        # ========== 路径配置 ==========
        local_root = "/root/projects/deep_past"
        self.is_local = True
        self.root_dir = local_root
        self.data_dir = os.path.join(self.root_dir, "data")
        # 使用 prepare_data.py 输出的预处理数据
        self.train_csv = (
            args.train_csv
            if args.train_csv
            else os.path.join(self.data_dir, "dpo_pairs_from_train_silver.csv")
        )
        # self.train_csv = os.path.join("/root/projects/deep_past/data_v2/", "train_mixed.csv")
        # self.test_csv = os.path.join(self.data_dir, "tetrain_st.csv")
        self.lexicon_csv = os.path.join(self.data_dir, "OA_Lexicon_eBL.csv")
        # 手动切分的长文档数据（优先替换同 oare_id 的样本，空字符串=不启用）
        # self.train_code_splits = os.path.join(self.data_dir, "train_manual_code_splits.csv")
        self.train_code_splits = os.path.join("")
        # 输出目录带时间戳
        from datetime import datetime
        timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        self._timestamp = timestamp
        self.output_dir = os.path.join(self.root_dir, "output", f"model_{timestamp}")  # 多fold时内部建foldN子目录
        
        # ========== 模型配置 ==========
        # 如需切换模型（含本地目录路径），直接修改此值
        self.model_name = "/root/projects/deep_past/output/model_20260318_175252/best_checkpoint"
        self.task_prefix = "translate Akkadian to English: "
        
        # ========== 训练参数 ==========
        self.num_epochs = 10
        self.learning_rate = 5e-6
        self.warmup_ratio = 0.1
        self.weight_decay = 0.01
        
        # ========== 序列长度 ==========
        self.max_input_length = 512
        self.max_target_length = 512

        # ========== DPO 参数 ==========
        self.dpo_beta = 0.3
        self.dpo_batch_size = 16
        self.dpo_gradient_accumulation = 1  # 有效 batch = 16
        self.dpo_eval_batch_size = 32     # eval 无梯度，可以更大
        

        # ========== 生成参数 (评估阶段) ==========
        self.num_beams = 10
        self.length_penalty = 1.5
        self.max_hints_per_sample = 10
        
        # ========== 评估和保存 ==========
        self.logging_steps = 10
        self.save_total_limit = 5
        self.gradient_checkpointing = True
        
        # ========== 数据策略 ==========
        self.test_size = 0.1
        self.random_seed = 42
        self.fold_group_col = "oare_id"

        # ========== Early Stopping ==========
        self.use_early_stopping = True
        self.early_stopping_patience = 5

        # ========== 精度 ==========
        self.dtype = "bf16"

        # ========== 多 Fold ==========
        self.num_folds = 1
        self.current_fold = args.fold if args.fold != -1 else 0

        # ========== 数据增强（DPO 中默认关闭） ==========
        self.use_simulate_damage = False
        self.damage_prob = 0.1
        self.damage_span_prob = 0.7
        self.use_curriculum = False
        self.use_rag = False
        
    
    # 精度辅助属性
    @property
    def use_bf16(self): return self.dtype == "bf16"
    
    def to_dict(self):
        """转换为字典用于日志"""
        return {k: str(v) if not isinstance(v, (int, float, bool, str)) else v 
                for k, v in self.__dict__.items()}
    
    def print_ablation_status(self):
        """打印消融开关状态"""
        print("\n🔬 消融开关状态:")
        print(f"   dtype: {self.dtype}")
        print(f"   use_ema: {self.use_ema} (decay={self.ema_decay})")
        print(f"   use_curriculum: {self.use_curriculum} (epochs={self.curriculum_epochs})")
        print(f"   use_early_stopping: {self.use_early_stopping}")
        print(f"   use_geom_mean: {self.use_geom_mean}")
        print(f"   use_simulate_damage: {self.use_simulate_damage} (prob={self.damage_prob}, span={self.damage_span_prob})")
        print(f"   name_handling_mode: {self.name_handling_mode}")
        if self.name_handling_mode == 'placeholder':
            print(f"      placeholder_ratio: {self.placeholder_ratio}")
        elif self.name_handling_mode == 'hint':
            print(f"      hint_ratio: {self.hint_ratio}")
            print(f"      max_hints_per_sample: {self.max_hints_per_sample}")
            print(f"      use_month_hints: {self.use_month_hints}")
        print(f"   use_logging: {self.use_logging}")
        print(f"   use_plot: {self.use_plot}")
        print(f"   dpo_beta: {self.dpo_beta}")
        print(f"   dpo_loss_type: {self.dpo_loss_type}")
        print(f"   dpo_max_prompt_length: {self.dpo_max_prompt_length}")
        print(f"   dpo_max_completion_length: {self.dpo_max_completion_length}")
        print("   预处理/质量过滤: 已移至 prepare_data.py")


config = Config()

if args.fold == -1 or args.fold == config.num_folds:
    if config.num_folds <= 1:
        print("⚠️ config.num_folds <= 1，将只进行一次训练。")
        args.fold = 0
    else:
        import subprocess
        import sys
        from datetime import datetime
        import shutil
        
        timestamp = args.timestamp if args.timestamp else datetime.now().strftime("%Y%m%d_%H%M%S")
        
        print(f"\n{'='*60}")
        print(f"🚀 自动启动 {config.num_folds} 折训练 (Timestamp: {timestamp})")
        print(f"{'='*60}\n")
        
        # ⚠️ 防止由于中途修改训练数据导致多折数据不一致，提前制作一个数据快照
        output_dir = os.path.join(config.root_dir, "output", f"model_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)
        snapshot_csv = os.path.join(output_dir, "train_snapshot.csv")
        try:
            shutil.copy2(config.train_csv, snapshot_csv)
            print(f"📦 已创建训练数据快照: {snapshot_csv} (多折共用，防中途修改)")
        except Exception as e:
            print(f"⚠️ 创建数据快照失败 ({e})，将回退使用原文件。")
            snapshot_csv = config.train_csv
        
        for f in range(config.num_folds):
            print(f"\n{'='*50}")
            print(f"🚀 开始训练 Fold {f}/{config.num_folds-1}...")
            print(f"{'='*50}")
            
            cmd = [sys.executable, sys.argv[0], "--fold", str(f), "--timestamp", timestamp, "--train_csv", snapshot_csv]
            # 传递其他可能存在的参数
            for arg in sys.argv[1:]:
                if not arg.startswith("--fold") and not arg.startswith("--timestamp") and not arg.startswith("--train_csv"):
                    cmd.append(arg)
                    
            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"\n❌ Fold {f} 训练失败，退出整个流程。")
                sys.exit(e.returncode)
                
        print("\n🎉 所有 Fold 训练完成！")
        sys.exit(0)

# 更新 config 的 fold 属性
config.current_fold = args.fold if args.fold != -1 else 0


# ============================================================================
# 加载词典并初始化 Onomasticon
# ============================================================================
print("\n📚 Loading lexicon...")
if os.path.exists(config.lexicon_csv):
    # 使用 prepare_data 中的函数加载
    _lexicon = load_lexicon(config.lexicon_csv)
    _onomasticon = build_onomasticon(_lexicon)
    print(f"   Lexicon size: {len(_lexicon)}")
    print(f"   Onomasticon size: {len(_onomasticon)}")
else:
    print(f"   ⚠️ Lexicon file not found at {config.lexicon_csv}")
    _lexicon = {}
    _onomasticon = set()

# 加载校准词典（用训练数据 TL↔TR 对校准后的拼写）
_calibrated_lexicon = {}
_cal_lex_path = os.path.join(config.data_dir, "calibrated_lexicon.json")
if os.path.exists(_cal_lex_path):
    import json as _json_loader
    with open(_cal_lex_path, 'r', encoding='utf-8') as f:
        _calibrated_lexicon = _json_loader.load(f)
    print(f"   📚 Calibrated lexicon: {len(_calibrated_lexicon)} entries")
else:
    print(f"   ℹ️ No calibrated lexicon found at {_cal_lex_path}, using raw lexicon")

# ============================================================================
# LOGGING SETUP (实时保存控制台输出)
# ============================================================================
import sys

# Ensure output directory exists
os.makedirs(config.output_dir, exist_ok=True)

class TeeLogger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    
    def isatty(self):
        return self.terminal.isatty() if hasattr(self.terminal, 'isatty') else False

# Redirect stdout to both terminal and file
log_file = os.path.join(config.output_dir, "training_log.txt")
sys.stdout = TeeLogger(log_file)
print(f"📝 Logging to: {log_file}")

print(f"\n📋 Configuration (Full Auto-Log):")
# 动态打印所有配置项，确保新添加的属性也能被自动记录
for key, value in sorted(config.to_dict().items()):
    print(f"   {key}: {value}")

# ============================================================================
# CELL 3: PREPROCESSING FUNCTIONS
# ============================================================================

# --- 模拟泥板损坏（数据增强）---
import random

# 创建独立的随机数生成器，使用固定种子确保可复现
_damage_rng = random.Random(config.random_seed)

def reset_damage_rng():
    """重置损坏随机数生成器（每个 epoch 开始时调用可确保一致性）"""
    global _damage_rng
    _damage_rng = random.Random(config.random_seed)

def merge_consecutive_gaps(text):
    """合并连续的 gap 标记，确保不会出现连续 gap"""
    if not isinstance(text, str):
        return text
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'<gap>\s+<gap>', '<gap>', text)
    return text


def simulate_damage(text, damage_prob=0.1, span_prob=0.7):
    """
    模拟泥板物理损坏，增强模型鲁棒性
    使用固定种子的随机数生成器确保可复现
    
    Args:
        text: 原始阿卡德语转写
        damage_prob: 触发损坏的概率
        span_prob: Span 损坏 vs 单词损坏 (<gap>) 的比例
    
    Returns:
        可能被"损坏"的文本
    """
    global _damage_rng
    if not isinstance(text, str) or _damage_rng.random() > damage_prob:
        return text
    
    words = text.split()
    n = len(words)
    if n < 3:
        return text  # 太短的不做损坏
    
    # 找到可以安全替换的位置（前后都不是 gap）
    safe_indices = []
    for i in range(n):
        if '<' in words[i]:
            continue  # 跳过已有的 gap
        # 检查前后是否有 gap
        prev_is_gap = (i > 0 and '<' in words[i-1])
        next_is_gap = (i < n-1 and '<' in words[i+1])
        if not prev_is_gap and not next_is_gap:
            safe_indices.append(i)
    
    if not safe_indices:
        return text  # 没有安全位置
    
    if _damage_rng.random() < span_prob:
        # --- Span 损坏 ---
        # 找连续的安全位置
        consecutive = []
        for i, idx in enumerate(safe_indices):
            if i > 0 and safe_indices[i-1] == idx - 1:
                consecutive.append(idx)
            else:
                consecutive = [idx]
            if len(consecutive) >= 2:
                break
        
        if len(consecutive) >= 2:
            span_len = min(_damage_rng.randint(2, 5), len(consecutive))
            start = consecutive[0]
            new_words = words[:start] + ['<gap>'] + words[start + span_len:]
        else:
            return text  # 没有连续安全位置
    else:
        # --- 单词损坏 (<gap>) ---
        idx = _damage_rng.choice(safe_indices)
        words[idx] = '<gap>'
        new_words = words
    
    result = ' '.join(new_words)
    return merge_consecutive_gaps(result)  # 最终合并检查


# --- 预处理函数（数据已由 prepare_data.py 预处理，这里只添加 task_prefix）---
def preprocess_input(text):
    """添加任务前缀，可选模拟损坏"""
    if not isinstance(text, str):
        return config.task_prefix
    
    # 训练时可选模拟泥板损坏
    if config.use_simulate_damage:
        text = simulate_damage(text, config.damage_prob, config.damage_span_prob)
    
    return config.task_prefix + text


# 阿卡德语高频功能词黑名单（这些词即使在词典里也不应作为 HINTS）
_HINT_MIN_FORM_LEN = 4  # 过短的 key 极易误匹配，跳过

# 已知阿卡德语语法后缀（只允许剥离这些，不做任意音节剥离）
_AKKADIAN_SUFFIXES = {
    'ma',   # 强调词
    'ni',   # 从句标记
    'šu',   # “他的”（物主代词）
    'ša',   # “她的”
    'a',    # 属格 / 主格变体
    'im',   # 属格
    'um',   # 主格
    'am',   # 宾格
    'kà',   # “你的”
    'ia',   # “我的”
    'ka',   # “你的”（变体）
    'at',   # 阴性标记
    'ti',   # “我的”（变体）
}

_HINT_STOPFORMS = {
    # 介词 / 连词
    "a-na", "um-ma", "i-na", "ù", "ú", "u",
    "ša", "ša-ma", "ki-ma", "ki-i", "ki",
    "la", "ul", "ul-ma",
    # 代词
    "a-ta", "a-ti",  # you (m/f)
    "a-tù-nu",       # you (pl)
    "a-hu-a",        # my brothers
    "a-ma", "i-ma",
    "ma-ma", "ma-ma-an",
    "a-ba", "a-bi",
    "šu-ma", "šu-ut",
    "i-li", "i-la",
    # 动词形式
    "a-dí-in",       # I gave
    "i-dí-in",       # he gave
    # 所有格（my brother / my father）
    "a-hi-a",        # my brother
    "a-bi-a",        # my father
    # 计量单位（极高频且词典里有同形专名）
    "ma-na",         # mina (重量单位)
    "ma-lá",         # 常见副词，非专名
    # 楔形文字限定词/质量词
    "sig5",          # good quality (限定词)
    # 常见商业词汇（布料/月份/法律用语）
    "ku-ta-nim",     # kutānum (布料单位，非专名)
    "sà-ra-tim",     # sarrātim (月份名，非专名)
    "ba-a-nim",      # bā'ānim (律师/代理人，非专名)
    "a-le-e",        # alê (疑问词"where is"，非专名)
    "am-ra-ma",      # amrāma (祝愿词"look!"，非专名)
    "a-lá-ni",       # alānē (城市复数，非专名)
}

# 月份音译形式 → 月份编号（官方月份表）
# 用于在音译中识别月名形式并生成 hint
_MONTH_TRANSLITERATION_FORMS = {
    # Month 2: ša-sarratim
    'ša sá-ra-tim': 2,
    # Month 3: Kenātim
    'ke-na-tim': 3,
    # Month 4: Mahur-ilī
    'ma-hu-ur-dingir': 4, 'ma-ḫu-ur-i-lí': 4,
    # Month 5: Abšarrani
    'áb-ša-ra-ni': 5, 'áb ša-ra-ni': 5, 'áb-ša-ra-nu': 5,
    # Month 6: Hubur
    'hu-bu-ur': 6,
    # Month 7: Ṣip'um
    'ṣí-ip-im': 7,
    # Month 8: Qarrātum
    'qá-ra-a-tí': 8, 'qá-ra-a-tim': 8,
    # Month 9: Kanwarta
    'kán-bar-ta': 9, 'kà-an-ma-ar-ta': 9,
    # Month 10: Te'inātum
    'té-i-na-tim': 10,
    # Month 11: Kuzallum
    'ku-zal-li': 11, 'ku-zal-lu': 11,
    # Month 12: Allanātum
    'a-lá-na-tum': 12, 'a-lá-na-tim': 12,
}

def _apply_month_hints(text: str) -> str:
    """检测音译中的阿卡德语月名形式，添加月份 hint。
    
    例如: 音译含 'ke-na-tim' → 添加 ## MONTH: ke-na-tim=month 3
    """
    if not isinstance(text, str) or not text.strip():
        return text
    text_lower = text.lower()
    month_hints = []
    for form, month_num in _MONTH_TRANSLITERATION_FORMS.items():
        if form.lower() in text_lower:
            month_hints.append(f"{form}=month {month_num}")
    if month_hints:
        # 附加到文本末尾（与 HINTS 格式类似但独立标记）
        if '## MONTH:' not in text:
            text = f"{text} ## MONTH: {'; '.join(month_hints)}"
    return text

def _apply_hints(text: str, lexicon: dict, max_hints: int = None) -> str:
    """基于限定词（Determinatives）的精准 Hint 系统
    
    只提取被限定词明确标记的专名，避免把动词词根误认为人名。
    支持的限定词：{d}(神名)、{ki}(地名)、DUMU(父名)、KIŠIB(印章名)
    
    策略："宁漏掉一万，不看错一个"
    """
    if not isinstance(text, str) or not text.strip():
        return text
    if "## HINTS:" in text or not lexicon:
        return text
    if max_hints is None:
        max_hints = config.max_hints_per_sample
        
    hints = []
    matched_forms = set()
    words = text.split()
    n = len(words)
    
    def _try_match(word_str):
        """Suffix-Aware 查词典：只对限定词相邻的词做后缀剥离"""
        clean = word_str.strip('.,;:?!()[]"\'>').lower()
        if not clean or len(clean) < _HINT_MIN_FORM_LEN or clean in _HINT_STOPFORMS:
            return None
        
        candidates = [clean]
        if '-' in clean:
            parts = clean.split('-')
            stem_parts = parts[:]
            for _ in range(min(2, len(parts) - 1)):
                if stem_parts[-1] in _AKKADIAN_SUFFIXES:
                    stem_parts = stem_parts[:-1]
                    stem = '-'.join(stem_parts)
                    if stem != clean:
                        candidates.append(stem)
                else:
                    break
        
        for cand in candidates:
            if len(cand) >= _HINT_MIN_FORM_LEN and cand not in _HINT_STOPFORMS:
                if cand in lexicon and cand not in matched_forms:
                    return cand
        return None
    
    for i, word in enumerate(words):
        if len(hints) >= max_hints:
            break
        
        # --- 限定词规则 ---
        
        # 1. {d}+专名：神名限定词，后面的词是神名
        if word == '{d}' and i + 1 < n:
            cand = _try_match(words[i + 1])
            if cand:
                hints.append(f"{cand}={lexicon[cand]}")
                matched_forms.add(cand)
            continue
        
        # 2. 词+{ki}：地名限定词，前面的词是地名
        if '{ki}' in word:
            place = word.replace('{ki}', '').strip('.,;:?!()[]"\'>').lower()
            if place and len(place) >= _HINT_MIN_FORM_LEN and place not in _HINT_STOPFORMS:
                if place in lexicon and place not in matched_forms:
                    hints.append(f"{place}={lexicon[place]}")
                    matched_forms.add(place)
            continue
        
        # 3. DUMU+专名："son of"，后面的词很可能是人名
        if word == 'DUMU' and i + 1 < n:
            cand = _try_match(words[i + 1])
            if cand:
                hints.append(f"{cand}={lexicon[cand]}")
                matched_forms.add(cand)
            continue
        
        # 4. KIŠIB+专名："seal of"，后面的词很可能是人名
        if word == 'KIŠIB' and i + 1 < n:
            cand = _try_match(words[i + 1])
            if cand:
                hints.append(f"{cand}={lexicon[cand]}")
                matched_forms.add(cand)
            continue

    if hints:
        return f"{text} ## HINTS: {'; '.join(hints)}"
    return text


def _apply_hints_persample(tl_text: str, tr_text: str, max_hints: int = None) -> str:
    """训练时 per-sample hint 提取：从 TL-TR 对中直接找到正确的名字拼写。
    
    比 _apply_hints (词典查表) 准确率高得多 (~95% vs ~40%)，因为直接从
    当前样本的翻译中提取名字，而不是依赖词典的固定拼写。
    
    仅在训练时使用（有 raw_target 可用）。推理时用 _apply_hints + 校准词典。
    """
    if not isinstance(tl_text, str) or not tl_text.strip():
        return tl_text
    if not isinstance(tr_text, str) or not tr_text.strip():
        return tl_text
    if "## HINTS:" in tl_text:
        return tl_text
    if max_hints is None:
        max_hints = config.max_hints_per_sample
    
    from difflib import SequenceMatcher
    
    # 提取 TR 中所有首字母大写的候选专名
    tr_names = re.findall(r'\b[A-Z][a-zāēīūšṣṭḫ]+(?:-[A-Za-zāēīūšṣṭḫ]+)*\b', tr_text)
    if not tr_names:
        return tl_text
    
    hints = []
    matched_forms = set()
    words = tl_text.split()
    n = len(words)
    
    def _find_in_tr(form_str):
        """在 TR 名字中模糊匹配 form"""
        form_syl = form_str.replace('-', '')
        best_match = None
        best_ratio = 0.0
        for name in tr_names:
            name_lower = name.lower().replace('-', '')
            ratio = SequenceMatcher(None, form_syl, name_lower).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_match = name
        if best_match and best_ratio >= 0.6:
            return best_match
        return None
    
    for i, word in enumerate(words):
        if len(hints) >= max_hints:
            break
        
        next_word_form = None
        if word in ('{d}', 'DUMU', 'KIŠIB') and i + 1 < n:
            next_word_form = words[i + 1].strip('.,;:?!()[]"\'>').lower()
        elif '{ki}' in word:
            next_word_form = word.replace('{ki}', '').strip('.,;:?!()[]"\'>').lower()
        
        if (next_word_form and len(next_word_form) >= _HINT_MIN_FORM_LEN 
                and next_word_form not in _HINT_STOPFORMS 
                and next_word_form not in matched_forms):
            match = _find_in_tr(next_word_form)
            if match:
                hints.append(f"{next_word_form}={match}")
                matched_forms.add(next_word_form)
    
    if hints:
        return f"{tl_text} ## HINTS: {'; '.join(hints)}"
    return tl_text


print("✓ Preprocessing functions defined (postprocess_output imported from prepare_data)")

# ============================================================================
# CELL 4: LOAD DATA
# ============================================================================

print("\n📂 Loading data...")

df = pd.read_csv(config.train_csv)
if 'translation' in df.columns:
    df['target'] = df['translation']
elif 'target_text' in df.columns:
    df['target'] = df['target_text']
elif 'chosen' in df.columns:
    df['target'] = df['chosen']
else:
    raise ValueError(
        "训练数据缺少可用目标列：需要 translation / target_text / chosen 之一。"
        f"当前列：{df.columns.tolist()}"
    )

# 对 DPO 成对数据优先复用已有 prompt/input_text；否则由 transliteration 组装
if 'input_text' not in df.columns:
    if 'prompt' in df.columns:
        df['input_text'] = df['prompt'].apply(lambda x: str(x) if pd.notna(x) else config.task_prefix)
    elif 'transliteration' in df.columns:
        df['input_text'] = df['transliteration'].apply(
            lambda x: config.task_prefix + str(x) if pd.notna(x) else config.task_prefix
        )
    else:
        raise ValueError(
            "训练数据缺少可用输入列：需要 input_text / prompt / transliteration 之一。"
            f"当前列：{df.columns.tolist()}"
        )
else:
    df['input_text'] = df['input_text'].apply(lambda x: str(x) if pd.notna(x) else config.task_prefix)

if 'target_text' not in df.columns:
    df['target_text'] = df['target'].apply(lambda x: str(x) if pd.notna(x) else "")
else:
    df['target_text'] = df['target_text'].apply(lambda x: str(x) if pd.notna(x) else "")

print(f"   Loaded: {len(df)} rows from {config.train_csv}")

# 手动切分数据替换：优先使用切分后的短段替换原始长文档
if config.train_code_splits and os.path.exists(config.train_code_splits):
    splits_df = pd.read_csv(config.train_code_splits)
    split_oids = set(splits_df['oare_id'].unique())
    n_replaced = df['oare_id'].isin(split_oids).sum()
    df = df[~df['oare_id'].isin(split_oids)]  # 删除被切分的原始文档
    # 添加切分后的段
    splits_df['target'] = splits_df['translation']
    df = pd.concat([df, splits_df[['oare_id', 'transliteration', 'translation', 'target']]], ignore_index=True)
    print(f"   🔀 手动切分替换: {n_replaced} 条原始文档 → {len(splits_df)} 段 (from {config.train_code_splits})")
else:
    if config.train_code_splits:
        print(f"   ℹ️ 手动切分文件不存在: {config.train_code_splits}")

# 数据已由 prepare_data.py 预处理和质量过滤
# 注意：simulate_damage 在 DataCollator 中动态应用，确保每个 epoch 看到不同的损坏

# Remove any empty rows
df = df[df['target_text'].str.len() > 0]
print(f"   After cleaning: {len(df)} rows")

# Show sample
print("\n📝 Sample data:")
print(f"   Input:  {df['input_text'].iloc[0]}")
print(f"   Target: {df['target_text'].iloc[0]}")

# 统计原始文本长度 (Character/Byte length)
def print_raw_length_stats(df, name="Raw Data"):
    print(f"\n📊 {name} 原始文本长度统计:")
    
    # 字符长度
    input_char_lens = df['input_text'].astype(str).apply(len)
    target_char_lens = df['target_text'].astype(str).apply(len)
    
    print(f"   [Char] Input:  min={input_char_lens.min()}, max={input_char_lens.max()}, avg={input_char_lens.mean():.1f}")
    print(f"   [Char] Target: min={target_char_lens.min()}, max={target_char_lens.max()}, avg={target_char_lens.mean():.1f}")

    # ByT5 Byte 长度 (UTF-8)
    input_byte_lens = df['input_text'].astype(str).apply(lambda x: len(x.encode('utf-8')))
    target_byte_lens = df['target_text'].astype(str).apply(lambda x: len(x.encode('utf-8')))
    
    print(f"   [Byte] Input:  min={input_byte_lens.min()}, max={input_byte_lens.max()}, avg={input_byte_lens.mean():.1f}")
    print(f"   [Byte] Target: min={target_byte_lens.min()}, max={target_byte_lens.max()}, avg={target_byte_lens.mean():.1f}")
    
    # 预估截断
    trunc_input = (input_byte_lens > config.max_input_length).sum()
    trunc_target = (target_byte_lens > config.max_target_length).sum()
    print(f"   预估截断 (Bytes > {config.max_input_length}/{config.max_target_length}):")
    print(f"   Input:  {trunc_input}/{len(df)} ({100*trunc_input/len(df):.1f}%)")
    print(f"   Target: {trunc_target}/{len(df)} ({100*trunc_target/len(df):.1f}%)")

print_raw_length_stats(df)

# ============================================================================
# CELL 5: CREATE DATASET
# ============================================================================

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

print("\n🔤 Loading tokenizer...")
# 直接查找本地 HF 缓存快照路径，绕过 hub 解析
import glob as _glob
_cache_hits = []
if os.path.isdir(config.model_name):
    _resolved_model_path = config.model_name
    _local_files_only = True
    print(f"   ✓ 使用本地模型目录: {_resolved_model_path}")
else:
    for _cache_root in ["/root/huggingface_cache/hub", os.path.expanduser("~/.cache/huggingface/hub")]:
        _cache_pattern = os.path.join(
            _cache_root,
            f"models--{config.model_name.replace('/', '--')}",
            "snapshots",
            "*",
            "config.json",
        )
        _cache_hits = _glob.glob(_cache_pattern)
        if _cache_hits:
            break

    if _cache_hits:
        _resolved_model_path = os.path.dirname(_cache_hits[0])
        _local_files_only = True
        print(f"   ✓ 使用本地缓存: {_resolved_model_path}")
    else:
        _resolved_model_path = config.model_name
        _local_files_only = False
        print(f"   ⚠️ 未找到本地缓存，使用 hub ID: {_resolved_model_path}")

try:
    tokenizer = AutoTokenizer.from_pretrained(_resolved_model_path, local_files_only=_local_files_only)
except Exception as e:
    tokenizer = None
    print(f"   ⚠️ 初始 tokenizer 加载失败（将延后到 DPO 阶段重试）: {e}")

# Split data
if config.num_folds > 1:
    # 多 Fold：按 oare_id 分组的 GroupKFold，防止同一文档泄露
    from sklearn.model_selection import GroupKFold
    groups = df[config.fold_group_col] if config.fold_group_col in df.columns else None
    if groups is not None:
        n_groups = groups.nunique()
        actual_folds = min(config.num_folds, n_groups)
        if actual_folds < config.num_folds:
            print(f"\n⚠️ 警告: 组数量({n_groups}) 小于 num_folds({config.num_folds})，自动将 n_splits 降为 {actual_folds}")
            if config.current_fold >= actual_folds:
                print(f"⚠️ 当前 fold({config.current_fold}) >= 实际最大 fold 数({actual_folds})，无数据可训，直接退出。")
                sys.exit(0)
        else:
            actual_folds = config.num_folds
            
        gkf = GroupKFold(n_splits=actual_folds)
        splits = list(gkf.split(df, groups=groups))
        train_idx, eval_idx = splits[config.current_fold]
        train_df = df.iloc[train_idx].copy()
        eval_df = df.iloc[eval_idx].copy()
        print(f"   GroupKFold: fold {config.current_fold+1}/{config.num_folds} (group_col={config.fold_group_col})")
    else:
        from sklearn.model_selection import KFold
        kf = KFold(n_splits=config.num_folds, shuffle=True, random_state=config.random_seed)
        splits = list(kf.split(df))
        train_idx, eval_idx = splits[config.current_fold]
        train_df = df.iloc[train_idx].copy()
        eval_df = df.iloc[eval_idx].copy()
        print(f"   KFold: fold {config.current_fold+1}/{config.num_folds} (无分组列，纯随机)")
else:
    # 单次划分（默认）— 按 oare_id 分组，防止同一文档的不同句子泄露到验证集
    if config.test_size == 0 or config.test_size is None:
        # test_size=0: 全量训练，不划分验证集
        train_df = df.copy()
        eval_df = df.sample(min(50, len(df)), random_state=config.random_seed).copy()
        print(f"   ⚠️ test_size=0: 全量训练，eval 使用 {len(eval_df)} 条采样（仅供监控，不做早停）")
    elif config.fold_group_col in df.columns:
        from sklearn.model_selection import GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=config.test_size, random_state=config.random_seed)
        train_idx, eval_idx = next(gss.split(df, groups=df[config.fold_group_col]))
        train_df = df.iloc[train_idx].copy()
        eval_df = df.iloc[eval_idx].copy()
        n_train_groups = df.iloc[train_idx][config.fold_group_col].nunique()
        n_eval_groups = df.iloc[eval_idx][config.fold_group_col].nunique()
        overlap = set(train_df[config.fold_group_col]) & set(eval_df[config.fold_group_col])
        print(f"   GroupShuffleSplit: {n_train_groups} train groups, {n_eval_groups} eval groups, overlap={len(overlap)}")
    else:
        from sklearn.model_selection import train_test_split
        train_df, eval_df = train_test_split(df, test_size=config.test_size, random_state=config.random_seed)
print(f"   Train: {len(train_df)}, Eval: {len(eval_df)}")

# 多 fold 时更新输出目录
if config.num_folds > 1:
    config.output_dir = os.path.join(config.root_dir, "output", f"model_{config._timestamp}", f"fold{config.current_fold}")
    os.makedirs(config.output_dir, exist_ok=True)
    
    print(f"   📁 Fold 输出目录: {config.output_dir}")

# Curriculum Learning: 按长度排序（短→长）
if config.use_curriculum:
    train_df['_length'] = train_df['input_text'].str.len() + train_df['target_text'].str.len()
    train_df = train_df.sort_values('_length').reset_index(drop=True)
    train_df = train_df.drop(columns=['_length'])
    print(f"   📚 Curriculum Learning: 按长度排序（短→长）")

# ============================================================================
# RAG 索引构建（BM25 词级 + TF-IDF 字符级 + RRF 融合 + 离线预计算）
# ⚠️ 严禁使用 target 端检索，否则会导致逻辑性数据泄露
# ============================================================================
_rag_index = None
_rag_cache = None  # 离线预计算的检索缓存

class HybridRAGIndex:
    """
    双路混合检索索引 + Reciprocal Rank Fusion + 离线预计算
    
    路1: BM25 (词级)     — 精确匹配 Sumerogram（KÙ.BABBAR）和专有名词
    路2: TF-IDF (字符级)  — 捕获形态变化（i-dí-in vs i-dí-nam）
    
    设计原则：
    - 不使用 Dense/Reranker（bge-m3 等现代语义模型不认识阿卡德语音译）
    - BM25 + TF-IDF 在字符/词形态层面匹配，对死语言最有效
    - 离线预计算每个训练样本的 top-K 检索结果，Collator 中 O(1) 查表
    """
    
    def __init__(self, train_df, cfg):
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer
        from rank_bm25 import BM25Okapi
        
        self.cfg = cfg
        
        # 1. 过滤干净示例：翻译足够长、无 gap 标记、有实际内容
        clean_mask = (
            (train_df['target_text'].str.len() >= cfg.rag_min_translation_len) &
            (~train_df['target_text'].str.contains('<gap>', case=False, na=False)) &
            (train_df['target_text'].str.strip().str.len() > 0) &
            (train_df['input_text'].str.strip().str.len() > len(cfg.task_prefix) + 5)
        )
        clean_df = train_df[clean_mask].reset_index(drop=True)
        
        # 2. 提取纯文本（去 task_prefix 的音译 + 英文翻译）
        self.transliterations = []
        self.translations = []
        for _, row in clean_df.iterrows():
            inp = str(row['input_text'])
            if inp.startswith(cfg.task_prefix):
                inp = inp[len(cfg.task_prefix):]
            tgt = str(row['target_text']).strip()
            words = tgt.split()
            if len(words) > 3 and len(set(words)) / len(words) < 0.3:
                continue
            self.transliterations.append(inp.strip())
            self.translations.append(tgt)
        
        n_examples = len(self.transliterations)
        
        # 3a. BM25 索引（词级，捕获 Sumerogram 和专名精确匹配）
        tokenized_corpus = [doc.split() for doc in self.transliterations]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 3b. TF-IDF 索引（字符级 n-gram，捕获形态/结构相似性）
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=8000,
            ngram_range=(3, 6),
            analyzer='char_wb',
            sublinear_tf=True,
        )
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.transliterations)
        
        print(f"   🔍 RAG 索引 (BM25+TF-IDF): {n_examples}/{len(train_df)} 条干净示例"
              f" (BM25 vocab~{len(self.bm25.idf)}, TF-IDF dim={self.tfidf_matrix.shape[1]})")
    
    def _rrf_fuse(self, rankings: list, k: int = 60) -> dict:
        """Reciprocal Rank Fusion：融合多路排序"""
        fused = {}
        for ranking in rankings:
            for rank, idx in enumerate(ranking):
                fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank + 1)
        return fused
    
    def retrieve(self, query_source: str, top_k: int = 3, exclude_source: str = None) -> list:
        """基于源语言（阿卡德语）检索最相似的 top_k 条示例"""
        if not query_source or not query_source.strip():
            return []
        
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        
        candidate_pool = top_k * 10
        all_rankings = []
        
        # 路1: BM25 词级检索
        bm25_scores = self.bm25.get_scores(query_source.split())
        bm25_ranking = np.argsort(bm25_scores)[::-1][:candidate_pool]
        all_rankings.append(bm25_ranking.tolist())
        
        # 路2: TF-IDF 字符级检索
        query_vec = self.tfidf_vectorizer.transform([query_source])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()
        tfidf_ranking = np.argsort(tfidf_scores)[::-1][:candidate_pool]
        all_rankings.append(tfidf_ranking.tolist())
        
        # RRF 融合
        fused_scores = self._rrf_fuse(all_rankings)
        sorted_candidates = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        
        # 收集结果（应用排除、质量过滤、字节预算）
        results = []
        budget = int(self.cfg.max_input_length * self.cfg.rag_budget_ratio)
        used_bytes = 0
        
        for idx, rrf_score in sorted_candidates:
            if len(results) >= top_k:
                break
            
            translit = self.transliterations[idx]
            trans = self.translations[idx]
            
            if exclude_source and translit.strip() == exclude_source.strip():
                continue
            
            # 质量门控
            has_bm25 = bm25_scores[idx] >= 0.5
            has_tfidf = tfidf_scores[idx] >= 0.15
            if not (has_bm25 or has_tfidf):
                continue
            
            # 字节预算（自然序列格式）
            example_str = f"{translit} => {trans}"
            example_bytes = len(example_str.encode('utf-8'))
            
            if example_bytes > self.cfg.rag_max_example_bytes:
                continue  # 宁缺毋滥：超长示例直接跳过，不截断加 ...（防止模型学到省略输出）
            
            if used_bytes + example_bytes > budget:
                break
            
            results.append({'source': translit, 'target': trans})
            used_bytes += example_bytes
        
        return results
    
    def precompute_cache(self, train_df):
        """离线预计算：为每个训练样本预先检索 top-K，训练时 O(1) 查表"""
        cache = {}
        top_k = self.cfg.rag_max_examples
        
        print(f"   📦 预计算 RAG 缓存 ({len(train_df)} 条, top_k={top_k})...")
        for idx, row in train_df.iterrows():
            inp = str(row['input_text'])
            if inp.startswith(self.cfg.task_prefix):
                inp = inp[len(self.cfg.task_prefix):]
            inp = inp.strip()
            if not inp:
                continue
            examples = self.retrieve(inp, top_k=top_k, exclude_source=inp)
            if examples:
                cache[inp] = examples
            if (idx + 1) % 500 == 0:
                print(f"      {idx+1}/{len(train_df)} ({len(cache)} cached)")
        
        print(f"   ✓ RAG 缓存完成: {len(cache)} 条 (命中率 {len(cache)/len(train_df)*100:.1f}%)")
        return cache

if config.use_rag:
    _rag_index = HybridRAGIndex(train_df, config)
    _rag_cache = _rag_index.precompute_cache(train_df)
else:
    _rag_index = None
    _rag_cache = None
    print("   📌 RAG 示例注入: 关闭")

def build_dpo_dataset(frame, tokenizer=None):
    """构建 DPO 所需的 prompt/chosen/rejected。优先使用显式 rejected。
    
    注意：TRL DPOTrainer 会自动添加 EOS 到 chosen/rejected 末尾，
    但 ByT5 的 tokenization 会导致 prompt 单独分词时末尾有 EOS，
    而 prompt+chosen 分词时中间没有 EOS，造成提取偏移错误。
    
    解决方案：确保 chosen/rejected 以 EOS 结尾，让 TRL 的 add_eos 函数不再重复添加。
    这样 prompt+chosen 的分词结果会在正确位置包含 EOS。
    """
    if "prompt" in frame.columns:
        prompts = frame["prompt"].astype(str).tolist()
    else:
        prompts = frame["input_text"].astype(str).tolist()

    if "chosen" in frame.columns:
        chosen = frame["chosen"].astype(str).tolist()
    else:
        chosen = frame["target_text"].astype(str).tolist()

    if "rejected" in frame.columns:
        rejected = frame["rejected"].astype(str).tolist()
    else:
        # 兜底：若数据中无 rejected，则退回到旧的"循环移位负样本"策略
        if len(chosen) <= 1:
            rejected = [(chosen[0] + " .") if chosen else "."]
        else:
            rejected = chosen[1:] + chosen[:1]
            for i in range(len(chosen)):
                if rejected[i] == chosen[i]:
                    rejected[i] = chosen[(i + 2) % len(chosen)]

    dpo_df = pd.DataFrame({"prompt": prompts, "chosen": chosen, "rejected": rejected})
    dpo_df = dpo_df[
        (dpo_df["chosen"].astype(str).str.strip().str.len() > 0)
        & (dpo_df["rejected"].astype(str).str.strip().str.len() > 0)
    ].copy()

    # 保底再过滤一次：去掉 chosen/rejected 完全一致样本
    chosen_norm = dpo_df["chosen"].astype(str).map(lambda x: re.sub(r"\s+", " ", x.strip()))
    rejected_norm = dpo_df["rejected"].astype(str).map(lambda x: re.sub(r"\s+", " ", x.strip()))
    same_mask = chosen_norm == rejected_norm
    same_count = int(same_mask.sum())
    if same_count > 0:
        print(f"   ℹ️ build_dpo_dataset 过滤掉 {same_count} 条 chosen==rejected 样本")
    dpo_df = dpo_df[~same_mask].reset_index(drop=True)

    # 关键修复：确保 chosen/rejected 以 EOS token 结尾
    # 这样 TRL 的 add_eos 不会重复添加，且 prompt+chosen 的分词会在正确位置有 EOS
    if tokenizer is not None and tokenizer.eos_token:
        eos = tokenizer.eos_token
        dpo_df["chosen"] = dpo_df["chosen"].apply(lambda x: x if x.endswith(eos) else x + eos)
        dpo_df["rejected"] = dpo_df["rejected"].apply(lambda x: x if x.endswith(eos) else x + eos)
        print(f"   🔧 已确保 chosen/rejected 以 EOS ('{eos}') 结尾")

    return Dataset.from_pandas(dpo_df.reset_index(drop=True), preserve_index=False)




def run_dpo_training(train_frame, eval_frame):
    """DPO training for encoder-decoder models (ByT5).
    
    TRL 0.29.0 removed encoder-decoder support, so we use a custom implementation.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSeq2SeqLM, AutoConfig, AutoTokenizer, get_cosine_schedule_with_warmup
    from datasets import Dataset
    import pandas as pd
    from tqdm import tqdm
    
    print("\n" + "=" * 60)
    print("🚀 STARTING DPO TRAINING (Custom Encoder-Decoder)")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   Device: {device}")
    
    model_id = _resolved_model_path
    local_files_only = os.path.isdir(model_id)
    model_cfg = AutoConfig.from_pretrained(model_id, local_files_only=local_files_only)
    
    # Check if encoder-decoder
    is_enc_dec = getattr(model_cfg, "is_encoder_decoder", False)
    print(f"   Model type: {'encoder-decoder' if is_enc_dec else 'decoder-only'}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=local_files_only)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimization
    load_dtype = torch.bfloat16 if (config.use_bf16 and torch.cuda.is_available()) else torch.float32
    policy_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    ).to(device)
    
    # Reference model on GPU for speed (reduce batch size if OOM)
    ref_model = AutoModelForSeq2SeqLM.from_pretrained(
        model_id,
        torch_dtype=load_dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
    print(f"   ✓ Reference model loaded on GPU")
    
    # Enable gradient checkpointing for memory efficiency
    if config.gradient_checkpointing and hasattr(policy_model, "gradient_checkpointing_enable"):
        policy_model.gradient_checkpointing_enable()
        print(f"   ✓ Gradient checkpointing enabled")
    
    # Build datasets
    def build_dpo_dataset_enc_dec(frame, tokenizer):
        """Build dataset for encoder-decoder DPO."""
        if "prompt" in frame.columns:
            prompts = frame["prompt"].astype(str).tolist()
        else:
            prompts = frame["input_text"].astype(str).tolist()
        
        if "chosen" in frame.columns:
            chosen = frame["chosen"].astype(str).tolist()
        else:
            chosen = frame["target_text"].astype(str).tolist()
        
        if "rejected" in frame.columns:
            rejected = frame["rejected"].astype(str).tolist()
        else:
            if len(chosen) <= 1:
                rejected = [(chosen[0] + " .") if chosen else "."]
            else:
                rejected = chosen[1:] + chosen[:1]
        
        data = []
        n_same = 0
        n_too_similar = 0
        for p, c, r in zip(prompts, chosen, rejected):
            c_norm = re.sub(r"\s+", " ", c.strip())
            r_norm = re.sub(r"\s+", " ", r.strip())
            if c_norm == r_norm or not c.strip() or not r.strip():
                n_same += 1
                continue
            # P1: 过滤 Jaccard > 0.8 的对（信号太弱，浪费梯度）
            c_words = set(c_norm.lower().split())
            r_words = set(r_norm.lower().split())
            union = c_words | r_words
            jaccard = len(c_words & r_words) / len(union) if union else 1.0
            if jaccard > 0.8:
                n_too_similar += 1
                continue
            data.append({
                "prompt": p,
                "chosen": c.strip(),
                "rejected": r.strip(),
            })
        if n_same > 0:
            print(f"   ℹ️ 过滤 chosen==rejected: {n_same} 条")
        if n_too_similar > 0:
            print(f"   ℹ️ 过滤 Jaccard>0.8（信号太弱）: {n_too_similar} 条")
        
        return Dataset.from_pandas(pd.DataFrame(data))
    
    train_dataset = build_dpo_dataset_enc_dec(train_frame, tokenizer)
    eval_dataset = build_dpo_dataset_enc_dec(eval_frame, tokenizer)
    
    print(f"   Train samples: {len(train_dataset)}")
    print(f"   Eval samples: {len(eval_dataset)}")
    print(f"   Beta: {config.dpo_beta}")
    print(f"   Learning rate: {config.learning_rate}")
    print(f"   Batch size: {config.dpo_batch_size}")
    print(f"   Gradient accumulation: {config.dpo_gradient_accumulation}")
    print(f"   Num epochs: {config.num_epochs}")
    
    # Data collator - use labels for T5, model will handle _shift_right internally
    def collate_fn(batch, tokenizer, max_length=512):
        prompts = [item["prompt"] for item in batch]
        chosen = [item["chosen"] for item in batch]
        rejected = [item["rejected"] for item in batch]
        
        # Encoder inputs
        enc_inputs = tokenizer(
            prompts, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt", add_special_tokens=True,    # <--- 改成 True
        )
        
        # For T5 labels: let tokenizer add proper EOS token ID (not string "</s>")
        # ByT5 is byte-level, so "</s>" string would be tokenized as 4 bytes, not as EOS token ID 1
        chosen_labels = tokenizer(
            chosen, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt", add_special_tokens=True,  # Let tokenizer add EOS
        )
        rejected_labels = tokenizer(
            rejected, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt", add_special_tokens=True,  # Let tokenizer add EOS
        )
        
        # Replace pad_token_id with -100 for loss masking
        chosen_labels_input_ids = chosen_labels["input_ids"].masked_fill(
            chosen_labels["input_ids"] == tokenizer.pad_token_id,
            -100,
        )
        rejected_labels_input_ids = rejected_labels["input_ids"].masked_fill(
            rejected_labels["input_ids"] == tokenizer.pad_token_id,
            -100,
        )
        
        return {
            "encoder_input_ids": enc_inputs["input_ids"],
            "encoder_attention_mask": enc_inputs["attention_mask"],
            "chosen_labels": chosen_labels_input_ids,
            "rejected_labels": rejected_labels_input_ids,
        }
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.dpo_batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=0,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config.dpo_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=0,
    )
    
    num_training_steps = len(train_loader) * config.num_epochs
    warmup_steps = int(num_training_steps * config.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    
    def get_log_probs(model, encoder_input_ids, encoder_attention_mask, labels):
        """Compute log probabilities for T5 encoder-decoder model.
        
        For T5/ByT5:
        - labels: target tokens [batch, seq_len]
        - model internally shifts labels right with decoder_start_token_id
        - When labels=labels is passed, logits[i] already predicts labels[i]
        - NO need for manual shift like Causal LM!
        """
        # Forward pass - T5 will internally create decoder_input_ids via _shift_right
        outputs = model(
            input_ids=encoder_input_ids,
            attention_mask=encoder_attention_mask,
            labels=labels,
        )
        
        # outputs.logits shape: [batch, seq_len, vocab_size]
        # For T5 with labels=labels, logits[i] already predicts labels[i] (aligned!)
        logits = outputs.logits
        
        # Create mask for valid (non -100) labels
        valid_mask = (labels != -100).float()
        
        # NO SHIFT NEEDED - logits and labels are already aligned
        # Replace -100 with 0 for gathering (will be masked out anyway)
        labels_gather = torch.where(labels == -100, torch.zeros_like(labels), labels)
        
        # Compute log probabilities
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, labels_gather.unsqueeze(-1)).squeeze(-1)
        
        # Mask out padding positions
        token_log_probs = token_log_probs * valid_mask
        
        # 不做长度归一化：DPO 的 chosen-rejected 差值会被 ref 差值抵消长度偏差，
        # 归一化反而会压缩梯度信号（尤其当 chosen/rejected 长度接近时）。
        return token_log_probs.sum(dim=1)
    
    def dpo_loss(policy_logps, ref_logps, beta=0.1):
        """Compute DPO loss with corrected accuracy and reward margin tracking."""
        policy_chosen, policy_rejected = policy_logps
        ref_chosen, ref_rejected = ref_logps
        
        logits = beta * ((policy_chosen - policy_rejected) - (ref_chosen - ref_rejected))
        loss = -F.logsigmoid(logits).mean()
        
        # 修正: 用 margin (而非绝对 log prob) 判断 accuracy
        accuracy = (logits > 0).float().mean()
        reward_margin = (policy_chosen - policy_rejected).mean().item()
        
        return loss, accuracy, reward_margin
    
    def evaluate_generation(model, eval_dataloader):
        """Run generation on eval set and compute BLEU/chrF++"""
        try:
            import sacrebleu
            gen_preds, gen_refs = [], []
            model.eval()
            with torch.no_grad():
                for gen_batch in tqdm(eval_dataloader, desc="GenEval", leave=False):
                    enc = gen_batch["encoder_input_ids"].to(device)
                    mask = gen_batch["encoder_attention_mask"].to(device)
                    outputs = model.generate(
                        input_ids=enc, attention_mask=mask,
                        max_new_tokens=config.max_target_length,
                        num_beams=config.num_beams,
                        length_penalty=config.length_penalty,
                    )
                    decoded = decode_sequences(outputs, tokenizer=tokenizer)
                    gen_preds.extend([postprocess_output(d) for d in decoded])
                    chosen_ids = gen_batch["chosen_labels"].clone()
                    chosen_ids[chosen_ids == -100] = tokenizer.pad_token_id
                    refs = decode_sequences(chosen_ids, tokenizer=tokenizer)
                    gen_refs.extend(refs)
            
            eval_bleu = sacrebleu.corpus_bleu(gen_preds, [gen_refs]).score
            eval_chrf = sacrebleu.corpus_chrf(gen_preds, [gen_refs], word_order=2).score
            eval_geom = math.sqrt(max(eval_bleu, 0.01) * max(eval_chrf, 0.01))
            return eval_bleu, eval_chrf, eval_geom
        except Exception as e:
            print(f"   [GenEval] Skipped: {e}")
            return None, None, None

    gen_eval_loader = DataLoader(
        eval_dataset, batch_size=config.dpo_eval_batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        num_workers=0,
    )

    # Initial evaluation before training
    print("\n   [GenEval] Running initial generation evaluation (Epoch 0)...")
    init_bleu, init_chrf, init_geom = evaluate_generation(policy_model, gen_eval_loader)
    if init_chrf is not None:
        print(f"   [GenEval] Pre-train BLEU={init_bleu:.2f}, chrF++={init_chrf:.2f}, geom={init_geom:.2f}")
    
    # Training loop
    global_step = 0
    best_loss = float("-inf") # Because we use chrf to save best model, larger is better
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        policy_model.train()
        epoch_loss = 0.0
        epoch_accuracy = 0.0
        num_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
        
        for batch in progress_bar:
            encoder_input_ids = batch["encoder_input_ids"].to(device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(device)
            chosen_labels = batch["chosen_labels"].to(device)
            rejected_labels = batch["rejected_labels"].to(device)
            
            # Policy log probs
            policy_chosen_logp = get_log_probs(
                policy_model, encoder_input_ids, encoder_attention_mask,
                chosen_labels
            )
            policy_rejected_logp = get_log_probs(
                policy_model, encoder_input_ids, encoder_attention_mask,
                rejected_labels
            )
            
            # Reference log probs
            with torch.no_grad():
                ref_chosen_logp = get_log_probs(
                    ref_model, encoder_input_ids, encoder_attention_mask,
                    chosen_labels
                )
                ref_rejected_logp = get_log_probs(
                    ref_model, encoder_input_ids, encoder_attention_mask,
                    rejected_labels
                )
            
            loss, accuracy, reward_margin = dpo_loss(
                (policy_chosen_logp, policy_rejected_logp),
                (ref_chosen_logp, ref_rejected_logp),
                beta=config.dpo_beta
            )
            
            loss = loss / config.dpo_gradient_accumulation
            loss.backward()
            
            if (global_step + 1) % config.dpo_gradient_accumulation == 0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * config.dpo_gradient_accumulation
            epoch_accuracy += accuracy.item()
            num_batches += 1
            global_step += 1
            
            progress_bar.set_postfix({
                "loss": f"{loss.item() * config.dpo_gradient_accumulation:.4f}",
                "acc": f"{accuracy.item():.2%}",
                "margin": f"{reward_margin:.3f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            if global_step % config.logging_steps == 0:
                print(f"\n   Step {global_step}: loss={loss.item() * config.dpo_gradient_accumulation:.4f}, acc={accuracy.item():.2%}, margin={reward_margin:.3f}")
        
        # 刷出 epoch 末尾残留梯度（batch 数无法被 grad_accum 整除时）
        if global_step % config.dpo_gradient_accumulation != 0:
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_loss = epoch_loss / max(1, num_batches)
        avg_accuracy = epoch_accuracy / max(1, num_batches)
        print(f"\n   [Train] Epoch {epoch+1} completed: avg_loss={avg_loss:.4f}, avg_acc={avg_accuracy:.2%}")
        
        # ==========================================
        # === EVAL (验证集) 监控循环 ===
        # ==========================================
        policy_model.eval()
        eval_loss_total = 0.0
        eval_acc_total = 0.0
        eval_batches = 0
        
        print(f"   [Eval] Running evaluation on {len(eval_loader)} batches...")
        with torch.no_grad():
            for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
                encoder_input_ids = batch["encoder_input_ids"].to(device)
                encoder_attention_mask = batch["encoder_attention_mask"].to(device)
                chosen_labels = batch["chosen_labels"].to(device)
                rejected_labels = batch["rejected_labels"].to(device)
                
                # Policy log probs
                policy_chosen_logp = get_log_probs(policy_model, encoder_input_ids, encoder_attention_mask, chosen_labels)
                policy_rejected_logp = get_log_probs(policy_model, encoder_input_ids, encoder_attention_mask, rejected_labels)
                
                # Reference log probs
                ref_chosen_logp = get_log_probs(ref_model, encoder_input_ids, encoder_attention_mask, chosen_labels)
                ref_rejected_logp = get_log_probs(ref_model, encoder_input_ids, encoder_attention_mask, rejected_labels)
                
                loss, accuracy, reward_margin = dpo_loss(
                    (policy_chosen_logp, policy_rejected_logp),
                    (ref_chosen_logp, ref_rejected_logp),
                    beta=config.dpo_beta
                )
                
                eval_loss_total += loss.item()
                eval_acc_total += accuracy.item()
                eval_batches += 1
        
        avg_eval_loss = eval_loss_total / max(1, eval_batches)
        avg_eval_accuracy = eval_acc_total / max(1, eval_batches)
        print(f"   [Eval] Epoch {epoch+1} results: eval_loss={avg_eval_loss:.4f}, eval_acc={avg_eval_accuracy:.2%}")
        
        # === 翻译质量评估 (每 epoch 做一次生成评估) ===
        eval_bleu, eval_chrf, eval_geom = evaluate_generation(policy_model, gen_eval_loader)
        if eval_chrf is not None:
            print(f"   [GenEval] BLEU={eval_bleu:.2f}, chrF++={eval_chrf:.2f}, geom={eval_geom:.2f}")
        
        # === 保存逻辑：优先用翻译质量，回退到 DPO loss ===
        current_metric = eval_chrf if eval_chrf is not None else -avg_eval_loss
        metric_improved = (eval_chrf is not None and current_metric > best_loss) or \
                          (eval_chrf is None and avg_eval_loss < best_loss)
        if epoch == 0:
            best_loss = current_metric
            metric_improved = True
        
        if metric_improved:
            best_loss = current_metric
            patience_counter = 0
            save_path = os.path.join(config.output_dir, f"checkpoint_epoch{epoch+1}")
            os.makedirs(save_path, exist_ok=True)
            policy_model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)
            print(f"   ⭐ New best model! Saved checkpoint to {save_path} (eval_loss: {avg_eval_loss:.4f})")
            
            # 维护 save_total_limit
            if hasattr(config, 'save_total_limit') and config.save_total_limit is not None:
                import glob
                import shutil
                # 查找所有 checkpoint 目录
                checkpoints = glob.glob(os.path.join(config.output_dir, "checkpoint_epoch*"))
                # 按修改时间排序（从旧到新）
                checkpoints.sort(key=os.path.getmtime)
                # 删除多余的旧 checkpoint
                if len(checkpoints) > config.save_total_limit:
                    for old_ckpt in checkpoints[:-config.save_total_limit]:
                        try:
                            shutil.rmtree(old_ckpt)
                            print(f"   🧹 Removed old checkpoint: {old_ckpt}")
                        except Exception as e:
                            print(f"   ⚠️ Failed to remove {old_ckpt}: {e}")
        else:
            patience_counter += 1
            print(f"   ⚠️ No improvement. Patience: {patience_counter}/{config.early_stopping_patience}")
            if patience_counter >= config.early_stopping_patience:
                print(f"   🛑 Early stopping triggered after {patience_counter} epochs without improvement")
                break
    
    print(f"\n✅ Training completed. Best checkpoint(s) kept under {config.output_dir}")
    
    return policy_model


# ============================================================================
# CALL DPO TRAINING
# ============================================================================
run_dpo_training(train_df, eval_df)
print("\n✅ DPO training completed.")
