#!/usr/bin/env python3
"""
数据准备脚本：预处理 + 质量过滤

功能：
1. 加载原始 train.csv
2. 应用与 train.py 完全一致的预处理（Gap标准化、字符规范化、分数转换等）
3. 质量过滤

输出（均已预处理）：
- data/train_clean.csv：官方-only 的 clean 数据（合并人工审校）
- data/train_clean_ocr_merged.csv：official + OCR 的 merged clean 数据（合并人工审校）
- data/train_sliding.csv：长文档滑动窗口数据
"""

import pandas as pd
import numpy as np
import os
import re
import unicodedata
from typing import List, Dict, Tuple, Set
from datetime import datetime
from collections import Counter
from difflib import SequenceMatcher
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端


# ============================================================
# 配置
# ============================================================

class Config:
    # 本地路径
    _local_root = "/data/lsb/deep_past"
    root_dir = _local_root
    # 本地有 data 子目录
    data_dir = os.path.join(root_dir, "data")
    
    # 输入文件
    # train_csv = os.path.join(data_dir, "train.csv")
    train_csv = "/data/lsb/deep_past/data/archibab/archibab_akk_en.csv"
    # train_csv = "/data/lsb/deep_past/output/akt6_mineru_ocr/train.csv"
    # train_csv = "/data/lsb/deep_past/data/data_k/train.csv"

    train_csv_ocr_merged = os.path.join(data_dir, "train_ocr_merged.csv")
    # 撇号恢复参考：原始版 train.csv（官方改进版误删了撇号）
    train_csv_original_v1 = os.path.join(data_dir, "train_original_v1.csv")
    # data_v2 回填参考：data_final 截断的翻译从 v2 回填
    train_csv_v2 = os.path.join(root_dir, "data_v2", "train.csv")
    # v2 TL 回填参考：v3 截断了 ~151 条 TL（hard cut ~966 chars），v2 有完整版（max=4057）
    # 注意：v1 原始也是截断的（max=932），只有 v2 有完整 TL
    train_csv_tl_source = os.path.join(root_dir, "data_v2", "train.csv")
    # 输出文件
    output_clean = os.path.join(data_dir, "train_clean.csv")
    output_clean_ocr_merged = os.path.join(data_dir, "train_clean_ocr_merged.csv")
    
    # 统计图输出目录
    output_dir = os.path.join(root_dir, "output")
    stats_plot = os.path.join(output_dir, "data_stats.png")
    
    # Silver 无标注数据
    published_texts_csv = os.path.join(data_dir, "published_texts.csv")
    output_silver = os.path.join(data_dir, "silver_unlabeled.csv")
    output_sliding = os.path.join(data_dir, "train_sliding.csv")
    cr_review_csv = os.path.join(data_dir, "train_clean_suspect_samples.csv")
    
    # 质量过滤参数
    min_translation_length = 10
    max_length_ratio = 3.5   # 处理后 tl_chars/tr_chars 上限（放宽：3.0→3.5 保留更多边界样本）
    min_length_ratio = 0.1    # 处理后 tl_chars/tr_chars 下限（放宽：0.15→0.1）
    max_raw_word_ratio = 6    # 原始 tl_words/tr_words 上限（放宽：5→6）
    
    # 文档长度阈值：用于区分 doc_short / doc_long
    max_doc_chars_for_split = 256
    
    # 任务前缀（与 train.py 一致）
    task_prefix = "translate Akkadian to English: "
    
    lexicon_csv = os.path.join(data_dir, "OA_Lexicon_eBL.csv")


config = Config()


def normalize_data_source_label(value) -> str:
    text = str(value).strip().lower()
    if text == "ocr":
        return "ocr"
    return "official"


def _parse_allowed_chars(block: str) -> Set[str]:
    return {line.strip() for line in block.splitlines() if line.strip()}


OFFICIAL_TRANSLITERATION_ALLOWED_CHARS = _parse_allowed_chars(
    """
    -
    a
    A
    i
    I
    u
    U
    m
    M
    š
    Š
    n
    N
    b
    B
    r
    R
    t
    T
    l
    L
    k
    K
    G
    g
    í
    Í
    D
    d
    Ù
    ù
    á
    Á
    .
    ú
    Ú
    p
    P
    e
    E
    h
    H
    q
    Q
    1
    ṣ
    Ṣ
    é
    É
    <
    >
    à
    À
    4
    z
    Z
    s
    S
    ì
    Ì
    5
    _
    2
    0
    ½
    w
    W
    3
    {
    }
    ṭ
    Ṭ
    6
    ⅓
    8
    ⅔
    7
    è
    È
    ⅚
    9
    ¼
    !
    +
    ⅙
    ı
    …
    ş
    İ
    :
    """
)


OFFICIAL_TRANSLATION_ALLOWED_CHARS = _parse_allowed_chars(
    """
    '
    ?
    e
    E
    a
    A
    i
    I
    t
    T
    n
    N
    s
    S
    o
    O
    r
    R
    l
    L
    h
    H
    u
    U
    m
    M
    d
    D
    F
    f
    š
    Š
    -
    p
    P
    w
    W
    b
    B
    g
    G
    y
    Y
    .
    K
    k
    ,
    C
    c
    v
    ā
    1
    )
    (
    <
    >
    z
    Z
    _
    Q
    q
    2
    ī
    ṭ
    Ṭ
    0
    :
    3
    ½
    5
    ;
    x
    ē
    4
    ū
    6
    ṣ
    Ṣ
    ⅓
    8
    ’
    !
    7
    j
    J
    ⅔
    “
    ”
    9
    –
    ⅚
    ¼
    ⅙
    "
    ‘
    ı
    —
    [
    ]
    ğ
    â
    +
    à
    ş
    """
)


def filter_to_official_allowed_characters(text: str, is_transliteration: bool = True) -> str:
    """最终字符封口：仅保留官方允许字符和空白。"""
    if not isinstance(text, str):
        return ""
    allowed_chars = (
        OFFICIAL_TRANSLITERATION_ALLOWED_CHARS
        if is_transliteration
        else OFFICIAL_TRANSLATION_ALLOWED_CHARS
    )
    text = unicodedata.normalize("NFC", text)
    filtered = []
    for char in text:
        if char in allowed_chars:
            filtered.append(char)
        elif char.isspace():
            filtered.append(" ")
        else:
            filtered.append(" ")
    return re.sub(r"\s+", " ", "".join(filtered)).strip()


# ============================================================
# 词典增强（Inline Dictionary Prompting）
# ============================================================

# 全局词典缓存
_lexicon_dict = None

def load_lexicon(lexicon_path: str = None) -> Dict[str, str]:
    """加载 OA_Lexicon_eBL.csv 构建 form → norm 映射表"""
    global _lexicon_dict
    if _lexicon_dict is not None and lexicon_path is None:
        return _lexicon_dict
    
    target_path = lexicon_path if lexicon_path else config.lexicon_csv
    
    if not os.path.exists(target_path):
        print(f"   ⚠️ 词典文件不存在: {target_path}")
        if lexicon_path is None: # Only cache empty result if using default path
            _lexicon_dict = {}
        return {}
    
    df = pd.read_csv(target_path)
    
    # 只保留 PN (人名), GN (地名), DN (神名)
    df = df[df['type'].isin(['PN', 'GN', 'DN'])]
    
    # 构建映射表：form → norm
    _lexicon_dict = {}
    for _, row in df.iterrows():
        # 1. 规范化 Key (用于匹配)
        form = str(row['form']).strip()
        form_norm = normalize_characters(form.lower(), is_transliteration=True)
        # 关键：也必须清理噪声（如括号），否则无法匹配到已清洗的输入文本
        # 例如词典 Key 为 "(d)IŠKUR"，输入文本被清洗为 "d IŠKUR"，不一致会导致匹配失败
        try:
            form_norm = clean_transliteration_noise(form_norm)
        except NameError:
            # 防止 clean_transliteration_noise 未定义（尽管通常不会发生）
            pass
        
        # 2. 规范化 Value (用于显示在 HINTS 中)
        # 必须同时调用 clean_translation_noise 确保不含括号，
        # 且调用 normalize_characters 确保字符集纯净 (ḫ -> h 等)
        norm = str(row['norm']).strip()
        norm_clean = normalize_characters(norm, is_transliteration=False)
        norm_clean = clean_translation_noise(norm_clean).strip('.,;:?! ')
        
        if form_norm and norm_clean and form_norm != 'nan' and norm_clean != 'nan':
            if form_norm not in _lexicon_dict:
                _lexicon_dict[form_norm] = norm_clean
    
    _lexicon_dict = dict(sorted(_lexicon_dict.items(), key=lambda x: len(x[0]), reverse=True))
    
    print(f"   📚 加载词典: {len(_lexicon_dict)} 条专名映射 (Key/Value 均已规范化)")
    return _lexicon_dict


# ============================================================
# 占位符服务（Dynamic Masking Pipeline）
# ============================================================

class PlaceholderService:
    """
    动态掩码管道：将专有名词替换为占位符
    - [[E000]], [[E001]], ... 用于专有名词
    - 数字/分数不掩码：ByT5 只需 1-2 token 即可生成，掩码反而增加困惑度
    
    ⚠️ Gap 标记不做占位符化！
    - Gap 是"状态描述"而非"实体"
    - 原文/译文 Gap 数量常不对等（40.7% 样本）
    - 让模型在 Byte 级别学习 <gap> 字符串
    
    专名匹配逻辑：
    - lexicon: {transliteration_form → english_norm} 字典
    - 在音译侧用 key 匹配，在翻译侧用 value 匹配
    - 同一实体共享占位符标签
    """
    
    # 常见阿卡德语功能词，不应被匹配为专名
    _STOP_FORMS = {
        'i-na', 'a-na', 'ša', 'la', 'ul', 'ma', 'u', 'ki', 'šu',
        'i-a', 'a-bi', 'um-ma', 'li', 'lu', 'ta', 'ni', 'e', 'ú',
        'im', 'an', 'il', 'en', 'be', 'a-ša', 'i-ša', 'ša-a',
        'a-tù', 'a-tú', 'at', 'it', 'ut', 'ib', 'ub', 'ab',
        'ma-ma-an', 'iš-tù', 'lá', 'ú-lá', 'šu-ma', 'ki-ma',
        'a-wa-at', 'a-nim', 'a-lim', 'ša-ma', 'na-áš',
    }
    MIN_FORM_LEN = 8  # 最短匹配长度（音译形式），过滤过短的误匹配
    
    def __init__(self, onomasticon: Set[str] = None, lexicon: Dict[str, str] = None):
        self.onomasticon = onomasticon or set()
        self.lexicon = lexicon or {}
        # 过滤：排除停用词和过短的形式
        filtered = {k: v for k, v in self.lexicon.items() 
                    if k not in self._STOP_FORMS 
                    and len(k) >= self.MIN_FORM_LEN
                    and not k.isdigit()}
        self._entity_lexicon = filtered
        # 按长度倒序排列，防止子串覆盖
        self._sorted_src_entities = sorted(filtered.keys(), key=len, reverse=True)
        # 构建正则：全词匹配（用 \b 或空格/行首行尾作为边界）
        # 阿卡德语音译用空格分词，所以 \b 对连字符形式不完美
        # 改用 (?:^|\s) 和 (?:\s|$) 作为边界
        self._entity_patterns = {}
        for src_form in self._sorted_src_entities:
            escaped = re.escape(src_form)
            self._entity_patterns[src_form] = re.compile(
                rf'(?:(?<=\s)|(?<=^)){escaped}(?=\s|$)', re.MULTILINE
            )
        # [DISABLED] 数字/分数不掩码 — ByT5 byte-level 生成 5 或 ½ 只需 1-2 token
        # self.copy_patterns = [
        #     r'[½⅓⅔¼¾⅕⅖⅗⅘⅙⅚⅛⅜⅝⅞⅒]',
        #     r'\b\d+(?:\.\d+)?\b'
        # ]
        # self.regex = re.compile(f"({'|'.join(self.copy_patterns)})")
    
    def apply_placeholders(self, text: str, target_text: str = None) -> Tuple[str, str, Dict[str, str]]:
        """
        对输入（和训练时的目标）同时应用占位符
        
        Returns:
            (masked_text, masked_target, mapping)
            mapping: {tag: (src_form, tgt_form)} 用于回填
        
        Note:
            使用三位数字填充 (如 [[N000]], [[E001]]) 避免 ByT5 字节级歧义
        """
        if not isinstance(text, str):
            return text, target_text, {}
        
        mapping = {}
        counter = {"N": 0, "E": 0}
        
        # 1. 处理专有名词：用 lexicon key 全词匹配音译侧
        for src_form in self._sorted_src_entities:
            pattern = self._entity_patterns.get(src_form)
            if pattern and pattern.search(text):
                tgt_form = self._entity_lexicon[src_form]
                tag = f"[[E{counter['E']:03d}]]"
                mapping[tag] = (src_form, tgt_form)
                text = pattern.sub(tag, text)
                # 翻译侧：用对应的英文形式替换（全词匹配）
                if target_text and tgt_form in target_text:
                    target_text = target_text.replace(tgt_form, tag)
                counter['E'] += 1
                if counter['E'] >= 20:  # 限制最大替换数，避免过度掩码
                    break
        
        # [DISABLED] 数字/分数不掩码 — 让 ByT5 直接学习生成
        # def _replacer(match):
        #     val = match.group(0)
        #     tag = f"[[N{counter['N']:03d}]]"
        #     mapping[tag] = (val, val)
        #     counter['N'] += 1
        #     return tag
        # text = self.regex.sub(_replacer, text)
        # if target_text:
        #     for tag, (src_val, tgt_val) in list(mapping.items()):
        #         if tag.startswith("[[N"):
        #             tgt_pattern = re.compile(r'\b' + re.escape(tgt_val) + r'\b')
        #             target_text = tgt_pattern.sub(tag, target_text)
        
        return text, target_text, mapping
    
    def restore(self, translated_text: str, mapping: Dict[str, str]) -> str:
        """翻译完成后回填占位符（推理时用英文形式）"""
        if not isinstance(translated_text, str):
            return translated_text
        
        for tag, val_tuple in mapping.items():
            # val_tuple = (src_form, tgt_form)
            # 回填时使用目标侧（英文）形式
            tgt_form = val_tuple[1] if isinstance(val_tuple, tuple) else val_tuple
            translated_text = translated_text.replace(tag, tgt_form)
        
        return translated_text


# 全局 PlaceholderService 实例（懒加载）
_placeholder_service = None

def get_placeholder_service(onomasticon: Set[str] = None, lexicon: Dict[str, str] = None) -> PlaceholderService:
    """获取或创建 PlaceholderService 实例"""
    global _placeholder_service
    if _placeholder_service is None or onomasticon is not None or lexicon is not None:
        _placeholder_service = PlaceholderService(onomasticon, lexicon)
    return _placeholder_service


# ============================================================
# 保护区机制（Protected Clean）
# ============================================================

def protected_clean(text: str, clean_func, protect_patterns: list = None) -> str:
    """
    保护区清洗：在执行清洗前保护关键结构，清洗后还原
    
    Args:
        text: 待清洗文本
        clean_func: 清洗函数 (接受 text 返回 cleaned_text)
        protect_patterns: 需要保护的正则模式列表
    """
    if not isinstance(text, str):
        return ""
    
    if protect_patterns is None:
        # 默认保护：限定词 {d}, {m} 等和 Gap 标记
        protect_patterns = [
            r'\{[^}]+\}',      # {d}, {m}, {tug₂} 等限定词
            r'\([^)]*\)',      # (d), (ki) 等括号限定词
            r'<[^>]+>',        # <gap> 等标记
        ]
    
    # 1. 提取并保护
    placeholders = []
    def _mask(m):
        placeholders.append(m.group())
        return f"__PROT_{len(placeholders)-1}__"
    
    protected_text = text
    for pattern in protect_patterns:
        protected_text = re.sub(pattern, _mask, protected_text)
    
    # 2. 执行清洗
    cleaned_text = clean_func(protected_text)
    
    # 3. 还原保护内容
    for i, val in enumerate(placeholders):
        cleaned_text = cleaned_text.replace(f"__PROT_{i}__", val)
    
    return cleaned_text


# ============================================================
# 学术备选标记 / 定制化修正映射（逐条人工审核）
# 官方确认测试集无 /，训练数据中 / 是译者标注的两种可能翻译
# 格式: (old_substring, new_substring)
# 注意: 匹配时文本已经过括号移除（step 1-4），gap 已被保护为 ___BIGGAP___
# ============================================================
SLASH_FIXES = [
    # ===== 名字拼合（损坏泥板的两种读法 → 合并） =====
    ('Uṣur-ša / i-Ištar', 'Uṣur-ša-Ištar'),        # [592] name
    ('Zul / niya', 'Zulniya'),                        # [872] name
    ('Elal / niuman', 'Elalniuman'),                   # [1537] name
    ('son of L / Zu', 'son of Zu'),                    # [1377,1396] damaged name
    ('Bursa / isum', 'Bursaisum'),                     # [1483] name
    ('son of B / Pususu', 'son of Pususu'),            # [1550] damaged name

    # ===== 具体上下文修正（长匹配优先） =====
    ('property / responsibility', 'property'),          # [6]
    ('quality / price', 'price'),                       # [27] after () removal
    ('of the / a tamkarum', 'of a tamkarum'),           # [65]
    ('nēreb / pu to', 'nēreb to'),                     # [71] lapis lazuli term
    ('Like / Instead of', 'Instead of'),                # [74] multi-word alt
    ('is / keeps', 'keeps'),                            # [74]
    ('its / his tithe', 'his tithe'),                   # [85]
    ('textiles to / as', 'textiles as'),                # [149]
    ('has / will indeed', 'will indeed'),               # [149]
    ('perished / disappeared', 'disappeared'),          # [165]
    ('shekel; a / the', 'shekel; the'),                 # [171] specific context
    ('consumed / paid during', 'paid during'),          # [171]
    ('during / from the', 'during the'),                # [171]
    ('less by / than', 'less than'),                    # [192]
    ('she / you can', 'she can'),                       # [236]
    ('will / can not', 'cannot'),                       # [323,1498]
    ('City of / concerning', 'City concerning'),         # [359]
    ('from / with Puzur', 'from Puzur'),                # [359] specific
    ('sons. I / he made', 'sons. He made'),             # [368]
    ('paid / provided to', 'paid to'),                  # [377]
    ('to / for Barānum', 'for Barānum'),                # [377] specific
    ('Subārum / the Subaraean', 'the Subaraean'),       # [383,480]
    ('house you / she brought', 'house she brought'),   # [475]
    ('month II / III', 'month II'),                     # [490,930,1350]
    ('wife. Of / with the', 'wife. With the'),          # [512] specific
    ('go with / in', 'go in'),                          # [512]
    ('for you for / as', 'for you as'),                 # [545] specific context
    ('receive / take', 'receive'),                      # [583]
    ('packets for / as', 'packets as'),                 # [626] specific
    ('this to / for', 'this for'),                      # [632]
    ('take the / a donkey', 'take a donkey'),           # [728]
    ('call for / obtain', 'obtain'),                    # [771]
    ('says / said', 'said'),                            # [771]
    ('misery / expenses', 'expenses'),                  # [913]
    ('tablet of / in the', 'tablet in the'),            # [913]
    ('place it / him before', 'place it before'),       # [965]
    ('bring it / him', 'bring it'),                     # [965]
    ('promised / been ordered', 'been ordered'),         # [986]
    ('from / in', 'from'),                              # [1042,1046] generic last resort
    ('his / its tax', 'its tax'),                       # [1100]
    ('house / firm', 'firm'),                           # [1211]
    ('to / within', 'within'),                          # [1225]
    ('has / shows', 'has'),                             # [1248] after <> removal
    ('replaces / represents', 'represents'),            # [1322]
    ('silver he / I', 'silver he'),                     # [1377,1396]
    ('palace from / at', 'palace at'),                  # [1432]
    ('person / donkey', 'donkey'),                      # [1432] after () removal
    ('gave to / for', 'gave for'),                      # [1433] specific
    ('import-tax to / for', 'import-tax for'),          # [1433]

    # ===== 无空格变体（sentence-level 数据格式不同） =====
    ('Like/Instead of', 'Instead of'),
    ('of/concerning', 'concerning'),
    ('from/with', 'from'),
    ('Of/with', 'With'),
    ('of/in', 'in'),
    ('as/the', 'as'),
    ('his/its', 'its'),
    ('house/firm', 'firm'),
]

# ============================================================
# 预处理函数
# ============================================================

def _resolve_slash_alternatives(text: str) -> str:
    """使用 SLASH_FIXES 查表处理翻译中的 / 标记，清除残留 /。"""
    if '/' not in text:
        return text
    for old, new in SLASH_FIXES:
        text = text.replace(old, new)
    # 保护残留的数字分数 N/M（step 4.0 grain rules 未覆盖的非 shekels 上下文）
    text = re.sub(r'\b(\d+)\s*/\s*(\d+)\b', r'\1/\2', text)  # 先归一化空格: 1 / 12 → 1/12
    # 清除残留的孤立 /（SLASH_FIXES 未覆盖的边缘情况）
    # 只替换非数字分数的 /（两侧不是数字的 /）
    text = re.sub(r'(?<!\d)/(?!\d)', ' ', text)
    return text


def _strip_editorial_parentheticals(text: str) -> str:
    patterns = [
        r'\(\s*(?:[A-Za-z][A-Za-z\-]*\s+)?preference\s*:?\s*[^)]*\)',
        r'\(\s*preferred(?:\s+translation)?\s*:?\s*[^)]*\)',
        r'\(\s*alternative\s*:?\s*[^)]*\)',
        r'\(\s*variant\s*:?\s*[^)]*\)',
    ]
    for pattern in patterns:
        text = re.sub(pattern, ' ', text, flags=re.I)
    return text


def decode_literal_whitespace_escapes(text: str) -> str:
    """Restore literal \\n / \\r / \\t sequences before downstream cleaning."""
    if not isinstance(text, str):
        return ""
    return (
        text.replace("\\r\\n", "\n")
        .replace("\\n", "\n")
        .replace("\\r", "\n")
        .replace("\\t", " ")
    )


def strip_translation_line_prefixes(text: str) -> str:
    """Remove OCR line-number prefixes that are glued to the next sentence token."""
    if not isinstance(text, str):
        return ""

    cleaned_lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            cleaned_lines.append("")
            continue

        # Examples: 3-13For, 5-I0[I]here, a-11 I pray, ns-2sAn owner, 35Your...
        line = re.sub(
            r'^\s*[a-z]{1,3}-[0-9IlSs]{1,3}(?:-[0-9IlSs]{1,3})?\s*(?=[A-Z\[])',
            '',
            line,
        )
        line = re.sub(
            r'^\s*[0-9IlSs]{1,3}-[0-9IlSs]{1,3}\s*(?=[A-Z\[])',
            '',
            line,
        )
        line = re.sub(
            r'^\s*[0-9IlSs]{1,3}(?=[A-Z\[])',
            '',
            line,
        )
        cleaned_lines.append(line.lstrip(" -–—.:;,'\""))

    return "\n".join(cleaned_lines)


def strip_translation_line_references(text: str) -> str:
    """Drop residual editorial line references that survive line-based cleanup."""
    if not isinstance(text, str):
        return ""

    patterns = [
        # 1-2 Tell..., 12 -14 [As for]..., 15-16 Do...
        r'(?:(?<=^)|(?<=\n)|(?<=\s))\d{1,2}\s*\'?\s*-\s*\d{1,2}(?=\s+(?:\[|\(|[A-Z]))',
        # 29(?)-iii-ZL 6'
        r'(?:(?<=^)|(?<=\n)|(?<=\s))\d{1,2}\s*(?:\(\?\))?\s*-\s*[ivxlcIVXLC]{1,4}(?:-ZL)?(?:\s*\d+[\'’]?)?(?=\s|$|[.,;:!?])',
        # 1.1' -16' damaged
        r'\b\d+(?:\.\d+)?\'\s*-\s*\d+\'(?:\s+damaged)?\b',
    ]
    for pattern in patterns:
        text = re.sub(pattern, ' ', text)
    return text


def repair_translation_linebreak_word_splits(text: str) -> str:
    """Heal common line-wrap/OCR splits that survive CSV import in translations."""
    if not isinstance(text, str):
        return ""

    letters = r"A-Za-zÀ-ÿĀ-žḀ-ỿ"
    lowercase_letters = r"a-zà-ÿā-žḁ-ỿ"
    # Most literal line wraps should become spaces, not glued words.
    text = re.sub(
        rf'(?<=[{letters}])\s*\n\s*(?=[{lowercase_letters}])',
        ' ',
        text,
    )
    # Join hyphenated wraps/names like Ibni- Adad -> Ibni-Adad.
    text = re.sub(
        rf'(?<=[{letters}])-\s+(?=[{letters}])',
        '-',
        text,
    )
    # Join short-token splits like Sin -ibni, mār -Enlil, Ea -rabi.
    text = re.sub(
        rf'\b([{letters}]{{1,4}})\s+-\s*(?=[{letters}])',
        r'\1-',
        text,
    )
    upper_letters = r"A-ZÀ-ÝĀĒĪŪŠṢṬḪḀ-Ỿ"
    text = re.sub(
        rf'(?<=[;:]\s)([{upper_letters}][{letters}]{{1,20}})\s+-\s*(?=[{letters}])',
        r'\1-',
        text,
    )
    return text


def unwrap_editorial_square_brackets(text: str) -> str:
    """
    Keep short lexical supplements from [...] while still collapsing pure damage markers.

    Examples:
      [one] -> one
      A[ssu]r -> Assur
      [...] -> <gap>
    """
    if not isinstance(text, str):
        return ""

    source = text
    pure_damage_re = re.compile(r'(?:[.\u2026xX\-–—]|\s)+')
    mixed_damage_re = re.compile(
        r'(?:\.\s*){2,}\.?'
        r'|…+'
        r'|(?:[xX]\s*){2,}[xX]?'
        r'|(?:[-–—]\s*){2,}[-–—]?'
    )

    def _normalize_space_local(value: str) -> str:
        return re.sub(r'\s+', ' ', str(value or '')).strip()

    def _replace(match):
        raw_content = str(match.group(1) or "")
        if not raw_content.strip():
            return " "

        if pure_damage_re.fullmatch(raw_content.strip()):
            return " ___GAP___ "

        content = mixed_damage_re.sub(" ___GAP___ ", raw_content)
        cleaned = re.sub(r"[^0-9A-Za-zÀ-ÿĀ-žḀ-ỿ' _-]", " ", content)
        cleaned = _normalize_space_local(cleaned)
        cleaned = re.sub(r'(?:___GAP___\s*){2,}', '___GAP___ ', cleaned).strip()
        if not cleaned:
            return " "
        if len(cleaned.split()) > 6 or len(cleaned) > 48:
            return " "
        if "___GAP___" in cleaned:
            return f" {cleaned} "

        start, end = match.span()
        prev_char = source[start - 1] if start > 0 else " "
        next_char = source[end] if end < len(source) else " "
        left_space = "" if (prev_char.isalnum() or prev_char in "-'") else " "
        right_space = "" if (next_char.isalnum() or next_char in "-'") else " "
        return f"{left_space}{cleaned}{right_space}"

    return re.sub(r'\[([^\[\]\n]{0,64})\]', _replace, text)


def repair_translation_escape_artifacts(text: str) -> str:
    """
    Repair OCR / CSV escape artifacts seen in Michel-derived translations.

    Examples:
      nTell ...      -> Tell ...
      n17-34Regarding -> Regarding ...
      Send me n of ... -> Send me of ...
    """
    if not isinstance(text, str):
        return ""
    trans_letters = r"A-Za-zÀ-ÿĀ-žḀ-ỿ"
    text = decode_literal_whitespace_escapes(text)
    text = text.replace('\u2013', '-').replace('\u2014', '-').replace('—', '-').replace('–', '-')
    text = text.replace('⸢', '').replace('⸣', '').replace('⸤', '').replace('⸥', '')
    text = text.replace('⌈', '').replace('⌉', '')
    text = repair_translation_linebreak_word_splits(text)
    # Repair lexical compounds broken by editorial uncertainty markers, e.g. Elali(?)-muštāl.
    text = re.sub(
        rf'(?<=[{trans_letters}])\(\?\)\s*-\s*(?=[{trans_letters}])',
        '-',
        text,
    )
    # Recover unmatched lexical supplements before visible damage, e.g. Baby[lon …] -> Babylon <gap>.
    text = re.sub(
        rf'(?<=[{trans_letters}])\[([{trans_letters}]{{1,16}})\s*(?=(?:\.\s*){{2,}}|…+|(?:[xX]\s*){{2,}})',
        r'\1 ',
        text,
    )
    def _restore_bracket_tail_hyphen(match):
        token = re.sub(r'\(\?\)', '', str(match.group(1) or '')).strip()
        return f' {token}-' if token else ' -'
    text = re.sub(
        rf'\[(?:[^\]\n]*?[\s,/])?([{trans_letters}]{{1,16}}(?:\(\?\))?)\]-(?=[{trans_letters}])',
        _restore_bracket_tail_hyphen,
        text,
    )
    text = re.sub(r'(?<=\d)-[\'"‘’`]+(?=[0-9IlSs])', '-', text)
    # Translation-side line-end hyphenation is usually editorial wrapping, not lexical '-'.
    text = re.sub(r'(?<=[A-Za-zÀ-ÿ])-\s*\n\s*(?=[A-Za-zÀ-ÿ])', '', text)
    text = re.sub(
        rf'(?:(?<=^)|(?<=\s))\[([{trans_letters}][{trans_letters}\' -]{{0,32}})\](?=-)',
        r'\1',
        text,
    )
    text = re.sub(
        rf'(?:(?<=^)|(?<=\s))\[([{trans_letters}]{{1,16}})\](?=[{trans_letters}])',
        r'\1',
        text,
    )
    text = re.sub(
        rf'(?:(?<=^)|(?<=\s))\[([{trans_letters}]{{1,3}})\](?=[{trans_letters}])',
        r'\1',
        text,
    )
    text = re.sub(
        rf'(?<=[{trans_letters}-])\[([{trans_letters}]{{1,16}})\](?=[{trans_letters}-])',
        r'\1',
        text,
    )
    text = re.sub(
        rf'(?<=-)\[([{trans_letters}]{{1,16}})\](?=\s|$|[.,;:!?])',
        r'\1',
        text,
    )
    text = re.sub(
        rf'(?<=[{trans_letters}-])\[([{trans_letters}]{{1,2}})\](?=\s|[.,;:!?-])',
        r'\1',
        text,
    )
    text = re.sub(
        rf'\b([{trans_letters}]{{1,3}})\[([{trans_letters}]{{2,16}})\](?=\s|$|[.,;:!?"])',
        r'\1\2',
        text,
    )
    text = re.sub(r'(?<=[A-Za-zÀ-ÿĀ-žḀ-ỿ])\d{2,}(?=\s|$|[.,;:!?"])', '', text)
    text = re.sub(r'(^|[\s"\'([{])n(?=[A-Z0-9])', r'\1\n', text)
    # Drop single-letter lower-case bracket notes before bracket unwrapping, e.g. alive[e] -> alive.
    text = re.sub(r'(?<=[a-z])\[[a-z]\](?=\W|$)', '', text)
    # Drop unmatched inline note markers such as sulu[l11, that are not lexical supplements.
    text = re.sub(r'(?<=[A-Za-zÀ-ÿ])\[[0-9IlSs]{1,4}(?=[,.;:\s])', '', text)
    # Recover open damage markers that were never properly closed, e.g. "[... l As soon as ...".
    text = re.sub(
        r'\[\s*(?:(?:\.\s*){2,}\.?|…+|(?:[xX]\s*){2,}[xX]?|(?:[-–—]\s*){2,}[-–—]?)\s*[A-Za-z0-9IlSs]?(?=\s)',
        ' <gap> ',
        text,
    )
    text = strip_translation_line_prefixes(text)
    text = strip_translation_line_references(text)
    text = re.sub(
        r'(?<=[\.\?!:;"\'\]\)])\s*(?:[a-z]{1,3}-[0-9IlSs]{1,3}(?:-[0-9IlSs]{1,3})?|[0-9IlSs]{1,3}-[0-9IlSs]{1,3}|[0-9IlSs]{1,3})(?=[A-Z\[])',
        ' ',
        text,
    )
    text = re.sub(rf'(?<![{trans_letters}])n(?![{trans_letters}])', ' ', text)
    return text


def clean_translation_light(text: str) -> str:
    """
    轻量级翻译清洗：保持原味，只处理真正的噪声
    - 保留原始拼写（ā, ī 等变音符）
    - 保留引号（chrF++ 会计入）
    - 智能处理括号：移除学术注释，保留补全性内容
    """
    if not isinstance(text, str):
        return ""

    text = repair_translation_escape_artifacts(text)
    
    # 0. 保护 Gap 标记和占位符
    text = text.replace('<gap>', '___GAP___')
    # 保护占位符 [[N000]], [[E000]] 等
    placeholders = []
    def _protect_placeholder(m):
        placeholders.append(m.group())
        return f"___PH{len(placeholders)-1}___"
    text = re.sub(r'\[\[[NE]\d{3}\]\]', _protect_placeholder, text)
    
    # 0.5 幽灵字符清理
    # 移除非拉丁外文字符（希伯来文、阿拉伯文等乱码）
    text = re.sub(r'[\u0590-\u05FF\u0600-\u06FF\u0980-\u09FF\u4E00-\u9FFF]+', ' ', text)
    # 学术转写残留字符 → 清洗（确保 translation 端 0 个非白名单 Unicode）
    text = text.replace('\u02BF', '')   # ʿ (modifier letter left half ring / ayin) → 删除
    text = text.replace('\u00FB', 'u')  # û → u
    text = text.replace('\u00EE', 'i')  # î → i
    text = text.replace('\u00EA', 'e')  # ê → e
    text = text.replace('\u00F4', 'o')  # ô → o
    text = text.replace('\u00E2', 'a')  # â → a
    text = text.replace('\u1EA1', 'a')  # ạ (a with dot below) → a
    text = text.replace('\u015B', 's')  # ś → s
    text = text.replace('\u1E2B', 'h')  # ḫ → h
    text = text.replace('\u1E2A', 'H')  # Ḫ → H
    text = text.replace('\u015F', 'ṣ')  # ş (Turkish s-cedilla) → ṣ
    text = text.replace('\u015E', 'Ṣ')  # Ş → Ṣ
    text = text.replace('\u011D', 'g')  # ĝ → g
    text = text.replace('\u011C', 'G')  # Ĝ → G
    text = text.replace('\u00CE', 'I')  # Î → I
    text = text.replace('\u00D4', 'O')  # Ô → O
    text = text.replace('\u00C2', 'A')  # Â → A
    # 智能引号 → ASCII 引号
    text = text.replace('\u00AB', '"').replace('\u00BB', '"')  # « » → "
    text = text.replace('\u201C', '"').replace('\u201D', '"')  # " " → "
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # ' ' → '
    text = text.replace('\u2013', '-').replace('\u2014', '-')  # – — → -
    text = _strip_editorial_parentheticals(text)
    # Common OCR word-wrap repairs that should stay joined after newline normalization.
    text = re.sub(r'\bpro per\b', 'proper', text, flags=re.I)
    text = re.sub(r'\bBaby lon\b', 'Babylon', text)
    text = re.sub(r'\bbroken\s*-\s*except\b', 'broken except', text, flags=re.I)
    text = re.sub(r'\bdeducted-\s+(?=\d)', 'deducted. ', text, flags=re.I)
    text = re.sub(r'\b(barley|flour)\s+-measured\b', r'\1 measured', text, flags=re.I)
    text = re.sub(r'\bstandard-(?=(?:for|as|was|out)\b|,)', 'standard ', text, flags=re.I)
    text = re.sub(r'\bmessenger\s+-who\b', 'messenger who', text, flags=re.I)
    text = re.sub(r'\bgentlemen\s+-your representatives-\b', 'gentlemen, your representatives,', text, flags=re.I)
    
    # 1. 删除学术注释括号（测试集无这些标注，但保留其他有意义的括号）
    # 匹配 (fem.), (sing.), (pl.), (plural), (fem. plur.), (fem. sing.), (?),
    # (masc.), (uncertain), (lit.), (meaning...), (var.), (sic), (reading...)
    text = re.sub(r'\((?:fem|plur|pl|sing|masc|uncertain|lit|meaning|var|sic|reading|absent)\.?(?:\s+(?:fem|plur|pl|sing|masc)\.?)*\s*[^)]*\)', ' ', text, flags=re.I)
    text = re.sub(r'\(\?\)', ' ', text)  # 显式移除 (?)
    text = re.sub(r'\(s\)', 's', text)   # god(s) → gods
    
    # 2. 保留括号！测试集有 (of silver), (and) 等有意义括号
    # 不再 unwrap 或删除普通括号
    
    # 3. 方括号：保留短补字内容，纯缺损继续映射为 <gap>
    text = unwrap_editorial_square_brackets(text)
    text = text.replace('[', ' ').replace(']', ' ')
    # 孤立花括号清理（保护限定词 {ki} 等，只删残留的孤立 }）
    # 先保护限定词，再删孤立花括号，再还原
    text = re.sub(r'\{(\w+)\}', r'__DET_\1__', text)
    text = text.replace('{', '').replace('}', '')
    text = re.sub(r'__DET_(\w+)__', r'{\1}', text)
    
    # 4.0 格令分数（/12 特殊转换，官方指定）— 必须在 SLASH 处理之前
    text = re.sub(r'\b5\s+11\s*/\s*12\s+shekels?\b', '6 shekels less 15 grains', text, flags=re.I)
    text = re.sub(r'\b(\d+)\s+7\s*/\s*12\s+shekels?\b', lambda m: f'{m.group(1)} ½ shekel 15 grains', text, flags=re.I)
    text = re.sub(r'\b7\s*/\s*12\s+shekels?\b', '½ shekel 15 grains', text, flags=re.I)
    text = re.sub(r'\b(\d+)\s+5\s*/\s*12\s+shekels?\b', lambda m: f'{int(m.group(1))} ⅔ shekel 15 grains', text, flags=re.I)
    text = re.sub(r'\b5\s*/\s*12\s+shekels?\b', '⅔ shekel 15 grains', text, flags=re.I)
    text = re.sub(r'\b(\d+)\s+1\s*/\s*12\s+shekels?\b', lambda m: f'{m.group(1)} shekel 15 grains', text, flags=re.I)
    text = re.sub(r'\b1\s*/\s*12\s*\(?shekels?\)?\b', '15 grains', text, flags=re.I)
    
    # 4.1 移除 forbidden chars（与 postprocess_output 对齐，确保 labels 和 predictions 一致）
    # ʾ (modifier letter right half ring) — 学术转写残留
    text = text.replace('\u02BE', '')
    # + — 数字标记如 "20+" 中的 +
    text = text.replace('+', '')
    # / — 学术不确定性标注，分类处理后删除残留
    text = _resolve_slash_alternatives(text)
    
    # 4.5 双尖括号 <<content>> → 保留内容（竞赛定义：erroneous/errant signs）
    text = re.sub(r'<<([^>]*)>>', r' \1 ', text)
    # 残留的单 << → 转成 < （处理不完整的双尖括号如 <<and>）
    text = text.replace('<<', '<')
    
    # 4.6 尖括号：保留内容，去掉括号（编辑补充，如 <the expenses> → the expenses）
    text = re.sub(r'<(?!/?gap\b)([^>]*)>', r' \1 ', text, flags=re.I)
    text = repair_translation_linebreak_word_splits(text)
    
    # 5. 连续引号压缩："" → " （嵌套引语闭合堆叠）
    text = re.sub(r'"{2,}', '"', text)
    
    # 6. 标点后缺空格修复
    text = re.sub(r'([a-z])\.([A-Z])', r'\1. \2', text)   # silver.Twice → silver. Twice
    text = re.sub(r';([A-Za-z0-9⅓⅔½¼¾⅙⅚⅛⅜⅝⅞])', r'; \1', text)  # Azuza;1 → Azuza; 1
    
    
    # 6.5 常见 typo 修复
    text = re.sub(r'\byou r\b', 'your', text)   # "you r firm" → "your firm"
    text = re.sub(r'\bconrol\b', 'control', text)  # typo fix
    
    # 6.6 学术备选标记 / 清理 — 已移至 step 4.1 的 _resolve_slash_alternatives()
    
    # 6.7 (保留 .. — data_final 中 39.9% 行含有 ..，是 ... gap 标记的边界残留，不应转成句号)
    
    # 6.8 官方建议的翻译替换
    # PN → <gap>（AKT 8 中的字面 PN token）
    text = re.sub(r'\bPN\b', '___GAP___', text)
    # Hebrew ד 字符清除
    text = text.replace('\u05d3', ' ')
    # 残留 x/X → <gap>（单独的 x 表示破損）
    text = re.sub(r'(?<=\s)[xX](?=\s|$|[.,;:!?])', '___GAP___', text)
    text = re.sub(r'^[xX](?=\s)', '___GAP___', text)
    # 官方要求：仅替换孤立连字符前缀形式（-gold, -tax, -textiles）
    # 不替换已有的 kutānu-textiles 和 bare textiles（22名选手反馈不含这些词分数更高）
    text = re.sub(r'(^|\s)-gold\b', r'\1pašallum gold', text)
    text = re.sub(r'(^|\s)-tax\b', r'\1šadduātum tax', text)
    text = re.sub(r'(^|\s)-textiles?\b', lambda m: m.group(1) + 'kutānum textile' + ('s' if m.group().endswith('s') else ''), text)
    # 月名粘连修复（数据typo：huburhe → hubur, he）
    text = re.sub(r'\b([Hh]ubur)(he)\b', r'\1, \2', text)
    # 月份罗马数字 → 阿拉伯数字
    _roman_map = {'XII': '12', 'XI': '11', 'VIII': '8', 'VII': '7',
                  'VI': '6', 'IV': '4', 'IX': '9', 'III': '3',
                  'II': '2', 'X': '10', 'V': '5', 'I': '1'}
    for roman, arabic in _roman_map.items():
        text = re.sub(r'(?i)\b(month)\s+' + roman + r'\b', r'\1 ' + arabic, text)
    
    # 月份阿卡德语月名 → 阿拉伯数字（官方月份表）
    # 格式: month (of/:) + 月名 → month N
    # 注意: Hubur 也是人名组件 (Šu-Hubur)，只匹配 month 后的
    _month_names = [
        # (regex_pattern, replacement_number) — 长模式优先
        # 复合月名（Narmak-Aššur-ša-kēnātim 等）必须最先匹配
        (r'(?:of\s+)?:?\s*Narmak[- ]Aššur[- ](?:ša[- ])?kēnātim', '3'),
        (r'(?:Narmak[- ]Aššur\s+)?(?:of\s+)?:?\s*[Šš]a[- ]kēnātim', '3'),
        (r'(?:of\s+)?:?\s*Allanāt(?:um|im)', '12'),
        (r'(?:of\s+)?:?\s*T[eē]\'?inātum', '10'),        # Te'inātum / Tē'inātum / Teinātum
        (r'(?:of\s+)?:?\s*Kanwartan?', '9'),               # Kanwarta / Kanwartan
        (r'(?:of\s+)?:?\s*Qarra\'?āt(?:um|im)', '8'),      # Qarrātum / Qarra'ātum / Qarrātim
        (r'(?:of\s+)?:?\s*[Ṣṣ]ip\'?(?:um|im)', '7'),      # Ṣipum / Ṣip'um / ṣipum / ṣipim
        (r'(?:of\s+)?:?\s*[Hh]ubur', '6'),                 # Hubur / hubur (NOT Šu-Hubur: 前面有 month)
        (r'(?:of\s+)?:?\s*Ab[- ]?[šŠ]arr(?:āni|ani)', '5'),# Abšarrani / Ab-šarrāni / Ab šarrāni
        (r'(?:of\s+)?:?\s*Ma[hḫ]+ur-ilī', '4'),            # Mahur-ilī / Mahhur-ilī
        (r'(?:of\s+)?:?\s*[Šš]a[- ]sarr(?:ātim|atim)', '2'),# ša-sarratim / Ša-sarrātim
        (r'(?:of\s+)?:?\s*Kuzall(?:um|u|i)', '11'),        # Kuzallum / Kuzallu / Kuzalli
        (r'(?:of\s+)?:?\s*Bēlat[- ]ekallim', '1'),         # Bēlat-ekallim / Bēlat ekallim (月1)
        (r'(?:of\s+)?:?\s*Narmak[- ]Aššur', '3'),          # 独立 Narmak-Aššur (月3别名)
    ]
    for pattern, num in _month_names:
        text = re.sub(r'(?i)\b(month)\s*' + pattern + r'\b', r'\1 ' + num, text)
    
    # 7. 标点前多余空格修复（方括号移除后残留，占总标点的 11.8%，大多数无前空格）
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # 7.5 下标/上标数字规范化（与 postprocess_output 对齐）
    _subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    text = text.translate(_subscripts)
    _superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    text = text.translate(_superscripts)
    
    # 8. 归一化标点和空格
    # 双连字符智能修复：小写词--小写词 → 破折号 " - "，其他 → 单连字符 "-"
    text = re.sub(r'(?<=[a-z])--+(?=[a-z])', ' - ', text)  # copper--your → copper - your
    text = re.sub(r'--+', '-', text)  # 其余: Aššuriš--tikal → Aššuriš-tikal
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 8.5 去重（与 postprocess_output 对齐，避免训练/推理不一致）
    # 删除重复词: "the the" → "the", "of of" → "of"
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)
    # 删除重复短语: "come here, come here," → "come here,"
    text = remove_phrase_repeats(text)
    
    # 8.9 清理连续脏标点残留（学术注释删除后的遗留，如 ,; 并列、空括号 ()）
    text = re.sub(r'\(\s*\)', '', text)           # 空括号删除
    text = re.sub(r',\s*[,;]+', ',', text)        # ,; ,, → ,
    text = re.sub(r';\s*[,;]+', ';', text)        # ;, ;; → ;
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 9. 还原保护的内容
    text = text.replace('___GAP___', '<gap>')
    for i, ph in enumerate(placeholders):
        text = text.replace(f"___PH{i}___", ph)
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def clean_translation_noise(text: str) -> str:
    """根据Kaggle最新官方规则执行翻译清洗"""
    if not isinstance(text, str):
        return ""

    text = repair_translation_escape_artifacts(text)
        
    # 0. 弯引号转直引号 (Curly to straight quotes)
    text = text.replace('«', '"').replace('»', '"')
    text = text.replace('“', '"').replace('”', '"').replace('‘', "'").replace('’', "'")
    
    # 官方要求：仅替换孤立连字符前缀形式（-gold, -tax, -textiles）
    text = re.sub(r'(?<=\s)-gold\b', ' pašallum gold', text)
    text = re.sub(r'(?<=\s)-tax\b', ' šadduātum tax', text)
    text = re.sub(r'(?<=\s)-textiles?\b', lambda m: ' kutānum textile' + ('s' if m.group().endswith('s') else ''), text)
    text = repair_translation_linebreak_word_splits(text)
    
    # Kaggle 规则：罗马数字月份转换 (Month I-XII to 1-12)
    month_map = {
        r'[Mm]onth XII': 'month 12',
        r'[Mm]onth XI': 'month 11',
        r'[Mm]onth X': 'month 10',
        r'[Mm]onth IX': 'month 9',
        r'[Mm]onth VIII': 'month 8',
        r'[Mm]onth VII': 'month 7',
        r'[Mm]onth VI': 'month 6',
        r'[Mm]onth IV': 'month 4',
        # 注意: 罗马数字里 V, III, II, I 必须按长短顺序匹配，否则会把 VII 替换成 V 等
        r'[Mm]onth V': 'month 5',
        r'[Mm]onth III': 'month 3',
        r'[Mm]onth II': 'month 2',
        r'[Mm]onth I': 'month 1',
    }
    for pat, rep in month_map.items():
        text = re.sub(pat, rep, text)

    """
    根据官方建议清理译文中的噪声：
    1. 移除现代抄录注释
    2. 彻底移除所有括号及其变体
    3. 清理重复标点、不平衡引号
    4. 移除由于泥板破损残留的孤立标点
    5. 处理特殊 Unicode 噪声并强制单空格归一化
    
    ⚠️ 注意：此函数较激进，建议使用 clean_translation_light() 保持原味
    """
    if not isinstance(text, str):
        return ""
    
    # 定义全局通用的字母字符集（包含变音符）
    # 用于正则的字符类
    word_chars_re = "A-Za-zšṣṭāáàīíìūúùŠṢṬĀÁÀĪÍÌŪÚÙ"
    # 用于 Python in 检查的集合
    word_chars_set = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
                         'šṣṭāáàīíìūúùŠṢṬĀÁÀĪÍÌŪÚÙ')
    
    # 0. 保护标准标签
    text = text.replace('<gap>', '___GAP___')
    text = _strip_editorial_parentheticals(text)

    # 1. 移除现代抄录注释
    scribal_patterns = [
        r'\(fem\.?\s*[^)]*\)', r'\(masc\.?\s*[^)]*\)',
        r'\(plur\.?\s*[^)]*\)', r'\(sing\.?\s*[^)]*\)',
        r'\(plural\)', r'\(singular\)',
        r'\(meaning\s+[^)]+\)', r'\(reading\s+[^)]+\)',
        r'\(uncertain\)', r'\(\?\)', 
        r'\((?:[a-z]{1,4}\.?\s+){1,2}[a-z]+\)',
    ]
    for pattern in scribal_patterns:
        text = re.sub(pattern, ' ', text, flags=re.I)
    
    # 2. 特殊符号映射
    unicode_map = {
        '—': '-', '–': '-',  # 破折号
        '“': '"', '”': '"',  # 智能引号
        '‘': "'", '’': "'",
        '⁄': '/', 'ד': ' ', '⌈': '', '⌉': '', 'ʾ': '', 'ˆ': '', '+': ' ',
    }
    for old, new in unicode_map.items():
        text = text.replace(old, new)

    # 2.5 已移至括号统一处理（见下方）

    # 3. 括号处理（智能区分注释性 vs 补全性括号）
    for b in ['[', ']', '{', '}', '［', '］', '（', '）', '【', '】']:
        text = text.replace(b, '')
    
    # 3.1 注释性关键词（整体删除）
    SCRIBAL_KEYWORDS = {
        'broken', 'lines', 'line', 'absent', 'plur', 'plural', 
        'fem', 'masc', 'sing', 'singular', 'lit', 'var', 
        'reading', 'meaning', 'sic', 'break', 'large'
    }
    
    # 3.2 智能括号处理
    def smart_paren_replace(match):
        content = match.group(1).strip()
        content_lower = content.lower()
        
        # 单字符注释（如 ?, !, s）-> 删除
        if len(content) <= 2 and not content.isalpha():
            return ' '
        if content_lower in {'s', 'x', '?', '!'}:
            return ' '
        
        # 含注释关键词 -> 删除
        words = content_lower.split()
        if any(w.rstrip('.') in SCRIBAL_KEYWORDS for w in words):
            return ' '
        
        # 纯字母单词（可能带连字符）-> 保留内容
        if re.match(r'^[A-Za-z]+(?:-[A-Za-z]+)?$', content):
            return content
        
        # 多个纯英文单词（最多5词）-> 保留内容
        if re.match(r'^[A-Za-z]+(?:\s+[A-Za-z]+){0,4}$', content):
            return content
        
        # 其他（含数字、标点等）-> 删除
        return ' '
    
    text = re.sub(r'\(([^()]*)\)', smart_paren_replace, text)
    
    # 处理不配对的孤立括号
    text = text.replace('(', '').replace(')', '')
    # 非标准尖括号（保护 <gap>）
    text = re.sub(r'<(?!/?gap\b)[^>]*>', ' ', text, flags=re.I)
    
    # 4. 极致循环清理
    prev = None
    while prev != text:
        prev = text
        
        # 4.1 彻底剥离所有双引号 (片段中多为噪声，且极易导致不平衡)
        text = text.replace('"', ' ')
        
        # 4.2 单引号：极致平衡与保护 (100% Purity 策略)
        def balance_single_quotes(t):
            # 定义嵌入式撇号保护：两侧必须紧贴字母 (包含变音符)
            chars = list(t)
            for i, char in enumerate(chars):
                if char == "'":
                    is_embedded = False
                    if 0 < i < len(t) - 1:
                        # 使用 word_chars_set 进行正确的成员检查
                        if t[i-1] in word_chars_set and t[i+1] in word_chars_set:
                            is_embedded = True
                    # 如果不是嵌入式撇号，在片段翻译中极大概率是残留噪声或不平衡引号，直接移除
                    if not is_embedded:
                        chars[i] = ''
            return "".join(chars)
        
        text = balance_single_quotes(text)

        # 4.3 彻底清除音译片段 (Akkadian residue)
        # 匹配纯小写、带连字符的音译片段，但必须是独立的（不是名字的一部分）
        # 关键：lookbehind/lookahead 必须包含所有变音符字母，否则会误删名字
        all_letters = "A-Za-zšṣṭāáàīíìūúùŠṢṬĀÁÀĪÍÌŪÚÙ"
        low_chars = "a-zšṣṭāáàīíìūúù"
        # 排除列表
        exclusions = ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 
                      'half', 'and', 'vis-a-vis', 'interest-free', 'toll-and-fees', 'toll-and-tax',
                      'side-by-side', 'day-to-day', 'year-by-year', 'month-by-month', 'tax-free',
                      'one-and-half', 'two-and-half', 'three-and-half', 'four-and-half', 'five-and-half']
        exc_pattern = "|".join([re.escape(x) for x in exclusions])
        # 匹配 2-6 个由连字符连接的小写片段，前后必须不是字母或连字符
        res_pattern = rf"(?<![{all_letters}\-])(?!(?:{exc_pattern})\b)[{low_chars}]{{1,3}}(?:-[{low_chars}]{{1,3}}){{1,5}}(?![{all_letters}\-])"
        text = re.sub(res_pattern, ' ', text)

        # 4.4 标点与格式深度清理
        # 4.4.1 移除单词首尾的非法连字符 (如 -copper -> copper)
        text = re.sub(rf"(?<![{word_chars_re}])-+([{word_chars_re}]+)", r"\1", text)
        text = re.sub(rf"([{word_chars_re}]+)-+(?![{word_chars_re}])", r"\1", text)
        
        # 4.4.2 标点堆叠处理
        text = text.replace(',.', '.').replace('.,', '.').replace(',,', ',').replace('..', '.')
        text = re.sub(r'[,;]{2,}', ',', text)
        text = re.sub(r'\.{2,}', '.', text)
        text = text.replace('!;', ';').replace('?;', ';').replace('!:', ':').replace('?:', ':')
        text = re.sub(r'([\!\?\:])[\.\,]+', r'\1', text)
        text = re.sub(r'[\.\,]+([\!\?\:])', r'\1', text)
        # 处理跨类标点粘连 (如 !?) -> 统一归约
        text = re.sub(r'([\.!\?,;:])([\.!\?,;:])+', r'\1', text)

        # 4.4 孤立标点 (移除前后是空格或边界的标点)
        # 注意：不移除冒号(:)，因为冒号在翻译中有实际语义（对话标记 said:、比率 6:1 等）
        text = re.sub(r'(^|[\s_])[\"\',;\?\!/\.-]+(?=[\s_]|$)', r'\1 ', text)
        
        # 4.5 标点前空格 (修复 Alpili , -> Alpili,)
        text = re.sub(r'([\w>_])\s+([\!\?\:\,;])', r'\1\2', text)
        text = re.sub(r'([\w>_])\s+\.(?!\d)', r'\1.', text)
        
        # 4.6 粘连修复
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # 4.7 归一化空格
        text = re.sub(r'\s+', ' ', text)

    # 5. De-stuttering (单词重复)
    def remove_stutter(m):
        word = m.group(1)
        full_match = m.group(0)
        if word.lower() in ['had', 'that'] and len(full_match.split()) == 2:
            return full_match
        return word

    prev = None
    while prev != text:
        prev = text
        text = re.sub(r'\b(\w+)(?:\s+\1)+\b', remove_stutter, text, flags=re.I)
        text = re.sub(r'\s+', ' ', text)

    # 6. 首尾彻底清理
    text = text.strip()
    text = re.sub(r'^[-\.,;:\?\!\s]+', '', text)
    def finalize_trailing(m):
        chars = m.group()
        for c in ['!', '?', '.']:
            if c in chars: return c
        return ''
    text = re.sub(r'[-\.,;:\?\!\"\']+$', finalize_trailing, text)
    
    # 7. 还原标准标签
    text = text.replace('___GAP___', '<gap>')
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # 8. 最终归一化
    text = re.sub(r'\s+', ' ', text).strip()
    text = unicodedata.normalize('NFC', text)
    
    return text


def normalize_gaps(text: str) -> str:
    """
    Gap 标准化：所有 gap 统一为 <gap>（官方最新规则，不区分大小间隙）
    """
    if not isinstance(text, str):
        return ""
    
    # 0. 字面量 gap 标签统一
    text = re.sub(r'<\s*big_gap\s*>', ' <gap> ', text, flags=re.I)
    text = re.sub(r'<\s*small_gap\s*>', ' <gap> ', text, flags=re.I)
    text = re.sub(r'\bbig_gap\b', ' <gap> ', text, flags=re.I)

    # 1. 大间隙标记 → <gap>
    text = re.sub(r'\{large\s+break\}', ' <gap> ', text, flags=re.I)
    text = re.sub(r'\(broken\s+lines?\)', ' <gap> ', text, flags=re.I)
    text = re.sub(r'\d+\s+broken\s+lines?', ' <gap> ', text, flags=re.I)
    # 编辑性括号包裹的缺损标记要先转成 <gap>，否则后续去括号时会被直接吃掉
    text = re.sub(r'\[\s*(?:\.{3,}|…+)\s*\]', ' <gap> ', text)
    text = re.sub(
        r'\(\s*(?:\.{3,}|…+)\s*(?:lacuna(?:e)?|break(?:en)?(?:\s+lines?)?)?\s*\)',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(r'\(\s*lacuna(?:e)?\s*\)', ' <gap> ', text, flags=re.I)
    text = re.sub(r'…+', ' <gap> ', text)
    text = re.sub(r'\.{3,}', ' <gap> ', text)
    text = re.sub(r'x{3,}', ' <gap> ', text, flags=re.I)
    text = re.sub(r'\bx(\s+x){2,}\b', ' <gap> ', text, flags=re.I)
    
    # 2. 小间隙标记 → <gap>
    text = re.sub(r'\(x\)', ' <gap> ', text, flags=re.I)
    text = re.sub(r'\[x\]', ' <gap> ', text, flags=re.I)
    text = re.sub(r'(?<![a-zA-Z\-₀₁₂₃₄₅₆₇₈₉])x(?![a-zA-Z\-₀₁₂₃₄₅₆₇₈₉])', ' <gap> ', text)
    
    # 3. 合并相邻 <gap>（两侧无连字符连接）
    prev = None
    while prev != text:
        prev = text
        # 激进合并：由空格、连字符、点分隔的连续 gap 统一归约为一个 <gap>
        text = re.sub(r'(?:<gap>[\s\-\.]*)+<gap>', '<gap>', text, flags=re.I)
        text = re.sub(r'\s+', ' ', text)

    # 4. 确保 <gap> 两侧有空格（连字符除外）
    #    修复 data_final 中 letter<gap> 和 <gap>letter 粘连问题
    text = re.sub(r'(?<![-\s])<gap>', ' <gap>', text, flags=re.I)
    text = re.sub(r'<gap>(?![-\s])', '<gap> ', text, flags=re.I)
    
    # 5. 去掉连字符与 <gap> 之间的多余空格：X- <gap> → X-<gap>，<gap> -X → <gap>-X
    #    仅当连字符连接词时粘连，破折号（- 后跟空格）保持不变
    text = re.sub(r'-\s+<gap>', '-<gap>', text, flags=re.I)
    text = re.sub(r'<gap>\s+-(?=\S)', '<gap>-', text, flags=re.I)  # <gap> -word → <gap>-word, but keep <gap> - word
    text = re.sub(r'<gap>\s+([.,;:!?])', r'<gap>\1', text, flags=re.I)
    
    # 6. 保持 <gap> 作为独立 token，不吞掉其右侧空格
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def normalize_characters(text: str, is_transliteration: bool = True) -> str:
    """字符规范化"""
    if not isinstance(text, str):
        return ""
    
    # 0. 强杀 ʾ (U+02BE) — 全局统一删除，不保留
    text = text.replace('\u02BE', '')
    
    # 0.1 CDLI/ORACC ASCII 到 Diacritics 转换 (如 sz -> š)
    if is_transliteration:
        text = text.replace('sz', 'š').replace('SZ', 'Š')
        text = text.replace('s,', 'ṣ').replace('S,', 'Ṣ')
        text = text.replace('t,', 'ṭ').replace('T,', 'Ṭ')
    
    # 1. 核心替换：ḫ (组合符或单字符) -> h
    text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    
    # 1.1 阿卡德语数字/下标标准化 (User Provided Logic)
    # 规则1: 元音 (a, e, i, u) + 2 或 3 (下标或普通数字) -> 变音符号
    # 规则2: 剩下的所有下标 -> 普通 ASCII 数字
    
    # 定义映射表
    accent_map = {
        'a': {'2': 'á', '3': 'à', '₂': 'á', '₃': 'à'},
        'e': {'2': 'é', '3': 'è', '₂': 'é', '₃': 'è'},
        'i': {'2': 'í', '3': 'ì', '₂': 'í', '₃': 'ì'},
        'u': {'2': 'ú', '3': 'ù', '₂': 'ú', '₃': 'ù'},
        # 大写支持 (Sumerograms)
        'A': {'2': 'Á', '3': 'À', '₂': 'Á', '₃': 'À'},
        'E': {'2': 'É', '3': 'È', '₂': 'É', '₃': 'È'},
        'I': {'2': 'Í', '3': 'Ì', '₂': 'Í', '₃': 'Ì'},
        'U': {'2': 'Ú', '3': 'Ù', '₂': 'Ú', '₃': 'Ù'}
    }

    def replace_accent_match(match):
        char = match.group(1)
        num = match.group(2)
        # 如果是元音，就查表替换
        if char in accent_map and num in accent_map[char]:
            return accent_map[char][num]
        return match.group(0) # 否则原样返回

    # 正则：匹配 [元音] 后面紧跟 [2/3/₂/₃]
    # (?i) 在 regex flag 中设置，或直接包含大小写
    if is_transliteration:
         text = re.sub(r"([aAeEiIuU])(2|3|₂|₃)", replace_accent_match, text)

    # 处理剩余数字 (Digits)
    # 此时剩下的 ₂/₃ 肯定是不贴元音的，或者是脏数据，直接转数字
    # 以及其他下标 ₀ ₁ ₄ ₅ ₆ ₇ ₈ ₉
    sub_to_normal = str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    text = text.translate(sub_to_normal)
    
    # 上标数字也统一转为普通数字
    superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    text = text.translate(superscript_map)
    
    # 1.2 移除半括号 ˹ ˺ (partially broken signs) - Dataset Instructions
    text = text.replace('˹', '').replace('˺', '')
    
    # 1.3 移除双尖括号 << >> (errant/erroneous signs) - Dataset Instructions
    text = re.sub(r'<<[^>]*>>', '', text)
    
    # 1.4 移除词分隔符 : 和 . (word divider) - 仅在音译中
    # 注意：保留句末标点，只移除词间分隔符
    
    # 1.5 限定词格式统一：(d) -> {d}（测试集使用花括号格式）
    # 根据官方说明：{tug₂} 或 {tug2} 应该是 {túg}
    # 常见限定词: d (神), m (男性), f (女性), ki (地名), túg (纺织品) 等
    if is_transliteration:
        text = re.sub(r'\(d\)', '{d}', text)
        text = re.sub(r'\(f\)', '{f}', text)
        text = re.sub(r'\(m\)', '{m}', text)
        text = re.sub(r'\(ki\)', '{ki}', text, flags=re.I)
        text = re.sub(r'\(lu[2₂]?\)', '{lú}', text, flags=re.I)
        text = re.sub(r'\(tug[2₂]?\)', '{túg}', text, flags=re.I)
        text = re.sub(r'\(giš\)', '{giš}', text, flags=re.I)
        text = re.sub(r'\(gisz\)', '{giš}', text, flags=re.I)
        text = re.sub(r'\(e[2₂]?\)', '{é}', text, flags=re.I)
        text = re.sub(r'\(id[2₂]?\)', '{íd}', text, flags=re.I)
        text = re.sub(r'\(u[2₂]?\)', '{ú}', text, flags=re.I)
        text = re.sub(r'\(uru\)', '{uru}', text, flags=re.I)
        text = re.sub(r'\(kur\)', '{kur}', text, flags=re.I)
        text = re.sub(r'\(geš\)', '{geš}', text, flags=re.I)
        text = re.sub(r'\(ĝeš\)', '{ĝeš}', text, flags=re.I)
        text = re.sub(r'\(mul\)', '{mul}', text, flags=re.I)
        text = re.sub(r'\(dub\)', '{dub}', text, flags=re.I)
        text = re.sub(r'\(mušen\)', '{mušen}', text, flags=re.I)
        text = re.sub(r'\(kuš\)', '{kuš}', text, flags=re.I)
        text = re.sub(r'\(na4\)', '{na4}', text, flags=re.I)
        # 已有花括号的也统一格式
        text = re.sub(r'\{tug2\}', '{túg}', text, flags=re.I)
        text = re.sub(r'\{lu2\}', '{lú}', text, flags=re.I)
        text = re.sub(r'\{e2\}', '{é}', text, flags=re.I)
        text = re.sub(r'\{id2\}', '{íd}', text, flags=re.I)
        text = re.sub(r'\{u2\}', '{ú}', text, flags=re.I)
    
    # 处理音译中的 + 为 -，翻译中则移除
    if is_transliteration:
        text = text.replace('+', '-')
    else:
        text = text.replace('+', ' ')
    
    # 2. 基础字符映射 (只处理非官方列表中的变音符)
    # 官方 Transliteration 字符列表包含：á à é è í ì ú ù (及大写)，必须保留！
    # 官方 Translation 字符列表包含：ā ī ū ē，必须保留！
    # 只转换非官方的变音符（如 ê ô â 等）
    if is_transliteration:
        # Transliteration: 保留 á à é è í ì ú ù，只转换非标准的
        norm_map = {
            'ē': 'e', 'ê': 'e', 'ě': 'e',  # 保留 é è
            'ī': 'i', 'î': 'i', 'ǐ': 'i',  # 保留 í ì
            'ō': 'o', 'ô': 'o', 'ǒ': 'o', 'ó': 'o', 'ò': 'o',  # o 系列全转（官方无）
            'ū': 'u', 'û': 'u', 'ǔ': 'u',  # 保留 ú ù
            'ā': 'a', 'â': 'a', 'ǎ': 'a',  # 保留 á à
            'Ē': 'E', 'Ê': 'E', 'Ě': 'E',
            'Ī': 'I', 'Î': 'I', 'Ǐ': 'I',
            'Ō': 'O', 'Ô': 'O', 'Ǒ': 'O', 'Ó': 'O', 'Ò': 'O',
            'Ū': 'U', 'Û': 'U', 'Ǔ': 'U',
            'Ā': 'A', 'Â': 'A', 'Ǎ': 'A',
        }
    else:
        # Translation: 保留 ā ī ū ē (长音符用于专名)，转换其他
        norm_map = {
            'ê': 'e', 'ě': 'e', 'é': 'e', 'è': 'e',  # 保留 ē
            'î': 'i', 'ǐ': 'i', 'í': 'i', 'ì': 'i',  # 保留 ī
            'ō': 'o', 'ô': 'o', 'ǒ': 'o', 'ó': 'o', 'ò': 'o',
            'û': 'u', 'ǔ': 'u', 'ú': 'u', 'ù': 'u',  # 保留 ū
            'â': 'a', 'ǎ': 'a', 'á': 'a', 'à': 'a',  # 保留 ā
            'Ê': 'E', 'Ě': 'E', 'É': 'E', 'È': 'E',
            'Î': 'I', 'Ǐ': 'I', 'Í': 'I', 'Ì': 'I',
            'Ō': 'O', 'Ô': 'O', 'Ǒ': 'O', 'Ó': 'O', 'Ò': 'O',
            'Û': 'U', 'Ǔ': 'U', 'Ú': 'U', 'Ù': 'U',
            'Â': 'A', 'Ǎ': 'A', 'Á': 'A', 'À': 'A',
        }
    for old, new in norm_map.items():
        text = text.replace(old, new)

    # 3. 彻底清除白名单外的 Unicode (只保留 ASCII + 官方变音符 + 分数)
    # 官方字符列表: š ṣ ṭ + 变音符 á à é è í ì ú ù (transliteration) + ā ī ū ē (translation)
    protected = set("šṣṭéèāáàēīíìūúùŠṢṬÉÈĀÁÀĒĪÍÌŪÚÙ" + "½⅓⅔¼¾⅕⅙⅚⅛⅜⅝⅞" + "ĝğ")  # ʾ (U+02BE) 已从保护名单移除，全局删除
    
    # 使用 NFC 归一化后检查每个字符
    text = unicodedata.normalize('NFC', text)
    res = []
    for char in text:
        if char in protected or ord(char) < 128:
            res.append(char)
        else:
            # 剥离变音标记 (Mn 类别)
            decomposed = unicodedata.normalize('NFKD', char)
            # 再次检查分解后的基础字符是否在允许范围内
            for c in decomposed:
                if unicodedata.category(c) != 'Mn':
                    if ord(c) < 128: res.append(c)
                    else: res.append(' ') # 完全未知的 Unicode 替换为空格保护边界
    
    text = "".join(res)
    
    # 4. 下标/上标数字还原
    subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉ₓ', '0123456789x')
    text = text.translate(subscript_map)
    
    # 上标数字
    superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    text = text.translate(superscript_map)
    
    # 4.1 浮点截断：防止浮点精度伪影 (如 1.3333300000000001 → 1.3333)
    #     保留最多 4 位小数，convert_fractions 后续会将其转为 Unicode 分数
    text = re.sub(r'(\d+\.\d{4})\d+', r'\1', text)
    
    # 5. 特殊字符
    text = text.replace('ş', 'ṣ').replace('ș', 'ṣ').replace('ț', 'ṭ').replace('İ', 'I').replace('ı', 'i')
    
    # 6. NFC 还原：确保最终输出为组合后的标准字符
    text = unicodedata.normalize('NFC', text)
    
    return text


def convert_fractions(text: str) -> str:
    """分数标准化：将小数和文字描述统一为符号（屏蔽译者习惯差异）"""
    if not isinstance(text, str):
        return ""
    
    # 1. 常见的文字描述转符号
    verbal_fractions = {
        r'\bone\s+half\b': '½',
        r'\bone-half\b': '½',
        r'\bone\s+third\b': '⅓',
        r'\btwo\s+thirds\b': '⅔',
        r'\bone\s+fourth\b': '¼',
        r'\bthree\s+fourths\b': '¾',
        r'\bone\s+quarter\b': '¼',
    }
    for pattern, replacement in verbal_fractions.items():
        text = re.sub(pattern, replacement, text, flags=re.I)

    # 2. 小数转分数（支持截断形式如 .3→⅓, .6/.66→⅔ 等）
    # 注意顺序：长模式先匹配，避免短模式误匹配
    patterns = [
        # 精确匹配（优先）
        (r'\b0\.125\b', '⅛'), (r'\b0\.375\b', '⅜'), (r'\b0\.625\b', '⅝'), (r'\b0\.875\b', '⅞'),
        (r'\b0\.25\b', '¼'), (r'\b0\.75\b', '¾'),
        (r'\b0\.5\b', '½'),
        # 重复小数（含截断形式）
        (r'\b0\.833+\d*\b', '⅚'), (r'\b0\.83\b', '⅚'),
        (r'\b0\.666+\d*\b', '⅔'), (r'\b0\.66\b', '⅔'), (r'\b0\.6\b', '⅔'),
        (r'\b0\.333+\d*\b', '⅓'), (r'\b0\.33\b', '⅓'), (r'\b0\.3\b', '⅓'),
        (r'\b0\.166+\d*\b', '⅙'), (r'\b0\.16\b', '⅙'),
        (r'(?<!\d)\.5\b', '½'),
    ]
    for pattern, replacement in patterns:
        text = re.sub(pattern, replacement, text)
    
    # 整数+分数之间必须有空格（测试集 ground truth 格式：2 ½ 而非 2½）
    # 同样支持截断形式
    text = re.sub(r'\b(\d+)\.125\b', r'\1 ⅛', text)
    text = re.sub(r'\b(\d+)\.375\b', r'\1 ⅜', text)
    text = re.sub(r'\b(\d+)\.625\b', r'\1 ⅝', text)
    text = re.sub(r'\b(\d+)\.875\b', r'\1 ⅞', text)
    text = re.sub(r'\b(\d+)\.25\b', r'\1 ¼', text)
    text = re.sub(r'\b(\d+)\.75\b', r'\1 ¾', text)
    text = re.sub(r'\b(\d+)\.5\b', r'\1 ½', text)
    text = re.sub(r'\b(\d+)\.833+\d*\b', r'\1 ⅚', text)
    text = re.sub(r'\b(\d+)\.83\b', r'\1 ⅚', text)
    text = re.sub(r'\b(\d+)\.666+\d*\b', r'\1 ⅔', text)
    text = re.sub(r'\b(\d+)\.66\b', r'\1 ⅔', text)
    text = re.sub(r'\b(\d+)\.6\b', r'\1 ⅔', text)
    text = re.sub(r'\b(\d+)\.333+\d*\b', r'\1 ⅓', text)
    text = re.sub(r'\b(\d+)\.33\b', r'\1 ⅓', text)
    text = re.sub(r'\b(\d+)\.3\b', r'\1 ⅓', text)
    text = re.sub(r'\b(\d+)\.166+\d*\b', r'\1 ⅙', text)
    text = re.sub(r'\b(\d+)\.16\b', r'\1 ⅙', text)
    
    # 3. ASCII 分数 X / Y → Unicode（官方确认测试集无 /）
    # 有精确 Unicode 的直接转换；无精确 Unicode 的用最近似
    ascii_fracs = [
        # 精确对应
        (r'\b1\s*/\s*2\b', '½'),
        (r'\b1\s*/\s*3\b', '⅓'),
        (r'\b2\s*/\s*3\b', '⅔'),
        (r'\b1\s*/\s*4\b', '¼'),
        (r'\b3\s*/\s*4\b', '¾'),
        (r'\b1\s*/\s*5\b', '⅕'),
        (r'\b1\s*/\s*6\b', '⅙'),
        (r'\b5\s*/\s*6\b', '⅚'),
        (r'\b1\s*/\s*8\b', '⅛'),
        (r'\b3\s*/\s*8\b', '⅜'),
        (r'\b5\s*/\s*8\b', '⅝'),
        (r'\b7\s*/\s*8\b', '⅞'),
        # /12 分数不转 Unicode，由 clean_translation_light 中的格令规则处理
    ]
    for pattern, replacement in ascii_fracs:
        text = re.sub(pattern, replacement, text)
    
    # 4. 归一化已有的整数紧贴Unicode分数（如 1½ → 1 ½）
    text = re.sub(r'(\d)([½¼¾⅓⅔⅙⅚⅛⅜⅝⅞])', r'\1 \2', text)
    
    return text


def normalize_punctuation(text: str) -> str:
    """
    标点符号规范化（保守策略）
    
    原则："只洗脏的，不洗丑的"
    - 不合并重复标点（保护 ... 的 Gap 语义）
    - 不删除标点前空格（避免 BLEU 不匹配）
    - 不删除首尾连字符（保护阿卡德语音节）
    """
    if not isinstance(text, str):
        return ""
    
    # 只做最基础的空白符清洗
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


# ============================================================
# 输出后处理（供 train.py 和 infer.py 使用）
# ============================================================

def build_onomasticon(lexicon: Dict[str, str]) -> Set[str]:
    """
    从词典构建 Onomasticon（专有名词标准形式集合）
    只包含人名、地名、神名
    """
    onomasticon = set()
    for eng_name in lexicon.values():
        if not isinstance(eng_name, str):
            continue
        name = eng_name.strip()
        if name and name[0].isupper() and len(name) > 2:
            common_words = {'The', 'This', 'That', 'And', 'But', 'For', 'Not', 'With', 'From', 'Into'}
            if name not in common_words:
                onomasticon.add(name)
    return onomasticon


def build_calibrated_lexicon(train_df: pd.DataFrame, lexicon: Dict[str, str],
                             output_path: str = None) -> Dict[str, str]:
    """
    用训练数据 TL↔TR 对反向校准词典拼写。
    
    原始 eBL 词典的 norm 值经常与翻译者使用的拼写不一致（如 Šunilka vs Šunilga），
    导致 59.6% 的 hint 是错误的。本函数通过统计训练数据中实际出现的拼写来校准。
    
    策略：
    1. 提取 TL 中限定词（DUMU/KIŠIB/{d}/{ki}）相邻的 form
    2. 在对应 TR 中用模糊匹配找到实际使用的名字拼写
    3. 取最高频的匹配作为校准后的 norm
    
    Returns:
        校准后的词典 {form: calibrated_norm}
    """
    from collections import defaultdict
    from difflib import SequenceMatcher
    
    _MIN_FORM_LEN = 4
    _STOPFORMS = {
        "a-na", "um-ma", "i-na", "ù", "ú", "u",
        "ša", "ša-ma", "ki-ma", "ki-i", "ki",
        "la", "ul", "ul-ma",
        "a-ta", "a-ti", "a-tù-nu", "a-hu-a", "a-ma", "i-ma",
        "ma-ma", "ma-ma-an", "a-ba", "a-bi",
        "šu-ma", "šu-ut", "i-li", "i-la",
        "a-dí-in", "i-dí-in", "a-hi-a", "a-bi-a",
        "ma-na", "ma-lá", "sig5",
        "ku-ta-nim", "sà-ra-tim", "ba-a-nim", "a-le-e", "am-ra-ma", "a-lá-ni",
    }
    
    def _extract_det_forms(tl):
        """从 TL 中提取限定词相邻的 form"""
        forms = []
        words = tl.split()
        n = len(words)
        for i, word in enumerate(words):
            next_form = None
            if word in ('DUMU', 'KIŠIB', '{d}') and i + 1 < n:
                next_form = words[i+1].strip('.,;:?!()[]"\'>').lower()
            elif '{ki}' in word:
                next_form = word.replace('{ki}', '').strip('.,;:?!()[]"\'>').lower()
            if next_form and len(next_form) >= _MIN_FORM_LEN and next_form not in _STOPFORMS:
                forms.append(next_form)
        return forms
    
    # Step 1: 收集 form → TR 名字匹配
    calibration_counts = defaultdict(lambda: defaultdict(int))
    
    for _, row in train_df.iterrows():
        tl = str(row.get('transliteration', ''))
        tr = str(row.get('translation', ''))
        if not tl or not tr:
            continue
        
        forms = _extract_det_forms(tl)
        if not forms:
            continue
        
        # 提取 TR 中所有首字母大写的词（候选专名）
        tr_names = re.findall(r'\b[A-Z][a-zāēīūšṣṭḫ]+(?:-[A-Za-zāēīūšṣṭḫ]+)*\b', tr)
        if not tr_names:
            continue
        
        for form in forms:
            form_syl = form.replace('-', '')
            
            # 先精确匹配词典
            if form in lexicon:
                norm = lexicon[form]
                if norm.lower() in tr.lower():
                    calibration_counts[form][norm] += 1
                    continue
            
            # 模糊匹配 TR 名字
            best_match = None
            best_ratio = 0.0
            for name in tr_names:
                name_lower = name.lower().replace('-', '')
                ratio = SequenceMatcher(None, form_syl, name_lower).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = name
            
            if best_match and best_ratio >= 0.6:
                calibration_counts[form][best_match] += 1
    
    # Step 2: 取最高频匹配作为校准后的 norm
    calibrated = {}
    ambiguous = 0
    for form, norms in calibration_counts.items():
        sorted_norms = sorted(norms.items(), key=lambda x: -x[1])
        top_norm, top_count = sorted_norms[0]
        second_count = sorted_norms[1][1] if len(sorted_norms) > 1 else 0
        
        # 最高频明显占优时选用（≥2倍第二名）
        if top_count >= max(2 * second_count, 1):
            calibrated[form] = top_norm
        else:
            ambiguous += 1
    
    # Step 3: 补充原始词典中未出现在训练数据的条目（fallback）
    for form, norm in lexicon.items():
        if form not in calibrated and len(form) >= _MIN_FORM_LEN and form not in _STOPFORMS:
            calibrated[form] = norm
    
    # 按 key 长度倒序（长的优先匹配）
    calibrated = dict(sorted(calibrated.items(), key=lambda x: len(x[0]), reverse=True))
    
    # 统计
    cal_only = sum(1 for f in calibrated if f in calibration_counts)
    fallback_only = len(calibrated) - cal_only
    print(f"   📚 校准词典: {len(calibrated)} 条 (训练数据校准={cal_only}, 原始词典 fallback={fallback_only}, 歧义跳过={ambiguous})")
    
    # 保存
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(calibrated, f, ensure_ascii=False, indent=2)
        print(f"   💾 保存校准词典: {output_path}")
    
    return calibrated


def build_high_precision_hint_lexicon(
    train_df: pd.DataFrame,
    output_path: str = None,
    min_support: int = 2,
    min_dominance: float = 0.8,
    enable_place_names: bool = False,
) -> Dict[str, str]:
    """
    构建高精度 hint 词典：只保留训练数据中高置信的 form→reference-style spelling 映射。
    
    与 build_calibrated_lexicon 的关键区别：
    1. 不依赖原始 eBL lexicon，完全从训练数据 TL↔TR 对中提取
    2. 不做 raw fallback——未命中的 form 直接跳过
    3. 默认不纳入 {ki} 触发（地名翻译风格不稳定）
    4. 只保留 support >= min_support 且 top_count/total >= min_dominance 的条目
    
    Args:
        train_df: 训练数据 DataFrame，需包含 transliteration 和 translation 列
        output_path: 输出 JSON 路径（可选）
        min_support: 最小观测次数（默认 2）
        min_dominance: 最高频候选占比下限（默认 0.8）
        enable_place_names: 是否纳入 {ki} 地名触发（默认 False）
    
    Returns:
        高精度词典 {form: reference_style_spelling}
    """
    from collections import defaultdict
    from difflib import SequenceMatcher
    
    _MIN_FORM_LEN = 4
    _STOPFORMS = {
        "a-na", "um-ma", "i-na", "ù", "ú", "u",
        "ša", "ša-ma", "ki-ma", "ki-i", "ki",
        "la", "ul", "ul-ma",
        "a-ta", "a-ti", "a-tù-nu", "a-hu-a", "a-ma", "i-ma",
        "ma-ma", "ma-ma-an", "a-ba", "a-bi",
        "šu-ma", "šu-ut", "i-li", "i-la",
        "a-dí-in", "i-dí-in", "a-hi-a", "a-bi-a",
        "ma-na", "ma-lá", "sig5",
        "ku-ta-nim", "sà-ra-tim", "ba-a-nim", "a-le-e", "am-ra-ma", "a-lá-ni",
    }
    
    # 高置信触发词
    _TRIGGERS = {'DUMU', 'KIŠIB', '{d}'}
    
    def _extract_det_forms(tl):
        """从 TL 中提取限定词相邻的 form（仅高置信触发）"""
        forms = []
        words = tl.split()
        n = len(words)
        for i, word in enumerate(words):
            next_form = None
            if word in _TRIGGERS and i + 1 < n:
                next_form = words[i + 1].strip('.,;:?!()[]"\'>').lower()
            elif enable_place_names and '{ki}' in word:
                next_form = word.replace('{ki}', '').strip('.,;:?!()[]"\'>').lower()
            if next_form and len(next_form) >= _MIN_FORM_LEN and next_form not in _STOPFORMS:
                forms.append(next_form)
        return forms
    
    # Step 1: 收集 form → TR 候选名字
    form_candidates = defaultdict(lambda: defaultdict(int))
    total_forms_seen = 0
    
    for _, row in train_df.iterrows():
        tl = str(row.get('transliteration', ''))
        tr = str(row.get('translation', ''))
        if not tl or not tr:
            continue
        
        forms = _extract_det_forms(tl)
        if not forms:
            continue
        
        # 提取 TR 中的首字母大写专名候选
        tr_names = re.findall(r'\b[A-Z][a-zāēīūšṣṭḫ]+(?:-[A-Za-zāēīūšṣṭḫ]+)*\b', tr)
        if not tr_names:
            continue
        
        for form in forms:
            total_forms_seen += 1
            form_syl = form.replace('-', '')
            
            # 模糊匹配 TR 中的名字
            best_match = None
            best_ratio = 0.0
            for name in tr_names:
                name_lower = name.lower().replace('-', '')
                ratio = SequenceMatcher(None, form_syl, name_lower).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = name
            
            if best_match and best_ratio >= 0.6:
                form_candidates[form][best_match] += 1
    
    # Step 2: 过滤——只保留高置信条目
    high_precision = {}
    stats = {
        'total_unique_forms': len(form_candidates),
        'total_observations': total_forms_seen,
        'accepted': 0,
        'rejected_low_support': 0,
        'rejected_low_dominance': 0,
        'rejected_ambiguous': 0,
    }
    
    for form, norms in form_candidates.items():
        sorted_norms = sorted(norms.items(), key=lambda x: -x[1])
        top_norm, top_count = sorted_norms[0]
        total_count = sum(c for _, c in sorted_norms)
        
        # 过滤条件
        if total_count < min_support:
            stats['rejected_low_support'] += 1
            continue
        
        dominance = top_count / total_count
        if dominance < min_dominance:
            stats['rejected_low_dominance'] += 1
            continue
        
        high_precision[form] = top_norm
        stats['accepted'] += 1
    
    # 按 key 长度倒序（长的优先匹配）
    high_precision = dict(sorted(high_precision.items(), key=lambda x: len(x[0]), reverse=True))
    
    stats['final_entries'] = len(high_precision)
    stats['min_support'] = min_support
    stats['min_dominance'] = min_dominance
    stats['enable_place_names'] = enable_place_names
    stats['raw_fallback'] = False
    
    print(f"   📚 High-precision hint lexicon: {len(high_precision)} entries")
    print(f"      unique forms seen: {stats['total_unique_forms']}, observations: {stats['total_observations']}")
    print(f"      accepted: {stats['accepted']}, low_support: {stats['rejected_low_support']}, "
          f"low_dominance: {stats['rejected_low_dominance']}")
    print(f"      min_support={min_support}, min_dominance={min_dominance}, "
          f"place_names={'ON' if enable_place_names else 'OFF'}, raw_fallback=OFF")
    
    # 保存
    if output_path:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(high_precision, f, ensure_ascii=False, indent=2)
        print(f"   💾 Saved: {output_path}")
        
        # 保存 stats
        stats_path = output_path.replace('.json', '_stats.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"   💾 Stats: {stats_path}")
    
    return high_precision


def fuzzy_fix_proper_nouns(text: str, onomasticon: Set[str], threshold: float = 0.85) -> str:
    """
    使用 fuzzy matching 纠正专有名词拼写
    """
    if not onomasticon:
        return text
    # threshold>=1 等价于关闭 fuzzy；继续全量 SequenceMatcher 只会浪费 CPU
    if threshold >= 1.0:
        return text
    
    # 常见首字母大写的普通英文词汇（停用词），避免被错误纠正为阿卡德语专有名词
    stop_words = {
        "The", "A", "An", "This", "That", "These", "Those",
        "I", "He", "She", "It", "We", "They", "You",
        "And", "But", "Or", "Nor", "For", "Yet", "So",
        "In", "On", "At", "To", "From", "By", "With", "About", "Against", "Under",
        "Is", "Are", "Was", "Were", "Be", "Been", "Being",
        "Have", "Has", "Had", "Do", "Does", "Did",
        "Will", "Would", "Shall", "Should", "Can", "Could", "May", "Might", "Must",
        "If", "Then", "Else", "When", "While", "Where", "Why", "How",
        "My", "Your", "His", "Her", "Its", "Our", "Their",
        "Here", "There", "Now", "Then",
        "Thus", "Therefore", "However", "Furthermore",
        "To", "From",
    }
    
    words = text.split()
    result = []
    
    for word in words:
        if not word or not word[0].isupper():
            result.append(word)
            continue
        
        # 如果是常见英文停用词，跳过纠正
        if word.strip('.,;:?!()[]"\'') in stop_words:
            result.append(word)
            continue
        
        clean_word = re.sub(r'[^\w\-\u0100-\u017F]', '', word)
        if len(clean_word) < 3:
            result.append(word)
            continue
        
        if clean_word in onomasticon:
            result.append(word)
            continue
        
        best_match = None
        best_ratio = 0.0
        
        for standard_name in onomasticon:
            if abs(len(clean_word) - len(standard_name)) > 3:
                continue
            ratio = SequenceMatcher(None, clean_word.lower(), standard_name.lower()).ratio()
            if ratio > best_ratio and ratio >= threshold:
                best_ratio = ratio
                best_match = standard_name
        
        if best_match and best_ratio < 1.0:
            suffix = word[len(clean_word):] if len(word) > len(clean_word) else ''
            result.append(best_match + suffix)
        else:
            result.append(word)
    
    return ' '.join(result)


def trim_trailing_fragment(text: str) -> str:
    """
    截断尾部不完整句子（来自 chunky_v1_5_0 的高分策略）
    条件：文本超过 100 字符 且 最后一个字符是字母（非标点）
    → 从末尾往前找最后一个 .?! 截断
    """
    if not text:
        return text
    text = text.rstrip()
    if not text:
        return text
    if len(text) > 100 and text[-1].isalpha():
        for i in range(len(text) - 1, -1, -1):
            if text[i] in ".?!":
                return text[: i + 1]
            if text[i] in "'" and i > 0 and text[i - 1] in ".?!":
                return text[: i + 1]
    return text


def postprocess_output(text: str, onomasticon: Set[str] = None, 
                       use_trim_fragment: bool = False) -> str:
    """
    模型输出后处理（统一版本，供 train.py 和 infer.py 使用）
    
    处理流程：
    1. Gap 合并（多个连续 <gap> → 单个 <gap>）
    2. ḫ/Ḫ → h/H 转换
    3. 下标/上标数字规范化
    4. 分数转换
    5. 删除重复词
    6. 删除重复短语
    7. Fuzzy 纠正专有名词
    8. 清理空白
    9. (可选) 截断尾部不完整句子
    """
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # 1. Gap 合并：多个连续独立 <gap> 合并为一个（保留连字符 gap 如 <gap>-word）
    text = re.sub(r'(?<!-)<gap>(?:\s*<gap>)+(?!-)', '<gap>', text, flags=re.I)
    
    # 1.5 移除 forbidden chars（保护 <gap> 标签后统一删除）
    # 注意：不删 !?;" — 训练标签中频率 13-29%，删了会与参考答案不一致
    text = text.replace('<gap>', '\x00GAP\x00')
    _forbidden_trans = str.maketrans("", "", "<>⌈⌋⌊[]+ʾ")
    text = text.translate(_forbidden_trans)
    text = text.replace('\x00GAP\x00', '<gap>')
    
    # 2. ḫ/Ḫ → h/H
    text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    
    # 3. 下标/上标规范化
    subscripts = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    text = text.translate(subscripts)
    superscripts = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')
    text = text.translate(superscripts)
    
    # 4. 分数转换
    text = convert_fractions(text)
    
    # 5. 删除重复词
    text = re.sub(r'\b(\w+)(?:\s+\1\b)+', r'\1', text)
    
    # 6. 删除重复短语 (使用滑动窗口算法，覆盖 2-8 词的短语)
    text = remove_phrase_repeats(text)
    
    # 7. Fuzzy 纠正专有名词
    if onomasticon:
        text = fuzzy_fix_proper_nouns(text, onomasticon, threshold=1)
    
    # 8. 清理空白
    text = normalize_punctuation(text)
    
    # 9. (可选) 截断尾部不完整句子（消融开关，默认 False）
    if use_trim_fragment:
        text = trim_trailing_fragment(text)
    
    return text


def remove_phrase_repeats(text: str) -> str:
    """
    使用滑动窗口去除重复短语 (借鉴 example1)
    处理 2-8 个词的重复短语
    """
    if not text or len(text) < 10:
        return text
        
    words = text.split()
    if len(words) < 4:
        return text
    
    # 从长到短尝试去重
    for phrase_len in range(8, 1, -1):
        i = 0
        result_words = []
        has_change = False
        
        while i < len(words):
            # 检查是否有足够的单词构成两个连续的短语
            if i + phrase_len * 2 <= len(words):
                phrase = words[i : i + phrase_len]
                next_phrase = words[i + phrase_len : i + phrase_len * 2]
                
                if phrase == next_phrase:
                    # 发现重复
                    has_change = True
                    # 保留一份
                    result_words.extend(phrase)
                    j = i + phrase_len
                    # 跳过所有后续的连续重复
                    while j + phrase_len <= len(words) and words[j : j + phrase_len] == phrase:
                        j += phrase_len
                    i = j
                    continue
            
            result_words.append(words[i])
            i += 1
            
        if has_change:
            words = result_words
            
    return " ".join(words)


def clean_transliteration_noise(text: str) -> str:
    """专门清理音译列的噪声，保留连字符和等号（对齐 OptimizedPreprocessor 逻辑）"""
    if not isinstance(text, str):
        return ""
    text = text.replace('_', ' ')
    text = re.sub(r'\((?:space|blank|vacat)\)', ' ', text, flags=re.I)
    text = text.replace('⸢', '').replace('⸣', '').replace('⸤', '').replace('⸥', '')
    text = text.replace('⌈', '').replace('⌉', '')
    # 0.5 缩写展开（在移除标点之前）
    text = re.sub(r'KÙ\.B(?:\.)?(?![A-Za-zÀ-ÿ])', 'KÙ.BABBAR', text)
    # 1. 移除噪音标点（官方指定的现代抄录符号）
    # 注意：保留 . (Sumerogram 分隔符 KÙ.BABBAR, AN.NA)
    _noise_trans = str.maketrans("", "", "!?/:")
    text = text.translate(_noise_trans)
    # 2. 移除 errant signs << >>
    text = re.sub(r'<<|>>', '', text)
    # 2.5 限定词花括号展开：{TÚG}→TÚG, {d}→{d}（保留短限定词，展开长限定词）
    # v2 数据中有 {TÚG}，v3 已转为 bare TÚG
    text = re.sub(r'\{(TÚG)\}', r'\1', text)
    # 3. 移除 scribal brackets ˹ ˺ 和方括号/圆括号（保留内容）
    for b in ['[', ']', '(', ')', '˹', '˺', '［', '］', '（', '）', '【', '】']:
        text = text.replace(b, '')
    # 4. 安全移除尖括号 < >（保护 <gap>）
    text = text.replace('<gap>', '\x00GAP\x00')
    text = text.replace('<', '').replace('>', '')
    text = text.replace('\x00GAP\x00', '<gap>')
    text = re.sub(r'\b(?:space|blank|vacat)\b', ' ', text, flags=re.I)
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'(?<=\w)\s*\.\s*(?=\w)', '.', text)
    # 5. 确保单空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def strip_archibab_transliteration_line_prefixes(text: str) -> str:
    """Remove edition line-number prefixes from ArchiBab transliterations when they form a clear sequence."""
    if not isinstance(text, str):
        return ""

    text = decode_literal_whitespace_escapes(text)
    if '\n' not in text and '\r' not in text:
        return text

    lines = text.splitlines()
    prefix_re = re.compile(r'^\s*(\d{1,3})(?:[\'’])?(?=\s+(?:[\[(<{]|[^\W\d_]))')
    numbered = []
    nonempty_lines = 0

    for line in lines:
        if line.strip():
            nonempty_lines += 1
        match = prefix_re.match(line)
        numbered.append((line, match))

    nums = [int(match.group(1)) for _, match in numbered if match]
    if len(nums) < 3 or nonempty_lines == 0:
        return text

    # Typical edition numbering is an almost monotone sequence with small forward jumps,
    # often mixed with continuation lines that have no prefix.
    forward_steps = [
        nums[idx] - nums[idx - 1]
        for idx in range(1, len(nums))
        if 0 < (nums[idx] - nums[idx - 1]) <= 4
    ]
    unmatched_nonempty = sum(1 for line, match in numbered if line.strip() and match is None)
    looks_like_line_numbers = (
        len(forward_steps) >= max(2, len(nums) - 2)
        and unmatched_nonempty > 0
        and len(nums) >= max(3, nonempty_lines // 3)
    )
    if not looks_like_line_numbers:
        return text

    cleaned_lines = []
    for line, match in numbered:
        if match:
            line = line[match.end():].lstrip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


_ARCHIBAB_TRANSLATION_HEADER_RE = re.compile(r'(?i)\bTraduction(?:\s+[A-Za-z.]+)?\s*:')
_ARCHIBAB_PLACEHOLDER_NAME_RE = re.compile(r'\b(?:PN(?:\d+)?|DN|GN|RN)\b')
_ARCHIBAB_STRONG_META_RE = re.compile(
    r'Traduction(?:\s+[A-Za-z.]+)?\s*:'
    r'|T moins et date'
    r'|too broken to be translated'
    r'|remainder lost'
    r'|reverse broken'
    r'|rest of the (?:(?:obv|rev)\.?(?: is)?|obverse|reverse)(?: and reverse)? (?:broken|lost|not inscribed)'
    r'|the standard Late OB greeting formulae'
    r'|greeting formulae ABCDE',
    re.I,
)
_ARCHIBAB_EDITORIAL_META_RE = re.compile(
    r'(?:^|[\s(])(?:lines?\s*\d+[^\s)]*'
    r'|fragmentary'
    r'|remainder lost'
    r'|reverse broken'
    r'|rest of the obverse'
    r'|rest of the reverse'
    r'|obv\.'
    r'|rev\.)',
    re.I,
)


def _roman_to_int(roman: str) -> int | None:
    values = {'I': 1, 'V': 5, 'X': 10}
    total = 0
    prev = 0
    for ch in reversed(str(roman or '').upper()):
        val = values.get(ch)
        if val is None:
            return None
        if val < prev:
            total -= val
        else:
            total += val
            prev = val
    return total if total > 0 else None


def clean_archibab_translation_residue(text: str) -> str:
    """Strip ArchiBab-specific editorial residue from English translations."""
    if not isinstance(text, str):
        return ""

    text = decode_literal_whitespace_escapes(text)
    text = text.replace('PN₂', 'PN2')
    text = text.replace('”', '"').replace('“', '"').replace('’', "'").replace('‘', "'")
    text = text.replace('–', '-').replace('—', '-')

    if _ARCHIBAB_TRANSLATION_HEADER_RE.search(text):
        segments = [
            seg.strip(" \n\r\t:;")
            for seg in _ARCHIBAB_TRANSLATION_HEADER_RE.split(text)
            if seg and seg.strip(" \n\r\t:;")
        ]
        if segments:
            text = segments[-1]

    text = _ARCHIBAB_TRANSLATION_HEADER_RE.sub(' ', text)

    def _date_repl(match):
        day = match.group(1)
        roman = match.group(2)
        year = (match.group(3) or '').strip(" ,.")
        month_num = _roman_to_int(roman)
        if month_num is None:
            return f" Day {day}, {year}. " if year else f" Day {day}. "
        if year:
            return f" Day {day}, month {month_num}, {year}. "
        return f" Day {day}, month {month_num}. "

    text = re.sub(
        r'(?:(?<=\s)|(?<=^)|(?<=[;,.]))-\s*(\d{1,2})\.\s*([IVX]{1,4})\s*,\s*([^.!?]+(?:\d+|king[^.!?]*))\.?',
        _date_repl,
        text,
        flags=re.I,
    )

    # Editorial gap remarks should collapse to a simple <gap>, not survive as prose.
    text = re.sub(
        r'\(\s*(?:the rest of the (?:obv|rev)\.?(?: is)? broken|remainder lost|remainder too broken to be translated|'
        r'rest of the reverse lost|rest of the obverse broken|reverse not inscribed|'
        r'lines?\s*\d+[^\)]*(?:fragmentary|too broken to be translated|too damaged to be translated)|'
        r'\d+\s*-\s*\d+[^\)]*(?:fragmentary|too broken to be translated|too damaged to be translated)|'
        r'break|three lines destroyed)\s*\)',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*(?:(?:obverse|reverse)(?:\s+and\s+(?:obverse|reverse))?|beginning of (?:the )?(?:obverse|reverse)|'
        r'remainder of (?:the )?(?:obverse|reverse)|rest of (?:the )?(?:obverse|reverse)|'
        r'beginning of the text|(?:upper|lower) edge|obverse and lower edge|beginning of reverse|beginning of obverse|'
        r'(?:one|two|three|four|five|\d+)\s+(?:largely\s+)?broken lines?)'
        r'[^)]*(?:broken(?: away)?|missing|lost|not(?: been)? preserved|not(?: been)? inscribed|not(?: been)? deciphered)\s*\.?\)',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*(?:beginning broken|rest broken away|obverse mostly broken away|obverse lost|reverse lost|'
        r'remainder of the text too (?:broken|damaged|fragmentary|poor) for translation|'
        r'remainder of too for translation(?:;\s*beginning of [^)]*)?|gap of (?:one|two|three|four|five|\d+) broken lines|'
        r'remainder of obverse and beginning of [^)]*|beginning of reverse missing|beginning of obverse missing|'
        r'(?:one|two|three|four|five|\d+)\s+(?:largely\s+)?broken lines?)\s*\)',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*remainder of (?:the text|too)[^)]*(?:translation|preserved)[^)]*\)',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*beginning of (?:one|two|three|four|five|\d+)\s+lines?\s*\)',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*\d+[\'’]?(?:\s*-\s*\d+[\'’]?)?(?:\s+\d+[\'’]?)?\s*\)',
        ' ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*(?:\d+[\'’]?\s*-\s*\d+[\'’]?(?:\s+\d+[\'’]?)?|l\.\s*\d+\s*-\s*\d+)[^)]*(?:too|fragmentary|translation)?[^)]*\)',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*\d+[\'’]?\s*-\s*\d+[\'’]?(?:[^)]*?\b(?:Witness(?:es)?|Month|Notes?|Summary|Heading|text|broken|fragmentary|translation)\b[^)]*)\)',
        ' ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\(\s*the standard Late OB greeting formulae[^)]*\)',
        ' ',
        text,
        flags=re.I,
    )
    text = re.sub(r'\(\s*(?:rev|obv)\s*\)\.?\s*["\']?', ' ', text, flags=re.I)
    text = re.sub(r'(?<![A-Za-z])(?:rev|obv)\.\s*["\']?', ' ', text, flags=re.I)
    text = re.sub(r'\(\s*lines?\s*\d+[^\)]*\)', ' ', text, flags=re.I)
    text = re.sub(r'\b(lines?\s*\d+[^\s,.;:!?)]*(?:\s*(?:fragmentary|too broken to be translated|too damaged to be translated))?)\b', ' ', text, flags=re.I)
    text = re.sub(
        r'(?:(?<=^)|(?<=\s)|(?<=[;,.]))\d+[\'’]?\s*-\s*\d+[\'’]?(?=\s+\b(?:Witness(?:es)?|Month|Notes?|Summary|text|broken|fragmentary|translation)\b)',
        ' ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'(?:(?<=^)|(?<=\s)|(?<=[;,.]))\d+[\'’]?\s*-\s*\d+[\'’]?(?=\s+[A-Z<])',
        ' ',
        text,
        flags=re.I,
    )
    text = re.sub(
        r'\b(?:the beginning\s+)?rest of the (?:(?:obverse|reverse)(?:\s+and\s+reverse)?|(?:obv|rev)\.?)'
        r'(?:[^.!?;:]*?(?:broken|lost|not inscribed|fragmentary|too broken to be translated|too damaged to be translated))',
        ' <gap> ',
        text,
        flags=re.I,
    )
    text = re.sub(r'\b(remainder lost|fragmentary\w*|reverse broken|remainder too)\b', ' ', text, flags=re.I)
    text = re.sub(r'\(\s*\)', ' ', text)

    # Common OCR and editorial residue.
    text = text.replace('as/for', 'as for').replace('with/of', 'with')
    text = text.replace('goodhealth', 'good health')
    text = text.replace('wwork', 'work').replace('officie', 'office')
    text = text.replace('epoynmy', 'eponymy').replace('corv e', 'corvee')
    text = text.replace('cam here', 'came here')
    text = text.replace('Nabium-mšallim', 'Nabium-mušallim')
    text = text.replace('p. office', 'office')
    text = text.replace('p. officie', 'office')
    text = text.replace('remainder too )', '<gap>')
    text = text.replace('(remainder of too )', '<gap>')
    text = text.replace(' ,', ',')
    text = text.replace(' .', '.')
    text = text.replace(' ;', ';')
    text = text.replace(' :', ':')
    text = text.replace(' do not give [the barley]', ' do not give the barley')

    # Placeholder names are better represented as a single gap token.
    text = _ARCHIBAB_PLACEHOLDER_NAME_RE.sub('<gap>', text)
    text = re.sub(r'\b(?:PI|GUR|IKU)\b', lambda m: m.group(0).lower(), text)

    # Drop isolated editorial junk tokens left by OCR.
    text = re.sub(r'\b(?:A\. George|IA)\b(?=:)', ' ', text)
    text = re.sub(r'\s*-\s*(?=Day \d+,\s*month \d+)', ' ', text)
    text = re.sub(r'(?<=\s)\.(?=\s)', ' ', text)
    text = normalize_gaps(text)
    text = re.sub(r'\s+', ' ', text).strip(" \t\n\r-;:,")
    return text


def filter_archibab_fragments(df: pd.DataFrame) -> pd.DataFrame:
    """Drop clearly low-value ArchiBab rows that are still dominated by editorial residue."""
    if df.empty:
        return df

    df = df.copy()
    tr_series = df['translation'].astype(str)
    tl_series = df['transliteration'].astype(str)

    tr_tokens = tr_series.str.split().str.len().clip(lower=1)
    tl_tokens = tl_series.str.split().str.len().clip(lower=1)
    tr_gap_count = tr_series.str.count(r'<gap>')
    tl_gap_count = tl_series.str.count(r'<gap>')
    combined_gap_density = (tr_gap_count + tl_gap_count) / (tr_tokens + tl_tokens)
    tr_gap_density = tr_gap_count / tr_tokens

    strong_meta = tr_series.str.contains(_ARCHIBAB_STRONG_META_RE, na=False)
    editorial_meta = tr_series.str.contains(_ARCHIBAB_EDITORIAL_META_RE, na=False)
    placeholder_names = tr_series.str.contains(_ARCHIBAB_PLACEHOLDER_NAME_RE, na=False)
    ugly_ocr = tr_series.str.contains(
        re.compile(r'wwork|officie|epoynmy|goodhealth|corv e|mšallim|shekelI\d+|b n \(|sila\d', re.I),
        na=False,
    )

    drop_mask = (
        strong_meta
        | (placeholder_names & ((tr_gap_count >= 1) | (tr_tokens <= 18)))
        | (editorial_meta & (combined_gap_density >= 0.18))
        | (combined_gap_density >= 0.38)
        | ((tr_gap_density >= 0.22) & (tr_tokens <= 24))
        | (ugly_ocr & (tr_gap_count >= 1))
    )

    n_drop = int(drop_mask.sum())
    if n_drop:
        print(f"   🧹 ArchiBab 碎片过滤: {n_drop} 条")
        df = df.loc[~drop_mask].copy()
    return df


_SHORT_BRACE_TOKEN_RE = re.compile(r'^[0-9A-Za-zÀ-ÿĝĜḫḪšŠṣṢṭṬ]+$')
_MAX_BRACE_TOKEN_LEN = 8


def sanitize_determinative_braces(text: str) -> str:
    """Drop obviously broken brace spans while preserving short determinative-style tokens."""
    if not isinstance(text, str):
        return ""

    nested_brace_pattern = re.compile(r'\{[^{}]*\{[^{}]*\}[^{}]*\}')
    while True:
        text, n_nested = nested_brace_pattern.subn(' ', text)
        if n_nested == 0:
            break

    def _replace_brace(match):
        content = match.group(1).strip()
        if not content:
            return " "
        if any(ch.isspace() for ch in content):
            return " "
        compact = content.replace('₂', '2').replace('₄', '4')
        if len(compact) > _MAX_BRACE_TOKEN_LEN:
            return " "
        if _SHORT_BRACE_TOKEN_RE.fullmatch(compact) is None:
            return " "
        return f"{{{compact}}}"

    text = re.sub(r'\{([^{}]*)\}', _replace_brace, text)
    placeholders = []

    def _protect_valid(match):
        placeholders.append(match.group(0))
        return f"__DETBR_{len(placeholders) - 1}__"

    text = re.sub(r'\{[^{}]+\}', _protect_valid, text)
    text = text.replace('{', ' ').replace('}', ' ')
    for idx, value in enumerate(placeholders):
        text = text.replace(f"__DETBR_{idx}__", value)
    return re.sub(r'\s+', ' ', text).strip()


def preprocess_transliteration(text: str) -> str:
    """完整的 transliteration 预处理流程"""
    text = decode_literal_whitespace_escapes(text)
    text = normalize_characters(text, is_transliteration=True)
    text = convert_fractions(text)
    # 1. 识别标记 (Gap 转换)
    text = normalize_gaps(text)
    # 1.5 删除明显错误的花括号块：长内容、嵌套、未配对等；保留短 determinative 样式 token
    text = sanitize_determinative_braces(text)
    # 2. 清除噪声 (使用音译专用逻辑)
    text = clean_transliteration_noise(text)
    # 3. 再次合并 (处理由于清理带来的标记相邻)
    text = normalize_gaps(text)
    # 4. 最终字符封口：仅保留官方允许字符
    text = filter_to_official_allowed_characters(text, is_transliteration=True)
    return text


def preprocess_translation(text: str, light_mode: bool = True) -> str:
    """
    完整的 translation 预处理流程
    
    Args:
        text: 待处理文本
        light_mode: True=轻量清洗（保持原味），False=激进清洗
    """
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = repair_translation_escape_artifacts(text)
    
    # Gap 标准化（先于分数转换，因为 "x0.6666" 中的 x 会阻止 \b 匹配）
    text = normalize_gaps(text)
    
    # 分数转换（保留 Unicode 分数）
    text = convert_fractions(text)
    
    # 清洗：轻量 vs 激进
    if light_mode:
        text = clean_translation_light(text)
    else:
        text = normalize_characters(text, is_transliteration=False)
        text = clean_translation_noise(text)
        text = text.replace('ḫ', 'h').replace('Ḫ', 'H')
    
    # 再次合并 Gap
    text = normalize_gaps(text)
    # 最终字符封口：仅保留官方允许字符
    text = filter_to_official_allowed_characters(text, is_transliteration=False)
    text = repair_translation_linebreak_word_splits(text)
    
    return text


def is_likely_non_parallel(akk: str, eng: str) -> Tuple[bool, str]:
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


# ============================================================
# 句子切分算法
# ============================================================

def fuzzy_match_word(word1: str, word2: str, threshold: float = 0.8) -> bool:
    """模糊匹配两个词（未被使用，保留作备用）"""
    if not word1 or not word2:
        return False
    
    w1 = normalize_characters(word1.lower().strip())
    w2 = normalize_characters(word2.lower().strip())
    
    if w1 == w2:
        return True
    # 长度差异过大时直接拒绝
    len_ratio = min(len(w1), len(w2)) / max(len(w1), len(w2))
    if len_ratio < 0.5:
        return False
    if len(w1) < 2 or len(w2) < 2:
        return w1 == w2
    
    similarity = SequenceMatcher(None, w1, w2).ratio()
    return similarity >= threshold


def is_balanced(words, idx):
    """检查词序列中到 idx 位置的括号是否平衡"""
    text = " ".join(words[:idx])
    if text.count('[') != text.count(']'): return False
    if text.count('(') != text.count(')'): return False
    if text.count('<') != text.count('>'): return False
    return True

def find_best_split_index(words, target_idx, window=20):
    """
    寻找一个启发式较好的切分点（优先在标点后切分，且避免截断括号内的内容）
    """
    target_idx = max(1, min(len(words), target_idx))
    
    # 1. 向前寻找符合括号平衡的备选点
    valid_indices = []
    start_search = max(1, target_idx - window)
    for i in range(target_idx, start_search - 1, -1):
        if is_balanced(words, i):
            valid_indices.append(i)
            
    if not valid_indices:
        # 如果向前找不到，向后找
        for i in range(target_idx + 1, min(len(words), target_idx + window + 1)):
            if is_balanced(words, i):
                return i
        return target_idx # 实在找不到就原样返回
        
    # 2. 在符合平衡的备选点中，优先选择刚好在标点符号后面的点
    best_idx = valid_indices[0] # 最接近 target_idx 且平衡的点
    for idx in valid_indices:
        prev_word = words[idx - 1]
        # 如果上一个词以句号、分号、叹号、问号结尾，则是一个完美的断句点
        if re.search(r'[.;!?]$', prev_word):
            return idx
            
    return best_idx

def sliding_window_split_doc(oare_id, translit, translation, max_bytes=512, overlap_bytes=128, global_ratio=0.9713):
    """
    通过滑动窗口切分长文档，基于阿卡德语/英语平均字符比例 (0.9713) 估算英语对应位置。
    """
    ak_words = str(translit).split()
    en_words = str(translation).split()
    
    if not ak_words or not en_words:
        return []
        
    ak_spans = []
    curr = 0
    for w in ak_words:
        ak_spans.append((curr, curr + len(w), w))
        curr += len(w) + 1
        
    en_spans = []
    curr = 0
    for w in en_words:
        en_spans.append((curr, curr + len(w), w))
        curr += len(w) + 1
        
    chunks = []
    start_word_idx = 0
    
    while start_word_idx < len(ak_words):
        end_word_idx = start_word_idx
        while end_word_idx < len(ak_words):
            chunk_words = [w[2] for w in ak_spans[start_word_idx:end_word_idx+1]]
            chunk_text = " ".join(chunk_words)
            if len(chunk_text.encode('utf-8')) > max_bytes and end_word_idx > start_word_idx:
                break
            end_word_idx += 1
            
        # [启发式平滑] 优化阿卡德语切分点 (end_word_idx 实际上是要切分的右边界开区间)
        if end_word_idx < len(ak_words):
            raw_ak_words = [w[2] for w in ak_spans]
            end_word_idx = find_best_split_index(raw_ak_words, end_word_idx, window=20)
            # 防止切分点倒退太多导致无限循环
            if end_word_idx <= start_word_idx:
                end_word_idx = start_word_idx + 1
            
        ak_chunk_words = [w[2] for w in ak_spans[start_word_idx:end_word_idx]]
        ak_chunk_text = " ".join(ak_chunk_words)
        
        ak_start_char = ak_spans[start_word_idx][0]
        ak_end_char = ak_spans[end_word_idx - 1][1]
        
        en_target_start = int(ak_start_char / global_ratio)
        en_target_end = int(ak_end_char / global_ratio)
        
        en_start_idx = 0
        min_start_diff = float('inf')
        for i, (s, e, w) in enumerate(en_spans):
            diff = abs(s - en_target_start)
            if diff < min_start_diff:
                min_start_diff = diff
                en_start_idx = i
                
        en_end_idx = len(en_spans) - 1
        min_end_diff = float('inf')
        for i, (s, e, w) in enumerate(en_spans):
            diff = abs(e - en_target_end)
            if diff < min_end_diff:
                min_end_diff = diff
                en_end_idx = i
                
        if end_word_idx == len(ak_words):
            en_end_idx = len(en_spans) - 1
        else:
            # [启发式平滑] 优化英语切分点
            raw_en_words = [w[2] for w in en_spans]
            # en_end_idx 对应闭区间，因此转换为切分点需要 + 1
            best_en_split = find_best_split_index(raw_en_words, en_end_idx + 1, window=15)
            en_end_idx = best_en_split - 1
            
        en_end_idx = max(en_start_idx, en_end_idx)
            
        en_chunk_words = [w[2] for w in en_spans[en_start_idx:en_end_idx + 1]]
        en_chunk_text = " ".join(en_chunk_words)
        
        chunks.append({
            'oare_id': f"{oare_id}_sl{len(chunks)}",
            'transliteration': ak_chunk_text,
            'translation': en_chunk_text,
            'source': 'sliding_window'
        })
        
        if end_word_idx == len(ak_words):
            break
            
        back_idx = end_word_idx - 1
        current_overlap_bytes = 0
        while back_idx > start_word_idx:
            w_bytes = len(ak_spans[back_idx][2].encode('utf-8')) + 1
            if current_overlap_bytes + w_bytes > overlap_bytes:
                break
            current_overlap_bytes += w_bytes
            back_idx -= 1
            
        if back_idx == end_word_idx - 1 and end_word_idx - 1 > start_word_idx:
            back_idx = end_word_idx - 2
            
        start_word_idx = max(start_word_idx + 1, back_idx)

    return chunks
# ============================================================
# N-gram 统计
# ============================================================

def compute_ngram_stats(texts: List[str], n_values: List[int] = [2, 3, 4, 5, 6, 8]) -> Dict:
    """
    计算文本的 n-gram 重复率统计
    """
    from collections import defaultdict
    
    stats = {}
    
    for n in n_values:
        repeat_rates = []
        
        for text in texts:
            if not isinstance(text, str) or len(text.split()) < n:
                continue
            
            words = text.split()
            ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
            
            if not ngrams:
                continue
            
            ngram_counts = Counter(ngrams)
            total = len(ngrams)
            repeated = sum(1 for ng, cnt in ngram_counts.items() if cnt > 1)
            repeat_rate = repeated / len(ngram_counts) if ngram_counts else 0
            repeat_rates.append(repeat_rate)
        
        if repeat_rates:
            stats[n] = {
                'mean': np.mean(repeat_rates),
                'median': np.median(repeat_rates),
                'max': np.max(repeat_rates),
                'samples_with_repeat': sum(1 for r in repeat_rates if r > 0)
            }
    
    return stats


# ============================================================
# 质量过滤
# ============================================================

def crop_half_translations(df: pd.DataFrame, ratio_threshold: float = 1.5, _uncertainty: dict = None) -> pd.DataFrame:
    """
    启发式尾部裁剪：修复"半截子"翻译
    
    检测特征：翻译末尾以 <gap> 结尾，且 transliteration 远长于 translation（ratio > threshold）
    修复方式：按正常样本的 tl/tr 词数比估算 transliteration 应有长度，裁剪多余尾部并补 <gap>
    
    这样模型学到的是 (裁剪后的音译 + <gap>) → (半截翻译)，
    而不是 (完整长音译) → (半截翻译)，避免学到"话说一半就停"的坏习惯。
    """
    df = df.copy()
    
    # 1. 计算正常样本的 tl/tr 词数比（作为裁剪基准）
    tl_wc = df['transliteration'].str.split().str.len()
    tr_wc = df['translation'].str.split().str.len()
    word_ratio = tl_wc / tr_wc.clip(lower=1)
    
    # 正常样本：不以 <gap> 结尾 且 ratio 在合理范围内
    ends_gap = df['translation'].str.contains(r'<gap>\s*$', case=False, na=False)
    normal_mask = ~ends_gap & (word_ratio < 1.5)
    normal_ratio = word_ratio[normal_mask].median()  # 正常 tl/tr 词数比中位数
    
    # 2. 检测半截子
    # 条件A：末尾 <gap> + ratio > threshold（经典半截子）
    # 条件B：ratio > 2.5 + tl > 20词（极端不平衡，即使翻译不以 gap 结尾；覆盖 v3 扩展 TL 导致的错位）
    # 条件C：TR以逗号/分号/破折号结尾 + ratio > 2.0 + tl > 10词（翻译明显截断但不以 gap 结尾）
    ends_abrupt = df['translation'].str.contains(r'[,;\-]\s*$', na=False)
    suspect_mask = (ends_gap & (word_ratio > ratio_threshold)) | \
                   ((word_ratio > 2.5) & (tl_wc > 20)) | \
                   (ends_abrupt & (word_ratio > 2.0) & (tl_wc > 10))
    n_suspect = suspect_mask.sum()
    
    if n_suspect == 0:
        return df
    
    # 3. 裁剪 transliteration
    # 用 P90 ratio（约 1.2）作为估算上限，再加 30% buffer
    crop_ratio = min(normal_ratio * 1.3, 1.5)  # 保守估算
    n_cropped = 0
    
    for idx in df[suspect_mask].index:
        tl = df.at[idx, 'transliteration']
        tr = df.at[idx, 'translation']
        tr_words = len(tr.split())
        tl_words_list = tl.split()
        
        # 估算应保留的 transliteration 词数
        expected_tl_words = int(tr_words * crop_ratio) + 2  # +2 buffer
        
        if expected_tl_words < len(tl_words_list):
            # 裁剪：保留前 expected_tl_words 个词 + <gap>
            cropped = ' '.join(tl_words_list[:expected_tl_words]) + ' <gap>'
            # 合并裁剪点产生的连续 <gap>（如果裁剪点前一个词已经是 <gap>）
            cropped = normalize_gaps(cropped)
            df.at[idx, 'transliteration'] = cropped
            n_cropped += 1
            if _uncertainty is not None and 'oare_id' in df.columns:
                oid = df.at[idx, 'oare_id']
                _uncertainty.setdefault(oid, set()).add('tl_cropped')
    
    print(f"   ✂️ 启发式尾部裁剪: {n_cropped}/{n_suspect} 条半截子翻译 "
          f"(正常 ratio 中位数={normal_ratio:.2f}, 裁剪 ratio={crop_ratio:.2f})")
    
    return df


def quality_filter(df: pd.DataFrame) -> pd.DataFrame:
    """质量过滤"""
    original_len = len(df)
    
    # 过滤翻译过短的样本
    df = df[df['translation'].str.len() >= config.min_translation_length].copy()
    
    # 计算长度比率
    df['_translit_len'] = df['transliteration'].str.len()
    df['_transl_len'] = df['translation'].str.len()
    df['_ratio'] = df['_translit_len'] / (df['_transl_len'] + 1)
    
    # 过滤比率异常的样本
    df = df[(df['_ratio'] >= config.min_length_ratio) & 
            (df['_ratio'] <= config.max_length_ratio)].copy()
    
    # 2.5 黑名单过滤（逐条人工审核确认的垃圾样本，按 oare_id 精确匹配）
    # 纯封印签名、极度截断、只有地址行、翻译为空或仅一个名字
    BLACKLIST_OARE_IDS = {
        # --- 纯封印签名 / 极度截断 ---
        'efc3ed8c-e0a0-4902-8a5d-226001b68d96',  # Seal of Kakua, 4tl/6tr (ratio=0.67 但仅6词)
        '8f46f357-9971-427f-ac9a-7de23c9ce3a5',  # Seal of Nimar-Ištar, 53tl/6tr (ratio=8.8)
        'e18c267c-d397-4318-b821-c053d6f32818',  # Seal of Ištar-bāni, 43tl/6tr (ratio=7.2)
        '0d14c5a8-c152-4417-90cc-36b6dff3226d',  # Seal of Zuba, 4tl/6tr (仅6词)
        '5db6ccde-0832-482f-8dcd-e4da3138cf06',  # Seal of huluba, 34tl/6tr (ratio=5.7)
        '23d4244d-9cea-4c20-b040-c32f1f722ce3',  # 只有地址行, 53tl/6tr (ratio=8.8)
        '08f7b5b5-8a52-43af-a30a-ad9c11ce8de7',  # 翻译只有 "Lullu,", 68tl/1tr
        '7d9a743f-d211-4afa-a173-347c5b4731a3',  # 翻译几乎为空 (Seal of <gap>)
        '8eb60135-fe29-4a3b-bad6-8f629e15e0b7',  # "Obverse too broken for translation"
        '194a79ff-646e-41a4-9f8b-670620da1e54',  # 翻译仅 "To Nabi-Suen," (2词), 11tl/2tr
        # --- data_final 新增：严重错配 ---
        '8376cbda-b423-42d4-abb5-188d04896392',  # 43tl/473tr, 翻译是长债务记录，与音译完全不匹配
        'e3aecf83-f197-4c23-ae53-56162c679468',  # 57tl/584tr, 翻译远超音译内容，明显错配
        # --- data_final 中已修复，从黑名单移除 ---
        # '9a208f3b' → 音译已修复为阿卡德语
        # 'c97bb594' → 翻译已修正匹配
        # --- 以下 6 条已通过 v2 回填修复，从黑名单移除 ---
        # 'aa963c56' → v2 回填完整契约 (74w)
        # '629f1e04' → v2 回填完整借条 (21w)
        # '2e1ab3e9' → v2 回填完整借条 (33w)
        # '315acabf' → v2 回填完整账单 (59w)
        # '529fcbd8' → v2 回填完整债务 (40w)
        # 'e05f20dc' → v2 回填完整交易 (33w)
        # --- data_final 新增：极短翻译（无 v2 完整版可回填）---
        '821b6253-72c8-43a5-93e1-0b40b5f9a7fb',  # 10tl/3tr "1 drink: Aššur-nādā;" ratio=3.3
    }
    if 'oare_id' in df.columns:
        bl_mask = df['oare_id'].isin(BLACKLIST_OARE_IDS)
        n_bl = bl_mask.sum()
        if n_bl > 0:
            print(f"   🚫 黑名单过滤 (by oare_id): {n_bl} 条")
            df = df[~bl_mask].copy()
    
    # 3. 辅助任务过滤 (来自 Probing Notebook 的建议)
    # 过滤掉字典定义、逆向翻译等非标准翻译任务
    def is_auxiliary(row):
        src = str(row['transliteration']).lower()
        tgt = str(row['translation']).lower()
        
        # 关键词过滤
        markers = ['sumerogram', 'means:', 'akkadian word', 'logogram', 'definition', 
                   'syllabic spelling', 'determinative', 'ideogram', 'translate english to']
        if any(m in src or m in tgt for m in markers):
            return True
            
        # 过短句子过滤 (可能是字典词条)
        # 如果原文单词数少于 2 且译文单词数少于 3，且没有 Gap 标记，可能是噪声
        if len(src.split()) < 2 and len(tgt.split()) < 3 and '<gap>' not in src:
            return True
            
        return False
    
    # 标记辅助任务
    aux_mask = df.apply(is_auxiliary, axis=1)
    n_aux = aux_mask.sum()
    if n_aux > 0:
        print(f"   🧹 过滤辅助任务/噪声数据: {n_aux} 条")
        df = df[~aux_mask].copy()

    # 4. 非平行内容过滤（索引表、metadata、页码引用等）
    non_parallel_results = [
        is_likely_non_parallel(tl, tr)
        for tl, tr in zip(df['transliteration'], df['translation'])
    ]
    non_parallel_mask = pd.Series(
        [flag for flag, _ in non_parallel_results],
        index=df.index,
    )
    n_non_parallel = int(non_parallel_mask.sum())
    if n_non_parallel > 0:
        reason_counter = Counter(
            reason for flag, reason in non_parallel_results if flag and reason
        )
        detail = ", ".join(f"{k}={v}" for k, v in sorted(reason_counter.items()))
        print(f"   🪓 过滤非平行内容: {n_non_parallel} 条" + (f" ({detail})" if detail else ""))
        df = df[~non_parallel_mask].copy()
    
    # 删除临时列
    df = df.drop(columns=['_translit_len', '_transl_len', '_ratio'])
    
    removed = original_len - len(df)
    print(f"   过滤掉 {removed} 条质量较差的样本")
    
    return df


# ============================================================
# Silver 无标注数据生成
# ============================================================

def prepare_silver_unlabeled(train_ids: set):
    """
    从 published_texts.csv 提取无标注数据，应用与 train 完全一致的预处理
    
    Args:
        train_ids: train.csv 中已有的 oare_id 集合（用于排除）
    """
    if not os.path.exists(config.published_texts_csv):
        print(f"\n⚠️ published_texts.csv 不存在: {config.published_texts_csv}，跳过 silver 生成")
        return
    
    print(f"\n{'='*60}")
    print("🥈 生成 Silver 无标注数据")
    print(f"{'='*60}")
    
    pub_df = pd.read_csv(config.published_texts_csv, usecols=['oare_id', 'transliteration', 'genre_label'])
    print(f"   published_texts.csv: {len(pub_df)} 条")
    
    # 过滤：去掉 train 中已有的
    before = len(pub_df)
    pub_df = pub_df[~pub_df['oare_id'].isin(train_ids)]
    print(f"   去掉 train 已有: {before} → {len(pub_df)} 条 (排除 {before - len(pub_df)} 条)")
    
    # 过滤：去掉无 transliteration 的
    pub_df = pub_df[pub_df['transliteration'].notna() & (pub_df['transliteration'].str.strip().str.len() > 0)]
    print(f"   去掉空 transliteration: {len(pub_df)} 条")
    
    # 应用与 train 完全一致的预处理
    pub_df['transliteration'] = pub_df['transliteration'].apply(preprocess_transliteration)
    
    # 列名映射：genre_label → genre，添加空 translation 列
    pub_df = pub_df.rename(columns={'genre_label': 'genre'})
    pub_df['translation'] = ''
    
    # 输出列顺序与现有 silver_unlabeled.csv 一致
    pub_df = pub_df[['oare_id', 'transliteration', 'translation', 'genre']]
    
    pub_df.to_csv(config.output_silver, index=False, encoding='utf-8')
    print(f"\n✅ 保存 silver 数据: {config.output_silver}")
    print(f"   总计: {len(pub_df)} 条")
    
    # 体裁分布
    top_genres = pub_df['genre'].value_counts().head(5)
    print(f"   Top 体裁: {dict(top_genres)}")


# ============================================================
# 撇号恢复（官方改进版误删了撇号，从原始版恢复）
# ============================================================

def restore_apostrophes(train_df: pd.DataFrame, original_csv_path: str) -> pd.DataFrame:
    """从原始版 train.csv 恢复被官方改进版误删的撇号。
    
    使用 SequenceMatcher 逐词对齐，只 cherry-pick 撇号差异，
    保留官方改进版的其他修正（半截子、括号、gap等）。
    """
    if not os.path.exists(original_csv_path):
        print(f"   ⚠️ 原始版 CSV 不存在: {original_csv_path}，跳过撇号恢复")
        return train_df
    
    original_df = pd.read_csv(original_csv_path)
    orig_dict = original_df.set_index('oare_id')['translation'].to_dict()
    
    total_changes = 0
    changed_samples = 0
    
    def _is_apo_only_diff(w1, w2):
        return w1.replace("'", "") == w2.replace("'", "")
    
    def _restore_row(row):
        nonlocal total_changes, changed_samples
        oid = row['oare_id']
        off_tr = str(row['translation']).strip()
        
        if oid not in orig_dict:
            return off_tr
        
        orig_tr = str(orig_dict[oid]).strip()
        if "'" not in orig_tr or orig_tr == off_tr:
            return off_tr
        
        orig_words = orig_tr.split()
        off_words = off_tr.split()
        
        matcher = SequenceMatcher(
            None,
            [w.replace("'", "") for w in off_words],
            [w.replace("'", "") for w in orig_words],
        )
        
        result = list(off_words)
        changes = 0
        for op, i1, i2, j1, j2 in matcher.get_opcodes():
            if op == 'equal':
                for off_idx, orig_idx in zip(range(i1, i2), range(j1, j2)):
                    if result[off_idx] != orig_words[orig_idx] and \
                       _is_apo_only_diff(orig_words[orig_idx], result[off_idx]):
                        result[off_idx] = orig_words[orig_idx]
                        changes += 1
        
        if changes > 0:
            total_changes += changes
            changed_samples += 1
        
        return ' '.join(result)
    
    train_df = train_df.copy()
    train_df['translation'] = train_df.apply(_restore_row, axis=1)
    print(f"   撇号恢复: {changed_samples} 条样本, {total_changes} 处撇号")
    return train_df


# ============================================================
# 主流程
# ============================================================


def build_sliding_dataset(train_df: pd.DataFrame, config: Config) -> pd.DataFrame:
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] ⚙️ 开始生成滑动窗口数据集...")
    
    sliding_data = []
    short_docs = 0
    long_docs_split = 0
    total_chunks = 0
    
    # 预留 task prefix + EOS token 的字节数，防止 tokenizer 截断
    # ByT5: 每个 byte → 1 token, 加 1 个 EOS token
    prefix_bytes = len(config.task_prefix.encode('utf-8'))
    max_tl_bytes = 512 - prefix_bytes - 1  # 512 - 31(prefix) - 1(EOS) = 480
    
    for _, row in train_df.iterrows():
        oare_id = row['oare_id']
        transliteration = str(row['transliteration'])
        translation = str(row['translation'])
        
        transliteration = preprocess_transliteration(transliteration)
        translation = preprocess_translation(translation)
        
        doc_len = len(transliteration.encode('utf-8'))
        
        if doc_len <= max_tl_bytes:
            sliding_data.append({
                'oare_id': oare_id,
                'transliteration': transliteration,
                'translation': translation,
                'source': 'doc_short',
                'data_source': row.get('data_source', 'official'),
            })
            short_docs += 1
        else:
            chunks = sliding_window_split_doc(oare_id, translit=transliteration, translation=translation, max_bytes=max_tl_bytes, overlap_bytes=128, global_ratio=0.9713)
            if chunks:
                for chunk in chunks:
                    sent_translit = preprocess_transliteration(chunk['transliteration'])
                    sent_translation = preprocess_translation(chunk['translation'])
                    
                    if len(sent_translation.split()) < 2 or len(sent_translit.split()) < 1:
                        continue
                        
                    sliding_data.append({
                        'oare_id': chunk['oare_id'],
                        'transliteration': sent_translit,
                        'translation': sent_translation,
                        'source': chunk['source'],
                        'data_source': row.get('data_source', 'official'),
                    })
                long_docs_split += 1
                total_chunks += len(chunks)
            else:
                sliding_data.append({
                    'oare_id': oare_id,
                    'transliteration': transliteration,
                    'translation': translation,
                    'source': 'doc_long',
                    'data_source': row.get('data_source', 'official'),
                })
                long_docs_split += 1 # keeping as is due to failure
                
    print(f"\n📊 滑动窗口切分统计:")
    print(f"   短文档（<= {max_tl_bytes} 字节）: {short_docs}")
    print(f"   长文档（> {max_tl_bytes} 字节）: {long_docs_split}")
    print(f"   切分出片段总数: {total_chunks}")
    
    sliding_df = pd.DataFrame(sliding_data)
    
    sliding_df = crop_half_translations(sliding_df)
    sliding_df = quality_filter(sliding_df)
    
    sliding_df.to_csv(config.output_sliding, index=False, encoding='utf-8')
    print(f"\n✅ 保存滑动窗口数据集: {config.output_sliding}")
    print(f"   总计: {len(sliding_df)} 条 (对比 train_df: {len(train_df)} 条)")
    
    return sliding_df


def apply_review_overrides(train_df: pd.DataFrame, config: Config) -> Tuple[pd.DataFrame, int]:
    """应用人工审校的修订。"""
    if not os.path.exists(config.cr_review_csv):
        print(f"\nℹ️ 未找到人工审校文件，跳过合并: {config.cr_review_csv}")
        return train_df, 0

    review_df = pd.read_csv(config.cr_review_csv)
    required_cols = {'oare_id', 'transliteration', 'translation'}
    if not required_cols.issubset(set(review_df.columns)):
        missing = sorted(required_cols - set(review_df.columns))
        raise ValueError(f"人工审校文件缺少必要列: {missing}")

    print(f"\n🛠️ 应用人工审校数据: {config.cr_review_csv}")
    print(f"   审校文件行数: {len(review_df)}")

    base_df = train_df.copy()
    base_df['oare_id'] = base_df['oare_id'].astype(str)
    review_df['oare_id'] = review_df['oare_id'].astype(str)

    base_meta = base_df[['oare_id', 'data_source', 'uncertainty']].copy()
    review_df = review_df[review_df['oare_id'].isin(set(base_df['oare_id']))].copy()
    review_df = review_df.drop_duplicates(subset=['oare_id'], keep='first')
    print(f"   命中当前 clean 数据的行数: {len(review_df)}")

    if 'data_source' in review_df.columns:
        review_df = review_df.rename(columns={'data_source': 'review_data_source'})
    if 'uncertainty' in review_df.columns:
        review_df = review_df.rename(columns={'uncertainty': 'review_uncertainty'})

    review_df['transliteration'] = review_df['transliteration'].astype(str).apply(preprocess_transliteration)
    review_df['translation'] = review_df['translation'].astype(str).apply(preprocess_translation)
    review_df = review_df.merge(base_meta, on='oare_id', how='left')

    if 'review_data_source' not in review_df.columns:
        review_df['review_data_source'] = ''
    else:
        review_df['review_data_source'] = review_df['review_data_source'].fillna('')
    if 'review_uncertainty' not in review_df.columns:
        review_df['review_uncertainty'] = ''
    else:
        review_df['review_uncertainty'] = review_df['review_uncertainty'].fillna('')

    review_df['data_source'] = np.where(
        review_df['review_data_source'].astype(str).str.strip() == '',
        review_df['data_source'],
        review_df['review_data_source'],
    )
    review_df['data_source'] = review_df['data_source'].apply(normalize_data_source_label)

    review_df['uncertainty'] = np.where(
        review_df['review_uncertainty'].astype(str).str.strip() == '',
        review_df['uncertainty'].fillna(''),
        review_df['review_uncertainty'],
    )

    review_df = review_df[['oare_id', 'transliteration', 'translation', 'data_source', 'uncertainty']]

    compare_df = review_df.merge(
        base_df[['oare_id', 'transliteration', 'translation']],
        on='oare_id',
        how='left',
        suffixes=('', '_base'),
    )
    changed_mask = (
        compare_df['transliteration'] != compare_df['transliteration_base']
    ) | (
        compare_df['translation'] != compare_df['translation_base']
    )
    merged_rows_df = compare_df.loc[changed_mask, ['oare_id', 'transliteration', 'translation', 'data_source', 'uncertainty']].copy()
    merged_rows_df = merged_rows_df.reset_index(drop=True)
    print(f"   ✅ 有效修订: {len(merged_rows_df)} 条")

    if merged_rows_df.empty:
        print("   ℹ️ 没有发现相对基准 clean 的实际改动，跳过合并")
        return train_df, 0

    merged_df = base_df.set_index('oare_id').copy()
    merged_indexed = merged_rows_df.set_index('oare_id')
    merged_df.update(merged_indexed[['transliteration', 'translation', 'data_source', 'uncertainty']])
    merged_df = merged_df.reset_index()
    print(f"   🔄 已合并人工修订到 clean 数据: {len(merged_rows_df)} 条")

    return merged_df, len(merged_rows_df)


def prepare_clean_dataset(input_csv_path: str, config: Config, dataset_label: str) -> pd.DataFrame:
    """从指定输入 CSV 生成 clean 数据。"""
    _uncertainty = {}
    is_archibab = 'archibab' in str(input_csv_path).lower()

    print(f"\n📂 加载{dataset_label}数据...")
    train_df = pd.read_csv(input_csv_path)
    print(f"   {os.path.basename(input_csv_path)}: {len(train_df)} 条")

    # 1.1 撇号恢复（新版 data_final 已恢复撇号，跳过此步骤）
    if os.path.exists(config.train_csv_original_v1):
        train_df = restore_apostrophes(train_df, config.train_csv_original_v1)
    else:
        print("   ℹ️ 跳过撇号恢复（新数据已含撇号）")

    # 1.15 定点翻译错误修正（翻译者把 TL ⅔ GÍN 15 ŠE = ¾ shekel 误写为其他值）
    n_fixed = 0
    if 'translation' in train_df.columns:
        train_df = train_df.copy()
        fix_map = {
            "3 ⅔ shekel": "¾ shekel",
            "3 ⅔ shekels": "¾ shekel",
        }
        for old, new in fix_map.items():
            mask = train_df['translation'].astype(str).str.contains(old, regex=False, na=False)
            if mask.any():
                train_df.loc[mask, 'translation'] = train_df.loc[mask, 'translation'].astype(str).str.replace(old, new, regex=False)
                n_fixed += int(mask.sum())
    if n_fixed:
        print(f"   🔧 定点翻译修正: {n_fixed} 处 (grain 数值纠错)")

    # 1.2 从 data_v2 动态回填截断翻译
    if os.path.exists(config.train_csv_v2):
        v2_df = pd.read_csv(config.train_csv_v2)
        v2_map = dict(zip(v2_df['oare_id'].astype(str), v2_df['translation'].astype(str)))
        backfill_count = 0
        for idx in train_df.index:
            oare_id = str(train_df.loc[idx, 'oare_id'])
            final_tr = str(train_df.loc[idx, 'translation'])
            v2_tr = v2_map.get(oare_id, '')
            if not v2_tr or v2_tr == 'nan':
                continue
            if final_tr.endswith('...') and len(v2_tr) > len(final_tr):
                train_df.loc[idx, 'translation'] = v2_tr
                backfill_count += 1
                _uncertainty.setdefault(oare_id, set()).add('tr_backfilled')
            elif final_tr.endswith('<gap>') and len(v2_tr) > len(final_tr):
                train_df.loc[idx, 'translation'] = v2_tr
                backfill_count += 1
                _uncertainty.setdefault(oare_id, set()).add('tr_backfilled')
            elif (final_tr.count(' ') < 8 and len(v2_tr) > len(final_tr) + 40):
                train_df.loc[idx, 'translation'] = v2_tr
                backfill_count += 1
                _uncertainty.setdefault(oare_id, set()).add('tr_backfilled')
            elif oare_id in v2_map:
                def _loose_prefix_match(a, b):
                    aa = re.sub(r'\\s+', ' ', str(a)).strip()
                    bb = re.sub(r'\\s+', ' ', str(b)).strip()
                    if len(aa) < 12:
                        return False
                    k = min(len(aa), len(bb))
                    return bb[:k].startswith(aa[:max(8, min(k, len(aa)))]) and len(bb) > len(aa) + 20
                backfilled = False
                if final_tr.rstrip().endswith(('/', '-')):
                    train_df.loc[idx, 'translation'] = v2_tr
                    backfill_count += 1
                    backfilled = True
                elif _loose_prefix_match(final_tr, v2_tr):
                    train_df.loc[idx, 'translation'] = v2_tr
                    backfill_count += 1
                    backfilled = True
                if backfilled:
                    _uncertainty.setdefault(oare_id, set()).add('tr_backfilled')
        if backfill_count > 0:
            print(f"   🔄 从 data_v2 动态回填截断翻译: {backfill_count} 条")
    else:
        print("   ℹ️ 跳过 v2 翻译回填（data_v2 不存在）")

    # 1.25 从 v2 恢复被截断的 TL
    if os.path.exists(config.train_csv_tl_source):
        tl_src_df = pd.read_csv(config.train_csv_tl_source)
        tl_src_by_oare = dict(zip(tl_src_df['oare_id'].astype(str), tl_src_df['transliteration'].astype(str)))
        tl_restore_count = 0
        for idx in train_df.index:
            oare = str(train_df.loc[idx, 'oare_id'])
            cur_tl = str(train_df.loc[idx, 'transliteration'])
            src_tl = tl_src_by_oare.get(oare, '')
            if len(src_tl) > len(cur_tl) + 100:
                train_df.loc[idx, 'transliteration'] = src_tl
                tl_restore_count += 1
                _uncertainty.setdefault(oare, set()).add('tl_restored')
        if tl_restore_count:
            print(f"   🔄 从 v2 恢复截断 TL: {tl_restore_count} 条")
        del tl_src_df, tl_src_by_oare
    else:
        print("   ℹ️ 跳过 TL 恢复（v2 数据不存在）")

    # 1.3 原始 ratio 预过滤
    raw_tl_wc = train_df['transliteration'].str.split().str.len()
    raw_tr_wc = train_df['translation'].str.split().str.len()
    raw_ratio = raw_tl_wc / raw_tr_wc.clip(lower=1)
    raw_bad = raw_ratio > config.max_raw_word_ratio
    if raw_bad.any():
        print(f"   🚫 原始 ratio>{config.max_raw_word_ratio} 预过滤: {raw_bad.sum()} 条")
        train_df = train_df[~raw_bad].copy()

    if 'data_source' not in train_df.columns:
        train_df['data_source'] = 'official'
    train_df['data_source'] = train_df['data_source'].apply(normalize_data_source_label)

    # 2. 预处理
    print("\n🔧 预处理 data_final...")
    if is_archibab:
        train_df['transliteration'] = train_df['transliteration'].apply(strip_archibab_transliteration_line_prefixes)
    train_df['transliteration'] = train_df['transliteration'].apply(preprocess_transliteration)
    train_df['translation'] = train_df['translation'].apply(preprocess_translation)
    if is_archibab:
        train_df['translation'] = train_df['translation'].apply(clean_archibab_translation_residue)

    # 2.8 启发式尾部裁剪
    train_df = crop_half_translations(train_df, _uncertainty=_uncertainty)
    if is_archibab:
        train_df = filter_archibab_fragments(train_df)
        train_df['translation'] = train_df['translation'].apply(clean_archibab_translation_residue)

    # 3. 质量过滤
    print("\n🔍 质量过滤...")
    train_df = quality_filter(train_df)
    if is_archibab:
        train_df['translation'] = train_df['translation'].apply(clean_archibab_translation_residue)
    print(f"   清洗后: {len(train_df)} 条")

    # 4. uncertainty + 审校覆盖
    train_df['uncertainty'] = train_df['oare_id'].apply(
        lambda oid: '|'.join(sorted(_uncertainty.get(oid, set()))) if oid in _uncertainty else ''
    )
    train_df, _ = apply_review_overrides(train_df, config)
    return train_df


def main():
    print("=" * 60)
    print("数据预处理脚本")
    print("=" * 60)

    train_df = prepare_clean_dataset(config.train_csv, config, dataset_label="官方")
    train_df.to_csv(config.output_clean, index=False, encoding='utf-8')
    print(f"\n✅ 保存官方-only clean 数据: {config.output_clean}")
    print(f"   总计: {len(train_df)} 条")

    if os.path.exists(config.train_csv_ocr_merged):
        merged_clean_df = prepare_clean_dataset(
            config.train_csv_ocr_merged,
            config,
            dataset_label="official+ocr merged",
        )
        merged_clean_df.to_csv(config.output_clean_ocr_merged, index=False, encoding='utf-8')
        print(f"\n✅ 保存 merged clean 数据: {config.output_clean_ocr_merged}")
        print(f"   总计: {len(merged_clean_df)} 条")
    else:
        print(f"\nℹ️ 未找到 merged 原始数据，跳过 merged clean 输出: {config.train_csv_ocr_merged}")

    # 6. 文档级统计（句子切分已停用）
    print("\n📝 句子切分已停用，当前输出为 train_clean.csv、train_clean_ocr_merged.csv 与 train_sliding.csv ...")

    short_docs = 0
    long_docs_kept = 0

    for _, row in train_df.iterrows():
        doc_len = len(row['transliteration'])
        source = 'doc_short' if doc_len <= config.max_doc_chars_for_split else 'doc_long'
        if source == 'doc_short':
            short_docs += 1
        else:
            long_docs_kept += 1

    print(f"\n📊 文档统计:")
    print(f"   短文档（<= {config.max_doc_chars_for_split} 字符）: {short_docs}")
    print(f"   长文档保留原样: {long_docs_kept}")

    # 清理已废弃输出，避免后续误用旧文件
    deprecated_outputs = [
        os.path.join(config.data_dir, "train_sentences.csv"),
        os.path.join(config.data_dir, "train_mixed.csv"),
        os.path.join(config.data_dir, "train_cr.csv"),
    ]
    for deprecated_path in deprecated_outputs:
        if os.path.exists(deprecated_path):
            os.remove(deprecated_path)
            print(f"   🧹 删除废弃输出: {deprecated_path}")

    stats_df = train_df.copy()
    stats_df['source'] = stats_df['transliteration'].apply(
        lambda x: 'doc_short' if len(str(x)) <= config.max_doc_chars_for_split else 'doc_long'
    )
    
    # 7.5 生成校准词典（用训练数据 TL↔TR 对校准 eBL 词典拼写）
    if os.path.exists(config.lexicon_csv):
        print("\n📚 生成校准词典...")
        _raw_lexicon = load_lexicon(config.lexicon_csv)
        cal_lex_path = os.path.join(config.data_dir, "calibrated_lexicon.json")
        build_calibrated_lexicon(stats_df, _raw_lexicon, output_path=cal_lex_path)
    else:
        print("\n⚠️ 跳过校准词典生成（词典文件不存在）")
    
    # 8. 统计信息
    print("\n📊 最终数据分布:")
    for source in stats_df['source'].unique():
        subset = stats_df[stats_df['source'] == source]
        lengths = subset['transliteration'].str.len()
        print(f"   {source}: {len(subset)} 条, 均值长度={lengths.mean():.0f}, 中位数={lengths.median():.0f}")
    
    # 9. 词数比例统计（阿卡德语 / 英语）
    print("\n📊 词数比例统计（阿卡德语词数 / 英语词数）:")
    stats_df['_akk_words'] = stats_df['transliteration'].apply(lambda x: len(str(x).split()))
    stats_df['_eng_words'] = stats_df['translation'].apply(lambda x: len(str(x).split()))
    stats_df['_word_ratio'] = stats_df['_akk_words'] / (stats_df['_eng_words'] + 1)
    
    print(f"   均值:   {stats_df['_word_ratio'].mean():.3f}")
    print(f"   中位数: {stats_df['_word_ratio'].median():.3f}")
    print(f"   最小值: {stats_df['_word_ratio'].min():.3f}")
    print(f"   最大值: {stats_df['_word_ratio'].max():.3f}")
    print(f"   标准差: {stats_df['_word_ratio'].std():.3f}")
    
    # 按来源分组
    print("\n   按来源分组:")
    for source in stats_df['source'].unique():
        subset = stats_df[stats_df['source'] == source]
        print(f"   {source}: 均值={subset['_word_ratio'].mean():.3f}, 中位数={subset['_word_ratio'].median():.3f}")
    
    # 10. N-gram 统计（需要在绘图前计算）
    print("\n📊 Translation N-gram 重复率统计:")
    ngram_stats = compute_ngram_stats(stats_df['translation'].tolist(), n_values=[2, 3, 4, 5, 6, 8])
    print(f"   {'n-gram':<8} {'均值':<10} {'中位数':<10} {'最大值':<10} {'有重复样本'}")
    print(f"   {'-'*50}")
    for n, stat in ngram_stats.items():
        print(f"   {n}-gram    {stat['mean']:.4f}     {stat['median']:.4f}     {stat['max']:.4f}     {stat['samples_with_repeat']}")
    
    # 11. 绘制统计图
    print("\n📈 生成统计图...")
    os.makedirs(config.output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Data Statistics', fontsize=14, fontweight='bold')
    
    # 图1: 数据来源分布
    ax1 = axes[0, 0]
    source_counts = stats_df['source'].value_counts()
    ax1.bar(source_counts.index, source_counts.values, color=['#4CAF50', '#2196F3', '#FF9800'])
    ax1.set_title('Data Source Distribution')
    ax1.set_xlabel('Source')
    ax1.set_ylabel('Count')
    for i, v in enumerate(source_counts.values):
        ax1.text(i, v + 10, str(v), ha='center', fontweight='bold')
    
    # 图2: Transliteration 长度分布
    ax2 = axes[0, 1]
    for source in stats_df['source'].unique():
        subset = stats_df[stats_df['source'] == source]
        ax2.hist(subset['transliteration'].str.len(), bins=50, alpha=0.6, label=source)
    ax2.set_title('Transliteration Length Distribution')
    ax2.set_xlabel('Length (chars)')
    ax2.set_ylabel('Count')
    ax2.legend()
    ax2.set_xlim(0, 1500)
    
    # 图3: Translation 长度分布（新增）
    ax3_new = axes[0, 2]
    for source in stats_df['source'].unique():
        subset = stats_df[stats_df['source'] == source]
        ax3_new.hist(subset['translation'].str.len(), bins=50, alpha=0.6, label=source)
    ax3_new.set_title('Translation Length Distribution')
    ax3_new.set_xlabel('Length (chars)')
    ax3_new.set_ylabel('Count')
    ax3_new.legend()
    ax3_new.set_xlim(0, 2000)
    
    # 图3: 词数比例分布
    ax3 = axes[1, 0]
    ax3.hist(stats_df['_word_ratio'], bins=50, color='#9C27B0', alpha=0.7, edgecolor='white')
    ratio_median = stats_df['_word_ratio'].median()
    ax3.axvline(ratio_median, color='red', linestyle='--', label=f'Median: {ratio_median:.2f}')
    ax3.set_title('Akkadian/English Word Ratio Distribution')
    ax3.set_xlabel('Ratio (Akkadian words / English words)')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.set_xlim(0, 2)
    
    # 图4: N-gram 重复率
    ax4 = axes[1, 1]
    n_values = list(ngram_stats.keys())
    means = [ngram_stats[n]['mean'] for n in n_values]
    samples = [ngram_stats[n]['samples_with_repeat'] for n in n_values]
    
    ax4_twin = ax4.twinx()
    bars = ax4.bar([f'{n}-gram' for n in n_values], means, color='#00BCD4', alpha=0.7, label='Mean Repeat Rate')
    line = ax4_twin.plot([f'{n}-gram' for n in n_values], samples, 'ro-', label='Samples with Repeat')
    ax4.set_title('N-gram Repeat Statistics')
    ax4.set_xlabel('N-gram Size')
    ax4.set_ylabel('Mean Repeat Rate', color='#00BCD4')
    ax4_twin.set_ylabel('Samples with Repeat', color='red')
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    # 图5: 阿卡德语词频分布（Top 30）
    ax5 = axes[1, 2]
    all_akk_words = []
    for text in train_df['transliteration']:
        if isinstance(text, str):
            all_akk_words.extend(text.split())
    word_freq = Counter(all_akk_words).most_common(30)
    words, freqs = zip(*word_freq) if word_freq else ([], [])
    ax5.barh(range(len(words)), freqs, color='#E91E63', alpha=0.7)
    ax5.set_yticks(range(len(words)))
    ax5.set_yticklabels(words, fontsize=8)
    ax5.invert_yaxis()
    ax5.set_title('Top 30 Akkadian Words')
    ax5.set_xlabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(config.stats_plot, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   保存统计图: {config.stats_plot}")
    
    # 清理临时列
    # 12. 生成 Silver 无标注数据（从 published_texts.csv 提取）
    prepare_silver_unlabeled(set(train_df['oare_id']))
    
    sliding_df = build_sliding_dataset(train_df, config)
    
    print("\n" + "=" * 60)
    print("数据预处理完成")
    print("=" * 60)
    
    return train_df


if __name__ == "__main__":
    import argparse
    _parser = argparse.ArgumentParser(description="数据预处理脚本")
    _parser.add_argument("--input", type=str, default=None,
                         help="覆盖 Config.train_csv 的输入文件路径")
    _args = _parser.parse_args()
    if _args.input:
        config.train_csv = _args.input
        # 自动推导输出路径：与输入同目录，文件名加 _clean 后缀
        _stem = os.path.splitext(os.path.basename(_args.input))[0]
        config.output_clean = os.path.join(os.path.dirname(_args.input), f"{_stem}_clean.csv")
        print(f"📂 CLI 覆盖输入: {config.train_csv}")
        print(f"📂 自动输出路径: {config.output_clean}")
    main()
