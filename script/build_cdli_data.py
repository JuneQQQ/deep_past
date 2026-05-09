#!/usr/bin/env python3
"""
CDLI 数据构建脚本（两阶段）

阶段 1 — 源头解析：
  从 cdliatf_unblocked.atf 解析全部带英文翻译的文本，
  去除 ATF 行号前缀，按语言分拆为 raw CSV：
    data/cdli/cdli_raw_oa.csv      (Akkadian)
    data/cdli/cdli_raw_sux.csv     (Sumerian)
    data/cdli/cdli_raw_elx.csv     (Elamite)
    data/cdli/cdli_raw_qpc.csv     (Proto-Cuneiform)
    ...

阶段 2 — 定制清洗（仅 oa / sux）：
  调用 prepare_data.py 中的 preprocess + quality_filter，
  输出：
    data/cdli/cdli_parallel_clean_oa.csv
    data/cdli/cdli_parallel_clean_sux.csv
    data/cdli/cdli_parallel_clean_all.csv   (oa + sux 合并，保留全部元数据)
"""

import re
import os
import sys
import pandas as pd
from pathlib import Path

import subprocess

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from prepare_data import preprocess_transliteration

# ── 路径 ──────────────────────────────────────────────
ATF_FILE = "/data/lsb/deep_past/data/cdliatf_unblocked.atf"
CATALOGUE_FILE = "/data/lsb/deep_past/data/cdli_cat.csv"
OUT_DIR = "/data/lsb/deep_past/data/cdli"

# ── 语言映射 ──────────────────────────────────────────
LANG_SUFFIX = {
    "Akkadian":            "oa",
    "Sumerian":            "sux",
    "Elamite":             "elx",
    "Proto-Cuneiform":     "qpc",
    "Hittite":             "hit",
    "Urartian":            "ura",
    "Hurrian":             "xur",
    "Akkadian-Sumerian":   "oa_sux",
    "Egyptian":            "egy",
}

BLOCKED_GENRES = [
    "lexical", "mathematical", "school", "exercise",
    "fake", "forgery", "seal", "bulla", "tag",
    "royal/monumental", "literary", "prayer/incantation",
    "omen", "private/votive",
]

# ── 方言分类（针对 Akkadian 大类的细分）──────────────
# period 字段 → dialect 标签
def classify_akkadian_dialect(period: str) -> str:
    """根据 CDLI period 字段将阿卡德语文本分类为具体方言"""
    p = str(period).lower()
    if "old assyrian" in p:
        return "OA"
    if "old babylonian" in p or "early old babylonian" in p:
        return "OB"
    if "old akkadian" in p:
        return "OAkk"
    if "neo-babylonian" in p or "achaemenid" in p or "early neo-babylonian" in p:
        return "NB"
    if "neo-assyrian" in p:
        return "NA"
    if "middle babylonian" in p:
        return "MB"
    if "middle assyrian" in p:
        return "MA"
    if "ur iii" in p:
        return "UrIII"
    if "ed iii" in p or "ed i" in p:
        return "ED"
    return "other"


# 方言 → (后缀, 方言标签, 是否清洗输出)
DIALECT_CONFIG = {
    "OA":    ("akk_oa",   "OA",      True),
    "OB":    ("akk_ob",   "OB",      True),
    "OAkk":  ("akk_oakk", "OAkk",    True),
    "NA":    ("akk_na",   "NA",      False),
    "NB":    ("akk_nb",   "NB",      False),
    "MA":    ("akk_ma",   "MA",      False),
    "MB":    ("akk_mb",   "MB",      False),
    "UrIII": ("akk_ur3",  "UrIII",   False),
    "ED":    ("akk_ed",   "ED",      False),
    "other": ("akk_misc", "other",   False),
}


# ── 工具函数 ──────────────────────────────────────────
def map_lang(raw: str) -> str:
    raw = str(raw).lower()
    if "akk" in raw and "sux" in raw:
        return "Akkadian-Sumerian"
    if "akk" in raw:
        return "Akkadian"
    if "sux" in raw:
        return "Sumerian"
    if "qpc" in raw:
        return "Proto-Cuneiform"
    if "elx" in raw:
        return "Elamite"
    if "hit" in raw:
        return "Hittite"
    if "urartian" in raw:
        return "Urartian"
    if "xur" in raw:
        return "Hurrian"
    if "egy" in raw:
        return "Egyptian"
    return raw


def is_genre_allowed(genre: str) -> bool:
    genre = str(genre).lower()
    return not any(b in genre for b in BLOCKED_GENRES)


def strip_atf_line_prefix(line: str) -> str:
    """去除 CDLI ATF 行号前缀，如 '1.', '1\\'.' , '1.a1.', '10.'"""
    return re.sub(r"^[0-9]+[a-zA-Z0-9\.\'\á\š]*[\.\']\s*", "", line).strip()


def normalize_cdli_transliteration(text: str) -> str:
    """将 CDLI ATF 格式特有的编码转换为训练数据使用的标准格式。
    
    在 preprocess_transliteration 之前执行（处理 ATF 源头编码差异）。
    """
    if not isinstance(text, str):
        return ""

    # 1. Ndiš → N（数字标记：1diš→1, 8diš→8, 12diš→12）
    text = re.sub(r'\b(\d+)diš\b', r'\1', text)

    # 1.1 分数+diš/aš → 分数（½diš→½, ⅔aš→⅔）
    text = re.sub(r'([½⅓⅔⅙⅚¼])\s*(?:diš|aš)\b', r'\1', text)

    # 2. N{ú} → N0（十位标记：1{ú}→10, 3{ú}→30）
    def _expand_u(m):
        return str(int(m.group(1)) * 10)
    text = re.sub(r'\b(\d+)\{ú\}', _expand_u, text)

    # 2.5 合并十位+个位（ATF 将 16 编码为 "10 6"，需要合并回 "16"）
    def _merge_tens_units(m):
        return str(int(m.group(1)) + int(m.group(2)))
    text = re.sub(r'\b(10|20|30|40|50) (\d)\b', _merge_tens_units, text)

    # 3. gin2 → GÍN（shekel 苏美尔表意字）
    text = re.sub(r'\bgin2\b', 'GÍN', text)

    # 4. šunigin / šu-nigin → ŠU.NÍGIN（总计）
    text = re.sub(r'\bšunigin\b', 'ŠU.NÍGIN', text, flags=re.IGNORECASE)
    text = re.sub(r'\bšu-nigin\b', 'ŠU.NÍGIN', text, flags=re.IGNORECASE)

    # 5. 小写苏美尔表意字 → 大写（仅独立词，不破坏阿卡德语音节）
    _SUMEROGRAM_MAP = {
        'dumu': 'DUMU',
        'lugal': 'LUGAL',
        'igi': 'IGI',
        'kišib': 'KIŠIB',
        'kišib3': 'KIŠIB',
        'ensí': 'ENSÍ',
        'ensi2': 'ENSÍ',
        'iti': 'ITI',
        'mu': 'MU',       # year name marker - only when standalone
    }
    for lc, uc in _SUMEROGRAM_MAP.items():
        # 只替换独立词（前后有空格或字符串边界），不破坏连字符词
        text = re.sub(rf'(?<![a-záàéèíìúùšṣṭḫ-])\b{re.escape(lc)}\b(?![a-záàéèíìúùšṣṭḫ-])', uc, text)

    # 5.5 去除 {diš} 人名限定词（REF 数据中不存在，CDLI 特有）
    text = re.sub(r'\{diš\}', '', text)

    # 6. 修复 stray space-hyphen（"ga -ki" → "ga-ki", "mu --<gap>" → "mu-<gap>"）
    text = re.sub(r'(\S) +--+', r'\1-', text)  # word --<gap> → word-<gap>
    text = re.sub(r'(\S) +-(\S)', r'\1-\2', text)

    # 7. 修复 hyphen-space（"a- na" → "a-na"）
    text = re.sub(r'(\S)- +(\S)', r'\1-\2', text)

    # 8. 压缩多余空格
    text = re.sub(r'  +', ' ', text).strip()

    return text


def normalize_cdli_translation(text: str) -> str:
    """CDLI 翻译特有的规范化（在 prepare_data.py 之前执行）。"""
    if not isinstance(text, str):
        return ""

    # 1. 修复断裂变音符：CDLI 用逗号表示下点符
    #    S,illī → Ṣillī, T,uppi → Ṭuppi, s,almu → ṣalmu, t,uppu → ṭuppu
    text = re.sub(r'S,(?=[a-záàéèíìúùāēīū])', 'Ṣ', text)
    text = re.sub(r'T,(?=[a-záàéèíìúùāēīū])', 'Ṭ', text)
    text = re.sub(r's,(?=[a-záàéèíìúùāēīū])', 'ṣ', text)
    text = re.sub(r't,(?=[a-záàéèíìúùāēīū])', 'ṭ', text)

    # 1.5 去除 TR 中的下划线（CDLI 偶尔用 _ 作词分隔符）
    text = text.replace('_', ' ')

    # 1.6 去除 'lit.' 学术直译注释（"sell him lit. give him for silver" → "sell him"）
    # 模式：lit. 后跟直到下一个分号/句号/逗号+大写字母的内容
    text = re.sub(r'\s*\blit\.\s*[^;.]*(?=[;.]|,\s*[A-Z]|$)', '', text)

    # 2. 处理学术编辑括号（区分旁白 vs 补全）
    # 2a. 旁白/metadata：含 ? 或学术术语的括号 → 整块删除
    text = re.sub(r'\([^)]*\?[^)]*\)', '', text)  # 含问号的学术猜测
    text = re.sub(r'\((?:i\.e\.|seal impression|lines? \d|broken|colophon|erasure|uninscribed)[^)]*\)', '', text, flags=re.IGNORECASE)
    # 2b. 补全/隐性补全：剥去括号，保留文字
    text = re.sub(r'\(([^)]{1,80})\)', r'\1', text)

    # 3. 修复双句点（".." → "."）
    text = re.sub(r'\.\.+', '.', text)

    # 4. 压缩多余空格
    text = re.sub(r'  +', ' ', text).strip()

    return text


def is_bad_boilerplate(eng: str) -> bool:
    eng = str(eng).lower().strip()
    return eng in {
        "basket-of-tablets: <gap>",
        "...",
        "<gap>",
        "basket-of-tablets:",
        "basket of tablets",
    }


# ══════════════════════════════════════════════════════
# 阶段 1：源头解析 → 按语言分拆 raw CSV
# ══════════════════════════════════════════════════════
def stage1_parse_and_split():
    print("=" * 60)
    print("阶段 1：解析 ATF 并按语言分拆 raw CSV")
    print("=" * 60)

    # 加载 catalogue 元数据
    print("  加载 catalogue 元数据 ...")
    df_cat = pd.read_csv(CATALOGUE_FILE, low_memory=False)
    df_cat["p_number"] = "P" + df_cat["id_text"].astype(str).str.zfill(6)
    meta_dict = df_cat.set_index("p_number")[["genre", "period"]].to_dict("index")

    # 逐行解析 ATF
    print("  解析 ATF 文件（去除行号前缀）...")
    current_p = None
    atf_dict: dict = {}

    with open(ATF_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("&P"):
                m = re.match(r"&(P\d+)", line)
                if m:
                    current_p = m.group(1)
                    atf_dict[current_p] = {
                        "tl": [],
                        "tr": [],
                        "lang_raw": "",
                    }
            elif current_p:
                if line.startswith("#atf: lang "):
                    atf_dict[current_p]["lang_raw"] = line.split("lang ")[1].strip()
                elif line.startswith("#tr.en:"):
                    atf_dict[current_p]["tr"].append(line[7:].strip())
                elif re.match(r"^\d+[\.\']", line):
                    clean = strip_atf_line_prefix(line)
                    if clean:
                        atf_dict[current_p]["tl"].append(clean)

    # 组装记录
    records = []
    for pid, d in atf_dict.items():
        if not d["tl"] or not d["tr"]:
            continue
        meta = meta_dict.get(pid, {})
        genre = str(meta.get("genre", ""))
        if not is_genre_allowed(genre):
            continue
        records.append(
            {
                "id": pid,
                "period": meta.get("period", ""),
                "genre": genre,
                "language": map_lang(d["lang_raw"]),
                "transliteration": " ".join(d["tl"]),
                "translation": " ".join(d["tr"]),
            }
        )

    df_all = pd.DataFrame(records)
    print(f"  共提取 {len(df_all)} 条带翻译的文本")

    # 按语言分拆输出 raw CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    for lang, grp in df_all.groupby("language"):
        if lang == "Akkadian":
            # 阿卡德语大类：按方言细分
            grp = grp.copy()
            grp["dialect"] = grp["period"].apply(classify_akkadian_dialect)
            print(f"\n  阿卡德语方言分布:")
            for dialect, dgrp in grp.groupby("dialect"):
                suffix, _, _ = DIALECT_CONFIG.get(dialect, ("akk_misc", "", False))
                out = os.path.join(OUT_DIR, f"cdli_raw_{suffix}.csv")
                dgrp.to_csv(out, index=False)
                print(f"    cdli_raw_{suffix}.csv  →  {len(dgrp)} 条  ({dialect})")
            # 同时输出一份完整的 oa raw（向后兼容）
            out_all_akk = os.path.join(OUT_DIR, "cdli_raw_oa.csv")
            grp.to_csv(out_all_akk, index=False)
            print(f"    cdli_raw_oa.csv  →  {len(grp)} 条  (Akkadian, all dialects)")
        else:
            suffix = LANG_SUFFIX.get(lang, lang.lower().replace(" ", "_"))
            out = os.path.join(OUT_DIR, f"cdli_raw_{suffix}.csv")
            grp.to_csv(out, index=False)
            print(f"    cdli_raw_{suffix}.csv  →  {len(grp)} 条  ({lang})")

    return df_all


# ══════════════════════════════════════════════════════
# 阶段 2：CDLI 定制化清洗 → prepare_data.py 自动化流水线
# ══════════════════════════════════════════════════════
PREPARE_DATA_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prepare_data.py")


def _cdli_normalize_and_export(raw_csv: str, dialect_label: str) -> str:
    """对 raw CSV 做 CDLI 特有预处理，输出中间 CSV 供 prepare_data.py 消费。
    
    返回中间 CSV 路径。
    """
    print(f"\n  [CDLI 预处理] {dialect_label} ({os.path.basename(raw_csv)}) ...")
    df = pd.read_csv(raw_csv)
    n0 = len(df)

    df = df.dropna(subset=["transliteration", "translation"]).copy()

    # 1) preprocess_transliteration（ATF raw → 标准音译字符）
    df["transliteration"] = df["transliteration"].astype(str).apply(preprocess_transliteration)
    # 2) CDLI 特有规范化（diš→数字、gin2→GÍN、苏美尔表意字大写、空格修复等）
    df["transliteration"] = df["transliteration"].apply(normalize_cdli_transliteration)
    # 3) CDLI 翻译规范化（断裂变音符、编辑括号等）
    df["translation"] = df["translation"].astype(str).apply(normalize_cdli_translation)
    # 4) 过滤 CDLI 特有 boilerplate
    df = df[~df["translation"].apply(is_bad_boilerplate)]
    df = df[(df["transliteration"] != "") & (df["translation"] != "")]

    # 列名对齐 prepare_data.py 期望的格式
    if "id" in df.columns:
        df = df.rename(columns={"id": "oare_id"})

    # 添加 dialect 和 data_source 元数据
    df["dialect"] = dialect_label
    df["data_source"] = "cdli"

    # 输出中间 CSV（与 raw 同目录，加 _normalized 后缀）
    stem = os.path.splitext(os.path.basename(raw_csv))[0]
    mid_csv = os.path.join(OUT_DIR, f"{stem}_normalized.csv")
    df.to_csv(mid_csv, index=False)
    print(f"    {n0} → {len(df)} 条  →  {mid_csv}")
    return mid_csv


def _run_prepare_data(mid_csv: str) -> str:
    """调用 prepare_data.py --input 对中间 CSV 执行完整清洗，返回 _clean.csv 路径。"""
    print(f"  [prepare_data.py] 清洗 {os.path.basename(mid_csv)} ...")
    cmd = [sys.executable, PREPARE_DATA_SCRIPT, "--input", mid_csv]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    ❌ prepare_data.py 失败:\n{result.stderr[-500:]}")
        return ""
    # 推导输出路径
    stem = os.path.splitext(os.path.basename(mid_csv))[0]
    clean_csv = os.path.join(os.path.dirname(mid_csv), f"{stem}_clean.csv")
    if os.path.exists(clean_csv):
        df = pd.read_csv(clean_csv)
        print(f"    → {len(df)} 条  →  {clean_csv}")
        return clean_csv
    print(f"    ⚠️ 清洗输出未找到: {clean_csv}")
    return ""


def main():
    # 阶段 1
    stage1_parse_and_split()

    # 阶段 2
    print("\n" + "=" * 60)
    print("阶段 2：自动化清洗流水线（CDLI normalize → prepare_data.py）")
    print("=" * 60)

    clean_csvs = []

    # 清洗阿卡德语方言（OA / OB / OAkk）
    for dialect, (suffix, label, should_clean) in DIALECT_CONFIG.items():
        raw_csv = os.path.join(OUT_DIR, f"cdli_raw_{suffix}.csv")
        if should_clean and os.path.exists(raw_csv):
            mid = _cdli_normalize_and_export(raw_csv, label)
            out = _run_prepare_data(mid)
            if out:
                clean_csvs.append((out, label))

    # 清洗苏美尔语
    sux_raw = os.path.join(OUT_DIR, "cdli_raw_sux.csv")
    if os.path.exists(sux_raw):
        mid = _cdli_normalize_and_export(sux_raw, "Sumerian")
        out = _run_prepare_data(mid)
        if out:
            clean_csvs.append((out, "Sumerian"))  # Sumerian 保持全称

    # 合并所有 _clean.csv → cdli_clean.csv
    dfs = []
    for csv_path, dialect_label in clean_csvs:
        df = pd.read_csv(csv_path)
        df["dialect"] = dialect_label
        df["data_source"] = "cdli"
        # 移除 uncertainty 列
        if "uncertainty" in df.columns:
            df = df.drop(columns=["uncertainty"])
        dfs.append(df)

    final_csv = os.path.join(OUT_DIR, "cdli_clean.csv")
    df_all = pd.concat(dfs, ignore_index=True)

    # 去重：完全相同的 TL+TR 对只保留第一条
    n_before = len(df_all)
    df_all = df_all.drop_duplicates(subset=["transliteration", "translation"], keep="first")
    n_dedup = n_before - len(df_all)
    if n_dedup:
        print(f"\n  去重: 移除 {n_dedup} 条完全重复的 TL+TR 对")

    # 后处理：修复 prepare_data.py 可能引入的双句点
    df_all["translation"] = df_all["translation"].astype(str).apply(
        lambda t: re.sub(r'\.\.+', '.', t)
    )

    # 后处理：PN 占位符 → <gap>
    df_all["translation"] = df_all["translation"].apply(
        lambda t: re.sub(r'\bPN\d?\b', '<gap>', str(t))
    )

    # 后处理：去除 TR 中残留的裸引号对（"word" → word）
    df_all["translation"] = df_all["translation"].apply(
        lambda t: re.sub(r'"([^"]{1,30})"', r'\1', str(t))
    )

    # 后处理：常见英文 typo
    for wrong, right in [('adjacent ot ', 'adjacent to '), (' teh ', ' the ')]:
        df_all["translation"] = df_all["translation"].str.replace(wrong, right, regex=False)

    # 后处理：短法语/德语标签黑名单 → 英文等价标签
    _FOREIGN_LABEL_REPLACEMENTS = [
        (r'\bMonat\s*:', 'Month:'),
        (r'\bJahr\s*:', 'Year:'),
        (r'\bZeugen?\s*:', 'Witnesses:'),
        (r'\bZeuge\s*:', 'Witness:'),
        (r'\bT[ée]moins?\s+et\s+date\b', 'Witnesses and date'),
        (r'\bT[ée]moins?\s*:', 'Witnesses:'),
        (r'\bTemoins\s+et\s+date\b', 'Witnesses and date'),
        (r'\bTemoins\s*:', 'Witnesses:'),
        (r'\breçu\s+par\b', 'received by'),
        (r'\bDis\s+à\b', 'Speak to'),
        (r'\bainsi\s+\(?parle\)?\b', 'thus says'),
        (r'\bfils\s+de\b', 'son of'),
        (r'\bfille\s+de\b', 'daughter of'),
    ]
    for pattern, repl in _FOREIGN_LABEL_REPLACEMENTS:
        df_all["translation"] = df_all["translation"].str.replace(pattern, repl, regex=True)

    # 后处理：移除高残损低价值 OA 记录
    def _is_high_damage_oa(row):
        if row.get("dialect") != "OA":
            return False
        tr = str(row["translation"])
        if len(tr) < 60 and tr.count("<gap>") >= 1:
            return True
        words = tr.split()
        frag = sum(1 for w in words if w.endswith(".") and len(w) <= 3)
        return len(words) > 0 and frag / len(words) > 0.3

    dmg_mask = df_all.apply(_is_high_damage_oa, axis=1)
    if dmg_mask.any():
        df_all = df_all[~dmg_mask]
        print(f"\n  移除高残损 OA: {dmg_mask.sum()} 条")

    # 后处理：移除非英语翻译（德语、法语等）
    _NON_ENG_MARKERS = re.compile(
        r'\b(?:Sekel|Silber|Gerste|Sohn|Gattin|Haus|Jahr|Monat|Schmied|bezahlt'
        r'|Zeuge|Zeugen|T[ée]moins?|Temoins?|Année|reçu|fils de|fille de'
        r'|le roi|la ville|dans|avec|pour)\b',
        re.I
    )
    non_eng_mask = df_all["translation"].apply(
        lambda t: len(_NON_ENG_MARKERS.findall(str(t))) >= 2
    )
    n_non_eng = non_eng_mask.sum()
    if n_non_eng:
        df_all = df_all[~non_eng_mask]
        print(f"\n  移除非英语翻译: {n_non_eng} 条")

    keep = ["oare_id", "transliteration", "translation", "data_source", "dialect"]
    df_all = df_all[[c for c in keep if c in df_all.columns]]

    df_all.to_csv(final_csv, index=False)

    # 清理中间文件（_normalized.csv、各子集 _clean.csv），只保留 raw + 最终合并
    for f in Path(OUT_DIR).glob("*_normalized*.csv"):
        f.unlink()
    for csv_path, _ in clean_csvs:
        p = Path(csv_path)
        if p.exists() and p.name != "cdli_clean.csv":
            p.unlink()
    # 清理不需要的 raw 文件
    keep_raws = {"cdli_raw_akk_oa.csv", "cdli_raw_akk_ob.csv",
                 "cdli_raw_akk_oakk.csv", "cdli_raw_sux.csv"}
    for f in Path(OUT_DIR).glob("cdli_raw_*.csv"):
        if f.name not in keep_raws:
            f.unlink()
    # 清理旧的 cdli_parallel_clean_all.csv
    old_all = Path(OUT_DIR) / "cdli_parallel_clean_all.csv"
    if old_all.exists():
        old_all.unlink()

    # 统计摘要
    print(f"\n{'='*60}")
    print(f"  最终输出 →  {len(df_all)} 条  →  {final_csv}")
    print(f"\n  方言分布:")
    for col_val in df_all["dialect"].unique():
        cnt = (df_all["dialect"] == col_val).sum()
        print(f"    {col_val}: {cnt} 条")
    print(f"\n  列: {list(df_all.columns)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
