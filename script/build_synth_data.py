#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import pandas as pd


@dataclass
class Config:
    input_train_csv: Path = Path("/data/lsb/deep_past/data/qwen_sentence_aligned_clean.csv")
    output_synth_csv: Path = Path("/data/lsb/deep_past/data/qwen_sentence_aligned_targeted_synth.csv")
    output_merged_csv: Path = Path("/data/lsb/deep_past/data/qwen_sentence_aligned_with_targeted_synth.csv")
    output_entity_json: Path = Path("/data/lsb/deep_past/data/qwen_sentence_aligned_targeted_entity_stats.json")
    total_numeric_rows: int = 200
    total_month_rows: int = 60
    total_vocab_rows: int = 60
    max_name_focus_rows: int = 30
    per_entity_examples: int = 2
    merge_ratio: float = 0.5
    random_seed: int = 42


ENTITY_PATTERN = re.compile(r"\b[A-ZŠṬṢḪĀĒĪŪ][a-zA-ZšṭṣḫāēīūŠṬṢḪĀĒĪŪ'\-’]*\b")
ENTITY_STOPWORDS = {
    "The", "A", "An", "To", "From", "He", "She", "It", "They",
    "In", "On", "At", "When", "If", "But", "And", "Or", "Then",
    "As", "For", "By", "With", "This", "That", "These", "Those",
    "I", "We", "You", "Your", "My", "His", "Her", "Their", "Our",
    "Since", "Of", "Also", "Send", "Please", "Thus", "So", "Now",
    "Let", "Thereof", "May",
    "Witnessed", "Sealed", "Month", "Year", "Day", "Is", "Are", "Was", "Were",
    "Do", "Sell", "City", "Palace", "Colony", "Witness", "Witnesses", "Seal",
    "Urgent", "Today", "Tomorrow", "House", "Silver", "Gold", "Copper", "Tin",
    "Barley", "Textiles", "Textile", "Donkeys", "Donkey", "Tablets", "Tablet",
    "Week", "Weeks", "Minas", "Mina", "Shekel", "Shekels", "Concerning", "Regarding",
}

FRACTIONS = ["", "½", "⅓", "⅔", "⅙", "⅚"]
FRACTION_VALUE = {
    "": Fraction(0, 1),
    "½": Fraction(1, 2),
    "⅓": Fraction(1, 3),
    "⅔": Fraction(2, 3),
    "⅙": Fraction(1, 6),
    "⅚": Fraction(5, 6),
}
VALUE_FRACTION = {value: symbol for symbol, value in FRACTION_VALUE.items() if symbol}
METALS = [
    ("KÙ.BABBAR", "silver"),
    ("KÙ.GI", "gold"),
    ("URUDU", "copper"),
    ("AN.NA", "tin"),
]
GOODS = [
    ("TÚG ku-ta-nim", "kutānum textiles", 20),
    ("ANŠE.HI.A", "donkeys", 12),
    ("DUG", "jars", 15),
]
DEBTORS = [
    ("a-lá-hi-im", "Ali-ahum"),
    ("i-dí-a-bu-um", "Iddin-abum"),
    ("en-na-sú-in", "Enna-Suen"),
    ("a-šùr-na-da", "Aššur-nādā"),
    ("bu-zu-ta-a", "Buzutaya"),
]
EPONYMS = [
    ("a-lá-hi-im", "Ali-ahum"),
    ("i-dí-a-bu-um", "Iddin-abum"),
    ("šu-da-a", "Šudāya"),
    ("bu-zu-ta-a", "Buzutaya"),
    ("a-šur-i-mì-tí", "Aššur-imittī"),
]
MONTH_TRANSLITERATION_FORMS = {
    1: ["bé-el-tí-É.GAL-lim", "be-el-té-É.GAL-lim"],
    2: ["ša sá-ra-tim"],
    3: ["ke-na-tim"],
    4: ["ma-hu-ur-dingir", "ma-ḫu-ur-i-lí"],
    5: ["áb-ša-ra-ni", "áb ša-ra-ni", "áb-ša-ra-nu"],
    6: ["hu-bu-ur"],
    7: ["ṣí-ip-im"],
    8: ["qá-ra-a-tí", "qá-ra-a-tim"],
    9: ["kán-wa-ar-ta", "kán-wa-ar-ta-an", "kán-wár-ta", "kán-mar-ta"],
    10: ["té-i-na-tim"],
    11: ["ku-zal-li", "ku-zal-lu"],
    12: ["a-lá-na-tim", "a-lá-na-tum"],
}

# 商业术语合成框架 (音译片段, 英文翻译, 语境框架列表)
TRADE_VOCAB_FRAMES = [
    ("ṣí-pá-ra-tim", "bronze pins", [
        ("{qty} GÍN KÙ.BABBAR ší-im {word}", "{qty} {shekel_pl} of silver, the price of {en}"),
        ("{qty} {word} a-na {debtor_tl} a-dí-in", "I gave {qty} {en} to {debtor_en}."),
    ]),
    ("e-ri-qá-tim", "wagons", [
        ("{qty} {word} iš-tí {debtor_tl}", "{qty} {en} are owed by {debtor_en}."),
    ]),
    ("sí-il5-qám", "boiled meat", [
        ("{word} ù ki-ra-am {debtor_tl}", "{en} and a jar: {debtor_en}."),
    ]),
    ("bi4-il5-tám", "porterage", [
        ("{qty} GÍN KÙ.BABBAR {word}", "{qty} {shekel_pl} of silver as {en}"),
        ("{word} ša {debtor_tl}", "{en} of {debtor_en}."),
    ]),
    ("kà-mu-nim", "cumin", [
        ("{qty} ma-na {word}", "{qty} {mina_pl} of {en}"),
    ]),
    ("mu-sà-ra-am", "girdle", [
        ("1 {word} a-na {debtor_tl} áš-qúl", "I paid 1 {en} to {debtor_en}."),
    ]),
    ("ta-ar-ki-is-tám", "debt-note", [
        ("{word} ša {debtor_tl} na-áš-a-kum", "{debtor_en} brings you {en}."),
    ]),
    ("ki-iṣ-ra-am", "lump", [
        ("1 {word} KÙ.BABBAR", "1 {en} of silver"),
    ]),
    ("a-ba-ra-am", "lead", [
        ("{qty} ma-na {word}", "{qty} {mina_pl} of {en}"),
    ]),
    ("na-ah-lá-áp-tám", "dress", [
        ("1 {word} a-na {debtor_tl} a-dí-in", "I gave 1 {en} to {debtor_en}."),
    ]),
    ("pá-ar-ší-am", "sash", [
        ("1 {word} ù 1 na-ah-lá-áp-tám", "1 {en} and 1 dress"),
    ]),
    ("ku-sí-a-am", "robe", [
        ("1 {word} a-na {debtor_tl}", "1 {en} for {debtor_en}."),
    ]),
]

# 数值合成的语法锁定框架（从训练集高频模式提取）
_SYNTAX_FRAMES = [
    # 债务声明
    ("{core_tl} i-ṣé-er {debtor_tl} i-šu",
     "{debtor_en} owes {core_tr}."),
    # 支付动作
    ("{core_tl} a-na {debtor_tl} áš-qúl",
     "I paid {core_tr} to {debtor_en}."),
    # thereof 分项
    ("ŠÀ.BA {core_tl} {debtor_tl} il5-qé",
     "Thereof {debtor_en} received {core_tr}."),
    # 封印运输
    ("{core_tl} ku-nu-ki-a {debtor_tl} na-áš-a-kum",
     "{debtor_en} brings you {core_tr} under my seal."),
    # 利息条款
    ("{core_tl} ṣí-ib-tám ú-ṣa-áb",
     "He will add {core_tr} as interest."),
    # 总计
    ("ŠU.NÍGIN {core_tl}",
     "In all: {core_tr}."),
]

# Logogram 人名模式（用于 build_name_focus_rows 兜底）
LOGOGRAM_NAME_PATTERNS = [
    r"\bLUGAL\b",
    r"\bNIN\.ŠUBUR\b",
    r"\bMAN\b",
    r"\bDU10\b",
]


def latest_predictions_csv() -> Path | None:
    paths = sorted(Path("/data/lsb/deep_past/output").glob("model_*/predictions.csv"))
    return paths[-1] if paths else None


def extract_entities(text: str) -> list[str]:
    text = str(text).replace("’", "'")
    entities: list[str] = []
    for word in ENTITY_PATTERN.findall(text):
        word = re.sub(r"'s$", "", word)
        if word in ENTITY_STOPWORDS:
            continue
        entities.append(word)
    return entities


def pluralize(unit: str, qty_text: str) -> str:
    if qty_text == "1":
        return unit[:-1] if unit.endswith("s") else unit
    return unit


def quantity_text(whole: int, fraction: str) -> str:
    if whole and fraction:
        return f"{whole} {fraction}"
    if whole:
        return str(whole)
    return fraction


def quantity_value(whole: int, fraction: str) -> Fraction:
    return Fraction(whole, 1) + FRACTION_VALUE[fraction]


def format_fraction_value(value: Fraction) -> str:
    whole = value.numerator // value.denominator
    frac = value - whole
    if whole and frac:
        frac_text = VALUE_FRACTION.get(frac)
        if not frac_text:
            raise ValueError(f"Unsupported fraction value: {value}")
        return f"{whole} {frac_text}"
    if whole:
        return str(whole)
    frac_text = VALUE_FRACTION.get(frac)
    if not frac_text:
        raise ValueError(f"Unsupported fraction value: {value}")
    return frac_text


def build_total_shekel_text(
    mina_whole: int,
    mina_fraction: str,
    shekel_whole: int,
    shekel_fraction: str,
) -> str:
    total = quantity_value(mina_whole, mina_fraction) * 60 + quantity_value(shekel_whole, shekel_fraction)
    return format_fraction_value(total)


def build_numeric_rows(cfg: Config) -> list[dict]:
    rng = random.Random(cfg.random_seed)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    idx = 1
    mode_counts: Counter[str] = Counter()
    mode_targets = {
        "metal_simple": 30,
        "metal_mixed_literal": 24,
        "goods_pair": 16,
        "metal_pair": 16,
        "total_shekel": 64,
        "hundred_mina": 50,
    }

    def add_row(tl: str, tr: str, tag: str) -> bool:
        nonlocal idx
        key = (tl, tr)
        if key in seen:
            return False
        seen.add(key)
        rows.append(
            {
                "oare_id": f"synth-num-{idx:04d}",
                "transliteration": tl,
                "translation": tr,
                "data_source": "synthetic",
                "uncertainty": f"numeric_synth:{tag}",
            }
        )
        idx += 1
        mode_counts[tag] += 1
        return True

    seed_cases = [
        ("⅔ ma-na 5 GÍN KÙ.BABBAR", "45 shekels of silver", "total_shekel"),
        ("⅔ ma-na 5 GÍN KÙ.BABBAR i-ṣé-er a-lá-hi-im i-šu", "45 shekels of silver is owed by Ali-ahum.", "total_shekel"),
        ("⅔ ma-na ⅚ GÍN KÙ.BABBAR", "40 ⅚ shekels of silver", "total_shekel"),
        ("⅔ ma-na ⅚ GÍN KÙ.BABBAR i-ṣé-er i-dí-a-bu-um i-šu", "40 ⅚ shekels of silver is owed by Iddin-abum.", "total_shekel"),
        ("1 me-at 6 ½ ma-na URUDU", "106 ½ minas of copper", "hundred_mina"),
        ("1 me-at 6 ½ ma-na URUDU i-ṣé-er en-na-sú-in i-šu", "106 ½ minas of copper is owed by Enna-Suen.", "hundred_mina"),
    ]
    for tl, tr, tag in seed_cases:
        add_row(tl, tr, tag)

    attempts = 0
    max_attempts = cfg.total_numeric_rows * 200
    while len(rows) < cfg.total_numeric_rows and attempts < max_attempts:
        attempts += 1
        eligible_modes = [mode for mode, target in mode_targets.items() if mode_counts[mode] < target]
        if not eligible_modes:
            break
        mode = rng.choice(eligible_modes)

        if mode == "metal_simple":
            whole = rng.randint(1, 30)
            fraction = rng.choice(FRACTIONS)
            metal_tl, metal_en = rng.choice(METALS)
            qty = quantity_text(whole, fraction)
            tl = f"{qty} ma-na {metal_tl}".strip()
            tr = f"{qty} {pluralize('minas', qty)} of {metal_en}"
            add_row(tl, tr, mode)
            continue

        if mode == "metal_mixed_literal":
            whole = rng.randint(1, 12)
            fraction = rng.choice(FRACTIONS)
            shekels = rng.randint(1, 15)
            metal_tl, metal_en = rng.choice(METALS)
            qty = quantity_text(whole, fraction)
            tl = f"{qty} ma-na {shekels} GÍN {metal_tl}".strip()
            tr = f"{qty} {pluralize('minas', qty)} {shekels} {pluralize('shekels', str(shekels))} of {metal_en}"
            add_row(tl, tr, mode)
            continue

        if mode == "goods_pair":
            item1_tl, item1_en, max1 = rng.choice(GOODS)
            remaining = [x for x in GOODS if x[0] != item1_tl]
            item2_tl, item2_en, max2 = rng.choice(remaining)
            qty1 = str(rng.randint(1, max1))
            qty2 = str(rng.randint(1, max2))
            tl = f"{qty1} {item1_tl} {qty2} {item2_tl}"
            tr = f"{qty1} {pluralize(item1_en, qty1)}, {qty2} {pluralize(item2_en, qty2)}"
            add_row(tl, tr, mode)
            continue

        if mode == "metal_pair":
            qty1 = quantity_text(rng.randint(1, 20), rng.choice(FRACTIONS))
            qty2 = quantity_text(rng.randint(1, 20), rng.choice(FRACTIONS))
            metal1_tl, metal1_en = rng.choice(METALS)
            metal2_tl, metal2_en = rng.choice(METALS)
            tl = f"{qty1} ma-na {metal1_tl} {qty2} ma-na {metal2_tl}".strip()
            tr = (
                f"{qty1} {pluralize('minas', qty1)} of {metal1_en}, "
                f"{qty2} {pluralize('minas', qty2)} of {metal2_en}"
            )
            add_row(tl, tr, mode)
            continue

        if mode == "total_shekel":
            mina_whole = rng.randint(0, 8)
            mina_fraction = rng.choice(["⅓", "⅔", "½", "⅙", "⅚"])
            if mina_whole == 0 and not mina_fraction:
                mina_fraction = "⅔"
            shekel_whole = rng.randint(0, 12)
            shekel_fraction = rng.choice(FRACTIONS)
            if shekel_whole == 0 and not shekel_fraction:
                shekel_fraction = rng.choice(["⅙", "⅓", "½", "⅔", "⅚"])
            mina_text = quantity_text(mina_whole, mina_fraction)
            shekel_text = quantity_text(shekel_whole, shekel_fraction)
            total_shekel_text = build_total_shekel_text(mina_whole, mina_fraction, shekel_whole, shekel_fraction)
            metal_tl, metal_en = rng.choice(METALS)
            debtor_tl, debtor_en = rng.choice(DEBTORS)
            core_tl = f"{mina_text} ma-na {shekel_text} GÍN {metal_tl}".strip()
            # merge_ratio controls merged ("45 shekels") vs split ("⅔ mina 5 shekels") form
            if rng.random() < cfg.merge_ratio:
                core_tr = f"{total_shekel_text} {pluralize('shekels', total_shekel_text)} of {metal_en}"
            else:
                core_tr = f"{mina_text} {pluralize('minas', mina_text)} {shekel_text} {pluralize('shekels', shekel_text)} of {metal_en}"
            # 50% isolated, 50% syntax-framed
            if rng.random() < 0.5:
                add_row(core_tl, core_tr, mode)
            else:
                frame_tl, frame_tr = rng.choice(_SYNTAX_FRAMES)
                tl = frame_tl.format(core_tl=core_tl, debtor_tl=debtor_tl)
                tr = frame_tr.format(core_tr=core_tr, debtor_en=debtor_en)
                add_row(tl, tr, mode)
            continue

        hundreds = rng.randint(1, 3)
        mina_whole = rng.randint(0, 24)
        mina_fraction = rng.choice(FRACTIONS)
        if mina_whole == 0 and not mina_fraction:
            mina_whole = 6
            mina_fraction = "½"
        mina_text = quantity_text(mina_whole, mina_fraction)
        total_minas = Fraction(hundreds * 100, 1) + quantity_value(mina_whole, mina_fraction)
        total_mina_text = format_fraction_value(total_minas)
        metal_tl, metal_en = rng.choice(METALS)
        debtor_tl, debtor_en = rng.choice(DEBTORS)
        core_tl = f"{hundreds} me-at {mina_text} ma-na {metal_tl}".strip()
        core_tr = f"{total_mina_text} {pluralize('minas', total_mina_text)} of {metal_en}"
        # 50% isolated, 50% syntax-framed
        if rng.random() < 0.5:
            add_row(core_tl, core_tr, "hundred_mina")
        else:
            frame_tl, frame_tr = rng.choice(_SYNTAX_FRAMES)
            tl = frame_tl.format(core_tl=core_tl, debtor_tl=debtor_tl)
            tr = frame_tr.format(core_tr=core_tr, debtor_en=debtor_en)
            add_row(tl, tr, "hundred_mina")

    if len(rows) < cfg.total_numeric_rows:
        raise RuntimeError(f"Unable to build enough numeric synth rows: {len(rows)}/{cfg.total_numeric_rows}")
    return rows[: cfg.total_numeric_rows]


def build_month_rows(cfg: Config) -> list[dict]:
    rng = random.Random(cfg.random_seed + 17)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    idx = 1
    mode_counts: Counter[str] = Counter()
    mode_targets = {
        "bare": 20,
        "deadline": 20,
        "eponym": 28,
        "debt": 28,
    }

    def add_row(tl: str, tr: str, tag: str) -> bool:
        nonlocal idx
        key = (tl, tr)
        if key in seen:
            return False
        seen.add(key)
        rows.append(
            {
                "oare_id": f"synth-month-{idx:04d}",
                "transliteration": tl,
                "translation": tr,
                "data_source": "synthetic",
                "uncertainty": f"month_synth:{tag}",
            }
        )
        idx += 1
        mode_counts[tag] += 1
        return True

    seed_cases = [
        ("ITU.1.KAM bé-el-tí-É.GAL-lim", "Month 1", "bare"),
        ("ITU.KAM hu-bu-ur", "Month 6", "bare"),
        ("ITU.KAM kán-wa-ar-ta", "Month 9", "bare"),
        ("ITU.KAM kán-wár-ta", "Month 9", "bare"),
        ("ITU.KAM a-lá-na-tim", "Month 12", "bare"),
        ("ITU.KAM hu-bu-ur li-mu-um i-dí-a-bu-um", "Month 6, eponymy Iddin-abum.", "eponym"),
        ("ITU.KAM kán-wa-ar-ta li-mu-um bu-zu-ta-a", "Month 9, eponymy Buzutaya.", "eponym"),
    ]
    for tl, tr, tag in seed_cases:
        add_row(tl, tr, tag)

    attempts = 0
    max_attempts = cfg.total_month_rows * 200
    while len(rows) < cfg.total_month_rows and attempts < max_attempts:
        attempts += 1
        eligible_modes = [mode for mode, target in mode_targets.items() if mode_counts[mode] < target]
        if not eligible_modes:
            break
        mode = rng.choice(eligible_modes)
        month_num = rng.choice(sorted(MONTH_TRANSLITERATION_FORMS))
        form = rng.choice(MONTH_TRANSLITERATION_FORMS[month_num])
        eponym_tl, eponym_en = rng.choice(EPONYMS)
        debtor_tl, debtor_en = rng.choice(DEBTORS)

        if mode == "bare":
            add_row(f"ITU.KAM {form}", f"Month {month_num}", mode)
            continue

        if mode == "deadline":
            tl = f"iš-tù ITU.1.KAM {form} a-na ITU.2.KAM i-ša-qal"
            tr = f"Reckoned from month {month_num}, he will pay in 2 months."
            add_row(tl, tr, mode)
            continue

        if mode == "eponym":
            tl = f"ITU.KAM {form} li-mu-um {eponym_tl}"
            tr = f"Month {month_num}, eponymy {eponym_en}."
            add_row(tl, tr, mode)
            continue

        tl = f"1 ma-na KÙ.BABBAR i-ṣé-er {debtor_tl} i-šu ITU.KAM {form} li-mu-um {eponym_tl}"
        tr = f"1 mina of silver is owed by {debtor_en}. Month {month_num}, eponymy {eponym_en}."
        add_row(tl, tr, mode)

    if len(rows) < cfg.total_month_rows:
        raise RuntimeError(f"Unable to build enough month synth rows: {len(rows)}/{cfg.total_month_rows}")
    return rows[: cfg.total_month_rows]


def build_vocab_rows(cfg: Config) -> list[dict]:
    """生成商业术语合成样本，每个词 3-5 条。"""
    rng = random.Random(cfg.random_seed + 31)
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    idx = 1

    def add_row(tl: str, tr: str, tag: str) -> bool:
        nonlocal idx
        key = (tl, tr)
        if key in seen:
            return False
        seen.add(key)
        rows.append(
            {
                "oare_id": f"synth-vocab-{idx:04d}",
                "transliteration": tl,
                "translation": tr,
                "data_source": "synthetic",
                "uncertainty": f"vocab_synth:{tag}",
            }
        )
        idx += 1
        return True

    for word_tl, word_en, frames in TRADE_VOCAB_FRAMES:
        for frame_tl, frame_tr in frames:
            for _ in range(5):
                if len(rows) >= cfg.total_vocab_rows:
                    break
                qty = str(rng.randint(1, 15))
                debtor_tl, debtor_en = rng.choice(DEBTORS)
                tl = frame_tl.format(qty=qty, word=word_tl, debtor_tl=debtor_tl)
                tr = frame_tr.format(
                    qty=qty, en=word_en, debtor_en=debtor_en,
                    shekel_pl=pluralize("shekels", qty),
                    mina_pl=pluralize("minas", qty),
                )
                add_row(tl, tr, word_tl)
        if len(rows) >= cfg.total_vocab_rows:
            break

    return rows[: cfg.total_vocab_rows]


def analyze_missed_entities(pred_csv: Path) -> tuple[list[tuple[str, int]], dict]:
    df = pd.read_csv(pred_csv)
    missing_counter: Counter[str] = Counter()
    example_map: dict[str, list[dict]] = {}
    for _, row in df.iterrows():
        ref = str(row.get("reference", "") or "")
        pred = str(row.get("prediction", "") or "")
        ref_entities = extract_entities(ref)
        pred_entities_lower = {x.lower() for x in extract_entities(pred)}
        missing = [ent for ent in ref_entities if ent.lower() not in pred_entities_lower]
        for ent in missing:
            missing_counter[ent] += 1
            example_map.setdefault(ent, []).append(
                {
                    "reference": ref,
                    "prediction": pred,
                    "step": int(row.get("step", 0) or 0),
                }
            )
    ranked = missing_counter.most_common()
    return ranked, example_map


def build_name_focus_rows(
    train_df: pd.DataFrame,
    ranked_entities: list[tuple[str, int]],
    cfg: Config,
) -> list[dict]:
    rows: list[dict] = []
    used_oare_ids: set[str] = set()
    train_df = train_df.copy()
    train_df["tl_len"] = train_df["transliteration"].astype(str).str.split().str.len()
    train_df["tr_len"] = train_df["translation"].astype(str).str.split().str.len()

    def score_row(translation: str) -> int:
        text = str(translation)
        score = 0
        if text.startswith("To "):
            score += 4
        if text.startswith("Seal of "):
            score += 4
        if " son of " in text:
            score += 3
        if " owes " in text:
            score += 2
        return score

    # 兜底：即使没有 predictions.csv，也从训练集提取含 logogram 人名的样本
    if not ranked_entities:
        fallback_entities: list[tuple[str, int]] = []
        for pat in LOGOGRAM_NAME_PATTERNS:
            mask = train_df["transliteration"].astype(str).str.contains(pat, regex=True, na=False)
            matched = train_df[mask]
            if not matched.empty:
                # 从翻译中提取实体名
                for _, row in matched.head(5).iterrows():
                    for ent in extract_entities(str(row.get("translation", ""))):
                        fallback_entities.append((ent, 1))
        # 去重
        seen_fb = set()
        deduped = []
        for ent, cnt in fallback_entities:
            if ent not in seen_fb:
                seen_fb.add(ent)
                deduped.append((ent, cnt))
        ranked_entities = deduped[:20]

    synth_idx = 1
    for entity, miss_count in ranked_entities:
        if len(rows) >= cfg.max_name_focus_rows:
            break
        mask = train_df["translation"].astype(str).str.contains(rf"\b{re.escape(entity)}\b", regex=True, na=False)
        candidates = train_df[mask].copy()
        if candidates.empty:
            continue
        candidates["focus_score"] = candidates["translation"].apply(score_row) - candidates["tl_len"] - candidates["tr_len"]
        candidates = candidates.sort_values(["focus_score", "tl_len", "tr_len"], ascending=[False, True, True])
        taken = 0
        for _, row in candidates.iterrows():
            oare_id = str(row["oare_id"])
            if oare_id in used_oare_ids:
                continue
            rows.append(
                {
                    "oare_id": f"synth-name-{synth_idx:04d}",
                    "transliteration": row["transliteration"],
                    "translation": row["translation"],
                    "data_source": "synthetic",
                    "uncertainty": f"name_focus:{entity}:{miss_count}",
                }
            )
            synth_idx += 1
            used_oare_ids.add(oare_id)
            taken += 1
            if taken >= cfg.per_entity_examples or len(rows) >= cfg.max_name_focus_rows:
                break
    return rows


def main() -> None:
    cfg = Config()
    random.seed(cfg.random_seed)

    train_df = pd.read_csv(cfg.input_train_csv)
    numeric_rows = build_numeric_rows(cfg)
    month_rows = build_month_rows(cfg)

    pred_csv = latest_predictions_csv()
    if pred_csv is None:
        ranked_entities: list[tuple[str, int]] = []
        entity_examples: dict = {}
    else:
        ranked_entities, entity_examples = analyze_missed_entities(pred_csv)

    vocab_rows = build_vocab_rows(cfg)
    name_focus_rows = build_name_focus_rows(train_df, ranked_entities, cfg)
    synth_df = pd.DataFrame(numeric_rows + month_rows + vocab_rows + name_focus_rows)
    merged_df = pd.concat([train_df, synth_df], ignore_index=True)

    cfg.output_synth_csv.parent.mkdir(parents=True, exist_ok=True)
    synth_df.to_csv(cfg.output_synth_csv, index=False)
    merged_df.to_csv(cfg.output_merged_csv, index=False)

    report = {
        "input_train_csv": str(cfg.input_train_csv),
        "predictions_csv": str(pred_csv) if pred_csv else None,
        "numeric_rows": len(numeric_rows),
        "month_rows": len(month_rows),
        "vocab_rows": len(vocab_rows),
        "name_focus_rows": len(name_focus_rows),
        "merged_rows": len(merged_df),
        "top_missing_entities": [
            {
                "entity": entity,
                "missing_count": count,
                "examples": entity_examples.get(entity, [])[:2],
            }
            for entity, count in ranked_entities[:20]
        ],
    }
    cfg.output_entity_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved synthetic rows: {len(synth_df)} -> {cfg.output_synth_csv}")
    print(f"Saved merged dataset: {len(merged_df)} -> {cfg.output_merged_csv}")
    if pred_csv:
        print(f"Used predictions file: {pred_csv}")
    print("Top missing entities:")
    for entity, count in ranked_entities[:10]:
        print(f"  {entity}: {count}")


if __name__ == "__main__":
    main()
