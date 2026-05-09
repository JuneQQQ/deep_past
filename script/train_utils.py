#!/usr/bin/env python3
"""
train_utils.py — 从 train.py 提取的独立工具类

包含：
- ErrorAnalyzer: 预测错误类型分析
- DynamicTokenBatchSampler: 按 token 预算贪心组 batch
- HybridRAGIndex: BM25 + TF-IDF 双路检索 + RRF 融合
"""

from __future__ import annotations

import os
import random
import re
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ============================================================================
# ErrorAnalyzer
# ============================================================================

class ErrorAnalyzer:
    """分析预测中的常见错误类型"""
    
    def __init__(self, data_dir: str = ""):
        self.error_types = {
            'proper_noun': [],      # 专有名词错误
            'number': [],           # 数字/度量错误
            'repeated': [],         # 重复生成
            'truncated': [],        # 截断/不完整
            'hallucination': [],    # 幻觉（生成不存在的内容）
            'gap_handling': [],     # gap 处理错误
            'placeholder_tag': [],  # 占位符标签丢失/多余
        }
        self._tag_pattern = re.compile(r'\[\[[EN]\d{3}\]\]')
        self.total_samples = 0
        
        # 从 Sentences_Oare_FirstWord_LinNum.csv 提取专有名词（人名/地名）
        self.proper_nouns_whitelist = set()
        oare_csv = os.path.join(data_dir, "Sentences_Oare_FirstWord_LinNum.csv")
        if os.path.exists(oare_csv):
            try:
                oare_df = pd.read_csv(oare_csv)
                if 'translation' in oare_df.columns:
                    for text in oare_df['translation'].dropna():
                        # 提取人名模式：
                        # 1. 带连字符的名字 (X-Y 或 X-Y-Z)，如 Puzur-Aššur, Šalim-aḫum
                        # 2. "s. NAME" 模式中的 NAME (son of)
                        # 3. 首字母大写且包含特殊字符的词
                        
                        # 模式1: 连字符名字 (至少一个大写字母开头)
                        hyphen_names = re.findall(r'\b[A-ZŠṢṬḪ][a-zāēīūšṣṭḫ]*(?:-[A-Za-zāēīūšṣṭḫ]+)+', str(text))
                        for name in hyphen_names:
                            self.proper_nouns_whitelist.add(name)
                        
                        # 模式2: "s. NAME" 或 "d. NAME" (son/daughter of)
                        son_of = re.findall(r'\bs\. ([A-ZŠṢṬḪ][a-zāēīūšṣṭḫ-]+)', str(text))
                        for name in son_of:
                            self.proper_nouns_whitelist.add(name)
                        
                        # 模式3: 带变音符的大写开头词（通常是阿卡德人名）
                        special_names = re.findall(r'\b[A-ZŠṢṬḪ][a-zāēīūšṣṭḫ]*[āēīūšṣṭḫ][a-zāēīūšṣṭḫ]*\b', str(text))
                        for name in special_names:
                            # 排除常见词
                            if name not in {'The', 'From', 'To', 'If', 'He', 'She', 'They', 'Month', 'Year', 'Witness', 'Witnesses', 'Seal'}:
                                self.proper_nouns_whitelist.add(name)
                        
                        # 模式4: 含神名的人名 (theophoric names)
                        god_names = r'(?:Šamaš|Ištar|Aššur|Adad|Ilabrat|Suen|Sin|Nergal|Enlil|Marduk|Nabû|Aya|Dagan|Ilī|Bēl|Kubum|Laban|Amurrum|Išḫara|Ana)'
                        theophoric = re.findall(rf'\b{god_names}[-]?[a-zāēīūšṣṭḫA-ZŠṢṬḪ-]*\b', str(text))
                        for name in theophoric:
                            self.proper_nouns_whitelist.add(name)
                        
                        # 模式5: 以神名结尾的人名 (X-Šamaš, X-Ištar, Šū-Enlil 等)
                        theophoric_suffix = re.findall(rf'\b[A-ZŠṢṬḪ][a-zāēīūšṣṭḫ]*-{god_names}\b', str(text))
                        for name in theophoric_suffix:
                            self.proper_nouns_whitelist.add(name)
                        
                        # 模式6: Šū-X 格式 (常见神名人名前缀)
                        shu_names = re.findall(r'\bŠū-[A-ZŠṢṬḪ][a-zāēīūšṣṭḫ-]*\b', str(text))
                        for name in shu_names:
                            self.proper_nouns_whitelist.add(name)
                
                print(f"   📘 从 OARE 提取专有名词: {len(self.proper_nouns_whitelist)} 个词条")
            except Exception as e:
                print(f"   ⚠️ 无法从 OARE 提取专有名词: {e}")
    
    def analyze(self, predictions, references):
        """分析一批预测结果"""
        self.error_types = {k: [] for k in self.error_types}
        self.total_samples = len(predictions)
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            pred_lower = pred.lower()
            ref_lower = ref.lower()
            
            # 1. 专有名词错误：基于官方词典 (Hints) 检测
            def get_proper_nouns(text):
                names = set()
                for w in text.split():
                    clean_w = w.strip(".,;:?!()[]{}'\"")
                    if clean_w in self.proper_nouns_whitelist:
                        names.add(clean_w)
                return names

            ref_names = get_proper_nouns(ref)
            pred_names = get_proper_nouns(pred)
            missing_names = ref_names - pred_names
            if missing_names:
                self.error_types['proper_noun'].append({
                    'idx': i, 'missing': list(missing_names)[:3],
                    'pred_snippet': pred, 'ref_snippet': ref
                })
            
            # 2. 数字错误：检查数字是否匹配
            ref_nums = set(re.findall(r'\b\d+(?:\.\d+)?(?:/\d+)?\b', ref))
            pred_nums = set(re.findall(r'\b\d+(?:\.\d+)?(?:/\d+)?\b', pred))
            if ref_nums and ref_nums != pred_nums:
                self.error_types['number'].append({
                    'idx': i, 'ref_nums': list(ref_nums), 'pred_nums': list(pred_nums),
                    'pred_snippet': pred
                })
            
            # 3. 重复生成：检查连续重复的短语
            words = pred.split()
            if len(words) >= 6:
                for n in [3, 4, 5]:
                    for j in range(len(words) - 2*n + 1):
                        if words[j:j+n] == words[j+n:j+2*n]:
                            self.error_types['repeated'].append({
                                'idx': i, 'repeated_phrase': ' '.join(words[j:j+n]),
                                'pred_snippet': pred,
                                'ref_snippet': ref
                            })
                            break
                    else:
                        continue
                    break
            
            # 4. 截断检测：预测明显短于参考
            if len(pred) < len(ref) * 0.5 and len(ref) > 50:
                self.error_types['truncated'].append({
                    'idx': i, 'pred_len': len(pred), 'ref_len': len(ref),
                    'pred_snippet': pred
                })
            
            # 5. Gap 处理：检查 gap 相关问题
            if '<gap>' in ref_lower:
                if '...' in pred or 'xxx' in pred_lower:
                    self.error_types['gap_handling'].append({
                        'idx': i, 'issue': 'gap_not_normalized',
                        'pred_snippet': pred
                    })
            
            # 6. 占位符标签匹配：检查 [[E0xx]]/[[N0xx]] 标签一致性
            ref_tags = set(self._tag_pattern.findall(ref))
            pred_tags = set(self._tag_pattern.findall(pred))
            if ref_tags or pred_tags:
                missing_tags = ref_tags - pred_tags
                extra_tags = pred_tags - ref_tags
                if missing_tags or extra_tags:
                    self.error_types['placeholder_tag'].append({
                        'idx': i,
                        'missing': sorted(missing_tags),
                        'extra': sorted(extra_tags),
                        'ref_tags': sorted(ref_tags),
                        'pred_tags': sorted(pred_tags),
                    })
        
        return self.get_summary()
    
    def get_summary(self):
        """返回错误摘要"""
        summary = {}
        for err_type, errors in self.error_types.items():
            summary[err_type] = {
                'count': len(errors),
                'ratio': len(errors) / max(self.total_samples, 1) * 100,
                'examples': errors[:2]  # 只保留前2个例子
            }
        return summary
    
    def print_report(self, step=None):
        """打印错误分析报告"""
        print(f"\n{'─'*60}")
        print(f"🔍 ERROR ANALYSIS" + (f" (Step {step})" if step else ""))
        print(f"{'─'*60}")
        
        # 按错误数量排序
        sorted_types = sorted(
            self.error_types.items(), 
            key=lambda x: len(x[1]), 
            reverse=True
        )
        
        for err_type, errors in sorted_types:
            count = len(errors)
            if count == 0:
                continue
            
            ratio = count / max(self.total_samples, 1) * 100
            type_labels = {
                'proper_noun': '📛 专有名词',
                'number': '🔢 数字/度量',
                'repeated': '🔄 重复生成',
                'truncated': '✂️ 截断/不完整',
                'hallucination': '👻 幻觉',
                'gap_handling': '⬜ Gap处理',
                'placeholder_tag': '🏷️ 占位符标签',
            }
            label = type_labels.get(err_type, err_type)
            print(f"\n{label}: {count} ({ratio:.1f}%)")
            
            # 打印示例
            for j, err in enumerate(errors[:2]):
                if err_type == 'proper_noun' and 'missing' in err:
                    print(f"   [{j+1}] 缺失: {err['missing']}")
                elif err_type == 'number':
                    print(f"   [{j+1}] 参考: {err.get('ref_nums', [])} → 预测: {err.get('pred_nums', [])}")
                elif err_type == 'repeated' and 'repeated_phrase' in err:
                    print(f"   [{j+1}] 重复: \"{err['repeated_phrase']}\"")
                    print(f"        预测: {err.get('pred_snippet', '')}")
                    print(f"        原文: {err.get('ref_snippet', '')}")
                elif err_type == 'truncated':
                    print(f"   [{j+1}] 长度: {err.get('pred_len', 0)} vs {err.get('ref_len', 0)}")
                elif err_type == 'placeholder_tag':
                    missing = err.get('missing', [])
                    extra = err.get('extra', [])
                    parts = []
                    if missing:
                        parts.append(f"丢失={missing}")
                    if extra:
                        parts.append(f"多余={extra}")
                    print(f"   [{j+1}] {', '.join(parts)}  (ref={err.get('ref_tags', [])}, pred={err.get('pred_tags', [])})")
        
        if all(len(errors) == 0 for errors in self.error_types.values()):
            print("   ✅ 未检测到明显错误模式")
        
        print(f"{'─'*60}\n")


# ============================================================================
# DynamicTokenBatchSampler
# ============================================================================

class DynamicTokenBatchSampler:
    """按 token 预算贪心组 batch，尽量贴近预算上限。"""

    def __init__(
        self,
        dataset,
        max_tokens: int,
        *,
        shuffle: bool,
        drop_last: bool,
        seed: int,
        bucket_size: int = 128,
        max_examples: int = 0,
    ):
        self.dataset = dataset
        self.max_tokens = max(int(max_tokens), 1)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = int(seed)
        self.bucket_size = max(int(bucket_size), 1)
        self.max_examples = max(int(max_examples), 0)
        self._epoch = 0
        self._lengths = []
        self._cached_epoch = None
        self._cached_batches = None

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            src_len = len(sample.get("input_ids", []))
            tgt_len = len(sample.get("labels", []))
            self._lengths.append((src_len, tgt_len))

    def _estimate_cost(self, count: int, max_src: int, max_tgt: int) -> int:
        return count * (max_src + max_tgt)

    def _ordered_indices(self, epoch: int) -> List[int]:
        indices = list(range(len(self._lengths)))
        if not self.shuffle:
            return indices

        rng = random.Random(self.seed + epoch)
        rng.shuffle(indices)

        ordered = []
        for start in range(0, len(indices), self.bucket_size):
            chunk = indices[start:start + self.bucket_size]
            chunk.sort(
                key=lambda idx: (
                    max(self._lengths[idx][0], self._lengths[idx][1]),
                    self._lengths[idx][0] + self._lengths[idx][1],
                ),
                reverse=True,
            )
            ordered.extend(chunk)
        return ordered

    def _build_batches(self, epoch: int) -> List[List[int]]:
        if self._cached_epoch == epoch and self._cached_batches is not None:
            return self._cached_batches

        batches: List[List[int]] = []
        current_batch: List[int] = []
        current_max_src = 0
        current_max_tgt = 0

        for idx in self._ordered_indices(epoch):
            src_len, tgt_len = self._lengths[idx]
            next_max_src = max(current_max_src, src_len)
            next_max_tgt = max(current_max_tgt, tgt_len)
            next_count = len(current_batch) + 1
            next_cost = self._estimate_cost(next_count, next_max_src, next_max_tgt)

            overflow_tokens = current_batch and next_cost > self.max_tokens
            overflow_examples = self.max_examples > 0 and next_count > self.max_examples
            if overflow_tokens or overflow_examples:
                batches.append(current_batch)
                current_batch = [idx]
                current_max_src = src_len
                current_max_tgt = tgt_len
                continue

            current_batch.append(idx)
            current_max_src = next_max_src
            current_max_tgt = next_max_tgt

        if current_batch and not self.drop_last:
            batches.append(current_batch)

        self._cached_epoch = epoch
        self._cached_batches = batches
        return batches

    def stats(self) -> Dict[str, float]:
        batches = self._build_batches(self._epoch)
        if not batches:
            return {
                "num_batches": 0.0,
                "avg_batch_size": 0.0,
                "max_batch_size": 0.0,
                "avg_batch_tokens": 0.0,
                "max_batch_tokens": 0.0,
                "mean_fill_ratio": 0.0,
            }

        batch_sizes = []
        batch_tokens = []
        for batch in batches:
            max_src = max(self._lengths[idx][0] for idx in batch)
            max_tgt = max(self._lengths[idx][1] for idx in batch)
            est_tokens = self._estimate_cost(len(batch), max_src, max_tgt)
            batch_sizes.append(len(batch))
            batch_tokens.append(est_tokens)

        return {
            "num_batches": float(len(batches)),
            "avg_batch_size": float(np.mean(batch_sizes)),
            "max_batch_size": float(np.max(batch_sizes)),
            "avg_batch_tokens": float(np.mean(batch_tokens)),
            "max_batch_tokens": float(np.max(batch_tokens)),
            "mean_fill_ratio": float(np.mean([tokens / self.max_tokens for tokens in batch_tokens])),
        }

    def __iter__(self):
        batches = self._build_batches(self._epoch)
        if self.shuffle:
            self._epoch += 1
        return iter(batches)

    def __len__(self):
        return len(self._build_batches(self._epoch))


# ============================================================================
# HybridRAGIndex
# ============================================================================

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
