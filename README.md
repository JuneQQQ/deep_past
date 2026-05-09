# Deep Past

> 低资源古亚述语机器翻译系统

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 项目概览

将古亚述楔形文字转写翻译为英文的低资源机器翻译系统，采用 ByT5 CPT+SFT 两阶段训练 + Hybrid MBR 推理集成。

---

## 问题定义

将 Old Assyrian 楔形文字转写翻译为英文，评估指标 `geom = √(BLEU × chrF++)`。

这不是常规机器翻译任务——平行语料仅约 1.5k 条，文本充满 `<gap>` 残损标记、OCR 噪声和拼写变体，专名与数字翻译一错即掉分。

### 系统架构

```
原始语料
  │
  ├─ 1. 严格数据清洗 ─── prepare_data.py
  │     字符归一化 / <gap> 守恒 / 词典过滤
  │
  ├─ 2. LLM 辅助句对齐 ─── qwen_align_sentences.py
  │     Qwen 切分 + 五层硬约束校验（gap/数字/变音符/覆盖率/边界）
  │
  ├─ 3. 持续预训练 CPT ─── build_cpt_bdlm_data.py + train_cpt_bdlm.py
  │     ByT5 BDLM span corruption + 词典增强 + SFT 数据混入防漂移
  │
  ├─ 4. 监督微调 SFT ─── train.py
  │     Meta Prefix 域标识 / R-Drop 一致性 / Coverage Loss 防欠翻
  │
  ├─ 5. 推理集成 ─── infer.py + merge_models.py
  │     Model Soup 权重平均 + Hybrid MBR 候选共识
  │
  └─ 提交
```

### 关键技术点

**数据工程——把脏数据变成可学数据**

- 统一残损标记 `<gap>`，确保源-目标端守恒，不一致则直接过滤
- 构建高精度 hint 词典，用支持度与优势比过滤保证 precision
- LLM 只负责"切分决策"，不负责"改写原文"，降低幻觉风险

**CPT 阶段——先学语言形态再学翻译映射**

- ByT5 字节级建模天然避免 BPE/OOV，对死语言的稀有字符和拼写变体极其友好
- BDLM span corruption 预训练 + 词典语义注解，让模型先理解古语分布
- 混入 50% SFT 翻译样本防止任务漂移，训练中监控 translation probes 确保不"只会填空"

**SFT 阶段——针对比赛痛点的多重约束**

- Meta Prefix 编码样本域信息（官方/噪声、完整/残损），让模型区分数据来源
- R-Drop 同样本双 dropout 输出做 KL 一致性约束，缓解低资源过拟合
- Coverage Loss 显式惩罚漏译和过短输出
- Group-aware split 按 `oare_id` 分组，防止同文档片段泄漏到验证集

**推理工程——最后几分来自集成与共识**

- Model Soup：多 checkpoint 权重平均，降低方差
- Hybrid MBR：Beam 候选 + Sampling 候选混合池，chrF/BLEU/Jaccard 加权共识选最终输出
- 异步流水线并行，生成/后处理/MBR 选择分阶段异步化提升吞吐

### 脚本索引

| 脚本 | 功能 |
|------|------|
| `script/prepare_data.py` | 数据清洗主干 |
| `script/qwen_align_sentences.py` | LLM 句对齐 + 硬约束校验 |
| `script/build_cpt_bdlm_data.py` | BDLM CPT 数据构建 |
| `script/train_cpt_bdlm.py` | ByT5 CPT 训练 |
| `script/train.py` | SFT 主训练 |
| `script/infer.py` | 多策略推理解码 |
| `script/merge_models.py` | Model Soup 融合 |
| `script/train_dpo.py` | DPO 偏好优化探索 |
| `script/train_grpo.py` | GRPO 强化学习探索 |

---

## 技术栈

- **模型训练**：PyTorch、Transformers、ms-swift、DeepSpeed ZeRO
- **推理部署**：vLLM、PagedAttention
- **数据工程**：MinerU OCR
- **评测**：BLEU、chrF++
