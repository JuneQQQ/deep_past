# Deep Past

> 低资源古亚述语机器翻译系统 + 学术论文 QA 数据合成智能体

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)]()

---

## 项目概览

本项目包含两个核心系统：

| 系统 | 任务 | 核心技术 |
|------|------|---------|
| **Deep Past 翻译引擎** | 古亚述楔形文字 → 英文翻译 | ByT5 CPT+SFT、Hybrid MBR、Model Soup |
| **SimQAAgent 数据工厂** | 从学术论文合成高质量 QA 训练数据 | 多智能体 ReAct、Quality Judge 反馈闭环、逆向任务 |

---

## 一、Deep Past 翻译引擎

### 问题定义

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

## 二、SimQAAgent 数据工厂

### 问题定义

从学术论文自动合成高质量 QA 训练数据，用于 LLM 的学术领域 SFT。

### 系统架构

```
论文全文 Markdown
  │
  ├─ 1. 段落提取与任务选择
  │     按空行切分 → 过滤标题/表格 → 随机选 3~5 种任务
  │
  ├─ 2. 多 Agent 并行生成 ─── agents.py
  │     5 类正向 Agent（Summary/Innovation/Experiment...）
  │     5 类逆向 Agent（DraftPolish/LogicRefinement/CritiqueRevision...）
  │     每个 Agent 内部走 ReAct 三轮推理：分析→生成→自省
  │
  ├─ 3. Quality Judge 反馈闭环 ─── generator.py
  │     Judge 独立评审 → 被拒 QA 反馈回原 Agent → 重新生成
  │     最多 3 轮反馈，超限丢弃
  │
  ├─ 4. 后处理与验证
  │     段落标签匹配 / 语言推断 / 长度检查
  │
  └─ PostgreSQL 入库 → SFT 训练集
```

### 关键技术点

**ReAct 多轮自省——不是一次调用就出结果**

每个 Agent 走三轮对话：第一轮分析论文关键内容，第二轮生成 QA 初稿，第三轮自查事实准确性。三轮共享对话历史，保证上下文连贯。

**逆向任务——可控退化生成负样本**

5 类逆向 Agent 只负责生成退化版 Question，Answer 直接注入论文原文段落。这保证了 Answer 部分零幻觉——不是 LLM 编的，而是真实的高质量学术文本。

**Quality Judge——独立评审 + 反馈闭环**

Judge 是单独的 Agent，temperature=0.3 保证评判稳定。逐条打分并输出拒绝原因，被拒的 QA 携带反馈重新生成，最多三轮。Judge 调用失败时走 fallback 全部通过，不丢数据。

**逆向任务的 5 种退化策略**

| 逆向任务 | 退化方式 |
|---------|---------|
| DraftPolish | 草稿润色——引入冗余和不精确表述 |
| LogicRefinement | 逻辑修复——制造推理链断裂 |
| CritiqueRevision | 评审修改——加入批评性误导 |
| HedgingTone | 语气调整——把确定性表述改为模糊表达 |
| AbstractInstruction | 摘要指令——从摘要生成脱离具体段落的泛化问题 |

### MyAutoConverter——MCQ 评测管线

基于 QASA 论文实现的自动选择题生成与评测系统：

```
QASA 样本 → 5 类干扰项生成(Concept/Reasoning/Nuance/Data/QuestionBias)
          → Reviewer 审核 → Refiner 改进 → Fusion 选优
          → Evaluator 评估 → Final Refiner（低分时触发）
          → 标准 4 选 1 MCQ → vLLM 评测 → accuracy
```

双模型设计：MiMo-V2-Flash 做生成审核（15 次调用/样本），gpt-5-mini 做决策（2~3 次调用/样本）。

---

## 技术栈

- **模型训练**：PyTorch、Transformers、ms-swift、DeepSpeed ZeRO
- **推理部署**：vLLM、PagedAttention
- **智能体**：OpenAI API、Pydantic、ThreadPoolExecutor
- **数据工程**：MinerU OCR、PostgreSQL、json_repair
- **评测**：BLEU、chrF++、MMLU、C-Eval、SQL 执行准确性
