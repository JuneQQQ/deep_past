# Deep Past Challenge 学习版 README（从 0 到银牌）

这份文档是给**第一次接触这个项目**的人看的。目标不是背术语，而是建立一套能在面试里讲清楚的思考框架：

1. 这个比赛到底难在哪；
2. 代码是怎么从 baseline 逐步进化到强系统的；
3. 每次改动为什么有效；
4. 你作为队员应该怎么讲，既专业又不夸大。

---

## 1. 先把问题想明白：这不是普通机器翻译

任务：把 Old Assyrian（古亚述语转写）翻成英文。  
评估：`geom_mean = sqrt(BLEU * chrF++)`（语料级 micro-average）。

这个任务和常见中英翻译很不一样：

- 低资源：官方平行数据只有约 1.5k 条量级，远不够端到端大模型直接学好。
- 噪声重：真实文本里有 `<gap>`、残损、OCR 伪字符、行号残留、拼写变体。
- 形态复杂：一个词里编码了很多语法信息，和英文词对齐不自然。
- 专名/数字极其关键：人名、地名、数量单位、分数写法一错就掉分。

所以核心思路不是“堆一个大模型”，而是：

- **先把数据做对**（清洗、对齐、扩充）；
- **再让模型学会领域字节模式**（CPT）；
- **再做翻译任务学习**（SFT）；
- **最后在推理端用候选共识减少随机性**（Hybrid MBR + Model Soup）。

---

## 2. 代码全景（按功能分层）

### 2.1 数据层（质量和规模）

- `script/prepare_data.py`  
  数据清洗主干：字符归一、`<gap>` 规范化、translation 噪声修复、质量过滤、高精度 hint 词典构建。

- `script/qwen_align_sentences.py`  
  用 Qwen 做句对齐和伪标签构建，带强约束校验（gap/数字/变音符/覆盖率/边界）。

- `script/build_cpt_bdlm_data.py`  
  构建 BDLM 持续预训练数据，离线预计算 span corruption，并混入 SFT 翻译对做冷启动。

- 其他数据构建脚本（语料扩充）  
  `build_cdli_data.py`、`build_archibab_data.py`、`build_ocr_sft_data.py`、`extract_*_mineru*.py` 等。

### 2.2 训练层（参数学习）

- `script/train_cpt_bdlm.py`：ByT5 的 BDLM CPT。
- `script/train.py`：主力 SFT（含 R-Drop、coverage loss、meta prefix、边界评估等）。
- `script/train_dpo.py` / `script/train_preference.py`：偏好优化探索。
- `script/train_grpo.py`：GRPO/RL rescue 探索。

### 2.3 推理层（效果落地）

- `script/infer.py`：多策略解码（beam / sampling_mbr / hybrid_mbr / multi_model_mbr）。
- `script/merge_models.py`：Model Soup 权重平均融合。

---

## 3. 进化链路（你面试时最该讲的主线）

下面是这套系统的“演化顺序”。你可以把它理解为五个阶段。

## 阶段 A：把“脏数据”变“可学数据”

### 做了什么

1. **统一字符与缺损标记**
   - 在 `prepare_data.py` 里把各种缺损表达统一成 `<gap>`；
   - 保留关键转写字符（如 š/ṭ/ḫ），去除无关 OCR 噪声。

2. **修复 translation 端格式污染**
   - 行号、跨行断词、编辑注释、残留符号等做规则化处理。

3. **构建高精度 hint 词典**
   - 从词典与训练对齐中挖名字映射，但做支持度与优势比过滤，保证 precision。

### 为什么提升

- BLEU/chrF 对字符和 n-gram 敏感；垃圾字符会直接拉低分数。
- `<gap>` 语义必须稳定，否则模型会学到冲突信号。
- 低资源场景里，错误标签比少标签更致命。

---

## 阶段 B：用 Qwen 做“高质量伪标签引擎”

对应脚本：`script/qwen_align_sentences.py`

### 做了什么

- 把长块文本切成句对，但不是盲信 LLM 输出；做了多层硬规则：
  - `<gap>` 数量不能减少；
  - 数字不能幻觉；
  - 变音符签名相似度约束；
  - transliteration 必须高度覆盖源文本；
  - 边界悬挂检测（防止把半句切成两句）。
- 通过 `force_source_transliteration` 只信 LLM 的“切点”，不信它改写原文。

### 为什么提升

- 你们真正要的是“可监督信号”，不是“看起来流畅的重写”。
- 这一步把 LLM 变成**数据对齐工具**，而不是直接当翻译模型，风险更可控。

---

## 阶段 C：CPT（持续预训练）让模型先“懂这种语言形态”

对应脚本：`script/build_cpt_bdlm_data.py` + `script/train_cpt_bdlm.py`

### 做了什么

- 使用 ByT5（字节级）做 BDLM span corruption，特别适合稀有字符和变体拼写。
- 利用 OA Lexicon + eBL Dictionary 给被 mask 片段加词义注解（可控比例）。
- 关键：CPT 数据里混入一定比例 SFT 翻译样本（脚本默认 50%），减少任务漂移。
- 训练时监控 translation probes + dictionary probes，避免“只会填空不会翻译”。

### 为什么提升

- 先学“古语分布”，再学“翻译映射”，比直接端到端 SFT 更稳。
- 字节级建模天然避免 BPE/OOV 问题，对死语言极其友好。

---

## 阶段 D：SFT 主干打分（真正冲榜核心）

对应脚本：`script/train.py`

### 关键机制

- **Meta Prefix**（如 `OAOI/OAOD`）：
  把“官方/噪声、完整/残损”等元信息编码进输入前缀，让模型区分样本域。

- **R-Drop 一致性约束**：
  同样本两次 dropout 输出做一致性，缓解低资源过拟合。

- **Coverage loss**：
  压制漏译和过短输出（这个任务里“欠翻”很常见）。

- **Group-aware split（按 `oare_id`）**：
  避免同文档片段泄漏到验证集，防止虚高。

- **多解码边界评估**：
  训练中比较 greedy/beam/sample_mbr/hybrid_mbr，不被单一 decode 偶然性误导。

### 为什么提升

- 这些改动全部针对比赛痛点：噪声域差异、欠翻、验证泄漏、解码不稳定。
- 从代码看，后期主力 checkpoint 在 `output/model_20260321_171137`。

---

## 阶段 E：推理工程化（最后几分的来源）

对应脚本：`script/infer.py` + `script/merge_models.py`

### 做了什么

1. **Model Soup**
   - 对多个同架构 checkpoint 做权重平均，降低方差，提高稳定性。

2. **Hybrid MBR**
   - 候选池 = Beam 候选 + Sampling 候选；
   - 用 chrF/BLEU/Jaccard 加权共识选最终输出。

3. **异步流水并行**
   - 生成、后处理、MBR 选择分阶段异步化，提高吞吐。

### 为什么提升

- 低资源下单次解码噪声大，MBR 的“群体共识”通常比单 best 更稳。
- Soup + MBR 的组合常常是 leaderboard 上最稳健的工程技巧。

---

## 4. 探索支线（面试要讲“做过但不吹”）

代码里有 DPO/GRPO：

- `script/train_dpo.py`
- `script/train_preference.py`
- `script/train_grpo.py`
- 输出目录里有 `dpo_* / grpo_* / grpo_rescue_*`

这些更像“后期探索/救火实验”。从现有日志与目录结构看，主提交强项仍然是
**数据工程 + CPT + SFT + 推理集成**。

面试建议说法：

- “我们系统性探索了 DPO/GRPO，但最终主分来自 SFT 主线和推理集成；RL 结论是对低资源翻译任务收益不稳定，需要更高质量偏好数据与更强 reward 设计。”

这样很专业，也诚实。

---

## 5. 你可以怎么讲你的贡献（小白可用版）

你不需要假装“算法全是我发明的”，你要做的是把**系统思维**讲清楚。

推荐口径（可直接背）：

1. “我先把任务拆成三层：数据质量、模型训练、推理稳定性。”
2. “在数据层，我们把 `<gap>`、数字、变音符当作硬约束，优先确保监督信号可信。”
3. “在训练层，我们采用 ByT5 的 CPT+SFT 两阶段，并用 R-Drop/coverage 解决低资源欠翻与过拟合。”
4. “在推理层，我们用 Model Soup + Hybrid MBR，把单模型随机性降下来，最终拿到更稳的线上分数。”
5. “我能解释每个模块解决的失败模式，而不是只会报模型名。”

---

## 6. 面试高频追问与标准回答

### Q1: 为什么不用纯 LLM 端到端翻译？

A: 数据量太小且噪声大，端到端极易幻觉和过拟合。先做强约束数据构建，再做小模型精调，整体更可控，离线评估也更稳定。

### Q2: 为什么用 ByT5 而不是常规 subword 模型？

A: 古语言转写符号和拼写变体多，字节级对 OOV 更鲁棒，不依赖现代语料 BPE 词表。

### Q3: MBR 为什么有效？

A: 低资源下不同候选常互补。MBR 通过候选间相互评分选“共识句”，显著减少偶然错误。

### Q4: 你们怎么防止伪标签把模型带偏？

A: 伪标签阶段加硬约束（gap/数字/变音符/覆盖率）并保守过滤，宁可少留，不留错样本。

### Q5: 你个人最大的成长点是什么？

A: 从“调参”转向“诊断失败模式 -> 设计可验证修复 -> 观察指标闭环”的工程化思维。

---

## 7. 一张图记住整套系统

`raw data -> strict cleaning -> qwen alignment + hard validation -> CPT(BDLM + dict aug + SFT mix) -> SFT(R-Drop/coverage/meta prefix) -> model soup -> hybrid MBR -> submission`

---

## 8. 给你的实战建议（最后一公里）

1. 把这份 README 每一节压缩成 1 分钟口述，总时长控制在 6~8 分钟。
2. 准备 2 个“失败案例 -> 修复策略”的具体故事（面试官最爱问）。
3. 强调你会用代码和指标做闭环，而不只是“我调了很多参数”。
4. 面试时别抢“核心算法作者”定位，改成“核心 pipeline 共建者/训练与推理工程负责人之一”，更稳。

---

## 9. 参考脚本索引（方便你临时翻代码）

- 数据清洗与预处理：`script/prepare_data.py`
- 句对齐与伪标签：`script/qwen_align_sentences.py`
- CPT 数据构建：`script/build_cpt_bdlm_data.py`
- CPT 训练：`script/train_cpt_bdlm.py`
- SFT 主训练：`script/train.py`
- 推理：`script/infer.py`
- 模型融合：`script/merge_models.py`
- DPO 数据：`script/build_dpo_pairs.py`
- DPO/GRPO 探索：`script/train_dpo.py`、`script/train_grpo.py`

---

如果你愿意，我下一步可以继续帮你做两件事：

1. 基于这份 README 给你写一个**中文 3 分钟面试口播稿**（技术面版本）；  
2. 再写一个**简历条目精简版**（两条 bullet，保证“能讲出来”）。

