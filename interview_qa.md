# 面试问题与参考答案

> 基于李少博简历，以项目为切入点，由项目问题引出底层原理（八股）

---

## 模块一：数据工程 & 数据对齐

### Q1：三层验证体系中，`<gap>` 守恒校验、悬挂边界检测、覆盖率过滤各解决什么问题？误报率如何量化？

**参考答案：**

- **`<gap>` 守恒校验**：古亚述楔形文字转写中，`<gap>` 标记表示泥板破损缺失片段。对齐时要求源语言与目标语言的 `<gap>` 数量一致，若不一致则说明句子边界切分错误或对齐偏移，直接过滤。
- **悬挂边界检测**：检测切分后句子是否以不完整的语法单元结尾（如句子截断在一个词的中间、或括号未闭合），通过正则 + 简单语法规则实现。
- **覆盖率阈值过滤**：统计源-目标句对的词汇覆盖率（即已知词典词汇在句子中的占比），低于阈值的句对视为噪声丢弃。
- **误报率量化**：人工抽样 200~500 条验证集，对过滤器输出做 precision/recall 统计；也可以观察过滤后训练集 BLEU 是否正向提升作为代理指标。

---

### Q2：LoRA SFT 三任务数据比例（翻译 40%、OCR 纠错 30%、切分 30%）如何确定？是否做消融？

**参考答案：**

- 初始比例基于任务难度与数据量的经验估计：翻译是核心任务，权重最高；OCR 纠错和句子切分是辅助任务，比例相对均等。
- 做了简单消融：分别测试仅翻译 / 翻译+纠错 / 三任务混合，观察下游句子对齐质量（以覆盖率和人工抽样评估），最终三任务混合效果最佳。
- 实践中比例调整也受到数据量约束——三类数据生产成本不同，纠错对和切分对可用规则批量生成，翻译对需要人工或模型审核，因此比例也部分由数据可用量决定。

---

### ⬇ 引出八股：LoRA 原理

**Q：LoRA 的原理是什么？为什么低秩分解有效？rank 怎么选？**

**参考答案：**

- **原理**：LoRA（Low-Rank Adaptation）冻结预训练权重 $W_0$，在旁路插入低秩矩阵 $\Delta W = BA$，其中 $B \in \mathbb{R}^{d \times r}$，$A \in \mathbb{R}^{r \times k}$，$r \ll \min(d,k)$。前向传播变为 $h = W_0 x + \frac{\alpha}{r} BAx$。
- **为什么有效**：研究发现预训练模型在下游任务上的权重更新矩阵本质上是低秩的（intrinsic dimensionality 假设），即有效的更新集中在少数几个方向上，全量更新大量参数是冗余的。
- **rank 选择**：通常从 r=4/8/16/32 开始实验；任务越复杂、领域偏移越大，r 需要越大；可通过 SVD 分析权重更新矩阵的奇异值分布来估计合适的 rank。
- **vs full fine-tuning**：LoRA 参数量仅为全量的 0.1%~1%，显存占用低，训练速度快，但对极端领域偏移场景表达能力有上限。

**Q：QLoRA 和 LoRA 的区别？NF4 量化原理？**

**参考答案：**

- **QLoRA**：在 LoRA 基础上，将冻结的基座模型权重量化为 4-bit（NF4），只有 LoRA adapter 保持 bf16/fp16 精度训练，大幅降低显存（70B 模型可在单卡 48GB 上训练）。
- **NF4（Normal Float 4）**：基于正态分布的非均匀量化。预训练权重近似服从正态分布，NF4 将量化区间按正态分布的等概率分位点划分，使量化误差在统计意义上最小，相比均匀量化（INT4）精度损失更小。
- **double quantization**：对量化常数本身再做量化，进一步压缩显存。

---

## 模块二：持续预训练（CPT）

### Q3：为什么选 ByT5（character-level）而不是 BPE token-level 模型？

**参考答案：**

- **死语言无 BPE 词表**：古亚述语没有现代语料，无法训练合适的 BPE tokenizer，强行用已有词表会把大量词切成 `<unk>` 或字节序列，信息严重损失。
- **ByT5 天然适配**：ByT5 直接在 UTF-8 字节上操作，无需词表，对任何语言的拉丁转写符号（š/ṭ/ḫ）都能精确表示，不存在 OOV 问题。
- **代价**：字节序列比 token 序列长 3~5 倍，attention 计算量更大，但在低资源场景下这个 trade-off 值得。

---

### Q4：BDLM Span Corruption 如何设计？与标准 T5 Span Corruption 有何不同？

**参考答案：**

- **标准 T5**：随机 mask 15% token，将连续 span 替换为单个哨兵 token，模型预测被 mask 的 span。
- **BDLM（Bilingual Dictionary-based Language Model）**：来自论文 arxiv:2103.07040，核心是在标准 T5 Span Corruption 的 **target 侧注入双语词典翻译**，而不是改变 mask 策略本身。
  - **标准 T5 target**：sentinel + 原始被 mask 的 token 序列。
  - **BDLM 增强 target**：以 50% 概率，将被 mask span 中的词附上词典释义注释，如 `ma-na [=mina] ṣa-ru-pá-am`，注释格式 `[={eng}]`，跳过停用词（to/and/if 等）。
  - **效果**：模型在预训练重建 span 时被迫学习"阿卡德语词 → 英语释义"的对齐关系，无需平行语料即可注入双语知识。
  - 混合 50% SFT 翻译数据做冷启动，避免纯 span corruption 预训练后切换到 seq2seq 任务时出现任务分布跳变。

---

### ⬇ 引出八股：Transformer 架构

**Q：Encoder-Decoder vs Decoder-only，各适合什么任务？**

**参考答案：**

- **Encoder-Decoder（T5/BART/ByT5）**：适合有明确输入-输出映射的任务（翻译、摘要、问答）；encoder 双向注意力对输入充分理解，decoder 自回归生成输出。
- **Decoder-only（GPT/LLaMA/Qwen）**：适合开放生成、指令跟随；统一的 causal attention，prompt+completion 一体化，扩展性强，当前 LLM 主流架构。
- **关键区别**：Encoder-Decoder 有 cross-attention 桥接两端，参数量在相同规模下分配给编解码各一半；Decoder-only 所有参数集中在解码，in-context learning 能力更强。

**Q：Attention 时间复杂度为何是 O(n²)？有哪些改进方案？**

**参考答案：**

- **O(n²) 原因**：计算 $QK^T$ 时，每个 query 需要与所有 n 个 key 做点积，共 n×n 次操作；显存存储 attention matrix 也是 O(n²)。
- **改进方案**：
  - **FlashAttention**：分块计算，不显式存储完整 attention matrix，显存 O(n)，速度提升 2~4x，但时间复杂度仍 O(n²)。
  - **Linear Attention（Performer/RWKV）**：用核函数近似 softmax，时间复杂度降至 O(n)，但精度有损。
  - **Sparse Attention（Longformer/BigBird）**：只计算局部窗口 + 少量全局 token，O(n·w)。
  - **MLA（DeepSeek）**：低秩压缩 KV cache，减少显存而非计算量。

---

## 模块三：SFT & 抗遗忘

### Q5：R-Drop 原理是什么？在哪个阶段引入？对 BLEU 提升多少？

**参考答案：**

- **R-Drop 原理**：对同一输入做两次前向传播（各自随机 dropout），要求两次输出的概率分布尽量一致（KL 散度约束）。最终 loss = 交叉熵 loss + α × KL(p1||p2)。这相当于一种数据增强 + 正则，迫使模型的表示更稳定、更不依赖特定神经元。
- **引入阶段**：在 SFT 微调阶段引入，CPT 阶段不用（CPT 数据量大，不需要额外正则）。
- **效果**：在古亚述语低资源场景下，BLEU 提升约 +1~2 点（具体数值视数据量和模型而定，需如实报告实验结果）。

---

### Q6：EMA 权重平滑和 Model Soup 有什么区别？

**参考答案：**

- **EMA（Exponential Moving Average）**：在**训练过程中**维护一份影子权重，每步更新 $\theta_{ema} = \beta \cdot \theta_{ema} + (1-\beta) \cdot \theta$，推理时使用 EMA 权重。抑制训练末期的权重震荡，类似于对训练轨迹做时间平滑。
- **Model Soup**：在**训练结束后**，对多个不同检查点（或不同超参数训练的模型）的权重做**平均**（uniform soup）或加权平均（greedy soup）。利用 loss landscape 平坦区域的特性，融合后的权重通常落在更好的泛化区域。
- **关键区别**：EMA 是训练中的在线平滑；Model Soup 是训练后的离线融合，可以跨不同运行融合。项目中两者都用：EMA 用于单次训练的稳定性，Model Soup 用于多检查点的最终集成。

---

### Q7：Meta Prefix 条件生成如何实现？Prefix 是可学习 token 还是 hard prompt？

**参考答案：**

- **实现方式**：采用可学习的 soft prefix（Prefix Tuning 思路），在 encoder 输入前拼接若干个连续向量，不同文本体裁（如行政文书体、商贸往来体、宗教文本）对应不同的 prefix 向量集合。
- **训练**：prefix 向量随任务联合训练，基座参数冻结或使用 LoRA。推理时根据输入文本的体裁标签选择对应 prefix。
- **vs hard prompt**：hard prompt 是离散 token（如 `[ADMIN]`、`[TRADE]`），梯度无法通过离散选择反传，优化困难；soft prefix 是连续向量，可端到端梯度优化，表达能力更强。

---

### ⬇ 引出八股：灾难性遗忘 & 归一化

**Q：灾难性遗忘的根本原因？主流解法有哪些？**

**参考答案：**

- **根本原因**：神经网络参数在新任务梯度更新时，会覆盖对旧任务重要的权重方向，因为 SGD 不区分参数对不同任务的重要性。
- **主流解法**：
  - **正则化**：EWC（Elastic Weight Consolidation）用 Fisher 信息矩阵估计参数重要性，对重要参数加惩罚项；R-Drop、EMA 也有类似效果。
  - **Replay**：混合旧任务数据（Experience Replay）或生成伪样本（Generative Replay）。
  - **结构隔离**：LoRA 冻结基座，只更新 adapter；不同任务用不同 LoRA 头。
  - **课程学习**：混合旧任务数据做 warmup，逐步过渡到新任务。

**Q：BN 和 LN 的区别？LLM 为什么用 LN？**

**参考答案：**

- **BN（Batch Normalization）**：对 batch 维度归一化，依赖 batch 统计量，训练和推理行为不同（需要维护移动均值/方差）；batch size 小时效果差；不适合序列变长场景。
- **LN（Layer Normalization）**：对单个样本的特征维度归一化，与 batch 无关，训练推理行为一致，天然适合变长序列和自回归生成。
- **LLM 选 LN 的原因**：自回归推理时 batch size=1，BN 完全失效；序列长度动态变化，LN 无额外约束；RMSNorm（LN 的简化版，去掉均值中心化）进一步减少计算量，LLaMA/Qwen 均采用 RMSNorm。

---

## 模块四：推理管线 & MBR

### Q8：Hybrid MBR 解码设计细节？为什么用 BLEU 和 chrF++ 几何平均？

**参考答案：**

- **MBR（Minimum Bayes Risk）原理**：从候选集 $\mathcal{H}$ 中选出期望 utility 最高的假设：$\hat{h} = \arg\max_{h \in \mathcal{H}} \sum_{h' \in \mathcal{H}} u(h, h')$，其中 $u$ 是 utility 函数（越高越好）。
- **Hybrid 候选**：Beam Search 生成高概率、确定性候选；Sampling 生成多样性候选。两路合并后做 MBR 选择，兼顾精度和多样性。
- **为何几何平均 BLEU + chrF++**：
  - BLEU 对短句精确率敏感，chrF++ 对字符级 n-gram 和召回率更鲁棒，古亚述语变音符多，chrF++ 能更好捕捉。
  - 几何平均比算术平均惩罚极端值，避免某一指标为 0 时整体被拉高。
- **复杂度**：候选集大小 N，MBR 需 O(N²) 次 utility 计算；实践中 N=20~50，可接受。

---

### Q9：双卡 DDP + 异步 MBR 流水线如何设计？

**参考答案：**

- **推理并行**：两张卡各自承担一半输入样本的解码（Beam + Sampling），通过 `torch.distributed` 或直接 vLLM 多 GPU 推理。
- **异步流水线**：Beam 解码速度快先完成，Sampling 解码慢，二者异步进行；MBR 计算在 Beam 结果就绪后立即开始，等 Sampling 结果陆续返回时做增量 MBR 更新，而不是等所有候选都生成完。
- **效果**：端到端吞吐提升约 30~40%（具体取决于 Beam/Sampling 速度比）。

---

### ⬇ 引出八股：分布式训练 & vLLM

**Q：DDP 和 FSDP 的区别？梯度同步在哪一步？all-reduce 通信量？**

**参考答案：**

- **DDP（DistributedDataParallel）**：每张卡保存完整模型副本，前向/反向各自独立，反向传播结束后对梯度做 **all-reduce**（求均值），再各自更新参数。
- **FSDP（Fully Sharded Data Parallel）**：将模型参数、梯度、优化器状态均分片到各卡，前向时 all-gather 拼回完整参数，反向后 reduce-scatter 聚合梯度分片，显存占用约为 DDP 的 1/N。
- **梯度同步时机**：DDP 在 `loss.backward()` 过程中，通过 bucket 机制与计算异步 overlap 地做 all-reduce，不是等反向完成后才同步。
- **all-reduce 通信量**：参数量为 P，共 N 卡，all-reduce 通信量 = $2P(N-1)/N \approx 2P$（ring all-reduce）。

**Q：vLLM 的 PagedAttention 原理？**

**参考答案：**

- **问题**：传统 KV Cache 为每个 request 预分配最大序列长度的连续显存，大量碎片化浪费。
- **PagedAttention**：借鉴操作系统虚拟内存分页思想，将 KV Cache 切成固定大小的 **block**（page），每个 sequence 的 KV Cache 由不连续的 block 组成，通过 **block table** 映射逻辑地址到物理地址。
- **优势**：显存碎片接近零，支持更大 batch size，吞吐提升 2~4x；同时支持 prefix sharing（相同 prompt 的 KV block 复用）。

---

## 模块五：GRPO & CHORD（图书馆项目）

### Q10：CHORD 算法如何将 SFT 重构为 RL 的 auxiliary objective？

**参考答案：**

- **标准 GRPO**：对一组采样回复按 reward 排名，用 group-relative advantage 做策略梯度更新，无 critic 网络。
- **CHORD 扩展**：SFT 数据和 GRPO 采样数据是**完全独立的两条数据流**，来自各自的 DataLoader，每个 step 分别取 batch，合并 loss 后一次反向传播：
  $$\mathcal{L}_{CHORD} = (1-\mu) \cdot \mathcal{L}_{GRPO} + \mu \cdot \mathcal{L}_{SFT}$$
- **CHORD-μ 动态加权**：μ 用 cosine 曲线从峰值（如 0.9）衰减到谷值（如 0.05），训练初期 SFT 主导稳定模型，后期 RL 主导——即"off/on-policy 过渡"。
- **CHORD-φ token 级加权（源码实现）**：不是对 SQL 关键字手动加权，而是按模型当前对每个专家 token 的预测概率**自适应**加权：
  $$\phi(y_t^*, \pi_\theta) = p_t \cdot (1 - p_t), \quad p_t = \pi_\theta(y_t^* \mid x, y_{<t}^*)$$
  - $p_t \approx 0.5$（模型不确定）→ φ 最大，重点学习；
  - $p_t \approx 0$ 或 $p_t \approx 1$（模型已确定）→ φ → 0，避免无效/破坏性更新；
  - 本质是**自适应课程学习**，自动聚焦于模型当前处于学习边界的 token。

---

### Q11：SQL 能力的 reward function 如何设计？

**参考答案：**

- **执行结果对比（首选）**：将模型生成的 SQL 在测试数据库上执行，与 ground truth SQL 的执行结果做集合比较（行级别 F1）。优点是绕过 SQL 语法等价性问题（不同写法可能等价）。
- **语法合法性 reward**：若 SQL 无法 parse/执行，给 -1 惩罚；合法但结果错误给 0；结果正确给 +1（或按 F1 连续奖励）。
- **格式 reward**：要求输出包含 `<sql>...</sql>` 标签，格式正确额外加分，避免 reward hacking（模型输出随机内容但恰好满足部分条件）。

---

### Q12：从 70% → 84% 的指标是什么？是否出现 reward hacking？

**参考答案：**

- **指标**：验证集上 SQL 执行准确率（Execution Accuracy，EX）或问答任务的精确匹配率（Exact Match），具体看任务类型。
- **Reward hacking 分析**：
  - 常见表现：模型学会输出固定模板触发格式 reward，但语义内容错误。
  - 检测方法：单独统计格式 reward 和内容 reward 的曲线，若格式 reward 快速饱和而内容 reward 停滞，说明 hacking 发生。
  - 应对：加入多维 reward（格式 + 执行结果 + 语言质量），提高 hacking 难度；或使用 Reward Model 而非规则 reward。

---

### ⬇ 引出八股：PPO/GRPO/DPO

**Q：PPO 和 GRPO 的区别？GRPO 为什么不需要 critic 网络？**

**参考答案：**

- **PPO**：需要 critic（value network）估计状态价值 V(s) 来计算 advantage = Q - V；critic 和 policy 参数量相当，额外占用大量显存和计算。
- **GRPO（Group Relative Policy Optimization）**：对同一问题采样 G 个回复，以 group 内 reward 的均值和方差做归一化计算 advantage：$A_i = \frac{r_i - \text{mean}(r)}{\text{std}(r)}$，用 group 统计量代替 critic，无需单独 value network。
- **优势**：显存减少约 50%，实现简单，对 LLM 的离散生成任务（reward 稀疏）更稳定。

**Q：DPO 的 loss 推导能讲一下吗？**

**参考答案：**

- **RLHF 的 KL 约束优化目标**：$\max_\pi \mathbb{E}[r(x,y)] - \beta \cdot KL(\pi || \pi_{ref})$
- **最优策略**的闭式解为：$\pi^*(y|x) \propto \pi_{ref}(y|x) \exp(r(x,y)/\beta)$
- 由此反推 reward：$r(x,y) = \beta \log \frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$
- 代入 Bradley-Terry 偏好模型（$p(y_w \succ y_l) = \sigma(r_w - r_l)$），$Z(x)$ 消去，得到 **DPO loss**：

$$\mathcal{L}_{DPO} = -\mathbb{E}\left[\log \sigma\left(\beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}\right)\right]$$

- 直觉：增大 chosen 回复相对 ref 的 log ratio，减小 rejected 回复的 log ratio，$\beta$ 控制偏离 ref 的幅度。

**Q：KL 散度和 JS 散度的区别？DPO 里 KL 的作用？**

**参考答案：**

- **KL 散度**：$KL(P||Q) = \sum P \log \frac{P}{Q}$，非对称，当 $Q(x)=0$ 而 $P(x)>0$ 时发散。
- **JS 散度**：$JS(P||Q) = \frac{1}{2}KL(P||M) + \frac{1}{2}KL(Q||M)$，$M=\frac{P+Q}{2}$，对称，有界 [0,1]，但梯度在分布不重叠时为零（GAN 训练不稳定的原因之一）。
- **DPO 里 KL 的作用**：约束 policy 不要偏离 ref model 太远（防止模型 collapse 或产生 reward hacking），$\beta$ 越大约束越强，生成越保守；$\beta$ 越小越激进地优化 reward。

---

## 模块六：多智能体 & RAG

### Q13：SimQAAgent 的 ReAct 三轮推理设计？QualityJudge 判断标准？

**参考答案：**

- **ReAct（Reasoning + Acting）三轮结构**：
  - Round 1：**Thought** - 分析论文片段，确定问题类型（事实性/推理性/写作改进）；**Action** - 调用论文检索工具获取相关段落。
  - Round 2：**Thought** - 结合检索结果生成候选 QA 对；**Action** - 调用答案回查工具验证答案可在原文中定位。
  - Round 3：**Thought** - 评估生成质量；**Action** - 输出最终 QA 对，或触发重生成（至多重试 3 次）。
- **为何固定三轮**：不是理论上的最优，而是工程权衡——推理成本与质量的 tradeoff。更多轮次会显著增加 token 消耗和延迟，实测三轮后质量提升边际递减。同时固定轮次便于 batch 并行和成本预估。
- **QualityJudge Agent**：基于 LLM scoring，从四个维度评分（每维 0-10 分）：
  - **忠实性**：答案是否有原文依据，不能靠模型自身知识编造；
  - **问题清晰度**：问题是否有歧义、是否能独立理解（不依赖上下文）；
  - **答案完整性**：答案是否完整覆盖问题，有无遗漏关键信息；
  - **学术语气**：是否符合学术写作规范，无口语化表达。
  - 总分 < 6 直接过滤，6-8 分进入 Refiner Agent 改写后重评，> 8 分直接通过。
- **DAG 结构与上下文传递**：整体为 DAG（有向无环图），节点是 Agent，边是数据流。QualityJudge 的反馈（含具体失分维度和改进建议）作为结构化 JSON 传递给上游 Generator Agent，而不是只传分数——Generator 拿到"忠实性不足"的具体反馈后，下一轮会显式在 prompt 中加入"请确保每句话都有原文出处"的约束。

**追问：DAG 里有没有环？如何防止死循环？**

- DAG 本身无环，但 QualityJudge → Generator 的重试形成了**有限次的循环**，通过 `max_retry=3` 硬截断。超出次数后强制输出当前最高分结果或标记为低质量跳过。

**追问：Agent 之间的上下文如何管理？**

- 每个 Agent 只接收"需要它完成任务的最小上下文"，而非全链路历史。Generator Agent 只看论文片段 + 检索结果 + Judge 反馈，不看前几轮的完整对话历史，避免 context 窗口爆炸。跨 Agent 的状态用结构化 Python dict 传递，不是拼字符串。

---

### Q14：BM25 + TF-IDF 双路 RAG 如何 merge？有没有做 Rerank？

**参考答案：**

- **双路 merge（RRF）**：两路各自返回 Top-K 文档，做 **Reciprocal Rank Fusion（RRF）** 合并：
  $$score(d) = \frac{1}{k + rank_{BM25}(d)} + \frac{1}{k + rank_{TF\text{-}IDF}(d)}$$
  k=60 是平滑常数，防止第 1 名得分过于悬殊，且让排名靠后的文档保留基本分。RRF 不依赖分数量纲，对异构检索系统天然友好。
- **双路互补的本质**：
  - **BM25 强于**：精确关键词匹配，短查询（术语查询），对停用词敏感度低；
  - **TF-IDF 强于**：在向量化预处理后的 dense 场景有一定优势；单纯词频统计，无饱和机制，对重复堆词的文档分更高（特定场景下反而有用）；
  - **双路失效的共同场景**：语义检索——用户问"怎么快速排序"，文档里写的是"时间复杂度 O(n log n) 的比较类算法"，两路都找不到，此时需要 dense embedding 第三路。
- **Rerank**：cross-encoder reranker（如 BGE-reranker）对 merge 后 Top-20 做精排，输出 Top-5 送入 LLM context。**bi-encoder（向量检索）vs cross-encoder（reranker）的区别**：bi-encoder 将 query 和 doc 分别编码为向量，做近似最近邻搜索，速度快但精度有损；cross-encoder 将 query+doc 拼接后过 BERT，能做细粒度的 token-level 交互，精度高但只适合对少量候选做精排（不能 index）。
- **k 值调参**：k=60 是经验默认值。k 越小，第 1 名和第 2 名得分差距越大，排名靠前的文档优势越明显；k 越大，各名次得分趋于平均，相当于更民主的 merge。业务中可以在验证集上用 NDCG 调参。

**追问：如果要加第三路 dense embedding，应该怎么 merge？**

- 同样用 RRF，三路各自独立检索，求和：$score(d) = \frac{1}{k+rank_{BM25}} + \frac{1}{k+rank_{TFIDF}} + \frac{1}{k+rank_{dense}}$。Dense 路用 FAISS/Milvus 做 ANN 检索，与稀疏检索完全解耦，merge 层无需改动。

---

### Q15：SimQAAgent 里的"逆向任务/受控退化"是怎么设计的？可控性如何保证？

**参考答案：**

- **背景**：除了正向的"论文片段 → 问答对"，还有一类逆向任务：给定一段"好的"原始学术文本，让 Agent 生成一个**有特定缺陷的退化版本**作为 question，原始文本作为 answer（对应写作改进训练场景，让模型学会"什么叫好的学术写作"）。
- **可控性两层保证**：
  - **第一层（生成侧）**：Generator Agent 的 system prompt 明确规定退化类型——必须是以下之一：逻辑跳跃（结论无前提）、语气不当（口语化/过度情绪化）、结构混乱（段落顺序错误或论证倒置）、信息冗余（重复表达同一意思）。禁止"随机损坏"（如随机删词），必须是有语言学意义的缺陷。
  - **第二层（Judge 侧）**：QualityJudge 单独验证两件事：① 退化版本**确实**有缺陷（而不是无意义的噪声）；② 原文**确实**是更好的版本（而不是两者质量相当）。两个条件都满足才通过，否则整对丢弃。
- **为什么比随机噪声注入好**：随机删词/替换词的"退化"文本没有语言学意义，模型无法从中学到"写作缺陷"的概念，只会学到"有噪声的文本 → 去噪"，本质是 denoising 而非写作改进。

**追问：退化类型如何平衡？会不会某类占比过高？**

- 在 prompt 里按类型轮换（round-robin）或加比例控制，类似多任务采样的数据配比，避免模型只学到一种退化模式。

---

### Q16：多智能体系统有哪些常见 failure mode？如何 debug？

**参考答案：**

- **常见 failure mode**：
  - **幻觉传染**：上游 Agent 输出的幻觉内容被下游 Agent 当作事实引用并扩展，形成幻觉链。根因是每个 Agent 相信上游输出，缺乏独立验证。
  - **格式不匹配**：Agent A 输出 JSON 格式略有变动（多一个字段），Agent B 解析失败、静默跳过或抛出异常，数据流中断。根因是 Agent 间接口约定不严格。
  - **上下文爆炸**：把所有历史对话都拼入 prompt，随轮次增加 context 超过窗口上限，模型截断导致"失忆"。
  - **死循环/无限重试**：Judge 永远给低分，Generator 永远在重试，没有截断机制。
  - **工具调用失败静默**：外部工具（检索 API）超时，Agent 拿到空结果但没有异常处理，继续生成毫无依据的答案。

- **debug 方法**：
  - **结构化日志**：每个 Agent 的输入/输出完整记录到结构化日志（含 Agent 名、轮次、耗时、token 数），异常时能快速回溯。
  - **单 Agent 单测**：每个 Agent 单独有 unit test，输入 mock 数据验证输出格式和内容边界，不依赖 end-to-end 测试发现问题。
  - **中间态可视化**：把 DAG 的每个节点输出持久化为文件，离线抽样人工检查，定位是哪个节点开始引入错误。
  - **Dry-run 模式**：用小批量（50条）先跑完整 pipeline，检查数据量、过滤率、每阶段质量分分布，再全量运行。

---

### Q17：多智能体系统中各 Agent 的 prompt 如何设计？如何避免幻觉？

**参考答案：**

- **Prompt 设计原则**：
  - **角色定义要精确**：不是"你是一个助手"，而是"你是一个专门从学术论文中提取信息的问答生成器，只能基于给定段落生成问答对"——限制角色边界降低幻觉风险。
  - **输出格式强约束**：用 JSON schema 或 XML tag 约束输出格式，配合后处理 validation，格式不合规直接重试而非修复。
  - **负样本示例**：在 few-shot 里加入"不应该这样做"的反例，比单纯正例更有效约束模型行为。
  - **最小上下文原则**：只把完成当前任务必要的信息传给 Agent，过多背景反而分散注意力（lost in the middle 问题）。

- **避免幻觉的具体手段**：
  - **引用约束**：Generator Agent 的 prompt 要求"每个答案句子后面必须标注 [原文第X段]"，Judge 在验证时逐条核查引用是否真实存在。
  - **拒绝选项**：给 Agent 显式的拒绝路径——"如果给定段落不足以回答，输出 `{\"skip\": true, \"reason\": \"...\"}`"，而不是强迫模型给答案。
  - **温度控制**：生成阶段 temperature=0.7 保证多样性，但 Judge 打分阶段 temperature=0 保证评判一致性。

**追问：system prompt 和 user prompt 有什么区别？**

- system prompt 设定 Agent 的角色、能力边界、输出格式约束，是不变的"规则"；user prompt 是每次请求的具体输入（论文片段 + 任务说明），是变化的"数据"。实践中把不变的约束放 system，把每次不同的输入放 user，能利用部分推理服务的 system prompt caching 降低成本。

---

### ⬇ 引出八股：RAG & BM25

**Q：RAG 和 long context 的 trade-off？**

**参考答案：**

| 维度 | RAG | Long Context |
|------|-----|-------------|
| 知识更新 | 实时更新知识库即可 | 需重新推理或 fine-tune |
| 噪声 | 检索噪声影响生成 | 上下文太长注意力分散（lost in the middle）|
| 计算成本 | 检索 O(1)，生成短 context | 长 context 推理 O(n²) |
| 适用场景 | 知识密集型、文档库大 | 需要全局理解、文档数量少 |

**Q：TF-IDF 和 BM25 公式是什么？IDF 为何要平滑？**

**参考答案：**

**TF-IDF：**

$$\text{TF-IDF}(t,d) = \underbrace{\frac{\text{count}(t,d)}{|d|}}_{\text{TF：词在本文的频率}} \times \underbrace{\log\frac{N}{df(t)}}_{\text{IDF：词在全局的稀缺度}}$$

- **TF**：这个词在当前文档里出现多少次（除以文档长度归一化）。
- **IDF**：$N$ 是语料总文档数，$df(t)$ 是含该词的文档数。越少文档包含这个词，IDF 越大，说明它越有区分度。"的/了/是"几乎出现在所有文档里 → $df(t) \approx N$ → IDF ≈ 0，自动被压制。
- **缺陷1（TF线性）**：词出现100次被认为比出现10次重要10倍，但边际效益应该递减——超过一定次数后再多出现意义不大。
- **缺陷2（忽略文档长度）**：同一个词出现5次，在100词短文里TF=0.05，在1000词长文里TF=0.005，长文档天然吃亏。

**BM25（修复上述两个缺陷）：**

$$BM25(q,d) = \sum_{t \in q} IDF(t) \cdot \frac{TF(t,d) \cdot (k_1+1)}{TF(t,d) + k_1 \cdot \left(1 - b + b \cdot \frac{|d|}{avgdl}\right)}$$

- **TF 饱和**：分母含 TF，使 TF 越大增长越慢（趋于饱和），解决缺陷1。$k_1 \in [1.2, 2.0]$ 控制饱和速度。
- **长度归一化**：$|d|/avgdl$ 是当前文档长度与平均长度之比，文档越长分母越大、得分被压低，解决缺陷2。$b \in [0,1]$ 控制归一化强度（$b=0$ 完全不归一化）。
- **IDF 平滑**：$IDF(t) = \log\frac{N - df(t) + 0.5}{df(t) + 0.5}$，加 0.5 是为了防止某词出现在所有文档时 $IDF=0$ 被完全忽略，加平滑后 IDF 保留最小正值。

---

## 模块七：工程 & 系统（Java 实习）

### Q15：Netty 连接泄漏排查，根因是什么？

**参考答案：**

- **常见根因**：
  1. Handler 中捕获了异常但未调用 `ctx.close()`，导致 Channel 对象一直存活在内存。
  2. `ByteBuf` 引用计数未正确释放（Netty 使用引用计数内存管理，`retain()` 后必须对应 `release()`）。
  3. 连接心跳超时逻辑未触发（`IdleStateHandler` 配置错误），死连接长期占用。
- **排查方式**：通过 `ChannelGroup` 统计活跃连接数；用 Netty 的 `ResourceLeakDetector`（设置 PARANOID 级别）定位 ByteBuf 泄漏；jmap + MAT 分析堆内存中 Channel 对象数量。

---

### Q16：gnet 和 Netty 的线程模型区别？

**参考答案：**

- **Netty**：Reactor 模型，Boss EventLoopGroup 负责 accept，Worker EventLoopGroup 负责 I/O 读写和 handler 处理，基于 Java NIO，JVM 上运行。
- **gnet（Go）**：同样是 Reactor 模型，但基于 Go 的 goroutine + epoll/kqueue，无 JVM overhead，GC 压力更小，适合追求极致 QPS 的网关/代理场景。gnet 的 event loop 直接绑定 OS 线程，避免 goroutine 调度开销。

---

### ⬇ 引出八股：NIO & 网络

**Q：NIO 和 BIO 的区别？Reactor 模式三种变体？**

**参考答案：**

- **BIO（Blocking I/O）**：每个连接对应一个线程，`read()` 阻塞直到数据到达，线程资源随连接数线性增长，C10K 问题无解。
- **NIO（Non-Blocking I/O）**：基于 Selector（epoll），单线程监听多个 Channel 事件，I/O 就绪时才处理，线程数与连接数解耦。
- **Reactor 三种变体**：
  1. **单 Reactor 单线程**：accept + I/O + 业务处理全在一个线程，简单但无法利用多核。
  2. **单 Reactor 多线程**：Reactor 线程只做 I/O，业务处理交给线程池，多核友好但 Reactor 成瓶颈。
  3. **主从 Reactor 多线程（Netty 默认）**：Main Reactor 只做 accept，Sub Reactor 池处理 I/O，业务线程池处理逻辑，最高性能。

**Q：TCP 粘包的原因和解决方案？Netty 里用哪个 Decoder？**

**参考答案：**

- **原因**：TCP 是字节流协议，不保留消息边界；发送方 Nagle 算法合并小包；接收方 read 不一定一次读完一个完整消息。
- **解决方案**：
  1. **固定长度**：每条消息固定 N 字节，`FixedLengthFrameDecoder`。
  2. **分隔符**：以特定字节（如 `\n`）分割，`DelimiterBasedFrameDecoder`。
  3. **长度字段**：消息头包含 body 长度字段，`LengthFieldBasedFrameDecoder`（最常用，IoT 协议多采用此方案）。
  4. **应用层协议**：HTTP/WebSocket 等自带边界定义。

---

## 压轴开放题

### Q17：ByT5 在古亚述语上的天花板在哪？下一步最值得做什么？

**参考答案（思路）：**

- **天花板**：ByT5 预训练数据中古亚述语极少，字节级别的 attention span 有限，对长距离语言结构（如动词结尾的 SOV 语序）捕捉能力弱；平行语料数量（约数千句对）是硬限制。
- **下一步建议**：
  1. **更大模型 + 更多预训练**：用包含更多古典语言（拉丁语、古希腊语）的多语言模型（如 mT5-XXL）做迁移，利用类型相似语言的知识。
  2. **词汇化表示**：结合楔形文字形态词典，引入词素级别（morpheme-level）tokenization，比纯字节更有语言学意义。
  3. **GRPO 对齐**：以翻译专家评分为 reward 做 RL 优化，超越纯 BLEU 优化的局限。
  4. **更多数据挖掘**：继续扩充学术 PDF 语料，探索 cross-lingual transfer（阿卡德语 → 古亚述语）。

---

### Q18：GRPO 训练中踩过什么坑？reward 快速收敛但泛化变差怎么处理？

**参考答案（思路）：**

- **常见坑**：
  1. **reward scale 不稳定**：不同任务 reward 量纲差异大，需做归一化或 clip。
  2. **KL 系数设置不当**：$\beta$ 太小导致模型偏离 ref 过多，输出退化（repetition、hallucination）；太大则 RL 没有足够的探索空间。
  3. **采样多样性不足**：temperature 太低时 G 个采样几乎一样，group advantage 方差为零，梯度消失。
- **reward 快速收敛但泛化差（过拟合 reward）**：
  - 分析：reward function 存在可被 exploit 的捷径（如格式 reward 被 hack）。
  - 解法：增加 reward 多样性（加 LLM judge reward 作为对抗项）；用 held-out 真实任务指标（EX/BLEU）监控而非只看 reward 曲线；适当增大 KL 惩罚。

---

*面试节奏建议：项目问题重点考察真实参与度（数字、细节、踩坑经历），八股考察理解深度而非背书，遇到答不上来的追问"你当时怎么解决的"比直接给答案更有诊断价值。*
