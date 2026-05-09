# 大模型算法工程师面试 Q&A

---

## 一、Deep Past 项目

---

**Q: 你为什么选 ByT5 而不是基于 BPE 的模型（如 mBART、NLLB）来做阿卡德语翻译？**

A: 核心原因是词表问题。BPE 模型的 tokenizer 在预训练时就固定了词表，对阿卡德语音译中的特殊符号（š/ṭ/ḫ 及各类变音符）几乎全部退化为 [UNK] 或逐字符切分，导致 embedding 层没有有效表示。ByT5 是 byte-level 模型，直接以 UTF-8 字节序列为输入，理论上对任意字符集都有完整表示能力，不存在 OOV 问题。

延伸知识点（Transformer tokenization）： BPE 的本质是基于频率的贪婪合并，在低资源死语言上稀有 n-gram 永远不会被合并，所有罕见字符都以单字节粒度输入，但这些单字节对应的 embedding 几乎没有被预训练充分，梯度信号很弱。ByT5 的解法是从一开始就放弃 subword，用 256 维 byte vocab 覆盖所有 Unicode，代价是序列更长（通常 3-5 倍），attention 计算量 O(n²) 显著上升，但对稀有字符任务反而是正确 tradeoff。

**Q: BDLM Span Corruption 和标准 T5 Span Corruption 有什么区别？**

A: 先理解标准 T5 的 Span Corruption：把一段连续的 subword token 遮住，用一个特殊占位符（`<extra_id_0>`）替代，训练目标是让 decoder 还原被遮住的内容。例如：

```
输入（encoder）: "The <extra_id_0> sat on the mat"
目标（decoder）: "<extra_id_0> cat <extra_id_1>"
```

ByT5 是 byte-level 模型，输入序列是 UTF-8 字节，所以 Span Corruption 操作的粒度也变成了字节——遮住的是一段连续字节而不是 subword。

BDLM（Byte-level Denoising LM）就是指这种以字节为粒度的去噪预训练目标，它天然适合 OCR 噪声修复，因为 OCR 的错误往往就在字节级别：比如阿卡德语的 `š`（两个字节）被 OCR 识别成 `s`（一个字节），在字节序列上就是一个局部替换，正好是 Span Corruption 要学会修复的那类扰动。

> **延伸（T5 vs BERT 的信息流差异）：** BERT 的 MLM 是 encoder-only，双向注意力可以同时看到左右上下文，直接"填空"。T5 的 decoder 是单向的（causal attention），只能看到已生成的部分，必须从 encoder 的上下文表示里"推断"缺失内容，对生成任务泛化性更好。这是 T5 在翻译、摘要等任务上比 BERT 类模型更强的根本原因。

**Q: 混合 50% SFT 翻译数据做 CPT 冷启动的目的是什么？**

A: 防止 catastrophic forgetting。纯文本 CPT 会让模型偏离翻译任务的 encoder-decoder 对齐能力。混合一定比例的翻译对，相当于在持续预训练时保留"任务锚点"，让模型在适应新领域分布的同时不忘记如何做翻译。50% 是经验值，过少则 CPT 后翻译能力退化，过多则 CPT 的域适应效果打折扣。

延伸知识点（灾难性遗忘与 Transformer）： 灾难性遗忘的根本原因是神经网络权重是共享的——新任务的梯度更新会覆盖旧任务学到的参数方向。在 Transformer 中，FFN 层（约占模型参数量 2/3）被认为是"知识存储"的主要载体（Geva et al. 2021），而 Attention 层负责"检索"。CPT 时 FFN 的权重会被大幅更新，因此混合旧任务数据是最直接的缓解方法。更高级的方法包括 EWC（弹性权重固化，对重要参数加惩罚项）或 LoRA（冻结主干只训练低秩矩阵，从架构上限制更新范围）。

**Q: R-Drop 在 Seq2Seq 上怎么用？**

A: R-Drop 的思路是对同一个输入做两次独立 forward（两次 dropout mask 不同），然后最小化两次输出分布之间的双向 KL 散度，作为额外的一致性正则项叠加在标准 CE loss 上：

```
L = 0.5 * (CE(p1, y) + CE(p2, y)) + α * KL(p1 || p2)
```

两次 forward 方向是对称的（双向 KL），让模型在不同 dropout 噪声下输出尽量一致，提高了表示的鲁棒性。在低资源场景下防止过拟合效果明显。

延伸知识点（Dropout 与 Transformer）： Transformer 中 dropout 通常加在 attention weight 之后和 FFN 激活之后。它本质上是在训练时对子网络做 ensemble，推理时关闭 dropout 相当于用期望网络。R-Drop 把这个隐式 ensemble 显式化——不只是训练时的随机正则，而是直接优化不同子网络输出的一致性。在极低资源（<10k 样本）时效果尤为突出，因为此时过拟合压力大，dropout 的随机性让两次 forward 差异明显，KL 惩罚能有效压制。

**Q: EMA 权重平滑在训练中怎么用的？**

A: EMA（Exponential Moving Average）在每个训练步后维护一份"影子权重"：

```
θ_ema = β * θ_ema + (1 - β) * θ_current
```

β 通常取 0.999。评估和推理时用 θ_ema 而非 θ_current。EMA 权重相当于对历史多个 checkpoint 的平滑集成，能有效过滤训练后期的梯度噪声，在低资源场景下比直接使用最终 checkpoint 的泛化性更好。

延伸知识点： EMA 和 Model Soup 都是权重空间 ensemble 的思路，区别在于：EMA 是时间维度的指数加权平均（同一训练轨迹上的不同步骤），Model Soup 是空间维度的算术平均（不同 checkpoint 或不同超参 run 的权重直接平均）。理论基础都来自 loss landscape 的平坦性假设——在 flat minima 附近，权重空间的线性插值仍在低 loss 区域（Garipov et al., 2018）。对于 Transformer，这个假设在大量实验中被验证，尤其是同一预训练权重不同微调路径之间的插值效果良好。

**Q: Meta Prefix 条件生成怎么实现的，体裁 token 怎么定义？**

A: Meta Prefix 是在输入序列前拼接一段元信息 token，显式告知模型当前翻译的文本体裁（如学术文献、商业信函、法律文书等），让模型在生成时条件化到特定风格。实现方式是在 tokenizer vocab 里加入若干 special token（如 <genre:academic>、<genre:commercial>），作为 source 序列的前缀。训练时在对应体裁数据上加上这些 prefix，推理时根据输入文本类型选择对应 prefix。

延伸知识点（Prompt / Prefix Tuning）： Meta Prefix 是 hard prefix（离散 token）的一种，另一种是 soft prefix（可学习的连续 embedding，不对应真实 token）。Prefix Tuning（Li & Liang, 2021）在 Transformer 的每层 attention K/V 前拼接可训练向量，参数量极少但能有效引导生成风格。两者的区别在于 hard prefix 的梯度流只通过 embedding lookup，而 soft prefix 的梯度直接更新 prefix embedding，表达能力更强但可解释性更差。

**Q: Hybrid MBR 里为什么用 chrF++ 和 BLEU 的几何平均而不是单一指标？**

A: chrF++ 和 BLEU 捕捉的信息互补：BLEU 是 n-gram precision（精确率导向，惩罚幻觉），chrF++ 是字符级 F-score（recall 和 precision 均衡，对形态变化更鲁棒）。几何平均 sqrt(chrF++ × BLEU) 要求两个指标都不能太差——如果一个候选 BLEU 很高但 chrF 很低（通常意味着输出很短、词选对了但字符形态错），几何平均会显著惩罚。阿卡德语有丰富的词形变化（词尾格变化），字符级指标比纯 n-gram 指标更敏感，所以 chrF 权重应当更高。

延伸知识点： MBR 解码的理论基础是贝叶斯决策理论——不选 argmax P(y|x)，而选 argmin E_{y'~P(y|x)}[L(y, y')]，即最小化期望损失（等价于最大化期望 utility）。在 NMT 里，候选集近似为从模型采样的 N 条输出，utility function 用 chrF/BLEU 这类自动指标代替真实人工评估。MBR 比 beam search 更鲁棒的直觉是：beam search 追求模型概率最大（可能是短、安全、通用的输出），MBR 追求与"大多数合理翻译"最接近（更倾向于信息完整的输出）。

## 二、远望谷项目

---

**Q: SimQAAgent 的逆向任务"受控退化"是怎么保证可控的？**

A: 逆向任务是给定"好的"原始论文段落，要求 Agent 生成一个"有特定缺陷"的退化版本作为 question，原始文本作为 answer（写作改进任务）。可控性通过两层保证：第一层是 Agent 的 system prompt 明确规定退化类型（逻辑跳跃、语气不当、结构混乱等），让模型生成定向错误而非随机损坏；第二层是 Quality Judge Agent 会检验退化版本是否真的有缺陷、原文是否真的是更好的版本，不符合条件的样本被过滤。这样产出的训练对比随机噪声注入有更明确的语言学意义，模型能学到"什么叫好的学术写作"。

**Q: DPO 的 loss 推导假设是什么？和 RLHF 等价的前提是什么？**

A: DPO 的推导起点是 RLHF 的目标：

```
max_π E[r(x,y)] - β * KL(π || π_ref)
```

这个目标有闭式解：`π*(y|x) ∝ π_ref(y|x) * exp(r(x,y)/β)`。反推得到奖励的隐式表示：

```
r(x,y) = β * log(π(y|x)/π_ref(y|x)) + β * log Z(x)
```

把这个代入 Bradley-Terry 偏好模型（人类偏好 y_w > y_l 的概率正比于奖励差的 sigmoid），Z(x) 抵消，得到 DPO 的 loss：

```
L_DPO = -E[log σ(β * log(π(y_w|x)/π_ref(y_w|x)) - β * log(π(y_l|x)/π_ref(y_l|x)))]
```

**等价前提：** 假设偏好数据来自同一策略（或接近策略分布），且偏好服从 Bradley-Terry 模型。实际中如果 chosen/rejected 都来自 π_ref（off-policy），则分布偏移可能导致 DPO 退化。SimPO、IPO 等变体正是为了解决这个问题。

**Q: GRPO 和 PPO 的核心区别是什么？为什么 GRPO 去掉了 critic？**

A: PPO 需要一个额外的 critic 网络估计 value function（V(s)），用来计算 advantage = r - V(s)，引导策略更新方向。GRPO（Group Relative Policy Optimization）的核心改动是：对同一个 prompt 采样一组输出（group），用组内的相对奖励排名代替绝对 value 估计，advantage 直接由组内归一化的奖励计算：

```
A_i = (r_i - mean(r_group)) / std(r_group)
```

**好处：** 去掉了 critic 模型（省一个同量级模型的显存和计算），对奖励函数稳定性要求更低，组内对比使得 advantage 信号天然去偏，适合 verifiable reward（如 SQL 执行结果）。**代价：** 需要每个 prompt 采样多个输出（batch size 膨胀），对采样多样性要求更高——如果同一 prompt 下所有输出质量相近，标准差趋近 0，梯度信号消失。CHORD 在此基础上加了 off-policy SFT 辅助目标，缓解了 on-policy 探索不足的问题。

**Q: 你的 SQL 奖励函数是怎么设计的？执行正确率从 70% 提升到 84% 是在什么数据集上测的？**

A: 奖励函数区分三档：SQL 执行结果与标准答案完全匹配得 1.0，SQL 可执行但结果错误得 0.1，SQL 不可执行得 0.0，另外叠加格式合规奖励（是否有 <think> 标签等）。这种阶梯式设计比二值奖励好处在于给了"方向正确但结果不对"的中间反馈，减少 GRPO 初期的稀疏奖励问题。验证集是业务内部的图书馆采编 SQL 测试集（非公开基准）。

延伸知识点（奖励稀疏性与 RL）： 稀疏奖励是 RL 在 LLM 上面临的主要挑战——如果大多数采样输出奖励为 0，GRPO 组内方差为 0，梯度消失。常见解法：reward shaping（引入 partial reward，如这里的 0.1 档）、curriculum（从简单 SQL 开始逐步提升难度）、auxiliary SFT loss（CHORD 的思路）。

## 三、通用基础

---

**Q: LoRA 的 rank 选多少，target modules 怎么选？**

A: rank 通常在 8-64 之间，低资源任务 rank=8 或 16 足够，large 以上模型且任务复杂时可以到 64。target modules 经验上：Q/V 矩阵是最重要的（注意力的查询和值），K 矩阵次之，FFN 的 up/down projection 加上后效果通常再提升一些。一般先试 Q+V，效果不够再加 K 和 FFN。

原理： LoRA 的假设是微调时的权重增量 ΔW 是低秩的（ΔW ≈ BA，B ∈ R^{d×r}, A ∈ R^{r×k}），r << min(d,k)。Transformer 的 attention 矩阵 W_Q/W_K/W_V 在预训练后已经具备很强的语义能力，微调只需要在低维子空间内调整方向，低秩假设成立。FFN 矩阵维度更大（4d × d），低秩假设相对弱一些，但参数量占比大，加上后整体效果通常还是正向。

**Q: DeepSpeed ZeRO Stage 2 和 Stage 3 的区别？**

A: ZeRO 把三类状态分片：optimizer states（Stage 1）、gradients（Stage 2）、model parameters（Stage 3）。

- **Stage 2**：在所有卡上分片 optimizer states 和 gradients，但每张卡都保留完整的模型参数。forward/backward 正常计算，reduce-scatter 后每卡只持有本卡负责的梯度分片，参数更新后 all-gather 同步。
- **Stage 3**：进一步分片模型参数，每张卡只持有部分层的参数。forward 时需要 all-gather 取到当前层完整参数，计算后立即释放（offload），显存峰值大幅降低，代价是通信量增加约 1.5 倍，且对网络带宽要求更高。

**怎么选：** 单机多卡 NVLink 高带宽用 Stage 2 更快，跨节点 InfiniBand 或单卡显存瓶颈严重时用 Stage 3。32B 模型 4 卡 A100 80G 通常 Stage 2 放不下，需要 Stage 3 或结合 CPU offload。

**Q: vLLM 的 PagedAttention 解决了什么问题？**

A: 传统推理框架对每个请求预分配连续的 KV cache 显存（按 max_length 预留），导致：1）显存碎片化严重——实际输出比 max_length 短时预留空间浪费；2）无法动态扩展——请求之间显存不能共享。

PagedAttention 借鉴 OS 的虚拟内存分页思想，把 KV cache 拆成固定大小的 block（page），用 block table 映射逻辑 KV 位置到物理显存地址。不同请求的 KV page 可以非连续存储，prefix sharing 场景（多个请求共享相同 prompt 前缀）可以直接引用同一物理 page，显存利用率提升显著（官方报告吞吐量提升 2-4 倍）。

延伸知识点： KV cache 的大小 = 2 × num_layers × num_heads × head_dim × seq_len × batch_size × dtype_bytes。对 Qwen3-VL-32B（64 层，128 heads，head_dim=128）来说，bs=1、seq_len=4096 的 KV cache 约 4GB，PagedAttention 的动态分配使得多请求并发时显存不被单个长请求独占。