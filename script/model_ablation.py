"""
model_ablation.py — ByT5 Structural Ablation Module
====================================================================

两种针对 ByT5 纯字节流模型的结构化增强方案，均可独立或组合使用：

1. Boundary-aware Side Embedding
   在 Encoder 的 byte embedding 旁并行注入"角色标签" side embedding，
   使模型在第一层即可感知阿卡德语音译的符号学边界。

2. Latent Bottleneck (Stride-2 Conv)
   在 Encoder 第 k 层后插入步长卷积瓶颈，将序列长度减半。
   浅层在原始字节分辨率运行（音节级特征），深层在压缩空间运行
   （更高语义密度），根治长序列注意力稀释问题。

核心思想
--------
ByT5 的 256 字节编码对所有字符一视同仁，但阿卡德语音译中：
  - `-` 连接音节，`=` 连接词素  → PUNCTUATION
  - `{d}`, `(m)`, `[…]`       → BOUNDARY（限定词/修复标注）
  - `<gap>`, `x`, `...`       → DAMAGE（信息缺失，禁止幻觉）
  - 数字 `10`, 分数 `½ ⅔`     → NUMERIC（精确对齐）
Side embedding 让模型在注意力计算前就区分这些角色，
无需从数据中隐式学习符号学规则。

消融开关
--------
>>> from model_ablation import ABLATION_CFG, install_boundary_aware
>>> ABLATION_CFG.enable = True          # 总开关
>>> ABLATION_CFG.use_punctuation = True # 连字符/音节连接符
>>> ABLATION_CFG.use_boundary = True    # 限定词括号 {d}, 修复括号 [...]
>>> ABLATION_CFG.use_damage = True      # <gap> 及损坏标记
>>> ABLATION_CFG.use_numeric = True     # 数字和分数
>>> ABLATION_CFG.combine_mode = "add"   # "add" | "gate" | "scale"
>>> install_boundary_aware(model)
"""

from __future__ import annotations

import math
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ╔══════════════════════════════════════════════════════════════╗
# ║                   ABLATION CONFIGURATION                    ║
# ╚══════════════════════════════════════════════════════════════╝

@dataclass
class BoundaryAblationConfig:
    """Boundary-aware frontend 消融配置。所有开关可在 install 前自由修改。"""

    # ---- 总开关 ----
    enable: bool = True

    # ---- 角色开关（关闭时该类别退化为 ALPHA）----
    use_punctuation: bool = True   # -, =, ., :
    use_boundary: bool = True      # {, }, (, ), [, ]
    use_damage: bool = True        # <gap>, x-in-bracket, ...
    use_numeric: bool = True       # 0-9, ½, ⅓, ⅔ …

    # ---- 融合策略 ----
    combine_mode: str = "add"      # "add" | "gate" | "scale"

    # ---- 初始化 ----
    side_embed_init_std: float = 0.01   # side embedding 初始标准差（小值保护预训练权重）
    gate_init_bias: float = -2.0        # gate 模式初始偏置（sigmoid(-2)≈0.12，偏向保留原始 embed）
    scale_init: float = 0.0             # scale 模式初始缩放因子（0 = 启动时无贡献）

    # ---- 冻结 ----
    freeze_base_embed: bool = False     # 冻结原始 byte embedding（shared weights）
    freeze_side_embed: bool = False     # 冻结 side embedding（仅训练 gate/scale）

    # ---- Encoder-only ----
    patch_decoder: bool = False         # 是否也给 decoder 加 side embedding（默认仅 encoder）


# 全局单例，train.py 中直接 import 后修改即可
ABLATION_CFG = BoundaryAblationConfig()


@dataclass
class BottleneckConfig:
    """Latent Bottleneck 消融配置。"""

    # ---- 总开关 ----
    enable: bool = False

    # ---- 结构 ----
    bottleneck_after_layer: int = 4    # 在第 k 层之后插入（0-indexed）
    stride: int = 2                     # 压缩步长（2 = 序列减半）
    kernel_size: int = 2                # 卷积核大小
    use_residual: bool = True           # 均值池化残差连接
    use_layernorm: bool = True          # 压缩后 LayerNorm

    # ---- 初始化 ----
    init_std: float = 0.02              # 卷积核初始标准差

    # ---- 冻结 ----
    freeze_bottleneck: bool = False     # 冻结瓶颈层（仅传递梯度，不更新参数）


BOTTLENECK_CFG = BottleneckConfig()


@dataclass
class HybridAttentionConfig:
    """Local + Global 混合注意力消融配置。

    局部滑动窗口让普通字节只看附近 W 个 token（音节级聚合），
    全局锚点（由 role tag 标识的关键 token）拥有全序列视野，
    充当跨句信息集线器。零新参数——仅操控 attention mask。
    """

    # ---- 总开关 ----
    enable: bool = False

    # ---- 窗口 ----
    window_size: int = 15               # 局部窗口半径（单侧），覆盖 2W+1 个 token

    # ---- 全局锚点 ----
    global_role_ids: Tuple[int, ...] = (3, 4, 5)  # BOUNDARY, DAMAGE, NUMERIC
    global_every_n: int = 0             # 额外每隔 N 个 token 设一个全局锚点（0=禁用）
    global_pad_token: bool = False      # PAD token 是否也作为全局锚点（通常 False）


HYBRID_CFG = HybridAttentionConfig()


@dataclass
class CoverageConfig:
    """Coverage Loss 消融配置。

    在训练时从 Decoder cross-attention 权重中计算覆盖率惩罚，
    压制重复翻译（复读机）和漏翻，同时抑制 Transliteration 泄漏。
    零新参数——纯 Loss 正则项。
    """

    # ---- 总开关 ----
    enable: bool = False

    # ---- 权重 ----
    alpha: float = 0.1                  # coverage loss 权重
    warmup_steps: int = 500             # 线性预热步数

    # ---- 层选择 ----
    layer_index: int = -1               # 使用哪层 decoder cross-attn（-1=最后一层）


COVERAGE_CFG = CoverageConfig()


# ╔══════════════════════════════════════════════════════════════╗
# ║                     ROLE CONSTANTS                          ║
# ╚══════════════════════════════════════════════════════════════╝

ROLE_PAD         = 0   # padding / EOS / UNK / sentinel
ROLE_ALPHA       = 1   # 普通字母（a-z, A-Z, 非 ASCII 字母如 š ṭ）
ROLE_PUNCTUATION = 2   # 音节连接符 - = . :  空格
ROLE_BOUNDARY    = 3   # 限定词/修复括号 { } ( ) [ ]
ROLE_DAMAGE      = 4   # <gap>、括号内 x、省略号 ...
ROLE_NUMERIC     = 5   # 数字 0-9 和 Unicode 分数 ½ ⅓ ⅔ ¼ ¾

NUM_ROLES = 6
ROLE_NAMES = ["PAD", "ALPHA", "PUNCTUATION", "BOUNDARY", "DAMAGE", "NUMERIC"]


# ╔══════════════════════════════════════════════════════════════╗
# ║                  BYTE → ROLE LOOKUP TABLE                   ║
# ╚══════════════════════════════════════════════════════════════╝

# ByT5 token 编码: token_id = utf8_byte_value + 3
# token 0=pad, 1=eos, 2=unk, 3..258 对应 byte 0..255
BYT5_BYTE_OFFSET = 3

def _build_byte_role_lut(max_vocab: int = 384) -> np.ndarray:
    """构建 token_id → 默认 role 的静态查找表。

    仅处理单字节可判定的角色；多字节模式（<gap>, ..., 分数）
    由 _compute_roles_1d 的上下文扫描补充。
    """
    lut = np.full(max_vocab, ROLE_ALPHA, dtype=np.int64)

    # 特殊 token
    lut[0] = ROLE_PAD  # <pad>
    lut[1] = ROLE_PAD  # </s> (EOS)
    lut[2] = ROLE_PAD  # <unk>
    # sentinel tokens (100 extra ids in ByT5) → PAD
    for i in range(259, min(max_vocab, 384)):
        lut[i] = ROLE_PAD

    # 逐字节分类 ASCII 范围
    for byte_val in range(128):
        tid = byte_val + BYT5_BYTE_OFFSET
        ch = chr(byte_val)
        if ch in "-=.:":
            lut[tid] = ROLE_PUNCTUATION
        elif ch in "{}()[]":
            lut[tid] = ROLE_BOUNDARY
        elif ch.isdigit():
            lut[tid] = ROLE_NUMERIC
        elif ch == " ":
            lut[tid] = ROLE_PUNCTUATION  # 空格视为分隔符
        # < > 留给上下文扫描判定（可能是 <gap> 的一部分）
        # 其余 ASCII 字母/符号保持 ALPHA

    return lut


_BYTE_ROLE_LUT = _build_byte_role_lut()

# ---- 多字节模式的 token_id 序列 ----
# <gap>
_GAP_TOKEN_IDS = np.array([ord(c) + BYT5_BYTE_OFFSET for c in "<gap>"], dtype=np.int64)
_GAP_LEN = len(_GAP_TOKEN_IDS)

# 省略号 ...
_DOT_TID = ord(".") + BYT5_BYTE_OFFSET

# 括号内 x（损坏标记）
_OPEN_BRACKET_TID  = ord("[") + BYT5_BYTE_OFFSET
_CLOSE_BRACKET_TID = ord("]") + BYT5_BYTE_OFFSET
_X_TID = ord("x") + BYT5_BYTE_OFFSET

# Unicode 分数 → UTF-8 字节序列对应的 token_id 序列
_FRACTION_PATTERNS: List[np.ndarray] = []
for _frac_char in "½⅓⅔¼¾":
    _bytes = _frac_char.encode("utf-8")
    _FRACTION_PATTERNS.append(
        np.array([b + BYT5_BYTE_OFFSET for b in _bytes], dtype=np.int64)
    )


# ╔══════════════════════════════════════════════════════════════╗
# ║                    ROLE ID COMPUTATION                      ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_role_ids_from_token_ids(
    input_ids: np.ndarray,
    cfg: Optional[BoundaryAblationConfig] = None,
) -> np.ndarray:
    """从 ByT5 token_ids 计算 role_ids。

    Parameters
    ----------
    input_ids : np.ndarray, shape [seq_len] 或 [batch, seq_len]
    cfg : BoundaryAblationConfig, optional（默认使用全局 ABLATION_CFG）

    Returns
    -------
    np.ndarray, 同 shape, dtype=int64
    """
    if cfg is None:
        cfg = ABLATION_CFG
    if input_ids.ndim == 1:
        return _compute_roles_1d(input_ids, cfg)
    batch_size, seq_len = input_ids.shape
    result = np.empty_like(input_ids, dtype=np.int64)
    for i in range(batch_size):
        result[i] = _compute_roles_1d(input_ids[i], cfg)
    return result


def _compute_roles_1d(
    token_ids: np.ndarray,
    cfg: BoundaryAblationConfig,
) -> np.ndarray:
    """单条序列的角色标注（LUT + 上下文扫描）。"""
    seq_len = len(token_ids)

    # Step 1: LUT 快速赋值
    clipped = np.clip(token_ids, 0, len(_BYTE_ROLE_LUT) - 1)
    roles = _BYTE_ROLE_LUT[clipped].copy()

    # Step 2: 上下文扫描 — <gap> 模式 → DAMAGE
    if cfg.use_damage:
        i = 0
        while i <= seq_len - _GAP_LEN:
            if np.array_equal(token_ids[i : i + _GAP_LEN], _GAP_TOKEN_IDS):
                roles[i : i + _GAP_LEN] = ROLE_DAMAGE
                i += _GAP_LEN
            else:
                i += 1

        # 省略号 ... → DAMAGE
        i = 0
        while i <= seq_len - 3:
            if (token_ids[i] == _DOT_TID
                    and token_ids[i + 1] == _DOT_TID
                    and token_ids[i + 2] == _DOT_TID):
                roles[i : i + 3] = ROLE_DAMAGE
                i += 3
            else:
                i += 1

        # 方括号内的 x → DAMAGE（如 [x x x]）
        in_bracket = False
        for i in range(seq_len):
            if token_ids[i] == _OPEN_BRACKET_TID:
                in_bracket = True
            elif token_ids[i] == _CLOSE_BRACKET_TID:
                in_bracket = False
            elif in_bracket and token_ids[i] == _X_TID:
                roles[i] = ROLE_DAMAGE

    # Step 3: Unicode 分数 → NUMERIC
    if cfg.use_numeric:
        for frac_tids in _FRACTION_PATTERNS:
            frac_len = len(frac_tids)
            i = 0
            while i <= seq_len - frac_len:
                if np.array_equal(token_ids[i : i + frac_len], frac_tids):
                    roles[i : i + frac_len] = ROLE_NUMERIC
                    i += frac_len
                else:
                    i += 1

    # Step 4: 消融降级 — 关闭的角色退化为 ALPHA
    if not cfg.use_punctuation:
        roles[roles == ROLE_PUNCTUATION] = ROLE_ALPHA
    if not cfg.use_boundary:
        roles[roles == ROLE_BOUNDARY] = ROLE_ALPHA
    if not cfg.use_damage:
        roles[roles == ROLE_DAMAGE] = ROLE_ALPHA
    if not cfg.use_numeric:
        roles[roles == ROLE_NUMERIC] = ROLE_ALPHA

    return roles


# ╔══════════════════════════════════════════════════════════════╗
# ║                  SIDE EMBEDDING MODULE                      ║
# ╚══════════════════════════════════════════════════════════════╝

class BoundarySideEmbedding(nn.Module):
    """并行 side embedding：将角色标签映射为与 byte embedding 同维度的向量。"""

    def __init__(
        self,
        num_roles: int,
        d_model: int,
        combine_mode: str = "add",
        init_std: float = 0.01,
        gate_init_bias: float = -2.0,
        scale_init: float = 0.0,
    ):
        super().__init__()
        self.num_roles = num_roles
        self.d_model = d_model
        self.combine_mode = combine_mode

        # Role embedding table
        self.role_embedding = nn.Embedding(num_roles, d_model)
        nn.init.normal_(self.role_embedding.weight, std=init_std)
        # PAD role → 零向量（不贡献任何信号）
        with torch.no_grad():
            self.role_embedding.weight[ROLE_PAD].zero_()

        # Combine-specific 参数
        if combine_mode == "gate":
            # gate: σ(Wg · [byte; role] + bg) ∈ (0,1)
            # 初始 bias < 0 → σ(bias) ≈ 0 → 保护预训练权重
            self.gate_proj = nn.Linear(d_model * 2, d_model)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, gate_init_bias)
        elif combine_mode == "scale":
            # 可学习全局缩放因子，初始 = 0 → 无贡献
            self.scale = nn.Parameter(torch.tensor(scale_init))

    def forward(
        self,
        byte_embeds: torch.Tensor,
        role_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        byte_embeds : [batch, seq_len, d_model] — 原始 byte embedding
        role_ids    : [batch, seq_len]           — 角色 ID (int64)

        Returns
        -------
        [batch, seq_len, d_model] — 融合后的 embedding
        """
        role_embeds = self.role_embedding(role_ids)  # [B, S, D]

        if self.combine_mode == "add":
            return byte_embeds + role_embeds

        elif self.combine_mode == "gate":
            gate_input = torch.cat([byte_embeds, role_embeds], dim=-1)  # [B, S, 2D]
            gate = torch.sigmoid(self.gate_proj(gate_input))           # [B, S, D]
            return byte_embeds + gate * role_embeds

        elif self.combine_mode == "scale":
            return byte_embeds + self.scale * role_embeds

        else:
            raise ValueError(f"Unknown combine_mode: {self.combine_mode!r}")

    def extra_repr(self) -> str:
        return (
            f"num_roles={self.num_roles}, d_model={self.d_model}, "
            f"combine_mode={self.combine_mode!r}"
        )


# ╔══════════════════════════════════════════════════════════════╗
# ║                 EMBEDDING WRAPPER                           ║
# ╚══════════════════════════════════════════════════════════════╝

class BoundaryAwareEmbeddingWrapper(nn.Module):
    """包装原始 embed_tokens，在 forward 中自动计算 role_ids 并注入 side embedding。

    对外暴露与 nn.Embedding 兼容的接口（weight, num_embeddings, embedding_dim），
    确保 HF Transformers 内部对 embed_tokens 的属性访问不会报错。
    """

    def __init__(
        self,
        original_embed: nn.Embedding,
        side_embedding: BoundarySideEmbedding,
        cfg: BoundaryAblationConfig,
    ):
        super().__init__()
        self.original_embed = original_embed
        self.side_embedding = side_embedding
        self._cfg = cfg

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        byte_embeds = self.original_embed(input_ids)

        if not self._cfg.enable:
            return byte_embeds

        # 从 token_ids 实时计算 role_ids（CPU numpy → GPU tensor）
        role_ids = _token_ids_to_role_tensor(input_ids, self._cfg)
        return self.side_embedding(byte_embeds, role_ids)

    # ---- nn.Embedding 兼容属性 ----
    @property
    def weight(self):
        return self.original_embed.weight

    @property
    def num_embeddings(self):
        return self.original_embed.num_embeddings

    @property
    def embedding_dim(self):
        return self.original_embed.embedding_dim

    def __repr__(self):
        return (
            f"BoundaryAwareEmbeddingWrapper(\n"
            f"  original={self.original_embed},\n"
            f"  side={self.side_embedding}\n"
            f")"
        )


def _token_ids_to_role_tensor(
    input_ids: torch.Tensor,
    cfg: BoundaryAblationConfig,
) -> torch.Tensor:
    """将 GPU 上的 input_ids 转为 role_ids tensor（同 device/dtype）。"""
    np_ids = input_ids.detach().cpu().numpy()
    role_np = compute_role_ids_from_token_ids(np_ids, cfg)
    return torch.from_numpy(role_np).to(device=input_ids.device, dtype=torch.long)


# ╔══════════════════════════════════════════════════════════════╗
# ║                    MODEL PATCHING                           ║
# ╚══════════════════════════════════════════════════════════════╝

def install_boundary_aware(
    model: nn.Module,
    cfg: Optional[BoundaryAblationConfig] = None,
) -> Optional[BoundaryAwareEmbeddingWrapper]:
    """一键安装：Patch 模型 encoder 的 embed_tokens，注入 boundary-aware side embedding。

    Parameters
    ----------
    model : T5ForConditionalGeneration
    cfg   : BoundaryAblationConfig（默认使用全局 ABLATION_CFG）

    Returns
    -------
    BoundaryAwareEmbeddingWrapper 或 None（若 cfg.enable=False）

    Usage
    -----
    >>> from model_ablation import install_boundary_aware, ABLATION_CFG
    >>> ABLATION_CFG.enable = True
    >>> ABLATION_CFG.use_damage = True
    >>> ABLATION_CFG.combine_mode = "scale"  # 最安全的融合方式
    >>> wrapper = install_boundary_aware(model)
    """
    if cfg is None:
        cfg = ABLATION_CFG

    if not cfg.enable:
        print("   ⬜ Boundary-aware frontend: DISABLED")
        return None

    d_model = model.config.d_model

    # 1. 创建 side embedding
    side_embed = BoundarySideEmbedding(
        num_roles=NUM_ROLES,
        d_model=d_model,
        combine_mode=cfg.combine_mode,
        init_std=cfg.side_embed_init_std,
        gate_init_bias=cfg.gate_init_bias,
        scale_init=cfg.scale_init,
    )

    # 匹配预训练权重的 dtype 和 device
    embed_param = model.shared.weight
    side_embed = side_embed.to(dtype=embed_param.dtype, device=embed_param.device)

    # 2. 创建 wrapper
    wrapper = BoundaryAwareEmbeddingWrapper(model.shared, side_embed, cfg)
    wrapper = wrapper.to(dtype=embed_param.dtype, device=embed_param.device)

    # 3. Patch encoder（decoder 默认不 patch，保留原始 byte embedding）
    model.encoder.embed_tokens = wrapper
    if cfg.patch_decoder:
        # Decoder 共享同一个 wrapper（可选）
        model.decoder.embed_tokens = wrapper

    # 4. 冻结控制
    if cfg.freeze_base_embed:
        model.shared.weight.requires_grad = False
        print("   🧊 Base byte embedding: FROZEN")
    if cfg.freeze_side_embed:
        for p in side_embed.parameters():
            p.requires_grad = False
        print("   🧊 Side embedding: FROZEN")

    # 5. 打印摘要
    _print_summary(cfg, side_embed)

    return wrapper


def uninstall_boundary_aware(model: nn.Module) -> None:
    """移除 boundary-aware patch，恢复原始 embed_tokens。"""
    if isinstance(model.encoder.embed_tokens, BoundaryAwareEmbeddingWrapper):
        original = model.encoder.embed_tokens.original_embed
        model.encoder.embed_tokens = original
        print("   🔄 Boundary-aware frontend: UNINSTALLED (encoder)")
    if isinstance(getattr(model, "decoder", None), nn.Module):
        if isinstance(model.decoder.embed_tokens, BoundaryAwareEmbeddingWrapper):
            original = model.decoder.embed_tokens.original_embed
            model.decoder.embed_tokens = original
            print("   🔄 Boundary-aware frontend: UNINSTALLED (decoder)")


def _print_summary(cfg: BoundaryAblationConfig, side_embed: BoundarySideEmbedding):
    """打印 boundary-aware frontend 安装摘要。"""
    side_total = sum(p.numel() for p in side_embed.parameters())
    side_trainable = sum(p.numel() for p in side_embed.parameters() if p.requires_grad)

    active_roles = []
    if cfg.use_punctuation:
        active_roles.append("PUNCTUATION")
    if cfg.use_boundary:
        active_roles.append("BOUNDARY")
    if cfg.use_damage:
        active_roles.append("DAMAGE")
    if cfg.use_numeric:
        active_roles.append("NUMERIC")
    active_str = ", ".join(active_roles) if active_roles else "(none — all collapsed to ALPHA)"

    print(f"   ✅ Boundary-aware frontend: ENABLED")
    print(f"      Combine mode : {cfg.combine_mode}")
    print(f"      Active roles : {active_str}")
    print(f"      Side params  : {side_total:,} (trainable: {side_trainable:,})")
    if cfg.combine_mode == "scale":
        print(f"      Initial scale: {cfg.scale_init}")
    elif cfg.combine_mode == "gate":
        print(f"      Gate init σ  : {torch.sigmoid(torch.tensor(cfg.gate_init_bias)).item():.3f}")
    print(f"      Patch decoder: {cfg.patch_decoder}")


# ╔══════════════════════════════════════════════════════════════╗
# ║                  LATENT BOTTLENECK MODULE                    ║
# ╚══════════════════════════════════════════════════════════════╝

class LatentBottleneckLayer(nn.Module):
    """Stride-2 1D 卷积瓶颈层，将 encoder 隐状态序列长度减半。

    浅层（前 k 层）在原始字节分辨率上运行，捕获音节级特征。
    瓶颈层将序列压缩，使深层在更高语义密度上运行。

    Architecture::

        hidden_states [B, L, D]
          → pad to even length (if needed)
          → Conv1d(D, D, kernel=2, stride=2) → [B, L//2, D]
          → (optional) + mean-pool residual
          → LayerNorm
    """

    def __init__(
        self,
        d_model: int,
        kernel_size: int = 2,
        stride: int = 2,
        use_residual: bool = True,
        use_layernorm: bool = True,
        init_std: float = 0.02,
    ):
        super().__init__()
        self.d_model = d_model
        self.stride = stride
        self.kernel_size = kernel_size
        self.use_residual = use_residual

        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, stride=stride)
        nn.init.normal_(self.conv.weight, std=init_std)
        nn.init.zeros_(self.conv.bias)

        self.norm = nn.LayerNorm(d_model) if use_layernorm else nn.Identity()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        hidden_states : [batch, seq_len, d_model]

        Returns
        -------
        [batch, ceil(seq_len / stride), d_model]
        """
        B, L, D = hidden_states.shape

        # Pad to multiple of stride
        pad_len = 0
        if L % self.stride != 0:
            pad_len = self.stride - (L % self.stride)
            hidden_states = torch.nn.functional.pad(hidden_states, (0, 0, 0, pad_len))
            L = L + pad_len

        # Conv1d: [B, D, L] → [B, D, L//stride]
        x = hidden_states.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)  # [B, L//stride, D]

        # Residual: mean-pool shortcut
        if self.use_residual:
            residual = hidden_states.reshape(B, L // self.stride, self.stride, D).mean(dim=2)
            x = x + residual

        x = self.norm(x)
        return x

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, use_residual={self.use_residual}"
        )


def _rebuild_encoder_causal_mask(
    attention_mask: torch.Tensor,
    new_seq_len: int,
    stride: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Downsample 2D attention mask and rebuild 4D encoder causal mask.

    Parameters
    ----------
    attention_mask : [batch, seq_len_original] — 1=valid, 0=pad
    new_seq_len    : compressed sequence length
    stride         : compression stride
    dtype          : target dtype for the mask

    Returns
    -------
    [batch, 1, 1, new_seq_len] — 0 for valid, -inf for padding
    """
    mask = attention_mask.unsqueeze(1).float()  # [B, 1, L]
    L = mask.shape[2]
    if L % stride != 0:
        pad_len = stride - (L % stride)
        mask = torch.nn.functional.pad(mask, (0, pad_len), value=0.0)

    mask = torch.nn.functional.max_pool1d(mask, kernel_size=stride, stride=stride)
    mask = mask[:, :, :new_seq_len]
    mask = mask.squeeze(1)  # [B, new_seq_len]

    causal_mask = mask[:, None, None, :]  # [B, 1, 1, new_seq_len]
    causal_mask = causal_mask.to(dtype=dtype)
    causal_mask = (1.0 - causal_mask) * torch.finfo(dtype).min
    return causal_mask


def _downsample_attention_mask_2d(
    attention_mask: torch.Tensor,
    new_seq_len: int,
    stride: int,
) -> torch.Tensor:
    """Downsample 2D attention mask [B, L] → [B, L//stride] via max-pool."""
    mask = attention_mask.unsqueeze(1).float()
    L = mask.shape[2]
    if L % stride != 0:
        pad_len = stride - (L % stride)
        mask = torch.nn.functional.pad(mask, (0, pad_len), value=0.0)
    mask = torch.nn.functional.max_pool1d(mask, kernel_size=stride, stride=stride)
    mask = mask[:, :, :new_seq_len].squeeze(1)
    return mask.to(dtype=attention_mask.dtype)


# ╔══════════════════════════════════════════════════════════════╗
# ║              HYBRID ATTENTION MASK BUILDER                   ║
# ╚══════════════════════════════════════════════════════════════╝

def _build_hybrid_causal_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    hybrid_cfg: HybridAttentionConfig,
    boundary_cfg: Optional[BoundaryAblationConfig],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build [B, 1, L, L] hybrid attention mask with local window + global anchors.

    Parameters
    ----------
    input_ids      : [B, L] — ByT5 token ids
    attention_mask  : [B, L] — 1=valid, 0=pad
    hybrid_cfg      : HybridAttentionConfig
    boundary_cfg    : BoundaryAblationConfig (for role computation, may be None)
    dtype           : target dtype

    Returns
    -------
    [B, 1, L, L] — 0.0 for allowed positions, -inf for masked positions
    """
    B, L = input_ids.shape
    device = input_ids.device
    W = hybrid_cfg.window_size

    # 1. Local sliding window: positions within distance W are visible
    positions = torch.arange(L, device=device)
    # [L, L] distance matrix
    dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()
    local_mask = dist <= W  # [L, L], bool

    # 2. Find global anchor positions from role IDs
    role_ids = _token_ids_to_role_tensor(input_ids, boundary_cfg or ABLATION_CFG)  # [B, L]
    is_global = torch.zeros(B, L, dtype=torch.bool, device=device)
    for rid in hybrid_cfg.global_role_ids:
        is_global = is_global | (role_ids == rid)

    # Optional: every-N global anchors
    if hybrid_cfg.global_every_n > 0:
        periodic = torch.zeros(L, dtype=torch.bool, device=device)
        periodic[::hybrid_cfg.global_every_n] = True
        is_global = is_global | periodic.unsqueeze(0)

    # Global row (query is global → can see everything) + column (key is global → everyone sees it)
    global_row = is_global.unsqueeze(2)   # [B, L, 1]
    global_col = is_global.unsqueeze(1)   # [B, 1, L]

    # 3. Combine: local OR query-is-global OR key-is-global
    hybrid_mask = local_mask.unsqueeze(0) | global_row | global_col  # [B, L, L]

    # 4. Apply padding mask (pad positions can't attend or be attended to)
    if attention_mask is not None:
        pad_valid = attention_mask.bool()  # [B, L]
        hybrid_mask = hybrid_mask & pad_valid.unsqueeze(1) & pad_valid.unsqueeze(2)

    # 5. Convert to float causal mask format: 0 for allowed, -inf for masked
    causal_mask = hybrid_mask.unsqueeze(1).to(dtype=dtype)  # [B, 1, L, L]
    causal_mask = (1.0 - causal_mask) * torch.finfo(dtype).min

    return causal_mask


# ╔══════════════════════════════════════════════════════════════╗
# ║               PATCHED ENCODER FORWARD                        ║
# ╚══════════════════════════════════════════════════════════════╝

def _make_patched_encoder_forward(
    bottleneck_layer, bottleneck_after_layer, stride,
    hybrid_cfg=None, boundary_cfg=None,
):
    """Create a patched T5Stack.forward for encoder with bottleneck injection.

    Mirrors the original T5Stack.forward (transformers 5.3.x) with two additions:
    (A) Hybrid attention mask override for encoder (local window + global anchors).
    (B) Bottleneck injection after layer ``bottleneck_after_layer``, which
        compresses hidden_states, rebuilds causal_mask, and resets position_bias.
    """
    from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
    from transformers.cache_utils import EncoderDecoderCache, DynamicCache

    # Import mask creation helpers (transformers 5.x)
    try:
        from transformers.models.t5.modeling_t5 import (
            create_causal_mask as _create_causal_mask,
            create_bidirectional_mask as _create_bidirectional_mask,
        )
    except ImportError:
        _create_causal_mask = None
        _create_bidirectional_mask = None

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        # ────────── Setup (mirrors T5Stack.forward 5.3.x) ──────────
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None
            else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and "
                f"{err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or "
                f"{err_msg_prefix}inputs_embeds"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                use_cache = False

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "You have to initialize the model with valid token embeddings"
                )
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # Encoder: do not pass cache
        if not self.is_decoder:
            past_key_values = None

        if self.is_decoder:
            if use_cache and past_key_values is None:
                if self.config.is_encoder_decoder:
                    past_key_values = EncoderDecoderCache(
                        DynamicCache(config=self.config),
                        DynamicCache(config=self.config),
                    )
                else:
                    past_key_values = DynamicCache(config=self.config)

        past_key_values_length = (
            past_key_values.get_seq_length() if past_key_values is not None else 0
        )
        if cache_position is None:
            cache_position = torch.arange(
                past_key_values_length,
                past_key_values_length + seq_length,
                device=inputs_embeds.device,
            )

        # ────────── Mask creation ──────────
        if self.config.is_decoder and _create_causal_mask is not None:
            causal_mask = _create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=(
                    past_key_values.self_attention_cache
                    if isinstance(past_key_values, EncoderDecoderCache)
                    else past_key_values
                ),
            )
        elif not self.is_decoder:
            # ====== HYBRID ATTENTION MASK (change A vs original) ======
            if (hybrid_cfg is not None and hybrid_cfg.enable
                    and input_ids is not None):
                causal_mask = _build_hybrid_causal_mask(
                    input_ids, attention_mask, hybrid_cfg, boundary_cfg,
                    inputs_embeds.dtype,
                )
            elif _create_bidirectional_mask is not None and attention_mask is not None:
                causal_mask = _create_bidirectional_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                )
            elif attention_mask is not None:
                # Fallback for older transformers
                causal_mask = attention_mask[:, None, None, :]
                causal_mask = causal_mask.to(dtype=inputs_embeds.dtype)
                causal_mask = (1.0 - causal_mask) * torch.finfo(inputs_embeds.dtype).min
            else:
                causal_mask = None
            # =============================================================
        else:
            causal_mask = None

        encoder_extended_attention_mask = None
        if self.is_decoder and encoder_hidden_states is not None:
            if _create_bidirectional_mask is not None:
                encoder_extended_attention_mask = _create_bidirectional_mask(
                    config=self.config,
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                )
            elif encoder_attention_mask is not None:
                encoder_extended_attention_mask = self.invert_attention_mask(
                    encoder_attention_mask,
                )

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = (
            () if (output_attentions and self.is_decoder) else None
        )
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        # ────────── Layer loop with bottleneck injection ──────────
        for i, layer_module in enumerate(self.block):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                causal_mask,
                position_bias,
                encoder_hidden_states,
                encoder_extended_attention_mask,
                encoder_decoder_position_bias,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                return_dict=return_dict,
                cache_position=cache_position,
            )

            hidden_states = layer_outputs[0]
            position_bias = layer_outputs[1]

            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    3 if output_attentions else 2
                ]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (
                        layer_outputs[4],
                    )

            # ====== BOTTLENECK INJECTION (change B vs original) ======
            if i == bottleneck_after_layer:
                hidden_states = bottleneck_layer(hidden_states)
                new_seq_len = hidden_states.shape[1]
                # Rebuild causal mask for compressed sequence
                if causal_mask is not None and attention_mask is not None:
                    causal_mask = _rebuild_encoder_causal_mask(
                        attention_mask, new_seq_len, stride, hidden_states.dtype,
                    )
                elif causal_mask is not None:
                    causal_mask = None
                # Reset position bias — next layer recomputes for new length
                position_bias = None
                # Store compressed 2D mask for decoder cross-attention
                if attention_mask is not None:
                    self._bottleneck_compressed_mask = _downsample_attention_mask_2d(
                        attention_mask, new_seq_len, stride,
                    )
            # ============================================================

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    past_key_values,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

    return patched_forward


# ╔══════════════════════════════════════════════════════════════╗
# ║        ENCODER STRUCTURAL PATCHES (INSTALL / UNINSTALL)      ║
# ╚══════════════════════════════════════════════════════════════╝

def install_encoder_patches(
    model: nn.Module,
    bottleneck_cfg: Optional[BottleneckConfig] = None,
    hybrid_cfg: Optional[HybridAttentionConfig] = None,
    boundary_cfg: Optional[BoundaryAblationConfig] = None,
) -> Tuple[Optional[LatentBottleneckLayer], bool]:
    """Unified installer for encoder forward patches (bottleneck + hybrid attention).

    Parameters
    ----------
    model          : T5ForConditionalGeneration
    bottleneck_cfg : BottleneckConfig (None → use global BOTTLENECK_CFG)
    hybrid_cfg     : HybridAttentionConfig (None → use global HYBRID_CFG)
    boundary_cfg   : BoundaryAblationConfig (for hybrid anchor roles; None → ABLATION_CFG)

    Returns
    -------
    (LatentBottleneckLayer or None, hybrid_enabled: bool)
    """
    import types

    if bottleneck_cfg is None:
        bottleneck_cfg = BOTTLENECK_CFG
    if hybrid_cfg is None:
        hybrid_cfg = HYBRID_CFG
    if boundary_cfg is None:
        boundary_cfg = ABLATION_CFG

    need_bottleneck = bottleneck_cfg.enable
    need_hybrid = hybrid_cfg.enable

    # ---- Bottleneck status ----
    bottleneck = None
    if not need_bottleneck:
        print("   ⬜ Latent bottleneck: DISABLED")
    # ---- Hybrid status ----
    if not need_hybrid:
        print("   ⬜ Hybrid attention: DISABLED")

    if not need_bottleneck and not need_hybrid:
        return None, False

    d_model = model.config.d_model
    num_encoder_layers = model.config.num_layers

    # ---- Create bottleneck module (if needed) ----
    bn_layer_idx = -1
    bn_stride = 2
    if need_bottleneck:
        if bottleneck_cfg.bottleneck_after_layer >= num_encoder_layers:
            raise ValueError(
                f"bottleneck_after_layer={bottleneck_cfg.bottleneck_after_layer} >= "
                f"num_encoder_layers={num_encoder_layers}"
            )
        bottleneck = LatentBottleneckLayer(
            d_model=d_model,
            kernel_size=bottleneck_cfg.kernel_size,
            stride=bottleneck_cfg.stride,
            use_residual=bottleneck_cfg.use_residual,
            use_layernorm=bottleneck_cfg.use_layernorm,
            init_std=bottleneck_cfg.init_std,
        )
        embed_param = model.shared.weight
        bottleneck = bottleneck.to(dtype=embed_param.dtype, device=embed_param.device)
        model.encoder._bottleneck_module = bottleneck
        model.encoder._bottleneck_compressed_mask = None
        bn_layer_idx = bottleneck_cfg.bottleneck_after_layer
        bn_stride = bottleneck_cfg.stride

    # ---- Install patched forward ----
    model.encoder._original_forward = model.encoder.forward
    patched_fwd = _make_patched_encoder_forward(
        bottleneck_layer=bottleneck,
        bottleneck_after_layer=bn_layer_idx,
        stride=bn_stride,
        hybrid_cfg=hybrid_cfg if need_hybrid else None,
        boundary_cfg=boundary_cfg,
    )
    model.encoder.forward = types.MethodType(patched_fwd, model.encoder)

    # ---- Decoder cross-attention mask hook (for bottleneck) ----
    # The compressed mask must persist across all decoder calls during generate().
    # During beam search, encoder_hidden_states are expanded by num_beams,
    # so the mask must be expanded to match.
    if need_bottleneck:
        def _decoder_mask_hook(module, args, kwargs):
            compressed = getattr(model.encoder, '_bottleneck_compressed_mask', None)
            if compressed is None:
                return args, kwargs
            # Detect beam expansion: compare batch dim of encoder_hidden_states vs mask
            enc_hs = kwargs.get('encoder_hidden_states')
            if enc_hs is None and len(args) >= 4:
                enc_hs = args[3] if args[3] is not None else None
            if enc_hs is not None and enc_hs.shape[0] != compressed.shape[0]:
                expand_factor = enc_hs.shape[0] // compressed.shape[0]
                if expand_factor > 1:
                    compressed = compressed.repeat_interleave(expand_factor, dim=0)
            kwargs['encoder_attention_mask'] = compressed
            return args, kwargs

        hook_handle = model.decoder.register_forward_pre_hook(
            _decoder_mask_hook, with_kwargs=True,
        )
        model.encoder._decoder_mask_hook_handle = hook_handle

    # ---- Freeze control (bottleneck) ----
    if need_bottleneck and bottleneck_cfg.freeze_bottleneck:
        for p in bottleneck.parameters():
            p.requires_grad = False
        print("   🧊 Bottleneck: FROZEN")

    # ---- Print summaries ----
    if need_bottleneck:
        k = bottleneck_cfg.bottleneck_after_layer
        total_params = sum(p.numel() for p in bottleneck.parameters())
        trainable = sum(p.numel() for p in bottleneck.parameters() if p.requires_grad)
        print(f"   ✅ Latent bottleneck: ENABLED")
        print(f"      After layer    : {k} (of {num_encoder_layers})")
        print(f"      Stride         : {bottleneck_cfg.stride}× compression")
        print(f"      Residual       : {bottleneck_cfg.use_residual}")
        print(f"      Params         : {total_params:,} (trainable: {trainable:,})")
        print(f"      Shallow layers : 0–{k} (byte-level)")
        print(f"      Deep layers    : {k+1}–{num_encoder_layers-1} (latent, L/{bottleneck_cfg.stride})")

    if need_hybrid:
        anchor_names = [ROLE_NAMES[r] for r in hybrid_cfg.global_role_ids if r < len(ROLE_NAMES)]
        print(f"   ✅ Hybrid attention: ENABLED")
        print(f"      Window size    : {hybrid_cfg.window_size} (covers {2*hybrid_cfg.window_size+1} tokens)")
        print(f"      Global anchors : {', '.join(anchor_names)}")
        if hybrid_cfg.global_every_n > 0:
            print(f"      Periodic global: every {hybrid_cfg.global_every_n} tokens")
        print(f"      New params     : 0 (mask-only, no new weights)")

    return bottleneck, need_hybrid


# Legacy alias for backward compatibility
def install_latent_bottleneck(
    model: nn.Module,
    cfg: Optional[BottleneckConfig] = None,
) -> Optional[LatentBottleneckLayer]:
    """Install latent bottleneck only. See install_encoder_patches for unified API."""
    bottleneck, _ = install_encoder_patches(model, bottleneck_cfg=cfg)
    return bottleneck


def uninstall_encoder_patches(model: nn.Module) -> None:
    """Remove all encoder forward patches, restoring original forward."""
    encoder = model.encoder
    if hasattr(encoder, '_original_forward'):
        encoder.forward = encoder._original_forward
        del encoder._original_forward
        print("   🔄 Encoder patches: UNINSTALLED (original forward restored)")
    if hasattr(encoder, '_decoder_mask_hook_handle'):
        encoder._decoder_mask_hook_handle.remove()
        del encoder._decoder_mask_hook_handle
    for attr in ['_bottleneck_module', '_bottleneck_compressed_mask']:
        if hasattr(encoder, attr):
            delattr(encoder, attr)


# Legacy alias
uninstall_latent_bottleneck = uninstall_encoder_patches


# ╔══════════════════════════════════════════════════════════════╗
# ║               CHECKPOINT SAVE / LOAD HELPERS                ║
# ╚══════════════════════════════════════════════════════════════╝

SIDE_EMBED_FILENAME = "boundary_side_embedding.pt"


def save_side_embedding(
    model: nn.Module,
    save_dir: str,
    cfg: Optional[BoundaryAblationConfig] = None,
) -> Optional[str]:
    """保存 side embedding 权重到 checkpoint 目录。

    Returns 保存路径，或 None（未安装 boundary-aware）。
    """
    import os
    wrapper = _find_wrapper(model)
    if wrapper is None:
        return None
    path = os.path.join(save_dir, SIDE_EMBED_FILENAME)
    state = {
        "side_embedding": wrapper.side_embedding.state_dict(),
        "config": (cfg or ABLATION_CFG).__dict__,
    }
    torch.save(state, path)
    return path


def load_side_embedding(
    model: nn.Module,
    load_dir: str,
    strict: bool = True,
) -> bool:
    """从 checkpoint 目录加载 side embedding 权重。

    Returns True if loaded, False if file not found.
    """
    import os
    path = os.path.join(load_dir, SIDE_EMBED_FILENAME)
    if not os.path.exists(path):
        return False
    wrapper = _find_wrapper(model)
    if wrapper is None:
        print(f"   ⚠️ Boundary-aware not installed, skipping load from {path}")
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    wrapper.side_embedding.load_state_dict(state["side_embedding"], strict=strict)
    print(f"   ✅ Loaded boundary side embedding from {path}")
    return True


def _find_wrapper(model: nn.Module) -> Optional[BoundaryAwareEmbeddingWrapper]:
    """在 encoder.embed_tokens 中查找 wrapper（兼容 DDP 包装）。"""
    # DDP wraps model as model.module
    _raw = getattr(model, "module", model)
    enc_embed = getattr(getattr(_raw, "encoder", None), "embed_tokens", None)
    if isinstance(enc_embed, BoundaryAwareEmbeddingWrapper):
        return enc_embed
    return None


BOTTLENECK_FILENAME = "latent_bottleneck.pt"


def save_bottleneck_weights(
    model: nn.Module,
    save_dir: str,
    cfg: Optional[BottleneckConfig] = None,
) -> Optional[str]:
    """保存 bottleneck 权重到 checkpoint 目录。

    Returns 保存路径，或 None（未安装 bottleneck）。
    """
    import os
    _raw = getattr(model, "module", model)
    bottleneck = getattr(getattr(_raw, "encoder", None), "_bottleneck_module", None)
    if bottleneck is None:
        return None
    path = os.path.join(save_dir, BOTTLENECK_FILENAME)
    state = {
        "bottleneck": bottleneck.state_dict(),
        "config": (cfg or BOTTLENECK_CFG).__dict__,
    }
    torch.save(state, path)
    return path


def load_bottleneck_weights(
    model: nn.Module,
    load_dir: str,
    strict: bool = True,
) -> bool:
    """从 checkpoint 目录加载 bottleneck 权重。

    Returns True if loaded, False if file not found.
    """
    import os
    path = os.path.join(load_dir, BOTTLENECK_FILENAME)
    if not os.path.exists(path):
        return False
    _raw = getattr(model, "module", model)
    bottleneck = getattr(getattr(_raw, "encoder", None), "_bottleneck_module", None)
    if bottleneck is None:
        print(f"   ⚠️ Bottleneck not installed, skipping load from {path}")
        return False
    state = torch.load(path, map_location="cpu", weights_only=True)
    bottleneck.load_state_dict(state["bottleneck"], strict=strict)
    print(f"   ✅ Loaded bottleneck weights from {path}")
    return True


# ╔══════════════════════════════════════════════════════════════╗
# ║                    DIAGNOSTIC TOOLS                         ║
# ╚══════════════════════════════════════════════════════════════╝

def visualize_roles(text: str, tokenizer=None) -> str:
    """可视化一段文本的角色标注结果（调试用）。

    Parameters
    ----------
    text : str — 阿卡德语音译文本
    tokenizer : ByT5Tokenizer（可选，用于对齐验证）

    Returns
    -------
    str — 格式化的角色可视化
    """
    # 手动模拟 ByT5 byte tokenization
    text_bytes = text.encode("utf-8")
    token_ids = np.array([b + BYT5_BYTE_OFFSET for b in text_bytes], dtype=np.int64)
    roles = compute_role_ids_from_token_ids(token_ids)

    lines = []
    lines.append(f"Text: {text}")
    lines.append(f"Bytes: {len(text_bytes)}, Tokens: {len(token_ids)}")
    lines.append("")

    # 颜色编码（终端 ANSI）
    role_colors = {
        ROLE_PAD: "\033[90m",         # gray
        ROLE_ALPHA: "\033[0m",        # default
        ROLE_PUNCTUATION: "\033[36m", # cyan
        ROLE_BOUNDARY: "\033[33m",    # yellow
        ROLE_DAMAGE: "\033[31m",      # red
        ROLE_NUMERIC: "\033[32m",     # green
    }
    reset = "\033[0m"

    # 按字符聚合（一个 Unicode 字符可能跨多个字节）
    char_roles = []
    byte_idx = 0
    for ch in text:
        ch_bytes = ch.encode("utf-8")
        n_bytes = len(ch_bytes)
        # 取该字符所有字节的角色（取众数）
        if byte_idx + n_bytes <= len(roles):
            ch_role = int(np.bincount(roles[byte_idx:byte_idx + n_bytes]).argmax())
        else:
            ch_role = ROLE_PAD
        char_roles.append((ch, ch_role))
        byte_idx += n_bytes

    # 输出带颜色的文本
    colored = ""
    for ch, role in char_roles:
        colored += f"{role_colors.get(role, '')}{ch}"
    colored += reset
    lines.append(f"Colored: {colored}")

    # 角色统计
    lines.append("")
    lines.append("Role distribution:")
    unique, counts = np.unique(roles, return_counts=True)
    for role_id, count in zip(unique, counts):
        pct = count / len(roles) * 100
        lines.append(f"  {ROLE_NAMES[role_id]:>12s}: {count:4d} ({pct:5.1f}%)")

    return "\n".join(lines)


def role_distribution_for_dataset(
    texts: List[str],
) -> Dict[str, float]:
    """统计一个数据集中各角色的占比（用于消融实验报告）。"""
    total_counts = np.zeros(NUM_ROLES, dtype=np.int64)
    for text in texts:
        text_bytes = text.encode("utf-8")
        token_ids = np.array([b + BYT5_BYTE_OFFSET for b in text_bytes], dtype=np.int64)
        roles = compute_role_ids_from_token_ids(token_ids)
        unique, counts = np.unique(roles, return_counts=True)
        for r, c in zip(unique, counts):
            total_counts[r] += c
    total = total_counts.sum()
    return {
        ROLE_NAMES[i]: float(total_counts[i] / max(total, 1))
        for i in range(NUM_ROLES)
    }


# ╔══════════════════════════════════════════════════════════════╗
# ║                    COVERAGE LOSS                             ║
# ╚══════════════════════════════════════════════════════════════╝

def compute_coverage_loss(
    cross_attentions: Tuple[torch.Tensor, ...],
    labels: torch.Tensor,
    cfg: Optional[CoverageConfig] = None,
) -> torch.Tensor:
    """Compute coverage loss from decoder cross-attention weights.

    Coverage loss penalizes the model for re-attending to already-covered
    encoder positions, suppressing repetition and transliteration leakage.

    Parameters
    ----------
    cross_attentions : tuple of [B, num_heads, T_dec, T_enc] tensors (one per layer)
    labels           : [B, T_dec] with -100 for padding
    cfg              : CoverageConfig (defaults to global COVERAGE_CFG)

    Returns
    -------
    Scalar coverage loss (un-weighted; caller applies alpha + warmup).
    """
    if cfg is None:
        cfg = COVERAGE_CFG

    if cross_attentions is None or len(cross_attentions) == 0:
        return torch.tensor(0.0)

    # Select decoder layer
    cross_attn = cross_attentions[cfg.layer_index]  # [B, H, T_dec, T_enc]

    # Average over heads → [B, T_dec, T_enc]
    attn = cross_attn.float().mean(dim=1)

    # Valid decoder positions (not padding)
    mask = (labels != -100)  # [B, T_dec]
    mask_f = mask.to(attn.dtype)

    # Cumulative coverage: how much each encoder position has been attended to
    # coverage[t] = sum of attn[0..t-1] — shifted so step t sees only prior coverage
    coverage = torch.cumsum(attn, dim=1) - attn  # [B, T_dec, T_enc]

    # Coverage loss: min(attn_t, coverage_t) summed over encoder positions
    # High when model re-attends to already-covered source positions
    cov_loss = torch.min(attn, coverage).sum(dim=-1)  # [B, T_dec]

    n_valid = mask_f.sum().clamp(min=1.0)
    return (cov_loss * mask_f).sum() / n_valid


# ╔══════════════════════════════════════════════════════════════╗
# ║                       SELF-TEST                             ║
# ╚══════════════════════════════════════════════════════════════╝

def _self_test():
    """基本功能自测。"""
    print("=" * 60)
    print("Boundary-aware frontend — self test")
    print("=" * 60)

    # Test 1: Role assignment
    test_text = "{d}aš-šur 10 ½ <gap> [x x x] a-na"
    print(f"\n--- Test 1: Role assignment ---")
    print(visualize_roles(test_text))

    # Test 2: Ablation — disable damage
    print(f"\n--- Test 2: Ablation (damage OFF) ---")
    saved = ABLATION_CFG.use_damage
    ABLATION_CFG.use_damage = False
    text_bytes = test_text.encode("utf-8")
    token_ids = np.array([b + BYT5_BYTE_OFFSET for b in text_bytes], dtype=np.int64)
    roles = compute_role_ids_from_token_ids(token_ids)
    assert ROLE_DAMAGE not in roles, "DAMAGE should be absent when disabled"
    print("  ✓ No DAMAGE roles when use_damage=False")
    ABLATION_CFG.use_damage = saved

    # Test 3: Module shapes (mock)
    print(f"\n--- Test 3: Module shapes ---")
    d_model = 64  # small for test
    side = BoundarySideEmbedding(NUM_ROLES, d_model, combine_mode="add")
    B, S = 2, 10
    byte_emb = torch.randn(B, S, d_model)
    role_ids = torch.randint(0, NUM_ROLES, (B, S))
    out = side(byte_emb, role_ids)
    assert out.shape == (B, S, d_model), f"Expected {(B, S, d_model)}, got {out.shape}"
    print(f"  ✓ add mode: {out.shape}")

    # gate mode
    side_gate = BoundarySideEmbedding(NUM_ROLES, d_model, combine_mode="gate")
    out_gate = side_gate(byte_emb, role_ids)
    assert out_gate.shape == (B, S, d_model)
    print(f"  ✓ gate mode: {out_gate.shape}")

    # scale mode
    side_scale = BoundarySideEmbedding(NUM_ROLES, d_model, combine_mode="scale", scale_init=0.0)
    out_scale = side_scale(byte_emb, role_ids)
    # scale=0 → output should equal byte_emb
    assert torch.allclose(out_scale, byte_emb, atol=1e-6), "scale=0 should produce identity"
    print(f"  ✓ scale mode (init=0): identity verified")

    # Test 4: Bottleneck module
    print(f"\n--- Test 4: Latent bottleneck ---")
    d_model_bn = 64
    bn = LatentBottleneckLayer(d_model_bn, kernel_size=2, stride=2)
    B_bn, L_bn = 2, 20
    hs = torch.randn(B_bn, L_bn, d_model_bn)
    out_bn = bn(hs)
    assert out_bn.shape == (B_bn, L_bn // 2, d_model_bn), \
        f"Expected {(B_bn, L_bn // 2, d_model_bn)}, got {out_bn.shape}"
    print(f"  ✓ Even length: {hs.shape} → {out_bn.shape}")

    # Odd length
    hs_odd = torch.randn(B_bn, 21, d_model_bn)
    out_odd = bn(hs_odd)
    assert out_odd.shape == (B_bn, 11, d_model_bn), \
        f"Expected {(B_bn, 11, d_model_bn)}, got {out_odd.shape}"
    print(f"  ✓ Odd length:  {hs_odd.shape} → {out_odd.shape}")

    # Test 5: Mask downsampling
    print(f"\n--- Test 5: Mask downsampling ---")
    mask = torch.tensor([[1, 1, 1, 1, 0, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.float32)
    causal = _rebuild_encoder_causal_mask(mask, 3, 2, torch.float32)
    assert causal.shape == (2, 1, 1, 3), f"Expected (2,1,1,3), got {causal.shape}"
    # First sample: positions 0,1 valid → compressed pos 0 valid (0.0)
    #               positions 2,3 half valid → compressed pos 1 valid (max-pool)
    #               positions 4,5 both pad → compressed pos 2 pad (-inf)
    assert causal[0, 0, 0, 0] == 0.0, "Position 0 should be valid"
    assert causal[0, 0, 0, 2] < -1e30, "Position 2 should be masked"
    # Second sample: only first 2 valid
    assert causal[1, 0, 0, 0] == 0.0, "Position 0 should be valid"
    assert causal[1, 0, 0, 1] < -1e30, "Position 1 should be masked"
    print(f"  ✓ Causal mask: {mask.shape} → {causal.shape}")

    mask_2d = _downsample_attention_mask_2d(mask, 3, 2)
    assert mask_2d.shape == (2, 3), f"Expected (2,3), got {mask_2d.shape}"
    print(f"  ✓ 2D mask:     {mask.shape} → {mask_2d.shape}")

    # Test 6: Hybrid attention mask
    print(f"\n--- Test 6: Hybrid attention mask ---")
    # Simulate ByT5 token ids for "{d}a-b"
    test_str = "{d}a-b"
    test_bytes = test_str.encode("utf-8")
    tok_ids = torch.tensor([[b + BYT5_BYTE_OFFSET for b in test_bytes]], dtype=torch.long)
    attn_mask = torch.ones(1, tok_ids.shape[1])
    h_cfg = HybridAttentionConfig(enable=True, window_size=1, global_role_ids=(ROLE_BOUNDARY,))
    hybrid_mask = _build_hybrid_causal_mask(tok_ids, attn_mask, h_cfg, ABLATION_CFG, torch.float32)
    L_h = tok_ids.shape[1]
    assert hybrid_mask.shape == (1, 1, L_h, L_h), f"Expected (1,1,{L_h},{L_h}), got {hybrid_mask.shape}"
    # '{' is BOUNDARY (global) → its row and column should be all 0.0 (allowed)
    bracket_pos = 0  # '{' is first byte
    assert hybrid_mask[0, 0, bracket_pos, :].max() == 0.0, "Global anchor row should be all-allowed"
    assert hybrid_mask[0, 0, :, bracket_pos].max() == 0.0, "Global anchor col should be all-allowed"
    # 'a' (pos 3) with window=1 should NOT see 'b' (pos 5) — they are 2 apart
    a_pos = 3  # '{','d','}','a'
    b_pos = 5  # '-','b'
    assert hybrid_mask[0, 0, a_pos, b_pos] < -1e30, "a should not see b with window=1"
    print(f"  ✓ Hybrid mask: {hybrid_mask.shape}, global anchors work, local window works")

    # Test 7: Coverage loss
    print(f"\n--- Test 7: Coverage loss ---")
    B_c, T_dec, T_enc, H = 2, 8, 12, 4
    # Simulate cross-attention: uniform → no coverage penalty at step 0,
    # increasing coverage as steps progress
    fake_attn = torch.randn(B_c, H, T_dec, T_enc).softmax(dim=-1)
    fake_labels = torch.randint(0, 100, (B_c, T_dec))
    fake_labels[:, -2:] = -100  # pad last 2 positions
    cov_loss = compute_coverage_loss((fake_attn,), fake_labels)
    assert cov_loss.ndim == 0, f"Expected scalar, got shape {cov_loss.shape}"
    assert cov_loss.item() >= 0.0, "Coverage loss should be non-negative"
    print(f"  ✓ Coverage loss: {cov_loss.item():.4f} (scalar, non-negative)")

    # Zero attention → zero coverage loss
    zero_attn = torch.zeros(B_c, H, T_dec, T_enc)
    cov_zero = compute_coverage_loss((zero_attn,), fake_labels)
    assert cov_zero.item() == 0.0, "Zero attention → zero coverage loss"
    print(f"  ✓ Zero attention: loss = {cov_zero.item():.4f}")

    print(f"\n{'=' * 60}")
    print("All tests passed ✓")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    _self_test()
