"""02: Transformer — Attention をブロックに組み上げ、言語モデルの背骨を作る。

`02_transformer.py` ノートブックの実装を整理したもの。Pre-LN 構成。
"""

from __future__ import annotations

import numpy as np

from .attention import causal_mask, init_mha_params, multi_head_attention

__all__ = [
    "gelu",
    "layer_norm",
    "feed_forward",
    "sinusoidal_positional_encoding",
    "init_block_params",
    "transformer_block",
    "TinyGPT",
]


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU（tanh 近似）。"""
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def layer_norm(
    x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> np.ndarray:
    """特徴次元方向の Layer Normalization。"""
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return gamma * (x - mu) / np.sqrt(var + eps) + beta


def feed_forward(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """位置ごとの FFN: GELU(x W1 + b1) W2 + b2。"""
    return gelu(x @ W1 + b1) @ W2 + b2


def sinusoidal_positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """オリジナル Transformer の sin/cos 位置エンコーディング (seq_len, d_model)。"""
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle = pos / np.power(10000, (2 * (i // 2)) / d_model)
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])
    return pe


def init_block_params(
    d_model: int, d_ff: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Transformer ブロック 1 段分の重み（attention + FFN + 2 つの LayerNorm）。"""
    s = 1.0 / np.sqrt(d_model)
    params = init_mha_params(d_model, rng)
    params.update(
        g1=np.ones(d_model), b1=np.zeros(d_model),  # LN (attention 前)
        g2=np.ones(d_model), b2=np.zeros(d_model),  # LN (FFN 前)
        W1=rng.standard_normal((d_model, d_ff)) * s, bf1=np.zeros(d_ff),
        W2=rng.standard_normal((d_ff, d_model)) / np.sqrt(d_ff), bf2=np.zeros(d_model),
    )
    return params


def transformer_block(
    x: np.ndarray,
    p: dict[str, np.ndarray],
    n_heads: int,
    mask: np.ndarray | None = None,
    return_weights: bool = False,
):
    """Pre-LN Transformer ブロック 1 段。

    x = x + MHA(LN(x));  x = x + FFN(LN(x))
    """
    ln1 = layer_norm(x, p["g1"], p["b1"])
    attn = multi_head_attention(
        ln1, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask, return_weights
    )
    if return_weights:
        attn, weights = attn
    x = x + attn
    ln2 = layer_norm(x, p["g2"], p["b2"])
    x = x + feed_forward(ln2, p["W1"], p["bf1"], p["W2"], p["bf2"])
    return (x, weights) if return_weights else x


class TinyGPT:
    """位置エンコーディング付き・因果マスクありの小型 GPT（デコーダのみ）。"""

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng(0)
        self.d_model = d_model
        self.n_heads = n_heads
        self.embed = rng.standard_normal((vocab_size, d_model)) * 0.1
        self.blocks = [init_block_params(d_model, d_ff, rng) for _ in range(n_layers)]

    def forward(self, token_ids: list[int] | np.ndarray) -> np.ndarray:
        """トークン ID 列 -> 各トークンの隠れ状態 (n, d_model)。"""
        token_ids = np.asarray(token_ids)
        n = len(token_ids)
        x = self.embed[token_ids]
        x = x + sinusoidal_positional_encoding(n, self.d_model)
        mask = causal_mask(n)
        for p in self.blocks:
            x = transformer_block(x, p, self.n_heads, mask)
        return x
