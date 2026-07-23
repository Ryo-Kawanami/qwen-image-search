"""01: Attention — VLM のすべての土台。

`01_attention.py` ノートブックの実装を、再利用できる関数・クラスに整理したもの。
すべて NumPy。学習済み重みは持たず、投影行列は外から渡す（または `init_mha_params` で初期化）。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "softmax",
    "scaled_dot_product_attention",
    "causal_mask",
    "init_mha_params",
    "multi_head_attention",
    "MultiHeadAttention",
]


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """数値的に安定な softmax（最大値を引いてから exp）。"""
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Attention(Q, K, V) = softmax(Q Kᵀ / √d_k) V。

    Parameters
    ----------
    Q : (n_q, d_k)   探す側
    K : (n_k, d_k)   見出し
    V : (n_k, d_v)   中身
    mask : (n_q, n_k) の bool。True の位置を -inf にして「見せない」。

    Returns
    -------
    out : (n_q, d_v)   加重平均された Value
    weights : (n_q, n_k)   attention 重み（各行が確率分布）
    """
    d_k = Q.shape[-1]
    scores = (Q @ K.T) / np.sqrt(d_k)
    if mask is not None:
        scores = np.where(mask, -np.inf, scores)
    weights = softmax(scores, axis=-1)
    out = weights @ V
    return out, weights


def causal_mask(n: int) -> np.ndarray:
    """(n, n) の因果マスク。True（上三角＝未来）の位置は attention から隠される。"""
    return np.triu(np.ones((n, n), dtype=bool), k=1)


def init_mha_params(
    d_model: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """Multi-Head Attention 用の投影行列 Wq, Wk, Wv, Wo をランダム初期化。"""
    s = 1.0 / np.sqrt(d_model)
    return {
        "Wq": rng.standard_normal((d_model, d_model)) * s,
        "Wk": rng.standard_normal((d_model, d_model)) * s,
        "Wv": rng.standard_normal((d_model, d_model)) * s,
        "Wo": rng.standard_normal((d_model, d_model)) * s,
    }


def multi_head_attention(
    X: np.ndarray,
    Wq: np.ndarray,
    Wk: np.ndarray,
    Wv: np.ndarray,
    Wo: np.ndarray,
    n_heads: int,
    mask: np.ndarray | None = None,
    return_weights: bool = False,
):
    """Multi-Head Self-Attention。

    X: (n, d_model)。d_model は n_heads で割り切れること。
    return_weights=True で head ごとの重み (n_heads, n, n) も返す。
    """
    n, d_model = X.shape
    if d_model % n_heads != 0:
        raise ValueError(f"d_model={d_model} must be divisible by n_heads={n_heads}")
    d_head = d_model // n_heads

    def split(M: np.ndarray) -> np.ndarray:  # (n, d_model) -> (n_heads, n, d_head)
        return M.reshape(n, n_heads, d_head).transpose(1, 0, 2)

    Qh, Kh, Vh = split(X @ Wq), split(X @ Wk), split(X @ Wv)

    outs, weights = [], []
    for h in range(n_heads):
        o, w = scaled_dot_product_attention(Qh[h], Kh[h], Vh[h], mask)
        outs.append(o)
        weights.append(w)

    out = np.concatenate(outs, axis=-1) @ Wo  # (n, d_model)
    if return_weights:
        return out, np.stack(weights)
    return out


class MultiHeadAttention:
    """重みを保持する薄いラッパ。`params` を与えなければランダム初期化する。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        rng: np.random.Generator | None = None,
        params: dict[str, np.ndarray] | None = None,
    ) -> None:
        if params is None:
            rng = rng or np.random.default_rng(0)
            params = init_mha_params(d_model, rng)
        self.d_model = d_model
        self.n_heads = n_heads
        self.params = params

    def __call__(
        self, X: np.ndarray, mask: np.ndarray | None = None, return_weights: bool = False
    ):
        p = self.params
        return multi_head_attention(
            X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], self.n_heads, mask, return_weights
        )
