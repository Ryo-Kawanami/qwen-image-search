"""04: CLIP — 画像とテキストを同じベクトル空間に並べる対照学習。

`04_clip.py` ノートブックの実装を整理したもの。
"""

from __future__ import annotations

import numpy as np

from .attention import softmax

__all__ = [
    "l2_normalize",
    "cosine_sim_matrix",
    "clip_loss",
    "zero_shot_classify",
]


def l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """ベクトルを長さ 1 に正規化する。"""
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def cosine_sim_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """A (n, d) と B (m, d) の全ペアのコサイン類似度 (n, m)。"""
    return l2_normalize(A) @ l2_normalize(B).T


def clip_loss(
    image_feats: np.ndarray, text_feats: np.ndarray, temperature: float = 0.07
) -> tuple[float, np.ndarray]:
    """対称 InfoNCE（CLIP loss）。正しいペアは対角と仮定する。

    Returns
    -------
    loss : float
    logits : (n, n)  コサイン類似度 / temperature
    """
    n = image_feats.shape[0]
    logits = cosine_sim_matrix(image_feats, text_feats) / temperature
    labels = np.arange(n)

    p_i2t = softmax(logits, axis=1)  # 各画像 -> 正しいテキスト
    loss_i2t = -np.mean(np.log(p_i2t[labels, labels] + 1e-9))
    p_t2i = softmax(logits, axis=0)  # 各テキスト -> 正しい画像
    loss_t2i = -np.mean(np.log(p_t2i[labels, labels] + 1e-9))

    return 0.5 * (loss_i2t + loss_t2i), logits


def zero_shot_classify(
    image_feat: np.ndarray,
    label_feats: np.ndarray,
    temperature: float = 0.1,
) -> tuple[int, np.ndarray]:
    """1 枚の画像特徴を、ラベル文特徴の中で最も近いものに分類する。

    Parameters
    ----------
    image_feat : (d,)
    label_feats : (n_labels, d)

    Returns
    -------
    pred_index : int
    probs : (n_labels,)  softmax 確率
    """
    sims = cosine_sim_matrix(image_feat[None, :], label_feats)[0]
    probs = softmax(sims / temperature)
    return int(np.argmax(sims)), probs
