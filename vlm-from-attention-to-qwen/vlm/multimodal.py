"""05: VLM (LLaVA スタイル) — 視覚特徴を projector で LLM に流し込む。

`05_vlm_llava.py` ノートブックの実装を整理したもの。
視覚エンコーダ -> projector -> テキストと連結 -> 因果 LLM、という流れを組む。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention import causal_mask, multi_head_attention
from .transformer import feed_forward, gelu, init_block_params, layer_norm

__all__ = [
    "simple_vision_encoder",
    "build_projector",
    "project",
    "assemble_sequence",
    "run_llm",
    "MultimodalSequence",
]


def simple_vision_encoder(
    img: np.ndarray, patch_size: int, d_vision: int, rng: np.random.Generator
) -> np.ndarray:
    """画像 -> パッチ特徴 (n_patch, d_vision)。ランダム重み（本物は CLIP ViT）。"""
    H, W, C = img.shape
    P = patch_size
    gh, gw = H // P, W // P
    patches = np.stack(
        [
            img[i * P:(i + 1) * P, j * P:(j + 1) * P, :].reshape(-1)
            for i in range(gh)
            for j in range(gw)
        ]
    )
    W_enc = rng.standard_normal((patches.shape[1], d_vision)) / np.sqrt(patches.shape[1])
    return np.tanh(patches @ W_enc)


def build_projector(
    d_vision: int, d_hidden: int, d_llm: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    """LLaVA の 2 層 MLP projector（視覚特徴 -> LLM 埋め込み空間）。"""
    return {
        "W1": rng.standard_normal((d_vision, d_hidden)) / np.sqrt(d_vision),
        "b1": np.zeros(d_hidden),
        "W2": rng.standard_normal((d_hidden, d_llm)) / np.sqrt(d_hidden),
        "b2": np.zeros(d_llm),
    }


def project(feats: np.ndarray, proj: dict[str, np.ndarray]) -> np.ndarray:
    """パッチ特徴を LLM 空間の画像トークンへ翻訳する。"""
    return gelu(feats @ proj["W1"] + proj["b1"]) @ proj["W2"] + proj["b2"]


@dataclass
class MultimodalSequence:
    """組み立て済みのマルチモーダル系列。"""

    embeddings: np.ndarray  # (seq_len, d_llm)
    segments: list[str]     # 各位置が "text" か "image" か
    labels: list[str]       # 表示用ラベル

    @property
    def image_indices(self) -> list[int]:
        return [i for i, s in enumerate(self.segments) if s == "image"]


def assemble_sequence(
    prefix_embed: np.ndarray,
    image_tokens: np.ndarray,
    suffix_embed: np.ndarray,
    prefix_labels: list[str],
    suffix_labels: list[str],
) -> MultimodalSequence:
    """テキスト前 + 画像トークン + テキスト後 を 1 列に連結する。"""
    embeddings = np.concatenate([prefix_embed, image_tokens, suffix_embed], axis=0)
    segments = (
        ["text"] * len(prefix_embed)
        + ["image"] * len(image_tokens)
        + ["text"] * len(suffix_embed)
    )
    labels = (
        list(prefix_labels)
        + [f"img{p}" for p in range(len(image_tokens))]
        + list(suffix_labels)
    )
    return MultimodalSequence(embeddings, segments, labels)


def run_llm(
    seq: np.ndarray,
    blocks: list[dict[str, np.ndarray]],
    n_heads: int,
) -> tuple[np.ndarray, np.ndarray]:
    """組み立て済み系列を因果 Transformer に通す。

    Returns
    -------
    hidden : (seq_len, d_llm)
    last_attn : (n_heads, seq_len, seq_len)  最終層の attention
    """
    n = seq.shape[0]
    mask = causal_mask(n)
    x = seq
    last_attn = None
    for bi, p in enumerate(blocks):
        ln1 = layer_norm(x, p["g1"], p["b1"])
        if bi == len(blocks) - 1:
            a, last_attn = multi_head_attention(
                ln1, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask, True
            )
        else:
            a = multi_head_attention(
                ln1, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads, mask
            )
        x = x + a
        ln2 = layer_norm(x, p["g2"], p["b2"])
        x = x + feed_forward(ln2, p["W1"], p["bf1"], p["W2"], p["bf2"])
    return x, last_attn


# init_block_params を再エクスポート（LLM ブロックの初期化に使う）
init_llm_block = init_block_params
__all__.append("init_llm_block")
