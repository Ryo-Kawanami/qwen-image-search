"""03: Vision Transformer — 画像を「パッチのトークン列」にして Transformer に通す。

`03_vision_transformer.py` ノートブックの実装を整理したもの。
"""

from __future__ import annotations

import numpy as np

from .transformer import init_block_params, transformer_block

__all__ = ["patchify", "VisionTransformer"]


def patchify(img: np.ndarray, patch_size: int) -> tuple[np.ndarray, tuple[int, int]]:
    """画像 (H, W, C) を patch_size 角のパッチに切り、平坦化する。

    Returns
    -------
    patches : (n_patch, patch_size*patch_size*C)
    grid : (gh, gw)  縦横のパッチ数
    """
    H, W, C = img.shape
    P = patch_size
    if H % P or W % P:
        raise ValueError(f"image {H}x{W} not divisible by patch_size {P}")
    gh, gw = H // P, W // P
    patches = [
        img[i * P:(i + 1) * P, j * P:(j + 1) * P, :].reshape(-1)
        for i in range(gh)
        for j in range(gw)
    ]
    return np.stack(patches), (gh, gw)


class VisionTransformer:
    """最小構成の ViT: patch embedding + [CLS] + 学習型位置埋め込み + 双方向 Transformer。

    学習はせず、重みはランダム初期化。forward は各トークンの隠れ状態と、
    最終層の attention 重みを返す（CLS の注目可視化などに使う）。
    """

    def __init__(
        self,
        patch_size: int,
        in_channels: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        rng = rng or np.random.default_rng(0)
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_heads = n_heads
        patch_dim = patch_size * patch_size * in_channels
        self._patch_dim = patch_dim
        self.W_embed = rng.standard_normal((patch_dim, d_model)) / np.sqrt(patch_dim)
        self.cls = rng.standard_normal((1, d_model)) * 0.02
        self.blocks = [init_block_params(d_model, d_ff, rng) for _ in range(n_layers)]
        self._rng = rng

    def forward(self, img: np.ndarray):
        """画像 -> (hidden (1+n_patch, d_model), grid, last_attn (n_heads, seq, seq))。"""
        patches, grid = patchify(img, self.patch_size)
        tokens = patches @ self.W_embed
        tokens = np.concatenate([self.cls, tokens], axis=0)
        # 学習型位置埋め込み（ここではランダム初期化で代用）
        pos = self._rng.standard_normal(tokens.shape) * 0.02
        x = tokens + pos

        last_attn = None
        for bi, p in enumerate(self.blocks):
            if bi == len(self.blocks) - 1:
                x, last_attn = transformer_block(
                    x, p, self.n_heads, mask=None, return_weights=True
                )
            else:
                x = transformer_block(x, p, self.n_heads, mask=None)
        return x, grid, last_attn

    def cls_attention_map(self, img: np.ndarray) -> np.ndarray:
        """[CLS] が各パッチへ向ける attention を (gh, gw) の 2D マップで返す。"""
        _, (gh, gw), last_attn = self.forward(img)
        cls_attn = last_attn[:, 0, 1:].mean(axis=0)  # head 平均、CLS 自身を除く
        return cls_attn.reshape(gh, gw)
