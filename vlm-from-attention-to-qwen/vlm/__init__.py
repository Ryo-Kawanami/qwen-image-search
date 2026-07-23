"""vlm — AttentionからQwen-VLまでを NumPy でゼロから実装した学習用ライブラリ。

`notebooks/` の各章に対応するモジュール:

    attention   01: scaled dot-product / multi-head attention, causal mask
    transformer 02: LayerNorm, FFN, 位置エンコーディング, Transformerブロック, TinyGPT
    vit         03: patchify, VisionTransformer
    clip        04: L2正規化, コサイン類似度, CLIP loss, ゼロショット分類
    multimodal  05: 視覚エンコーダ, projector, マルチモーダル系列, 因果LLM
    qwen        06: RoPE, M-RoPE, 動的解像度, vision merger

すべて学習済み重みは持たず、乱数初期化した重みで「形と仕組み」を確認するための実装。
"""

from __future__ import annotations

from . import attention, clip, multimodal, qwen, transformer, vit

__all__ = ["attention", "transformer", "vit", "clip", "multimodal", "qwen"]

__version__ = "0.1.0"
