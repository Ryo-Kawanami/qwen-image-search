"""06: Qwen-VL の中核技術 — RoPE / M-RoPE / 動的解像度 / vision merger。

`06_qwen_vl.py` ノートブックの実装を整理したもの。
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "rope_angles",
    "apply_rope",
    "assign_mrope_positions",
    "apply_mrope",
    "n_vision_tokens",
    "patch_merger",
    "make_merger",
]


def rope_angles(positions: np.ndarray, dim: int, base: float = 10000.0) -> np.ndarray:
    """各次元ペアの回転角 (n, dim/2)。θ_i = base^(-2i/dim)。"""
    i = np.arange(0, dim, 2)
    inv_freq = base ** (-(i / dim))
    return positions[:, None] * inv_freq[None, :]


def apply_rope(x: np.ndarray, positions: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """RoPE: x (n, dim) を位置 positions で回転する。dim は偶数。"""
    n, dim = x.shape
    if dim % 2:
        raise ValueError("dim must be even for RoPE")
    ang = rope_angles(np.asarray(positions, dtype=float), dim, base)
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[:, 0::2], x[:, 1::2]
    out = np.empty_like(x, dtype=float)
    out[:, 0::2] = x1 * cos - x2 * sin
    out[:, 1::2] = x1 * sin + x2 * cos
    return out


def assign_mrope_positions(
    n_text_before: int, grid_hw: tuple[int, int], n_text_after: int
) -> tuple[np.ndarray, list[str]]:
    """テキスト -> 画像(gh×gw) -> テキスト に (t, h, w) 位置 ID を割り当てる（Qwen2-VL 簡略版）。

    - テキストトークン: t = h = w = 系列位置（1 次元に退化）
    - 画像トークン: t 固定, h = パッチ行, w = パッチ列（2 次元を保持）
    - 画像後のテキストは、画像が使った最大 ID の次から続ける

    Returns
    -------
    positions : (N, 3) int
    segments : list[str]  各トークンの "text"/"image"
    """
    gh, gw = grid_hw
    positions: list[list[int]] = []
    segments: list[str] = []

    p = 0
    for _ in range(n_text_before):
        positions.append([p, p, p])
        segments.append("text")
        p += 1

    t0 = p
    for r in range(gh):
        for c in range(gw):
            positions.append([t0, t0 + r, t0 + c])
            segments.append("image")

    p = t0 + max(gh, gw)
    for _ in range(n_text_after):
        positions.append([p, p, p])
        segments.append("text")
        p += 1

    return np.array(positions), segments


def apply_mrope(x: np.ndarray, pos3: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """M-RoPE: head 次元を (t, h, w) の 3 セクションに分け、各々の位置で RoPE を回す。

    x : (n, dim)
    pos3 : (n, 3) の (t, h, w) 位置 ID
    """
    n, dim = x.shape
    sec = dim // 3
    sec -= sec % 2  # 各セクションを偶数に
    parts = []
    for k in range(3):
        xs = x[:, k * sec:(k + 1) * sec]
        parts.append(apply_rope(xs, pos3[:, k].astype(float), base))
    rest = x[:, 3 * sec:]  # 割り切れない端数はそのまま
    return np.concatenate(parts + [rest], axis=1)


def n_vision_tokens(
    h_px: int, w_px: int, patch: int = 14, merge: int = 2
) -> int:
    """解像度 -> 視覚トークン数（naive dynamic resolution + merge×merge 統合後）。"""
    gh, gw = h_px // patch, w_px // patch
    return (gh // merge) * (gw // merge)


def make_merger(d: int, rng: np.random.Generator, merge: int = 2) -> np.ndarray:
    """vision merger の射影行列 (merge*merge*d, d)。"""
    fan_in = merge * merge * d
    return rng.standard_normal((fan_in, d)) / np.sqrt(fan_in)


def patch_merger(
    feats_grid: np.ndarray, W_merge: np.ndarray, merge: int = 2
) -> np.ndarray:
    """隣接 merge×merge のパッチ特徴を concat -> 線形射影で 1 トークンに統合。

    feats_grid : (gh, gw, d)
    Returns    : (gh//merge, gw//merge, d)
    """
    gh, gw, d = feats_grid.shape
    gh2, gw2 = gh // merge, gw // merge
    out = np.zeros((gh2, gw2, d))
    for i in range(gh2):
        for j in range(gw2):
            block = feats_grid[
                merge * i:merge * i + merge, merge * j:merge * j + merge, :
            ].reshape(-1)
            out[i, j] = block @ W_merge
    return out
