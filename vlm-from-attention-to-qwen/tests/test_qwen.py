import numpy as np

from vlm.qwen import (
    apply_mrope,
    apply_rope,
    assign_mrope_positions,
    make_merger,
    n_vision_tokens,
    patch_merger,
)


def test_rope_preserves_norm():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 32))
    y = apply_rope(x, np.arange(5))
    assert np.allclose(np.linalg.norm(x, axis=1), np.linalg.norm(y, axis=1))


def test_rope_score_depends_on_relative_position():
    """位置 m,n を同じだけずらしても q·k が保たれる（相対位置に依存）。"""
    rng = np.random.default_rng(1)
    dim = 32
    q = rng.standard_normal((1, dim))
    k = rng.standard_normal((1, dim))
    # (m, n) = (3, 5) と (10, 12): どちらも相対 -2
    s1 = (apply_rope(q, [3]) @ apply_rope(k, [5]).T)[0, 0]
    s2 = (apply_rope(q, [10]) @ apply_rope(k, [12]).T)[0, 0]
    assert np.isclose(s1, s2, atol=1e-6)


def test_mrope_positions_layout():
    pos, seg = assign_mrope_positions(3, (4, 4), 4)
    assert len(seg) == 3 + 16 + 4
    assert pos.shape == (23, 3)
    # テキスト区間は t=h=w
    text_idx = [i for i, s in enumerate(seg) if s == "text"]
    for i in text_idx:
        assert pos[i, 0] == pos[i, 1] == pos[i, 2]
    # 画像区間では h,w が広がる（2 次元）
    img_idx = [i for i, s in enumerate(seg) if s == "image"]
    hs = pos[img_idx, 1]
    ws = pos[img_idx, 2]
    assert len(set(hs.tolist())) > 1
    assert len(set(ws.tolist())) > 1


def test_mrope_preserves_section_norm():
    rng = np.random.default_rng(2)
    pos, _ = assign_mrope_positions(3, (4, 4), 4)
    x = rng.standard_normal((pos.shape[0], 36))  # 3 セクション×12
    y = apply_mrope(x, pos)
    # 最初のセクション（t）のノルムは回転で不変
    assert np.allclose(
        np.linalg.norm(x[:, :12], axis=1), np.linalg.norm(y[:, :12], axis=1)
    )


def test_dynamic_resolution_scales_with_area():
    small = n_vision_tokens(224, 224)
    large = n_vision_tokens(448, 448)
    assert large > small
    # 一辺 2 倍 -> 面積 4 倍 -> トークン約 4 倍
    assert large == small * 4


def test_patch_merger_reduces_tokens_by_four():
    rng = np.random.default_rng(3)
    d = 16
    feats = rng.standard_normal((8, 8, d))
    W = make_merger(d, rng)
    merged = patch_merger(feats, W)
    assert merged.shape == (4, 4, d)  # 64 -> 16 tokens
