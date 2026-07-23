import numpy as np

from vlm.multimodal import (
    assemble_sequence,
    build_projector,
    init_llm_block,
    project,
    run_llm,
    simple_vision_encoder,
)


def _make_image(size=32):
    rng = np.random.default_rng(0)
    return rng.random((size, size, 3))


def test_vision_encoder_and_projector_shapes():
    rng = np.random.default_rng(1)
    feats = simple_vision_encoder(_make_image(32), patch_size=8, d_vision=24, rng=rng)
    assert feats.shape == (16, 24)  # 4x4 patches
    proj = build_projector(24, 48, 32, rng)
    tokens = project(feats, proj)
    assert tokens.shape == (16, 32)


def test_assemble_sequence_layout():
    rng = np.random.default_rng(2)
    d_llm = 32
    prefix = rng.standard_normal((2, d_llm))
    image_tokens = rng.standard_normal((16, d_llm))
    suffix = rng.standard_normal((5, d_llm))
    seq = assemble_sequence(
        prefix, image_tokens, suffix, ["<bos>", "USER:"], ["a", "b", "c", "d", "e"]
    )
    assert seq.embeddings.shape == (23, d_llm)
    assert seq.segments.count("image") == 16
    assert seq.image_indices == list(range(2, 18))


def test_run_llm_shapes():
    rng = np.random.default_rng(3)
    d_llm = 32
    seq = rng.standard_normal((23, d_llm))
    blocks = [init_llm_block(d_llm, 64, rng) for _ in range(2)]
    hidden, attn = run_llm(seq, blocks, n_heads=4)
    assert hidden.shape == (23, d_llm)
    assert attn.shape == (4, 23, 23)


def test_assistant_can_attend_to_image_tokens():
    """因果マスク下で、末尾テキスト位置は前方の画像トークンを見られる（重み > 0）。"""
    rng = np.random.default_rng(4)
    d_llm = 16
    prefix = rng.standard_normal((2, d_llm))
    image_tokens = rng.standard_normal((4, d_llm))
    suffix = rng.standard_normal((3, d_llm))
    seq = assemble_sequence(prefix, image_tokens, suffix, ["a", "b"], ["x", "y", "z"])
    blocks = [init_llm_block(d_llm, 32, rng) for _ in range(2)]
    _, attn = run_llm(seq.embeddings, blocks, n_heads=2)
    last_row = attn.mean(axis=0)[-1]  # 末尾トークンの注目
    img_weight = last_row[seq.image_indices].sum()
    assert img_weight > 0.0
