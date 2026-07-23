import numpy as np

from vlm.transformer import (
    TinyGPT,
    init_block_params,
    layer_norm,
    sinusoidal_positional_encoding,
    transformer_block,
)


def test_layer_norm_normalizes():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 16)) * 5 + 3
    y = layer_norm(x, np.ones(16), np.zeros(16))
    assert np.allclose(y.mean(axis=-1), 0.0, atol=1e-6)
    assert np.allclose(y.std(axis=-1), 1.0, atol=1e-3)


def test_positional_encoding_shape_and_range():
    pe = sinusoidal_positional_encoding(50, 64)
    assert pe.shape == (50, 64)
    assert pe.min() >= -1.0 and pe.max() <= 1.0


def test_transformer_block_residual_shape():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((8, 32))
    p = init_block_params(32, 64, rng)
    y = transformer_block(x, p, n_heads=4)
    assert y.shape == x.shape


def test_transformer_block_returns_weights():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((5, 16))
    p = init_block_params(16, 32, rng)
    y, w = transformer_block(x, p, n_heads=2, return_weights=True)
    assert y.shape == (5, 16)
    assert w.shape == (2, 5, 5)


def test_tiny_gpt_forward():
    gpt = TinyGPT(vocab_size=20, d_model=32, n_heads=4, d_ff=64, n_layers=3,
                  rng=np.random.default_rng(3))
    h = gpt.forward([3, 1, 4, 1, 5, 9, 2, 6])
    assert h.shape == (8, 32)
    assert np.isfinite(h).all()
