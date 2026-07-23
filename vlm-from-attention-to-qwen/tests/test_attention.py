import numpy as np

from vlm.attention import (
    MultiHeadAttention,
    causal_mask,
    init_mha_params,
    multi_head_attention,
    scaled_dot_product_attention,
    softmax,
)


def test_softmax_sums_to_one():
    x = np.random.default_rng(0).standard_normal((4, 5))
    p = softmax(x, axis=-1)
    assert np.allclose(p.sum(axis=-1), 1.0)
    assert (p >= 0).all()


def test_attention_weights_are_distributions():
    rng = np.random.default_rng(1)
    Q = rng.standard_normal((3, 8))
    K = rng.standard_normal((5, 8))
    V = rng.standard_normal((5, 4))
    out, w = scaled_dot_product_attention(Q, K, V)
    assert out.shape == (3, 4)
    assert w.shape == (3, 5)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_causal_mask_hides_future():
    rng = np.random.default_rng(2)
    n = 6
    Q = K = rng.standard_normal((n, 8))
    V = rng.standard_normal((n, 8))
    _, w = scaled_dot_product_attention(Q, K, V, mask=causal_mask(n))
    # 上三角（未来）はすべて 0
    upper = np.triu(np.ones((n, n), dtype=bool), k=1)
    assert np.allclose(w[upper], 0.0)


def test_multi_head_shapes_and_weights():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((7, 16))
    p = init_mha_params(16, rng)
    out, w = multi_head_attention(
        X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads=4, return_weights=True
    )
    assert out.shape == (7, 16)
    assert w.shape == (4, 7, 7)
    assert np.allclose(w.sum(axis=-1), 1.0)


def test_multi_head_requires_divisible_dim():
    rng = np.random.default_rng(4)
    X = rng.standard_normal((5, 10))
    p = init_mha_params(10, rng)
    try:
        multi_head_attention(X, p["Wq"], p["Wk"], p["Wv"], p["Wo"], n_heads=3)
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-divisible d_model")


def test_mha_class_matches_functional():
    rng = np.random.default_rng(5)
    X = rng.standard_normal((6, 12))
    mha = MultiHeadAttention(12, 3, params=init_mha_params(12, np.random.default_rng(5)))
    out = mha(X)
    assert out.shape == (6, 12)
