import numpy as np

from vlm.clip import clip_loss, cosine_sim_matrix, l2_normalize, zero_shot_classify


def test_l2_normalize_unit_length():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((5, 8))
    n = l2_normalize(x)
    assert np.allclose(np.linalg.norm(n, axis=-1), 1.0)


def test_cosine_sim_range():
    rng = np.random.default_rng(1)
    A = rng.standard_normal((4, 8))
    B = rng.standard_normal((6, 8))
    S = cosine_sim_matrix(A, B)
    assert S.shape == (4, 6)
    assert S.min() >= -1.0001 and S.max() <= 1.0001


def test_aligned_features_have_lower_loss_than_random():
    rng = np.random.default_rng(2)
    n, d = 6, 16
    shared = rng.standard_normal((n, d))
    img_a = shared + 0.1 * rng.standard_normal((n, d))
    txt_a = shared + 0.1 * rng.standard_normal((n, d))
    img_r = rng.standard_normal((n, d))
    txt_r = rng.standard_normal((n, d))
    loss_a, _ = clip_loss(img_a, txt_a)
    loss_r, _ = clip_loss(img_r, txt_r)
    assert loss_a < loss_r


def test_zero_shot_picks_nearest_label():
    rng = np.random.default_rng(3)
    d = 16
    labels = rng.standard_normal((3, d))
    # クエリはラベル 1 に近い
    query = labels[1] + 0.05 * rng.standard_normal(d)
    pred, probs = zero_shot_classify(query, labels)
    assert pred == 1
    assert np.allclose(probs.sum(), 1.0)
