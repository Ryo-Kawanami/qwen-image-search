import numpy as np

from vlm.vit import VisionTransformer, patchify


def _make_image(size=48):
    rng = np.random.default_rng(0)
    return rng.random((size, size, 3))


def test_patchify_counts():
    img = _make_image(48)
    patches, (gh, gw) = patchify(img, 8)
    assert (gh, gw) == (6, 6)
    assert patches.shape == (36, 8 * 8 * 3)


def test_patchify_rejects_indivisible():
    img = _make_image(48)
    try:
        patchify(img, 7)
    except ValueError:
        return
    raise AssertionError("expected ValueError for indivisible patch size")


def test_vit_forward_shapes():
    vit = VisionTransformer(
        patch_size=8, in_channels=3, d_model=32, n_heads=4, d_ff=64, n_layers=2,
        rng=np.random.default_rng(1),
    )
    hidden, grid, attn = vit.forward(_make_image(48))
    assert grid == (6, 6)
    assert hidden.shape == (1 + 36, 32)  # [CLS] + 36 patches
    assert attn.shape == (4, 37, 37)


def test_cls_attention_map_is_grid_and_normalized():
    vit = VisionTransformer(
        patch_size=8, in_channels=3, d_model=32, n_heads=4, d_ff=64, n_layers=2,
        rng=np.random.default_rng(2),
    )
    amap = vit.cls_attention_map(_make_image(48))
    assert amap.shape == (6, 6)
    assert (amap >= 0).all()
