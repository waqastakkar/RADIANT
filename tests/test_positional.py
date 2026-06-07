import pytest
import torch

from radiant import apply_rope, build_rope_cache


def test_cache_shape():
    cos, sin = build_rope_cache(32, 16)
    assert cos.shape == (32, 8)
    assert sin.shape == (32, 8)


def test_apply_rope_preserves_shape_and_norm():
    B, S, H, D = 2, 5, 3, 16
    cos, sin = build_rope_cache(S, D)
    x = torch.randn(B, S, H, D)
    y = apply_rope(x, cos, sin)
    assert y.shape == x.shape
    # Rotation preserves per-token norm because each (a,b) pair is rotated.
    assert torch.allclose(y.norm(dim=-1), x.norm(dim=-1), atol=1e-5)


def test_apply_rope_position_zero_is_identity():
    B, S, H, D = 1, 1, 1, 8
    cos, sin = build_rope_cache(1, D)  # angle = 0; cos=1, sin=0
    x = torch.randn(B, S, H, D)
    y = apply_rope(x, cos, sin)
    assert torch.allclose(y, x, atol=1e-6)


def test_odd_head_dim_rejected():
    with pytest.raises(ValueError):
        build_rope_cache(8, 7)


def test_short_cache_rejected():
    cos, sin = build_rope_cache(4, 8)
    x = torch.randn(1, 6, 1, 8)
    with pytest.raises(ValueError):
        apply_rope(x, cos, sin)
