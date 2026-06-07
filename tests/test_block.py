import torch

from radiant import TransformerBlock, build_rope_cache, tiny_config


def test_dense_block_shape_and_no_aux():
    cfg = tiny_config()
    blk = TransformerBlock(cfg, moe=False).eval()
    cos, sin = build_rope_cache(8, cfg.head_dim)
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        h, aux = blk(x, cos, sin, attn_mask=None, is_causal=True)
    assert h.shape == x.shape
    assert aux is None


def test_moe_block_returns_aux_when_enabled():
    cfg = tiny_config(use_moe=True, n_experts=4, n_active_experts=2)
    blk = TransformerBlock(cfg, moe=True).eval()
    cos, sin = build_rope_cache(8, cfg.head_dim)
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        h, aux = blk(x, cos, sin, attn_mask=None, is_causal=True)
    assert h.shape == x.shape
    assert aux is not None and torch.isfinite(aux)


def test_block_residual_at_init_close_to_input():
    """At init, attn/ffn outputs are small; pre-norm residual makes |h_out - x| << |x|."""
    cfg = tiny_config()
    blk = TransformerBlock(cfg, moe=False).eval()
    cos, sin = build_rope_cache(8, cfg.head_dim)
    x = torch.randn(2, 8, cfg.d_model)
    with torch.no_grad():
        h, _ = blk(x, cos, sin, attn_mask=None, is_causal=True)
    # The residuals are non-zero but the change should be modest.
    rel = (h - x).norm() / x.norm()
    assert rel < 1.0
