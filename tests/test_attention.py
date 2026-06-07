import torch

from radiant import GQAAttention, build_rope_cache, tiny_config
from radiant.utils import expand_attention_mask


def _rope_for(cfg, S):
    return build_rope_cache(S, cfg.head_dim, theta=cfg.rope_theta)


def test_attention_shape():
    cfg = tiny_config()
    attn = GQAAttention(cfg).eval()
    cos, sin = _rope_for(cfg, 12)
    x = torch.randn(3, 12, cfg.d_model)
    with torch.no_grad():
        y = attn(x, cos, sin, attn_mask=None, is_causal=True)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_gqa_kv_groups():
    cfg = tiny_config(n_query_heads=4, n_kv_heads=2)
    attn = GQAAttention(cfg)
    # k/v projections produce n_kv * head_dim
    assert attn.k_proj.weight.shape[0] == cfg.n_kv_heads * cfg.head_dim
    assert attn.q_proj.weight.shape[0] == cfg.n_query_heads * cfg.head_dim


def test_pad_mask_keeps_padded_query_rows_finite():
    cfg = tiny_config()
    attn = GQAAttention(cfg).eval()
    S = 8
    cos, sin = _rope_for(cfg, S)
    x = torch.randn(2, S, cfg.d_model)
    # Only first 5 tokens are real. The mask must not create all-False rows
    # for padded queries because SDPA can return NaNs for all-masked rows.
    keep = torch.zeros(2, S, dtype=torch.long)
    keep[:, :5] = 1
    mask = expand_attention_mask(keep, batch_size=2, seq_len=S, device=x.device, causal=True)
    assert mask[:, :, 5:, :].any(dim=-1).all()
    with torch.no_grad():
        y = attn(x, cos, sin, attn_mask=mask, is_causal=False)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()
