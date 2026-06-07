import pytest
import torch

from radiant import RadiantModel, tiny_config


@pytest.mark.parametrize("B,S", [(1, 4), (2, 7), (4, 16)])
def test_forward_shape(B, S):
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (B, S))
    with torch.no_grad():
        out = model(x)
    assert out.logits.shape == (B, S, cfg.vocab_size)
    assert out.last_hidden_state.shape == (B, S, cfg.d_model)


def test_seq_len_too_long_raises():
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, cfg.max_seq_len + 1))
    with pytest.raises(ValueError):
        model(x)


def test_with_pad_attention_mask():
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 8))
    mask = torch.ones(2, 8, dtype=torch.long)
    mask[0, 5:] = 0  # row 0 has 5 valid tokens
    with torch.no_grad():
        out = model(x, attention_mask=mask)
    assert torch.isfinite(out.logits).all()


def test_with_moe_in_stem_returns_aux():
    cfg = tiny_config(use_moe=True, moe_placement="stem_exit", n_experts=4, n_active_experts=2)
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (2, 6))
    with torch.no_grad():
        out = model(x)
    assert len(out.aux_losses) > 0
    assert out.aux_loss is not None and torch.isfinite(out.aux_loss)


def test_no_moe_no_aux():
    cfg = tiny_config()
    model = RadiantModel(cfg).eval()
    x = torch.randint(0, cfg.vocab_size, (1, 4))
    with torch.no_grad():
        out = model(x)
    assert out.aux_losses == []
    assert out.aux_loss is None


def test_tied_embedding_weight_shared():
    cfg = tiny_config(tie_word_embeddings=True)
    model = RadiantModel(cfg).eval()
    assert model.lm_head.proj.weight.data_ptr() == model.stem.token_embed.weight.data_ptr()


def test_untied_embedding_weight_separate():
    cfg = tiny_config(tie_word_embeddings=False)
    model = RadiantModel(cfg).eval()
    assert model.lm_head.proj.weight.data_ptr() != model.stem.token_embed.weight.data_ptr()
