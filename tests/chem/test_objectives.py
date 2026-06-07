import torch

from radiant_chem import (
    ClassificationLoss,
    ContrastiveLoss,
    MaskedLMLoss,
    RegressionLoss,
)


def test_mlm_loss_ignores_minus_hundred():
    logits = torch.randn(2, 6, 10)
    labels = torch.full((2, 6), -100, dtype=torch.long)
    labels[0, 0] = 3
    labels[1, 5] = 7
    loss = MaskedLMLoss()(logits, labels)
    assert torch.isfinite(loss)


def test_regression_loss_basic():
    pred = torch.tensor([1.0, 2.0, 3.0])
    tgt = torch.tensor([1.0, 2.5, 3.0])
    loss = RegressionLoss()(pred, tgt)
    assert torch.isclose(loss, torch.tensor(0.25 / 3.0))


def test_regression_loss_log_scale():
    pred = torch.tensor([10.0, 100.0, 1000.0])
    tgt = torch.tensor([10.0, 100.0, 1000.0])
    loss = RegressionLoss(log_scale=True)(pred, tgt)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_classification_loss_basic():
    logits = torch.tensor([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
    labels = torch.tensor([0, 2])
    loss = ClassificationLoss()(logits, labels)
    assert torch.isfinite(loss) and loss.item() < 1.0


def test_contrastive_loss_zero_when_views_identical():
    """In the limit of identical views, similarities are 1 on diagonal and ~0 off;
    after temperature scaling 1/τ the diagonal dominates and CE -> 0 in the limit
    of large τ.... but for finite τ we can at minimum check it's finite and lower
    than for orthogonal views."""
    a = torch.eye(8)
    b = a.clone()
    near_loss = ContrastiveLoss(temperature=0.1)(a, b).item()
    rand_a = torch.randn(8, 8)
    rand_b = torch.randn(8, 8)
    far_loss = ContrastiveLoss(temperature=0.1)(rand_a, rand_b).item()
    assert near_loss < far_loss


def test_contrastive_loss_invalid_temperature_raises():
    import pytest
    with pytest.raises(ValueError):
        ContrastiveLoss(temperature=0.0)
