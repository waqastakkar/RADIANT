"""Combined pretraining objective: MLM + contrastive.

Given a model that exposes both an LM head (``forward_mlm``) and a
pooled-embedding helper (``embed_pooled``), this function computes::

    loss = mlm_weight * cross_entropy(masked_logits, labels)
         + contrastive_weight * InfoNCE(pool(view_a), pool(view_b))

The contrastive term uses a fresh, *unmasked* forward of view A so it
isn't biased by the MLM mask pattern.
"""

from __future__ import annotations

import torch

from radiant_chem.objectives import ContrastiveLoss, MaskedLMLoss


def combined_pretrain_loss(
    model,
    batch: dict[str, torch.Tensor],
    *,
    n_loops: int | None = None,
    mlm_weight: float = 1.0,
    contrastive_weight: float = 0.1,
    scaffold_contrastive_weight: float = 0.05,
    rgroup_contrastive_weight: float = 0.05,
    rgroup_mlm_weight: float = 0.25,
    contrastive_temperature: float = 0.07,
) -> tuple[torch.Tensor, dict[str, float]]:
    mlm_loss_fn = MaskedLMLoss()
    contrastive_loss_fn = ContrastiveLoss(temperature=contrastive_temperature)

    # 1. MLM forward.
    mlm_logits = model.forward_mlm(
        batch["mlm_input_ids"],
        attention_mask=batch["mlm_attention_mask"],
        n_loops=n_loops,
    )
    loss_mlm = mlm_loss_fn(mlm_logits, batch["mlm_labels"])

    metrics = {"loss_mlm": float(loss_mlm.detach().item())}
    total = mlm_weight * loss_mlm

    if "view_b_input_ids" in batch and contrastive_weight > 0:
        # 2. Contrastive: fresh forward of unmasked view A and view B.
        # Use the unmasked view A (shipped by the collator) so the
        # contrastive embedding isn't biased by [MASK] / random token
        # replacements from the MLM objective.
        view_a_ids = batch.get("view_a_input_ids", batch["mlm_input_ids"])
        view_a_mask = batch.get("view_a_attention_mask", batch["mlm_attention_mask"])
        emb_a = model.embed_pooled(
            view_a_ids,
            attention_mask=view_a_mask,
            n_loops=n_loops,
        )
        emb_b = model.embed_pooled(
            batch["view_b_input_ids"],
            attention_mask=batch["view_b_attention_mask"],
            n_loops=n_loops,
        )
        loss_contrastive = contrastive_loss_fn(emb_a, emb_b)
        total = total + contrastive_weight * loss_contrastive
        metrics["loss_contrastive"] = float(loss_contrastive.detach().item())

        if "scaffold_input_ids" in batch and scaffold_contrastive_weight > 0:
            emb_scaf = model.embed_pooled(
                batch["scaffold_input_ids"],
                attention_mask=batch["scaffold_attention_mask"],
                n_loops=n_loops,
            )
            loss_scaf = contrastive_loss_fn(emb_a, emb_scaf)
            total = total + scaffold_contrastive_weight * loss_scaf
            metrics["loss_scaffold_contrastive"] = float(loss_scaf.detach().item())

        if "rgroup_input_ids" in batch and rgroup_contrastive_weight > 0:
            emb_rg = model.embed_pooled(
                batch["rgroup_input_ids"],
                attention_mask=batch["rgroup_attention_mask"],
                n_loops=n_loops,
            )
            loss_rg_ctr = contrastive_loss_fn(emb_a, emb_rg)
            total = total + rgroup_contrastive_weight * loss_rg_ctr
            metrics["loss_rgroup_contrastive"] = float(loss_rg_ctr.detach().item())

    if "rgroup_mlm_input_ids" in batch and rgroup_mlm_weight > 0:
        rg_logits = model.forward_mlm(
            batch["rgroup_mlm_input_ids"],
            attention_mask=batch["rgroup_mlm_attention_mask"],
            n_loops=n_loops,
        )
        loss_rg_mlm = mlm_loss_fn(rg_logits, batch["rgroup_mlm_labels"])
        total = total + rgroup_mlm_weight * loss_rg_mlm
        metrics["loss_rgroup_mlm"] = float(loss_rg_mlm.detach().item())

    metrics["loss_total"] = float(total.detach().item())
    return total, metrics
