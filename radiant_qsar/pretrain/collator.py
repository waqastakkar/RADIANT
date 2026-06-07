"""Collator: SMILES -> token tensors + MLM masking + paired views.

Given a batch of ``(smiles_a, smiles_b)`` pairs from
:class:`CompoundCorpusDataset`, the collator produces a dictionary
suitable for the combined MLM + contrastive objective:

    {
        "mlm_input_ids":      (B, L) -- view A with MLM mask applied
        "mlm_labels":         (B, L) -- ground-truth ids at masked positions, -100 elsewhere
        "mlm_attention_mask": (B, L)
        "view_b_input_ids":   (B, L') -- view B unmasked (used for contrastive)
        "view_b_attention_mask": (B, L')
    }

The MLM masking is BERT-style: a configurable fraction of non-special
positions are selected, of which 80% become ``[MASK]``, 10% are replaced
with a random vocab id, and 10% are kept unchanged.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from radiant_chem import SmilesTokenizer


@dataclass
class MLMContrastiveCollator:
    tokenizer: SmilesTokenizer
    max_len: int = 256
    mlm_mask_prob: float = 0.15
    replace_random_prob: float = 0.10
    keep_orig_prob: float = 0.10
    seed: int = 0

    def __post_init__(self) -> None:
        if not (0 < self.mlm_mask_prob < 1):
            raise ValueError("mlm_mask_prob must be in (0,1)")
        if self.replace_random_prob + self.keep_orig_prob > 1.0:
            raise ValueError("replace+keep <= 1")
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    def _encode(self, smiles_list: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        return self.tokenizer.encode_batch(smiles_list, max_len=self.max_len)

    # ------------------------------------------------------------------
    def _mask(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tok = self.tokenizer
        specials = {tok.pad_id, tok.bos_id, tok.eos_id, tok.mask_id}
        masked = input_ids.clone()
        labels = torch.full_like(input_ids, -100)
        B, L = input_ids.shape
        for b in range(B):
            eligible = [
                j for j in range(L)
                if attention_mask[b, j].item() == 1 and int(input_ids[b, j].item()) not in specials
            ]
            if not eligible:
                continue
            n_mask = max(1, int(round(len(eligible) * self.mlm_mask_prob)))
            chosen = self._rng.sample(eligible, k=min(n_mask, len(eligible)))
            for j in chosen:
                labels[b, j] = input_ids[b, j]
                r = self._rng.random()
                if r < self.replace_random_prob:
                    masked[b, j] = self._rng.randrange(tok.vocab_size)
                elif r < self.replace_random_prob + self.keep_orig_prob:
                    pass
                else:
                    masked[b, j] = tok.mask_id
        return masked, labels

    # ------------------------------------------------------------------
    def __call__(self, batch: list) -> dict[str, torch.Tensor]:
        # Each element is either str, (full, randomized), or
        # (full, randomized, scaffold, rgroup).
        if isinstance(batch[0], (tuple, list)):
            view_a = [b[0] for b in batch]
            view_b = [b[1] for b in batch]
            scaffold_view = [b[2] for b in batch] if len(batch[0]) > 2 else None
            rgroup_view = [b[3] for b in batch] if len(batch[0]) > 3 else None
        else:
            view_a = list(batch)
            view_b = None
            scaffold_view = None
            rgroup_view = None

        a_ids, a_attn = self._encode(view_a)
        a_masked, a_labels = self._mask(a_ids, a_attn)

        out = {
            "mlm_input_ids": a_masked,
            "mlm_labels": a_labels,
            "mlm_attention_mask": a_attn,
        }
        if view_b is not None:
            # Ship the *unmasked* view A for contrastive learning so the
            # embedding isn't biased by [MASK] tokens / random replacements.
            out["view_a_input_ids"] = a_ids
            out["view_a_attention_mask"] = a_attn
            b_ids, b_attn = self._encode(view_b)
            out["view_b_input_ids"] = b_ids
            out["view_b_attention_mask"] = b_attn
        if scaffold_view is not None:
            s_ids, s_attn = self._encode(scaffold_view)
            out["scaffold_input_ids"] = s_ids
            out["scaffold_attention_mask"] = s_attn
        if rgroup_view is not None:
            rg_ids, rg_attn = self._encode(rgroup_view)
            rg_masked, rg_labels = self._mask(rg_ids, rg_attn)
            out["rgroup_input_ids"] = rg_ids
            out["rgroup_attention_mask"] = rg_attn
            out["rgroup_mlm_input_ids"] = rg_masked
            out["rgroup_mlm_labels"] = rg_labels
            out["rgroup_mlm_attention_mask"] = rg_attn
        return out
