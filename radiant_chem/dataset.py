"""ChEMBL-shaped CSV dataset loaders.

Two flavors:

* :class:`ChemblCsvDataset` -- loads ``(smiles, target_columns...)`` rows
  from a CSV and yields ``(input_ids, attention_mask, targets)``. Suitable
  for property prediction.
* :class:`MaskedSmilesDataset` -- yields ``(input_ids, mask_positions,
  labels, attention_mask)`` for masked-token modeling. Wraps an underlying
  iterable of tokenized SMILES.

Neither uses pandas; both rely on the standard-library ``csv`` module so
the package installs cleanly with only torch + numpy.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch.utils.data import Dataset

from radiant_chem.tokenizer import SmilesTokenizer


class ChemblCsvDataset(Dataset):
    """A CSV file with one SMILES column and zero-or-more target columns.

    The dataset reads the entire CSV at construction (intentional: ChEMBL
    property tables fit in memory). Rows whose SMILES fail to tokenize
    against ``tokenizer`` will still tokenize -- unknown atoms become
    ``[UNK]``. Rows with non-floatable targets are skipped with a count.
    """

    def __init__(
        self,
        csv_path: str | Path,
        tokenizer: SmilesTokenizer,
        *,
        smiles_column: str = "smiles",
        target_columns: Sequence[str] = (),
        max_len: int = 256,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> None:
        self.tokenizer = tokenizer
        self.smiles_column = smiles_column
        self.target_columns = list(target_columns)
        self.max_len = max_len
        self.add_bos = add_bos
        self.add_eos = add_eos

        self.smiles: list[str] = []
        self.targets: list[list[float]] = []
        self.skipped = 0

        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if smiles_column not in (reader.fieldnames or []):
                raise KeyError(
                    f"Column {smiles_column!r} not found in CSV; available: "
                    f"{reader.fieldnames}"
                )
            for row in reader:
                s = row[smiles_column]
                if not s:
                    self.skipped += 1
                    continue
                try:
                    tgt = [float(row[c]) for c in self.target_columns]
                except (ValueError, KeyError):
                    self.skipped += 1
                    continue
                self.smiles.append(s)
                self.targets.append(tgt)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.tokenizer.encode(
            self.smiles[idx],
            add_bos=self.add_bos,
            add_eos=self.add_eos,
            max_len=self.max_len,
        )
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[idx], dtype=torch.float32),
        }

    def collate(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        L = max(item["input_ids"].size(0) for item in batch)
        B = len(batch)
        input_ids = torch.full((B, L), self.tokenizer.pad_id, dtype=torch.long)
        attn = torch.zeros((B, L), dtype=torch.long)
        targets = torch.stack([item["targets"] for item in batch], dim=0)
        for i, item in enumerate(batch):
            n = item["input_ids"].size(0)
            input_ids[i, :n] = item["input_ids"]
            attn[i, :n] = 1
        return {"input_ids": input_ids, "attention_mask": attn, "targets": targets}


def make_mlm_batch(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    tokenizer: SmilesTokenizer,
    mask_prob: float = 0.15,
    replace_random_prob: float = 0.10,
    keep_orig_prob: float = 0.10,
    rng: random.Random | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """BERT-style MLM masking.

    Of the ``mask_prob`` fraction of *non-pad / non-special* tokens chosen
    for masking:
      * ``replace_random_prob`` -> replaced with a random vocab id
      * ``keep_orig_prob``       -> left unchanged
      * remainder                 -> replaced with ``[MASK]``

    Returns ``(masked_input_ids, mask_positions, labels)``. ``labels`` is
    ``input_ids`` at masked positions and ``-100`` everywhere else (the
    standard "ignore" sentinel for ``F.cross_entropy``).
    """
    if rng is None:
        rng = random.Random()
    masked = input_ids.clone()
    labels = torch.full_like(input_ids, -100)

    specials = {tokenizer.pad_id, tokenizer.bos_id, tokenizer.eos_id, tokenizer.mask_id}

    B, L = input_ids.shape
    mask_positions = torch.zeros_like(input_ids, dtype=torch.bool)
    for b in range(B):
        eligible = [
            j for j in range(L)
            if attention_mask[b, j].item() == 1 and int(input_ids[b, j].item()) not in specials
        ]
        if not eligible:
            continue
        n_mask = max(1, int(round(len(eligible) * mask_prob)))
        chosen = rng.sample(eligible, k=min(n_mask, len(eligible)))
        for j in chosen:
            labels[b, j] = input_ids[b, j]
            mask_positions[b, j] = True
            r = rng.random()
            if r < replace_random_prob:
                masked[b, j] = rng.randrange(tokenizer.vocab_size)
            elif r < replace_random_prob + keep_orig_prob:
                pass  # keep
            else:
                masked[b, j] = tokenizer.mask_id
    return masked, mask_positions, labels


class MaskedSmilesDataset(Dataset):
    """Wraps an iterable of tokenized SMILES (already as id lists) for MLM training."""

    def __init__(
        self,
        smiles: Iterable[str],
        tokenizer: SmilesTokenizer,
        *,
        max_len: int = 256,
        mask_prob: float = 0.15,
        replace_random_prob: float = 0.10,
        keep_orig_prob: float = 0.10,
        seed: int = 0,
    ) -> None:
        self.smiles = list(smiles)
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.mask_prob = mask_prob
        self.replace_random_prob = replace_random_prob
        self.keep_orig_prob = keep_orig_prob
        self.rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = torch.tensor(
            self.tokenizer.encode(self.smiles[idx], max_len=self.max_len),
            dtype=torch.long,
        )
        attn = torch.ones_like(ids)
        masked, positions, labels = make_mlm_batch(
            ids.unsqueeze(0),
            attn.unsqueeze(0),
            tokenizer=self.tokenizer,
            mask_prob=self.mask_prob,
            replace_random_prob=self.replace_random_prob,
            keep_orig_prob=self.keep_orig_prob,
            rng=self.rng,
        )
        return {
            "input_ids": masked.squeeze(0),
            "labels": labels.squeeze(0),
            "mask_positions": positions.squeeze(0),
            "attention_mask": attn,
        }

    def collate(
        self, batch: list[dict[str, torch.Tensor]]
    ) -> dict[str, torch.Tensor]:
        L = max(item["input_ids"].size(0) for item in batch)
        B = len(batch)
        ids = torch.full((B, L), self.tokenizer.pad_id, dtype=torch.long)
        labels = torch.full((B, L), -100, dtype=torch.long)
        positions = torch.zeros((B, L), dtype=torch.bool)
        attn = torch.zeros((B, L), dtype=torch.long)
        for i, item in enumerate(batch):
            n = item["input_ids"].size(0)
            ids[i, :n] = item["input_ids"]
            labels[i, :n] = item["labels"]
            positions[i, :n] = item["mask_positions"]
            attn[i, :n] = item["attention_mask"]
        return {
            "input_ids": ids,
            "labels": labels,
            "mask_positions": positions,
            "attention_mask": attn,
        }
