"""SMILES / SELFIES tokenizer.

The default :class:`SmilesTokenizer` is an atom-level regex tokenizer with
no external dependencies. It splits a SMILES string into tokens of these
kinds (in priority order):

1. Two-letter atom symbols (Cl, Br) and their lowercase aromatic variants
   ('cl', 'br' don't occur in SMILES but the regex covers all length-2
   organic-subset cases).
2. Bracketed atoms like ``[NH4+]``, ``[13C]``, ``[C@@H]`` -- treated atomically.
3. Single-character atoms / bonds / structure characters ``- = # / \ ( ) % 0-9``.

Special tokens:

    [PAD]  -- padding (id 0 by default)
    [BOS]  -- beginning of sequence
    [EOS]  -- end of sequence
    [MASK] -- masked-token target
    [UNK]  -- unknown character

A :class:`SelfiesTokenizer` is exposed but only usable when the optional
``selfies`` package is installed; instantiating it without that package
raises ``ImportError``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import torch


# Greedy two-letter atoms first, then bracketed groups, then single chars.
# Note: Cl/Br aren't lowercase in SMILES, but uppercase is what we need.
_SMILES_TOKEN_RE = re.compile(
    r"(?P<atom2>Cl|Br)"        # two-letter elements
    r"|(?P<bracket>\[[^\]]+\])"  # bracketed (any length)
    r"|(?P<other>%[0-9]{2}|.)"   # %nn for 2-digit ring closures, or any single char
)


SPECIAL_TOKENS: tuple[str, ...] = ("[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]")


def smiles_tokenize(s: str) -> list[str]:
    """Atom-level token list for ``s`` (no special tokens)."""
    out: list[str] = []
    for m in _SMILES_TOKEN_RE.finditer(s):
        out.append(m.group(0))
    # Sanity: re-joining must reconstruct the input exactly.
    return out


class SmilesTokenizer:
    """Atom-level SMILES tokenizer with a learned vocab.

    Build the vocab from a corpus with :py:meth:`build_vocab` or
    :py:meth:`from_corpus`, then use :py:meth:`encode` / :py:meth:`decode`
    or the higher-level :py:meth:`encode_batch`.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        if vocab is None:
            self.token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        else:
            self.token_to_id = dict(vocab)
            for tok in SPECIAL_TOKENS:
                if tok not in self.token_to_id:
                    raise ValueError(f"Vocab missing required special token {tok!r}")
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    # ------------------------------------------------------------------
    # Vocab management
    # ------------------------------------------------------------------
    @classmethod
    def from_corpus(cls, smiles: Iterable[str], min_count: int = 1) -> "SmilesTokenizer":
        tok = cls()
        tok.build_vocab(smiles, min_count=min_count)
        return tok

    def build_vocab(self, smiles: Iterable[str], min_count: int = 1) -> None:
        counts: Counter[str] = Counter()
        for s in smiles:
            counts.update(smiles_tokenize(s))
        # Reset to specials, then add observed tokens in deterministic order.
        self.token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, c in sorted(counts.items()):
            if c < min_count:
                continue
            if tok in self.token_to_id:
                continue
            self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    @property
    def vocab_size(self) -> int:
        return len(self.token_to_id)

    # ------------------------------------------------------------------
    # Special-token id helpers
    # ------------------------------------------------------------------
    @property
    def pad_id(self) -> int:
        return self.token_to_id["[PAD]"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["[BOS]"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["[EOS]"]

    @property
    def mask_id(self) -> int:
        return self.token_to_id["[MASK]"]

    @property
    def unk_id(self) -> int:
        return self.token_to_id["[UNK]"]

    # ------------------------------------------------------------------
    # Encoding / decoding
    # ------------------------------------------------------------------
    def encode(
        self,
        s: str,
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
    ) -> list[int]:
        toks = smiles_tokenize(s)
        ids = [self.token_to_id.get(t, self.unk_id) for t in toks]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        if max_len is not None and len(ids) > max_len:
            # Truncate but always preserve EOS as the final token so the
            # model sees a proper end-of-sequence marker.
            if add_eos:
                ids = ids[: max_len - 1] + [self.eos_id]
            else:
                ids = ids[:max_len]
        return ids

    def decode(self, ids: list[int] | "torch.Tensor") -> str:
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        skip = {self.pad_id, self.bos_id, self.eos_id}
        out: list[str] = []
        for i in ids:
            if i in skip:
                continue
            out.append(self.id_to_token.get(int(i), "[UNK]"))
        return "".join(out)

    def encode_batch(
        self,
        smiles_list: list[str],
        *,
        add_bos: bool = True,
        add_eos: bool = True,
        max_len: int | None = None,
        pad_to: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns ``(input_ids, attention_mask)`` LongTensors of shape (B, L).

        ``L`` is the longest sequence in the batch (or ``pad_to`` if larger).
        """
        encoded = [
            self.encode(s, add_bos=add_bos, add_eos=add_eos, max_len=max_len)
            for s in smiles_list
        ]
        L = max((len(e) for e in encoded), default=1)
        if pad_to is not None:
            L = max(L, pad_to)
        input_ids = torch.full((len(encoded), L), self.pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(encoded), L), dtype=torch.long)
        for i, e in enumerate(encoded):
            input_ids[i, : len(e)] = torch.tensor(e, dtype=torch.long)
            attention_mask[i, : len(e)] = 1
        return input_ids, attention_mask

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.token_to_id, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "SmilesTokenizer":
        return cls(vocab=json.loads(Path(path).read_text(encoding="utf-8")))


class SelfiesTokenizer(SmilesTokenizer):
    """SELFIES-based tokenizer; requires the optional ``selfies`` dependency.

    SELFIES strings are inherently tokenized -- each ``[Symbol]`` is one
    token. We delegate the tokenization to the ``selfies`` package and
    inherit vocab management from :class:`SmilesTokenizer`.
    """

    def __init__(self, vocab: dict[str, int] | None = None) -> None:
        try:
            import selfies as _sf  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "SelfiesTokenizer requires the 'selfies' package; install with "
                "`pip install selfies`."
            ) from exc
        super().__init__(vocab=vocab)

    @staticmethod
    def _tokenize(s: str) -> list[str]:
        import selfies as sf

        return list(sf.split_selfies(s))

    def build_vocab(self, smiles: Iterable[str], min_count: int = 1) -> None:
        counts: Counter[str] = Counter()
        for s in smiles:
            counts.update(self._tokenize(s))
        self.token_to_id = {tok: i for i, tok in enumerate(SPECIAL_TOKENS)}
        for tok, c in sorted(counts.items()):
            if c < min_count or tok in self.token_to_id:
                continue
            self.token_to_id[tok] = len(self.token_to_id)
        self.id_to_token = {i: t for t, i in self.token_to_id.items()}

    def encode(self, s: str, **kw) -> list[int]:
        toks = self._tokenize(s)
        ids = [self.token_to_id.get(t, self.unk_id) for t in toks]
        if kw.get("add_bos", True):
            ids = [self.bos_id] + ids
        if kw.get("add_eos", True):
            ids = ids + [self.eos_id]
        max_len = kw.get("max_len")
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        return ids
