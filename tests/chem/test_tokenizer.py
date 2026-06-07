import torch

from radiant_chem import SmilesTokenizer
from radiant_chem.tokenizer import smiles_tokenize


CORPUS = [
    "CCO",
    "c1ccccc1",
    "CC(=O)O",
    "Cc1ccncc1",
    "Brc1ccc(N)cc1",
    "[NH4+]",
    "[13C]C",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1",
    "C/C=C/C",  # cis/trans
    "C[C@H](N)C(=O)O",
    "CCO%11",  # 2-digit ring closure
]


def test_two_letter_atoms_recognized():
    assert smiles_tokenize("BrCCBr") == ["Br", "C", "C", "Br"]
    assert smiles_tokenize("ClCCCl") == ["Cl", "C", "C", "Cl"]


def test_brackets_atomic():
    assert smiles_tokenize("[NH4+]") == ["[NH4+]"]
    assert smiles_tokenize("[13C]C") == ["[13C]", "C"]
    assert smiles_tokenize("C[C@@H](N)C") == ["C", "[C@@H]", "(", "N", ")", "C"]


def test_two_digit_ring_closure():
    assert smiles_tokenize("CCO%11") == ["C", "C", "O", "%11"]


def test_round_trip_after_vocab_build():
    tok = SmilesTokenizer.from_corpus(CORPUS)
    for s in CORPUS:
        ids = tok.encode(s, add_bos=False, add_eos=False)
        assert tok.decode(ids) == s


def test_special_tokens_present():
    tok = SmilesTokenizer.from_corpus(CORPUS)
    for name in ("[PAD]", "[BOS]", "[EOS]", "[MASK]", "[UNK]"):
        assert name in tok.token_to_id


def test_special_token_ids_consistent():
    tok = SmilesTokenizer.from_corpus(CORPUS)
    assert tok.pad_id == 0
    assert tok.token_to_id["[BOS]"] == tok.bos_id


def test_unknown_token_maps_to_unk():
    tok = SmilesTokenizer.from_corpus(["CC"])  # vocab knows only C and specials
    ids = tok.encode("CN", add_bos=False, add_eos=False)
    assert tok.unk_id in ids


def test_encode_batch_padding_and_mask():
    tok = SmilesTokenizer.from_corpus(CORPUS)
    ids, attn = tok.encode_batch(["CC", "CCO", "Cc1ccncc1"])
    assert ids.shape[0] == 3
    L = ids.shape[1]
    assert attn.shape == (3, L)
    # Each row's mask sums to its non-pad length.
    assert int(attn[0].sum()) < int(attn[2].sum())  # shorter SMILES has fewer real tokens


def test_save_load_round_trip(tmp_path):
    tok = SmilesTokenizer.from_corpus(CORPUS)
    p = tmp_path / "vocab.json"
    tok.save(p)
    tok2 = SmilesTokenizer.load(p)
    assert tok.token_to_id == tok2.token_to_id


def test_max_len_truncation():
    tok = SmilesTokenizer.from_corpus(CORPUS)
    ids = tok.encode("CCCCCCCCCC", max_len=4)
    assert len(ids) == 4
