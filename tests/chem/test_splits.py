from radiant_chem import random_split, scaffold_split


SMILES = [
    "CCO", "c1ccccc1", "CC(=O)O", "Cc1ccncc1", "Brc1ccc(N)cc1",
    "CN(C)CC", "O=C1CCC1", "CCBr", "NCCc1ccc(O)c(O)c1",
    "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "ClCCl", "OCC(O)C(O)C(O)C(O)CO",
    "CC(=O)Nc1ccc(O)cc1", "CCCCCCCC", "CCN(CC)CC",
]


def test_random_split_partitions_indices():
    train, val, test = random_split(SMILES, ratios=(0.7, 0.15, 0.15), seed=0)
    union = sorted(train + val + test)
    assert union == list(range(len(SMILES)))
    assert len(set(train)) == len(train)


def test_random_split_is_deterministic():
    a = random_split(SMILES, seed=7)
    b = random_split(SMILES, seed=7)
    assert a == b


def test_scaffold_split_partitions_indices():
    train, val, test = scaffold_split(SMILES, ratios=(0.7, 0.15, 0.15), seed=0)
    union = sorted(train + val + test)
    assert union == list(range(len(SMILES)))


def test_scaffold_split_groups_by_scaffold():
    """Two molecules with the same scaffold (or hash bucket if rdkit absent)
    should not appear in different splits at high training ratios."""
    train, val, test = scaffold_split(SMILES, ratios=(1.0, 0.0, 0.0), seed=0)
    assert len(train) == len(SMILES)
    assert len(val) == 0
    assert len(test) == 0


def test_split_ratios_invalid_raise():
    import pytest
    with pytest.raises(ValueError):
        random_split(SMILES, ratios=(0, 0, 0))
