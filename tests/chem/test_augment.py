from radiant_chem.augment import canonicalize_smiles, randomize_smiles


def test_randomize_returns_string():
    out = randomize_smiles("CCO")
    assert isinstance(out, str)
    assert len(out) > 0


def test_canonicalize_returns_string():
    out = canonicalize_smiles("CCO")
    assert isinstance(out, str)


def test_randomize_handles_invalid_input():
    # Garbage SMILES; rdkit returns None internally and we fall back to identity.
    out = randomize_smiles("not_a_molecule_!!!")
    assert isinstance(out, str)
