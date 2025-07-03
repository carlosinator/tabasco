from rdkit import Chem
from typing import List
from torch import Tensor


def largest_component(molecules: List[Chem.Mol]) -> List[Chem.Mol]:
    """Return the largest connected component for each molecule.

    Args:
        molecules: Iterable of RDKit Mol objects with 3-D coordinates.

    Returns:
        list[Chem.Mol]: Each entry is the fragment with the most atoms for
        the corresponding input molecule.

    From: https://github.com/jostorge/diffusion-hopping/blob/main/diffusion_hopping/analysis/util.py
    """
    return [
        max(
            Chem.GetMolFrags(mol, asMols=True),
            key=lambda x: x.GetNumAtoms(),
            default=mol,
        )
        for mol in molecules
    ]


def attempt_sanitize(mol: Chem.Mol):
    """Run `Chem.SanitizeMol` and return `None` on failure.

    Args:
        mol: RDKit Mol to sanitize.

    Returns:
        Chem.Mol | None: Sanitised molecule or `None` if RDKit raises.

    Credits: Charles Harris
    """
    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception as e:
        print(f"Sanitization failed: {e}")
        return None


def reorder_molecule_by_smiles(mol: Chem.Mol) -> Chem.Mol | None:
    """Renumber atoms to follow canonical SMILES indexing.

    Args:
        mol: RDKit Mol. If `None`, the function returns `None`.

    Returns:
        Chem.Mol | None: Renumbered copy of the molecule.

    Raises:
        ValueError: If canonicalisation or substructure matching fails.
    """

    if mol is None:
        return None

    mol_copy = Chem.Mol(mol)
    canonical_mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol_copy))

    if canonical_mol is None:
        raise ValueError("Failed to canonicalize molecule")

    match = mol_copy.GetSubstructMatch(canonical_mol)
    if not match:
        raise ValueError("Failed to find substructure match")

    return Chem.RenumberAtoms(mol_copy, match)


def write_xyz_file(coords: Tensor, atom_types: List[str], filename: str) -> None:
    """Write an XYZ file.

    Args:
        coords: Array-like of shape `(N, 3)` with Cartesian coordinates.
        atom_types: Sequence of length `N` with element symbols.
        filename: Path to the output `.xyz` file.
    """
    out = f"{len(coords)}\n\n"
    assert len(coords) == len(atom_types)
    for i in range(len(coords)):
        out += f"{atom_types[i]} {coords[i, 0]:.3f} {coords[i, 1]:.3f} {coords[i, 2]:.3f}\n"
    with open(filename, "w") as f:
        f.write(out)
