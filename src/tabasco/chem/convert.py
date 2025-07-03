import contextlib as clib
from copy import deepcopy
from typing import List, Optional, Tuple

import biotite.structure as struc  # codespell: ignore
import numpy as np
import rdkit.Chem.rdDetermineBonds as rdDetermineBonds
import torch
from biotite.structure import AtomArray
from rdkit import Chem
from tensordict import TensorDict
from torch import Tensor
import tempfile

from tabasco.chem.utils import attempt_sanitize, write_xyz_file
from tabasco.chem.constants import ATOM_COLOR_MAP, ATOM_NAMES
from tabasco.data.utils import batch_to_list
from tabasco.utils import RankedLogger
from rdkit import RDLogger

log = RankedLogger(__name__, rank_zero_only=True)


class MoleculeConverter:
    """Bidirectional converter between RDKit molecules and TensorDicts.

    Coordinates are optionally centred and divided by `dataset_normalizer`
    to improve numerical stability for learning tasks.
    """

    def __init__(
        self,
        atom_names=ATOM_NAMES,
        atom_color_map=ATOM_COLOR_MAP,
        dataset_normalizer=2.0,
    ):
        """Args:
        atom_names: Allowed element symbols; `'*'` is treated as a dummy.
        atom_color_map: Parallel list of RGB colour triples.
        dataset_normalizer: Value used to scale coordinates; see
            `to_tensor` / `from_tensor`.
        """
        self._atom_types = atom_names
        self._mol_colors = list(atom_color_map.values())
        self._dummy_atom_idx = self._atom_types.index("*")

        # TODO: refactor: make this a param and rename to smth better
        self.dataset_normalizer = dataset_normalizer

    def _get_atomics(self, mol: Chem.Mol) -> Tensor:
        """Return one-hot atom-type matrix of shape `(N, n_elements)`."""
        atom_names = [atom.GetSymbol() for atom in mol.GetAtoms()]
        atomics = torch.zeros(
            len(atom_names), len(self._atom_types), dtype=torch.float32
        )
        for i, atom in enumerate(atom_names):
            if atom not in self._atom_types:
                atomics[i, self._dummy_atom_idx] = 1.0
            else:
                atomics[i, self._atom_types.index(atom)] = 1.0
        return atomics

    def _get_atom_types(
        self, atomics: Tensor, padding_mask: Optional[Tensor] = None
    ) -> List[str]:
        """Return list of element symbols, ignoring rows flagged in
        `padding_mask` if provided."""
        if padding_mask is not None:
            atomics = atomics[~padding_mask]

        return [self._atom_types[i] for i in torch.argmax(atomics, dim=1)]

    def _pad_to_size(
        self, coords: Tensor, atomics: Tensor, max_size: int
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Pad `coords` and `atomics` to `max_size` in the first dimension.

        Returns:
            Tuple of (padded_coords, padded_atomics, padding_mask) where
            `padding_mask` is shape `(max_size,)` and `False` for real atoms.
        """

        assert coords.shape[0] <= max_size, (
            f"coords must be less than or equal to max_size, got {coords.shape[0]} and max_size: {max_size}"
        )

        assert coords.shape[0] == atomics.shape[0], (
            f"coords and atomics must have the same number of atoms, got coords: {coords.shape[0]} and atomics: {atomics.shape[0]}"
        )

        padding_mask = torch.ones(max_size, dtype=torch.bool)
        padding_mask[: coords.shape[0]] = False

        padded_coords = torch.cat(
            [coords, torch.zeros(max_size - coords.shape[0], *coords.shape[1:])], dim=0
        )

        dummy_atoms = torch.zeros(
            (max_size - atomics.shape[0], len(self._atom_types)), dtype=torch.float32
        )
        dummy_atoms[:, self._dummy_atom_idx] = 1

        padded_atomics = torch.cat(
            [atomics, dummy_atoms],
            dim=0,
        )

        return padded_coords, padded_atomics, padding_mask

    def to_tensor(
        self,
        mol: Chem.Mol,
        pad_to_size: Optional[int] = None,
        normalize_coords: bool = True,
        remove_hydrogens: bool = True,
    ) -> TensorDict:
        """Convert an RDKit mol to a TensorDict.

        Args:
            mol: Input molecule with 3-D conformer.
            pad_to_size: If given, output is padded to this atom count.
            normalize_coords: If True, centre of mass is removed and
                divided by `dataset_normalizer`.
            remove_hydrogens: If True, strip explicit H atoms.

        Returns:
            TensorDict with keys:
                • `coords`:         `(N, 3)` float32
                • `atomics`:        `(N, n_elements)` one-hot
                • `padding_mask`:   `(N,)` bool (optional)
        """

        if remove_hydrogens:
            mol = deepcopy(mol)
            mol = Chem.RemoveAllHs(mol)

        raw_coords = torch.tensor(
            mol.GetConformer().GetPositions(), dtype=torch.float32
        )

        if normalize_coords:
            raw_coords = (raw_coords - raw_coords.mean(dim=0)) / self.dataset_normalizer

        atomics = self._get_atomics(mol)

        if pad_to_size is None:
            return TensorDict(
                {
                    "coords": raw_coords,
                    "atomics": atomics,
                }
            )

        padded_coords, padded_atomics, padding_mask = self._pad_to_size(
            raw_coords, atomics, pad_to_size
        )

        return TensorDict(
            {
                "coords": padded_coords,
                "atomics": padded_atomics,
                "padding_mask": padding_mask,
            }
        )

    def _make_mol_simple_imputation(
        self, coords: Tensor, atom_types: List[str]
    ) -> Chem.Mol:
        """Quick bond inference via RDKit's `DetermineConnectivity`.

        Note: Bond orders are crude and stereochemistry is ignored.
        """

        mol = Chem.RWMol()

        # add atom types
        for i, (atom_type, _) in enumerate(zip(atom_types, coords)):
            atom = Chem.Atom(atom_type)
            mol.AddAtom(atom)

        # make conformer
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(mol.GetNumAtoms()):
            conf.SetAtomPosition(
                i, (coords[i, 0].item(), coords[i, 1].item(), coords[i, 2].item())
            )
        mol.AddConformer(conf)

        rdDetermineBonds.DetermineConnectivity(mol)

        return mol

    def _make_mol_openbabel(self, coords: Tensor, atom_types: List[str]) -> Chem.Mol:
        """Bond inference through OpenBabel round-trip XYZ -> SDF.

        Requires the `openbabel` Python bindings to be installed.
        """
        from openbabel import openbabel

        with tempfile.NamedTemporaryFile() as tmp:
            tmp_file = tmp.name

            # Write xyz file
            write_xyz_file(coords, atom_types, tmp_file)

            # Convert to sdf file with openbabel
            # openbabel will add bonds
            obConversion = openbabel.OBConversion()
            obConversion.SetInAndOutFormats("xyz", "sdf")
            ob_mol = openbabel.OBMol()
            obConversion.ReadFile(ob_mol, tmp_file)

            obConversion.WriteFile(ob_mol, tmp_file)

            # Read sdf file with RDKit
            tmp_mol = Chem.SDMolSupplier(tmp_file, sanitize=False)[0]

        if tmp_mol is None:
            return None

        # Build new molecule. This is a workaround to remove radicals.
        mol = Chem.RWMol()
        for atom in tmp_mol.GetAtoms():
            mol.AddAtom(Chem.Atom(atom.GetSymbol()))
        mol.AddConformer(tmp_mol.GetConformer(0))

        for bond in tmp_mol.GetBonds():
            mol.AddBond(
                bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond.GetBondType()
            )

        return mol

    def from_tensor(
        self,
        mol_tensor: TensorDict,
        rescale_coords: bool = True,
        sanitize: bool = True,
        use_openbabel: bool = True,
    ):
        """Inverse of `to_tensor`.

        Args:
            mol_tensor: Unbatched TensorDict.
            rescale_coords: Multiply coords back by `dataset_normalizer`.
            sanitize: Run `Chem.SanitizeMol`; may fail on exotic molecules.
            use_openbabel: Toggle OpenBabel bond inference.

        Returns:
            RDKit Mol or `None` if any step fails.
        """
        assert mol_tensor["coords"].shape[0] == mol_tensor["atomics"].shape[0]
        assert mol_tensor["coords"].shape[1] == 3, (
            "positions must be (N, 3), might have passed node features instead"
        )
        assert len(mol_tensor["coords"].shape) == 2, "function expects unbatched data"

        coords = mol_tensor["coords"]
        atomics = mol_tensor["atomics"]

        if "padding_mask" in mol_tensor:
            padding_mask = mol_tensor["padding_mask"]
            coords = coords[~padding_mask]
            atom_types = self._get_atom_types(atomics, padding_mask)
        else:
            atom_types = self._get_atom_types(atomics)

        if rescale_coords:
            coords = coords * self.dataset_normalizer

        assert len(atom_types) == coords.shape[0], (
            f"atom_types and coords must have the same number of atoms, got {len(atom_types)} and {coords.shape[0]}"
        )

        if use_openbabel:
            mol = self._make_mol_openbabel(coords, atom_types)
            if mol is None:
                return None
        else:
            mol = self._make_mol_simple_imputation(coords, atom_types)

        # remove radicals if any and set charge to 0 and set implicit valence
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                atom.SetNumRadicalElectrons(0)
            atom.SetFormalCharge(0)

        if sanitize:
            # useful to suppress output, comment out when debugging
            with (
                clib.redirect_stdout(open("/dev/null", "w")),
                clib.redirect_stderr(open("/dev/null", "w")),
            ):
                mol = attempt_sanitize(mol)

        return deepcopy(mol)

    def from_batch(self, batch: TensorDict, **kwargs) -> List[Chem.Mol]:
        """Vectorised wrapper around `from_tensor` for batched data.

        Unconvertible items are returned as `None` and logged as warnings.
        """
        mol_list = []
        gen_list = batch_to_list(batch)

        rdkit_logger = RDLogger.logger()
        rdkit_logger.setLevel(RDLogger.CRITICAL)

        # suppress rdkit warnings, comment out when debugging
        for gen in gen_list:
            try:
                mol = self.from_tensor(gen, **kwargs)
                mol_list.append(mol)
            except Exception as e:
                log.warning(f"Error building molecule: {e}")
                mol_list.append(None)
                continue

        rdkit_logger.setLevel(RDLogger.INFO)

        return mol_list

    def data_to_atom_array(
        self,
        mol_tensor: TensorDict,
        rescale_coords: bool = True,
        add_bonds=True,
        add_hydrogens=True,
        sanitize=True,
    ) -> AtomArray:
        """Shortcut: TensorDict -> RDKit Mol -> Biotite AtomArray."""

        assert len(mol_tensor["coords"].shape) == 2, "function expects unbatched data"

        mol = self.from_tensor(
            mol_tensor, rescale_coords, add_bonds, add_hydrogens, sanitize
        )
        return self.mol_to_atom_array(mol)

    def mol_to_atom_array(self, mol: Chem.Mol) -> AtomArray:
        """Convert an RDKit Mol to a Biotite `AtomArray` (with bonds)."""

        # Create atom array from RDKit mol
        atoms = []
        for atom in mol.GetAtoms():
            # Get 3D coordinates
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            coord = np.array([pos.x, pos.y, pos.z])

            # Create Biotite Atom with required annotations
            atoms.append(
                struc.Atom(
                    coord=coord,
                    chain_id="A",  # Default chain
                    res_id=1,  # Default residue
                    res_name="UNK",  # Unknown residue
                    hetero=True,  # Hetero atom
                    atom_name=atom.GetSymbol(),  # Use element symbol as name
                    element=atom.GetSymbol(),  # Element symbol
                )
            )

        # Create atom array from atoms
        atom_array = struc.array(atoms)

        # Add bonds if present
        if mol.GetNumBonds() > 0:
            bonds = []
            for bond in mol.GetBonds():
                # Get atom indices for each bond
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                # Add bond (BondType.ANY = 0)
                bonds.append([i, j, 0])

            # Create bond list and add to atom array
            bond_list = struc.BondList(atom_array.array_length(), np.array(bonds))
            atom_array.bonds = bond_list

        return atom_array

    def tensor_obj_to_points(self, tensor_obj: TensorDict) -> Tuple[Tensor, Tensor]:
        """Return `(coords, atom_type_idx)` with padding rows removed."""

        assert len(tensor_obj["coords"].shape) == 2, "function expects unbatched data"

        coords = tensor_obj["coords"]
        atomics = tensor_obj["atomics"]
        padding_mask = tensor_obj["padding_mask"]

        coords = coords[~padding_mask]
        atomics = atomics[~padding_mask]

        if atomics.shape[1] == 0:
            atom_idx = torch.zeros(coords.shape[0], dtype=torch.long)
        else:
            atom_idx = torch.argmax(atomics, dim=1)

        return coords, atom_idx
