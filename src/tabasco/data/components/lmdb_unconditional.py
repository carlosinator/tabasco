import os
import yaml
import pickle
from collections import Counter
import lmdb
from tqdm import tqdm
import torch
from tabasco.data.components.lmdb_base import BaseLMDBDataset
from tensordict import TensorDict
from tabasco.chem.convert import MoleculeConverter
from tabasco.chem.utils import reorder_molecule_by_smiles
from rdkit import Chem
from tabasco.data.transforms import random_rotation, permute_atoms

from tabasco.utils import RankedLogger

logger = RankedLogger(__name__)


class UnconditionalLMDBDataset(BaseLMDBDataset):
    """Adapted from Charlie Harris' code, originally based on a PyG dataset of
    protein-ligand complexes for experimental structures found in CrossDocked2020 dataset.
    Unconditional ligand dataset backed by an LMDB file."""

    def __init__(
        self,
        data_dir: str,
        split: str,
        add_random_rotation: bool = True,
        add_random_permutation: bool = True,
        reorder_to_smiles_order: bool = False,
        remove_hydrogens: bool = True,
        single_sample: bool = False,
        limit_samples: int = None,
        lmdb_dir: str = None,
    ):
        """Initialize the dataset.

        Args:
            data_dir: Path to the serialized list of `(Optional[Protein], rdkit.Chem.Mol)` tuples.
            split: Name of the dataset split.
            add_random_rotation: Randomly rotate ligand coordinates on access.
            add_random_permutation: Randomly permute atom order on access.
            reorder_to_smiles_order: Sort atoms to match canonical SMILES before tensor conversion.
            remove_hydrogens: Strip explicit hydrogens prior to processing.
            single_sample: When `True`, always return the first entry (useful for debugging).
            limit_samples: Optional cap on number of molecules processed.
            lmdb_dir: Directory in which the LMDB and stats YAML will be stored.
        """
        super().__init__(
            split=split,
            single_sample=single_sample,
            limit_samples=limit_samples,
            lmdb_dir=lmdb_dir,
        )
        self.split = split
        self.data_dir = data_dir
        self.mol_converter = MoleculeConverter()

        self.add_random_rotation = add_random_rotation
        self.add_random_permutation = add_random_permutation
        self.reorder_to_smiles_order = reorder_to_smiles_order
        self.remove_hydrogens = remove_hydrogens

        self.db = None
        self.keys = None

        self.mol_num_atoms_list = []
        self.all_smiles = []

        if not (os.path.exists(self.lmdb_path)):
            self._process()
        else:
            self._load_stats()

    def _update_stats(self, tensor_repr: TensorDict):
        mol_num_atoms = tensor_repr["coords"].shape[0]
        self.mol_num_atoms_list.append(mol_num_atoms)

        molecule = self.mol_converter.from_tensor(tensor_repr)

        try:
            self.all_smiles.append(Chem.MolToSmiles(molecule))
        except Exception as e:
            logger.warning(f"mol processing error: {str(e)[:100]}")

    def compute_stats(self):
        """Populate summary statistics and write to disk."""
        num_atoms_histogram = Counter(self.mol_num_atoms_list)

        self.max_mol_num_atoms = max(num_atoms_histogram.keys())

        example_datapoint = self.__getitem__(0)

        stats_dict = {
            "num_atoms_histogram": dict(num_atoms_histogram),
            "max_num_atoms": self.max_mol_num_atoms,
            "spatial_dim": example_datapoint["coords"].shape[1],
            "atom_dim": example_datapoint["atomics"].shape[1],
            "all_smiles": self.all_smiles,
        }

        self.stats_dict = stats_dict
        yaml.dump(
            stats_dict,
            open(os.path.join(self.lmdb_dir, f"{self.split}_stats.yaml"), "w"),
        )

    def _load_stats(self):
        """Load previously computed statistics from YAML file."""
        with open(os.path.join(self.lmdb_dir, f"{self.split}_stats.yaml"), "r") as f:
            self.stats_dict = yaml.safe_load(f)

        self.max_mol_num_atoms = self.stats_dict["max_num_atoms"]
        self.all_smiles = self.stats_dict["all_smiles"]
        return

    def get_stats(self):
        """Get the dataset statistics."""
        logger.info(f"len all smiles: {len(self.all_smiles)}")
        logger.info(f"example smiles: {self.all_smiles[:5]}")
        return self.stats_dict

    def _process(self):
        """Create the LMDB and stats from `data_dir` if they are missing.

        Note:
            This operation may be slow because it converts every `rdkit.Chem.Mol`
            object to the internal tensor representation.
            Furthermore, this function allows for failure when processing molecules,
            simply not adding them to the dataset.
        """
        db = lmdb.open(
            self.lmdb_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=True,
            subdir=False,
            readonly=False,  # Writable
        )

        mol_list = torch.load(self.data_dir, weights_only=False)  # nosec B614

        with db.begin(write=True, buffers=True) as txn:
            for i, (_, dp) in enumerate(tqdm(mol_list, total=len(mol_list))):
                if self.remove_hydrogens:
                    try:
                        dp = Chem.RemoveAllHs(dp)
                    except Exception as e:
                        logger.warning(f"Error removing hydrogens: {e}, at index {i}")
                        continue

                if self.reorder_to_smiles_order:
                    try:
                        dp = reorder_molecule_by_smiles(dp)
                    except Exception as e:
                        logger.warning(
                            f"Error reordering molecule to smiles order: {e}, at index {i}"
                        )
                        continue
                try:
                    tensor_repr = self.mol_converter.to_tensor(dp)
                    self._update_stats(tensor_repr)
                except Exception as e:
                    logger.warning(
                        f"Error converting molecule to tensor: {e}, at index {i}"
                    )
                    continue

                data_dict = {
                    "molecule": dp,
                }

                # only add to LMDB if the molecule was successfully processed in all steps
                txn.put(key=str(i).encode(), value=pickle.dumps(data_dict))

        db.close()

        self.compute_stats()

    def get_data_dict(self, index: int) -> TensorDict:
        """Return the raw LMDB entry (no tensor conversion)."""
        return super().__getitem__(index)

    def __getitem__(self, index: int) -> TensorDict:
        """Return a tensor representation of the ligand ready for model input.
        Note: Overrides the default __getitem__ for custom conversion and transforms methods."""
        data_dict = super().__getitem__(index)

        data_tensor = self.mol_converter.to_tensor(
            mol=data_dict["molecule"],
            pad_to_size=self.max_mol_num_atoms,
        )

        if "id" in data_dict:
            data_tensor["index"] = data_dict["id"]

        if self.add_random_rotation:
            data_tensor = random_rotation(data_tensor)

        if self.add_random_permutation:
            data_tensor = permute_atoms(data_tensor)

        return data_tensor
