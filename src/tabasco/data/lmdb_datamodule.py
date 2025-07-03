from typing import Optional

from lightning import LightningDataModule
from tabasco.data.utils import TensorDictCollator
from tabasco.data.components.lmdb_unconditional import UnconditionalLMDBDataset
from torch.utils.data import DataLoader
from tabasco.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class LmdbDataModule(LightningDataModule):
    """PyTorch Lightning `DataModule` for unconditional ligand generation."""

    def __init__(
        self,
        data_dir: str,
        lmdb_dir: str,
        add_random_rotation: bool = False,
        add_random_permutation: bool = False,
        reorder_to_smiles_order: bool = False,
        remove_hydrogens: bool = True,
        batch_size: int = 256,
        num_workers: int = 0,
        val_data_dir: Optional[str] = None,
        test_data_dir: Optional[str] = None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.val_data_dir = val_data_dir
        self.test_data_dir = test_data_dir
        self.lmdb_dir = lmdb_dir
        self.dataset_kwargs = {
            "add_random_rotation": add_random_rotation,
            "add_random_permutation": add_random_permutation,
            "reorder_to_smiles_order": reorder_to_smiles_order,
            "remove_hydrogens": remove_hydrogens,
        }
        """Args:
            data_dir: Path to the training set .pt file produced by preprocessing.
            lmdb_dir: Directory where LMDB files and stats are stored.
            add_random_rotation: Apply random rotations inside each dataset item.
            add_random_permutation: Randomly permute heavy-atom order in each item.
            reorder_to_smiles_order: Re-index atoms to canonical SMILES before tensorization.
            remove_hydrogens: Strip explicit hydrogens before conversion to tensors.
            batch_size: Number of molecules per batch.
            num_workers: DataLoader worker count.
            val_data_dir: Optional path to a separate validation set; if None, a train/val split is created.
            test_data_dir: Optional path to a held-out test set.
        """

    def prepare_data(self):
        """Create LMDB files if they are missing (handled lazily by dataset)."""
        return

    def setup(self, stage: Optional[str] = None):
        """Instantiate train/val/test datasets.

        If val_data_dir is None, the training file is randomly split into
        train and validation indices. Otherwise the provided paths are used.
        """
        if self.val_data_dir is None:
            train_indices, val_indices = self._compute_train_val_split()  # nosec B614

            log.info("Initializing train dataset...")
            self.train_dataset = UnconditionalLMDBDataset(
                data_dir=self.data_dir,
                split_indices=train_indices,
                **self.dataset_kwargs,
            )
            log.info(
                f"Train dataset initialized with {len(self.train_dataset)} samples"
            )

            log.info("Initializing val dataset...")
            self.val_dataset = UnconditionalLMDBDataset(
                data_dir=self.data_dir,
                split_indices=val_indices,
                **self.dataset_kwargs,
            )
            log.info(f"Val dataset initialized with {len(self.val_dataset)} samples")

        else:
            self.train_dataset = UnconditionalLMDBDataset(
                data_dir=self.data_dir,
                split="train",
                lmdb_dir=self.lmdb_dir,
                **self.dataset_kwargs,
            )
            self.val_dataset = UnconditionalLMDBDataset(
                data_dir=self.val_data_dir,
                split="val",
                lmdb_dir=self.lmdb_dir,
                **self.dataset_kwargs,
            )
            if self.test_data_dir is not None:
                self.test_dataset = UnconditionalLMDBDataset(
                    data_dir=self.test_data_dir,
                    split="test",
                    lmdb_dir=self.lmdb_dir,
                    **self.dataset_kwargs,
                )
            else:
                self.test_dataset = UnconditionalLMDBDataset(
                    data_dir=self.val_data_dir,
                    split="val",
                    lmdb_dir=self.lmdb_dir,
                    **self.dataset_kwargs,
                )

    def get_dataset_stats(self):
        """Return statistics dictionary computed by the training dataset."""
        return self.train_dataset.get_stats()

    def train_dataloader(self):
        """Return the training `DataLoader`."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
            shuffle=True,
        )

    def val_dataloader(self):
        """Return the validation `DataLoader`."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
            shuffle=False,
        )

    def test_dataloader(self):
        """Return the test `DataLoader` (falls back to validation set when absent)."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=TensorDictCollator(),
            shuffle=False,
        )
