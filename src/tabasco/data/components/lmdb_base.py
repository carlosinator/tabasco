import os
import pickle
from typing import Callable, List, Optional

import lmdb
from torch.utils.data import Dataset
from tqdm import tqdm
from abc import abstractmethod, ABC


class BaseLMDBDataset(Dataset, ABC):
    def __init__(
        self,
        split: str,
        lmdb_dir: str,
        transform: Optional[Callable] = None,
        single_sample: bool = False,
        limit_samples: Optional[int] = None,
        pre_filter: Optional[List[Callable]] = None,
    ):
        """Credits: Charlie Harris.
        Dataset class for LMDB-based datasets.

        Args:
            split (str): The split of the dataset to use (e.g., 'train', 'val', 'test').
            transform (Callable, optional): A function/transform that takes in a sample and returns a transformed version.
                Defaults to None.
            single_sample (bool, optional): Whether to return a single sample. Defaults to False.
            limit_samples (int, optional): The number of samples to limit the dataset to. Defaults to None.
            pre_filter (Callable, optional): A function that takes in a sample and returns True if the sample should be
                included in the dataset, and False otherwise. Defaults to None.
        """
        super().__init__()
        self.split = split
        self.single_sample = single_sample
        self.limit_samples = limit_samples
        self.pre_filter = pre_filter
        self.lmdb_dir = lmdb_dir
        self.lmdb_path = os.path.join(lmdb_dir, split + ".lmdb")
        self.index = None

        self.transform = transform
        self.db = None
        self.keys = None

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, "A connection has already been opened."
        self.db = lmdb.open(
            self.lmdb_path,
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

        if self.limit_samples is not None:
            self.keys = self.keys[: self.limit_samples]

        if self.pre_filter is not None:
            self._pre_filter_dataset()

    def _pre_filter_dataset(self):
        """
        Loop through the dataset and apply the pre-filter function
        Remove keys that do not pass the filter
        """
        removed = 0
        for key in tqdm(self.keys, desc=f"Pre-filtering dataset {self.split}"):
            with self.db.begin() as txn:
                data = pickle.loads(txn.get(key))
                mol = data["mol"]
                if not all(fn(mol) for fn in self.pre_filter):
                    self.keys.remove(key)
                    removed += 1
        print(f"Removed {removed} samples from the dataset.")

    def _close_db(self):
        """close database"""
        self.db.close()
        self.db = None
        self.keys = None

    @abstractmethod
    def _process(self):
        """abstract method to process the dataset i.e. add key-value pairs to the database"""
        pass

    def __len__(self):
        """Return the number of entries in the dataset split."""
        if self.db is None:
            self._connect_db()
        return len(self.keys)

    def __repr__(self):
        """Debug-friendly string with split name and dataset length."""
        return f"{self.__class__.__name__}(split={self.split}, len={len(self)})"

    def __getitem__(self, idx):
        """Fetch an item by index, optionally applying `transform`."""
        if self.db is None:
            self._connect_db()
        if self.single_sample:
            idx = 0
        key = self.keys[idx]
        data = pickle.loads(self.db.begin().get(key))
        data["id"] = idx

        if self.transform is not None:
            data = self.transform(data)
        return data

    def update_data(self, update_fn: Callable):
        """
        Update data in the LMDB database using a provided update function.

        Args:
            update_fn (Callable): A function that takes a single data item and returns an updated version of it.
        """
        if self.db is None:
            self._connect_db()

        # Ensure the database is not opened in readonly mode
        self.db.close()
        self.db = lmdb.open(
            self.processed_path,  # Assuming this is the path to your LMDB database
            map_size=10 * (1024 * 1024 * 1024),  # 10GB
            create=False,
            subdir=False,
            readonly=False,  # Important: open in write mode
            lock=False,
            readahead=False,
            meminit=False,
        )

        with self.db.begin(write=True) as txn:
            for key in tqdm(self.keys, desc=f"Updating dataset {self.split}"):
                data = pickle.loads(txn.get(key))
                updated_data = update_fn(data)
                txn.put(key, pickle.dumps(updated_data))

        # Close the database after updating
        self._close_db()
