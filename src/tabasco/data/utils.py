from typing import Dict, List

from torch import Tensor
from tensordict import TensorDict


def batch_to_list(batch: Dict[str, Tensor]) -> List[Dict[str, Tensor]]:
    """Split a stacked batch dict into a list of per-sample dicts."""
    return [
        {k: v[i] for k, v in batch.items()} for i in range(batch["coords"].shape[0])
    ]


class TensorDictCollator:
    """Stack a list of TensorDict objects along batch dimension."""

    def __call__(self, batch):
        """Stack incoming list into a single TensorDict.

        Args:
            batch: Sequence of TensorDict objects returned by the dataset.

        Returns:
            Stacked TensorDict.
        """
        if isinstance(batch[0], TensorDict):
            return TensorDict.stack(batch, dim=0)
        raise TypeError(f"Expected TensorDict, got {type(batch[0])}")
