from typing import Optional, Callable
import torch
from tensordict import TensorDict
from torch import Tensor, nn

from tabasco.flow.path import FlowPath
from tabasco.utils.metric_utils import split_losses_by_time


class InterDistancesLoss(nn.Module):
    """Mean-squared error between predicted and reference inter-atomic distance matrices."""

    def __init__(
        self,
        distance_threshold: Optional[float] = None,
        sqrd: bool = False,
        key: str = "coords",
        key_pad_mask: str = "padding_mask",
        time_factor: Optional[Callable] = None,
    ):
        """Initialize the loss module.

        Args:
            distance_threshold: If provided, only atom pairs with distance <= threshold
                contribute to the loss. Units must match the coordinate system.
            sqrd: When `True` the raw *squared* distances are used instead of their square-root.
                Set this to `True` if you have pre-squared your training targets.
            key: Key that stores coordinates inside `TensorDict` objects.
            key_pad_mask: Key that stores the boolean padding mask inside `TensorDict` objects.
            time_factor: Optional callable `f(t)` that rescales the per-pair loss as a
                function of the interpolation time `t`.
        """
        super().__init__()
        self.key = key
        self.key_pad_mask = key_pad_mask
        self.distance_threshold = distance_threshold
        self.sqrd = sqrd
        self.mse_loss = nn.MSELoss(reduction="none")
        self.time_factor = time_factor

    def inter_distances(self, coords1, coords2, eps: float = 1e-6) -> Tensor:
        """Compute pairwise distances between two coordinate sets.

        Args:
            coords1: Coordinate tensor of shape `(N, 3)`.
            coords2: Coordinate tensor of shape `(M, 3)`.
            eps: Numerical stability term added before `sqrt` when `sqrd` is `False`.

        Returns:
            Tensor of shape `(N, M)` containing pairwise distances. Values are squared
            distances when the instance was created with `sqrd=True`.
        """
        if self.sqrd:
            return torch.cdist(coords1, coords2, p=2) ** 2
        else:
            squared_dist = torch.cdist(coords1, coords2, p=2) ** 2
            return torch.sqrt(squared_dist + eps)

    def forward(
        self, path: FlowPath, pred: TensorDict, compute_stats: bool = True
    ) -> Tensor:
        """Compute the inter-distance MSE loss.

        Args:
            path: `FlowPath` containing ground-truth endpoint coordinates and the
                interpolation time `t`.
            pred: `TensorDict` with predicted coordinates under the same `key` as
                specified during initialization.
            compute_stats: If `True` additionally returns distance-loss statistics binned
                by time for logging purposes.

        Returns:
            - loss:         Scalar tensor with the mean loss.
            - stats_dict:   Dictionary with binned loss statistics. Empty when
                `compute_stats` is `False`.
        """
        real_mask = 1 - path.x_1[self.key_pad_mask].float()
        real_mask = real_mask.unsqueeze(-1)

        pred_coords = pred[self.key]
        true_coords = path.x_1[self.key]

        pred_dists = self.inter_distances(pred_coords, pred_coords)
        true_dists = self.inter_distances(true_coords, true_coords)

        mask_2d = torch.matmul(real_mask, real_mask.transpose(-1, -2))

        # Add distance threshold mask (0 for pairs where distance > threshold)
        if self.distance_threshold is not None:
            distance_mask = (true_dists <= self.distance_threshold).float()
            combined_mask = mask_2d * distance_mask
        else:
            combined_mask = mask_2d

        dists_loss = self.mse_loss(pred_dists, true_dists)
        dists_loss = dists_loss * combined_mask

        if self.time_factor:
            dists_loss = dists_loss * self.time_factor(path.t)

        if compute_stats:
            binned_losses = split_losses_by_time(path.t, dists_loss, 5)
            stats_dict = {
                **{f"dists_loss_bin_{i}": loss for i, loss in enumerate(binned_losses)},
            }
        else:
            stats_dict = {}

        dists_loss = dists_loss.mean()
        return dists_loss, stats_dict
