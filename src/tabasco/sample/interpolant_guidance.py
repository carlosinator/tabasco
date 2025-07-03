from typing import List
from abc import ABC, abstractmethod
from tensordict import TensorDict

import torch

from rdkit import Chem
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from lightning.pytorch import LightningModule
from tabasco.utils.tensor_ops import apply_mask

from tabasco.chem.convert import MoleculeConverter


class InterpolantGuidance(ABC):
    """Interface for guidance modules that modify intermediate samples."""

    @abstractmethod
    def __call__(
        self, lightning_module: LightningModule, x_t: TensorDict, t: int
    ) -> TensorDict:
        """Modify the partial sample `x_t` produced at timestep `t`."""
        pass


class UFFBoundGuidance(InterpolantGuidance):
    """Move atom coordinates towards locations that are within UFF distance bounds of their current conformation."""

    def __init__(
        self,
        mol_converter: MoleculeConverter,
        lr: float = 0.01,
        n_steps: int = 3,
        regress_to_center: bool = False,
    ):
        """Args:
        mol_converter: Helper that converts between tensors and `rdkit.Chem.Mol`.
        lr: Step size applied to the sign of the gradient.
        n_steps: Number of gradient steps taken per guidance call.
        regress_to_center: If True, pull distances to interval centers instead of edges.
        """
        self.mol_converter = mol_converter
        self.lr = lr
        self.n_steps = n_steps
        self.regress_to_center = regress_to_center

    def _get_uff_bounds(self, mol: Chem.Mol):
        """Return the 2D matrix of UFF lower/upper bounds scaled to tensor space."""

        if mol is None:
            return None

        bound_matrix_params = {
            "set15bounds": True,
            "scaleVDW": True,
            "doTriangleSmoothing": True,
            "useMacrocycle14config": False,
        }

        bounds = GetMoleculeBoundsMatrix(mol, **bound_matrix_params)

        return (
            torch.from_numpy(bounds).to(dtype=torch.float32)
            / self.mol_converter.dataset_normalizer
        )

    def __call__(
        self, lightning_module: LightningModule, x_t: TensorDict, t: int
    ) -> TensorDict:
        """Return a steered version of `x_t` after `n_steps` bound-guided updates."""
        x_t = x_t.detach().clone()
        x_t["coords"].requires_grad = True

        for _ in range(self.n_steps):
            endpoint = lightning_module.model._call_net(x_t, t)
            endpoint_mols = self.mol_converter.from_batch(endpoint)
            endpoint_bounds = [self._get_uff_bounds(mol) for mol in endpoint_mols]

            loss = self._compute_loss(endpoint, endpoint_bounds)
            loss.backward()

            with torch.no_grad():
                x_t["coords"].sub_(self.lr * torch.sign(x_t["coords"].grad))
                x_t["coords"].grad.zero_()

                masked_coords = apply_mask(x_t["coords"], x_t["padding_mask"])
                x_t["coords"].copy_(masked_coords)

        return x_t

    def _loss_to_interval_edge(self, lb, ub, pwd):
        """Loss to the interval edge."""
        if lb <= pwd <= ub:
            dist_loss = 0
        elif pwd > ub:
            dist_loss = (pwd - ub) ** 2
        elif pwd <= lb:
            dist_loss = (pwd - lb) ** 2
        else:
            raise ValueError(f"Invalid distance: {pwd}, lb: {lb}, ub: {ub}")

        return dist_loss

    def _loss_to_interval_center(self, lb, ub, pwd):
        """Loss to the center of the interval."""
        interval_center = (ub + lb) / 2
        if lb <= pwd <= ub:
            dist_loss = 0
        else:
            dist_loss = (pwd - interval_center) ** 2

        return dist_loss

    def loss_from_bounds(self, tensor_repr, bounds):
        """Return mean squared deviation of pairwise distances from allowed bounds."""
        pairwise_dist = torch.cdist(
            tensor_repr["coords"][~tensor_repr["padding_mask"]],
            tensor_repr["coords"][~tensor_repr["padding_mask"]],
        )

        bounds = bounds.to(device=pairwise_dist.device)

        num_atoms = pairwise_dist.shape[0]

        ub_matrix = torch.triu(bounds, diagonal=0)
        lb_matrix = torch.tril(bounds, diagonal=0).transpose(0, 1)
        pw_dist_matrix = torch.triu(pairwise_dist, diagonal=0)

        above_ub_mask = pw_dist_matrix > ub_matrix
        below_lb_mask = pw_dist_matrix < lb_matrix

        if self.regress_to_center:
            interval_center = (ub_matrix + lb_matrix) / 2
            vectorized_loss = (
                pw_dist_matrix[above_ub_mask] - interval_center[above_ub_mask]
            ).pow(2).sum() + (
                pw_dist_matrix[below_lb_mask] - interval_center[below_lb_mask]
            ).pow(2).sum()
        else:
            vectorized_loss = (
                pw_dist_matrix[above_ub_mask] - ub_matrix[above_ub_mask]
            ).pow(2).sum() + (
                pw_dist_matrix[below_lb_mask] - lb_matrix[below_lb_mask]
            ).pow(2).sum()

        num_pairs = num_atoms * (num_atoms - 1) / 2

        return vectorized_loss / num_pairs

    def _compute_loss(self, pred: TensorDict, bounds: List[torch.Tensor]):
        """Aggregate distance-bound loss over a batch, skipping None bounds."""

        len_valid_bounds = sum(1 for bound in bounds if bound is not None)

        loss_sum = 0
        for i, tensor_bound in enumerate(bounds):
            if tensor_bound is None:
                continue
            loss_sum += self.loss_from_bounds(pred[i], tensor_bound)
        loss = loss_sum / len_valid_bounds

        return loss
