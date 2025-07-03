from typing import Dict

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.utilities import grad_norm
from torch.optim.optimizer import Optimizer

from tabasco.chem.convert import MoleculeConverter
from tabasco.utils.metrics import (
    MolecularConnectivity,
    MolecularLipinski,
    MolecularLogP,
    MolecularNovelty,
    MolecularQEDValue,
    MolecularUniqueness,
    MolecularValidity,
    AtomTypeDistribution,
    AtomFractionMetric,
)


class LightningTabasco(L.LightningModule):
    """Thin Lightning wrapper around a flow-matching molecule generator.

    Provides:
    - training / validation loops that delegate to `model`.
    - sampling convenience (`sample`).
    - molecule metrics computed on-device via `torchmetrics`.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """Args:
        model: `nn.Module` that implements `forward` and `sample` like
            `FlowMatchingModel`.
        optimizer: Callable that returns a `torch.optim.Optimizer` when
            passed the model parameters.
        """
        super().__init__()
        self.model = model
        self.save_hyperparameters()

        self.mol_converter = MoleculeConverter()
        self.mol_metrics = torch.nn.ModuleDict(
            {
                "validity": MolecularValidity(sync_on_compute=False),
                "connectivity": MolecularConnectivity(sync_on_compute=False),
                "lipinski": MolecularLipinski(sync_on_compute=False),
                "mol_logp": MolecularLogP(sync_on_compute=False),
                "qed": MolecularQEDValue(sync_on_compute=False),
                "uniqueness": MolecularUniqueness(sync_on_compute=False),
                "fraction_carbon": AtomFractionMetric(
                    atom_symbol="C", sync_on_compute=False
                ),
                "fraction_nitrogen": AtomFractionMetric(
                    atom_symbol="N", sync_on_compute=False
                ),
                "fraction_oxygen": AtomFractionMetric(
                    atom_symbol="O", sync_on_compute=False
                ),
            }
        )

    def set_data_stats(self, stats: Dict):
        """Pass dataset statistics to sub-modules and init metrics that need them."""
        # TODO: fix: this is a hack to get the novelty metric to work
        self.model.set_data_stats(stats)
        all_smiles = stats["all_smiles"]
        self.mol_metrics["novelty"] = MolecularNovelty(
            original_smiles=all_smiles, sync_on_compute=False
        )
        self.mol_metrics["atom_type_distribution"] = AtomTypeDistribution(
            original_smiles=all_smiles, sync_on_compute=False
        )

    def training_step(self, batch):
        """Perform a single training step."""
        loss, _ = self.model(batch, compute_stats=False)
        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )
        return loss

    def sample(self, **kwargs):
        """Sample from the model."""
        with torch.no_grad():
            return self.model.sample(**kwargs)

    def on_before_optimizer_step(self, optimizer: Optimizer) -> None:
        """Log total L2 grad-norm prior to the optimiser update."""
        norms = grad_norm(self, norm_type=2)
        self.log(
            "train/grad_norm",
            norms["grad_2.0_norm_total"].item(),
            on_step=True,
            on_epoch=True,
            sync_dist=True,
        )

    def validation_step(self, batch):
        """Perform a single validation step."""
        loss, stats_dict = self.model(batch, compute_stats=True)

        for k, v in stats_dict.items():
            self.log(f"val/{k}", v, on_epoch=True, sync_dist=True)
        self.log("val/loss", loss, on_epoch=True, sync_dist=True)

    def test_step(self, batch):
        """Perform a single test step."""
        loss = self.model_step(batch)
        self.log("test/loss", loss, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        """Return the optimiser instantiated with current model parameters."""
        return self.hparams.optimizer(params=self.trainer.model.parameters())

    def on_save_checkpoint(self, checkpoint):
        """Add `data_stats` to checkpoint so sampling works after resume."""
        checkpoint["data_stats"] = self.model.data_stats

    def on_load_checkpoint(self, checkpoint):
        """Restore `data_stats` from checkpoint if present."""
        if "data_stats" in checkpoint:
            self.model.data_stats = checkpoint["data_stats"]
