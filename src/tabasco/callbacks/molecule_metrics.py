import lightning as L
from lightning import Callback

from tabasco.chem.convert import MoleculeConverter
from tabasco.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class MoleculeMetricsCallback(Callback):
    """Periodically sample molecules and log evaluation metrics.

    Intended for use in validation where sampling is cheap relative to
    training. Executes only on the main process (`global_rank == 0`) to avoid
    duplicate work and noisy logging.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_sampling_steps: int = 100,
        compute_every: int = 1000,
    ):
        """Args:
        num_samples: Number of molecules to draw per evaluation.
        num_sampling_steps: Iterations for the model's sampler.
        compute_every: Global-step interval between evaluations.
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_sampling_steps = num_sampling_steps
        self.mol_converter = MoleculeConverter()

        self.next_compute = 0
        self.compute_every = compute_every

    def on_validation_epoch_end(
        self, trainer: L.Trainer, lightning_module: L.LightningModule
    ) -> None:
        """Sample molecules and update metric objects.

        Args:
            trainer: Lightning Trainer instance.
            lightning_module: LightningModule with `sample` and `mol_metrics` attrs.

        Notes:
            - Runs only on rank-0 for distributed training.
            - Heavy operation: sampling + metric computation; thus spaced by
              `compute_every` steps.
            - Metrics are expected to be torchmetrics.Metric callables stored in
              `lightning_module.mol_metrics` and to be compatible with a list
              of RDKit Mol objects.
        """
        if trainer.global_rank != 0:
            return

        if trainer.global_step < self.next_compute:
            return
        self.next_compute += self.compute_every

        generated_batch = lightning_module.sample(
            batch_size=self.num_samples, num_steps=self.num_sampling_steps
        )
        mol_list = lightning_module.mol_converter.from_batch(generated_batch)

        # compute and log metrics
        for k, metric in lightning_module.mol_metrics.items():
            metric(mol_list)
            lightning_module.log(f"val/{k}", metric)
