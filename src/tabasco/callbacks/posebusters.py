import lightning as L
from lightning import Callback

from tabasco.utils import RankedLogger

from posebusters import PoseBusters
from yaml import safe_load

log = RankedLogger(__name__, rank_zero_only=True)


class PoseBustersCallback(Callback):
    """Compute PoseBusters quality metrics during validation.

    Only rank-0 computes the metrics to avoid redundant heavy work.
    """

    def __init__(
        self,
        num_samples: int = 1,
        num_sampling_steps: int = 100,
        compute_every: int = 1000,
        config_file: str = "./src/tabasco/utils/posebusters_no_strain.yaml",
    ):
        """Args:
        num_samples: Molecules to sample per evaluation.
        num_sampling_steps: Sampling iterations per molecule.
        compute_every: Global-step interval between evaluations.
        config_file: YAML file consumed by PoseBusters for thresholds.
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_sampling_steps = num_sampling_steps

        self.cfg_file = safe_load(open(config_file, encoding="utf-8"))
        self.posebusters = PoseBusters(config=self.cfg_file)

        self.compute_every = compute_every
        self.next_compute = 0

    def on_validation_epoch_end(
        self, trainer: L.Trainer, lightning_module: L.LightningModule
    ) -> None:
        """Sample molecules and log PoseBusters metrics.

        Skips execution on non-zero ranks and throttles by `compute_every`.
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
        total_num_mols = len(mol_list)

        valid_mols = [mol for mol in mol_list if mol is not None]

        if len(valid_mols) == 0:
            return

        try:
            results = self.posebusters.bust(mol_pred=valid_mols)
        except RuntimeError:
            log.error("Posebusters computation raised an error.")
            return

        posebusters_sum = 0.0
        for _, row in results.iterrows():
            posebusters_sum += 0 if row.isin([False]).any() else 1

        lightning_module.log("val/pb_intersection", posebusters_sum / total_num_mols)

        for column in results.columns:
            fraction_valid = (1.0 * results[column].sum()) / total_num_mols
            lightning_module.log(f"val/pb_{column}", fraction_valid)
