import lightning as L
from lightning import Callback

from tabasco.utils import RankedLogger


log = RankedLogger(__name__, rank_zero_only=True)


class DatasetStatsCallback(Callback):
    """Sync dataset statistics and effective batch size to the Lightning model.

    The datamodule must expose `get_dataset_stats()` and the model must
    implement `set_data_stats(stats)`; otherwise the callback logs a warning
    and continues without raising.
    """

    def _set_data_stats(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Push datamodule statistics to the model.

        Args:
            trainer: PyLightning trainer instance.
            pl_module: The LightningModule that will receive the stats.

        Note:
            - No-op if either side lacks the requisite method.
            - Executed on every rank.
        """
        if hasattr(trainer.datamodule, "get_dataset_stats"):
            stats = trainer.datamodule.get_dataset_stats()
            if hasattr(pl_module, "set_data_stats"):
                log.info("Adding dataset stats from datamodule to model")
                pl_module.set_data_stats(stats=stats)
            else:
                log.warning("Model doesn't implement set_data_stats method")
        else:
            log.warning("Datamodule doesn't implement get_dataset_stats method")

    def _log_effective_batch_size(
        self, trainer: L.Trainer, pl_module: L.LightningModule
    ):
        """Compute and log the *effective* batch size.

        Accounts for random 3-D rotations (`num_random_augmentations`). If a
        W&B logger is attached, the value is written to the run config so it
        appears alongside hyper-parameters.
        """
        if pl_module.model.num_random_augmentations is None:
            effective_batch_size = trainer.datamodule.batch_size
        else:
            size_factor = 1 + pl_module.model.num_random_augmentations
            effective_batch_size = trainer.datamodule.batch_size * size_factor

        log.info(f"Using rotated copies, effective batch size: {effective_batch_size}")

        if hasattr(pl_module, "logger") and pl_module.logger is not None:
            if hasattr(pl_module.logger, "experiment") and hasattr(
                pl_module.logger.experiment, "config"
            ):
                pl_module.logger.experiment.config.update(
                    {
                        "effective_batch_size": effective_batch_size,
                        "augmentation_factor": pl_module.model.num_random_augmentations,
                        "dataset_normalizer": pl_module.mol_converter.dataset_normalizer,
                    }
                )

    def on_fit_start(self, trainer, pl_module):
        """Hook called once at the start of training."""
        self._set_data_stats(trainer, pl_module)

        # Only rank-zero should emit logs to avoid duplication.
        if trainer.global_rank == 0:
            self._log_effective_batch_size(trainer, pl_module)
        return
