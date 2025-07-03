import os
import tempfile
from typing import List

import datamol as dm
import lightning as L
import wandb
import contextlib as clib
from lightning import Callback
from rdkit import Chem
from tensordict import TensorDict

from tabasco.chem.convert import MoleculeConverter
from tabasco.utils import RankedLogger
from tabasco.chem.utils import largest_component


log = RankedLogger(__name__, rank_zero_only=True)


class SaveGeneratedMolsCallback(Callback):
    """Log generated molecules, trajectories, and SDF files to Weights & Biases.

    Performs several logging passes:
        - Sample preview molecules (`_log_mols_to_display`).
        - Highlight novel molecules not present in the training set.
        - Store invalid molecules for debugging.
        - Persist full SDF and selected trajectories as W&B artifacts.
    Heavy I/O is gated by `compute_every` and executed only on rank-0.
    """

    def __init__(
        self,
        num_samples: int = 100,
        num_trajectories: int = 10,
        num_display_mols: int = 10,
        num_sampling_steps: int = 100,
        compute_every: int = 1000,
    ):
        """Args:
        num_samples: Molecules sampled each evaluation.
        num_trajectories: Trajectories to save as SDF files.
        num_display_mols: Maximum molecules displayed inline in W&B.
        num_sampling_steps: Reverse-diffusion steps in sampling.
        compute_every: Scheduler interval in global steps.
        """
        super().__init__()
        self.num_samples = num_samples
        self.num_trajectories = num_trajectories
        self.num_display_mols = num_display_mols
        self.num_sampling_steps = num_sampling_steps
        self.mol_converter = MoleculeConverter()

        self.compute_every = compute_every
        self.next_compute = 0

    def _log_mols_to_display(
        self, lightning_module: L.LightningModule, full_mol_list: List[Chem.Mol]
    ):
        """Log a small gallery of valid molecules to W&B media panel."""
        wandb_dict = {}

        valid_mol_list = [mol for mol in full_mol_list if mol is not None][
            : self.num_display_mols
        ]
        for i, mol in enumerate(valid_mol_list):
            try:
                wandb_dict[f"gen_mol_{i}"] = wandb.Molecule.from_rdkit(
                    mol, convert_to_3d_and_optimize=False
                )
            except Exception as e:
                log.warning(f"Error logging molecule {i}: {e}")

        lightning_module.logger.experiment.log(wandb_dict)

    def _log_novel_mols(
        self, lightning_module: L.LightningModule, mol_list: List[Chem.Mol]
    ):
        """Log molecules whose SMILES are unseen in the training set to W&B."""
        valid_mols = [mol for mol in mol_list if mol is not None]
        valid_mols = largest_component(valid_mols)

        mol_smiles = [Chem.MolToSmiles(mol) for mol in valid_mols]
        existing_smiles = set(lightning_module.model.data_stats["all_smiles"])

        novel_mols = []
        for mol, smiles in zip(valid_mols, mol_smiles):
            if smiles not in existing_smiles:
                novel_mols.append(mol)
        novel_mols = novel_mols[: self.num_display_mols]

        wandb_dict = {}
        for i, mol in enumerate(novel_mols):
            wandb_dict[f"novel_mol_{i}"] = wandb.Molecule.from_rdkit(
                mol, convert_to_3d_and_optimize=False
            )

        lightning_module.logger.experiment.log(wandb_dict)

    def _log_invalid_mols(
        self,
        lightning_module: L.LightningModule,
        mol_list: List[Chem.Mol],
        generated_batch: TensorDict,
    ):
        """Render invalid reconstructions for debugging purposes.

        A molecule is deemed invalid when RDKit sanitisation of the sampled
        graph fails; the raw tensor is converted with `sanitize=False` to
        visualise what went wrong.
        """
        invalid_mols = []
        for mol, tensor_repr in zip(mol_list, generated_batch):
            if mol is None:
                raw_mol = self.mol_converter.from_tensor(tensor_repr, sanitize=False)
                if raw_mol is None:
                    print("FAILED to convert tensor to mol")
                    continue

                invalid_mols.append(raw_mol)

        invalid_mols = invalid_mols[: self.num_display_mols]

        wandb_dict = {}
        for i, mol in enumerate(invalid_mols):
            wandb_dict[f"invalid_mol_{i}"] = wandb.Molecule.from_rdkit(
                mol, convert_to_3d_and_optimize=False
            )

        lightning_module.logger.experiment.log(wandb_dict)

    def _log_as_sdf(
        self, lightning_module: L.LightningModule, mol_list: List[Chem.Mol]
    ):
        """Save the full set of molecules as an SDF artifact in W&B."""
        with tempfile.TemporaryDirectory() as tmpdir:
            sdf_path = os.path.join(tmpdir, "generated_mols.sdf")

            dm.to_sdf(mol_list, urlpath=sdf_path)

            artifact = wandb.Artifact(
                f"generated_mols_epoch_{lightning_module.current_epoch}", type="sdf"
            )
            artifact.add_file(sdf_path)
            lightning_module.logger.experiment.log_artifact(artifact)

    def _log_trajectories(
        self, lightning_module: L.LightningModule, trajectories: List[List[Chem.Mol]]
    ):
        """Log select diffusion trajectories as SDF artifacts in W&B.

        Each trajectory is a list of RDKit molecules representing snapshots
        over time; files are added under a single artifact for convenience.
        """

        mol_snapshots = []
        for batch_at_time in trajectories:
            mols_at_time = lightning_module.mol_converter.from_batch(
                batch_at_time, sanitize=False
            )
            mol_snapshots.append(mols_at_time)

        mol_trajectories = []
        for i in range(len(mol_snapshots[0])):
            one_trajectory = []
            for mol_snapshot in mol_snapshots:
                one_trajectory.append(mol_snapshot[i])
            mol_trajectories.append(one_trajectory)

        # create artifact outside the loop
        artifact = wandb.Artifact(
            f"trajectory_{lightning_module.current_epoch}", type="sdf"
        )

        # track if successfully added any trajectories
        any_trajectory_added = False

        for i in range(min(self.num_trajectories, len(mol_trajectories))):
            try:
                with tempfile.TemporaryDirectory() as tmpdir:
                    sdf_path = os.path.join(tmpdir, f"trajectory_{i}.sdf")

                    with clib.suppress(Exception):
                        dm.to_sdf(mol_trajectories[i], urlpath=sdf_path)
                        if os.path.exists(sdf_path) and os.path.getsize(sdf_path) > 0:
                            artifact.add_file(sdf_path, policy="mutable")
                            any_trajectory_added = True
            except Exception as e:
                log.warning(f"Failed to log trajectory {i}: {e}")
                continue

        # only log the artifact if we successfully added at least one trajectory
        if any_trajectory_added:
            try:
                lightning_module.logger.experiment.log_artifact(artifact)
            except Exception as e:
                log.warning(f"Failed to log artifact: {e}")

    def on_validation_epoch_end(
        self, trainer: L.Trainer, lightning_module: L.LightningModule
    ) -> None:
        """Core hook: sample molecules and trigger logging helpers."""
        # Only run on the main process (rank 0) to avoid duplicate logging
        if trainer.global_rank != 0:
            return

        if trainer.global_step < self.next_compute:
            return
        self.next_compute += self.compute_every

        if lightning_module.logger is None:
            log.warning("Logger is not set, skipping logging of molecules")
            return

        generated_batch, trajectories = lightning_module.sample(
            batch_size=self.num_samples,
            num_steps=self.num_sampling_steps,
            return_trajectories=True,
        )
        mol_list = lightning_module.mol_converter.from_batch(generated_batch)

        self._log_mols_to_display(lightning_module, mol_list)
        self._log_novel_mols(lightning_module, mol_list)
        self._log_invalid_mols(lightning_module, mol_list, generated_batch)

        self._log_as_sdf(lightning_module, mol_list)

        # TODO: fix: randomly throws logging errors
        try:
            self._log_trajectories(lightning_module, trajectories)
        except Exception as e:
            log.warning(f"Failed to log trajectories: {e}")
