"""Copyright (c) Meta Platforms, Inc.

and affiliates.
"""

import logging
import os
from functools import partial
from typing import Dict

import numpy as np
import torch
from openbabel import openbabel
from posebusters import PoseBusters
from pymatgen.core import Molecule
from rdkit import Chem, RDLogger
from rdkit.Chem import Draw

import wandb
from tabasco.utils import joblib_map

RDLogger.DisableLog("rdApp.*")

openbabel.obErrorLog.StopLogging()

logging.getLogger("posebusters").setLevel(logging.CRITICAL)


class MoleculeGenerationEvaluator:
    """Evaluator for molecule generation tasks.

    Can be used within a Lightning module by appending sampled structures and computing metrics at
    the end of an epoch.
    """

    def __init__(self, dataset_smiles_list, removeHs=True, tolerance=0.01, device="cpu"):
        self.dataset_smiles_list = dataset_smiles_list
        self.removeHs = removeHs
        # self.matcher = MoleculeMatcher(tolerance=tolerance)
        self.buster = PoseBusters(config="mol")
        self.pred_arrays_list = []
        self.pred_mol_list = []
        self.pred_rdkit_list = []
        self.device = device

    def append_pred_array(self, pred: Dict):
        """Append a prediction to the evaluator."""
        self.pred_arrays_list.append(pred)

    def clear(self):
        """Clear the stored predictions, to be used at the end of an epoch."""
        self.pred_arrays_list = []
        self.pred_mol_list = []
        self.pred_rdkit_list = []

    def _arrays_to_molecules(self, save: bool = False, save_dir: str = "", n_jobs: int = -4):
        """Convert stored predictions and ground truths to Molecule objects for evaluation."""
        self.pred_mol_list = joblib_map(
            partial(
                array_dict_to_molecule,
                save=save,
                save_dir_name=save_dir,
            ),
            self.pred_arrays_list,
            n_jobs=n_jobs,
            inner_max_num_threads=1,
            desc="    Pred to Molecule",
            total=len(self.pred_arrays_list),
        )

    def get_metrics(self, save: bool = True, save_dir: str = "", n_jobs: int = -4):
        """Compute metrics for the generated molecules."""
        assert len(self.pred_arrays_list) > 0, "No predictions to evaluate."
        assert save, "Metric computation currently requires saving as pdb files."

        # Convert predictions to Molecule objects
        self._arrays_to_molecules(save, save_dir, n_jobs=n_jobs)

        valid_molecules = []
        valid_smiles = []
        for idx in range(len(self.pred_mol_list)):
            sample_idx = self.pred_mol_list[idx].properties["sample_idx"]
            try:
                m = Chem.MolFromPDBFile(
                    os.path.join(save_dir, f"molecule_{sample_idx}.pdb"),
                    removeHs=self.removeHs,
                )
                pred_smiles = Chem.MolToSmiles(m, isomericSmiles=True)
                pred_2d = wandb.Image(Draw.MolToImage(m))

                # simple fragment-based validity check
                m_frags = Chem.rdmolops.GetMolFrags(m, asMols=True)
                largest_frag = max(m_frags, default=m, key=lambda frag: frag.GetNumAtoms())
                pred_smiles = Chem.MolToSmiles(largest_frag, isomericSmiles=True)
                valid = True
                pb_valid = True
                valid_molecules.append(m)
                valid_smiles.append(pred_smiles)

            except Exception as e:
                # log.error(f"Failed to convert molecule {sample_idx} to Smiles")
                m = None
                pred_smiles = ""
                pred_2d = None
                valid = False
                pb_valid = False

            # Update list (used for wandb table)
            self.pred_rdkit_list.append((m, pred_smiles, pred_2d, valid, pb_valid))

        # Define consistent set of keys for PoseBusters metrics
        pb_keys = [
            "mol_pred_loaded",
            "sanitization",
            "inchi_convertible",
            "all_atoms_connected",
            "bond_lengths",
            "bond_angles",
            "internal_steric_clash",
            "aromatic_ring_flatness",
            "double_bond_flatness",
            "internal_energy",
            "posebusters_rate",
        ]

        if len(valid_smiles) > 0:
            unique_smiles = set(valid_smiles)
            novel_smiles = (
                unique_smiles.difference(set(self.dataset_smiles_list))
                if self.dataset_smiles_list is not None
                else set()
            )
            validity_metrics_dict = {
                "valid_rate": torch.tensor(
                    len(valid_smiles) / len(self.pred_rdkit_list), device=self.device
                ),
                "unique_rate": torch.tensor(
                    len(unique_smiles) / len(valid_smiles), device=self.device
                ),
                "novel_rate": torch.tensor(
                    len(novel_smiles) / len(valid_smiles), device=self.device
                ),
            }
            pb_metrics = self.buster.bust(valid_molecules, None, None)
            pb_metrics["posebusters_rate"] = pb_metrics.all(axis=1)
            pb_metrics_dict_raw = pb_metrics.mean().to_dict()

            # Ensure all keys are present
            pb_metrics_dict = {k: pb_metrics_dict_raw.get(k, 0.0) for k in pb_keys}

            # Update pred_rdkit_list with PoseBusters validity after bulk screening
            valid_idx = 0
            for idx in range(len(self.pred_rdkit_list)):
                if self.pred_rdkit_list[idx][3]:  # valid
                    self.pred_rdkit_list[idx] = (
                        self.pred_rdkit_list[idx][0],
                        self.pred_rdkit_list[idx][1],
                        self.pred_rdkit_list[idx][2],
                        self.pred_rdkit_list[idx][3],
                        bool(pb_metrics["posebusters_rate"].iloc[valid_idx]),
                    )
                    valid_idx += 1

        else:
            validity_metrics_dict = {
                "valid_rate": torch.tensor(0.0, device=self.device),
                "unique_rate": torch.tensor(0.0, device=self.device),
                "novel_rate": torch.tensor(0.0, device=self.device),
            }
            # Ensure all keys are present with default 0.0 value
            pb_metrics_dict = {k: 0.0 for k in pb_keys}

        metrics_dict = {**validity_metrics_dict, **pb_metrics_dict}
        return metrics_dict

    def get_wandb_table(self, current_epoch: int = 0, save_dir: str = ""):
        """Create a wandb.Table for the predicted molecules."""
        # Log molecule structures and metrics to wandb
        pred_table = wandb.Table(
            columns=[
                "Global step",
                "Sample idx",
                "Num atoms",
                "Valid?",
                "PoseBusters valid?",
                "Pred atom types",
                "Pred Smiles",
                "Pred 2D",
                "Pred 3D",
            ]
        )

        for idx in range(len(self.pred_mol_list)):
            sample_idx = self.pred_mol_list[idx].properties["sample_idx"]

            num_atoms = len(self.pred_mol_list[idx].atomic_numbers)

            pred_atom_types = " ".join(
                [str(int(t)) for t in self.pred_mol_list[idx].atomic_numbers]
            )

            pred_smiles = self.pred_rdkit_list[idx][1]

            pred_2d = self.pred_rdkit_list[idx][2]

            valid = self.pred_rdkit_list[idx][3]

            pb_valid = self.pred_rdkit_list[idx][4]

            pred_3d = wandb.Molecule(os.path.join(save_dir, f"molecule_{sample_idx}.pdb"))

            # Update table
            pred_table.add_data(
                current_epoch,
                sample_idx,
                num_atoms,
                valid,
                pb_valid,
                pred_atom_types,
                pred_smiles,
                pred_2d,
                pred_3d,
            )

        return pred_table


def array_dict_to_molecule(
    x: dict[str, np.ndarray],
    save: bool = False,
    save_dir_name: str = "",
) -> Molecule:
    """Method to convert a dictionary of numpy arrays to a Molecule object which is compatible with
    MoleculeMatcher (used for evaluations).

    Args:
        x: Dictionary of numpy arrays with keys:
            - 'atom_types': Atomic numbers of atoms.
            - 'pos': 3D coordinates of atoms.
            - 'sample_idx': Index of the sample in the dataset.
        save: Whether to save the molecule as a pdb file.
        save_dir_name: Directory to save the pdb file.

    Returns:
        Molecule: Molecule object, optionally saved as a pdb file.
    """
    mol = Molecule(
        species=x["atom_types"],
        coords=x["pos"],
        properties={"sample_idx": x["sample_idx"]},
    )
    if save:
        os.makedirs(save_dir_name, exist_ok=True)
        mol.to(os.path.join(save_dir_name, f"molecule_{x['sample_idx']}.pdb"), fmt="pdb")
    return mol
