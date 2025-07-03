"""Standalone entry-point for sampling molecules with PocketSynth.

Run this file with a trained checkpoint to generate molecules and export them
to SDF / pickle while reporting basic validity metrics. Example:

```bash
python src/sample.py --checkpoint model.ckpt --num_mols 1000 --num_steps 100 -o output
```
"""

import torch
import argparse
import pickle
import lightning as L
from tabasco.callbacks.ema import EMAOptimizer
from tabasco.models.lightning_tabasco import LightningTabasco
from tabasco.chem.convert import MoleculeConverter
from tensordict import TensorDict

torch.set_float32_matmul_precision("high")
L.seed_everything(42)


def sample_batch(
    lightning_module: L.LightningModule,
    ema_optimizer: EMAOptimizer | None,
    num_molecules: int,
    num_steps: int,
) -> TensorDict:
    """Generate a batch of molecules.

    Args:
        lightning_module: Loaded PocketSynth checkpoint.
        ema_optimizer: Optional EMA wrapper; if given, swaps in EMA weights.
        num_molecules: How many molecules to sample.
        num_steps: Number of diffusion steps per trajectory.

    Returns:
        TensorDict with keys `coords`, `atomics`, `padding_mask`.
    """

    if ema_optimizer is None:
        with torch.no_grad():
            out_batch = lightning_module.sample(
                batch_size=num_molecules, num_steps=num_steps
            )
    else:
        with torch.no_grad() and ema_optimizer.swap_ema_weights():
            out_batch = lightning_module.sample(
                batch_size=num_molecules, num_steps=num_steps
            )

    return out_batch


def export_batch_to_pickle(out_batch: TensorDict, out_path: str):
    """Serialize generated molecules and basic metrics.

    Saves two files: `<out_path>.pkl` containing a Python list of RDKit
    molecules (with `None` for invalid ones) and `<out_path>.sdf` containing
    only the valid molecules.
    """
    mol_converter = MoleculeConverter()
    generated_mols = mol_converter.from_batch(out_batch, sanitize=False)

    print(f"Saving generated mols to {out_path}...")
    with open(out_path, "wb") as f:
        pickle.dump(generated_mols, f)


def parse_args():
    """Return CLI arguments parsed with `argparse`."""
    parser = argparse.ArgumentParser(description="Run PocketSynth generation")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the model checkpoint file",
    )
    parser.add_argument(
        "--num_mols",
        type=int,
        default=100,
        help="Number of molecules to generate (default: 100)",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of steps to generate (default: 100)",
    )
    parser.add_argument(
        "--ema_strength", type=float, default=None, help="EMA strength (default: None)"
    )
    parser.add_argument(
        "-o",
        "--output_path",
        type=str,
        default=None,
        help="Path to the output file ending in .pkl (default: None)",
    )
    return parser.parse_args()


def main():
    """Main entry-point: parse args, load model, sample, export."""
    args = parse_args()
    num_mols = args.num_mols
    num_steps = args.num_steps
    lightning_module = LightningTabasco.load_from_checkpoint(args.checkpoint)
    lightning_module.model.net.eval()

    if torch.cuda.is_available():
        lightning_module.to("cuda")

    if args.ema_strength is not None:
        adam_opt = torch.optim.Adam(lightning_module.model.net.parameters())
        ema_optimizer = EMAOptimizer(
            adam_opt,
            "cuda" if torch.cuda.is_available() else "cpu",
            args.ema_strength,
        )
    else:
        ema_optimizer = None

    out_batch = sample_batch(
        lightning_module=lightning_module,
        ema_optimizer=ema_optimizer,
        num_molecules=num_mols,
        num_steps=num_steps,
    )

    if args.output_path is not None:
        export_batch_to_pickle(out_batch, args.output_path)


if __name__ == "__main__":
    main()
