"""Sample molecules with optional UFF-bound guidance.

This helper script first denoises a batch without guidance for a configurable
number of steps, then switches to `UFFBoundGuidance` to fine-tune structures.
"""

import torch
import argparse
import pickle
import os
from tabasco.models.lightning_tabasco import LightningTabasco
from tabasco.chem.convert import MoleculeConverter
from tabasco.callbacks.ema import EMAOptimizer

from tabasco.sample.interpolant_guidance import UFFBoundGuidance
from tabasco.sample.guided_sampling import GuidedSampling


def main():
    """Parse CLI args, run guided sampling, and save results."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step-switch", type=int, help="Step at which to start guidance", required=True
    )
    parser.add_argument(
        "--ckpt", type=str, help="Path to model checkpoint", required=True
    )
    parser.add_argument("--batch-size", type=int, help="Batch size", required=True)
    parser.add_argument(
        "--output-dir", type=str, help="Path to output directory", default=None
    )
    parser.add_argument(
        "--guidance", type=float, help="Guidance strength", default=0.01
    )
    parser.add_argument(
        "--to-center", type=bool, help="Regress to center", default=True
    )
    args = parser.parse_args()

    lightning_module = LightningTabasco.load_from_checkpoint(args.ckpt)
    lightning_module = lightning_module.to("cuda")
    ema_optimizer = torch.optim.Adam(lightning_module.model.net.parameters())
    ema_optimizer = EMAOptimizer(ema_optimizer, "cuda", 0.999)

    lightning_module.eval()

    mol_converter = MoleculeConverter()

    uff_bound_guidance = UFFBoundGuidance(
        mol_converter=mol_converter,
        lr=args.guidance,
        n_steps=1,
        regress_to_center=args.to_center,
    )

    unguided_sampling = GuidedSampling(
        lightning_module=lightning_module,
        inpaint_function=None,
        steer_interpolant=None,
        ema_optimizer=ema_optimizer,
    )

    guided_sampling = GuidedSampling(
        lightning_module=lightning_module,
        inpaint_function=None,
        steer_interpolant=uff_bound_guidance,
        ema_optimizer=ema_optimizer,
    )

    noisy_batch = lightning_module.model._sample_noise_like_batch(
        batch_size=args.batch_size
    )

    T = lightning_module.model._get_sample_schedule(num_steps=100)
    T = T.to(noisy_batch.device)[:, None]
    T = T.repeat(1, noisy_batch["coords"].shape[0])

    partially_denoised_batch = unguided_sampling.sample(
        x_t=noisy_batch, timesteps=T[: args.step_switch]
    )

    boosted_batch = guided_sampling.sample(
        x_t=partially_denoised_batch.detach().clone(),
        timesteps=T[args.step_switch :],
    )

    boosted_mols = mol_converter.from_batch(boosted_batch)

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "boosted_mols.pkl"), "wb") as f:
            pickle.dump(boosted_mols, f)


if __name__ == "__main__":
    main()
