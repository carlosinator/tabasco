import torch
from tqdm import tqdm
from tabasco.utils.tensor_ops import apply_mask
from typing import Callable, Optional
from pytorch_lightning import LightningModule
from torch.optim import Optimizer


class GuidedSampling:
    """Utility that performs guided sampling over predefined timesteps."""

    def __init__(
        self,
        lightning_module: LightningModule,
        inpaint_function: Optional[Callable] = None,
        steer_interpolant: Optional[Callable] = None,
        ema_optimizer: Optional[Optimizer] = None,
    ):
        """Args:
        lightning_module: Trained LightningModule containing the diffusion model.
        inpaint_function: Optional callable that fills or fixes parts of `x_t` after each step.
        steer_interpolant: Optional callable applied before each model step to guide the sample.
        ema_optimizer: If provided, swaps EMA parameters for inference.
        """
        self.lightning_module = lightning_module
        self.inpaint_function = inpaint_function
        self.steer_interpolant = steer_interpolant
        self.ema_optimizer = ema_optimizer

    def sample(self, x_t, timesteps):
        """Iteratively denoise `x_t` following `timesteps`.

        Args:
            x_t: Initial noisy TensorDict at the first timestep.
            timesteps: 1-D tensor or list of monotonically increasing timesteps.

        Returns:
            TensorDict representing the final denoised sample.
        """
        self.lightning_module.model.net.eval()

        x_t = x_t.detach().clone()
        pbar = tqdm(range(1, len(timesteps)))

        for i in pbar:
            t = timesteps[i - 1]
            dt = timesteps[i] - timesteps[i - 1]

            if self.steer_interpolant is not None:
                x_t = self.steer_interpolant(self.lightning_module, x_t, t)

            x_t["coords"] = apply_mask(x_t["coords"], x_t["padding_mask"])

            if self.ema_optimizer is not None:
                with torch.no_grad() and self.ema_optimizer.swap_ema_weights():
                    x_t = self.lightning_module.model._step(x_t, t, dt)
            else:
                with torch.no_grad():
                    x_t = self.lightning_module.model._step(x_t, t, dt)

            if self.inpaint_function is not None:
                x_t = self.inpaint_function(x_t, t)

        x_t["coords"] = apply_mask(x_t["coords"], x_t["padding_mask"])

        return x_t
