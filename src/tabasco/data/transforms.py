import torch
from scipy.spatial.transform import Rotation
from tensordict import TensorDict
from math import prod


def random_rotation(data: TensorDict) -> TensorDict:
    """Apply a single random 3D rotation to data['coords']."""

    R = Rotation.random()
    data["coords"] = torch.tensor(R.apply(data["coords"]), dtype=torch.float32)
    return data


def permute_atoms(data: TensorDict) -> TensorDict:
    """Randomly permute the order of non-padded atoms in a molecule tensor."""
    real_atom_indices = torch.where(~data["padding_mask"])[0]
    num_real_atoms = len(real_atom_indices)

    # Create a permutation for the real atom indices
    perm_indices = real_atom_indices[torch.randperm(num_real_atoms)]

    # Create the full permutation vector
    perm = torch.arange(data["coords"].shape[0])
    perm[real_atom_indices] = perm_indices

    # Apply the permutation
    data["coords"] = data["coords"][perm]
    data["atomics"] = data["atomics"][perm]
    data["padding_mask"] = data["padding_mask"][perm]

    return data


def sample_uniform_rotation(
    shape: torch.Size, dtype: torch.dtype, device: torch.device
) -> torch.Tensor:
    """Sample uniform random rotation matrices.

    Reference: NVIDIA-Digital-Bio/proteina implementation.
    """
    return torch.tensor(
        Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


def apply_random_rotation(batch: TensorDict, n_augmentations=10) -> TensorDict:
    """Augment a batch with `n_augmentations` additional random rotations.

    Args:
        batch: TensorDict with keys `coords`, `padding_mask`, `atomics`.
        n_augmentations: Number of extra rotated copies to generate.

    Returns:
        TensorDict with keys `coords`, `padding_mask`, `atomics` and
        `n_augmentations` extra copies.

    Reference: NVIDIA-Digital-Bio/proteina implementation.
    """
    naug = n_augmentations + 1
    assert batch["coords"].ndim == 3, (
        f"Augmetations can only be used for simple (x_1) batches [b, n, 3], current shape is {batch['coords'].shape}"
    )
    assert batch["padding_mask"].ndim == 2, (
        f"Augmetations can only be used for simple (mask) batches [b, n], current shape is {batch['padding_mask'].shape}"
    )
    assert naug >= 1, f"Number of augmentations (int) should >= 1, currently {naug}"

    x = batch["coords"].repeat(naug, 1, 1).to(batch.device)
    mask = batch["padding_mask"].repeat(naug, 1).to(batch.device)
    atomics = batch["atomics"].repeat(naug, 1, 1).to(batch.device)

    rotations = sample_uniform_rotation(
        shape=x.shape[:-2], dtype=x.dtype, device=x.device
    )

    x_rot = torch.matmul(x, rotations).to(batch.device)

    augmented_batch = TensorDict(
        {
            "coords": x_rot,
            "padding_mask": mask,
            "atomics": atomics,
        },
        batch_size=x_rot.shape[0],
    ).to(batch.device)

    return augmented_batch
