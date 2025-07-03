from dataclasses import dataclass

from tensordict import TensorDict
from torch import Tensor


@dataclass
class FlowPath:
    x_1: TensorDict
    x_t: TensorDict
    dx_t: TensorDict
    x_0: TensorDict
    t: Tensor
