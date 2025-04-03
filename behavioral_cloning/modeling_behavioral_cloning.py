"""Behavioral Cloning imitation learning implementation for use with HuggingFace LeRobot for use with SO100
robotic arms from The Robot Studio.

Basic structure used here is:
image + robot state (6 servo positions) --> behavioral cloning policy --> "action" (6 target servo positions)


"""

# import packages

import math
from collections import deque
from typing import Callable

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn

from lerobot.lerobot.common.constants import OBS_ENV, OBS_ROBOT
from configuration_behavioral_cloning import BehavioralCloningConfig
from lerobot.lerobot.common.policies.normalize import Normalize, Unnormalize
from lerobot.lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.lerobot.common.policies.diffusion.modeling_diffusion import DiffusionRgbEncoder
from lerobot.lerobot.common.policies.diffusion.modeling_diffusion import SpatialSoftmax
from lerobot.lerobot.common.policies.utils import (
    get_device_from_parameters,
    get_dtype_from_parameters,
    get_output_shape,
    populate_queues,
)

# policy 
class BehavioralCloning(nn.Module):
    """Behavioral cloning policy implemented as basic supervised 
    learning for performance comparison with Diffusion Policy
    """

    config_class = BehavioralCloningConfig
    name = "bclone"

    def __init__(
        self,
        config: BehavioralCloningConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
    ):
        """_summary_

        Args:
            config (BehavioralCloningConfig): Policy configuration class instance or None, in which case the default instantiation of
                the configuration class is used
            dataset_stats: Dataset statistics to be used for normalization. If not passed here, it is expected
                that they will be passed with a call to `load_state_dict` before the policy is used.
                """
        
        super().__init__(config)
        config.validate_features()
        self.config = config

        # normalize inputs and outputs to maintain consistency with lerobot
        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        def get_optim_params(self) -> dict:
            return self.diffusion.parameters()

        @torch.no_grad
        def action_selection(self, batch: dict[str, Tensor]) -> Tensor:
            """_summary_

            Args:
                batch (dict[str, Tensor]): Observation from environment (images + robot position)

            Returns:
                Tensor: Action for robot to take (list of servo positions to move to)
            """

            batch = self.normalize_inputs(batch)
            if self.config.image_features:
                batch = dict(batch) # shallow copy so that adding a key doesn't modify the original
                batch["observation.images"] = torch.stack(
                    [batch[key] for key in self.config.image_features], dim=-4
                )
            batch = self.normalize_targets(batch)

# policy network




