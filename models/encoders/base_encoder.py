"""
models/encoders/base_encoder.py

Abstract base class for all backbone encoders.

Defines the contract that every encoder must fulfill so that the rest
of the system — PrototypeModule, SimilarityModule, UNetDecoder,
FewShotModel — can work with any backbone without modification.

Contract:
    1. forward(x) must return a dict with keys "layer1".."layer4",
       each mapping to the feature tensor at that scale.
    2. out_channels must expose the number of channels in "layer4".

Any class that inherits from BaseEncoder and implements these two
things is a valid encoder for the system. Nothing else is required.

Why a separate module?
    FewShotModel should depend on this interface, not on ResNetEncoder
    or SwinEncoder directly. This is the Dependency Inversion Principle:
    high-level modules (FewShotModel) depend on abstractions (BaseEncoder),
    not on concrete implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn


class BaseEncoder(ABC, nn.Module):
    """Abstract base class for all backbone encoders.

    Subclasses must implement:
        - forward():     extract multi-scale features from an input tensor.
        - out_channels:  number of channels in the deepest feature map (layer4).

    Expected output format for forward():
        {
            "layer1": B × C1 × (H/4)  × (W/4),
            "layer2": B × C2 × (H/8)  × (W/8),
            "layer3": B × C3 × (H/16) × (W/16),
            "layer4": B × C4 × (H/32) × (W/32),
        }

    The channel counts C1..C4 vary by backbone — the decoder reads
    them dynamically via out_channels and skip_channels, so no
    hardcoded values are needed anywhere else in the system.

    Example (implementing a custom encoder):
        class MyEncoder(BaseEncoder):
            def __init__(self, cfg):
                super().__init__()
                # build your backbone here

            @property
            def out_channels(self) -> int:
                return 512  # channels in layer4

            def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
                # extract and return features
                return {"layer1": f1, "layer2": f2, "layer3": f3, "layer4": f4}
    """

    def __init__(self) -> None:
        # nn.Module.__init__ must be called explicitly when using multiple
        # inheritance with ABC. ABC.__init__ is a no-op so order does not matter.
        super().__init__()

    @property
    @abstractmethod
    def out_channels(self) -> int:
        """Number of channels in the layer4 (deepest) feature map.

        Used by FewShotModel to compute the bottleneck size:
            bottleneck_channels = encoder.out_channels + 2  (+ similarity map)

        Returns:
            Integer channel count for layer4 output.
        """
        ...

    @property
    @abstractmethod
    def skip_channels(self) -> List[int]:
        """Channels for skip connections, ordered layer3 → layer2 → layer1."""
        ...

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from an input image tensor.

        Args:
            x: Input tensor of shape B × C × H × W.
               C is typically 3 for RGB or preprocessed radiographic images.

        Returns:
            Dictionary with exactly four keys:
                "layer1": B × C1 × (H/4)  × (W/4)
                "layer2": B × C2 × (H/8)  × (W/8)
                "layer3": B × C3 × (H/16) × (W/16)
                "layer4": B × C4 × (H/32) × (W/32)
        """
        ...