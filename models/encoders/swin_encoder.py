"""
models/encoders/swin_encoder.py

Swin Transformer encoder for few-shot segmentation, built on timm.

Uses timm's features_only=True API to extract intermediate feature maps
at four scales, matching the same interface as ResNetEncoder so the rest
of the system works without any modification.

Supported backbones (any timm model with features_only support):
    "swin_tiny_patch4_window7_224"
    "swin_small_patch4_window7_224"
    "swin_base_patch4_window7_224"
    "swin_base_patch4_window12_384"
    "swinv2_tiny_window8_256"
    "swinv2_base_window8_256"
    "convnext_tiny", "convnext_base", ...  ← also works

Shapes (Swin-Tiny, 512×512 input):
    layer1: B × 96  × 128 × 128   (stride 4)
    layer2: B × 192 × 64  × 64    (stride 8)
    layer3: B × 384 × 32  × 32    (stride 16)
    layer4: B × 768 × 16  × 16    (stride 32)

Important: timm returns Swin features in channels-last format
(B × H × W × C). This encoder permutes them to channels-first
(B × C × H × W) before returning, so downstream modules are unaware
of this implementation detail.
"""

from typing import Dict, List

import torch
import timm

from config.base_config import EncoderConfig
from models.encoders.base_encoder import BaseEncoder


class SwinEncoder(BaseEncoder):
    """Multi-scale Swin Transformer encoder for few-shot segmentation.

    Wraps any timm backbone that supports features_only=True and exposes
    four feature maps under the same "layer1".."layer4" interface as
    ResNetEncoder. The rest of the system needs no changes.

    The encoder is Siamese — the same instance is used for both support
    and query in FewShotModel, just like ResNetEncoder.

    Args:
        cfg: EncoderConfig — backbone field must be a valid timm model name
             that supports features_only=True (e.g. Swin, ConvNeXt).
             frozen_layers is not supported for timm models and is ignored
             with a warning.

    Example:
        cfg = EncoderConfig(
            backbone="swin_tiny_patch4_window7_224",
            pretrained=True,
            in_channels=3,
        )
        encoder = SwinEncoder(cfg)

        features = encoder(x)       # x: B × 3 × 512 × 512
        features["layer4"]          # B × 768 × 16 × 16
        encoder.out_channels        # 768
        encoder.skip_channels       # [96, 192, 384]
    """

    def __init__(self, cfg: EncoderConfig) -> None:
        super().__init__()

        if cfg.in_channels != 3:
            raise ValueError(
                f"SwinEncoder requires in_channels=3 (pretrained Swin uses RGB). "
                f"Got in_channels={cfg.in_channels}. "
                f"If your images are grayscale, replicate the channel before passing."
            )

        if cfg.frozen_layers:
            import warnings
            warnings.warn(
                f"frozen_layers={cfg.frozen_layers} is not supported for SwinEncoder "
                f"and will be ignored. Fine-grained layer freezing for timm models "
                f"is not yet implemented.",
                UserWarning,
            )

        # timm's features_only=True discards the classification head and
        # returns intermediate feature maps at each stage.
        # out_indices=(0,1,2,3) selects all four stages.
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=cfg.img_size,
        )

        # feature_info contains channel counts and strides per stage.
        # We read them once here so out_channels and skip_channels are
        # always consistent with the actual backbone, never hardcoded.
        # feature_info.channels() → [C0, C1, C2, C3]
        channels: List[int] = self.backbone.feature_info.channels()

        # layer4 channels — used by FewShotModel to build the bottleneck
        self._out_channels: int = channels[3]

        # layer1..layer3 channels — used by FewShotModel for skip connections
        # ordered shallow → deep to match _SKIP_CHANNELS convention:
        # [layer3_ch, layer2_ch, layer1_ch]
        self._skip_channels: List[int] = [channels[2], channels[1], channels[0]]

    # ------------------------------------------------------------------
    # BaseEncoder contract
    # ------------------------------------------------------------------

    @property
    def out_channels(self) -> int:
        """Channels in layer4 (deepest feature map).

        Swin-Tiny:  768
        Swin-Small: 768
        Swin-Base:  1024
        Swin-Large: 1536
        """
        return self._out_channels

    @property
    def skip_channels(self) -> List[int]:
        """Channels for skip connections, ordered layer3 → layer2 → layer1.

        Used by FewShotModel to build the decoder dynamically.
        ResNetEncoder does not need this because its values are hardcoded
        in _SKIP_CHANNELS — SwinEncoder exposes them explicitly instead.

        Swin-Tiny:  [384, 192, 96]
        Swin-Small: [384, 192, 96]
        Swin-Base:  [512, 256, 128]
        Swin-Large: [768, 384, 192]
        """
        return self._skip_channels

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from input image tensor.

        Args:
            x: Input tensor of shape B × 3 × H × W.

        Returns:
            Dictionary with keys "layer1".."layer4":
                "layer1": B × C0 × (H/4)  × (W/4)
                "layer2": B × C1 × (H/8)  × (W/8)
                "layer3": B × C2 × (H/16) × (W/16)
                "layer4": B × C3 × (H/32) × (W/32)

            Channel counts depend on the backbone variant — read them
            from encoder.out_channels and encoder.skip_channels.
        """
        # timm returns a list of 4 tensors, one per stage.
        # Swin uses channels-last internally: B × H × W × C
        # We permute to channels-first: B × C × H × W
        stages = self.backbone(x)

        # _to_channels_first handles both channels-last (Swin) and
        # channels-first (ConvNeXt) outputs transparently.
        f1 = self._to_channels_first(stages[0])  # B × C0 × H/4  × W/4
        f2 = self._to_channels_first(stages[1])  # B × C1 × H/8  × W/8
        f3 = self._to_channels_first(stages[2])  # B × C2 × H/16 × W/16
        f4 = self._to_channels_first(stages[3])  # B × C3 × H/32 × W/32

        return {
            "layer1": f1,
            "layer2": f2,
            "layer3": f3,
            "layer4": f4,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_channels_first(x: torch.Tensor) -> torch.Tensor:
        """Convert tensor to channels-first format if needed.

        Args:
            x: Tensor of shape B × H × W × C  or  B × C × H × W.

        Returns:
            Tensor of shape B × C × H × W.
        """
        # If last dim is smaller than dim 1, already channels-first
        if x.shape[1] <= x.shape[-1]:
            return x.permute(0, 3, 1, 2).contiguous()
        return x