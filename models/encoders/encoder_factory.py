"""
models/encoders/encoder_factory.py

Factory function for building backbone encoders from config.

This is the single point of contact between FewShotModel and the encoder
implementations. FewShotModel calls build_encoder(cfg) and receives a
BaseEncoder — it never imports ResNetEncoder or SwinEncoder directly.

Adding a new backbone:
    - If it is a torchvision ResNet: add it to ResNetEncoder._BACKBONE_REGISTRY.
    - If it is a timm model with features_only support: just set backbone to
      its timm name in config — SwinEncoder handles it automatically.
    - If it needs a completely new class: create it, inherit from BaseEncoder,
      and add a branch in build_encoder().

Why a factory and not a class?
    A function is sufficient here — there is no state to manage and no
    need for inheritance. A factory function is simpler and easier to read
    than a Factory class with a single create() method.
"""

from config.base_config import EncoderConfig
from models.encoders.base_encoder import BaseEncoder
from models.encoders.resnet_encoder import ResNetEncoder, _BACKBONE_REGISTRY as _RESNET_REGISTRY
from models.encoders.swin_encoder import SwinEncoder


def build_encoder(cfg: EncoderConfig) -> BaseEncoder:
    """Instantiate and return the correct encoder for the given config.

    Decision logic:
        - backbone name in ResNet registry → ResNetEncoder (torchvision)
        - anything else                    → SwinEncoder (timm)

    This means any timm backbone that supports features_only=True works
    out of the box: Swin variants, SwinV2, ConvNeXt, EfficientNet, etc.

    Args:
        cfg: EncoderConfig — backbone field determines which class is used.

    Returns:
        A BaseEncoder instance ready to use as a Siamese encoder.

    Raises:
        No errors for unknown backbone names — timm will raise its own
        clear error if the model name is not found in its registry.

    Example:
        # ResNet (torchvision)
        cfg = EncoderConfig(backbone="resnet34", pretrained=True)
        encoder = build_encoder(cfg)   # → ResNetEncoder
        encoder.out_channels           # 512

        # Swin Transformer (timm)
        cfg = EncoderConfig(backbone="swin_tiny_patch4_window7_224", pretrained=True)
        encoder = build_encoder(cfg)   # → SwinEncoder
        encoder.out_channels           # 768
        encoder.skip_channels          # [384, 192, 96]

        # ConvNeXt (timm) — works without any additional code
        cfg = EncoderConfig(backbone="convnext_tiny", pretrained=True)
        encoder = build_encoder(cfg)   # → SwinEncoder (timm path)
        encoder.out_channels           # 768
    """
    if cfg.backbone in _RESNET_REGISTRY:
        return ResNetEncoder(cfg)

    # Everything else goes through timm.
    # SwinEncoder handles any timm backbone with features_only support.
    return SwinEncoder(cfg)