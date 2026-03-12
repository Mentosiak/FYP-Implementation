"""
Model builder utilities.
"""

from src.models import ResNet18, ResNet34, WideResNet
from src.utils.config import ModelConfig


def build_model(model_cfg: ModelConfig, num_classes: int = 10):
    """
    Build model from configuration.
    
    Args:
        model_cfg: Model configuration
        num_classes: Number of output classes
        
    Returns:
        Model instance
    """
    arch = model_cfg.architecture.lower()
    
    if arch == 'resnet18':
        return ResNet18(num_classes=num_classes)
    elif arch == 'resnet34':
        return ResNet34(num_classes=num_classes)
    elif arch == 'wideresnet':
        return WideResNet(
            depth=model_cfg.depth,
            widen_factor=model_cfg.widen_factor,
            num_classes=num_classes,
            dropout_rate=model_cfg.dropout_rate
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
