"""Model architectures for image classification."""

from .resnet import ResNet18, ResNet34
from .wideresnet import WideResNet
from .builder import build_model

__all__ = ['ResNet18', 'ResNet34', 'WideResNet', 'build_model']
