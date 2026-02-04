"""Model architectures for image classification."""

from .resnet import ResNet18, ResNet34
from .wideresnet import WideResNet

__all__ = ['ResNet18', 'ResNet34', 'WideResNet']
