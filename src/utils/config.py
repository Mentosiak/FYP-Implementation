"""
Configuration loading utilities.
Handles YAML config parsing and provides structured config objects.
"""

from __future__ import annotations
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    """Dataset configuration."""
    name: str = "cifar10"
    data_dir: str = "./data"
    num_classes: int = 10


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    architecture: str = "resnet18"
    dropout_rate: float = 0.0
    widen_factor: int = 2  # For WideResNet
    depth: int = 28  # For WideResNet


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    epochs: int = 200
    batch_size: int = 128
    learning_rate: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 0.0005
    warmup_epochs: int = 0
    validation_split: float = 0.0
    stop_loss_threshold: float | None = None
    stop_loss_warmup_epochs: int = 12
    supervised_algorithm: str = "standard"  # standard, mixup, or cutmix
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    cutmix_prob: float = 0.5
    save_best: bool = True
    save_last: bool = True


@dataclass
class SSLConfig:
    """Semi-supervised learning specific configuration."""
    enabled: bool = False
    algorithm: str = "pseudolabel"  # pseudolabel, mixmatch, fixmatch, etc.
    labels_per_class: int | None = 250  # None means use label_fraction
    label_fraction: float | None = None  # e.g., 0.1 for 10%
    unlabeled_batch_size: int = 256
    confidence_threshold: float = 0.95
    unlabeled_loss_weight: float = 1.0
    strong_augment: bool = True
    temperature: float = 1.0  # For pseudo-label sharpening
    mixmatch_alpha: float = 0.75
    mixmatch_temperature: float = 0.5
    seed: int = 42


@dataclass
class SystemConfig:
    """System and environment configuration."""
    seed: int = 42
    num_workers: int = 2
    device: str = "cuda"
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    experiment_dir: str = "experiments"


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    ssl: SSLConfig = field(default_factory=SSLConfig)
    system: SystemConfig = field(default_factory=SystemConfig)
    run_name: str = "experiment"

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> ExperimentConfig:
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML configuration file
            
        Returns:
            ExperimentConfig instance
        """
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> ExperimentConfig:
        """
        Create ExperimentConfig from dictionary.
        
        Args:
            config_dict: Configuration dictionary
            
        Returns:
            ExperimentConfig instance
        """
        dataset_cfg = DatasetConfig(**config_dict.get('dataset', {}))
        model_cfg = ModelConfig(**config_dict.get('model', {}))
        training_cfg = TrainingConfig(**config_dict.get('training', {}))
        ssl_cfg = SSLConfig(**config_dict.get('ssl', {}))
        system_cfg = SystemConfig(**config_dict.get('system', {}))
        run_name = config_dict.get('run_name', 'experiment')
        
        return cls(
            dataset=dataset_cfg,
            model=model_cfg,
            training=training_cfg,
            ssl=ssl_cfg,
            system=system_cfg,
            run_name=run_name
        )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'dataset': {
                'name': self.dataset.name,
                'data_dir': self.dataset.data_dir,
                'num_classes': self.dataset.num_classes,
            },
            'model': {
                'architecture': self.model.architecture,
                'dropout_rate': self.model.dropout_rate,
                'widen_factor': self.model.widen_factor,
                'depth': self.model.depth,
            },
            'training': {
                'epochs': self.training.epochs,
                'batch_size': self.training.batch_size,
                'learning_rate': self.training.learning_rate,
                'momentum': self.training.momentum,
                'weight_decay': self.training.weight_decay,
                'warmup_epochs': self.training.warmup_epochs,
                'validation_split': self.training.validation_split,
                'stop_loss_threshold': self.training.stop_loss_threshold,
                'stop_loss_warmup_epochs': self.training.stop_loss_warmup_epochs,
                'supervised_algorithm': self.training.supervised_algorithm,
                'mixup_alpha': self.training.mixup_alpha,
                'cutmix_alpha': self.training.cutmix_alpha,
                'cutmix_prob': self.training.cutmix_prob,
                'save_best': self.training.save_best,
                'save_last': self.training.save_last,
            },
            'ssl': {
                'enabled': self.ssl.enabled,
                'algorithm': self.ssl.algorithm,
                'labels_per_class': self.ssl.labels_per_class,
                'label_fraction': self.ssl.label_fraction,
                'unlabeled_batch_size': self.ssl.unlabeled_batch_size,
                'confidence_threshold': self.ssl.confidence_threshold,
                'unlabeled_loss_weight': self.ssl.unlabeled_loss_weight,
                'strong_augment': self.ssl.strong_augment,
                'temperature': self.ssl.temperature,
                'mixmatch_alpha': self.ssl.mixmatch_alpha,
                'mixmatch_temperature': self.ssl.mixmatch_temperature,
                'seed': self.ssl.seed,
            },
            'system': {
                'seed': self.system.seed,
                'num_workers': self.system.num_workers,
                'device': self.system.device,
                'checkpoint_dir': self.system.checkpoint_dir,
                'log_dir': self.system.log_dir,
                'experiment_dir': self.system.experiment_dir,
            },
            'run_name': self.run_name,
        }
    
    def save_yaml(self, yaml_path: str | Path):
        """Save configuration to YAML file."""
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(yaml_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_config(yaml_path: str | Path) -> ExperimentConfig:
    """
    Convenience function to load configuration from YAML.
    
    Args:
        yaml_path: Path to YAML configuration file
        
    Returns:
        ExperimentConfig instance
    """
    return ExperimentConfig.from_yaml(yaml_path)
