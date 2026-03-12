"""
Supervised training with limited labels (for fair SSL comparison).
Uses only a subset of labeled data to match SSL experiments.
"""

import argparse
import os

import torch
from src.data import get_cifar_ssl_loaders
from src.data.cifar import SplitConfig
from src.models import build_model
from src.training import SupervisedTrainer
from src.utils import set_seed, get_logger, load_config, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description='Supervised Learning with Limited Labels')
    parser.add_argument('--config', type=str, default='configs/supervised_limited_cifar10.yaml',
                        help='Path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of epochs (overrides config)')
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with command-line arguments
    if args.device:
        config.system.device = args.device
    if args.epochs:
        config.training.epochs = args.epochs
    
    # Set device
    device = config.system.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Set random seed
    set_seed(config.system.seed)
    
    # Setup logging
    run_name = config.run_name
    log_dir = os.path.join(config.system.log_dir, run_name)
    checkpoint_dir = os.path.join(config.system.checkpoint_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger = get_logger(
        name=run_name,
        log_dir=log_dir,
        level=20  # INFO
    )
    
    logger.info("="*60)
    logger.info("Supervised Training with Limited Labels")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {config.system.seed}")
    logger.info("="*60)
    
    # Data loaders - use SSL loader but only labeled data
    logger.info("Loading data with limited labels...")
    
    # Create split configuration
    split_cfg = SplitConfig(
        labels_per_class=config.ssl.labels_per_class if hasattr(config.ssl, 'labels_per_class') else None,
        label_ratio=config.ssl.label_fraction if hasattr(config.ssl, 'label_fraction') else None,
        seed=config.ssl.seed,
    )
    
    # Get SSL loaders (we'll only use labeled/val/test)
    loader_outputs = get_cifar_ssl_loaders(
        dataset=config.dataset.name,
        data_dir=config.dataset.data_dir,
        batch_size=config.training.batch_size,
        unlabeled_batch_size=config.training.batch_size,
        num_workers=config.system.num_workers,
        split_cfg=split_cfg,
        val_split=getattr(config.training, 'validation_split', 0.0),
        strong_augment=False,  # Standard augmentation for supervised
        logger=logger,
    )

    if len(loader_outputs) == 4:
        labeled_loader, _, val_loader, test_loader = loader_outputs
    else:
        labeled_loader, _, test_loader = loader_outputs
        val_loader = None
    
    # Build model
    logger.info("Building model...")
    model = build_model(config.model, num_classes=config.dataset.num_classes)
    logger.info(f"Model: {config.model.architecture}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer (use labeled data as train data)
    trainer = SupervisedTrainer(
        model=model,
        train_loader=labeled_loader,  # Only labeled data
        test_loader=test_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        logger=logger,
        save_best=config.training.save_best,
        save_last=config.training.save_last,
        num_classes=config.dataset.num_classes,
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            start_epoch = trainer.load_checkpoint(args.resume, load_optimizer=True)
        else:
            logger.warning(f"Checkpoint not found: {args.resume}, starting from scratch")
    
    # Save configuration
    config_save_path = os.path.join(checkpoint_dir, f"{run_name}_config.yaml")
    config.save_yaml(config_save_path)
    logger.info(f"Saved configuration to {config_save_path}")
    
    # Train
    history = trainer.train(num_epochs=config.training.epochs, start_epoch=start_epoch)

    # Save training visualizations
    plot_training_curves(history, save_dir=log_dir, run_name=run_name)
    
    logger.info("Training complete!")
    logger.info(f"Best test accuracy: {trainer.best_acc:.2f}%")
    logger.info(f"Checkpoints saved to: {checkpoint_dir}")
    logger.info(f"Logs saved to: {log_dir}")


if __name__ == '__main__':
    main()
