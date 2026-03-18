"""
Supervised training script with YAML configuration support.
Supports both YAML config files and command-line arguments for backwards compatibility.
"""

import argparse
import os

import torch
from src.data import get_cifar_supervised_loaders
from src.models import build_model
from src.training import SupervisedTrainer
from src.utils import set_seed, get_logger, load_config, plot_training_curves


def main():
    parser = argparse.ArgumentParser(description='Supervised Learning Training')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML configuration file (overrides other args)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Learning rate')
    args = parser.parse_args()
    
    # Load or create configuration
    if args.config:
        print(f"Loading configuration from {args.config}")
        config = load_config(args.config)
        # Override with CLI args if provided
        if args.device:
            config.system.device = args.device
    else:
        print("Using default configuration (configs/supervised_cifar10.yaml)")
        # Try to load default config, fallback to creating one
        try:
            config = load_config('configs/supervised_cifar10.yaml')
        except FileNotFoundError:
            print("Default config not found, using hardcoded defaults")
            from src.utils.config import ExperimentConfig
            config = ExperimentConfig()
            config.training.epochs = args.epochs
            config.training.batch_size = args.batch_size
            config.training.learning_rate = args.lr
    
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
    logger.info("Supervised Training")
    logger.info("="*60)
    logger.info(f"Configuration: {args.config if args.config else 'default'}")
    logger.info(f"Run name: {run_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Random seed: {config.system.seed}")
    logger.info("="*60)
    
    # Data loaders
    logger.info("Loading data...")
    train_loader, test_loader = get_cifar_supervised_loaders(
        dataset=config.dataset.name,
        data_dir=config.dataset.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.system.num_workers,
        augment=True,
        logger=logger,
    )
    
    # Build model
    logger.info("Building model...")
    model = build_model(config.model, num_classes=config.dataset.num_classes)
    logger.info(f"Model: {config.model.architecture}")
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=config.training.learning_rate,
        momentum=config.training.momentum,
        weight_decay=config.training.weight_decay,
        checkpoint_dir=checkpoint_dir,
        run_name=run_name,
        logger=logger,
        save_best=config.training.save_best,
        save_last=config.training.save_last,
        stop_loss_threshold=config.training.stop_loss_threshold,
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
