"""
Main script for supervised learning baseline on CIFAR-10.
This is Sprint 1: Basic supervised training loop.
"""

import argparse
from datetime import datetime

from src.data import get_cifar10_loaders
from src.models import ResNet18, WideResNet
from src.training import SupervisedTrainer
from src.utils import set_seed, count_parameters, get_device, get_logger, plot_training_curves


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Supervised Learning on CIFAR-10')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='cifar10', 
                        choices=['cifar10', 'cifar100'],
                        help='Dataset to use')
    parser.add_argument('--data-dir', type=str, default='./data',
                        help='Directory to store datasets')
    
    # Model
    parser.add_argument('--model', type=str, default='resnet18',
                        choices=['resnet18', 'resnet34', 'wideresnet'],
                        help='Model architecture')
    parser.add_argument('--num-classes', type=int, default=10,
                        help='Number of classes')
    
    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1,
                        help='Initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        help='Weight decay')
    
    # System
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--num-workers', type=int, default=2,
                        help='Number of data loading workers')

    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--log-dir', type=str, default='./logs',
                        help='Directory to save training logs')
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)

    run_name = f"{args.dataset}_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    logger = get_logger(run_name, log_dir=args.log_dir)
    
    # Get device
    device = get_device()
    
    logger.info("==================================================")
    logger.info("SUPERVISED LEARNING - SPRINT 1")
    logger.info("==================================================")
    logger.info("Dataset: %s", args.dataset.upper())
    logger.info("Model: %s", args.model)
    logger.info("Epochs: %d", args.epochs)
    logger.info("Batch Size: %d", args.batch_size)
    logger.info("Learning Rate: %.4f", args.lr)
    logger.info("Run Name: %s", run_name)
    logger.info("==================================================")
    
    # Load data
    logger.info("Loading data...")
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            logger=logger,
        )
    else:
        from src.data import get_cifar100_loaders
        train_loader, test_loader = get_cifar100_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            logger=logger,
        )

    logger.info("Training batches: %d", len(train_loader))
    logger.info("Test batches: %d", len(test_loader))
    
    # Create model
    logger.info("Creating model...")
    if args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes)
    elif args.model == 'resnet34':
        from src.models import ResNet34
        model = ResNet34(num_classes=args.num_classes)
    elif args.model == 'wideresnet':
        model = WideResNet(depth=28, widen_factor=2, num_classes=args.num_classes)
    
    logger.info("Model: %s", args.model)
    logger.info("Parameters: %s", f"{count_parameters(model):,}")
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        run_name=run_name,
        logger=logger,
    )
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    plot_training_curves(history, save_dir=args.log_dir, run_name=run_name)
    
    logger.info("==================================================")
    logger.info("TRAINING COMPLETE")
    logger.info("==================================================")
    logger.info("Best Test Accuracy: %.2f%%", max(history['test_acc']))
    logger.info("Final Test Accuracy: %.2f%%", history['test_acc'][-1])
    logger.info("==================================================")


if __name__ == '__main__':
    main()
