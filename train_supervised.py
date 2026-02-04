"""
Main script for supervised learning baseline on CIFAR-10.
This is Sprint 1: Basic supervised training loop.
"""

import torch
import argparse
from src.data import get_cifar10_loaders
from src.models import ResNet18, WideResNet
from src.training import SupervisedTrainer
from src.utils import set_seed, count_parameters, get_device


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
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    print("\n" + "="*50)
    print("SUPERVISED LEARNING - SPRINT 1")
    print("="*50)
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Model: {args.model}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print("="*50 + "\n")
    
    # Load data
    print("Loading data...")
    if args.dataset == 'cifar10':
        train_loader, test_loader = get_cifar10_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    else:
        from src.data import get_cifar100_loaders
        train_loader, test_loader = get_cifar100_loaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers
        )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Create model
    print("Creating model...")
    if args.model == 'resnet18':
        model = ResNet18(num_classes=args.num_classes)
    elif args.model == 'resnet34':
        from src.models import ResNet34
        model = ResNet34(num_classes=args.num_classes)
    elif args.model == 'wideresnet':
        model = WideResNet(depth=28, widen_factor=2, num_classes=args.num_classes)
    
    print(f"Model: {args.model}")
    print(f"Parameters: {count_parameters(model):,}\n")
    
    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Train
    history = trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Best Test Accuracy: {max(history['test_acc']):.2f}%")
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print("="*50)


if __name__ == '__main__':
    main()
