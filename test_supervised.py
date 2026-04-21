"""
Quick test script to verify the supervised training setup works.
Runs for just 2 epochs to quickly validate the pipeline.
"""

import argparse

import torch
from src.data import get_cifar10_loaders
from src.models import ResNet18
from src.training import SupervisedTrainer
from src.utils import set_seed, get_device


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Quick supervised training smoke test')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs to run')
    parser.add_argument('--algorithm', type=str, default='standard', choices=['standard', 'mixup', 'cutmix'])
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--cutmix-alpha', type=float, default=1.0)
    parser.add_argument('--cutmix-prob', type=float, default=0.5)
    return parser.parse_args()

def quick_test():
    """Run a quick test to verify everything works."""
    args = parse_args()
    print("="*60)
    print("QUICK TEST - Supervised Training Pipeline")
    print("="*60)
    
    # Setup
    set_seed(42)
    device = get_device()
    
    # Small batch for quick testing
    print("\nLoading CIFAR-10...")
    train_loader, test_loader = get_cifar10_loaders(
        data_dir='./data',
        batch_size=128,
        num_workers=0  # 0 for Windows compatibility
    )
    
    print(f"Training batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create small model
    print("\nCreating ResNet-18...")
    model = ResNet18(num_classes=10)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.1,
        supervised_algorithm=args.algorithm,
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_prob=args.cutmix_prob,
    )
    
    # Quick train for 2 epochs
    print(f"\nTraining for {args.epochs} epochs (quick validation)...\n")
    history = trainer.train(num_epochs=args.epochs)
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY! ✅")
    print("="*60)
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print("\nThe supervised training pipeline is working correctly.")
    print("="*60)

if __name__ == '__main__':
    quick_test()
