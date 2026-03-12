"""
Quick test script to verify SSL training pipeline works.
Runs for just 2 epochs with limited data to validate the implementation.
"""

import torch
from src.data import get_cifar_ssl_loaders
from src.data.cifar import SplitConfig
from src.models import WideResNet
from src.training import PseudoLabelTrainer
from src.utils import set_seed, get_logger


def quick_ssl_test():
    """Run a quick test to verify SSL pipeline works."""
    print("="*60)
    print("QUICK TEST - SSL Pseudo-Label Training Pipeline")
    print("="*60)
    
    # Setup
    set_seed(42)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    # Create split config (very few labels for quick test)
    split_cfg = SplitConfig(
        labels_per_class=10,  # Only 10 labels per class = 100 total
        seed=42
    )
    
    # Load data
    print("\nLoading CIFAR-10 with SSL split...")
    labeled_loader, unlabeled_loader, test_loader = get_cifar_ssl_loaders(
        dataset='cifar10',
        data_dir='./data',
        batch_size=32,  # Smaller batch for quick test
        unlabeled_batch_size=64,
        num_workers=0,  # 0 for Windows compatibility
        split_cfg=split_cfg,
        strong_augment=True,
    )
    
    print(f"Labeled batches: {len(labeled_loader)}")
    print(f"Unlabeled batches: {len(unlabeled_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create model
    print("\nCreating WideResNet-28-2...")
    model = WideResNet(depth=28, widen_factor=2, num_classes=10, dropout_rate=0.0)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create logger
    logger = get_logger(name='ssl_test', log_dir=None, level=20)
    
    # Create trainer
    print("\nInitializing SSL Pseudo-Label trainer...")
    trainer = PseudoLabelTrainer(
        model=model,
        labeled_loader=labeled_loader,
        unlabeled_loader=unlabeled_loader,
        test_loader=test_loader,
        device=device,
        learning_rate=0.03,
        confidence_threshold=0.95,
        unlabeled_loss_weight=1.0,
        checkpoint_dir='checkpoints/test',
        run_name='ssl_test',
        logger=logger,
        save_best=False,
        save_last=False,
        num_classes=10,
    )
    
    # Quick train for 2 epochs
    print("\nTraining for 2 epochs (quick validation)...\n")
    history = trainer.train(num_epochs=2)
    
    print("\n" + "="*60)
    print("TEST COMPLETED SUCCESSFULLY! ✅")
    print("="*60)
    print(f"Final Test Accuracy: {history['test_acc'][-1]:.2f}%")
    print(f"Pseudo-label ratio: {history['pseudo_label_ratio'][-1]:.2%}")
    print("\nThe SSL Pseudo-Label training pipeline is working correctly.")
    print("="*60)


if __name__ == '__main__':
    quick_ssl_test()
