"""
Comparison script: SSL vs Supervised Learning.
Runs experiments and compares performance, using checkpoints if available.
"""

import argparse
import os
import json
from datetime import datetime

import torch
import matplotlib.pyplot as plt

from src.data import get_cifar_supervised_loaders, get_cifar_ssl_loaders
from src.data.cifar import SplitConfig
from src.models import build_model
from src.training import SupervisedTrainer, PseudoLabelTrainer, FixMatchTrainer
from src.utils import set_seed, get_logger, load_config, plot_per_class_comparison


def plot_comparison(supervised_history, ssl_history, save_dir, metric='test_acc'):
    """Plot comparison of supervised vs SSL training."""
    plt.figure(figsize=(10, 6))
    
    if metric in supervised_history and metric in ssl_history:
        plt.plot(supervised_history[metric], label='Supervised', linewidth=2)
        plt.plot(ssl_history[metric], label='SSL', linewidth=2)
        plt.xlabel('Epoch')
        plt.ylabel(metric.replace('_', ' ').title())
        plt.title(f'{metric.replace("_", " ").title()} Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(save_dir, f'comparison_{metric}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved plot: {save_path}")


def plot_per_class_results(supervised_history, ssl_history, save_dir):
    """Plot per-class comparison using final-epoch per-class metrics."""
    if not supervised_history.get('per_class_acc') or not ssl_history.get('per_class_acc'):
        return

    supervised_per_class = supervised_history['per_class_acc'][-1]
    ssl_per_class = ssl_history['per_class_acc'][-1]
    save_path = os.path.join(save_dir, 'comparison_per_class_acc.png')
    plot_per_class_comparison(
        supervised_per_class=supervised_per_class,
        ssl_per_class=ssl_per_class,
        save_path=save_path,
    )
    print(f"Saved plot: {save_path}")


def save_comparison_results(results, save_path):
    """Save comparison results to JSON."""
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Saved comparison results: {save_path}")


def train_or_load_supervised(config, device, logger, checkpoint_dir, force_train=False):
    """Train supervised model or load from checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f"{config.run_name}_best.pt")
    
    # Build model
    model = build_model(config.model, num_classes=config.dataset.num_classes)
    
    # Load data
    train_loader, test_loader = get_cifar_supervised_loaders(
        dataset=config.dataset.name,
        data_dir=config.dataset.data_dir,
        batch_size=config.training.batch_size,
        num_workers=config.system.num_workers,
        augment=True,
        logger=logger,
    )
    
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
        run_name=config.run_name,
        logger=logger,
        save_best=config.training.save_best,
        save_last=config.training.save_last,
        num_classes=config.dataset.num_classes,
    )
    
    # Load or train
    if os.path.exists(checkpoint_path) and not force_train:
        logger.info(f"Loading supervised model from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
        history = trainer.history
    else:
        logger.info("Training supervised model from scratch...")
        history = trainer.train(num_epochs=config.training.epochs)
    
    return trainer, history


def train_or_load_ssl(config, device, logger, checkpoint_dir, force_train=False):
    """Train SSL model or load from checkpoint."""
    checkpoint_path = os.path.join(checkpoint_dir, f"{config.run_name}_best.pt")
    
    # Build model
    model = build_model(config.model, num_classes=config.dataset.num_classes)
    
    # Create split configuration for SSL
    split_cfg = SplitConfig(
        labels_per_class=config.ssl.labels_per_class,
        label_ratio=config.ssl.label_fraction,
        seed=config.ssl.seed,
    )
    
    # Load data
    loader_outputs = get_cifar_ssl_loaders(
        dataset=config.dataset.name,
        data_dir=config.dataset.data_dir,
        batch_size=config.training.batch_size,
        unlabeled_batch_size=config.ssl.unlabeled_batch_size,
        num_workers=config.system.num_workers,
        split_cfg=split_cfg,
        val_split=getattr(config.training, 'validation_split', 0.0),
        strong_augment=config.ssl.strong_augment,
        logger=logger,
    )

    if len(loader_outputs) == 4:
        labeled_loader, unlabeled_loader, val_loader, test_loader = loader_outputs
    else:
        labeled_loader, unlabeled_loader, test_loader = loader_outputs
        val_loader = None
    
    # Create trainer
    algorithm = config.ssl.algorithm.lower()
    if algorithm == 'fixmatch':
        trainer = FixMatchTrainer(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
            confidence_threshold=config.ssl.confidence_threshold,
            unlabeled_loss_weight=config.ssl.unlabeled_loss_weight,
            checkpoint_dir=checkpoint_dir,
            run_name=config.run_name,
            logger=logger,
            save_best=config.training.save_best,
            save_last=config.training.save_last,
            num_classes=config.dataset.num_classes,
        )
    else:
        trainer = PseudoLabelTrainer(
            model=model,
            labeled_loader=labeled_loader,
            unlabeled_loader=unlabeled_loader,
            test_loader=test_loader,
            val_loader=val_loader,
            device=device,
            learning_rate=config.training.learning_rate,
            momentum=config.training.momentum,
            weight_decay=config.training.weight_decay,
            confidence_threshold=config.ssl.confidence_threshold,
            unlabeled_loss_weight=config.ssl.unlabeled_loss_weight,
            temperature=config.ssl.temperature,
            checkpoint_dir=checkpoint_dir,
            run_name=config.run_name,
            logger=logger,
            save_best=config.training.save_best,
            save_last=config.training.save_last,
            num_classes=config.dataset.num_classes,
        )
    
    # Load or train
    if os.path.exists(checkpoint_path) and not force_train:
        logger.info(f"Loading SSL model from checkpoint: {checkpoint_path}")
        trainer.load_checkpoint(checkpoint_path, load_optimizer=False)
        history = trainer.history
    else:
        logger.info("Training SSL model from scratch...")
        history = trainer.train(num_epochs=config.training.epochs)
    
    return trainer, history


def main():
    parser = argparse.ArgumentParser(description='Compare SSL vs Supervised Learning')
    parser.add_argument('--supervised-config', type=str, default='configs/supervised_cifar10.yaml',
                        help='Path to supervised configuration file')
    parser.add_argument('--ssl-config', type=str, default='configs/ssl_pseudolabel_cifar10.yaml',
                        help='Path to SSL configuration file')
    parser.add_argument('--force-train', action='store_true',
                        help='Force training even if checkpoints exist')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--output-dir', type=str, default='experiments/comparison',
                        help='Directory to save comparison results')
    args = parser.parse_args()
    
    # Setup
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logger
    logger = get_logger(
        name='comparison',
        log_dir=output_dir,
        level=20  # INFO
    )
    
    logger.info("="*60)
    logger.info("SSL vs Supervised Learning Comparison")
    logger.info("="*60)
    logger.info(f"Supervised config: {args.supervised_config}")
    logger.info(f"SSL config: {args.ssl_config}")
    logger.info(f"Device: {device}")
    logger.info(f"Force train: {args.force_train}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)
    
    # Load configurations
    supervised_config = load_config(args.supervised_config)
    ssl_config = load_config(args.ssl_config)
    
    # Set consistent seed
    set_seed(42)
    
    # Train/load supervised model
    logger.info("\n" + "="*60)
    logger.info("SUPERVISED LEARNING")
    logger.info("="*60)
    supervised_checkpoint_dir = os.path.join(
        supervised_config.system.checkpoint_dir,
        supervised_config.run_name
    )
    supervised_trainer, supervised_history = train_or_load_supervised(
        supervised_config,
        device,
        logger,
        supervised_checkpoint_dir,
        force_train=args.force_train
    )
    
    # Train/load SSL model
    logger.info("\n" + "="*60)
    logger.info("SEMI-SUPERVISED LEARNING (Pseudo-Label)")
    logger.info("="*60)
    ssl_checkpoint_dir = os.path.join(
        ssl_config.system.checkpoint_dir,
        ssl_config.run_name
    )
    ssl_trainer, ssl_history = train_or_load_ssl(
        ssl_config,
        device,
        logger,
        ssl_checkpoint_dir,
        force_train=args.force_train
    )
    
    # Compare results
    logger.info("\n" + "="*60)
    logger.info("COMPARISON RESULTS")
    logger.info("="*60)
    
    results = {
        'timestamp': timestamp,
        'supervised': {
            'config': args.supervised_config,
            'best_test_acc': supervised_trainer.best_acc,
            'final_test_acc': supervised_history['test_acc'][-1] if supervised_history['test_acc'] else 0,
            'num_epochs': len(supervised_history['test_acc']),
            'total_time': sum(supervised_history['epoch_time']) if 'epoch_time' in supervised_history else 0,
        },
        'ssl': {
            'config': args.ssl_config,
            'best_test_acc': ssl_trainer.best_acc,
            'final_test_acc': ssl_history['test_acc'][-1] if ssl_history['test_acc'] else 0,
            'num_epochs': len(ssl_history['test_acc']),
            'total_time': sum(ssl_history['epoch_time']) if 'epoch_time' in ssl_history else 0,
            'labels_per_class': ssl_config.ssl.labels_per_class,
        }
    }
    
    logger.info(f"Supervised - Best Acc: {results['supervised']['best_test_acc']:.2f}%")
    logger.info(f"Supervised - Final Acc: {results['supervised']['final_test_acc']:.2f}%")
    logger.info(f"Supervised - Training Time: {results['supervised']['total_time']:.1f}s")
    logger.info("")
    logger.info(f"SSL - Best Acc: {results['ssl']['best_test_acc']:.2f}%")
    logger.info(f"SSL - Final Acc: {results['ssl']['final_test_acc']:.2f}%")
    logger.info(f"SSL - Training Time: {results['ssl']['total_time']:.1f}s")
    logger.info(f"SSL - Labels per class: {results['ssl']['labels_per_class']}")
    logger.info("")
    
    accuracy_gap = results['supervised']['best_test_acc'] - results['ssl']['best_test_acc']
    logger.info(f"Accuracy Gap: {accuracy_gap:.2f}%")
    
    if accuracy_gap > 0:
        logger.info("Supervised model performs better (expected with full labels)")
    else:
        logger.info("SSL model performs comparably or better!")
    
    # Save results
    results_path = os.path.join(output_dir, 'comparison_results.json')
    save_comparison_results(results, results_path)
    
    # Plot comparisons
    logger.info("\nGenerating comparison plots...")
    plot_comparison(supervised_history, ssl_history, output_dir, 'test_acc')
    plot_comparison(supervised_history, ssl_history, output_dir, 'train_loss')
    plot_comparison(supervised_history, ssl_history, output_dir, 'test_loss')
    plot_per_class_results(supervised_history, ssl_history, output_dir)
    
    logger.info("="*60)
    logger.info("Comparison complete!")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
