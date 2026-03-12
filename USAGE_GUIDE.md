# SSL vs Supervised Learning - Quick Start Guide

## Overview

This project implements and compares Semi-Supervised Learning (SSL) with Pseudo-Label algorithm against standard Supervised Learning for image classification on CIFAR-10.

## Quick Start Commands

### 1. Test the Pipeline (Quick Validation)

```bash
# Test supervised pipeline (2 epochs)
python test_supervised.py

# Test SSL pipeline (2 epochs)  
python test_ssl.py
```

### 2. Train Supervised Baseline (Full Labels)

Train with all 50,000 labeled images:

```bash
python train_supervised_yaml.py --config configs/supervised_cifar10.yaml
```

### 3. Train SSL with Pseudo-Label

Train with only 2,500 labeled images (5% of data):

```bash
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```

### 4. Train Supervised with Limited Labels (Fair Comparison)

Train supervised model with same labels as SSL (2,500 images):

```bash
python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml
```

### 5. Compare All Methods

Run complete comparison (uses existing checkpoints if available):

```bash
python compare_methods.py \
    --supervised-config configs/supervised_limited_cifar10.yaml \
    --ssl-config configs/ssl_pseudolabel_cifar10.yaml \
    --output-dir experiments/comparison
```

Force retraining:
```bash
python compare_methods.py \
    --supervised-config configs/supervised_limited_cifar10.yaml \
    --ssl-config configs/ssl_pseudolabel_cifar10.yaml \
    --force-train
```

## Configuration Files

### Available Configs

1. **supervised_cifar10.yaml** - Full supervised baseline (50K labels)
2. **supervised_limited_cifar10.yaml** - Supervised with 2.5K labels
3. **ssl_pseudolabel_cifar10.yaml** - SSL Pseudo-Label with 2.5K labels

### Key Parameters

#### SSL Configuration
```yaml
ssl:
  labels_per_class: 250              # 250 × 10 classes = 2,500 labels
  confidence_threshold: 0.95         # Minimum confidence for pseudo-labels
  unlabeled_loss_weight: 1.0         # Weight for unlabeled data loss
  unlabeled_batch_size: 256          # Batch size for unlabeled data
  strong_augment: true               # Use RandAugment
```

#### Training Configuration
```yaml
training:
  epochs: 300                        # Number of training epochs
  batch_size: 64                     # Labeled batch size
  learning_rate: 0.03                # Initial learning rate
  momentum: 0.9                      # SGD momentum
  weight_decay: 0.0005               # L2 regularization
```

## Expected Results

### Performance Comparison

| Method | Labels Used | Expected Accuracy | Training Time* |
|--------|-------------|-------------------|----------------|
| Supervised (Full) | 50,000 | ~94-95% | ~3-4 hours |
| Supervised (Limited) | 2,500 | ~70-75% | ~1-2 hours |
| SSL Pseudo-Label | 2,500 | ~85-88% | ~2-3 hours |

*Approximate times on NVIDIA RTX 3080

### Label Efficiency

SSL achieves **~90% of full supervised performance** while using only **5% of labels**!

## Output Structure

```
checkpoints/
├── supervised_cifar10/
│   ├── supervised_cifar10_best.pt
│   ├── supervised_cifar10_last.pt
│   └── supervised_cifar10_config.yaml
├── supervised_limited_cifar10/
│   └── ...
└── ssl_pseudolabel_cifar10/
    └── ...

logs/
├── supervised_cifar10/
│   └── supervised_cifar10_TIMESTAMP.log
├── supervised_limited_cifar10/
│   └── ...
└── ssl_pseudolabel_cifar10/
    └── ...

experiments/
└── comparison/
    └── TIMESTAMP/
        ├── comparison_results.json
        ├── comparison_test_acc.png
        ├── comparison_train_loss.png
        └── logs/
```

## Resuming Training

Resume from a checkpoint:

```bash
# Supervised
python train_supervised_yaml.py \
    --config configs/supervised_cifar10.yaml \
    --resume checkpoints/supervised_cifar10/supervised_cifar10_last.pt

# SSL
python train_ssl.py \
    --config configs/ssl_pseudolabel_cifar10.yaml \
    --resume checkpoints/ssl_pseudolabel_cifar10/ssl_pseudolabel_cifar10_last.pt
```

## Modifying Configurations

### Change Label Amount

Edit `configs/ssl_pseudolabel_cifar10.yaml`:

```yaml
ssl:
  labels_per_class: 100  # 1,000 total labels (2% of data)
  # OR
  label_fraction: 0.1    # 10% of data (~5,000 labels)
```

### Change Model Architecture

```yaml
model:
  architecture: resnet18  # Options: resnet18, resnet34, wideresnet
  # WideResNet specific:
  depth: 28
  widen_factor: 2
```

### Adjust SSL Hyperparameters

```yaml
ssl:
  confidence_threshold: 0.9    # Lower = more pseudo-labels (less confident)
  unlabeled_loss_weight: 2.0   # Higher = more weight on unlabeled data
  temperature: 0.5             # Lower = sharper pseudo-label distribution
```

## Metrics Tracked

All experiments track:

- **Test Accuracy** - Overall accuracy on test set
- **Per-class Accuracy** - Accuracy for each of 10 classes
- **Training Loss** - Cross-entropy loss on labeled data
- **Test Loss** - Loss on test set
- **Pseudo-label Ratio** (SSL only) - % of unlabeled data used
- **Training Time** - Time per epoch and total

## Troubleshooting

### CUDA Out of Memory

Reduce batch sizes:
```yaml
training:
  batch_size: 32  # Reduce from 64
ssl:
  unlabeled_batch_size: 128  # Reduce from 256
```

### Slow Training

- Use GPU: `--device cuda`
- Reduce `num_workers` if CPU bottleneck:
  ```yaml
  system:
    num_workers: 2  # Try 0, 1, or 2
  ```

### Poor SSL Performance

Try adjusting:
- `confidence_threshold`: Lower to 0.9 or 0.85
- `unlabeled_loss_weight`: Try 0.5 or 2.0
- `learning_rate`: Try 0.05 or 0.02
- Ensure `strong_augment: true`

## Command-Line Overrides

Override config settings via CLI:

```bash
# Change device
python train_ssl.py --config CONFIG.yaml --device cpu

# Change epochs
python train_ssl.py --config CONFIG.yaml --epochs 100

# Both
python train_ssl.py --config CONFIG.yaml --device cuda --epochs 200
```

## Next Steps

1. Run `python test_ssl.py` to verify installation
2. Train supervised baseline with limited labels
3. Train SSL model with same label amount
4. Compare results using `compare_methods.py`
5. Experiment with different hyperparameters
6. Try different label fractions (1%, 10%, etc.)

## References

- Lee (2013): Pseudo-Label algorithm paper
- configs/ directory: All configuration examples
- README_NEW.md: Full project documentation
