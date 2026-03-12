# Quick Reference Card

## 🚀 Common Commands

### Verification
```bash
python verify_installation.py                    # Check all imports
```

### Quick Tests (2 epochs each)
```bash
python test_supervised.py                        # Test supervised pipeline
python test_ssl.py                               # Test SSL pipeline
```

### Training
```bash
# Full supervised (50K labels)
python train_supervised_yaml.py --config configs/supervised_cifar10.yaml

# Limited supervised (2.5K labels)
python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml

# SSL Pseudo-Label (2.5K labels + 47.5K unlabeled)
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```

### Comparison
```bash
# Use existing checkpoints
python compare_methods.py \
    --supervised-config configs/supervised_limited_cifar10.yaml \
    --ssl-config configs/ssl_pseudolabel_cifar10.yaml

# Force retrain
python compare_methods.py \
    --supervised-config configs/supervised_limited_cifar10.yaml \
    --ssl-config configs/ssl_pseudolabel_cifar10.yaml \
    --force-train
```

### Windows Helper Script
```bash
experiment_helper.bat verify              # Verify installation
experiment_helper.bat test                # Quick tests
experiment_helper.bat train-supervised    # Train supervised (limited)
experiment_helper.bat train-ssl           # Train SSL
experiment_helper.bat compare             # Compare methods
experiment_helper.bat full-experiment     # Complete workflow
```

---

## 📁 Important Files

### Configurations
- `configs/supervised_cifar10.yaml` - Full supervised (50K labels)
- `configs/supervised_limited_cifar10.yaml` - Limited (2.5K labels)
- `configs/ssl_pseudolabel_cifar10.yaml` - SSL (2.5K + unlabeled)

### Training Scripts
- `train_supervised_yaml.py` - Supervised with YAML config
- `train_supervised_limited.py` - Supervised with limited labels
- `train_ssl.py` - SSL Pseudo-Label training
- `compare_methods.py` - Compare SSL vs Supervised

### Testing
- `test_supervised.py` - Quick supervised test (2 epochs)
- `test_ssl.py` - Quick SSL test (2 epochs)
- `verify_installation.py` - Check installation

### Documentation
- `README_NEW.md` - Full documentation
- `USAGE_GUIDE.md` - Quick start guide
- `PROJECT_COMPLETE.md` - Implementation summary

---

## ⚙️ Key Configuration Parameters

### SSL Settings
```yaml
ssl:
  labels_per_class: 250              # Labels per class (2500 total)
  confidence_threshold: 0.95         # Pseudo-label threshold
  unlabeled_loss_weight: 1.0         # Weight for unlabeled loss
  unlabeled_batch_size: 256          # Unlabeled batch size
  strong_augment: true               # Use strong augmentation
```

### Training Settings
```yaml
training:
  epochs: 300                        # Number of epochs
  batch_size: 64                     # Labeled batch size
  learning_rate: 0.03                # Initial learning rate
  momentum: 0.9                      # SGD momentum
  weight_decay: 0.0005               # L2 regularization
```

### Model Settings
```yaml
model:
  architecture: wideresnet           # resnet18, resnet34, wideresnet
  depth: 28                          # WideResNet depth
  widen_factor: 2                    # WideResNet width
```

---

## 📊 Output Locations

```
checkpoints/
  <run_name>/
    <run_name>_best.pt              # Best model
    <run_name>_last.pt              # Last epoch
    <run_name>_config.yaml          # Saved config

logs/
  <run_name>/
    <run_name>_TIMESTAMP.log        # Training logs

experiments/
  comparison/
    TIMESTAMP/
      comparison_results.json       # Numerical results
      comparison_test_acc.png       # Accuracy plot
      comparison_train_loss.png     # Loss plot
      logs/                         # Comparison logs
```

---

## 🎯 Expected Results

| Method | Labels | Accuracy |
|--------|--------|----------|
| Supervised (Full) | 50,000 | ~94-95% |
| Supervised (Limited) | 2,500 | ~70-75% |
| SSL Pseudo-Label | 2,500 | ~85-88% |

**Label Efficiency:** SSL gets ~90% of full performance with 5% of labels!

---

## 🔧 Resume Training

```bash
# Resume supervised
python train_supervised_limited.py \
    --config configs/supervised_limited_cifar10.yaml \
    --resume checkpoints/supervised_limited_cifar10/supervised_limited_cifar10_last.pt

# Resume SSL
python train_ssl.py \
    --config configs/ssl_pseudolabel_cifar10.yaml \
    --resume checkpoints/ssl_pseudolabel_cifar10/ssl_pseudolabel_cifar10_last.pt
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
Reduce batch sizes in config:
```yaml
training:
  batch_size: 32          # Reduce from 64
ssl:
  unlabeled_batch_size: 128  # Reduce from 256
```

### Slow Training
- Ensure using GPU: `--device cuda`
- Reduce `num_workers`:
  ```yaml
  system:
    num_workers: 0  # Try 0, 1, or 2
  ```

### Poor SSL Performance
Adjust hyperparameters:
```yaml
ssl:
  confidence_threshold: 0.9   # Lower threshold
  unlabeled_loss_weight: 2.0  # More weight on unlabeled
```

---

## 📚 Quick Links

- **Full Docs:** [README_NEW.md](README_NEW.md)
- **Usage Guide:** [USAGE_GUIDE.md](USAGE_GUIDE.md)
- **Implementation:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)
- **Project Summary:** [PROJECT_COMPLETE.md](PROJECT_COMPLETE.md)

---

## ✅ Typical Workflow

1. **Verify:** `python verify_installation.py`
2. **Test:** `python test_ssl.py`
3. **Train SSL:** `python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml`
4. **Train Supervised:** `python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml`
5. **Compare:** `python compare_methods.py`
6. **Analyze:** Check `experiments/comparison/TIMESTAMP/`

---

## 💡 Tips

- Use `--device cpu` for testing without GPU
- Start with quick tests before full training
- Compare methods uses existing checkpoints automatically
- Save configs with checkpoints for reproducibility
- Check logs for detailed training progress

---

**Ready to run SSL experiments! 🚀**
