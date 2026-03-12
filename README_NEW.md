# Comparative Study of Semi-Supervised and Supervised Learning for Image Classification

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)

This repository implements and compares **Semi-Supervised Learning (SSL)** and **Supervised Learning** algorithms for image classification on CIFAR-10/100 datasets. Part of a Final Year Project investigating label efficiency in deep learning.

## 🎯 Project Overview

Modern computer vision relies on large labeled datasets, but labeling is expensive and time-consuming. This project explores **Semi-Supervised Learning** as a solution, leveraging small labeled datasets alongside large amounts of unlabeled data.

**Implemented Algorithms:**
- ✅ Supervised Baseline (Full labels)
- ✅ **Pseudo-Label** (SSL)
- ✅ **FixMatch (Initial Version)**
- 🚧 MixMatch (Planned)
- 🚧 FlexMatch (Planned)

## 📁 Project Structure

```
.
├── configs/                          # YAML configuration files
│   ├── supervised_cifar10.yaml       # Supervised learning config
│   └── ssl_pseudolabel_cifar10.yaml  # Pseudo-Label SSL config
├── src/
│   ├── data/                         # Dataset loaders
│   │   ├── cifar.py                  # CIFAR-10/100 loaders with SSL support
│   │   └── __init__.py
│   ├── models/                       # Neural network architectures
│   │   ├── resnet.py                 # ResNet-18/34
│   │   ├── wideresnet.py             # WideResNet-28-2
│   │   ├── builder.py                # Model builder utility
│   │   └── __init__.py
│   ├── training/                     # Training loops
│   │   ├── supervised.py             # Supervised trainer
│   │   ├── ssl_pseudolabel.py        # Pseudo-Label SSL trainer
│   │   └── __init__.py
│   └── utils/                        # Utilities
│       ├── config.py                 # YAML configuration loader
│       ├── helpers.py                # Helper functions
│       ├── logging_utils.py          # Logging utilities
│       └── __init__.py
├── train_supervised_yaml.py          # Supervised training script (YAML)
├── train_ssl.py                      # SSL training script
├── compare_methods.py                # Compare SSL vs Supervised
├── test_supervised.py                # Quick validation test
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (recommended) or CPU
- Docker (optional)

### Installation

#### Option 1: Local Python Environment

```bash
# Clone repository
git clone <repository-url>
cd FYP-Implementation

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option 2: Docker

```bash
# Build Docker image
docker build -t fyp-implementation -f docker/Dockerfile .

# Run with GPU
docker run -it --rm --gpus all -v ${PWD}:/workspace fyp-implementation

# Run with CPU only
docker run -it --rm -v ${PWD}:/workspace fyp-implementation
```

## 📊 Usage

### 1. Supervised Learning (Baseline)

Train a fully supervised model using all available labels:

```bash
# Using YAML configuration
python train_supervised_yaml.py --config configs/supervised_cifar10.yaml

# Using command-line arguments (legacy)
python train_supervised.py --epochs 200 --batch-size 128 --lr 0.1
```

**Key Features:**
- Full CIFAR-10 training set (50,000 labeled images)
- ResNet-18 or WideResNet-28-2 architecture
- Cosine annealing learning rate schedule
- Automatic checkpointing (best and last models)
- Per-class accuracy tracking

### 2. Semi-Supervised Learning (Pseudo-Label)

Train with limited labels using the Pseudo-Label algorithm:

```bash
# Train SSL model
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml

# Resume from checkpoint
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml --resume checkpoints/ssl_pseudolabel_cifar10/ssl_pseudolabel_cifar10_last.pt
```

### 2b. Semi-Supervised Learning (FixMatch - Initial)

```bash
# Train FixMatch from dedicated config
python train_fixmatch.py --config configs/ssl_fixmatch_cifar10.yaml

# Equivalent via generic SSL entrypoint
python train_ssl.py --config configs/ssl_fixmatch_cifar10.yaml
```

**Configuration Highlights:**
- **Labels:** 250 per class (2,500 total for CIFAR-10 = 5% of data)
- **Unlabeled data:** Remaining 47,500 images
- **Confidence threshold:** 0.95 (only use high-confidence pseudo-labels)
- **Strong augmentation:** RandAugment for unlabeled data

### 3. Compare SSL vs Supervised

Run both methods and generate comparison plots:

```bash
python compare_methods.py \
    --supervised-config configs/supervised_cifar10.yaml \
    --ssl-config configs/ssl_pseudolabel_cifar10.yaml \
    --output-dir experiments/comparison
```

**Features:**
- Automatically loads existing checkpoints (use `--force-train` to retrain)
- Generates comparison plots (accuracy curves, loss curves)
- Saves detailed comparison report (JSON)
- Calculates accuracy gap and label efficiency

**Output:**
```
experiments/comparison/YYYYMMDD_HHMMSS/
├── comparison_results.json      # Quantitative comparison
├── comparison_test_acc.png      # Accuracy plot
├── comparison_train_loss.png    # Training loss plot
├── comparison_test_loss.png     # Test loss plot
└── logs/                        # Execution logs
```

### 4. Quick Validation Test

Run a quick 2-epoch test to verify the pipeline:

```bash
python test_supervised.py
```

## ⚙️ Configuration

All experiments use YAML configuration files for reproducibility. Key parameters:

### Supervised Learning Config

```yaml
run_name: supervised_cifar10

dataset:
  name: cifar10
  num_classes: 10

model:
  architecture: resnet18

training:
  epochs: 200
  batch_size: 128
  learning_rate: 0.1

system:
  seed: 42
  device: cuda
```

### SSL Pseudo-Label Config

```yaml
run_name: ssl_pseudolabel_cifar10

ssl:
  enabled: true
  algorithm: pseudolabel
  labels_per_class: 250              # 2,500 total labels
  confidence_threshold: 0.95         # Only use high-confidence predictions
  unlabeled_loss_weight: 1.0         # Balance labeled/unlabeled loss
  unlabeled_batch_size: 256          # Larger batch for unlabeled data
  strong_augment: true               # Use RandAugment
```

## 📈 Metrics and Logging

Both trainers track comprehensive metrics:

- **Accuracy:** Overall and per-class test accuracy
- **Loss:** Training and test loss
- **Pseudo-label stats:** Confidence ratio, usage rate (SSL only)
- **Training time:** Per-epoch and total
- **Best model tracking:** Automatic checkpoint saving

Logs are saved to:
- `logs/<run_name>/` - Timestamped log files
- `checkpoints/<run_name>/` - Model checkpoints
- `experiments/` - Comparison results

## 🔬 Pseudo-Label Algorithm

Implementation based on Lee (2013): *"Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks"*

**Key Steps:**
1. Train on labeled data with cross-entropy loss
2. Generate predictions on unlabeled data (weak augmentation)
3. Filter predictions by confidence threshold (default: 0.95)
4. Use high-confidence predictions as pseudo-labels
5. Train on both labeled and pseudo-labeled data (strong augmentation)

**Advantages:**
- Simple and efficient
- No additional networks or adversarial training
- Works with any architecture
- Proven effectiveness on CIFAR benchmarks

## 🧪 Expected Results

Based on literature and preliminary experiments:

| Method | Labels Used | Expected Accuracy |
|--------|-------------|-------------------|
| Supervised | 50,000 (100%) | ~94-95% |
| Pseudo-Label | 2,500 (5%) | ~85-88% |

**Label Efficiency:** SSL achieves ~90% of supervised performance with only 5% of labels!

## 📚 References

1. Lee, D. H. (2013). "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks"
2. Berthelot et al. (2019). "MixMatch: A Holistic Approach to Semi-Supervised Learning"
3. Sohn et al. (2020). "FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence"
4. Zhang et al. (2021). "FlexMatch: Boosting Semi-Supervised Learning with Curriculum Pseudo Labeling"

## 🛠️ Development

### Running Tests

```bash
# Quick pipeline validation (2 epochs)
python test_supervised.py

# Full test with both methods
python compare_methods.py --force-train
```

### Docker Helper (Windows)

```bash
# Build image
docker_helper.bat build

# Quick test
docker_helper.bat test

# Full training (GPU)
docker_helper.bat train-gpu

# Interactive shell
docker_helper.bat shell
```

## 📝 TODOs

- [x] Supervised baseline implementation
- [x] Pseudo-Label SSL implementation
- [x] YAML configuration system
- [x] Comprehensive logging and metrics
- [x] Checkpoint loading/resuming
- [x] Comparison framework
- [ ] MixMatch implementation
- [ ] FixMatch implementation
- [ ] FlexMatch implementation
- [ ] Hyperparameter tuning
- [ ] Extended experiments (CIFAR-100, different label fractions)
- [ ] Visualization dashboard

## 👤 Author

**Milosz Momot**  
Final Year Project - BSc Software Development  
Munster Technological University Cork  
December 2025

## 📄 License

This project is for academic purposes as part of a Final Year Project submission.

---

**Note:** This is an active research project. Results and implementations may change as experiments progress.
