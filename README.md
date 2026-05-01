# Semi-Supervised Learning Benchmark: Comparative Study on CIFAR-10

Implements and compares multiple semi-supervised learning (SSL) algorithms against supervised baselines for image classification on CIFAR-10.

## Implemented Algorithms

**Supervised:**
- Standard supervised learning with limited labels

**Semi-Supervised:**
- **Pseudo-Label** — confidence-based pseudo-labeling (Lee, 2013)
- **FixMatch** — weak & strong augmentation with confidence thresholding
- **MixMatch** — mixing labeled & unlabeled samples
- **FlexMatch** — flexible threshold adjustment per class

## Quick Start

**Interactive Launcher (Windows):**
Double-click \launch_docker.cmd\

**Command Line:**
\\\ash
python launch_docker.py --dataset cifar10 --algorithm pseudolabel --split 250labels
\\\

**Docker Direct:**
\\\ash
docker run --rm --gpus all -v \C:\Users\momot\Documents\GitHub\FYP-Implementation:/workspace fyp-ssl:latest \
  python train_ssl.py --config configs/benchmarks/ssl_pseudolabel_cifar10_250labels.yaml
\\\

## Repository Structure

\\\
├── src/
│   ├── algorithms/          # Augmentation utilities
│   ├── data/                # CIFAR-10 loading and splits
│   ├── models/              # ResNet and WideResNet
│   ├── training/            # Trainer classes (pseudolabel, fixmatch, mixmatch, flexmatch, supervised)
│   └── utils/               # Logging, config, visualization
├── configs/benchmarks/      # YAML configs for each algorithm x label budget
├── docker/                  # Container setup
├── checkpoints/             # Saved model weights
├── experiments/             # Training logs and results
├── launch_docker.py         # Interactive launcher
├── train_ssl.py             # SSL training entrypoint
├── train_supervised_limited.py  # Supervised training entrypoint
└── requirements.txt
\\\

## Configuration

Benchmark configs in \configs/benchmarks/\ specify algorithm, hyperparameters, and label budget (40, 250, 1000, 4000 labels).

## Running Experiments

**List available benchmarks:**
\\\ash
python launch_docker.py --list
\\\

**Train a model:**
\\\ash
python launch_docker.py --dataset cifar10 --algorithm pseudolabel --split 250labels
\\\

**With overrides:**
\\\ash
python launch_docker.py --dataset cifar10 --algorithm fixmatch --split 1000labels --epochs 100
\\\

## Requirements

- Python 3.8+, PyTorch 2.0+
- Docker (recommended)
- NVIDIA GPU (optional)

Local install:
\\\ash
pip install -r requirements.txt
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
\\\

Docker:
\\\ash
docker build -t fyp-ssl:latest -f docker/Dockerfile .
\\\

## Output

Training outputs to \experiments/\ and \logs/\:
- Convergence curves, confusion matrices, calibration diagrams
- Final checkpoints and metrics

## References

- Lee, D. (2013). Pseudo-Label: Simple and Efficient Semi-Supervised Learning
- Berthelot et al. (2019). MixMatch: A Holistic Approach to Semi-Supervised Learning
- Zhang et al. (2020). FlexMatch: A Curriculum Pseudo-Labeling Framework for Semi-Supervised Learning
