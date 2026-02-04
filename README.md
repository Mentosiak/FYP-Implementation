Comparative Study of Semi‑Supervised and Supervised Learning for Image Classification

This repository currently contains a simplified supervised baseline for CIFAR‑10/100 using PyTorch. It is modular and ready for later SSL extensions, but for now it focuses on a single supervised training loop.

What’s included (current state)

    Supervised training loop (cross‑entropy + SGD)
    CIFAR‑10 and CIFAR‑100 loaders (no augmentation)
    ResNet‑18/34 and WideResNet backbones
    Docker setup for reproducible runs (CPU or GPU)

Repository structure

.
├── configs/                 # Example config (not required by code)
├── data/                    # Downloaded datasets
├── docker/                  # Dockerfile
├── src/
│   ├── data/                # Dataset loaders
│   ├── models/              # ResNet / WideResNet definitions
│   ├── training/            # Supervised trainer
│   └── utils/               # Helpers (seed, device)
├── train_supervised.py      # Main training script
├── test_supervised.py       # Quick validation test
└── requirements.txt         # Python dependencies

Setup

Option 1: Local Python

    pip install -r requirements.txt
    python train_supervised.py --epochs 50

Option 2: Docker

    docker build -t fyp-implementation -f docker/Dockerfile .
    docker run -it --rm -v ${PWD}:/workspace fyp-implementation python train_supervised.py --epochs 50

GPU (optional but highly recommended if you have a GPU) 

    docker run -it --rm --gpus all -v ${PWD}:/workspace fyp-implementation python train_supervised.py --epochs 50

Usage

    Quick sanity test:
    python test_supervised.py

    Full training (200 epochs):
    python train_supervised.py --epochs 200

Notes

    Data augmentation, label‑fraction support, and checkpointing are intentionally removed for now to keep the code simple.
