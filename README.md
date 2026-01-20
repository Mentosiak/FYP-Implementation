Comparative Study of Semi‑Supervised and Supervised Learning for Image Classification

This repository contains the code for a final year project that benchmarks multiple semi‑supervised learning (SSL) algorithms against supervised baselines on standard image classification datasets such as CIFAR‑10 and CIFAR‑100.​

The goal is to implement a unified experimentation pipeline that fairly compares methods under varying label budgets and provides practical guidance on when SSL is preferable to supervised learning.​
Project overview

This project investigates how far unlabelled data can reduce the need for expensive manual labelling while maintaining strong classification performance. The system trains and evaluates several supervised and semi‑supervised algorithms under consistent conditions (shared backbone, data splits, and metrics).^​

Key ideas:

    Use a small labelled subset and a large unlabelled pool to train SSL methods such as Pseudo‑Label and FixMatch.​

    Compare them against strong supervised baselines using the same backbone (e.g. ResNet / WideResNet).^​

    Analyse accuracy, label efficiency, convergence, and computational cost across different label fractions (e.g. 1%, 5%, 10%, 25%, 100%).​

Features

    Unified training pipeline for supervised and semi‑supervised algorithms (shared training loop with algorithm‑specific strategies).^​

    Modular dataset handling with controllable labelled/unlabelled splits and reproducible random seeds.​

    Support for classic vision benchmarks (CIFAR‑10, CIFAR‑100, STL‑10, Intel Image Classification, with CIFAR‑10 as the primary focus).^​

    Config‑driven experiments (YAML/JSON configs to choose dataset, algorithm, label fraction, and hyperparameters).^​

    Reproducible environment via Docker with GPU support (PyTorch + CUDA).^​

    Evaluation tools for top‑1 accuracy, per‑class accuracy, confusion matrices, and training curves.​

Planned / optional:

    Additional SSL methods (e.g. MixMatch, Mean Teacher, FlexMatch) depending on time and resources.​

    Calibration metrics (e.g. Expected Calibration Error) and simple statistical tests over multiple seeds.​

Repository structure

The codebase is designed to be modular and easy to extend with new algorithms or datasets.​

text
.
├── configs/                 # YAML/JSON experiment configs (algo, dataset, label fraction) [file:1]
├── data/                    # Downloaded datasets and generated splits [file:1]
│   ├── cifar10/
│   │   ├── labelled/
│   │   ├── unlabelled/
│   │   └── test/
├── src/
│   ├── data/                # Dataset loaders, augmentations, split utilities [file:1]
│   ├── models/              # Backbone definitions (ResNet, WideResNet, etc.) [file:1]
│   ├── algorithms/          # Algorithm-specific training logic (supervised, FixMatch, etc.) [file:1]
│   ├── training/            # Base trainer, loops, logging, checkpointing [file:1]
│   ├── evaluation/          # Metrics, confusion matrices, plotting helpers [file:1]
│   └── utils/               # Config parsing, seeding, logging, CLI wrappers [file:1]
├── experiments/             # Saved runs, metrics (CSV/JSON), plots [file:1]
├── docker/
│   ├── Dockerfile           # Reproducible GPU-enabled environment [file:1]
│   └── docker-compose.yml   # Optional convenience for local runs [file:1]
├── README.md                # Project documentation
└── requirements.txt         # Python dependencies (PyTorch, NumPy, etc.) [file:1]

You can adjust folder names to match your actual layout, but try to preserve the separation between data, models, algorithms, training, and evaluation as described in your architecture section.​
Installation and setup
Prerequisites

    Python 3.10+

    NVIDIA GPU with CUDA support (recommended)^​

    Docker + NVIDIA Container Toolkit if using the containerised setup.​

Option 1: Local Python environment

    Clone the repository:

    bash
    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>

    Create and activate a virtual environment:

    bash
    python -m venv .venv
    source .venv/bin/activate  # Windows: .venv\Scripts\activate

    Install dependencies:

    bash
    pip install -r requirements.txt

    (Optional) Verify GPU visibility in PyTorch:

    bash
    python -c "import torch; print('CUDA available:', torch.cuda.is_available())"

Option 2: Docker (recommended)

    Build the image:

    bash
    docker build -t ssl-benchmark ./docker

    Run a container with GPU access:

    bash
    docker run --gpus all -it --rm \
      -v $(pwd):/workspace \
      -w /workspace \
      ssl-benchmark bash

The Docker image bundles Python, PyTorch, CUDA, and other dependencies to make experiments reproducible across machines.​
Usage

All experiments are configured via a command‑line interface and external config files so that no code changes are needed to rerun a setup.​
Training a supervised baseline

Example: ResNet‑18 on CIFAR‑10 with all labels:

bash
python -m src.training.run_experiment \
  --config configs/cifar10_supervised_100.yaml

This will:

    Load CIFAR‑10 and apply standard weak augmentations.​

    Train a supervised baseline using cross‑entropy on labelled data only.​

    Save checkpoints, logs, and metrics into experiments/.​

Training a semi‑supervised method

Example: FixMatch on CIFAR‑10 with 10% labelled data:

bash
python -m src.training.run_experiment \
  --config configs/cifar10_fixmatch_10pct.yaml

This will:

    Create a fixed 10% labelled / 90% unlabelled split with saved indices.​

    Apply weak + strong augmentations required by FixMatch.​

    Optimise a combined supervised + consistency loss on labelled and unlabelled batches.​

You can add more configs (e.g. 1%, 5%, 25% labels or different algorithms) by copying and editing the YAML files in configs/.​
Evaluating and plotting

After training, you can run evaluation scripts to compute metrics and generate figures:

bash
python -m src.evaluation.evaluate_run \
  --run-dir experiments/cifar10_fixmatch_10pct_seed1

bash
python -m src.evaluation.plot_label_efficiency \
  --results-dir experiments/cifar10_all_methods

Typical outputs:

    Test accuracy and per‑class accuracy.​

    Confusion matrices for selected models.​

    Accuracy vs label‑fraction plots comparing supervised and SSL methods.​

Experiments

Planned core experiments (matching the thesis):
Dataset	Algorithms (initial)	Label fractions	Notes
CIFAR‑10	Supervised baseline (ResNet / WRN), Pseudo‑Label, FixMatch	1%, 5%, 10%, 25%, 100%	Primary benchmark for the study.​
CIFAR‑100 (optional)	Supervised, FixMatch	10%	If resources allow.​
STL‑10 / Intel Image (optional)	Subset of above	Selected	For additional generality.​

Each experiment is repeated for at least two random seeds, with metrics aggregated to compute means and standard deviations.​
Reproducibility

The project follows best practices for reproducible SSL research:​

    Fixed random seeds recorded in each run directory.

    Saved labelled/unlabelled split indices for each dataset/label fraction.​

    All hyperparameters and settings stored in config files and experiment metadata.​

    Dockerised environment with pinned dependency versions.​

The goal is that any researcher can re‑run the experiments (given similar hardware) and recover the reported results.​
Roadmap

Short‑term:

    Implement supervised baseline on CIFAR‑10.

    Implement Pseudo‑Label as the first SSL method.

    Implement FixMatch with weak/strong augmentations.

    Finish full CIFAR‑10 grid of label fractions for all implemented methods.

Long‑term / stretch:

    Add additional SSL algorithms (MixMatch, Mean Teacher, FlexMatch, FreeMatch, SoftMatch, SimMatch).^​

    Add calibration metrics and basic statistical tests.​

    Extend to more datasets (CIFAR‑100, STL‑10, Intel).^​

Acknowledgements

This project was developed as part of the BSc in Software Development at Munster Technological University Cork. The research is heavily inspired by recent SSL work including Pseudo‑Label, MixMatch, FixMatch, UDA, FlexMatch, FreeMatch, SoftMatch, SimMatch and related benchmarks such as USB.
