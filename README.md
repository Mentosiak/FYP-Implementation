# Semi-Supervised Learning Benchmark (CIFAR-10)

This project was completed as a Final Year Project (FYP) for Munster Technological University (MTU) in Software Development.

It implements and compares multiple semi-supervised learning (SSL) algorithms against supervised baselines for image classification on CIFAR-10.

## Implemented Methods

- Supervised (limited labels)
- Pseudo-Label
- FixMatch
- MixMatch
- FlexMatch

## Quickstart (Windows + Docker)

### 1) Build the Docker image

```powershell
docker build -t fyp-ssl:latest -f docker/Dockerfile .
```

### 2) Interactive Launcher (recommended on Windows)

- Double-click `launch_docker.cmd`

You will be prompted to choose the dataset, algorithm, backbone, label budget/split, and whether to use GPU.

### 3) Launcher from the command line

List the available runs:

```powershell
python launch_docker.py --list
```

Run an experiment (example):

```powershell
python launch_docker.py --dataset cifar10 --algorithm pseudolabel --split 250labels
```

### 4) Docker direct (no launcher)

GPU:

```powershell
docker run --rm --gpus all -v ${PWD}:/workspace -w /workspace fyp-ssl:latest `
  python train_ssl.py --config configs/benchmarks/ssl_pseudolabel_cifar10_250labels.yaml
```

CPU (no NVIDIA GPU):

```powershell
docker run --rm -v ${PWD}:/workspace -w /workspace fyp-ssl:latest `
  python train_ssl.py --config configs/benchmarks/ssl_pseudolabel_cifar10_250labels.yaml
```

## Running Experiments (what to expect)

- Benchmark splits used in this project: `250labels`, `1000labels`, `4000labels`.
- Outputs are written to `experiments/`, `logs/`, and `checkpoints/`.
- If you want to run multiple experiments, build the image once and re-run the launcher for each selection.

## Requirements

- Docker Desktop
- NVIDIA GPU + NVIDIA Container Toolkit (optional, for `--gpus all`)
- Python 3.x (only required to run `launch_docker.py` on the host)
