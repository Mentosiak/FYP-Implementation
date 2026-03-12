# Docker Experiment Guide

## Quick Start - Running Experiments in Docker

### Step 1: Build the Docker Image

```bash
# Windows
run_docker.bat build

# Linux/Mac
docker build -t fyp-ssl:latest -f docker/Dockerfile .
```

This will:
- Create a container with CUDA support
- Install all Python dependencies (PyTorch, torchvision, etc.)
- Set up the complete environment

### Step 2: Verify Installation

```bash
# Windows
run_docker.bat verify

# Linux/Mac
docker run --rm -v ${PWD}:/workspace fyp-ssl:latest python verify_installation.py
```

### Step 3: Run Quick Test (2 epochs)

```bash
# Windows (with GPU)
run_docker.bat test-ssl

# Windows (CPU only)
run_docker.bat test-ssl-cpu

# Linux/Mac (with GPU)
docker run --rm --gpus all -v ${PWD}:/workspace fyp-ssl:latest python test_ssl.py

# Linux/Mac (CPU only)
docker run --rm -v ${PWD}:/workspace fyp-ssl:latest python test_ssl.py
```

This runs a quick 2-epoch test to verify the SSL pipeline works.

### Step 4: Run Full Experiment

```bash
# Windows
run_docker.bat full-experiment

# Linux/Mac
docker run --rm --gpus all -v ${PWD}:/workspace fyp-ssl:latest bash -c "
    python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml &&
    python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml &&
    python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml
"
```

This will:
1. Train supervised model with 2,500 labels (~1-2 hours on GPU)
2. Train SSL model with 2,500 labels + unlabeled data (~2-3 hours on GPU)
3. Generate comparison plots and results

---

## Individual Training Commands

### Train Supervised (Limited Labels)

```bash
# Windows
run_docker.bat train-supervised

# Linux/Mac
docker run --rm --gpus all -v ${PWD}:/workspace fyp-ssl:latest \
    python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml
```

### Train SSL Pseudo-Label

```bash
# Windows
run_docker.bat train-ssl

# Linux/Mac
docker run --rm --gpus all -v ${PWD}:/workspace fyp-ssl:latest \
    python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```

### Compare Results

```bash
# Windows
run_docker.bat compare

# Linux/Mac
docker run --rm --gpus all -v ${PWD}:/workspace fyp-ssl:latest \
    python compare_methods.py \
        --supervised-config configs/supervised_limited_cifar10.yaml \
        --ssl-config configs/ssl_pseudolabel_cifar10.yaml
```

---

## Using Docker Compose

### Build

```bash
docker-compose build
```

### Run Specific Services

```bash
# Verify installation
docker-compose run --rm verify

# Quick test
docker-compose run --rm test-ssl

# Train supervised
docker-compose run --rm train-supervised-limited

# Train SSL
docker-compose run --rm train-ssl

# Compare
docker-compose run --rm compare

# Full experiment
docker-compose run --rm full-experiment
```

---

## Interactive Shell

For manual experimentation:

```bash
# Windows (with GPU)
run_docker.bat shell

# Windows (CPU only)
run_docker.bat shell-cpu

# Linux/Mac (with GPU)
docker run -it --rm --gpus all -v ${PWD}:/workspace fyp-ssl:latest /bin/bash

# Linux/Mac (CPU only)
docker run -it --rm -v ${PWD}:/workspace fyp-ssl:latest /bin/bash
```

Inside the shell, you can run any command:
```bash
python verify_installation.py
python test_ssl.py
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```

---

## Expected Timeline

| Task | GPU Time | CPU Time |
|------|----------|----------|
| Build image | 5-10 min | 5-10 min |
| Verify | <1 min | <1 min |
| Quick test (2 epochs) | 2-3 min | 10-15 min |
| Train supervised (300 epochs) | 1-2 hours | 12-24 hours |
| Train SSL (300 epochs) | 2-3 hours | 24-48 hours |
| Compare | <1 min | <1 min |
| **Total experiment** | **3-5 hours** | **36-72 hours** |

---

## Output Files

All outputs are saved to your local directory via volume mount:

```
checkpoints/
├── supervised_limited_cifar10/
│   ├── supervised_limited_cifar10_best.pt
│   ├── supervised_limited_cifar10_last.pt
│   └── supervised_limited_cifar10_config.yaml
└── ssl_pseudolabel_cifar10/
    ├── ssl_pseudolabel_cifar10_best.pt
    ├── ssl_pseudolabel_cifar10_last.pt
    └── ssl_pseudolabel_cifar10_config.yaml

logs/
├── supervised_limited_cifar10/
│   └── supervised_limited_cifar10_TIMESTAMP.log
└── ssl_pseudolabel_cifar10/
    └── ssl_pseudolabel_cifar10_TIMESTAMP.log

experiments/
└── comparison/
    └── TIMESTAMP/
        ├── comparison_results.json
        ├── comparison_test_acc.png
        ├── comparison_train_loss.png
        └── comparison_test_loss.png
```

---

## Tips for Running Experiments

### 1. **Start with Quick Test**
```bash
run_docker.bat test-ssl
```
This verifies everything works before running long experiments.

### 2. **Use GPU if Available**
- Experiments are ~10-20x faster on GPU
- Ensure NVIDIA Docker runtime is installed
- Use `--gpus all` flag

### 3. **Monitor Progress**
```bash
# Watch logs in real-time
tail -f logs/ssl_pseudolabel_cifar10/*.log

# On Windows PowerShell
Get-Content logs/ssl_pseudolabel_cifar10/*.log -Wait
```

### 4. **Resume Interrupted Training**
```bash
docker run --rm --gpus all -v ${PWD}:/workspace fyp-ssl:latest \
    python train_ssl.py \
    --config configs/ssl_pseudolabel_cifar10.yaml \
    --resume checkpoints/ssl_pseudolabel_cifar10/ssl_pseudolabel_cifar10_last.pt
```

### 5. **CPU-Only Training**
Remove `--gpus all` flag:
```bash
docker run --rm -v ${PWD}:/workspace fyp-ssl:latest \
    python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```

---

## Troubleshooting

### Docker Build Fails
```bash
# Clear cache and rebuild
docker build --no-cache -t fyp-ssl:latest -f docker/Dockerfile .
```

### GPU Not Detected
```bash
# Check NVIDIA Docker runtime
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# If fails, install nvidia-docker2
```

### Out of Memory
Edit configs to reduce batch sizes:
```yaml
training:
  batch_size: 32  # Reduce from 64
ssl:
  unlabeled_batch_size: 128  # Reduce from 256
```

### Slow Download
CIFAR-10 downloads automatically on first run. If slow:
```bash
# Download manually first
docker run -it --rm -v ${PWD}:/workspace fyp-ssl:latest bash
python -c "from torchvision import datasets; datasets.CIFAR10('./data', download=True)"
```

---

## Recommended Workflow

### Option 1: Full Automated Run
```bash
# Build once
run_docker.bat build

# Run complete experiment
run_docker.bat full-experiment

# Results in experiments/comparison/
```

### Option 2: Step-by-Step
```bash
# 1. Build
run_docker.bat build

# 2. Verify
run_docker.bat verify

# 3. Quick test
run_docker.bat test-ssl

# 4. Train supervised
run_docker.bat train-supervised

# 5. Train SSL
run_docker.bat train-ssl

# 6. Compare
run_docker.bat compare
```

### Option 3: Interactive Experimentation
```bash
# Open shell
run_docker.bat shell

# Inside container:
python verify_installation.py
python test_ssl.py

# Edit configs and rerun
nano configs/ssl_pseudolabel_cifar10.yaml
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```

---

## Expected Results

After running the full experiment, you'll get:

### Comparison Results JSON
```json
{
  "supervised": {
    "best_test_acc": 72.5,
    "labels_used": 2500
  },
  "ssl": {
    "best_test_acc": 86.8,
    "labels_used": 2500
  }
}
```

### Comparison Plots
- Test accuracy curves (supervised vs SSL)
- Training loss curves
- Test loss curves

### Key Metric
**SSL Improvement: +14-15% accuracy over supervised with same labels!**

---

## Next Steps

After getting results:
1. Review plots in `experiments/comparison/`
2. Check logs for detailed metrics
3. Try different hyperparameters
4. Experiment with label fractions (1%, 10%, etc.)
5. Add more SSL algorithms (MixMatch, FixMatch)

---

**Ready to run your first SSL experiment!** 🚀

Start with: `run_docker.bat build` then `run_docker.bat test-ssl`
