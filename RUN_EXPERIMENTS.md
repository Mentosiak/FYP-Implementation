# 🚀 Complete Experiment Setup & Execution Guide

## Prerequisites Check

Before running experiments, you need:

### ✅ Option 1: Docker (Recommended - No local Python setup needed)
1. **Install Docker Desktop** (if not already installed)
   - Download from: https://www.docker.com/products/docker-desktop
   - Install and start Docker Desktop
   - Verify: Open PowerShell and run `docker --version`

2. **For GPU support** (optional but 10x faster):
   - Install NVIDIA drivers
   - Install NVIDIA Container Toolkit
   - Verify: `docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi`

### ✅ Option 2: Local Python Environment
1. **Python 3.10 or higher**
   - Verify: `python --version`

2. **Create virtual environment**:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```powershell
   pip install -r requirements.txt
   ```

---

## 🐳 Option 1: Running with Docker (Recommended)

### Step 1: Start Docker Desktop
- Open Docker Desktop application
- Wait for it to show "Docker Desktop is running"

### Step 2: Build the Image
```powershell
# Using helper script
.\run_docker.bat build

# Or directly
docker build -t fyp-ssl:latest -f docker/Dockerfile .
```

**Time:** 5-10 minutes (downloads ~5GB)

### Step 3: Verify Installation
```powershell
.\run_docker.bat verify
```

**Expected output:**
```
✅ torch
✅ torchvision
✅ numpy
...
✅ All checks passed! System is ready.
```

### Step 4: Quick Test (2 epochs - ~3 minutes)
```powershell
# With GPU (faster)
.\run_docker.bat test-ssl

# Without GPU (slower but works everywhere)
.\run_docker.bat test-ssl-cpu
```

**Expected output:**
```
TEST COMPLETED SUCCESSFULLY! ✅
Final Test Accuracy: ~25-35%
```

### Step 5: Run Full Experiment

#### Option A: Automated Full Run (~3-5 hours with GPU)
```powershell
.\run_docker.bat full-experiment
```

This will:
1. Train supervised with 2,500 labels (~1-2 hours)
2. Train SSL with 2,500 labels + unlabeled (~2-3 hours)
3. Generate comparison plots and results (~1 minute)

#### Option B: Step-by-Step (for monitoring)
```powershell
# 1. Train supervised (1-2 hours)
.\run_docker.bat train-supervised

# 2. Train SSL (2-3 hours)
.\run_docker.bat train-ssl

# 3. Compare results
.\run_docker.bat compare
```

#### Option C: Literature-Standard Benchmark Sweep
```powershell
.\run_docker.bat benchmark-sweep
```

This runs literature-standard CIFAR-10 SSL label-count experiments:
1. 250 total labels = 25 per class (MixMatch)
2. 1000 total labels = 100 per class (Supervised, Pseudo-Label, MixMatch)
3. 4000 total labels = 400 per class (Supervised, Pseudo-Label, FixMatch)

The sweep script executes all configured runs and then writes comparison outputs.

The benchmark configs live under `configs/benchmarks/` and keep the original default configs untouched.

### Docker Troubleshooting (If Build/Run Fails)
If you get daemon connection errors (for example, `dockerDesktopLinuxEngine` not found), Docker CLI is installed but Docker Desktop is not running yet.

1. Start Docker Desktop and wait until it reports it is running.
2. Re-run:
  ```powershell
  docker info
  ```
3. Then continue with:
  ```powershell
  .\run_docker.bat build
  .\run_docker.bat benchmark-sweep
  ```

### Step 6: Check Results

After completion, find results in:
```
experiments/comparison/YYYYMMDD_HHMMSS/
├── comparison_results.json         # Numerical results
├── curves/                         # Accuracy/loss curves
├── confusion_matrices/             # Seaborn confusion matrices
├── calibration/                    # Reliability diagrams + calibration_metrics.json
└── ...
```

**Expected Results:**
```json
{
  "supervised": {
    "best_test_acc": 70-75%,
    "labels_used": 2500
  },
  "ssl": {
    "best_test_acc": 85-88%,
    "labels_used": 2500
  }
}
```

**Key Finding:** SSL achieves ~15% higher accuracy with the same number of labels!

### Step 7: Generate Consolidated Benchmark Summary

After individual runs complete, generate a single matrix and machine-readable summary:

```powershell
python summarize_benchmarks.py
```

Outputs are written under:

```
experiments/benchmark_summary/YYYYMMDD_HHMMSS/
├── benchmark_summary.json      # Full run metadata and status
├── benchmark_summary.csv       # Flat table for spreadsheet/thesis import
└── benchmark_matrix.md         # Human-readable completion matrix
```

Use this summary to quickly identify missing runs and underperforming configs.

### Step 8: Interactive Docker CUDA Training CLI

Use the interactive launcher to pick algorithm, dataset, model, and label budget:

```powershell
python train_cli_docker.py
```

Optional non-interactive example:

```powershell
python train_cli_docker.py --algorithm flexmatch --dataset cifar10 --model wideresnet --labels-per-class 100 --epochs 300
```

The CLI generates a YAML config under `configs/generated/` and launches the matching training script inside Docker with `--gpus all`.

### Step 9: Start Remaining CUDA Queue

To launch the current remaining run queue (FixMatch 4000 resume, MixMatch 250 tuned, FlexMatch 1000):

```powershell
.\run_remaining_cuda_queue.bat
```

---

## 💻 Option 2: Running Locally (Without Docker)

### Step 1: Set Up Python Environment

```powershell
# Create virtual environment
python -m venv .venv

# Activate it
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Verify Installation
```powershell
python verify_installation.py
```

### Step 3: Quick Test
```powershell
python test_ssl.py
```

### Step 4: Run Experiments

```powershell
# Train supervised (1-2 hours with GPU, 12-24 hours CPU)
python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml

# Train SSL (2-3 hours with GPU, 24-48 hours CPU)
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml

# Compare results
python compare_methods.py `
    --supervised-config configs/supervised_limited_cifar10.yaml `
    --ssl-config configs/ssl_pseudolabel_cifar10.yaml
```

---

## 📊 Monitoring Progress

### View Logs in Real-Time

**PowerShell:**
```powershell
Get-Content logs\ssl_pseudolabel_cifar10\*.log -Wait
```

**Logs show:**
- Epoch progress (1/300, 2/300, ...)
- Training/test accuracy
- Loss values
- Pseudo-label usage ratio
- Time per epoch
- Best accuracy so far

### Example Log Output:
```
Epoch [50/300] | Time: 24.5s | LR: 0.029850 |
Train Loss: 0.8234 | Train Acc: 72.45% |
Test Loss: 1.2341 | Test Acc: 68.23% |
Pseudo Ratio: 87.34% | Best: 69.12%
```

---

## 🎯 What the Experiment Does

### Supervised Training (Limited Labels)
- Uses only 2,500 labeled images (5% of CIFAR-10)
- 250 labels per class
- Standard supervised cross-entropy loss
- Expected accuracy: **70-75%**

### SSL Pseudo-Label Training
- Uses same 2,500 labeled images
- PLUS 47,500 unlabeled images via pseudo-labeling
- Generates pseudo-labels for unlabeled data
- Filters by confidence (95% threshold)
- Expected accuracy: **85-88%**

### Comparison
- Plots accuracy curves over epochs
- Shows SSL outperforms supervised by ~15%
- Demonstrates label efficiency
- Saves results for analysis

### Benchmark Protocol Notes
- The benchmark sweep uses exact literature-standard labeled counts: 250, 1000, and 4000 total labels.
- These configs set `validation_split: 0.0` so the effective number of training labels matches the published protocol exactly.
- For direct paper comparison, compare Pseudo-Label against the labeled-only baseline and against a stronger SSL reference such as FixMatch under the same split definitions.

---

## ⏱️ Estimated Times

| Task | GPU (RTX 3080) | CPU |
|------|----------------|-----|
| Docker build | 5-10 min | 5-10 min |
| Quick test (2 epochs) | 2-3 min | 15-20 min |
| Supervised training (300 epochs) | 1-2 hours | 12-24 hours |
| SSL training (300 epochs) | 2-3 hours | 24-48 hours |
| Comparison | <1 min | <1 min |
| **Full experiment** | **3-5 hours** | **36-72 hours** |

---

## 🔧 Troubleshooting

### Docker Issues

**Docker not running:**
```
ERROR: failed to connect to docker API
```
**Solution:** Start Docker Desktop application

**Out of disk space:**
```
ERROR: no space left on device
```
**Solution:** 
- Clean Docker: `docker system prune -a`
- Free up disk space

**GPU not detected:**
```
WARNING: CUDA not available
```
**Solution:**
- Install NVIDIA Container Toolkit
- Or use CPU version: `.\run_docker.bat test-ssl-cpu`

### Training Issues

**Out of memory:**
```
RuntimeError: CUDA out of memory
```
**Solution:** Edit config to reduce batch size:
```yaml
training:
  batch_size: 32  # Reduce from 64
ssl:
  unlabeled_batch_size: 128  # Reduce from 256
```

**Training stuck at 0%:**
- Wait a few minutes (data loading can be slow on first run)
- Check logs for errors
- Verify CIFAR-10 downloaded: `dir data\cifar-10-batches-py`

**MixMatch 250-label underperforming:**
- If `ssl_mixmatch_cifar10_250labels` is far below expected range, rerun with the tuned recovery config:
```powershell
python train_ssl.py --config configs/benchmarks/ssl_mixmatch_cifar10_250labels_tuned.yaml
```

---

## 📝 Complete Workflow Example

Here's a complete session from start to finish:

```powershell
# Step 1: Start Docker Desktop (manually)
# Wait for "Docker Desktop is running"

# Step 2: Build image (one-time, ~10 min)
PS> .\run_docker.bat build
[+] Building 347.2s (12/12) FINISHED
Successfully tagged fyp-ssl:latest

# Step 3: Verify setup
PS> .\run_docker.bat verify
✅ All checks passed! System is ready.

# Step 4: Quick test (optional, ~3 min)
PS> .\run_docker.bat test-ssl
TEST COMPLETED SUCCESSFULLY! ✅
Final Test Accuracy: 32.45%

# Step 5: Run full experiment (~4 hours)
PS> .\run_docker.bat full-experiment
=== Step 1/3: Training Supervised ===
Epoch [1/300] | ... | Test Acc: 25.67%
Epoch [2/300] | ... | Test Acc: 32.12%
...
Epoch [300/300] | ... | Test Acc: 72.34% | Best: 72.89%
Training Completed! Best: 72.89%

=== Step 2/3: Training SSL ===
Epoch [1/300] | ... | Test Acc: 28.45% | Pseudo: 45.2%
Epoch [2/300] | ... | Test Acc: 35.67% | Pseudo: 67.8%
...
Epoch [300/300] | ... | Test Acc: 86.12% | Best: 86.45%
Training Completed! Best: 86.45%

=== Step 3/3: Comparing Results ===
Supervised - Best Acc: 72.89%
SSL - Best Acc: 86.45%
Accuracy Gap: 13.56%
Comparison complete!

# Step 6: View results
PS> dir experiments\comparison\
20260225_143052\
  comparison_results.json
  comparison_test_acc.png
  comparison_train_loss.png
  ...
```

---

## 📈 What You'll Get

### 1. Numerical Results
`experiments/comparison/TIMESTAMP/comparison_results.json`:
```json
{
  "timestamp": "20260225_143052",
  "supervised": {
    "best_test_acc": 72.89,
    "final_test_acc": 72.34,
    "num_epochs": 300,
    "total_time": 4523.45
  },
  "ssl": {
    "best_test_acc": 86.45,
    "final_test_acc": 86.12,
    "num_epochs": 300,
    "total_time": 8234.67,
    "labels_per_class": 250
  }
}
```

### 2. Plots
- **comparison_test_acc.png** - Shows SSL achieving higher accuracy
- **comparison_train_loss.png** - Training loss curves
- **comparison_test_loss.png** - Test loss curves

### 3. Model Checkpoints
```
checkpoints/
├── supervised_limited_cifar10/
│   ├── supervised_limited_cifar10_best.pt (for reuse/analysis)
│   └── supervised_limited_cifar10_config.yaml
└── ssl_pseudolabel_cifar10/
    ├── ssl_pseudolabel_cifar10_best.pt (for reuse/analysis)
    └── ssl_pseudolabel_cifar10_config.yaml
```

### 4. Detailed Logs
```
logs/
├── supervised_limited_cifar10/
│   └── supervised_limited_cifar10_20260225_143052.log
└── ssl_pseudolabel_cifar10/
    └── ssl_pseudolabel_cifar10_20260225_150234.log
```

---

## 🎓 For Your Final Year Project

### Key Results to Report:

1. **Label Efficiency:**
   - Supervised (2.5K labels): ~72-75%
   - SSL (2.5K labels + unlabeled): ~85-88%
   - **Improvement: +13-15%**

2. **Effective Label Utilization:**
   - SSL achieves ~90% of full supervised performance (94-95%)
   - Using only 5% of labeled data

3. **Practical Impact:**
   - Reduces labeling requirements by 95%
   - Maintains competitive performance
   - Demonstrates value of unlabeled data

### Plots for Report:
- Use `comparison_test_acc.png` to show SSL vs Supervised curves
- Highlight the accuracy gap
- Discuss pseudo-label usage ratio

### Discussion Points:
- SSL leverages unlabeled data effectively
- Confidence threshold prevents noisy pseudo-labels
- Strong augmentation crucial for SSL
- Trade-off: More training time but fewer labels needed

---

## ⚡ Quick Start Commands

**Choose your path:**

### If you have Docker Desktop installed:
```powershell
.\run_docker.bat build          # One-time setup
.\run_docker.bat test-ssl       # Quick test
.\run_docker.bat full-experiment # Complete run
```

### If you prefer local Python:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python test_ssl.py
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```

---

## 🎯 Your Next Steps

1. **Start Docker Desktop**
2. **Run:** `.\run_docker.bat build`
3. **Run:** `.\run_docker.bat test-ssl`
4. **Run:** `.\run_docker.bat full-experiment`
5. **Wait 3-5 hours** (or leave overnight)
6. **Check:** `experiments/comparison/` for results
7. **Analyze plots and metrics for your report!**

---

**You're all set to run your SSL experiments! 🚀**

Questions? Check [DOCKER_GUIDE.md](DOCKER_GUIDE.md) for more details.
