# 🚀 Experiment Execution Checklist

## Before You Start

- [ ] Decide: Docker or Local Python?
  - **Docker:** No Python setup, clean environment, easier
  - **Local:** More control, faster iteration

---

## Option A: Docker Path ⭐ (Recommended)

### Pre-Run Checklist
- [ ] Docker Desktop installed
- [ ] Docker Desktop running (check system tray)
- [ ] At least 10GB free disk space
- [ ] (Optional) NVIDIA drivers for GPU support

### Execution Steps

```powershell
# 1. Build Docker image (10 min, one-time)
.\run_docker.bat build
```
- [ ] Build completed successfully
- [ ] No error messages

```powershell
# 2. Verify installation (1 min)
.\run_docker.bat verify
```
- [ ] All checks passed ✅
- [ ] CUDA available (or CPU fallback is OK)

```powershell
# 3. Quick test (3 min)
.\run_docker.bat test-ssl
```
- [ ] Test completed successfully
- [ ] Accuracy > 25%

```powershell
# 4. Run full experiment (3-5 hours)
.\run_docker.bat full-experiment
```
- [ ] Supervised training started
- [ ] SSL training started
- [ ] Comparison generated

### Results Checklist
- [ ] Files created in `experiments/comparison/TIMESTAMP/`
- [ ] `comparison_results.json` exists
- [ ] `comparison_test_acc.png` shows SSL > Supervised
- [ ] Checkpoints saved in `checkpoints/`
- [ ] Logs available in `logs/`

---

## Option B: Local Python Path

### Pre-Run Checklist
- [ ] Python 3.10+ installed
- [ ] Virtual environment created
- [ ] Dependencies installed from requirements.txt
- [ ] At least 5GB free disk space

### Execution Steps

```powershell
# 1. Activate environment
.\.venv\Scripts\activate
```
- [ ] Environment activated (prompt shows `.venv`)

```powershell
# 2. Verify installation
python verify_installation.py
```
- [ ] All imports successful
- [ ] Config loading works
- [ ] Model building works

```powershell
# 3. Quick test
python test_ssl.py
```
- [ ] Test completed
- [ ] Model trained for 2 epochs
- [ ] No errors

```powershell
# 4. Train supervised
python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml
```
- [ ] Training started
- [ ] Progress visible in console
- [ ] Checkpoint saved after completion

```powershell
# 5. Train SSL
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
```
- [ ] Training started
- [ ] Pseudo-label ratio visible
- [ ] Checkpoint saved after completion

```powershell
# 6. Compare
python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml
```
- [ ] Comparison completed
- [ ] Plots generated
- [ ] Results saved

### Results Checklist
Same as Docker path above.

---

## During Training Monitoring

### Things to Watch
- [ ] Epoch number increasing (1/300, 2/300, ...)
- [ ] Test accuracy improving over time
- [ ] No error messages or warnings
- [ ] Disk space not full
- [ ] (GPU) Temperature reasonable (<85°C)

### Expected Progress

**Supervised Training:**
```
Epoch [1/300] | Test Acc: ~25%
Epoch [50/300] | Test Acc: ~60%
Epoch [150/300] | Test Acc: ~70%
Epoch [300/300] | Test Acc: ~72-75%
```

**SSL Training:**
```
Epoch [1/300] | Test Acc: ~28% | Pseudo: ~45%
Epoch [50/300] | Test Acc: ~65% | Pseudo: ~75%
Epoch [150/300] | Test Acc: ~82% | Pseudo: ~85%
Epoch [300/300] | Test Acc: ~85-88% | Pseudo: ~90%
```

### Warning Signs
- [ ] Accuracy stuck at 10% for >50 epochs → Check config
- [ ] "Out of memory" error → Reduce batch size
- [ ] Training very slow → Check if using GPU
- [ ] Loss = NaN → Learning rate too high

---

## After Completion

### Verify Results
- [ ] Navigate to `experiments/comparison/TIMESTAMP/`
- [ ] Open `comparison_results.json`
- [ ] Confirm:
  - Supervised accuracy: 70-75%
  - SSL accuracy: 85-88%
  - SSL improvement: ~15%

### View Plots
- [ ] Open `comparison_test_acc.png`
- [ ] Verify SSL curve above supervised curve
- [ ] Check convergence (both curves flatten)

### Check Checkpoints
- [ ] `checkpoints/supervised_limited_cifar10/supervised_limited_cifar10_best.pt` exists
- [ ] `checkpoints/ssl_pseudolabel_cifar10/ssl_pseudolabel_cifar10_best.pt` exists
- [ ] Both >100MB in size

### Review Logs
- [ ] Open log files in `logs/`
- [ ] Verify 300 epochs completed
- [ ] Check final "Best Test Accuracy" line

---

## Success Criteria

Your experiment is successful if:

- [x] Both models trained for 300 epochs
- [x] Supervised accuracy: 70-75%
- [x] SSL accuracy: 85-88%
- [x] SSL shows ~15% improvement
- [x] Comparison plots generated
- [x] Results saved as JSON
- [x] No errors or crashes

---

## If Something Goes Wrong

### Training Fails
1. Check logs in `logs/` directory
2. Look for error messages
3. Common fixes:
   - Reduce batch size in config
   - Use CPU if GPU issues
   - Free up disk space

### Results Look Wrong
1. Verify correct configs used
2. Check if 300 epochs completed
3. Ensure CIFAR-10 data downloaded correctly
4. Try rerunning comparison only:
   ```powershell
   .\run_docker.bat compare
   ```

### Need to Restart
1. Stop current run (Ctrl+C)
2. Resume from checkpoint:
   ```powershell
   .\run_docker.bat shell
   python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml --resume checkpoints/ssl_pseudolabel_cifar10/ssl_pseudolabel_cifar10_last.pt
   ```

---

## Time Estimates

**With GPU (e.g., RTX 3080):**
- Total time: 3-5 hours
- Can run overnight or during class

**With CPU:**
- Total time: 36-72 hours
- Recommended: Run over weekend

**Planning:**
```
Start Friday evening
→ Completes by Monday morning (GPU)
→ Completes by Tuesday morning (CPU)
```

---

## For Your Report

### What to Include:

1. **Methodology:**
   - Dataset: CIFAR-10
   - Label amount: 2,500 (5% of data)
   - Algorithms: Supervised vs SSL Pseudo-Label
   - Architecture: WideResNet-28-2
   - Training: 300 epochs

2. **Results:**
   - Supervised: 72-75% accuracy
   - SSL: 85-88% accuracy
   - Improvement: +13-15%

3. **Figures:**
   - Include `comparison_test_acc.png`
   - Caption: "SSL achieves higher accuracy with same label budget"

4. **Analysis:**
   - SSL leverages unlabeled data effectively
   - 95% reduction in labeling requirements
   - Near-full-supervised performance (90% of 94-95%)

---

## Quick Reference

### Docker Commands
```powershell
.\run_docker.bat build              # Setup
.\run_docker.bat verify             # Check
.\run_docker.bat test-ssl           # Quick test
.\run_docker.bat full-experiment    # Complete run
.\run_docker.bat shell              # Debug
```

### Python Commands
```powershell
python verify_installation.py       # Check
python test_ssl.py                  # Quick test
python train_supervised_limited.py --config configs/supervised_limited_cifar10.yaml
python train_ssl.py --config configs/ssl_pseudolabel_cifar10.yaml
python compare_methods.py --supervised-config configs/supervised_limited_cifar10.yaml --ssl-config configs/ssl_pseudolabel_cifar10.yaml
```

### Files to Check
```
experiments/comparison/TIMESTAMP/comparison_results.json  # Results
experiments/comparison/TIMESTAMP/comparison_test_acc.png  # Main plot
checkpoints/*/  # Trained models
logs/*/  # Training logs
```

---

## Ready to Run?

### Fastest Path to Results:

1. **Start Docker Desktop**
2. **Open PowerShell in project directory**
3. **Run:** `.\run_docker.bat build` (wait 10 min)
4. **Run:** `.\run_docker.bat full-experiment` (wait 3-5 hours)
5. **Check:** `experiments\comparison\`
6. **Done!** You have your results! 🎉

---

**Print this checklist and tick off items as you go!** ✓
