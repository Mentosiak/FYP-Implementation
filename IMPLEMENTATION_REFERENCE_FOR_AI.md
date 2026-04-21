# Implementation Reference for Thesis Writing

This document summarizes what is actually implemented in the repository, with enough detail for another AI to use as a reference when helping write or revise the thesis. It focuses on the codebase state rather than the prose in the thesis draft.

## 1. Project Purpose

The project is a PyTorch-based image classification framework for comparing supervised learning with semi-supervised learning on CIFAR datasets, especially CIFAR-10 under restricted label budgets.

The main goals are:

1. Train a supervised baseline on full-label and reduced-label settings.
2. Train semi-supervised models using unlabeled data in addition to a small labeled subset.
3. Compare methods under the same backbone, optimizer family, and experiment structure.
4. Save checkpoints, logs, curves, confusion matrices, reliability diagrams, and comparison JSON outputs for analysis and thesis figures.

## 2. Current Implemented Algorithms

The codebase currently implements the following training modes:

1. Supervised training.
2. Supervised training with limited labels.
3. Pseudo-Label semi-supervised training.
4. FixMatch-style semi-supervised training.
5. MixMatch-style semi-supervised training.
6. FlexMatch-style adaptive-threshold pseudo-label training.

The repository also contains orchestration scripts for training, comparison, verification, and Docker-based runs.

## 3. Repository Architecture

The project is organized around these modules:

1. `src/data/` handles CIFAR loading, augmentation, and labeled/unlabeled splits.
2. `src/models/` defines the CIFAR-adapted neural network architectures.
3. `src/training/` contains the supervised and SSL trainer implementations.
4. `src/utils/` contains configuration loading, logging, seeding, and plotting helpers.
5. Top-level scripts provide CLI entry points for training and comparison.
6. YAML files under `configs/` define reproducible experiment settings.

## 4. Data Pipeline

### 4.1 Supported datasets

The data layer supports:

1. CIFAR-10.
2. CIFAR-100.

The implementation downloads datasets through `torchvision.datasets.CIFAR10` and `torchvision.datasets.CIFAR100`.

### 4.2 Normalization values

The CIFAR statistics are hard-coded in `src/data/cifar.py`:

1. CIFAR-10 mean: `(0.4914, 0.4822, 0.4465)`.
2. CIFAR-10 std: `(0.2023, 0.1994, 0.2010)`.
3. CIFAR-100 mean: `(0.5071, 0.4867, 0.4408)`.
4. CIFAR-100 std: `(0.2675, 0.2565, 0.2761)`.

### 4.3 Supervised loaders

`get_cifar_supervised_loaders()` creates:

1. A training loader.
2. A test loader.

It applies standard augmentation when `augment=True`:

1. Random crop with padding 4.
2. Random horizontal flip.
3. Tensor conversion.
4. CIFAR normalization.

When augmentation is disabled, it uses only tensor conversion and normalization.

### 4.4 SSL loaders

`get_cifar_ssl_loaders()` creates:

1. A labeled training loader.
2. An unlabeled training loader.
3. An optional validation loader.
4. A test loader.

The unlabeled loader returns weak and strong augmented views when strong augmentation is enabled. The data wrapper supports returning either:

1. `(weak_view, strong_view)` for unlabeled data.
2. `(image, label)` for labeled or validation data.

### 4.5 Split logic

Splitting is stratified by class, which is important for small label budgets.

The split logic supports three ways of specifying labeled data:

1. `labels_per_class`.
2. `num_labeled`.
3. `label_ratio`.

If `labels_per_class` is set, every class gets that many labeled examples, capped by class availability. If total labels are specified indirectly, the labels are spread across classes as evenly as possible, with remainders distributed cyclically.

Validation splitting is also stratified over the labeled subset using `val_split`. When a class has enough samples, the split keeps at least one item in train and one in validation for that class.

The split functions use `numpy.random.default_rng(seed)` so the labeled/unlabeled partitions are reproducible.

### 4.6 Augmentation strategy

The code distinguishes between weak and strong augmentation:

1. Weak augmentation: random crop + horizontal flip.
2. Strong augmentation: weak augmentation plus RandAugment.

This matches the expectations of SSL methods that use weak views to generate pseudo-labels and strong views to train on filtered unlabeled samples.

## 5. Model Implementations

### 5.1 ResNet

`src/models/resnet.py` implements CIFAR-adapted ResNet variants:

1. ResNet-18.
2. ResNet-34.

The network uses:

1. A 3x3 stem convolution for 32x32 images.
2. Four residual stages.
3. Batch normalization and ReLU activations.
4. Global average pooling.
5. A final linear classifier.

This is a CIFAR-style ResNet rather than an ImageNet-style version.

### 5.2 WideResNet

`src/models/wideresnet.py` implements WideResNet for CIFAR.

Key characteristics:

1. The depth must satisfy `6n + 4`.
2. The default configuration is WideResNet-28-2.
3. The channel progression is `16 -> 16k -> 32k -> 64k`.
4. Dropout is supported inside the residual blocks.
5. The model uses global average pooling and a linear classifier.

### 5.3 Model builder

`src/models/builder.py` dispatches model construction from configuration.

It supports:

1. `resnet18`.
2. `resnet34`.
3. `wideresnet`.

The builder is used by the training scripts so experiments can be configured from YAML without changing code.

## 6. Supervised Training

### 6.1 Supervised trainer

`src/training/supervised.py` defines `SupervisedTrainer`.

It supports three supervised modes:

1. Standard cross-entropy training.
2. Mixup.
3. CutMix.

### 6.2 Optimizer and scheduler

The supervised trainer uses:

1. SGD optimizer.
2. Momentum of 0.9 by default.
3. Weight decay of 5e-4 by default.
4. CosineAnnealingLR scheduler.

### 6.3 Mixup behavior

Mixup is implemented by sampling a Beta-distributed mixing coefficient, shuffling the batch, mixing inputs linearly, and computing a convex combination of the two cross-entropy losses.

### 6.4 CutMix behavior

CutMix is implemented by:

1. Sampling a Beta-distributed mixing coefficient.
2. Computing a bounding box.
3. Replacing the patch region with a patch from a shuffled sample.
4. Recomputing the effective lambda from the cut area.
5. Using the same convex-combination loss pattern as mixup.

### 6.5 Training loop behavior

Each epoch:

1. Runs forward/backward passes on labeled data.
2. Computes training loss and training accuracy.
3. Evaluates validation data when available.
4. Evaluates test data every epoch.
5. Steps the cosine scheduler.
6. Records timing and GPU telemetry when available.

### 6.6 Checkpointing

The trainer saves:

1. A best checkpoint.
2. A last checkpoint.

The checkpoint state includes:

1. Epoch index.
2. Model weights.
3. Optimizer state.
4. Scheduler state.
5. Best accuracy.
6. Best validation loss.
7. Training history.

Checkpoint naming follows:

1. `<run_name>_best.pt`.
2. `<run_name>_last.pt`.

### 6.7 Early stop safety

The shared trainer utility `should_trigger_stop_loss()` can stop training if monitored loss exceeds a threshold after a warmup period. This is used as a guardrail for unstable low-label runs.

## 7. SSL Training Implementations

### 7.1 Pseudo-Label trainer

`src/training/ssl_pseudolabel.py` defines `PseudoLabelTrainer`.

Behavior:

1. Train on labeled batches with cross-entropy.
2. Predict pseudo-labels on weakly augmented unlabeled samples.
3. Compute confidence scores with softmax.
4. Keep only unlabeled samples above `confidence_threshold`.
5. Train on the strong unlabeled view using the filtered pseudo-labels.
6. Combine labeled and unlabeled losses.

Important parameters:

1. `confidence_threshold` defaults to 0.95.
2. `unlabeled_loss_weight` defaults to 1.0.
3. `temperature` controls optional sharpening before softmax.

The implementation tracks:

1. Labeled loss.
2. Unlabeled loss.
3. Train accuracy.
4. Pseudo-label acceptance ratio.
5. Per-class accuracy.
6. GPU memory/utilization when available.

### 7.2 FixMatch trainer

`src/training/ssl_fixmatch.py` defines `FixMatchTrainer`.

Behavior:

1. Generate pseudo-labels from the weak unlabeled view.
2. Keep only samples whose maximum probability exceeds `confidence_threshold`.
3. Train the model on the strong unlabeled view using the selected pseudo-labels.
4. Add unlabeled loss to labeled loss.

This is a simplified FixMatch-style implementation centered on confidence-thresholded consistency training.

### 7.3 MixMatch trainer

`src/training/ssl_mixmatch.py` defines `MixMatchTrainer`.

Behavior:

1. Produce unlabeled predictions from weak and strong views.
2. Average the probabilities.
3. Sharpen the averaged probabilities using `mixmatch_temperature`.
4. Convert labeled targets to one-hot vectors.
5. Concatenate labeled and unlabeled samples.
6. Apply MixUp across the combined batch.
7. Use soft cross-entropy on mixed outputs and mixed soft targets.

Important parameters:

1. `mixmatch_alpha` controls the Beta distribution for MixUp.
2. `mixmatch_temperature` controls sharpening.
3. `unlabeled_loss_weight` weights the unlabeled term.

The implementation is a practical MixMatch-style version rather than a full paper reproduction with all possible auxiliary refinements.

### 7.4 FlexMatch trainer

`src/training/ssl_flexmatch.py` defines `FlexMatchTrainer`, which inherits from `PseudoLabelTrainer`.

Behavior:

1. Compute unlabeled prediction confidences.
2. Track the confidence distribution.
3. Update an adaptive threshold from a confidence percentile.
4. Smooth the threshold with momentum.
5. Clamp the threshold to floor and ceiling values.
6. Apply the updated threshold in the pseudo-label filter.

Key parameters:

1. `confidence_floor` defaults to 0.75.
2. `confidence_ceiling` defaults to 0.99.
3. `confidence_percentile` defaults to 80.0.
4. `threshold_momentum` defaults to 0.95.

This implementation is best described as adaptive-threshold pseudo-labeling in a FlexMatch style.

## 8. Shared Training Infrastructure

### 8.1 Shared trainer utilities

`src/training/trainer_utils.py` provides:

1. `should_trigger_stop_loss()`.
2. `ensure_history_lists()`.
3. GPU telemetry collection via `nvidia-smi`.

GPU metrics are queried per epoch and return memory used and utilization percent when CUDA telemetry is available.

### 8.2 History tracking

All trainers store metrics in history dictionaries. Common fields include:

1. Train loss.
2. Train accuracy.
3. Validation loss.
4. Validation accuracy.
5. Test loss.
6. Test accuracy.
7. Per-class accuracy.
8. Epoch time.
9. GPU memory used.
10. GPU utilization.

SSL trainers additionally track:

1. Labeled loss.
2. Unlabeled loss.
3. Pseudo-label acceptance ratio.

## 9. Configuration System

### 9.1 Dataclass-based config model

`src/utils/config.py` defines a structured configuration system using dataclasses:

1. `DatasetConfig`.
2. `ModelConfig`.
3. `TrainingConfig`.
4. `SSLConfig`.
5. `SystemConfig`.
6. `ExperimentConfig`.

### 9.2 YAML loading

The project loads YAML into `ExperimentConfig` with:

1. `load_config()`.
2. `ExperimentConfig.from_yaml()`.

This allows one YAML file to control dataset, model, training, SSL, and system settings.

### 9.3 Main config values

Common defaults and important options include:

1. Dataset name and class count.
2. Model architecture, depth, widen factor, and dropout.
3. Epochs, batch size, learning rate, momentum, and weight decay.
4. Validation split.
5. Stop-loss threshold and warmup.
6. SSL algorithm selection.
7. Labels per class or label fraction.
8. Unlabeled batch size.
9. Confidence threshold.
10. Strong augmentation toggle.
11. Temperature and MixMatch-specific parameters.
12. Device, checkpoint directory, log directory, and experiment directory.

## 10. Logging and Visualization

### 10.1 Logging

`src/utils/logging_utils.py` creates a logger that writes to:

1. Console.
2. A timestamped log file.

The logger is used by the training scripts and comparison script to capture run metadata and epoch-by-epoch metrics.

### 10.2 Training curves

`src/utils/visualization.py` generates:

1. Accuracy curves.
2. Loss curves.
3. Per-class comparison plots.
4. Confusion matrices.
5. Reliability diagrams.

### 10.3 Calibration metric

The visualization utilities implement expected calibration error (ECE). The reliability diagram function plots model confidence against empirical accuracy and returns the ECE value.

## 11. Entry Point Scripts

### 11.1 `train_supervised.py`

This is the original supervised baseline script. It supports:

1. Dataset selection.
2. Model selection.
3. Learning rate and optimizer settings.
4. Mixup or CutMix as supervised variants.
5. Training on full CIFAR-10 or CIFAR-100.

It is the older, argument-driven entry point.

### 11.2 `train_supervised_yaml.py`

This is the YAML-based supervised training script.

It:

1. Loads an experiment config.
2. Builds the model from configuration.
3. Trains using `SupervisedTrainer`.
4. Saves the config alongside the checkpoint.
5. Writes training plots to the log directory.

### 11.3 `train_supervised_limited.py`

This script trains the supervised baseline on a limited labeled subset.

It reuses the SSL data loader but only consumes the labeled portion, which makes it suitable for fair comparisons against SSL methods under the same label budget.

### 11.4 `train_ssl.py`

This is the main SSL training entry point.

It:

1. Loads the YAML config.
2. Applies CLI overrides for device or epochs.
3. Creates labeled, unlabeled, validation, and test loaders.
4. Builds the model.
5. Dispatches to the configured SSL trainer.
6. Optionally resumes from a checkpoint.
7. Saves the config and plots after training.

The SSL algorithm is chosen from `fixmatch`, `mixmatch`, `flexmatch`, or the default pseudo-label path.

### 11.5 `train_fixmatch.py`

This is a thin wrapper around `train_ssl.py` that forces the default configuration to the FixMatch YAML file.

### 11.6 `train_cli_docker.py`

This is an interactive Docker launcher.

It:

1. Prompts for algorithm, dataset, model, labels per class, epochs, and learning rate.
2. Generates a YAML config on the fly.
3. Writes the generated config to `configs/generated/`.
4. Launches a Docker container to run either supervised or SSL training.

This script is intended for reproducible containerized runs.

### 11.7 `compare_methods.py`

This is the main comparison and analysis script.

It can:

1. Train or load a supervised model.
2. Train or load an SSL model.
3. Collect predictions for evaluation.
4. Generate comparison plots.
5. Save confusion matrices and calibration diagrams.
6. Save comparison metrics to JSON.

It is the script that supports side-by-side thesis figures and benchmark comparisons.

### 11.8 `verify_installation.py`

This is an environment sanity-check script.

It verifies:

1. Core dependencies such as `torch` and `torchvision`.
2. Project module imports.
3. Config loading.
4. Model building.
5. CUDA availability.

## 12. Configuration Files

The repository contains YAML configs for supervised and SSL experiments, plus benchmark-specific variants.

Representative files include:

1. `configs/supervised_cifar10.yaml`.
2. `configs/supervised_limited_cifar10.yaml`.
3. `configs/ssl_pseudolabel_cifar10.yaml`.
4. `configs/ssl_fixmatch_cifar10.yaml`.
5. `configs/ssl_mixmatch_cifar10.yaml`.
6. `configs/ssl_flexmatch_cifar10.yaml`.
7. Benchmark configs under `configs/benchmarks/` for 250, 1000, and 4000 label runs.

Typical defaults in the shipped configs:

1. Supervised baseline uses ResNet-18, 200 epochs, batch size 128, learning rate 0.1.
2. SSL experiments use WideResNet-28-2, 300 epochs, batch size 64, learning rate 0.03.
3. SSL experiments usually enable a 10% validation split on the labeled subset.
4. SSL experiments usually use 250 labels per class for CIFAR-10 benchmark runs, with benchmark variants covering other budgets.

## 13. Experiment Outputs

The project produces the following artifacts:

1. Checkpoints in `checkpoints/<run_name>/`.
2. Logs in `logs/<run_name>/`.
3. Plots generated from training history.
4. Comparison artifacts in `experiments/comparison/<timestamp>/`.
5. JSON summaries of comparison results.

Comparison runs can include:

1. Accuracy curves.
2. Loss curves.
3. Per-class accuracy plots.
4. Confusion matrices.
5. Reliability diagrams.

## 14. Reproducibility Controls

Reproducibility is handled through:

1. Fixed random seeds.
2. YAML configs.
3. Checkpoint saving and resuming.
4. Timestamped logs.
5. Stratified splits.
6. Deterministic CUDA settings in the seeding helper when CUDA is available.

## 15. What Is Implemented Versus What Is Not

### Implemented

1. Supervised CIFAR training.
2. Limited-label supervised training.
3. Pseudo-Label SSL.
4. FixMatch-style SSL.
5. MixMatch-style SSL.
6. FlexMatch-style adaptive threshold SSL.
7. Config-driven experiment management.
8. Checkpointing and resume support.
9. Logging and plotting.
10. Comparison and calibration analysis.
11. Docker-based run orchestration.

### Not present as a full system feature

1. A separate model zoo beyond the CIFAR ResNet and WideResNet variants.
2. A full web UI or dashboard.
3. Distributed multi-GPU training orchestration.
4. Automated hyperparameter search.
5. A complete reproduction harness for the original SSL papers.

## 16. Important Caveats For Thesis Writing

These are the main interpretation points the thesis should respect:

1. The implemented SSL methods are practical project implementations, not guaranteed one-to-one reproductions of the original papers.
2. The code uses the same general backbone and training framework across comparisons to make the benchmark fair at the project level.
3. Low-label runs are more sensitive to threshold choice, split randomness, and checkpoint selection.
4. The comparison results in the thesis should be framed as project outcomes under this local protocol, not as canonical benchmark replications.
5. The codebase now includes MixMatch and FlexMatch implementations, even though older repository text may describe them as planned.

## 17. Short Reference of the Main Files

1. `src/data/cifar.py` - CIFAR loading, augmentations, stratified splits, SSL dataset wrappers.
2. `src/models/resnet.py` - CIFAR ResNet implementations.
3. `src/models/wideresnet.py` - WideResNet implementation.
4. `src/models/builder.py` - Model factory.
5. `src/training/supervised.py` - Supervised trainer with mixup and cutmix.
6. `src/training/ssl_pseudolabel.py` - Pseudo-Label trainer.
7. `src/training/ssl_fixmatch.py` - FixMatch-style trainer.
8. `src/training/ssl_mixmatch.py` - MixMatch-style trainer.
9. `src/training/ssl_flexmatch.py` - FlexMatch-style trainer.
10. `src/utils/config.py` - YAML/dataclass config system.
11. `src/utils/visualization.py` - plots and calibration metrics.
12. `src/utils/logging_utils.py` - logging setup.
13. `train_supervised_yaml.py` - YAML supervised entry point.
14. `train_supervised_limited.py` - limited-label supervised entry point.
15. `train_ssl.py` - main SSL entry point.
16. `compare_methods.py` - comparison and evaluation runner.
17. `train_cli_docker.py` - interactive Docker launcher.

## 18. Thesis-Writing Guidance

If another AI uses this document to help write the thesis, the safest framing is:

1. Describe the project as a modular PyTorch CIFAR benchmark framework.
2. State that it compares supervised training with multiple SSL methods under label budgets.
3. Emphasize the shared backbone, split protocol, and evaluation pipeline.
4. Present the SSL methods as implemented project variants of Pseudo-Label, FixMatch, MixMatch, and FlexMatch.
5. Treat the thesis results as protocol-specific experimental outcomes.

That framing is consistent with the codebase and avoids over-claiming exact reproduction of external benchmark papers.