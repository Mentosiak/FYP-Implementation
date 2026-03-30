"""Visualization helpers for experiment analysis."""

from __future__ import annotations

import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


_CIFAR10_CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def plot_training_curves(history: dict, save_dir: str, run_name: str):
    """Plot train/val/test accuracy and loss over epochs."""
    _ensure_dir(save_dir)

    epochs = np.arange(1, len(history.get('train_loss', [])) + 1)
    if len(epochs) == 0:
        return

    # Accuracy plot
    plt.figure(figsize=(10, 6))
    if history.get('train_acc'):
        plt.plot(epochs, history['train_acc'], label='Train Accuracy', linewidth=2)
    if history.get('val_acc') and any(v is not None for v in history['val_acc']):
        val_acc = [np.nan if v is None else v for v in history['val_acc']]
        plt.plot(epochs, val_acc, label='Validation Accuracy', linewidth=2)
    if history.get('test_acc'):
        plt.plot(epochs, history['test_acc'], label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'{run_name}: Accuracy Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{run_name}_accuracy_curves.png'), dpi=150)
    plt.close()

    # Loss plot
    plt.figure(figsize=(10, 6))
    if history.get('train_loss'):
        plt.plot(epochs, history['train_loss'], label='Train Loss', linewidth=2)
    if history.get('val_loss') and any(v is not None for v in history['val_loss']):
        val_loss = [np.nan if v is None else v for v in history['val_loss']]
        plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
    if history.get('test_loss'):
        plt.plot(epochs, history['test_loss'], label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{run_name}: Loss Curves')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{run_name}_loss_curves.png'), dpi=150)
    plt.close()


def plot_per_class_comparison(
    supervised_per_class: dict[str, float],
    ssl_per_class: dict[str, float],
    save_path: str,
    class_names: Sequence[str] | None = None,
):
    """Plot per-class accuracy bars for supervised vs SSL."""
    class_names = list(class_names) if class_names is not None else _CIFAR10_CLASS_NAMES

    class_keys = [f'class_{i}' for i in range(len(class_names))]
    supervised_vals = [float(supervised_per_class.get(k, 0.0)) for k in class_keys]
    ssl_vals = [float(ssl_per_class.get(k, 0.0)) for k in class_keys]

    x = np.arange(len(class_names))
    width = 0.38

    plt.figure(figsize=(13, 6))
    plt.bar(x - width / 2, supervised_vals, width, label='Supervised')
    plt.bar(x + width / 2, ssl_vals, width, label='SSL (Pseudo-Label/FixMatch)')

    plt.xticks(x, class_names, rotation=30, ha='right')
    plt.ylabel('Accuracy (%)')
    plt.title('Per-Class Accuracy Comparison')
    plt.ylim(0, 100)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute expected calibration error (ECE)."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        bin_weight = in_bin.mean()
        ece += bin_weight * abs(bin_acc - bin_conf)

    return float(ece)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    class_names: Sequence[str] | None = None,
    title: str = 'Confusion Matrix',
):
    """Render and save a confusion matrix heatmap."""
    if class_names is None:
        num_classes = int(max(np.max(y_true), np.max(y_pred)) + 1)
        class_names = [f'class_{i}' for i in range(num_classes)]

    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
    )
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()


def plot_reliability_diagram(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str,
    title: str = 'Reliability Diagram',
    n_bins: int = 15,
):
    """Render a reliability diagram and return ECE."""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = []
    bin_accs = []

    for i in range(n_bins):
        lower = bin_edges[i]
        upper = bin_edges[i + 1]
        in_bin = (confidences > lower) & (confidences <= upper)
        if not np.any(in_bin):
            continue
        bin_centers.append((lower + upper) / 2)
        bin_accs.append(float(accuracies[in_bin].mean()))

    ece = compute_ece(y_true, y_prob, n_bins=n_bins)

    plt.figure(figsize=(8, 6))
    sns.lineplot(x=bin_centers, y=bin_accs, marker='o', label='Model')
    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.title(f'{title} (ECE={ece:.4f})')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=160)
    plt.close()
    return ece
