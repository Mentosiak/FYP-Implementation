"""FixMatch semi-supervised trainer.

Initial implementation based on:
Sohn et al. (2020) - "FixMatch: Simplifying Semi-Supervised Learning with
Consistency and Confidence"
"""

from __future__ import annotations
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


class FixMatchTrainer:
    """Initial FixMatch training loop for image classification."""

    def __init__(
        self,
        model: nn.Module,
        labeled_loader: DataLoader,
        unlabeled_loader: DataLoader,
        test_loader: DataLoader,
        val_loader: DataLoader | None = None,
        device: str = 'cuda',
        learning_rate: float = 0.03,
        momentum: float = 0.9,
        weight_decay: float = 5e-4,
        confidence_threshold: float = 0.95,
        unlabeled_loss_weight: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
        run_name: str = 'ssl_fixmatch',
        logger: logging.Logger | None = None,
        save_best: bool = True,
        save_last: bool = True,
        num_classes: int = 10,
    ):
        self.model = model.to(device)
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.unlabeled_loss_weight = unlabeled_loss_weight
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.logger = logger or logging.getLogger(__name__)
        self.save_best = save_best
        self.save_last = save_last
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=300)

        self.history = {
            'train_loss': [],
            'labeled_loss': [],
            'unlabeled_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'pseudo_label_ratio': [],
            'epoch_time': [],
            'per_class_acc': [],
        }

        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
        }

        if is_best and self.save_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_best.pt")
            torch.save(state, best_path)
            self.logger.info("Saved best checkpoint: %s", best_path)

        if is_last and self.save_last:
            last_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_last.pt")
            torch.save(state, last_path)
            self.logger.info("Saved final checkpoint: %s", last_path)

    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        self.logger.info("Loading checkpoint from %s", checkpoint_path)
        # These checkpoints are produced by this project and include optimizer/history state.
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state'])

        if load_optimizer and 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
            if 'scheduler_state' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state'])

        if 'best_acc' in checkpoint:
            self.best_acc = checkpoint['best_acc']
        if 'best_val_loss' in checkpoint:
            self.best_val_loss = checkpoint['best_val_loss']
        if 'history' in checkpoint:
            self.history = checkpoint['history']

        start_epoch = checkpoint.get('epoch', 0) + 1
        self.logger.info("Loaded checkpoint from epoch %d", checkpoint.get('epoch', 0))
        return start_epoch

    def train_epoch(self):
        self.model.train()

        labeled_loss_sum = 0.0
        unlabeled_loss_sum = 0.0
        total_loss_sum = 0.0
        correct = 0
        total = 0

        pseudo_label_count = 0
        unlabeled_total = 0

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        num_iterations = max(len(self.labeled_loader), len(self.unlabeled_loader))

        for _ in range(num_iterations):
            try:
                inputs_labeled, targets_labeled = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                inputs_labeled, targets_labeled = next(labeled_iter)

            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

            inputs_labeled = inputs_labeled.to(self.device)
            targets_labeled = targets_labeled.to(self.device)

            if isinstance(unlabeled_batch, (tuple, list)) and len(unlabeled_batch) == 2:
                inputs_unlabeled_weak, inputs_unlabeled_strong = unlabeled_batch
                inputs_unlabeled_weak = inputs_unlabeled_weak.to(self.device)
                inputs_unlabeled_strong = inputs_unlabeled_strong.to(self.device)
            else:
                if isinstance(unlabeled_batch, torch.Tensor):
                    inputs_unlabeled_weak = unlabeled_batch.to(self.device)
                else:
                    inputs_unlabeled_weak = unlabeled_batch[0].to(self.device)
                inputs_unlabeled_strong = inputs_unlabeled_weak

            self.optimizer.zero_grad()

            logits_labeled = self.model(inputs_labeled)
            loss_labeled = self.criterion(logits_labeled, targets_labeled)

            with torch.no_grad():
                logits_unlabeled_weak = self.model(inputs_unlabeled_weak)
                probs_unlabeled_weak = torch.softmax(logits_unlabeled_weak, dim=1)
                max_probs, pseudo_labels = torch.max(probs_unlabeled_weak, dim=1)
                confidence_mask = max_probs >= self.confidence_threshold

            if confidence_mask.sum() > 0:
                logits_unlabeled_strong = self.model(inputs_unlabeled_strong)
                loss_unlabeled = self.criterion(
                    logits_unlabeled_strong[confidence_mask],
                    pseudo_labels[confidence_mask],
                )
            else:
                loss_unlabeled = torch.tensor(0.0, device=self.device)

            loss = loss_labeled + self.unlabeled_loss_weight * loss_unlabeled
            loss.backward()
            self.optimizer.step()

            labeled_loss_sum += loss_labeled.item()
            unlabeled_loss_sum += float(loss_unlabeled.item())
            total_loss_sum += loss.item()

            _, predicted = logits_labeled.max(1)
            total += targets_labeled.size(0)
            correct += predicted.eq(targets_labeled).sum().item()

            pseudo_label_count += confidence_mask.sum().item()
            unlabeled_total += confidence_mask.size(0)

        epoch_loss = total_loss_sum / num_iterations
        epoch_labeled_loss = labeled_loss_sum / num_iterations
        epoch_unlabeled_loss = unlabeled_loss_sum / num_iterations
        epoch_acc = 100.0 * correct / total
        pseudo_ratio = pseudo_label_count / unlabeled_total if unlabeled_total > 0 else 0.0
        return epoch_loss, epoch_labeled_loss, epoch_unlabeled_loss, epoch_acc, pseudo_ratio

    def validate(self):
        if self.val_loader is None:
            return None, None

        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        return val_loss / len(self.val_loader), 100.0 * correct / total

    def test(self):
        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0

        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                for i in range(self.num_classes):
                    class_mask = targets == i
                    class_total[i] += class_mask.sum().item()
                    class_correct[i] += (predicted[class_mask] == i).sum().item()

        per_class_acc = {
            f'class_{i}': (100.0 * class_correct[i] / class_total[i]) if class_total[i] > 0 else 0.0
            for i in range(self.num_classes)
        }
        return test_loss / len(self.test_loader), 100.0 * correct / total, per_class_acc

    def train(self, num_epochs: int = 300, start_epoch: int = 0):
        self.logger.info("=" * 60)
        self.logger.info("Starting FixMatch SSL Training")
        self.logger.info("=" * 60)

        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()

            train_loss, labeled_loss, unlabeled_loss, train_acc, pseudo_ratio = self.train_epoch()
            val_loss, val_acc = self.validate()
            test_loss, test_acc, per_class_acc = self.test()
            self.scheduler.step()

            epoch_time = time.time() - start_time

            self.history['train_loss'].append(train_loss)
            self.history['labeled_loss'].append(labeled_loss)
            self.history['unlabeled_loss'].append(unlabeled_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['pseudo_label_ratio'].append(pseudo_ratio)
            self.history['per_class_acc'].append(per_class_acc)
            self.history['epoch_time'].append(epoch_time)

            is_best = False
            if val_loss is not None:
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_acc = test_acc
                    is_best = True
            else:
                is_best = test_acc > self.best_acc
                if is_best:
                    self.best_acc = test_acc

            if is_best:
                self._save_checkpoint(epoch, is_best=True)

            current_lr = self.optimizer.param_groups[0]['lr']
            if val_loss is not None and val_acc is not None:
                self.logger.info(
                    "Epoch [%d/%d] | Time: %.1fs | LR: %.6f | Train Loss: %.4f | Train Acc: %.2f%% | "
                    "Val Loss: %.4f | Val Acc: %.2f%% | Test Loss: %.4f | Test Acc: %.2f%% | "
                    "Pseudo Ratio: %.2f%% | Best (val): %.4f",
                    epoch + 1,
                    num_epochs,
                    epoch_time,
                    current_lr,
                    train_loss,
                    train_acc,
                    val_loss,
                    val_acc,
                    test_loss,
                    test_acc,
                    pseudo_ratio * 100.0,
                    self.best_val_loss,
                )
            else:
                self.logger.info(
                    "Epoch [%d/%d] | Time: %.1fs | LR: %.6f | Train Loss: %.4f | Train Acc: %.2f%% | "
                    "Test Loss: %.4f | Test Acc: %.2f%% | Pseudo Ratio: %.2f%% | Best: %.2f%%",
                    epoch + 1,
                    num_epochs,
                    epoch_time,
                    current_lr,
                    train_loss,
                    train_acc,
                    test_loss,
                    test_acc,
                    pseudo_ratio * 100.0,
                    self.best_acc,
                )

        if self.save_last:
            self._save_checkpoint(num_epochs - 1, is_last=True)

        self.logger.info("=" * 60)
        self.logger.info("Training Completed!")
        self.logger.info("Best Test Accuracy: %.2f%%", self.best_acc)
        self.logger.info("=" * 60)

        return self.history
