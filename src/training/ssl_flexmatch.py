"""
FlexMatch-style Semi-Supervised Learning Trainer.

This implementation extends pseudo-labeling with an adaptive confidence threshold
derived from recent unlabeled prediction confidence.
"""

from __future__ import annotations

import logging
from collections import deque

import torch

from .ssl_pseudolabel import PseudoLabelTrainer


class FlexMatchTrainer(PseudoLabelTrainer):
    """FlexMatch trainer with adaptive confidence thresholding."""

    def __init__(
        self,
        *args,
        confidence_floor: float = 0.75,
        confidence_ceiling: float = 0.99,
        confidence_percentile: float = 80.0,
        threshold_momentum: float = 0.95,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.confidence_floor = max(0.0, min(confidence_floor, 1.0))
        self.confidence_ceiling = max(self.confidence_floor, min(confidence_ceiling, 1.0))
        self.confidence_percentile = max(50.0, min(confidence_percentile, 99.0))
        self.threshold_momentum = max(0.0, min(threshold_momentum, 0.999))
        self.adaptive_threshold = self.confidence_threshold
        self.conf_window = deque(maxlen=128)

    def _update_threshold(self, max_probs: torch.Tensor) -> float:
        if max_probs.numel() == 0:
            return self.adaptive_threshold

        target = torch.quantile(
            max_probs.detach(),
            q=float(self.confidence_percentile / 100.0),
        ).item()
        target = max(self.confidence_floor, min(self.confidence_ceiling, target))

        self.adaptive_threshold = (
            self.threshold_momentum * self.adaptive_threshold
            + (1.0 - self.threshold_momentum) * target
        )
        self.adaptive_threshold = max(
            self.confidence_floor,
            min(self.confidence_ceiling, self.adaptive_threshold),
        )
        return self.adaptive_threshold

    def train_epoch(self, epoch: int):
        self.model.train()

        labeled_loss_sum = 0
        unlabeled_loss_sum = 0
        total_loss_sum = 0
        correct = 0
        total = 0

        pseudo_label_count = 0
        unlabeled_total = 0

        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        num_iterations = max(len(self.labeled_loader), len(self.unlabeled_loader))

        for _ in range(num_iterations):
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                labeled_batch = next(labeled_iter)

            inputs_labeled, targets_labeled = labeled_batch
            inputs_labeled = inputs_labeled.to(self.device)
            targets_labeled = targets_labeled.to(self.device)

            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)

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

            outputs_labeled = self.model(inputs_labeled)
            loss_labeled = self.criterion(outputs_labeled, targets_labeled)

            with torch.no_grad():
                logits_unlabeled = self.model(inputs_unlabeled_weak)
                probs_unlabeled = self._sharpen_predictions(logits_unlabeled)
                max_probs, pseudo_labels = torch.max(probs_unlabeled, dim=1)
                current_threshold = self._update_threshold(max_probs)
                confidence_mask = max_probs >= current_threshold

            if confidence_mask.sum() > 0:
                outputs_unlabeled_strong = self.model(inputs_unlabeled_strong)
                loss_unlabeled = self.criterion(
                    outputs_unlabeled_strong[confidence_mask],
                    pseudo_labels[confidence_mask],
                )
            else:
                loss_unlabeled = torch.tensor(0.0).to(self.device)

            loss = loss_labeled + self.unlabeled_loss_weight * loss_unlabeled
            loss.backward()
            self.optimizer.step()

            labeled_loss_sum += loss_labeled.item()
            unlabeled_loss_sum += loss_unlabeled.item() if isinstance(loss_unlabeled, torch.Tensor) else 0
            total_loss_sum += loss.item()

            _, predicted = outputs_labeled.max(1)
            total += targets_labeled.size(0)
            correct += predicted.eq(targets_labeled).sum().item()

            pseudo_label_count += confidence_mask.sum().item()
            unlabeled_total += confidence_mask.size(0)

        epoch_loss = total_loss_sum / num_iterations
        epoch_labeled_loss = labeled_loss_sum / num_iterations
        epoch_unlabeled_loss = unlabeled_loss_sum / num_iterations
        epoch_acc = 100.0 * correct / total
        pseudo_ratio = pseudo_label_count / unlabeled_total if unlabeled_total > 0 else 0
        self.conf_window.append(self.adaptive_threshold)

        return epoch_loss, epoch_labeled_loss, epoch_unlabeled_loss, epoch_acc, pseudo_ratio

    def train(self, num_epochs: int = 300, start_epoch: int = 0):
        self.logger.info("Using FlexMatch adaptive thresholding")
        self.logger.info(
            "Adaptive threshold params: floor=%.2f ceiling=%.2f percentile=%.1f momentum=%.3f",
            self.confidence_floor,
            self.confidence_ceiling,
            self.confidence_percentile,
            self.threshold_momentum,
        )
        history = super().train(num_epochs=num_epochs, start_epoch=start_epoch)
        if self.conf_window:
            self.logger.info("Final adaptive threshold: %.4f", self.conf_window[-1])
        return history
