"""
Pseudo-Label Semi-Supervised Learning Trainer.

Based on "Pseudo-Label: The Simple and Efficient Semi-Supervised Learning Method 
for Deep Neural Networks" by Lee (2013).

The algorithm:
1. Train on labeled data with standard cross-entropy loss
2. Generate pseudo-labels for unlabeled data using current model predictions
3. Filter pseudo-labels by confidence threshold
4. Train on both labeled and high-confidence pseudo-labeled data
"""

from __future__ import annotations
import os
import time
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np


class PseudoLabelTrainer:
    """
    Pseudo-Label SSL trainer for image classification.
    """
    
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
        temperature: float = 1.0,
        checkpoint_dir: str = 'checkpoints',
        run_name: str = 'ssl_pseudolabel',
        logger: logging.Logger | None = None,
        save_best: bool = True,
        save_last: bool = True,
        num_classes: int = 10,
    ):
        """
        Initialize Pseudo-Label SSL trainer.
        
        Args:
            model: Neural network model
            labeled_loader: Labeled data loader
            unlabeled_loader: Unlabeled data loader (returns (weak, strong) augmented pairs)
            test_loader: Test data loader
            val_loader: Optional validation data loader (used for model selection)
            device: Device to train on
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay (L2 regularization)
            confidence_threshold: Minimum confidence to use pseudo-label (0-1)
            unlabeled_loss_weight: Weight for unlabeled loss term
            temperature: Temperature for pseudo-label sharpening (lower = sharper)
            checkpoint_dir: Directory to save checkpoints
            run_name: Name for this training run
            logger: Logger instance
            save_best: Whether to save best checkpoint
            save_last: Whether to save last checkpoint
            num_classes: Number of classes
        """
        self.model = model.to(device)
        self.labeled_loader = labeled_loader
        self.unlabeled_loader = unlabeled_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.unlabeled_loss_weight = unlabeled_loss_weight
        self.temperature = temperature
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.logger = logger or logging.getLogger(__name__)
        self.save_best = save_best
        self.save_last = save_last
        self.num_classes = num_classes
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=300
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'labeled_loss': [],
            'unlabeled_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'pseudo_label_ratio': [],  # Ratio of unlabeled samples used
            'pseudo_label_accuracy': [],  # Accuracy of pseudo-labels (if we track true labels)
            'epoch_time': [],
            'per_class_acc': [],  # Per-class accuracy on test set
        }
        
        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
    
    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False):
        """Save model checkpoint."""
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
            self.logger.info(f"Saved best checkpoint: {best_path}")
        
        if is_last and self.save_last:
            last_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_last.pt")
            torch.save(state, last_path)
            self.logger.info(f"Saved final checkpoint: {last_path}")
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
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
        self.logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 0)}")
        self.logger.info(f"Best accuracy: {self.best_acc:.2f}%")
        
        return start_epoch
    
    def _sharpen_predictions(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Sharpen predictions using temperature scaling.
        
        Args:
            logits: Model logits
            
        Returns:
            Sharpened probabilities
        """
        if self.temperature == 1.0:
            return F.softmax(logits, dim=1)
        
        # Apply temperature
        sharpened_logits = logits / self.temperature
        return F.softmax(sharpened_logits, dim=1)
    
    def train_epoch(self, epoch: int):
        """Train for one epoch using pseudo-labeling."""
        self.model.train()
        
        labeled_loss_sum = 0
        unlabeled_loss_sum = 0
        total_loss_sum = 0
        correct = 0
        total = 0
        
        # Metrics for pseudo-labels
        pseudo_label_count = 0
        unlabeled_total = 0
        
        # Create iterators
        labeled_iter = iter(self.labeled_loader)
        unlabeled_iter = iter(self.unlabeled_loader)
        
        # Determine number of iterations (match unlabeled data)
        num_iterations = max(len(self.labeled_loader), len(self.unlabeled_loader))
        
        for batch_idx in range(num_iterations):
            # Get labeled batch (cycle if necessary)
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(self.labeled_loader)
                labeled_batch = next(labeled_iter)
            
            inputs_labeled, targets_labeled = labeled_batch
            inputs_labeled = inputs_labeled.to(self.device)
            targets_labeled = targets_labeled.to(self.device)
            
            # Get unlabeled batch (cycle if necessary)
            try:
                unlabeled_batch = next(unlabeled_iter)
            except StopIteration:
                unlabeled_iter = iter(self.unlabeled_loader)
                unlabeled_batch = next(unlabeled_iter)
            
            # Unlabeled data returns (weak_aug, strong_aug)
            if isinstance(unlabeled_batch, (tuple, list)) and len(unlabeled_batch) == 2:
                inputs_unlabeled_weak, inputs_unlabeled_strong = unlabeled_batch
                inputs_unlabeled_weak = inputs_unlabeled_weak.to(self.device)
                inputs_unlabeled_strong = inputs_unlabeled_strong.to(self.device)
            else:
                # If only one augmentation provided, use it for both
                inputs_unlabeled_weak = unlabeled_batch.to(self.device) if isinstance(unlabeled_batch, torch.Tensor) else unlabeled_batch[0].to(self.device)
                inputs_unlabeled_strong = inputs_unlabeled_weak
            
            self.optimizer.zero_grad()
            
            # === Labeled loss ===
            outputs_labeled = self.model(inputs_labeled)
            loss_labeled = self.criterion(outputs_labeled, targets_labeled)
            
            # === Unlabeled loss (pseudo-labeling) ===
            # Generate pseudo-labels using weak augmentation
            with torch.no_grad():
                logits_unlabeled = self.model(inputs_unlabeled_weak)
                probs_unlabeled = self._sharpen_predictions(logits_unlabeled)
                max_probs, pseudo_labels = torch.max(probs_unlabeled, dim=1)
                
                # Create mask for high-confidence predictions
                confidence_mask = max_probs >= self.confidence_threshold
            
            # Calculate loss on strong augmentation for high-confidence samples
            if confidence_mask.sum() > 0:
                outputs_unlabeled_strong = self.model(inputs_unlabeled_strong)
                loss_unlabeled = self.criterion(
                    outputs_unlabeled_strong[confidence_mask],
                    pseudo_labels[confidence_mask]
                )
            else:
                loss_unlabeled = torch.tensor(0.0).to(self.device)
            
            # Combined loss
            loss = loss_labeled + self.unlabeled_loss_weight * loss_unlabeled
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Metrics tracking
            labeled_loss_sum += loss_labeled.item()
            unlabeled_loss_sum += loss_unlabeled.item() if isinstance(loss_unlabeled, torch.Tensor) else 0
            total_loss_sum += loss.item()
            
            _, predicted = outputs_labeled.max(1)
            total += targets_labeled.size(0)
            correct += predicted.eq(targets_labeled).sum().item()
            
            # Pseudo-label metrics
            pseudo_label_count += confidence_mask.sum().item()
            unlabeled_total += confidence_mask.size(0)
        
        # Calculate epoch metrics
        epoch_loss = total_loss_sum / num_iterations
        epoch_labeled_loss = labeled_loss_sum / num_iterations
        epoch_unlabeled_loss = unlabeled_loss_sum / num_iterations
        epoch_acc = 100. * correct / total
        pseudo_ratio = pseudo_label_count / unlabeled_total if unlabeled_total > 0 else 0
        
        return epoch_loss, epoch_labeled_loss, epoch_unlabeled_loss, epoch_acc, pseudo_ratio
    
    def test(self):
        """Evaluate on test set."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # Per-class metrics
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
                
                # Per-class accuracy
                for i in range(self.num_classes):
                    class_mask = targets == i
                    class_total[i] += class_mask.sum().item()
                    class_correct[i] += (predicted[class_mask] == i).sum().item()
        
        epoch_loss = test_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        
        # Calculate per-class accuracy
        per_class_acc = {}
        for i in range(self.num_classes):
            if class_total[i] > 0:
                per_class_acc[f'class_{i}'] = 100. * class_correct[i] / class_total[i]
            else:
                per_class_acc[f'class_{i}'] = 0.0
        
        return epoch_loss, epoch_acc, per_class_acc

    def validate(self):
        """Evaluate on validation set."""
        if self.val_loader is None:
            return None, None

        self.model.eval()
        val_loss = 0
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

        epoch_loss = val_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs: int = 300, start_epoch: int = 0):
        """
        Train for specified number of epochs.
        
        Args:
            num_epochs: Total number of epochs to train
            start_epoch: Starting epoch (for resuming training)
        """
        self.logger.info("="*60)
        self.logger.info("Starting Pseudo-Label SSL Training")
        self.logger.info("="*60)
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info(f"Labeled batches: {len(self.labeled_loader)}")
        self.logger.info(f"Unlabeled batches: {len(self.unlabeled_loader)}")
        if self.val_loader is not None:
            self.logger.info(f"Validation batches: {len(self.val_loader)}")
        self.logger.info(f"Test batches: {len(self.test_loader)}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"Unlabeled loss weight: {self.unlabeled_loss_weight}")
        self.logger.info("="*60)
        
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, labeled_loss, unlabeled_loss, train_acc, pseudo_ratio = self.train_epoch(epoch)

            # Validate
            val_loss, val_acc = self.validate()
            
            # Test
            test_loss, test_acc, per_class_acc = self.test()
            
            # Update scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Update history
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
            
            # Check if best model (prefer lowest validation loss if available)
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
            
            # Logging
            current_lr = self.optimizer.param_groups[0]['lr']
            if val_loss is not None and val_acc is not None:
                self.logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] | "
                    f"Time: {epoch_time:.1f}s | "
                    f"LR: {current_lr:.6f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Test Acc: {test_acc:.2f}% | "
                    f"Pseudo Ratio: {pseudo_ratio:.2%} | "
                    f"Best (val): {self.best_val_loss:.4f}"
                )
            else:
                self.logger.info(
                    f"Epoch [{epoch+1}/{num_epochs}] | "
                    f"Time: {epoch_time:.1f}s | "
                    f"LR: {current_lr:.6f} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Train Acc: {train_acc:.2f}% | "
                    f"Test Loss: {test_loss:.4f} | "
                    f"Test Acc: {test_acc:.2f}% | "
                    f"Pseudo Ratio: {pseudo_ratio:.2%} | "
                    f"Best: {self.best_acc:.2f}%"
                )
            
            # Log per-class accuracy every 50 epochs
            if (epoch + 1) % 50 == 0:
                self.logger.info("Per-class accuracy:")
                for cls, acc in per_class_acc.items():
                    self.logger.info(f"  {cls}: {acc:.2f}%")
        
        # Save final checkpoint
        if self.save_last:
            self._save_checkpoint(num_epochs - 1, is_last=True)
        
        self.logger.info("="*60)
        self.logger.info("Training Completed!")
        self.logger.info(f"Best Test Accuracy: {self.best_acc:.2f}%")
        self.logger.info("="*60)
        
        return self.history
