"""
Supervised learning trainer.
"""

import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from .trainer_utils import should_trigger_stop_loss


class SupervisedTrainer:
    """
    Supervised training loop for image classification.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        val_loader=None,
        device='cuda',
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        checkpoint_dir='checkpoints',
        run_name='supervised',
        logger: logging.Logger | None = None,
        save_best: bool = True,
        save_last: bool = True,
        num_classes: int = 10,
        stop_loss_threshold: float | None = None,
        stop_loss_warmup_epochs: int = 12,
        supervised_algorithm: str = 'standard',
        mixup_alpha: float = 0.2,
        total_epochs: int = 200,
    ):
        """
        Initialize supervised trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            val_loader: Optional validation data loader (used for model selection)
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay (L2 regularization)
            checkpoint_dir: Directory to save checkpoints
            run_name: Name for this training run
            logger: Logger instance
            save_best: Whether to save best checkpoint
            save_last: Whether to save last checkpoint
            num_classes: Number of classes
            stop_loss_threshold: Stop training when monitored loss exceeds this threshold
            stop_loss_warmup_epochs: Ignore stop-loss check during first N epochs
            supervised_algorithm: Training algorithm ('standard' or 'mixup')
            mixup_alpha: Mixup beta distribution alpha
            total_epochs: Total planned epochs (used for scheduler horizon)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.logger = logger or logging.getLogger(__name__)
        self.save_best = save_best
        self.save_last = save_last
        self.num_classes = num_classes
        self.stop_loss_threshold = stop_loss_threshold
        self.stop_loss_warmup_epochs = max(0, stop_loss_warmup_epochs)
        self.supervised_algorithm = supervised_algorithm.lower()
        self.mixup_alpha = mixup_alpha
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate 
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(1, total_epochs)
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'test_loss': [],
            'test_acc': [],
            'per_class_acc': [],
            'epoch_time': []
        }

        self.best_acc = 0.0
        self.best_val_loss = float('inf')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _mixup_batch(self, inputs: torch.Tensor, targets: torch.Tensor):
        """Apply Mixup augmentation to a labeled batch."""
        if self.mixup_alpha <= 0.0:
            return inputs, targets, targets, 1.0
        lam = float(np.random.beta(self.mixup_alpha, self.mixup_alpha))
        lam = max(lam, 1.0 - lam)
        index = torch.randperm(inputs.size(0), device=inputs.device)
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
        targets_a = targets
        targets_b = targets[index]
        return mixed_inputs, targets_a, targets_b, lam

    def _mixup_criterion(self, outputs, targets_a, targets_b, lam: float):
        """Compute mixup loss as convex combination of CE losses."""
        return lam * self.criterion(outputs, targets_a) + (1.0 - lam) * self.criterion(outputs, targets_b)

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
            self.logger.info("Saved best checkpoint: %s", best_path)

        if is_last and self.save_last:
            last_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_last.pt")
            torch.save(state, last_path)
            self.logger.info("Saved final checkpoint: %s", last_path)
    
    def load_checkpoint(self, checkpoint_path: str, load_optimizer: bool = True):
        """
        Load model from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            load_optimizer: Whether to load optimizer state
            
        Returns:
            start_epoch: Epoch to resume from
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
    
    def train_epoch(self):
        # Training for one epoch
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            if self.supervised_algorithm == 'mixup':
                mixed_inputs, targets_a, targets_b, lam = self._mixup_batch(inputs, targets)
                outputs = self.model(mixed_inputs)
                loss = self._mixup_criterion(outputs, targets_a, targets_b, lam)
                with torch.no_grad():
                    clean_outputs = self.model(inputs)
                    _, predicted = clean_outputs.max(1)
            else:
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                _, predicted = outputs.max(1)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = train_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def test(self):
        """Evaluate on test set with per-class metrics."""
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        # Per-class metrics
        class_correct = np.zeros(self.num_classes)
        class_total = np.zeros(self.num_classes)
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
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
    
    def train(self, num_epochs=200, start_epoch: int = 0):
        """
        Train for specified number of epochs.
        
        Args:
            num_epochs: Total number of epochs to train
            start_epoch: Starting epoch (for resuming training)
        """
        self.logger.info("="*60)
        self.logger.info("Starting Supervised Training")
        self.logger.info("="*60)
        self.logger.info("Device: %s", self.device)
        self.logger.info("Model parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")
        self.logger.info("Supervised algorithm: %s", self.supervised_algorithm)
        if self.supervised_algorithm == 'mixup':
            self.logger.info("Mixup alpha: %.3f", self.mixup_alpha)
        if self.stop_loss_threshold is not None:
            self.logger.info(
                "Stop-loss threshold: %.4f (active after epoch %d)",
                self.stop_loss_threshold,
                self.stop_loss_warmup_epochs + 1,
            )
        self.logger.info("Training batches: %d", len(self.train_loader))
        if self.val_loader is not None:
            self.logger.info("Validation batches: %d", len(self.val_loader))
        self.logger.info("Test batches: %d", len(self.test_loader))
        self.logger.info("="*60)
        
        last_completed_epoch = start_epoch - 1
        for epoch in range(start_epoch, num_epochs):
            start_time = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc = self.validate()
            
            # Test
            test_loss, test_acc, per_class_acc = self.test()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track time
            epoch_time = time.time() - start_time
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['per_class_acc'].append(per_class_acc)
            self.history['epoch_time'].append(epoch_time)
            last_completed_epoch = epoch
            
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

            monitored_loss = val_loss if val_loss is not None else test_loss
            if should_trigger_stop_loss(
                epoch=epoch,
                monitored_loss=monitored_loss,
                threshold=self.stop_loss_threshold,
                warmup_epochs=self.stop_loss_warmup_epochs,
            ):
                self.logger.warning(
                    "Stop-loss triggered at epoch %d: monitored_loss=%.4f > threshold=%.4f",
                    epoch + 1,
                    monitored_loss,
                    self.stop_loss_threshold,
                )
                break
            
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
                    f"Best: {self.best_acc:.2f}%"
                )
            
            # Log per-class accuracy every 50 epochs
            if (epoch + 1) % 50 == 0:
                self.logger.info("Per-class accuracy:")
                for cls, acc in per_class_acc.items():
                    self.logger.info(f"  {cls}: {acc:.2f}%")
        
        if self.save_last and last_completed_epoch >= start_epoch:
            self._save_checkpoint(last_completed_epoch, is_best=False, is_last=True)

        self.logger.info("="*60)
        self.logger.info("Training Completed!")
        self.logger.info("Best Test Accuracy: %.2f%%", self.best_acc)
        self.logger.info("="*60)
        
        return self.history
