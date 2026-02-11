"""
Supervised learning trainer.
"""

import logging
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim


class SupervisedTrainer:
    """
    Supervised training loop for image classification.
    """
    
    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        device='cuda',
        learning_rate=0.1,
        momentum=0.9,
        weight_decay=5e-4,
        checkpoint_dir='checkpoints',
        run_name='supervised',
        logger: logging.Logger | None = None,
        save_best: bool = True,
        save_last: bool = True,
    ):
        """
        Initialize supervised trainer.
        
        Args:
            model: Neural network model
            train_loader: Training data loader
            test_loader: Test data loader
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            momentum: SGD momentum
            weight_decay: Weight decay (L2 regularization)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.run_name = run_name
        self.logger = logger or logging.getLogger(__name__)
        self.save_best = save_best
        self.save_last = save_last
        
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
            self.optimizer, T_max=200
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'epoch_time': []
        }

        self.best_acc = 0.0
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _save_checkpoint(self, epoch: int, is_best: bool = False, is_last: bool = False):
        state = {
            'epoch': epoch,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict(),
            'best_acc': self.best_acc,
            'history': self.history,
        }

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_best.pt")
            torch.save(state, best_path)
            self.logger.info("Saved best checkpoint: %s", best_path)

        if is_last:
            last_path = os.path.join(self.checkpoint_dir, f"{self.run_name}_last.pt")
            torch.save(state, last_path)
            self.logger.info("Saved final checkpoint: %s", last_path)
    
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
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Track metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        epoch_loss = train_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def test(self):
#Testing on test set
        self.model.eval()
        test_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(self.test_loader):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        epoch_loss = test_loss / len(self.test_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=200):
#Passing in number of epochs to train
        self.logger.info("Training on %s", self.device)
        self.logger.info("Model parameters: %s", f"{sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            self.logger.info("Epoch %d/%d", epoch + 1, num_epochs)
            self.logger.info("Learning Rate: %.6f", self.scheduler.get_last_lr()[0])
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Test
            test_loss, test_acc = self.test()
            
            # Update learning rate
            self.scheduler.step()
            
            # Track time
            epoch_time = time.time() - start_time
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['test_loss'].append(test_loss)
            self.history['test_acc'].append(test_acc)
            self.history['epoch_time'].append(epoch_time)
            
            # Print summary
            self.logger.info("Train Loss: %.4f | Train Acc: %.2f%%", train_loss, train_acc)
            self.logger.info("Test Loss: %.4f | Test Acc: %.2f%%", test_loss, test_acc)
            self.logger.info("Epoch Time: %.2fs", epoch_time)
            
            # Track best accuracy (no checkpoint saving)
            if test_acc > self.best_acc:
                self.best_acc = test_acc
                if self.save_best:
                    self._save_checkpoint(epoch, is_best=True, is_last=False)
        
        if self.save_last:
            self._save_checkpoint(num_epochs - 1, is_best=False, is_last=True)

        self.logger.info("Training completed")
        self.logger.info("Best Test Accuracy: %.2f%%", self.best_acc)
        
        return self.history
