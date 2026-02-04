"""
Supervised learning trainer.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import time


class SupervisedTrainer:
    """
    Supervised training loop for image classification.
    """
    
    def __init__(self, model, train_loader, test_loader, device='cuda', 
                 learning_rate=0.1, momentum=0.9, weight_decay=5e-4):
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
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer - using SGD with momentum as is standard for CIFAR
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler - cosine annealing
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
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch_idx, (inputs, targets) in enumerate(pbar):
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
            
            # Update progress bar
            pbar.set_postfix({
                'loss': train_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
        
        epoch_loss = train_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def test(self):
        """Evaluate on test set."""
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
        """
        Train the model for a specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        print(f"Training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        best_acc = 0
        
        for epoch in range(num_epochs):
            start_time = time.time()
            
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
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
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.2f}%")
            print(f"Epoch Time: {epoch_time:.2f}s")
            
            # Track best accuracy (no checkpoint saving)
            if test_acc > best_acc:
                best_acc = test_acc
        
        print(f"\nTraining completed!")
        print(f"Best Test Accuracy: {best_acc:.2f}%")
        
        return self.history
