"""
Training loop and trainer class for twin face verification.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
import logging

from ..models.twin_verifier import TwinVerifier
from .losses import CombinedLoss, AdaptiveMarginLoss
from .mining import TwinHardNegativeMiner, BatchHardNegativeMiner, CurriculumMiner
from ..utils.metrics import compute_verification_metrics, compute_roc_auc
from ..utils.visualization import plot_attention_maps, plot_training_curves


class TwinVerificationTrainer:
    """
    Main trainer class for twin face verification.
    Handles training loop, validation, checkpointing, and logging.
    """
    
    def __init__(self, config, model=None, device=None):
        self.config = config
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        if model is None:
            self.model = TwinVerifier(
                backbone=config['model']['backbone'],
                feat_dim=config['model']['feat_dim'],
                num_heads=config['model']['num_heads'],
                dropout=config['model']['dropout'],
                use_gradient_checkpointing=config['optimization']['gradient_checkpointing']
            )
        else:
            self.model = model
        
        self.model.to(self.device)
        
        # Initialize loss function
        self._setup_loss_function()
        
        # Initialize optimizer and scheduler
        self._setup_optimizer()
        
        # Initialize hard negative mining
        self._setup_mining()
        
        # Mixed precision training
        if config['optimization']['mixed_precision']:
            self.scaler = GradScaler()
        else:
            self.scaler = None
        
        # Logging and checkpointing
        self._setup_logging()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_auc = 0.0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'twin_auc': []
        }
    
    def _setup_loss_function(self):
        """Setup loss function based on config."""
        self.criterion = CombinedLoss(
            twin_margin=self.config['training']['twin_margin'],
            other_margin=self.config['training']['other_margin'],
            margin_weight=self.config['training']['twin_margin_weight'],
            classification_weight=self.config['training']['difference_head_weight'],
            use_focal_loss=True
        )
        
        # Optional adaptive margin
        if self.config.get('adaptive_margin', False):
            self.adaptive_criterion = AdaptiveMarginLoss(
                initial_twin_margin=0.3,
                final_twin_margin=self.config['training']['twin_margin'],
                warmup_epochs=self.config['training']['warmup_epochs']
            )
    
    def _setup_optimizer(self):
        """Setup optimizer and learning rate scheduler."""
        # Optimizer
        if self.config['optimization']['optimizer'] == 'adamw':
            self.optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                weight_decay=self.config['training']['weight_decay']
            )
        elif self.config['optimization']['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.config['training']['learning_rate'],
                momentum=0.9,
                weight_decay=self.config['training']['weight_decay']
            )
        
        # Learning rate scheduler
        if self.config['optimization']['scheduler'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['epochs'],
                eta_min=self.config['optimization']['min_lr']
            )
        elif self.config['optimization']['scheduler'] == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config['training']['epochs'] // 3,
                gamma=0.1
            )
    
    def _setup_mining(self):
        """Setup hard negative mining strategies."""
        self.twin_miner = TwinHardNegativeMiner(
            hard_ratio=self.config['training']['hard_twin_ratio'],
            update_frequency=1000
        )
        
        self.batch_miner = BatchHardNegativeMiner(margin=0.5)
        
        self.curriculum_miner = CurriculumMiner(
            easy_epochs=2,
            medium_epochs=self.config['training']['warmup_epochs'],
            hard_epochs=self.config['training']['epochs']
        )
    
    def _setup_logging(self):
        """Setup logging and tensorboard."""
        # Create directories
        os.makedirs(self.config['logging']['checkpoint_dir'], exist_ok=True)
        os.makedirs(self.config['logging']['log_dir'], exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.config['logging']['log_dir'], 'train.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Tensorboard
        self.writer = SummaryWriter(log_dir=self.config['logging']['log_dir'])
        
        # Wandb (optional)
        if self.config['logging']['use_wandb']:
            import wandb
            wandb.init(
                project=self.config['logging']['wandb_project'],
                config=self.config
            )
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': [], 'margin': [], 'classification': []}
        
        # Regenerate pairs for this epoch
        train_loader.dataset.regenerate_pairs()
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            img1 = batch['img1'].to(self.device)
            img2 = batch['img2'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if self.scaler:
                with autocast():
                    outputs = self.model(img1, img2)
                    loss_dict = self.criterion(outputs, labels)
                    loss = loss_dict['total_loss']
                
                # Backward pass
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['optimization']['gradient_clip'] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['optimization']['gradient_clip']
                    )
                
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(img1, img2)
                loss_dict = self.criterion(outputs, labels)
                loss = loss_dict['total_loss']
                
                loss.backward()
                
                # Gradient clipping
                if self.config['optimization']['gradient_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config['optimization']['gradient_clip']
                    )
                
                self.optimizer.step()
            
            # Update mining
            with torch.no_grad():
                self.twin_miner.mine_hard_pairs(
                    outputs['embedding1'], labels, batch['img1_path']
                )
            
            # Record losses
            epoch_losses['total'].append(loss.item())
            epoch_losses['margin'].append(loss_dict['margin_loss'].item())
            epoch_losses['classification'].append(loss_dict['classification_loss'].item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })
            
            # Logging
            if batch_idx % self.config['logging']['log_interval'] == 0:
                self.writer.add_scalar('train/total_loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/margin_loss', loss_dict['margin_loss'].item(), self.global_step)
                self.writer.add_scalar('train/classification_loss', loss_dict['classification_loss'].item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', self.optimizer.param_groups[0]['lr'], self.global_step)
            
            self.global_step += 1
        
        # Calculate epoch averages
        avg_losses = {k: np.mean(v) for k, v in epoch_losses.items()}
        return avg_losses
    
    def validate(self, val_loader):
        """Validate the model."""
        self.model.eval()
        val_losses = {'total': [], 'margin': [], 'classification': []}
        
        # Collect predictions for metrics
        all_embeddings1, all_embeddings2 = [], []
        all_labels, all_predictions = [], []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc='Validation'):
                img1 = batch['img1'].to(self.device)
                img2 = batch['img2'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Forward pass
                outputs = self.model(img1, img2)
                loss_dict = self.criterion(outputs, labels)
                
                # Record losses
                val_losses['total'].append(loss_dict['total_loss'].item())
                val_losses['margin'].append(loss_dict['margin_loss'].item())
                val_losses['classification'].append(loss_dict['classification_loss'].item())
                
                # Collect for metrics
                all_embeddings1.append(outputs['embedding1'].cpu())
                all_embeddings2.append(outputs['embedding2'].cpu())
                all_labels.append(labels.cpu())
                all_predictions.append(outputs['difference_logits'].cpu())
        
        # Calculate metrics
        all_embeddings1 = torch.cat(all_embeddings1)
        all_embeddings2 = torch.cat(all_embeddings2)
        all_labels = torch.cat(all_labels)
        all_predictions = torch.cat(all_predictions)
        
        # Compute verification metrics
        metrics = compute_verification_metrics(
            all_embeddings1, all_embeddings2, all_labels
        )
        
        # Classification metrics
        classification_acc = (all_predictions.argmax(dim=1) == all_labels).float().mean().item()
        
        metrics['classification_accuracy'] = classification_acc
        avg_losses = {k: np.mean(v) for k, v in val_losses.items()}
        
        return avg_losses, metrics
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'config': self.config,
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.config['logging']['checkpoint_dir'],
            f'checkpoint_epoch_{epoch}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(
                self.config['logging']['checkpoint_dir'],
                'best_model.pth'
            )
            torch.save(checkpoint, best_path)
            self.logger.info(f"New best model saved with validation AUC: {self.best_val_auc:.4f}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_auc = checkpoint['best_val_auc']
        self.training_history = checkpoint['training_history']
        
        self.logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def train(self, train_loader, val_loader):
        """Main training loop."""
        self.logger.info("Starting training...")
        
        # Freeze backbone for initial epochs if specified
        if self.config['training']['freeze_backbone_epochs'] > 0:
            self._freeze_backbone()
        
        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            
            # Update curriculum mining
            self.curriculum_miner.update_epoch(epoch)
            
            # Unfreeze backbone after warmup
            if epoch == self.config['training']['freeze_backbone_epochs']:
                self._unfreeze_backbone()
            
            # Train epoch
            start_time = time.time()
            train_losses = self.train_epoch(train_loader)
            train_time = time.time() - start_time
            
            # Validation
            start_time = time.time()
            val_losses, val_metrics = self.validate(val_loader)
            val_time = time.time() - start_time
            
            # Update scheduler
            self.scheduler.step()
            
            # Log epoch results
            self._log_epoch_results(epoch, train_losses, val_losses, val_metrics, train_time, val_time)
            
            # Check for best model
            current_val_auc = val_metrics['overall_auc']
            is_best = current_val_auc > self.best_val_auc
            if is_best:
                self.best_val_auc = current_val_auc
            
            # Save checkpoint
            if epoch % self.config['logging']['save_interval'] == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
            
            # Update training history
            self.training_history['train_loss'].append(train_losses['total'])
            self.training_history['val_loss'].append(val_losses['total'])
            self.training_history['val_auc'].append(val_metrics['overall_auc'])
            self.training_history['twin_auc'].append(val_metrics.get('twin_auc', 0.0))
        
        self.logger.info("Training completed!")
        
        # Save final model
        self.save_checkpoint(self.config['training']['epochs'] - 1)
        
        # Plot training curves
        plot_training_curves(self.training_history, save_path=os.path.join(
            self.config['logging']['log_dir'], 'training_curves.png'
        ))
        
        self.writer.close()
    
    def _freeze_backbone(self):
        """Freeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = False
        self.logger.info("Backbone frozen for initial training")
    
    def _unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        for param in self.model.backbone.parameters():
            param.requires_grad = True
        self.logger.info("Backbone unfrozen")
    
    def _log_epoch_results(self, epoch, train_losses, val_losses, val_metrics, train_time, val_time):
        """Log results for one epoch."""
        # Console logging
        self.logger.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_losses['total']:.4f} | "
            f"Val Loss: {val_losses['total']:.4f} | "
            f"Val AUC: {val_metrics['overall_auc']:.4f} | "
            f"Twin AUC: {val_metrics.get('twin_auc', 0.0):.4f} | "
            f"Time: {train_time + val_time:.1f}s"
        )
        
        # Tensorboard logging
        self.writer.add_scalar('epoch/train_loss', train_losses['total'], epoch)
        self.writer.add_scalar('epoch/val_loss', val_losses['total'], epoch)
        self.writer.add_scalar('epoch/val_auc', val_metrics['overall_auc'], epoch)
        self.writer.add_scalar('epoch/twin_auc', val_metrics.get('twin_auc', 0.0), epoch)
        self.writer.add_scalar('epoch/classification_acc', val_metrics['classification_accuracy'], epoch)
        
        # Wandb logging
        if self.config['logging']['use_wandb']:
            import wandb
            wandb.log({
                'epoch': epoch,
                'train_loss': train_losses['total'],
                'val_loss': val_losses['total'],
                'val_auc': val_metrics['overall_auc'],
                'twin_auc': val_metrics.get('twin_auc', 0.0),
                'classification_acc': val_metrics['classification_accuracy']
            })
