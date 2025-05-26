"""
Training script for twin face verification model.
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.twin_verifier import TwinVerifier
from src.data.dataset import create_data_loaders
from src.training.trainer import TwinVerificationTrainer
from src.training.losses import CombinedLoss
from src.utils.metrics import VerificationMetrics, TwinSpecificMetrics
from src.utils.visualization import TrainingVisualizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train twin face verification model')
    
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--pairs_json', type=str, 
                       help='Path to pairs.json file (optional)')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='Output directory for models and logs')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to train on')
    parser.add_argument('--wandb', action='store_true',
                       help='Use Weights & Biases for logging')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug mode')
    
    return parser.parse_args()


def setup_device(device_arg):
    """Setup compute device."""
    if device_arg == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"Using CUDA device: {torch.cuda.get_device_name()}")
        else:
            device = torch.device('cpu')
            print("Using CPU device")
    else:
        device = torch.device(device_arg)
        print(f"Using {device_arg} device")
    
    return device


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def create_model(config, device):
    """Create the twin verification model."""
    model_config = config['model']
    
    model = TwinVerifier(
        backbone=model_config['backbone'],
        embedding_dim=model_config['embedding_dim'],
        attention_type=model_config.get('attention_type', 'cbam'),
        dropout=model_config.get('dropout', 0.5),
        pretrained=model_config.get('pretrained', True)
    )
    
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model created:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_loss_function(config):
    """Create the loss function."""
    loss_config = config['training']['loss']
    
    if loss_config['type'] == 'combined':
        loss_fn = CombinedLoss(
            margin=loss_config.get('margin', 1.0),
            twin_weight=loss_config.get('twin_weight', 2.0),
            same_weight=loss_config.get('same_weight', 1.0),
            other_weight=loss_config.get('other_weight', 1.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_config['type']}")
    
    return loss_fn


def setup_wandb(config, args):
    """Setup Weights & Biases logging."""
    if args.wandb:
        # Create run name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"twin_verification_{timestamp}"
        
        wandb.init(
            project="twin-face-verification",
            name=run_name,
            config=config,
            tags=["twin-verification", config['model']['backbone']]
        )
        
        print("Weights & Biases logging enabled")
    else:
        print("Weights & Biases logging disabled")


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Update config with command line arguments
    config['data']['dataset_path'] = args.data_path
    if args.pairs_json:
        config['data']['pairs_json_path'] = args.pairs_json
    
    # Setup device
    device = setup_device(args.device)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Setup logging
    setup_wandb(config, args)
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, test_loader, train_dataset = create_data_loaders(config)
    
    print(f"Data loaded:")
    print(f"  Train samples per epoch: {len(train_loader.dataset)}")
    print(f"  Validation samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Twin pairs in training: {len(train_dataset.split_twin_pairs)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config, device)
    
    # Create loss function
    loss_fn = create_loss_function(config)
    
    # Create optimizer
    optimizer_config = config['training']['optimizer']
    if optimizer_config['type'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=optimizer_config['lr'],
            weight_decay=optimizer_config.get('weight_decay', 1e-4)
        )
    elif optimizer_config['type'] == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=optimizer_config['lr'],
            momentum=optimizer_config.get('momentum', 0.9),
            weight_decay=optimizer_config.get('weight_decay', 1e-4)
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_config['type']}")
    
    # Create scheduler
    scheduler_config = config['training'].get('scheduler', {})
    if scheduler_config.get('type') == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['training']['epochs'],
            eta_min=scheduler_config.get('min_lr', 1e-6)
        )
    elif scheduler_config.get('type') == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1)
        )
    else:
        scheduler = None
    
    # Create trainer
    trainer = TwinVerificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        loss_fn=loss_fn,
        device=device,
        config=config,
        save_dir=args.output_dir,
        use_wandb=args.wandb
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        start_epoch = trainer.load_checkpoint(args.resume)
    
    # Training loop
    print("Starting training...")
    try:
        trainer.train(
            num_epochs=config['training']['epochs'],
            start_epoch=start_epoch
        )
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training failed with error: {e}")
        raise
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_metrics = trainer.evaluate(test_loader, save_visualizations=True)
    
    print("Test Results:")
    for metric, value in test_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    # Save final model
    final_model_path = os.path.join(args.output_dir, 'final_model.pth')
    trainer.save_checkpoint(final_model_path, is_best=False)
    
    print(f"Training completed. Final model saved to {final_model_path}")
    
    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
