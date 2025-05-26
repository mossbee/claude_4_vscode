"""
Visualization utilities for twin face verification.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from typing import List, Dict, Tuple, Optional


class AttentionVisualizer:
    """Visualize attention maps from the model."""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_maps = {}
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        def get_attention_hook(name):
            def hook(module, input, output):
                if hasattr(output, 'attention_weights'):
                    self.attention_maps[name] = output.attention_weights.detach()
                elif isinstance(output, tuple) and len(output) > 1:
                    # Some attention modules return (output, attention_weights)
                    if hasattr(output[1], 'shape'):
                        self.attention_maps[name] = output[1].detach()
            return hook
        
        # Register hooks for attention modules
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'cbam' in name.lower():
                module.register_forward_hook(get_attention_hook(name))
    
    def visualize_attention(self, image1, image2, save_dir=None):
        """
        Visualize attention maps for a pair of images.
        
        Args:
            image1: First image tensor [C, H, W]
            image2: Second image tensor [C, H, W]
            save_dir: Directory to save attention visualizations
        """
        self.model.eval()
        self.attention_maps.clear()
        
        # Prepare input
        if len(image1.shape) == 3:
            image1 = image1.unsqueeze(0)
        if len(image2.shape) == 3:
            image2 = image2.unsqueeze(0)
        
        images = torch.cat([image1, image2], dim=0).to(self.device)
        
        with torch.no_grad():
            _ = self.model(images)
        
        # Convert images to numpy for visualization
        img1_np = self._tensor_to_numpy(image1[0])
        img2_np = self._tensor_to_numpy(image2[0])
        
        # Create visualization
        fig, axes = plt.subplots(2, len(self.attention_maps) + 1, 
                                figsize=(4 * (len(self.attention_maps) + 1), 8))
        
        if len(self.attention_maps) == 0:
            # No attention maps captured, just show images
            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            axes[0].imshow(img1_np)
            axes[0].set_title('Image 1')
            axes[0].axis('off')
            
            axes[1].imshow(img2_np)
            axes[1].set_title('Image 2')
            axes[1].axis('off')
        else:
            # Show original images
            axes[0, 0].imshow(img1_np)
            axes[0, 0].set_title('Image 1')
            axes[0, 0].axis('off')
            
            axes[1, 0].imshow(img2_np)
            axes[1, 0].set_title('Image 2')
            axes[1, 0].axis('off')
            
            # Show attention maps
            for idx, (name, attention) in enumerate(self.attention_maps.items()):
                col = idx + 1
                
                # Process attention for both images
                if attention.shape[0] >= 2:
                    att1 = attention[0]
                    att2 = attention[1]
                    
                    # Average across channels if needed
                    if len(att1.shape) > 2:
                        att1 = att1.mean(dim=0)
                        att2 = att2.mean(dim=0)
                    
                    # Resize to image size
                    att1_resized = self._resize_attention(att1, img1_np.shape[:2])
                    att2_resized = self._resize_attention(att2, img2_np.shape[:2])
                    
                    # Overlay on images
                    overlay1 = self._overlay_attention(img1_np, att1_resized)
                    overlay2 = self._overlay_attention(img2_np, att2_resized)
                    
                    axes[0, col].imshow(overlay1)
                    axes[0, col].set_title(f'{name} - Img1')
                    axes[0, col].axis('off')
                    
                    axes[1, col].imshow(overlay2)
                    axes[1, col].set_title(f'{name} - Img2')
                    axes[1, col].axis('off')
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'attention_visualization.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig
    
    def _tensor_to_numpy(self, tensor):
        """Convert tensor to numpy array for visualization."""
        if tensor.device != 'cpu':
            tensor = tensor.cpu()
        
        # Denormalize if needed (assume ImageNet normalization)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        
        img = tensor.numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        
        return img
    
    def _resize_attention(self, attention, target_size):
        """Resize attention map to target size."""
        attention_np = attention.cpu().numpy()
        
        # Use OpenCV for resizing
        resized = cv2.resize(attention_np, (target_size[1], target_size[0]))
        
        # Normalize to [0, 1]
        resized = (resized - resized.min()) / (resized.max() - resized.min() + 1e-8)
        
        return resized
    
    def _overlay_attention(self, image, attention, alpha=0.6):
        """Overlay attention map on image."""
        # Create heatmap
        heatmap = plt.cm.jet(attention)[:, :, :3]  # Remove alpha channel
        
        # Blend with original image
        overlay = alpha * heatmap + (1 - alpha) * image
        
        return np.clip(overlay, 0, 1)


class EmbeddingVisualizer:
    """Visualize embedding spaces and clusters."""
    
    def __init__(self):
        self.embeddings = []
        self.labels = []
        self.pair_types = []
        self.image_paths = []
    
    def add_embeddings(self, embeddings, labels, pair_types=None, image_paths=None):
        """Add embeddings for visualization."""
        if torch.is_tensor(embeddings):
            embeddings = embeddings.cpu().numpy()
        
        self.embeddings.append(embeddings)
        self.labels.extend(labels)
        
        if pair_types is not None:
            self.pair_types.extend(pair_types)
        
        if image_paths is not None:
            self.image_paths.extend(image_paths)
    
    def visualize_tsne(self, perplexity=30, save_path=None):
        """Create t-SNE visualization of embeddings."""
        if not self.embeddings:
            raise ValueError("No embeddings added for visualization")
        
        # Combine all embeddings
        all_embeddings = np.vstack(self.embeddings)
        
        # Perform t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d = tsne.fit_transform(all_embeddings)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Color by identity if labels available
        if self.labels:
            unique_labels = list(set(self.labels))
            colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(self.labels) == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[colors[i]], label=f'Identity {label}', alpha=0.7)
        else:
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        
        plt.title('t-SNE Visualization of Face Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        if len(unique_labels) <= 20:  # Only show legend if not too many labels
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def visualize_twin_clusters(self, twin_pairs, save_path=None):
        """Visualize how well twins are separated in embedding space."""
        if not self.embeddings:
            raise ValueError("No embeddings added for visualization")
        
        all_embeddings = np.vstack(self.embeddings)
        
        # Perform PCA for 2D visualization
        pca = PCA(n_components=2)
        embeddings_2d = pca.fit_transform(all_embeddings)
        
        plt.figure(figsize=(12, 8))
        
        # Plot all points
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                   c='lightgray', alpha=0.5, s=20)
        
        # Highlight twin pairs
        colors = plt.cm.tab10(np.linspace(0, 1, len(twin_pairs)))
        
        for i, (twin_a, twin_b) in enumerate(twin_pairs):
            # Find embeddings for this twin pair
            twin_a_indices = [j for j, label in enumerate(self.labels) if label == twin_a]
            twin_b_indices = [j for j, label in enumerate(self.labels) if label == twin_b]
            
            if twin_a_indices and twin_b_indices:
                # Plot twin A
                plt.scatter(embeddings_2d[twin_a_indices, 0], 
                           embeddings_2d[twin_a_indices, 1],
                           c=[colors[i]], marker='o', s=100, 
                           label=f'Twin Pair {i+1} - A')
                
                # Plot twin B
                plt.scatter(embeddings_2d[twin_b_indices, 0], 
                           embeddings_2d[twin_b_indices, 1],
                           c=[colors[i]], marker='s', s=100, 
                           label=f'Twin Pair {i+1} - B')
                
                # Draw lines connecting twins
                for idx_a in twin_a_indices:
                    for idx_b in twin_b_indices:
                        plt.plot([embeddings_2d[idx_a, 0], embeddings_2d[idx_b, 0]],
                                [embeddings_2d[idx_a, 1], embeddings_2d[idx_b, 1]],
                                c=colors[i], alpha=0.3, linewidth=1)
        
        plt.title('Twin Pair Separation in Embedding Space')
        plt.xlabel('PCA Component 1')
        plt.ylabel('PCA Component 2')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt


class TrainingVisualizer:
    """Visualize training progress and metrics."""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.epochs = []
    
    def update(self, epoch, train_loss, val_loss, train_metrics=None, val_metrics=None):
        """Update training history."""
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        if train_metrics:
            self.train_metrics.append(train_metrics)
        if val_metrics:
            self.val_metrics.append(val_metrics)
    
    def plot_losses(self, save_path=None):
        """Plot training and validation losses."""
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.epochs, self.train_losses, 'b-', label='Training Loss', linewidth=2)
        plt.plot(self.epochs, self.val_losses, 'r-', label='Validation Loss', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def plot_metrics(self, metric_name, save_path=None):
        """Plot specific metrics over time."""
        if not self.train_metrics or not self.val_metrics:
            print(f"No metrics data available for {metric_name}")
            return None
        
        train_values = [m.get(metric_name, 0) for m in self.train_metrics]
        val_values = [m.get(metric_name, 0) for m in self.val_metrics]
        
        plt.figure(figsize=(10, 6))
        
        plt.plot(self.epochs, train_values, 'b-', label=f'Training {metric_name}', linewidth=2)
        plt.plot(self.epochs, val_values, 'r-', label=f'Validation {metric_name}', linewidth=2)
        
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Training and Validation {metric_name}')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt
    
    def create_training_dashboard(self, save_dir=None):
        """Create a comprehensive training dashboard."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Loss plot
        axes[0, 0].plot(self.epochs, self.train_losses, 'b-', label='Training', linewidth=2)
        axes[0, 0].plot(self.epochs, self.val_losses, 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # AUC plot
        if self.train_metrics and self.val_metrics:
            train_auc = [m.get('auc', 0) for m in self.train_metrics]
            val_auc = [m.get('auc', 0) for m in self.val_metrics]
            
            axes[0, 1].plot(self.epochs, train_auc, 'b-', label='Training', linewidth=2)
            axes[0, 1].plot(self.epochs, val_auc, 'r-', label='Validation', linewidth=2)
            axes[0, 1].set_title('AUC')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('AUC')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # EER plot
            train_eer = [m.get('eer', 0) for m in self.train_metrics]
            val_eer = [m.get('eer', 0) for m in self.val_metrics]
            
            axes[1, 0].plot(self.epochs, train_eer, 'b-', label='Training', linewidth=2)
            axes[1, 0].plot(self.epochs, val_eer, 'r-', label='Validation', linewidth=2)
            axes[1, 0].set_title('Equal Error Rate (EER)')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('EER')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Accuracy plot
            train_acc = [m.get('optimal_accuracy', 0) for m in self.train_metrics]
            val_acc = [m.get('optimal_accuracy', 0) for m in self.val_metrics]
            
            axes[1, 1].plot(self.epochs, train_acc, 'b-', label='Training', linewidth=2)
            axes[1, 1].plot(self.epochs, val_acc, 'r-', label='Validation', linewidth=2)
            axes[1, 1].set_title('Optimal Accuracy')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Accuracy')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            plt.savefig(os.path.join(save_dir, 'training_dashboard.png'), 
                       dpi=300, bbox_inches='tight')
        
        return fig


def create_pair_comparison_grid(image_pairs, predictions, labels, 
                               twin_pairs=None, save_path=None, max_pairs=16):
    """
    Create a grid showing image pairs with predictions and ground truth.
    
    Args:
        image_pairs: List of (img1_path, img2_path) tuples
        predictions: List of model predictions
        labels: List of ground truth labels
        twin_pairs: List of twin pair identifiers (optional)
        save_path: Path to save the grid
        max_pairs: Maximum number of pairs to show
    """
    n_pairs = min(len(image_pairs), max_pairs)
    n_cols = 4  # img1, img2, prediction, label
    n_rows = n_pairs
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n_pairs):
        img1_path, img2_path = image_pairs[i]
        pred = predictions[i]
        label = labels[i]
        
        # Load and display images
        try:
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            axes[i, 0].imshow(img1)
            axes[i, 0].set_title('Image 1')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(img2)
            axes[i, 1].set_title('Image 2')
            axes[i, 1].axis('off')
            
            # Show prediction
            pred_text = f"Pred: {pred:.3f}"
            axes[i, 2].text(0.5, 0.5, pred_text, transform=axes[i, 2].transAxes,
                           fontsize=14, ha='center', va='center')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
            
            # Show ground truth
            label_names = {0: 'Same', 1: 'Twin', 2: 'Other'}
            label_text = label_names.get(label, f"Label: {label}")
            color = 'green' if label == 0 else 'red' if label == 1 else 'blue'
            axes[i, 3].text(0.5, 0.5, label_text, transform=axes[i, 3].transAxes,
                           fontsize=14, ha='center', va='center', color=color)
            axes[i, 3].set_title('Ground Truth')
            axes[i, 3].axis('off')
            
        except Exception as e:
            print(f"Error loading images for pair {i}: {e}")
            for j in range(n_cols):
                axes[i, j].text(0.5, 0.5, 'Error loading image', 
                               transform=axes[i, j].transAxes,
                               ha='center', va='center')
                axes[i, j].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig
