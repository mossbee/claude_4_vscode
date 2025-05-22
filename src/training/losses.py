"""
Custom loss functions for twin face verification.
Implements twin-aware margin loss and auxiliary classification loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TwinAwareMarginLoss(nn.Module):
    """
    Twin-aware margin loss for face verification.
    
    Implements the twin-aware margin from the blueprint:
    L = max(0, m1 + d+ - dt) + max(0, m2 + d+ - do)
    
    Where:
    - d+: distance for positive pairs (same person)
    - dt: distance for twin pairs (hard negatives)
    - do: distance for other pairs (easy negatives)
    - m1: margin for twins (typically 0.5)
    - m2: margin for others (typically 0.3)
    """
    
    def __init__(self, twin_margin=0.5, other_margin=0.3, distance_metric='l2'):
        super(TwinAwareMarginLoss, self).__init__()
        self.twin_margin = twin_margin
        self.other_margin = other_margin
        self.distance_metric = distance_metric
    
    def forward(self, embeddings1, embeddings2, labels):
        """
        Args:
            embeddings1, embeddings2: Normalized embeddings [B, D]
            labels: Labels [B] where 0=same, 1=twin, 2=other
            
        Returns:
            loss: Scalar loss value
        """
        # Compute distances
        if self.distance_metric == 'l2':
            distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        elif self.distance_metric == 'cosine':
            distances = 1 - F.cosine_similarity(embeddings1, embeddings2, dim=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
        
        # Separate distances by label type
        same_mask = (labels == 0)
        twin_mask = (labels == 1)
        other_mask = (labels == 2)
        
        # Extract distances for each type
        same_distances = distances[same_mask]
        twin_distances = distances[twin_mask]
        other_distances = distances[other_mask]
        
        loss = 0.0
        
        # Twin margin loss: push twins away from same-person pairs
        if len(same_distances) > 0 and len(twin_distances) > 0:
            # All pairs of (same, twin) distances
            same_expanded = same_distances.unsqueeze(1)  # [N_same, 1]
            twin_expanded = twin_distances.unsqueeze(0)  # [1, N_twin]
            
            twin_loss = torch.clamp(
                self.twin_margin + same_expanded - twin_expanded, min=0
            ).mean()
            loss += twin_loss
        
        # Other margin loss: standard contrastive loss
        if len(same_distances) > 0 and len(other_distances) > 0:
            same_expanded = same_distances.unsqueeze(1)  # [N_same, 1]
            other_expanded = other_distances.unsqueeze(0)  # [1, N_other]
            
            other_loss = torch.clamp(
                self.other_margin + same_expanded - other_expanded, min=0
            ).mean()
            loss += other_loss
        
        return loss


class ContrastiveLoss(nn.Module):
    """Standard contrastive loss for face verification."""
    
    def __init__(self, margin=0.5, distance_metric='l2'):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, embeddings1, embeddings2, labels):
        """
        Args:
            embeddings1, embeddings2: Embeddings [B, D]
            labels: Binary labels [B] where 0=same, 1=different
        """
        # Convert twin/other labels to binary (0=same, 1=different)
        binary_labels = (labels > 0).float()
        
        # Compute distances
        if self.distance_metric == 'l2':
            distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        elif self.distance_metric == 'cosine':
            distances = 1 - F.cosine_similarity(embeddings1, embeddings2, dim=1)
        
        # Contrastive loss
        positive_loss = (1 - binary_labels) * torch.pow(distances, 2)
        negative_loss = binary_labels * torch.pow(
            torch.clamp(self.margin - distances, min=0), 2
        )
        
        return (positive_loss + negative_loss).mean()


class TripletLoss(nn.Module):
    """Triplet loss with online hard negative mining."""
    
    def __init__(self, margin=0.3, distance_metric='l2'):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.distance_metric = distance_metric
    
    def forward(self, embeddings1, embeddings2, labels):
        """
        Args:
            embeddings1, embeddings2: Embeddings [B, D]
            labels: Labels [B] where 0=same, 1=twin, 2=other
        """
        batch_size = embeddings1.size(0)
        
        # Compute pairwise distances
        if self.distance_metric == 'l2':
            anchor_distances = torch.norm(
                embeddings1.unsqueeze(1) - embeddings2.unsqueeze(0), p=2, dim=2
            )
        elif self.distance_metric == 'cosine':
            anchor_distances = 1 - torch.mm(embeddings1, embeddings2.t())
        
        # Create triplets
        losses = []
        
        for i in range(batch_size):
            # Anchor
            anchor_label = labels[i]
            
            # Positive: same label
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
            if positive_mask.any():
                positive_distances = anchor_distances[i][positive_mask]
                hard_positive_dist = positive_distances.max()
                
                # Negative: different label
                negative_mask = (labels != anchor_label)
                if negative_mask.any():
                    negative_distances = anchor_distances[i][negative_mask]
                    hard_negative_dist = negative_distances.min()
                    
                    # Triplet loss
                    triplet_loss = torch.clamp(
                        hard_positive_dist - hard_negative_dist + self.margin, min=0
                    )
                    losses.append(triplet_loss)
        
        return torch.stack(losses).mean() if losses else torch.tensor(0.0)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        """
        Args:
            logits: Prediction logits [B, num_classes]
            targets: Ground truth labels [B]
        """
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combined loss function for twin face verification.
    
    Combines:
    - Twin-aware margin loss for embeddings
    - Cross-entropy/focal loss for auxiliary classification
    """
    
    def __init__(self, 
                 twin_margin=0.5,
                 other_margin=0.3,
                 margin_weight=1.0,
                 classification_weight=0.5,
                 use_focal_loss=True):
        super(CombinedLoss, self).__init__()
        
        self.margin_loss = TwinAwareMarginLoss(twin_margin, other_margin)
        self.margin_weight = margin_weight
        self.classification_weight = classification_weight
        
        if use_focal_loss:
            self.classification_loss = FocalLoss(alpha=1.0, gamma=2.0)
        else:
            self.classification_loss = nn.CrossEntropyLoss()
    
    def forward(self, outputs, labels):
        """
        Args:
            outputs: Dictionary containing:
                - embedding1, embedding2: Embeddings
                - difference_logits: Classification logits
            labels: Ground truth labels [B]
        """
        embeddings1 = outputs['embedding1']
        embeddings2 = outputs['embedding2']
        difference_logits = outputs['difference_logits']
        
        # Margin loss on embeddings
        margin_loss = self.margin_loss(embeddings1, embeddings2, labels)
        
        # Classification loss on difference head
        classification_loss = self.classification_loss(difference_logits, labels)
        
        # Combined loss
        total_loss = (self.margin_weight * margin_loss + 
                     self.classification_weight * classification_loss)
        
        return {
            'total_loss': total_loss,
            'margin_loss': margin_loss,
            'classification_loss': classification_loss
        }


class OnlineHardNegativeMining:
    """Online hard negative mining for improved training."""
    
    def __init__(self, hard_ratio=0.3, semi_hard_ratio=0.5):
        self.hard_ratio = hard_ratio
        self.semi_hard_ratio = semi_hard_ratio
    
    def mine_hard_negatives(self, embeddings1, embeddings2, labels, distances):
        """
        Mine hard negatives based on current embeddings.
        
        Args:
            embeddings1, embeddings2: Current embeddings
            labels: Current labels
            distances: Computed distances
            
        Returns:
            Indices of hard negative samples
        """
        batch_size = len(labels)
        
        # Separate by label type
        same_indices = torch.where(labels == 0)[0]
        twin_indices = torch.where(labels == 1)[0]
        other_indices = torch.where(labels == 2)[0]
        
        hard_indices = []
        
        # Hard twin negatives (closest to same-person pairs)
        if len(same_indices) > 0 and len(twin_indices) > 0:
            twin_distances = distances[twin_indices]
            n_hard_twins = max(1, int(len(twin_indices) * self.hard_ratio))
            
            # Sort twins by distance (ascending = hardest first)
            sorted_indices = torch.argsort(twin_distances)
            hard_twin_indices = twin_indices[sorted_indices[:n_hard_twins]]
            hard_indices.extend(hard_twin_indices.tolist())
        
        # Semi-hard other negatives
        if len(other_indices) > 0:
            other_distances = distances[other_indices]
            n_semi_hard_others = max(1, int(len(other_indices) * self.semi_hard_ratio))
            
            # Sort others by distance (descending = semi-hard)
            sorted_indices = torch.argsort(other_distances, descending=True)
            semi_hard_other_indices = other_indices[sorted_indices[:n_semi_hard_others]]
            hard_indices.extend(semi_hard_other_indices.tolist())
        
        # Always include all same-person pairs
        hard_indices.extend(same_indices.tolist())
        
        return torch.tensor(hard_indices, dtype=torch.long)


class AdaptiveMarginLoss(nn.Module):
    """Adaptive margin loss that adjusts margins based on training progress."""
    
    def __init__(self, 
                 initial_twin_margin=0.3,
                 final_twin_margin=0.7,
                 initial_other_margin=0.2,
                 final_other_margin=0.4,
                 warmup_epochs=5):
        super(AdaptiveMarginLoss, self).__init__()
        
        self.initial_twin_margin = initial_twin_margin
        self.final_twin_margin = final_twin_margin
        self.initial_other_margin = initial_other_margin
        self.final_other_margin = final_other_margin
        self.warmup_epochs = warmup_epochs
        self.current_epoch = 0
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_current_margins(self):
        """Get current margins based on training progress."""
        if self.current_epoch < self.warmup_epochs:
            progress = self.current_epoch / self.warmup_epochs
            
            twin_margin = (self.initial_twin_margin + 
                          progress * (self.final_twin_margin - self.initial_twin_margin))
            other_margin = (self.initial_other_margin + 
                           progress * (self.final_other_margin - self.initial_other_margin))
        else:
            twin_margin = self.final_twin_margin
            other_margin = self.final_other_margin
        
        return twin_margin, other_margin
    
    def forward(self, embeddings1, embeddings2, labels):
        twin_margin, other_margin = self.get_current_margins()
        
        loss_fn = TwinAwareMarginLoss(twin_margin, other_margin)
        return loss_fn(embeddings1, embeddings2, labels)
