"""
Evaluation metrics for twin face verification.
"""

import numpy as np
import torch
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import matplotlib.pyplot as plt
from collections import defaultdict


class VerificationMetrics:
    """Comprehensive metrics for face verification."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.distances = []
        self.labels = []
        self.predictions = []
        self.pair_types = []  # 0: same, 1: twin, 2: other
        
    def update(self, embeddings1, embeddings2, labels, pair_types=None):
        """
        Update metrics with a batch of embeddings.
        
        Args:
            embeddings1: First set of embeddings [batch_size, feature_dim]
            embeddings2: Second set of embeddings [batch_size, feature_dim]
            labels: Ground truth labels [batch_size]
            pair_types: Type of pairs [batch_size] (0: same, 1: twin, 2: other)
        """
        # Convert to numpy if needed
        if torch.is_tensor(embeddings1):
            embeddings1 = embeddings1.cpu().numpy()
        if torch.is_tensor(embeddings2):
            embeddings2 = embeddings2.cpu().numpy()
        if torch.is_tensor(labels):
            labels = labels.cpu().numpy()
        
        # Compute distances
        distances = self._compute_distances(embeddings1, embeddings2)
        
        self.distances.extend(distances)
        self.labels.extend(labels)
        
        if pair_types is not None:
            if torch.is_tensor(pair_types):
                pair_types = pair_types.cpu().numpy()
            self.pair_types.extend(pair_types)
    
    def _compute_distances(self, emb1, emb2, metric='cosine'):
        """Compute pairwise distances between embeddings."""
        if metric == 'cosine':
            # Cosine distance = 1 - cosine similarity
            similarities = np.sum(emb1 * emb2, axis=1) / (
                np.linalg.norm(emb1, axis=1) * np.linalg.norm(emb2, axis=1)
            )
            distances = 1 - similarities
        elif metric == 'euclidean':
            distances = np.linalg.norm(emb1 - emb2, axis=1)
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
        
        return distances.tolist()
    
    def compute_roc(self):
        """Compute ROC curve and AUC."""
        if not self.distances or not self.labels:
            return None, None, None
        
        # Convert same/different labels to binary
        # Label 0 (same person) -> positive class (1)
        # Labels 1,2 (twin, other) -> negative class (0)
        binary_labels = [1 if label == 0 else 0 for label in self.labels]
        
        # For verification, lower distance should indicate same person
        # So we use negative distances as scores
        scores = [-d for d in self.distances]
        
        fpr, tpr, thresholds = roc_curve(binary_labels, scores)
        auc_score = auc(fpr, tpr)
        
        return fpr, tpr, auc_score
    
    def compute_eer(self):
        """Compute Equal Error Rate."""
        fpr, tpr, _ = self.compute_roc()
        if fpr is None:
            return None
        
        # EER is the point where FPR = 1 - TPR (or FPR + TPR = 1)
        fnr = 1 - tpr
        eer_idx = np.argmin(np.abs(fpr - fnr))
        eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
        
        return eer
    
    def compute_accuracy_at_threshold(self, threshold):
        """Compute accuracy at a specific distance threshold."""
        if not self.distances or not self.labels:
            return None
        
        # Predict same person if distance < threshold
        predictions = [1 if d < threshold else 0 for d in self.distances]
        
        # Convert labels to binary (same vs different)
        binary_labels = [1 if label == 0 else 0 for label in self.labels]
        
        accuracy = accuracy_score(binary_labels, predictions)
        return accuracy
    
    def find_optimal_threshold(self):
        """Find optimal threshold that maximizes accuracy."""
        if not self.distances:
            return None
        
        thresholds = np.linspace(0, max(self.distances), 1000)
        best_acc = 0
        best_threshold = 0
        
        for threshold in thresholds:
            acc = self.compute_accuracy_at_threshold(threshold)
            if acc > best_acc:
                best_acc = acc
                best_threshold = threshold
        
        return best_threshold, best_acc
    
    def compute_detailed_metrics(self):
        """Compute detailed metrics by pair type."""
        if not self.pair_types:
            return self._compute_overall_metrics()
        
        metrics = {}
        
        # Overall metrics
        metrics['overall'] = self._compute_overall_metrics()
        
        # Metrics by pair type
        pair_type_names = {0: 'same', 1: 'twin', 2: 'other'}
        
        for pair_type in [0, 1, 2]:
            if pair_type in self.pair_types:
                # Filter data for this pair type
                type_distances = [d for d, t in zip(self.distances, self.pair_types) if t == pair_type]
                type_labels = [l for l, t in zip(self.labels, self.pair_types) if t == pair_type]
                
                if type_distances:
                    type_metrics = self._compute_metrics_for_subset(type_distances, type_labels)
                    metrics[pair_type_names[pair_type]] = type_metrics
        
        return metrics
    
    def _compute_overall_metrics(self):
        """Compute overall metrics."""
        if not self.distances:
            return {}
        
        fpr, tpr, auc_score = self.compute_roc()
        eer = self.compute_eer()
        optimal_threshold, optimal_acc = self.find_optimal_threshold()
        
        return {
            'auc': auc_score,
            'eer': eer,
            'optimal_threshold': optimal_threshold,
            'optimal_accuracy': optimal_acc,
            'mean_distance': np.mean(self.distances),
            'std_distance': np.std(self.distances),
            'num_pairs': len(self.distances)
        }
    
    def _compute_metrics_for_subset(self, distances, labels):
        """Compute metrics for a subset of data."""
        # For pair type analysis, we compute statistics
        return {
            'mean_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'min_distance': np.min(distances),
            'max_distance': np.max(distances),
            'num_pairs': len(distances)
        }


class TwinSpecificMetrics:
    """Metrics specifically designed for twin face verification."""
    
    def __init__(self):
        self.same_distances = []
        self.twin_distances = []
        self.other_distances = []
    
    def update(self, distances, pair_types):
        """Update with distances and pair types."""
        if torch.is_tensor(distances):
            distances = distances.cpu().numpy()
        if torch.is_tensor(pair_types):
            pair_types = pair_types.cpu().numpy()
        
        for dist, pair_type in zip(distances, pair_types):
            if pair_type == 0:  # same person
                self.same_distances.append(dist)
            elif pair_type == 1:  # twin
                self.twin_distances.append(dist)
            elif pair_type == 2:  # other person
                self.other_distances.append(dist)
    
    def compute_twin_separation_metrics(self):
        """Compute metrics for how well the model separates twins."""
        if not self.same_distances or not self.twin_distances:
            return {}
        
        same_mean = np.mean(self.same_distances)
        twin_mean = np.mean(self.twin_distances)
        other_mean = np.mean(self.other_distances) if self.other_distances else None
        
        # Twin separation ratio: how much larger twin distances are vs same distances
        twin_separation = twin_mean / same_mean if same_mean > 0 else float('inf')
        
        # Overlap analysis
        same_max = np.max(self.same_distances)
        twin_min = np.min(self.twin_distances)
        overlap_ratio = max(0, same_max - twin_min) / (twin_mean - same_mean) if twin_mean > same_mean else 1.0
        
        metrics = {
            'same_distance_mean': same_mean,
            'same_distance_std': np.std(self.same_distances),
            'twin_distance_mean': twin_mean,
            'twin_distance_std': np.std(self.twin_distances),
            'twin_separation_ratio': twin_separation,
            'overlap_ratio': overlap_ratio,
        }
        
        if other_mean is not None:
            metrics.update({
                'other_distance_mean': other_mean,
                'other_distance_std': np.std(self.other_distances),
                'other_separation_ratio': other_mean / same_mean if same_mean > 0 else float('inf')
            })
        
        return metrics
    
    def plot_distance_distributions(self, save_path=None):
        """Plot distance distributions for different pair types."""
        plt.figure(figsize=(12, 8))
        
        # Plot histograms
        bins = np.linspace(0, max(max(self.same_distances, default=0), 
                                 max(self.twin_distances, default=0),
                                 max(self.other_distances, default=0)), 50)
        
        if self.same_distances:
            plt.hist(self.same_distances, bins=bins, alpha=0.7, label='Same Person', color='green')
        
        if self.twin_distances:
            plt.hist(self.twin_distances, bins=bins, alpha=0.7, label='Twin Pairs', color='red')
        
        if self.other_distances:
            plt.hist(self.other_distances, bins=bins, alpha=0.7, label='Other Persons', color='blue')
        
        plt.xlabel('Distance')
        plt.ylabel('Frequency')
        plt.title('Distance Distributions by Pair Type')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt


def compute_identification_metrics(query_embeddings, gallery_embeddings, 
                                 query_labels, gallery_labels, top_k=[1, 5, 10]):
    """
    Compute identification metrics (CMC curve).
    
    Args:
        query_embeddings: Query set embeddings [N_q, feature_dim]
        gallery_embeddings: Gallery set embeddings [N_g, feature_dim]
        query_labels: Query labels [N_q]
        gallery_labels: Gallery labels [N_g]
        top_k: List of k values for top-k accuracy
    
    Returns:
        Dictionary with identification metrics
    """
    if torch.is_tensor(query_embeddings):
        query_embeddings = query_embeddings.cpu().numpy()
    if torch.is_tensor(gallery_embeddings):
        gallery_embeddings = gallery_embeddings.cpu().numpy()
    
    # Compute similarity matrix
    similarities = cosine_similarity(query_embeddings, gallery_embeddings)
    
    # For each query, rank gallery images by similarity
    results = {}
    
    for k in top_k:
        correct = 0
        
        for i, query_label in enumerate(query_labels):
            # Get similarities for this query
            query_similarities = similarities[i]
            
            # Get top-k most similar gallery images
            top_k_indices = np.argsort(query_similarities)[::-1][:k]
            top_k_labels = [gallery_labels[idx] for idx in top_k_indices]
            
            # Check if query label is in top-k
            if query_label in top_k_labels:
                correct += 1
        
        accuracy = correct / len(query_labels)
        results[f'top_{k}_accuracy'] = accuracy
    
    return results


def plot_roc_curve(fpr, tpr, auc_score, save_path=None):
    """Plot ROC curve."""
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return plt
