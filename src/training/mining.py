"""
Hard negative mining strategies for twin face verification.
"""

import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict


class TwinHardNegativeMiner:
    """
    Hard negative mining specifically designed for twin pairs.
    Focuses on finding the most challenging twin pairs for training.
    """
    
    def __init__(self, 
                 hard_ratio=0.5,
                 semi_hard_ratio=0.3,
                 update_frequency=1000,
                 memory_size=10000):
        """
        Args:
            hard_ratio: Ratio of hardest twin negatives to use
            semi_hard_ratio: Ratio of semi-hard negatives to use
            update_frequency: How often to update the mining statistics
            memory_size: Size of embedding memory for mining
        """
        self.hard_ratio = hard_ratio
        self.semi_hard_ratio = semi_hard_ratio
        self.update_frequency = update_frequency
        self.memory_size = memory_size
        
        # Memory banks for embeddings and metadata
        self.embedding_memory = {}
        self.label_memory = {}
        self.path_memory = {}
        self.update_counter = 0
        
        # Mining statistics
        self.twin_pair_difficulties = defaultdict(list)
        self.hard_twin_pairs = []
        
    def update_memory(self, embeddings, labels, paths):
        """Update the embedding memory with new samples."""
        for emb, label, path in zip(embeddings, labels, paths):
            self.embedding_memory[path] = emb.detach().cpu()
            self.label_memory[path] = label.item()
            self.path_memory[path] = path
            
        # Prune memory if too large
        if len(self.embedding_memory) > self.memory_size:
            # Remove oldest entries (simple FIFO)
            oldest_keys = list(self.embedding_memory.keys())[:-self.memory_size]
            for key in oldest_keys:
                del self.embedding_memory[key]
                del self.label_memory[key]
                del self.path_memory[key]
    
    def mine_hard_pairs(self, current_embeddings, current_labels, current_paths):
        """
        Mine hard negative pairs from current batch and memory.
        
        Returns:
            Dictionary with hard negative information
        """
        self.update_counter += 1
        
        # Update memory with current batch
        self.update_memory(current_embeddings, current_labels, current_paths)
        
        # Mine hard negatives if we have enough samples
        if (self.update_counter % self.update_frequency == 0 and 
            len(self.embedding_memory) > 100):
            return self._perform_hard_mining()
        
        return {}
    
    def _perform_hard_mining(self):
        """Perform actual hard negative mining."""
        # Convert memory to tensors
        paths = list(self.embedding_memory.keys())
        embeddings = torch.stack([self.embedding_memory[p] for p in paths])
        labels = torch.tensor([self.label_memory[p] for p in paths])
        
        # Separate by label types
        same_mask = (labels == 0)
        twin_mask = (labels == 1) 
        other_mask = (labels == 2)
        
        same_embeddings = embeddings[same_mask]
        twin_embeddings = embeddings[twin_mask]
        other_embeddings = embeddings[other_mask]
        
        same_paths = [p for i, p in enumerate(paths) if same_mask[i]]
        twin_paths = [p for i, p in enumerate(paths) if twin_mask[i]]
        other_paths = [p for i, p in enumerate(paths) if other_mask[i]]
        
        hard_pairs = {
            'hard_twins': [],
            'semi_hard_others': [],
            'easy_positives': []
        }
        
        # Mine hard twin pairs
        if len(same_embeddings) > 0 and len(twin_embeddings) > 0:
            hard_twins = self._mine_hard_twins(
                same_embeddings, twin_embeddings, same_paths, twin_paths
            )
            hard_pairs['hard_twins'] = hard_twins
        
        # Mine semi-hard others
        if len(same_embeddings) > 0 and len(other_embeddings) > 0:
            semi_hard_others = self._mine_semi_hard_others(
                same_embeddings, other_embeddings, same_paths, other_paths
            )
            hard_pairs['semi_hard_others'] = semi_hard_others
        
        return hard_pairs
    
    def _mine_hard_twins(self, same_embeddings, twin_embeddings, same_paths, twin_paths):
        """Mine hardest twin pairs (closest to same-person embeddings)."""
        # Compute distances between same-person and twin embeddings
        distances = torch.cdist(same_embeddings, twin_embeddings, p=2)
        
        # Find closest twin for each same-person embedding
        min_distances, min_indices = torch.min(distances, dim=1)
        
        # Create hard twin pairs
        hard_twins = []
        for i, (dist, twin_idx) in enumerate(zip(min_distances, min_indices)):
            same_path = same_paths[i]
            twin_path = twin_paths[twin_idx]
            
            hard_twins.append({
                'same_path': same_path,
                'twin_path': twin_path,
                'distance': dist.item()
            })
        
        # Sort by distance (hardest first)
        hard_twins.sort(key=lambda x: x['distance'])
        
        # Return top hard examples
        n_hard = max(1, int(len(hard_twins) * self.hard_ratio))
        return hard_twins[:n_hard]
    
    def _mine_semi_hard_others(self, same_embeddings, other_embeddings, same_paths, other_paths):
        """Mine semi-hard other pairs (not too easy, not too hard)."""
        # Compute distances
        distances = torch.cdist(same_embeddings, other_embeddings, p=2)
        
        # Find median distance for each same-person embedding (semi-hard)
        median_distances, median_indices = torch.median(distances, dim=1)
        
        # Create semi-hard other pairs
        semi_hard_others = []
        for i, (dist, other_idx) in enumerate(zip(median_distances, median_indices)):
            same_path = same_paths[i]
            other_path = other_paths[other_idx]
            
            semi_hard_others.append({
                'same_path': same_path,
                'other_path': other_path,
                'distance': dist.item()
            })
        
        # Return random subset
        n_semi_hard = max(1, int(len(semi_hard_others) * self.semi_hard_ratio))
        return np.random.choice(semi_hard_others, n_semi_hard, replace=False).tolist()


class BatchHardNegativeMiner:
    """Hard negative mining within a single batch."""
    
    def __init__(self, margin=0.5):
        self.margin = margin
    
    def mine_batch_hard(self, embeddings1, embeddings2, labels):
        """
        Mine hard negatives within the current batch.
        
        Args:
            embeddings1, embeddings2: Batch embeddings
            labels: Batch labels
            
        Returns:
            Indices of hard negative samples
        """
        batch_size = len(labels)
        
        # Compute all pairwise distances
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        
        # Separate by label types
        same_indices = torch.where(labels == 0)[0]
        twin_indices = torch.where(labels == 1)[0]
        other_indices = torch.where(labels == 2)[0]
        
        hard_indices = []
        
        # Always include all same-person pairs
        hard_indices.extend(same_indices.tolist())
        
        # Mine hard twins (closest to same-person pairs)
        if len(same_indices) > 0 and len(twin_indices) > 0:
            twin_distances = distances[twin_indices]
            same_distances = distances[same_indices]
            
            # Find twins that are closer than the farthest same-person pair
            max_same_distance = same_distances.max()
            hard_twin_mask = twin_distances < (max_same_distance + self.margin)
            hard_twin_indices = twin_indices[hard_twin_mask]
            
            hard_indices.extend(hard_twin_indices.tolist())
        
        # Mine semi-hard others
        if len(other_indices) > 0:
            other_distances = distances[other_indices]
            
            # Select others that are not too easy (above median distance)
            median_distance = other_distances.median()
            semi_hard_mask = other_distances >= median_distance
            semi_hard_indices = other_indices[semi_hard_mask]
            
            hard_indices.extend(semi_hard_indices.tolist())
        
        return torch.tensor(hard_indices, dtype=torch.long)
    
    def compute_weights(self, embeddings1, embeddings2, labels):
        """Compute sample weights based on difficulty."""
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        
        weights = torch.ones_like(distances)
        
        # Increase weight for hard examples
        same_mask = (labels == 0)
        twin_mask = (labels == 1)
        other_mask = (labels == 2)
        
        if same_mask.any():
            same_distances = distances[same_mask]
            # Higher weight for farther same-person pairs (harder positives)
            same_weights = 1.0 + same_distances / same_distances.max()
            weights[same_mask] = same_weights
        
        if twin_mask.any():
            twin_distances = distances[twin_mask]
            # Higher weight for closer twin pairs (harder negatives)
            twin_weights = 2.0 - twin_distances / twin_distances.max()
            weights[twin_mask] = twin_weights
        
        if other_mask.any():
            other_distances = distances[other_mask]
            # Moderate weight for others
            other_weights = 1.0 + 0.5 * (1.0 - other_distances / other_distances.max())
            weights[other_mask] = other_weights
        
        return weights


class CurriculumMiner:
    """Curriculum learning for gradually increasing mining difficulty."""
    
    def __init__(self, easy_epochs=5, medium_epochs=10, hard_epochs=20):
        self.easy_epochs = easy_epochs
        self.medium_epochs = medium_epochs
        self.hard_epochs = hard_epochs
        self.current_epoch = 0
    
    def update_epoch(self, epoch):
        self.current_epoch = epoch
    
    def get_mining_strategy(self):
        """Get current mining strategy based on training progress."""
        if self.current_epoch < self.easy_epochs:
            return 'easy'
        elif self.current_epoch < self.medium_epochs:
            return 'medium'
        else:
            return 'hard'
    
    def mine_curriculum_samples(self, embeddings1, embeddings2, labels):
        """Mine samples based on current curriculum stage."""
        strategy = self.get_mining_strategy()
        distances = torch.norm(embeddings1 - embeddings2, p=2, dim=1)
        
        if strategy == 'easy':
            # Easy stage: mostly random sampling
            indices = torch.randperm(len(labels))
            return indices
        
        elif strategy == 'medium':
            # Medium stage: mix of easy and hard
            hard_miner = BatchHardNegativeMiner(margin=0.3)
            hard_indices = hard_miner.mine_batch_hard(embeddings1, embeddings2, labels)
            
            # Mix with random samples
            random_indices = torch.randperm(len(labels))[:len(labels)//2]
            combined_indices = torch.cat([hard_indices, random_indices])
            return torch.unique(combined_indices)
        
        else:  # hard
            # Hard stage: full hard negative mining
            hard_miner = BatchHardNegativeMiner(margin=0.5)
            return hard_miner.mine_batch_hard(embeddings1, embeddings2, labels)


class OnlineTripletMiner:
    """Online triplet mining for triplet loss."""
    
    def __init__(self, strategy='hardest'):
        self.strategy = strategy  # 'hardest', 'semi_hard', 'all'
    
    def mine_triplets(self, embeddings, labels):
        """
        Mine triplets from batch embeddings.
        
        Args:
            embeddings: Batch embeddings [B, D]
            labels: Batch labels [B]
            
        Returns:
            triplet_indices: (anchor, positive, negative) indices
        """
        batch_size = embeddings.size(0)
        
        # Compute pairwise distances
        distances = torch.cdist(embeddings, embeddings, p=2)
        
        triplets = []
        
        for i in range(batch_size):
            anchor_label = labels[i]
            
            # Find positive examples (same label, different index)
            positive_mask = (labels == anchor_label) & (torch.arange(batch_size) != i)
            positive_indices = torch.where(positive_mask)[0]
            
            # Find negative examples (different label)
            negative_mask = (labels != anchor_label)
            negative_indices = torch.where(negative_mask)[0]
            
            if len(positive_indices) == 0 or len(negative_indices) == 0:
                continue
            
            anchor_pos_distances = distances[i][positive_indices]
            anchor_neg_distances = distances[i][negative_indices]
            
            if self.strategy == 'hardest':
                # Hardest positive (farthest)
                hardest_pos_idx = positive_indices[torch.argmax(anchor_pos_distances)]
                # Hardest negative (closest)
                hardest_neg_idx = negative_indices[torch.argmin(anchor_neg_distances)]
                
                triplets.append((i, hardest_pos_idx.item(), hardest_neg_idx.item()))
            
            elif self.strategy == 'semi_hard':
                # Semi-hard negatives: closer than hardest positive but still negative
                hardest_pos_distance = torch.max(anchor_pos_distances)
                semi_hard_mask = (anchor_neg_distances < hardest_pos_distance)
                
                if semi_hard_mask.any():
                    semi_hard_neg_indices = negative_indices[semi_hard_mask]
                    # Pick random semi-hard negative
                    neg_idx = semi_hard_neg_indices[torch.randint(len(semi_hard_neg_indices), (1,))]
                    pos_idx = positive_indices[torch.randint(len(positive_indices), (1,))]
                    
                    triplets.append((i, pos_idx.item(), neg_idx.item()))
        
        return triplets
