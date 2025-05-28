"""
Dataset classes for twin face verification.
Handles the twin dataset structure with pairs.json file.
"""

import os
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from collections import defaultdict
import json


class TwinFaceDataset(Dataset):
    """
    Dataset for twin face verification.
    
    Generates three types of pairs:
    - P+: Same individual (positive pairs)
    - P-t: Identical twin pairs (hard negatives)
    - P-o: Other identity pairs (ordinary negatives)
    """
    
    def __init__(self, 
                 dataset_path,
                 pairs_json_path=None,
                 split='train',
                 transform=None,
                 pair_sampling_strategy='balanced',
                 twin_ratio=0.5,
                 same_ratio=0.25,
                 other_ratio=0.25):
        """
        Args:
            dataset_path: Path to dataset with img_folder_1, img_folder_2, etc.
            pairs_json_path: Path to pairs.json file containing twin pairs
            split: 'train', 'val', or 'test'
            transform: Image transformations
            pair_sampling_strategy: How to sample pairs ('balanced', 'hard', 'easy')
            twin_ratio: Ratio of twin pairs in each batch
            same_ratio: Ratio of same person pairs
            other_ratio: Ratio of other person pairs
        """
        self.dataset_path = dataset_path
        self.pairs_json_path = pairs_json_path or os.path.join(dataset_path, 'pairs.json')
        self.split = split
        self.transform = transform
        self.pair_sampling_strategy = pair_sampling_strategy
        self.twin_ratio = twin_ratio
        self.same_ratio = same_ratio
        self.other_ratio = other_ratio
        
        # Load dataset structure
        self._load_dataset_structure()
        
        # Create split
        self._create_split()
        
        # Generate pairs for this epoch
        self.regenerate_pairs()
    
    def _load_dataset_structure(self):
        """Load the dataset structure and identify twin pairs from pairs.json."""
        self.twin_pairs = []
        self.identity_images = defaultdict(list)
        
        # Load twin pairs from JSON file
        if os.path.exists(self.pairs_json_path):
            with open(self.pairs_json_path, 'r') as f:
                twin_pairs_data = json.load(f)
            
            # Convert to list of tuples
            self.twin_pairs = [tuple(pair) for pair in twin_pairs_data]
            print(f"Loaded {len(self.twin_pairs)} twin pairs from {self.pairs_json_path}")
        else:
            print(f"Warning: pairs.json not found at {self.pairs_json_path}")
            self.twin_pairs = []
        
        # Scan dataset directory and collect images for each folder
        for folder_name in os.listdir(self.dataset_path):
            folder_path = os.path.join(self.dataset_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            
            # Collect images for this identity
            images = []
            for img_file in os.listdir(folder_path):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff')):
                    img_path = os.path.join(folder_path, img_file)
                    images.append(img_path)
            
            if images:
                self.identity_images[folder_name] = images
        
        # Verify that all twin pairs exist in the dataset
        valid_twin_pairs = []
        for pair in self.twin_pairs:
            folder1, folder2 = pair
            if folder1 in self.identity_images and folder2 in self.identity_images:
                valid_twin_pairs.append(pair)
            else:
                print(f"Warning: Twin pair {pair} not found in dataset")
        
        self.twin_pairs = valid_twin_pairs
        
        print(f"Found {len(self.twin_pairs)} valid twin pairs")
        print(f"Found {len(self.identity_images)} identity folders")
    
    def _create_split(self):
        """Create train/val/test splits."""
        # Split twin pairs into train/val/test
        twin_pairs = list(self.twin_pairs)
        random.shuffle(twin_pairs)
        
        n_train = int(0.8 * len(twin_pairs))
        n_val = int(0.1 * len(twin_pairs))
        
        if self.split == 'train':
            self.split_twin_pairs = twin_pairs[:n_train]
        elif self.split == 'val':
            self.split_twin_pairs = twin_pairs[n_train:n_train + n_val]
        else:  # test
            self.split_twin_pairs = twin_pairs[n_train + n_val:]
        
        # Collect all identities in this split
        self.split_identities = set()
        for twin_a, twin_b in self.split_twin_pairs:
            self.split_identities.add(twin_a)
            self.split_identities.add(twin_b)
        
        print(f"{self.split} split: {len(self.split_twin_pairs)} twin pairs, "
              f"{len(self.split_identities)} identities")
    
    def regenerate_pairs(self, num_pairs=None):
        """Regenerate pairs for training. Call this at the start of each epoch."""
        if num_pairs is None:
            # Default: generate pairs based on available data
            num_pairs = len(self.split_twin_pairs) * 10  # 10x oversampling
        
        self.pairs = []
        self.labels = []  # 0: same, 1: twin, 2: other
        
        # Calculate number of pairs for each type
        n_twin = int(num_pairs * self.twin_ratio)
        n_same = int(num_pairs * self.same_ratio)
        n_other = num_pairs - n_twin - n_same
        
        # Generate twin pairs (hard negatives)
        for _ in range(n_twin):
            twin_a, twin_b = random.choice(self.split_twin_pairs)
            img1 = random.choice(self.identity_images[twin_a])
            img2 = random.choice(self.identity_images[twin_b])
            self.pairs.append((img1, img2))
            self.labels.append(1)  # twin
        
        # Generate same person pairs (positives)
        for _ in range(n_same):
            identity = random.choice(list(self.split_identities))
            if len(self.identity_images[identity]) >= 2:
                img1, img2 = random.sample(self.identity_images[identity], 2)
            else:
                # If only one image, use it twice (with different augmentations)
                img1 = img2 = self.identity_images[identity][0]
            self.pairs.append((img1, img2))
            self.labels.append(0)  # same
        
        # Generate other person pairs (easy negatives)
        identities_list = list(self.split_identities)
        for _ in range(n_other):
            # Ensure we don't pick twins
            attempts = 0
            while attempts < 10:  # Avoid infinite loop
                id1, id2 = random.sample(identities_list, 2)
                # Check if they are twins
                is_twin_pair = False
                for twin_a, twin_b in self.split_twin_pairs:
                    if (id1 == twin_a and id2 == twin_b) or (id1 == twin_b and id2 == twin_a):
                        is_twin_pair = True
                        break
                
                if not is_twin_pair:
                    break
                attempts += 1
            
            img1 = random.choice(self.identity_images[id1])
            img2 = random.choice(self.identity_images[id2])
            self.pairs.append((img1, img2))
            self.labels.append(2)  # other
        
        # Shuffle pairs
        combined = list(zip(self.pairs, self.labels))
        random.shuffle(combined)
        self.pairs, self.labels = zip(*combined)
        
        print(f"Generated {len(self.pairs)} pairs: "
              f"{n_same} same, {n_twin} twin, {n_other} other")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        img1_path, img2_path = self.pairs[idx]
        label = self.labels[idx]
        
        # Load images
        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')
        
        # Apply transforms
        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        
        return {
            'img1': img1,
            'img2': img2,
            'label': label,
            'img1_path': img1_path,
            'img2_path': img2_path
        }


class SingleImageDataset(Dataset):
    """Dataset for single image encoding (gallery/probe sets)."""
    
    def __init__(self, image_paths, transform=None):
        self.image_paths = image_paths
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return {
            'image': img,
            'path': img_path
        }


class HardNegativeMiner:
    """Mine hard negative pairs for improved training."""
    
    def __init__(self, model, dataset, device):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.embeddings_cache = {}
    
    def compute_embeddings(self, data_loader):
        """Compute embeddings for all images in the dataset."""
        self.model.eval()
        embeddings = {}
        
        with torch.no_grad():
            for batch in data_loader:
                images = batch['image'].to(self.device)
                paths = batch['path']
                
                embs = self.model(images)
                
                for emb, path in zip(embs, paths):
                    embeddings[path] = emb.cpu().numpy()
        
        return embeddings
    
    def mine_hard_twins(self, embeddings, twin_pairs, top_k=100):
        """Mine hardest twin pairs based on current embeddings."""
        hard_pairs = []
        
        for twin_a, twin_b in twin_pairs:
            images_a = self.dataset.identity_images[twin_a]
            images_b = self.dataset.identity_images[twin_b]
            
            min_distance = float('inf')
            hardest_pair = None
            
            for img_a in images_a:
                for img_b in images_b:
                    if img_a in embeddings and img_b in embeddings:
                        emb_a = embeddings[img_a]
                        emb_b = embeddings[img_b]
                        
                        # Compute L2 distance
                        distance = np.linalg.norm(emb_a - emb_b)
                        
                        if distance < min_distance:
                            min_distance = distance
                            hardest_pair = (img_a, img_b, distance)
            
            if hardest_pair:
                hard_pairs.append(hardest_pair)
        
        # Sort by distance (hardest first)
        hard_pairs.sort(key=lambda x: x[2])
        
        return hard_pairs[:top_k]


def create_data_loaders(config):
    """Create data loaders for training, validation, and testing."""
    from .transforms import get_transforms
    
    train_transform, val_transform = get_transforms(config)
      # Create datasets
    train_dataset = TwinFaceDataset(
        dataset_path=config['data']['dataset_path'],
        pairs_json_path=config['data'].get('pairs_json_path'),
        split='train',
        transform=train_transform,
        twin_ratio=config['training']['hard_twin_ratio'],
        same_ratio=1 - config['training']['hard_twin_ratio'] - config['training']['other_ratio'],
        other_ratio=config['training']['other_ratio']
    )
    
    val_dataset = TwinFaceDataset(
        dataset_path=config['data']['dataset_path'],
        pairs_json_path=config['data'].get('pairs_json_path'),
        split='val',
        transform=val_transform,
        twin_ratio=0.5,  # Balanced for validation
        same_ratio=0.25,
        other_ratio=0.25
    )
    
    test_dataset = TwinFaceDataset(
        dataset_path=config['data']['dataset_path'],
        pairs_json_path=config['data'].get('pairs_json_path'),
        split='test',
        transform=val_transform,
        twin_ratio=0.5,
        same_ratio=0.25,
        other_ratio=0.25
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory'],
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    return train_loader, val_loader, test_loader, train_dataset
