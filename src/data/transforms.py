"""
Data transformations and augmentations for twin face verification.
Implements twin-aware augmentations (no horizontal flipping).
"""

import torch
import torchvision.transforms as transforms
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np
from PIL import Image


class TwinAwareAugmentation:
    """
    Augmentation pipeline designed for twin face verification.
    Key principle: NO horizontal flipping as twins often style hair differently.
    """
    
    def __init__(self, 
                 input_size=224,
                 color_jitter=0.4,
                 cutmix_alpha=1.0,
                 cutmix_prob=0.5,
                 is_training=True):
        self.input_size = input_size
        self.color_jitter = color_jitter
        self.cutmix_alpha = cutmix_alpha
        self.cutmix_prob = cutmix_prob
        self.is_training = is_training
        
        self.train_transform = A.Compose([
            # Geometric transforms (NO horizontal flip!)
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=10,
                p=0.5
            ),
            
            # Photometric transforms
            A.ColorJitter(
                brightness=color_jitter,
                contrast=color_jitter,
                saturation=color_jitter,
                hue=color_jitter/2,
                p=0.8
            ),
            
            # Quality degradation (important for twins - test robustness)
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.3),
            
            # Resize and normalize
            A.Resize(input_size, input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(input_size, input_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image):
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        if self.is_training:
            transformed = self.train_transform(image=image)
        else:
            transformed = self.val_transform(image=image)
        
        return transformed['image']


class CutMix:
    """CutMix augmentation for face verification."""
    
    def __init__(self, alpha=1.0, prob=0.5):
        self.alpha = alpha
        self.prob = prob
    
    def __call__(self, batch):
        if np.random.rand() > self.prob:
            return batch
        
        images, labels = batch['images'], batch['labels']
        batch_size = images.size(0)
        
        # Generate CutMix parameters
        lam = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(batch_size)
        
        # Generate bounding box
        _, _, H, W = images.shape
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        images[:, :, bby1:bby2, bbx1:bbx2] = images[rand_index, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda for exact area
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
        
        return {
            'images': images,
            'labels': labels,
            'cutmix_labels': labels[rand_index],
            'cutmix_lambda': lam
        }


class FaceAlignment:
    """Face alignment using 5-point landmarks."""
    
    def __init__(self, target_size=224):
        self.target_size = target_size
        # Standard 5-point landmarks for face alignment
        self.reference_points = np.array([
            [30.2946, 51.6963],  # left eye
            [65.5318, 51.5014],  # right eye  
            [48.0252, 71.7366],  # nose tip
            [33.5493, 92.3655],  # left mouth corner
            [62.7299, 92.2041]   # right mouth corner
        ], dtype=np.float32)
        
        # Scale reference points to target size
        self.reference_points *= (target_size / 96.0)
    
    def align_face(self, image, landmarks):
        """
        Align face using similarity transformation.
        
        Args:
            image: Input image (PIL or numpy)
            landmarks: 5-point landmarks [(x1,y1), (x2,y2), ...]
            
        Returns:
            Aligned face image
        """
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        landmarks = np.array(landmarks, dtype=np.float32)
        
        # Compute similarity transformation
        transformation_matrix = cv2.estimateAffinePartial2D(
            landmarks, self.reference_points
        )[0]
        
        # Apply transformation
        aligned = cv2.warpAffine(
            image, 
            transformation_matrix,
            (self.target_size, self.target_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101
        )
        
        return aligned


class MultiScaleTransform:
    """Multi-scale transformation for training."""
    
    def __init__(self, scales=[224, 256, 288], is_training=True):
        self.scales = scales
        self.is_training = is_training
        
        if is_training:
            self.transforms = {
                scale: TwinAwareAugmentation(scale, is_training=True)
                for scale in scales
            }
        else:
            self.transforms = {
                scale: TwinAwareAugmentation(scale, is_training=False)
                for scale in scales
            }
    
    def __call__(self, image):
        if self.is_training:
            # Randomly select scale during training
            scale = np.random.choice(self.scales)
        else:
            # Use largest scale for validation/testing
            scale = max(self.scales)
        
        return self.transforms[scale](image)


class HighResolutionTransform:
    """High-resolution transformation for inference."""
    
    def __init__(self, size=640):
        self.transform = A.Compose([
            A.Resize(size, size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2()
        ])
    
    def __call__(self, image):
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        transformed = self.transform(image=image)
        return transformed['image']


def get_transforms(config):
    """
    Get training and validation transforms based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        train_transform, val_transform
    """
    input_size = config['data'].get('input_size', config['model'].get('input_size', 224))
    color_jitter = config['data']['color_jitter']
    
    train_transform = TwinAwareAugmentation(
        input_size=input_size,
        color_jitter=color_jitter,
        cutmix_alpha=config['data']['cutmix_alpha'],
        cutmix_prob=config['data']['cutmix_prob'],
        is_training=True
    )
    
    val_transform = TwinAwareAugmentation(
        input_size=input_size,
        is_training=False
    )
    
    return train_transform, val_transform


def get_inference_transform(high_res=False):
    """Get transform for inference."""
    if high_res:
        return HighResolutionTransform(size=640)
    else:
        return TwinAwareAugmentation(input_size=224, is_training=False)


# Test-time augmentation for improved inference
class TestTimeAugmentation:
    """Test-time augmentation for robust inference."""
    
    def __init__(self, n_augmentations=5):
        self.n_augmentations = n_augmentations
        self.base_transform = TwinAwareAugmentation(is_training=False)
        
        # Light augmentations for TTA
        self.tta_transforms = [
            A.Compose([
                A.Rotate(limit=5, p=1.0),
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ]) for _ in range(n_augmentations)
        ]
    
    def __call__(self, image):
        """Return list of augmented versions."""
        if isinstance(image, Image.Image):
            image = np.array(image)
        
        augmented = []
        
        # Original
        augmented.append(self.base_transform(image))
        
        # TTA versions
        for transform in self.tta_transforms:
            aug_img = transform(image=image)['image']
            augmented.append(aug_img)
        
        return torch.stack(augmented)
