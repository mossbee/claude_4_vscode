"""
Inference module for twin face verification.
Provides easy-to-use interface for model deployment.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import json
import yaml
from typing import Union, List, Tuple, Dict, Optional

from ..models.twin_verifier import TwinVerifier
from ..data.transforms import get_transforms
from ..utils.metrics import VerificationMetrics


class TwinVerificationInference:
    """
    Inference class for twin face verification.
    Provides methods for verification, identification, and embedding extraction.
    """
    
    def __init__(self, 
                 model_path: str,
                 config_path: str = None,
                 device: str = 'auto'):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the trained model checkpoint
            config_path: Path to the config file (optional)
            device: Device to run inference on ('auto', 'cpu', 'cuda')
        """
        self.device = self._setup_device(device)
        self.model = None
        self.transform = None
        self.config = None
        
        # Load config if provided
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        
        # Load model
        self.load_model(model_path)
        
        # Setup transforms
        self._setup_transforms()
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup the compute device."""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                device = 'cpu'
        
        return torch.device(device)
    
    def load_model(self, model_path: str):
        """Load the trained model."""
        print(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Extract config from checkpoint if not provided
        if self.config is None and 'config' in checkpoint:
            self.config = checkpoint['config']
        
        # Create model
        if self.config:
            model_config = self.config.get('model', {})
        else:
            # Default config if none provided
            model_config = {
                'backbone': 'resnet50',
                'embedding_dim': 512,
                'attention_type': 'cbam',
                'dropout': 0.5
            }
        
        self.model = TwinVerifier(
            backbone=model_config.get('backbone', 'resnet50'),
            embedding_dim=model_config.get('embedding_dim', 512),
            attention_type=model_config.get('attention_type', 'cbam'),
            dropout=model_config.get('dropout', 0.5)
        )
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        print(f"Model loaded successfully on {self.device}")
    
    def _setup_transforms(self):
        """Setup image transforms for inference."""
        if self.config and 'data' in self.config:
            # Use config transforms
            _, self.transform = get_transforms(self.config)
        else:
            # Default transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
    
    def preprocess_image(self, image: Union[str, Image.Image, np.ndarray]) -> torch.Tensor:
        """
        Preprocess an image for inference.
        
        Args:
            image: Image path, PIL Image, or numpy array
            
        Returns:
            Preprocessed image tensor
        """
        if isinstance(image, str):
            # Load from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy to PIL
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        image_tensor = self.transform(image)
        
        return image_tensor
    
    def extract_embedding(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """
        Extract embedding for a single image.
        
        Args:
            image: Input image
            
        Returns:
            Embedding vector as numpy array
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image)
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.model.extract_features(image_batch)
            
        return embedding.cpu().numpy().squeeze()
    
    def extract_embeddings_batch(self, images: List[Union[str, Image.Image, np.ndarray]], 
                                batch_size: int = 32) -> np.ndarray:
        """
        Extract embeddings for a batch of images.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            Array of embeddings [N, embedding_dim]
        """
        embeddings = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                img_tensor = self.preprocess_image(img)
                batch_tensors.append(img_tensor)
            
            batch_tensor = torch.stack(batch_tensors).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                batch_embeddings = self.model.extract_features(batch_tensor)
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)
    
    def verify_pair(self, 
                   image1: Union[str, Image.Image, np.ndarray],
                   image2: Union[str, Image.Image, np.ndarray],
                   return_distance: bool = False) -> Union[float, Tuple[float, float]]:
        """
        Verify if two images are of the same person.
        
        Args:
            image1: First image
            image2: Second image
            return_distance: Whether to return the distance as well
            
        Returns:
            Similarity score (and distance if requested)
        """
        # Extract embeddings
        emb1 = self.extract_embedding(image1)
        emb2 = self.extract_embedding(image2)
        
        # Compute similarity
        similarity = self.compute_similarity(emb1, emb2)
        
        if return_distance:
            distance = self.compute_distance(emb1, emb2)
            return similarity, distance
        
        return similarity
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Compute cosine similarity between embeddings."""
        # Normalize embeddings
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
        
        # Compute cosine similarity
        similarity = np.dot(emb1_norm, emb2_norm)
        
        return float(similarity)
    
    def compute_distance(self, emb1: np.ndarray, emb2: np.ndarray, metric: str = 'euclidean') -> float:
        """Compute distance between embeddings."""
        if metric == 'euclidean':
            distance = np.linalg.norm(emb1 - emb2)
        elif metric == 'cosine':
            similarity = self.compute_similarity(emb1, emb2)
            distance = 1 - similarity
        else:
            raise ValueError(f"Unknown distance metric: {metric}")
        
        return float(distance)
    
    def identify(self, 
                query_image: Union[str, Image.Image, np.ndarray],
                gallery_embeddings: np.ndarray,
                gallery_labels: List[str],
                top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Identify a query image against a gallery.
        
        Args:
            query_image: Query image
            gallery_embeddings: Gallery embeddings [N, embedding_dim]
            gallery_labels: Gallery labels
            top_k: Number of top matches to return
            
        Returns:
            List of (label, similarity) tuples sorted by similarity
        """
        # Extract query embedding
        query_embedding = self.extract_embedding(query_image)
        
        # Compute similarities with gallery
        similarities = []
        for i, gallery_emb in enumerate(gallery_embeddings):
            similarity = self.compute_similarity(query_embedding, gallery_emb)
            similarities.append((gallery_labels[i], similarity))
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def batch_verify(self, 
                    image_pairs: List[Tuple[Union[str, Image.Image, np.ndarray], 
                                          Union[str, Image.Image, np.ndarray]]],
                    batch_size: int = 32) -> List[float]:
        """
        Verify multiple image pairs.
        
        Args:
            image_pairs: List of (image1, image2) tuples
            batch_size: Batch size for processing
            
        Returns:
            List of similarity scores
        """
        similarities = []
        
        for i in range(0, len(image_pairs), batch_size):
            batch_pairs = image_pairs[i:i + batch_size]
            
            # Prepare batch
            batch_images1 = []
            batch_images2 = []
            
            for img1, img2 in batch_pairs:
                batch_images1.append(self.preprocess_image(img1))
                batch_images2.append(self.preprocess_image(img2))
            
            batch_tensor1 = torch.stack(batch_images1).to(self.device)
            batch_tensor2 = torch.stack(batch_images2).to(self.device)
            
            # Extract embeddings
            with torch.no_grad():
                emb1 = self.model.extract_features(batch_tensor1)
                emb2 = self.model.extract_features(batch_tensor2)
                
                # Compute similarities
                batch_similarities = F.cosine_similarity(emb1, emb2, dim=1)
                similarities.extend(batch_similarities.cpu().numpy().tolist())
        
        return similarities
    
    def evaluate_on_dataset(self, 
                           test_pairs: List[Tuple[str, str]],
                           test_labels: List[int],
                           batch_size: int = 32) -> Dict[str, float]:
        """
        Evaluate the model on a test dataset.
        
        Args:
            test_pairs: List of (image1_path, image2_path) tuples
            test_labels: List of labels (0: same, 1: different)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary of evaluation metrics
        """
        print(f"Evaluating on {len(test_pairs)} pairs...")
        
        # Compute similarities
        similarities = self.batch_verify(test_pairs, batch_size)
        
        # Compute metrics
        metrics = VerificationMetrics()
        
        # Convert similarities to distances and create dummy embeddings for metrics
        distances = [1 - sim for sim in similarities]
        dummy_emb1 = np.zeros((len(test_pairs), 1))
        dummy_emb2 = np.ones((len(test_pairs), 1))
        
        # Update metrics with distances
        metrics.distances = distances
        metrics.labels = test_labels
        
        # Compute detailed metrics
        results = metrics.compute_detailed_metrics()['overall']
        
        print("Evaluation Results:")
        for metric, value in results.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
        
        return results
    
    def save_embeddings(self, 
                       images: List[str],
                       output_path: str,
                       batch_size: int = 32):
        """
        Extract and save embeddings for a list of images.
        
        Args:
            images: List of image paths
            output_path: Path to save embeddings
            batch_size: Batch size for processing
        """
        print(f"Extracting embeddings for {len(images)} images...")
        
        embeddings = self.extract_embeddings_batch(images, batch_size)
        
        # Save as numpy file
        np.save(output_path, embeddings)
        
        # Also save image paths
        paths_file = output_path.replace('.npy', '_paths.json')
        with open(paths_file, 'w') as f:
            json.dump(images, f, indent=2)
        
        print(f"Embeddings saved to {output_path}")
        print(f"Image paths saved to {paths_file}")


def load_inference_model(model_path: str, 
                        config_path: str = None,
                        device: str = 'auto') -> TwinVerificationInference:
    """
    Convenience function to load an inference model.
    
    Args:
        model_path: Path to the trained model
        config_path: Path to the config file
        device: Device to run inference on
        
    Returns:
        TwinVerificationInference instance
    """
    return TwinVerificationInference(model_path, config_path, device)


# Example usage functions
def example_verify_twins():
    """Example of using the inference engine for twin verification."""
    # Load model
    inference = load_inference_model('path/to/model.pth', 'path/to/config.yaml')
    
    # Verify a pair of images
    similarity = inference.verify_pair('twin1_img1.jpg', 'twin1_img2.jpg')
    print(f"Similarity: {similarity:.4f}")
    
    # Extract embeddings
    embedding1 = inference.extract_embedding('person1.jpg')
    embedding2 = inference.extract_embedding('person2.jpg')
    
    # Compute custom similarity
    similarity = inference.compute_similarity(embedding1, embedding2)
    print(f"Custom similarity: {similarity:.4f}")


def example_identification():
    """Example of using the inference engine for identification."""
    # Load model
    inference = load_inference_model('path/to/model.pth')
    
    # Build gallery
    gallery_images = ['person1.jpg', 'person2.jpg', 'person3.jpg']
    gallery_embeddings = inference.extract_embeddings_batch(gallery_images)
    gallery_labels = ['Person 1', 'Person 2', 'Person 3']
    
    # Identify query
    query_image = 'unknown_person.jpg'
    matches = inference.identify(query_image, gallery_embeddings, gallery_labels, top_k=3)
    
    print("Top matches:")
    for label, similarity in matches:
        print(f"  {label}: {similarity:.4f}")


if __name__ == "__main__":
    # Run examples
    print("Example 1: Twin Verification")
    example_verify_twins()
    
    print("\nExample 2: Identification")
    example_identification()
