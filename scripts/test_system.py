"""
System integration test script for twin face verification.
Tests all components to ensure they work together correctly.
"""

import os
import sys
import tempfile
import shutil
import yaml
import torch
import numpy as np
from PIL import Image
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.twin_verifier import TwinVerifier
from src.data.dataset import TwinFaceDataset, create_data_loaders
from src.data.transforms import get_transforms
from src.training.trainer import TwinVerificationTrainer
from src.training.losses import CombinedLoss
from src.inference.inference import TwinVerificationInference
from src.utils.metrics import VerificationMetrics, TwinSpecificMetrics
from src.utils.visualization import TrainingVisualizer


def create_dummy_dataset(data_dir, num_pairs=3, images_per_identity=5):
    """Create a dummy dataset for testing."""
    print(f"Creating dummy dataset in {data_dir}")
    
    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    
    # Create twin pairs
    twin_pairs = []
    folder_counter = 1
    
    for pair_id in range(num_pairs):
        # Create two folders for each twin pair
        folder1 = f"img_folder_{folder_counter}"
        folder2 = f"img_folder_{folder_counter + 1}"
        folder_counter += 2
        
        twin_pairs.append([folder1, folder2])
        
        # Create directories
        os.makedirs(os.path.join(data_dir, folder1), exist_ok=True)
        os.makedirs(os.path.join(data_dir, folder2), exist_ok=True)
        
        # Create dummy images
        for i in range(images_per_identity):
            # Create similar images for twins (same base pattern)
            base_color1 = (100 + pair_id * 30, 150 + pair_id * 20, 200 + pair_id * 10)
            base_color2 = (110 + pair_id * 30, 160 + pair_id * 20, 210 + pair_id * 10)
            
            # Add some variation
            img1 = Image.new('RGB', (224, 224), 
                           (base_color1[0] + i * 5, base_color1[1] + i * 5, base_color1[2] + i * 5))
            img2 = Image.new('RGB', (224, 224), 
                           (base_color2[0] + i * 5, base_color2[1] + i * 5, base_color2[2] + i * 5))
            
            img1.save(os.path.join(data_dir, folder1, f"image_{i:03d}.jpg"))
            img2.save(os.path.join(data_dir, folder2, f"image_{i:03d}.jpg"))
    
    # Create additional non-twin identities
    for identity_id in range(num_pairs):
        folder = f"img_folder_{folder_counter}"
        folder_counter += 1
        
        os.makedirs(os.path.join(data_dir, folder), exist_ok=True)
        
        # Create different colored images
        base_color = (50 + identity_id * 60, 50 + identity_id * 40, 50 + identity_id * 80)
        
        for i in range(images_per_identity):
            img = Image.new('RGB', (224, 224), 
                          (base_color[0] + i * 10, base_color[1] + i * 10, base_color[2] + i * 10))
            img.save(os.path.join(data_dir, folder, f"image_{i:03d}.jpg"))
    
    # Save pairs.json
    pairs_path = os.path.join(data_dir, 'pairs.json')
    with open(pairs_path, 'w') as f:
        json.dump(twin_pairs, f, indent=2)
    
    print(f"Created {len(twin_pairs)} twin pairs and {num_pairs} additional identities")
    return twin_pairs


def create_test_config():
    """Create a test configuration."""
    config = {
        'model': {
            'backbone': 'resnet18',  # Use smaller model for testing
            'embedding_dim': 128,
            'attention_type': 'cbam',
            'dropout': 0.5,
            'pretrained': True
        },
        'data': {
            'dataset_path': '',  # Will be set by test
            'num_workers': 0,  # No multiprocessing for testing
            'pin_memory': False,
            'image_size': [224, 224],
            'augmentation': {
                'horizontal_flip': False,  # Twin-aware: no horizontal flip
                'rotation': 10,
                'color_jitter': {
                    'brightness': 0.2,
                    'contrast': 0.2,
                    'saturation': 0.2,
                    'hue': 0.1
                },
                'random_crop': True,
                'normalize': {
                    'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]
                }
            }
        },
        'training': {
            'epochs': 2,  # Short training for testing
            'batch_size': 4,
            'hard_twin_ratio': 0.5,
            'other_ratio': 0.25,
            'optimizer': {
                'type': 'adam',
                'lr': 0.001,
                'weight_decay': 1e-4
            },
            'scheduler': {
                'type': 'cosine',
                'min_lr': 1e-6
            },
            'loss': {
                'type': 'combined',
                'margin': 1.0,
                'twin_weight': 2.0,
                'same_weight': 1.0,
                'other_weight': 1.0
            },
            'mixed_precision': False,  # Disable for testing
            'gradient_clipping': 1.0,
            'save_frequency': 1,
            'validation_frequency': 1
        }
    }
    
    return config


def test_model_creation():
    """Test model creation and basic functionality."""
    print("Testing model creation...")
    
    model = TwinVerifier(
        backbone='resnet18',
        embedding_dim=128,
        attention_type='cbam',
        dropout=0.5
    )
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        # Test feature extraction
        features = model.extract_features(dummy_input)
        assert features.shape == (2, 128), f"Expected (2, 128), got {features.shape}"
        
        # Test forward pass
        output = model(dummy_input)
        assert output.shape == (2, 128), f"Expected (2, 128), got {output.shape}"
    
    print("✓ Model creation and forward pass successful")


def test_dataset_loading(data_dir):
    """Test dataset loading and data loaders."""
    print("Testing dataset loading...")
    
    config = create_test_config()
    config['data']['dataset_path'] = data_dir
    
    # Test dataset creation
    _, val_transform = get_transforms(config)
    
    dataset = TwinFaceDataset(
        dataset_path=data_dir,
        split='train',
        transform=val_transform,
        twin_ratio=0.5,
        same_ratio=0.25,
        other_ratio=0.25
    )
    
    print(f"Dataset loaded: {len(dataset)} pairs")
    print(f"Twin pairs: {len(dataset.split_twin_pairs)}")
    print(f"Identities: {len(dataset.split_identities)}")
    
    # Test data loader
    try:
        train_loader, val_loader, test_loader, train_dataset = create_data_loaders(config)
        
        # Test loading a batch
        batch = next(iter(train_loader))
        
        assert 'img1' in batch
        assert 'img2' in batch
        assert 'label' in batch
        assert batch['img1'].shape[0] == config['training']['batch_size']
        
        print("✓ Dataset loading and data loaders successful")
        return train_loader, val_loader, test_loader
        
    except Exception as e:
        print(f"✗ Dataset loading failed: {e}")
        raise


def test_training_loop(train_loader, val_loader, temp_dir):
    """Test training loop."""
    print("Testing training loop...")
    
    device = torch.device('cpu')  # Use CPU for testing
    
    # Create model
    model = TwinVerifier(
        backbone='resnet18',
        embedding_dim=128,
        attention_type='cbam',
        dropout=0.5
    ).to(device)
    
    # Create loss function
    loss_fn = CombinedLoss(
        margin=1.0,
        twin_weight=2.0,
        same_weight=1.0,
        other_weight=1.0
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create config
    config = create_test_config()
    
    # Create trainer
    trainer = TwinVerificationTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=None,
        loss_fn=loss_fn,
        device=device,
        config=config,
        save_dir=temp_dir,
        use_wandb=False
    )
    
    # Test one training step
    try:
        trainer.train_epoch(0)
        print("✓ Training epoch successful")
        
        # Test validation
        val_metrics = trainer.validate()
        print(f"✓ Validation successful, AUC: {val_metrics.get('auc', 'N/A')}")
        
        # Test saving
        save_path = os.path.join(temp_dir, 'test_model.pth')
        trainer.save_checkpoint(save_path, is_best=False)
        
        assert os.path.exists(save_path), "Model checkpoint not saved"
        print("✓ Model saving successful")
        
        return save_path
        
    except Exception as e:
        print(f"✗ Training failed: {e}")
        raise


def test_inference(model_path, data_dir, config):
    """Test inference functionality."""
    print("Testing inference...")
    
    # Create temporary config file
    config_path = os.path.join(os.path.dirname(model_path), 'test_config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        # Load inference model
        inference_model = TwinVerificationInference(
            model_path=model_path,
            config_path=config_path,
            device='cpu'
        )
        
        # Test embedding extraction
        # Find some test images
        test_images = []
        for folder in os.listdir(data_dir):
            if folder.startswith('img_folder_'):
                folder_path = os.path.join(data_dir, folder)
                for img_file in os.listdir(folder_path):
                    if img_file.endswith('.jpg'):
                        test_images.append(os.path.join(folder_path, img_file))
                        if len(test_images) >= 4:
                            break
                if len(test_images) >= 4:
                    break
        
        assert len(test_images) >= 2, "Not enough test images found"
        
        # Test single embedding extraction
        embedding = inference_model.extract_embedding(test_images[0])
        assert embedding.shape == (128,), f"Expected (128,), got {embedding.shape}"
        print("✓ Single embedding extraction successful")
        
        # Test batch embedding extraction
        embeddings = inference_model.extract_embeddings_batch(test_images[:2])
        assert embeddings.shape == (2, 128), f"Expected (2, 128), got {embeddings.shape}"
        print("✓ Batch embedding extraction successful")
        
        # Test pair verification
        similarity = inference_model.verify_pair(test_images[0], test_images[1])
        assert isinstance(similarity, float), f"Expected float, got {type(similarity)}"
        assert -1 <= similarity <= 1, f"Similarity out of range: {similarity}"
        print(f"✓ Pair verification successful, similarity: {similarity:.4f}")
        
        # Test batch verification
        pairs = [(test_images[0], test_images[1]), (test_images[2], test_images[3])]
        similarities = inference_model.batch_verify(pairs)
        assert len(similarities) == 2, f"Expected 2 similarities, got {len(similarities)}"
        print("✓ Batch verification successful")
        
        print("✓ All inference tests successful")
        
    except Exception as e:
        print(f"✗ Inference failed: {e}")
        raise


def test_metrics():
    """Test metrics computation."""
    print("Testing metrics...")
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 100
    
    # Simulate embeddings and labels
    embeddings1 = np.random.randn(n_samples, 128)
    embeddings2 = np.random.randn(n_samples, 128)
    labels = np.random.choice([0, 1, 2], n_samples)  # 0: same, 1: twin, 2: other
    
    # Test verification metrics
    metrics = VerificationMetrics()
    metrics.update(embeddings1, embeddings2, labels, labels)
    
    results = metrics.compute_detailed_metrics()
    
    assert 'overall' in results
    assert 'auc' in results['overall']
    assert 'eer' in results['overall']
    print("✓ Verification metrics successful")
    
    # Test twin-specific metrics
    distances = [np.linalg.norm(e1 - e2) for e1, e2 in zip(embeddings1, embeddings2)]
    
    twin_metrics = TwinSpecificMetrics()
    twin_metrics.update(distances, labels)
    
    twin_results = twin_metrics.compute_twin_separation_metrics()
    
    assert 'same_distance_mean' in twin_results
    assert 'twin_distance_mean' in twin_results
    print("✓ Twin-specific metrics successful")


def run_full_system_test():
    """Run a comprehensive system test."""
    print("="*60)
    print("TWIN FACE VERIFICATION SYSTEM TEST")
    print("="*60)
    
    # Create temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        data_dir = os.path.join(temp_dir, 'test_dataset')
        output_dir = os.path.join(temp_dir, 'test_output')
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Test 1: Create dummy dataset
            print("\n1. Creating test dataset...")
            twin_pairs = create_dummy_dataset(data_dir, num_pairs=3, images_per_identity=5)
            
            # Test 2: Model creation
            print("\n2. Testing model creation...")
            test_model_creation()
            
            # Test 3: Dataset loading
            print("\n3. Testing dataset loading...")
            train_loader, val_loader, test_loader = test_dataset_loading(data_dir)
            
            # Test 4: Training loop
            print("\n4. Testing training loop...")
            model_path = test_training_loop(train_loader, val_loader, output_dir)
            
            # Test 5: Inference
            print("\n5. Testing inference...")
            config = create_test_config()
            config['data']['dataset_path'] = data_dir
            test_inference(model_path, data_dir, config)
            
            # Test 6: Metrics
            print("\n6. Testing metrics...")
            test_metrics()
            
            print("\n" + "="*60)
            print("✓ ALL TESTS PASSED SUCCESSFULLY!")
            print("✓ The twin face verification system is working correctly.")
            print("="*60)
            
            return True
            
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            print("="*60)
            return False


def test_individual_components():
    """Test individual components separately."""
    print("Testing individual components...")
    
    # Test imports
    print("Testing imports...")
    try:
        from src.models.attention import SEBlock, CBAM, NonLocalBlock, CrossAttention
        from src.models.backbone import AttentionResNet
        from src.models.twin_verifier import TwinVerifier
        from src.data.dataset import TwinFaceDataset
        from src.data.transforms import get_transforms
        from src.training.losses import TwinAwareMarginLoss, ContrastiveLoss, CombinedLoss
        from src.training.mining import HardNegativeMiner
        from src.training.trainer import TwinVerificationTrainer
        from src.utils.metrics import VerificationMetrics, TwinSpecificMetrics
        from src.utils.visualization import AttentionVisualizer, EmbeddingVisualizer
        from src.inference.inference import TwinVerificationInference
        print("✓ All imports successful")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False
    
    # Test attention mechanisms
    print("Testing attention mechanisms...")
    try:
        se = SEBlock(64)
        cbam = CBAM(64)
        
        x = torch.randn(2, 64, 56, 56)
        se_out = se(x)
        cbam_out = cbam(x)
        
        assert se_out.shape == x.shape
        assert cbam_out.shape == x.shape
        print("✓ Attention mechanisms working")
    except Exception as e:
        print(f"✗ Attention test failed: {e}")
        return False
    
    # Test loss functions
    print("Testing loss functions...")
    try:
        loss_fn = CombinedLoss()
        
        embeddings1 = torch.randn(4, 128)
        embeddings2 = torch.randn(4, 128)
        labels = torch.tensor([0, 1, 2, 0])  # same, twin, other, same
        
        loss = loss_fn(embeddings1, embeddings2, labels)
        assert isinstance(loss.item(), float)
        print("✓ Loss functions working")
    except Exception as e:
        print(f"✗ Loss test failed: {e}")
        return False
    
    print("✓ All individual component tests passed")
    return True


def main():
    """Main function."""
    print("Starting twin face verification system tests...")
    
    # Test 1: Individual components
    print("\n" + "-"*40)
    print("TESTING INDIVIDUAL COMPONENTS")
    print("-"*40)
    
    if not test_individual_components():
        print("Individual component tests failed. Stopping.")
        return
    
    # Test 2: Full system integration
    print("\n" + "-"*40)
    print("TESTING FULL SYSTEM INTEGRATION")
    print("-"*40)
    
    if not run_full_system_test():
        print("Full system test failed.")
        return
    
    print("\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
    print("The twin face verification system is ready for use.")


if __name__ == "__main__":
    main()
