"""
Evaluation script for twin face verification model.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
import json
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.inference import TwinVerificationInference
from src.data.dataset import TwinFaceDataset
from src.data.transforms import get_transforms
from src.utils.metrics import VerificationMetrics, TwinSpecificMetrics
from src.utils.visualization import (
    EmbeddingVisualizer, 
    create_pair_comparison_grid,
    plot_roc_curve
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Evaluate twin face verification model')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to dataset directory')
    parser.add_argument('--pairs_json', type=str,
                       help='Path to pairs.json file')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                       help='Output directory for evaluation results')
    parser.add_argument('--split', type=str, default='test',
                       choices=['train', 'val', 'test'],
                       help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--save_embeddings', action='store_true',
                       help='Save extracted embeddings')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run evaluation on')
    
    return parser.parse_args()


def load_test_dataset(args, config):
    """Load the test dataset."""
    _, val_transform = get_transforms(config)
    
    dataset = TwinFaceDataset(
        dataset_path=args.data_path,
        pairs_json_path=args.pairs_json,
        split=args.split,
        transform=val_transform,
        twin_ratio=0.5,
        same_ratio=0.25,
        other_ratio=0.25
    )
    
    return dataset


def evaluate_verification(inference_model, dataset, batch_size=32):
    """Evaluate verification performance."""
    print("Evaluating verification performance...")
    
    # Prepare data
    image_pairs = []
    labels = []
    pair_types = []
    
    for i in tqdm(range(len(dataset)), desc="Loading test pairs"):
        sample = dataset[i]
        image_pairs.append((sample['img1_path'], sample['img2_path']))
        labels.append(1 if sample['label'] == 0 else 0)  # Convert to binary (same=1, different=0)
        pair_types.append(sample['label'])
    
    # Compute similarities
    print("Computing similarities...")
    similarities = inference_model.batch_verify(image_pairs, batch_size)
    
    # Convert to distances
    distances = [1 - sim for sim in similarities]
    
    # Compute standard verification metrics
    metrics = VerificationMetrics()
    dummy_emb1 = np.zeros((len(labels), 1))
    dummy_emb2 = np.ones((len(labels), 1))
    
    # Use the distances we computed
    metrics.distances = distances
    metrics.labels = labels
    metrics.pair_types = pair_types
    
    results = metrics.compute_detailed_metrics()
    
    # Compute twin-specific metrics
    twin_metrics = TwinSpecificMetrics()
    twin_metrics.update(distances, pair_types)
    twin_results = twin_metrics.compute_twin_separation_metrics()
    
    return results, twin_results, similarities, distances, labels, pair_types


def evaluate_identification(inference_model, dataset, gallery_size=100):
    """Evaluate identification performance."""
    print(f"Evaluating identification performance with gallery size {gallery_size}...")
    
    # Sample gallery and query sets
    all_identities = list(dataset.identity_images.keys())
    
    if len(all_identities) < gallery_size:
        gallery_identities = all_identities
    else:
        gallery_identities = np.random.choice(all_identities, gallery_size, replace=False)
    
    # Build gallery
    gallery_images = []
    gallery_labels = []
    
    for identity in gallery_identities:
        images = dataset.identity_images[identity]
        if images:
            gallery_images.append(images[0])  # Use first image as gallery
            gallery_labels.append(identity)
    
    # Extract gallery embeddings
    print("Extracting gallery embeddings...")
    gallery_embeddings = inference_model.extract_embeddings_batch(gallery_images)
    
    # Create query set (using different images of same identities)
    query_images = []
    query_labels = []
    
    for identity in gallery_identities:
        images = dataset.identity_images[identity]
        if len(images) > 1:
            query_images.append(images[1])  # Use second image as query
            query_labels.append(identity)
    
    if not query_images:
        print("Warning: No query images available for identification test")
        return {}
    
    # Evaluate identification
    print("Running identification...")
    results = {}
    
    for k in [1, 5, 10]:
        if k <= len(gallery_identities):
            correct = 0
            
            for query_img, query_label in tqdm(zip(query_images, query_labels), 
                                             desc=f"Top-{k} accuracy"):
                matches = inference_model.identify(
                    query_img, gallery_embeddings, gallery_labels, top_k=k
                )
                
                # Check if correct identity is in top-k
                top_k_labels = [match[0] for match in matches]
                if query_label in top_k_labels:
                    correct += 1
            
            accuracy = correct / len(query_images)
            results[f'top_{k}_accuracy'] = accuracy
    
    return results


def create_visualizations(args, similarities, distances, labels, pair_types, 
                         twin_results, dataset):
    """Create evaluation visualizations."""
    print("Creating visualizations...")
    
    viz_dir = os.path.join(args.output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    # ROC curve
    from src.utils.metrics import plot_roc_curve
    from sklearn.metrics import roc_curve, auc
    
    fpr, tpr, _ = roc_curve(labels, similarities)
    auc_score = auc(fpr, tpr)
    
    roc_fig = plot_roc_curve(fpr, tpr, auc_score, 
                            save_path=os.path.join(viz_dir, 'roc_curve.png'))
    
    # Distance distributions
    twin_metrics = TwinSpecificMetrics()
    twin_metrics.update(distances, pair_types)
    
    dist_fig = twin_metrics.plot_distance_distributions(
        save_path=os.path.join(viz_dir, 'distance_distributions.png')
    )
    
    # Sample predictions visualization
    sample_pairs = []
    sample_predictions = []
    sample_labels = []
    
    # Sample some pairs for visualization
    n_samples = min(16, len(dataset))
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        sample_pairs.append((sample['img1_path'], sample['img2_path']))
        sample_predictions.append(similarities[idx])
        sample_labels.append(sample['label'])
    
    comparison_fig = create_pair_comparison_grid(
        sample_pairs, sample_predictions, sample_labels,
        save_path=os.path.join(viz_dir, 'sample_predictions.png')
    )
    
    print(f"Visualizations saved to {viz_dir}")


def save_results(args, verification_results, twin_results, identification_results):
    """Save evaluation results to files."""
    results = {
        'verification_metrics': verification_results,
        'twin_specific_metrics': twin_results,
        'identification_metrics': identification_results,
        'config': {
            'model_path': args.model,
            'data_path': args.data_path,
            'split': args.split,
            'batch_size': args.batch_size
        }
    }
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    results = convert_numpy(results)
    
    # Save to JSON
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")


def print_results(verification_results, twin_results, identification_results):
    """Print evaluation results."""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    print("\nVERIFICATION METRICS:")
    print("-" * 30)
    for category, metrics in verification_results.items():
        print(f"\n{category.upper()}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\nTWIN-SPECIFIC METRICS:")
    print("-" * 30)
    for metric, value in twin_results.items():
        if isinstance(value, (int, float)):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    if identification_results:
        print("\nIDENTIFICATION METRICS:")
        print("-" * 30)
        for metric, value in identification_results.items():
            print(f"  {metric}: {value:.4f}")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model}")
    inference_model = TwinVerificationInference(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load dataset
    print(f"Loading {args.split} dataset...")
    dataset = load_test_dataset(args, config)
    print(f"Dataset loaded: {len(dataset)} pairs")
    
    # Evaluate verification
    verification_results, twin_results, similarities, distances, labels, pair_types = \
        evaluate_verification(inference_model, dataset, args.batch_size)
    
    # Evaluate identification
    identification_results = evaluate_identification(inference_model, dataset)
    
    # Print results
    print_results(verification_results, twin_results, identification_results)
    
    # Save results
    save_results(args, verification_results, twin_results, identification_results)
    
    # Create visualizations
    if args.visualize:
        create_visualizations(args, similarities, distances, labels, pair_types,
                            twin_results, dataset)
    
    # Save embeddings
    if args.save_embeddings:
        print("Extracting and saving embeddings...")
        
        # Collect all unique images
        all_images = set()
        for i in range(len(dataset)):
            sample = dataset[i]
            all_images.add(sample['img1_path'])
            all_images.add(sample['img2_path'])
        
        all_images = list(all_images)
        
        # Extract embeddings
        embeddings = inference_model.extract_embeddings_batch(all_images, args.batch_size)
        
        # Save
        embeddings_path = os.path.join(args.output_dir, 'embeddings.npy')
        paths_path = os.path.join(args.output_dir, 'embedding_paths.json')
        
        np.save(embeddings_path, embeddings)
        with open(paths_path, 'w') as f:
            json.dump(all_images, f, indent=2)
        
        print(f"Embeddings saved to {embeddings_path}")
        print(f"Image paths saved to {paths_path}")
    
    print("\nEvaluation completed!")


if __name__ == "__main__":
    main()
