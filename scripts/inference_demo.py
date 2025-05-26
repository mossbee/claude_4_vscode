"""
Demo script for twin face verification inference.
Provides examples of how to use the trained model for various tasks.
"""

import os
import sys
import argparse
import yaml
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.inference.inference import TwinVerificationInference


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Twin face verification inference demo')
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str,
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    
    # Task selection
    parser.add_argument('--task', type=str, required=True,
                       choices=['verify', 'identify', 'extract', 'compare_gallery'],
                       help='Task to perform')
    
    # For verification task
    parser.add_argument('--image1', type=str,
                       help='Path to first image (for verification)')
    parser.add_argument('--image2', type=str,
                       help='Path to second image (for verification)')
    
    # For identification task
    parser.add_argument('--query', type=str,
                       help='Path to query image (for identification)')
    parser.add_argument('--gallery_dir', type=str,
                       help='Path to gallery directory (for identification)')
    
    # For extraction task
    parser.add_argument('--input_dir', type=str,
                       help='Input directory containing images')
    parser.add_argument('--output_file', type=str,
                       help='Output file for embeddings')
    
    # General options
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--output_dir', type=str, default='./inference_output',
                       help='Output directory for results')
    
    return parser.parse_args()


def verify_pair(inference_model, image1_path, image2_path, visualize=False, output_dir=None):
    """Verify if two images are of the same person."""
    print(f"Verifying pair:")
    print(f"  Image 1: {image1_path}")
    print(f"  Image 2: {image2_path}")
    
    # Compute similarity
    similarity, distance = inference_model.verify_pair(
        image1_path, image2_path, return_distance=True
    )
    
    print(f"\nResults:")
    print(f"  Similarity: {similarity:.4f}")
    print(f"  Distance: {distance:.4f}")
    
    # Interpret result
    if similarity > 0.7:
        result = "SAME PERSON (High confidence)"
    elif similarity > 0.5:
        result = "SAME PERSON (Medium confidence)"
    elif similarity > 0.3:
        result = "UNCERTAIN"
    else:
        result = "DIFFERENT PERSON"
    
    print(f"  Prediction: {result}")
    
    # Create visualization if requested
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        
        # Load and display images
        img1 = Image.open(image1_path).convert('RGB')
        img2 = Image.open(image2_path).convert('RGB')
        
        axes[0].imshow(img1)
        axes[0].set_title('Image 1')
        axes[0].axis('off')
        
        axes[1].imshow(img2)
        axes[1].set_title('Image 2')
        axes[1].axis('off')
        
        # Add result text
        fig.suptitle(f'Similarity: {similarity:.4f} | {result}', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'verification_result.png'), 
                   dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to {output_dir}/verification_result.png")
    
    return similarity, distance


def identify_person(inference_model, query_path, gallery_dir, top_k=5, 
                   visualize=False, output_dir=None):
    """Identify a person against a gallery."""
    print(f"Identifying person:")
    print(f"  Query: {query_path}")
    print(f"  Gallery: {gallery_dir}")
    
    # Build gallery
    gallery_images = []
    gallery_labels = []
    
    for img_file in os.listdir(gallery_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            img_path = os.path.join(gallery_dir, img_file)
            gallery_images.append(img_path)
            # Use filename (without extension) as label
            gallery_labels.append(os.path.splitext(img_file)[0])
    
    if not gallery_images:
        print("Error: No images found in gallery directory")
        return
    
    print(f"  Gallery size: {len(gallery_images)} images")
    
    # Extract gallery embeddings
    print("Extracting gallery embeddings...")
    gallery_embeddings = inference_model.extract_embeddings_batch(gallery_images)
    
    # Perform identification
    print("Performing identification...")
    matches = inference_model.identify(
        query_path, gallery_embeddings, gallery_labels, top_k=top_k
    )
    
    print(f"\nTop {top_k} matches:")
    for i, (label, similarity) in enumerate(matches):
        print(f"  {i+1}. {label}: {similarity:.4f}")
    
    # Create visualization if requested
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        n_show = min(top_k, 5)  # Show top 5 matches
        fig, axes = plt.subplots(2, n_show + 1, figsize=(4 * (n_show + 1), 8))
        
        # Show query image
        query_img = Image.open(query_path).convert('RGB')
        axes[0, 0].imshow(query_img)
        axes[0, 0].set_title('Query')
        axes[0, 0].axis('off')
        axes[1, 0].axis('off')
        
        # Show top matches
        for i, (label, similarity) in enumerate(matches[:n_show]):
            # Find corresponding image path
            gallery_img_path = None
            for path, gal_label in zip(gallery_images, gallery_labels):
                if gal_label == label:
                    gallery_img_path = path
                    break
            
            if gallery_img_path:
                gallery_img = Image.open(gallery_img_path).convert('RGB')
                axes[0, i + 1].imshow(gallery_img)
                axes[0, i + 1].set_title(f'#{i+1}: {label}')
                axes[0, i + 1].axis('off')
                
                axes[1, i + 1].text(0.5, 0.5, f'{similarity:.4f}', 
                                   transform=axes[1, i + 1].transAxes,
                                   fontsize=14, ha='center', va='center')
                axes[1, i + 1].set_title('Similarity')
                axes[1, i + 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'identification_result.png'),
                   dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to {output_dir}/identification_result.png")
    
    return matches


def extract_embeddings(inference_model, input_dir, output_file):
    """Extract embeddings for all images in a directory."""
    print(f"Extracting embeddings:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output file: {output_file}")
    
    # Collect all images
    image_paths = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_paths.append(os.path.join(root, file))
    
    if not image_paths:
        print("Error: No images found in input directory")
        return
    
    print(f"  Found {len(image_paths)} images")
    
    # Extract embeddings
    print("Extracting embeddings...")
    inference_model.save_embeddings(image_paths, output_file)
    
    print("Embeddings extracted successfully!")


def compare_gallery(inference_model, gallery_dir, visualize=False, output_dir=None):
    """Compare all images in a gallery and find most similar pairs."""
    print(f"Comparing gallery:")
    print(f"  Gallery: {gallery_dir}")
    
    # Collect images
    image_paths = []
    for img_file in os.listdir(gallery_dir):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_paths.append(os.path.join(gallery_dir, img_file))
    
    if len(image_paths) < 2:
        print("Error: Need at least 2 images for comparison")
        return
    
    print(f"  Found {len(image_paths)} images")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = inference_model.extract_embeddings_batch(image_paths)
    
    # Compute all pairwise similarities
    print("Computing pairwise similarities...")
    similarities = []
    pairs = []
    
    for i in range(len(image_paths)):
        for j in range(i + 1, len(image_paths)):
            similarity = inference_model.compute_similarity(embeddings[i], embeddings[j])
            similarities.append(similarity)
            pairs.append((image_paths[i], image_paths[j], similarity))
    
    # Sort by similarity (highest first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    print(f"\nTop 10 most similar pairs:")
    for i, (img1, img2, sim) in enumerate(pairs[:10]):
        img1_name = os.path.basename(img1)
        img2_name = os.path.basename(img2)
        print(f"  {i+1}. {img1_name} <-> {img2_name}: {sim:.4f}")
    
    print(f"\nTop 10 least similar pairs:")
    for i, (img1, img2, sim) in enumerate(pairs[-10:]):
        img1_name = os.path.basename(img1)
        img2_name = os.path.basename(img2)
        print(f"  {i+1}. {img1_name} <-> {img2_name}: {sim:.4f}")
    
    # Create visualization if requested
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Show top 5 most similar pairs
        fig, axes = plt.subplots(5, 2, figsize=(8, 20))
        
        for i in range(5):
            if i < len(pairs):
                img1_path, img2_path, similarity = pairs[i]
                
                img1 = Image.open(img1_path).convert('RGB')
                img2 = Image.open(img2_path).convert('RGB')
                
                axes[i, 0].imshow(img1)
                axes[i, 0].set_title(f'Pair {i+1} - Image 1\nSimilarity: {similarity:.4f}')
                axes[i, 0].axis('off')
                
                axes[i, 1].imshow(img2)
                axes[i, 1].set_title(f'Pair {i+1} - Image 2\nSimilarity: {similarity:.4f}')
                axes[i, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'gallery_comparison.png'),
                   dpi=300, bbox_inches='tight')
        
        print(f"Visualization saved to {output_dir}/gallery_comparison.png")
    
    return pairs


def main():
    """Main function."""
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
    
    print("Model loaded successfully!\n")
    
    # Perform requested task
    if args.task == 'verify':
        if not args.image1 or not args.image2:
            print("Error: --image1 and --image2 are required for verification task")
            return
        
        verify_pair(inference_model, args.image1, args.image2, 
                   args.visualize, args.output_dir)
    
    elif args.task == 'identify':
        if not args.query or not args.gallery_dir:
            print("Error: --query and --gallery_dir are required for identification task")
            return
        
        identify_person(inference_model, args.query, args.gallery_dir,
                       visualize=args.visualize, output_dir=args.output_dir)
    
    elif args.task == 'extract':
        if not args.input_dir or not args.output_file:
            print("Error: --input_dir and --output_file are required for extraction task")
            return
        
        extract_embeddings(inference_model, args.input_dir, args.output_file)
    
    elif args.task == 'compare_gallery':
        if not args.gallery_dir:
            print("Error: --gallery_dir is required for gallery comparison task")
            return
        
        compare_gallery(inference_model, args.gallery_dir,
                       args.visualize, args.output_dir)
    
    print("\nInference completed!")


if __name__ == "__main__":
    main()
