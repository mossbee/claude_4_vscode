"""
Script to prepare and validate the twin face dataset.
Helps convert datasets to the expected format and create pairs.json file.
"""

import os
import sys
import argparse
import json
import shutil
from collections import defaultdict
import re

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Prepare twin face dataset')
    
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing the dataset')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for processed dataset')
    parser.add_argument('--format', type=str, default='auto',
                       choices=['auto', 'twins_db', 'custom'],
                       help='Input dataset format')
    parser.add_argument('--pairs_file', type=str,
                       help='Existing pairs file (for custom format)')
    parser.add_argument('--min_images_per_identity', type=int, default=2,
                       help='Minimum number of images per identity')
    parser.add_argument('--dry_run', action='store_true',
                       help='Only validate, do not copy files')
    
    return parser.parse_args()


def detect_dataset_format(input_dir):
    """Automatically detect the dataset format."""
    print("Detecting dataset format...")
    
    # Check if there's already a pairs.json file
    if os.path.exists(os.path.join(input_dir, 'pairs.json')):
        print("Found pairs.json file - assuming custom format")
        return 'custom'
    
    # Look for folder patterns
    folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    
    # Check for twins_db format (folders like "001_1", "001_2" for twins)
    twin_pattern = re.compile(r'(\d+)_(\d+)')
    twins_db_folders = [f for f in folders if twin_pattern.match(f)]
    
    if len(twins_db_folders) > len(folders) * 0.8:  # If 80% of folders match twin pattern
        print("Detected twins_db format")
        return 'twins_db'
    
    print("Could not detect format automatically - please specify --format")
    return 'auto'


def process_twins_db_format(input_dir, output_dir, min_images_per_identity):
    """Process dataset in twins_db format (folders like 001_1, 001_2)."""
    print("Processing twins_db format...")
    
    # Group folders by twin pair
    twin_groups = defaultdict(list)
    twin_pattern = re.compile(r'(\d+)_(\d+)')
    
    for folder in os.listdir(input_dir):
        folder_path = os.path.join(input_dir, folder)
        if not os.path.isdir(folder_path):
            continue
        
        match = twin_pattern.match(folder)
        if match:
            pair_id = match.group(1)
            twin_id = match.group(2)
            twin_groups[pair_id].append((folder, twin_id))
    
    # Process twin pairs
    twin_pairs = []
    processed_folders = []
    folder_counter = 1
    
    for pair_id, twins in twin_groups.items():
        if len(twins) != 2:
            print(f"Warning: Twin pair {pair_id} has {len(twins)} members, skipping")
            continue
        
        # Sort by twin_id to ensure consistent ordering
        twins.sort(key=lambda x: x[1])
        
        valid_twins = []
        for orig_folder, twin_id in twins:
            orig_path = os.path.join(input_dir, orig_folder)
            
            # Count images
            images = [f for f in os.listdir(orig_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            if len(images) >= min_images_per_identity:
                new_folder = f"img_folder_{folder_counter}"
                valid_twins.append((orig_folder, new_folder))
                folder_counter += 1
            else:
                print(f"Warning: {orig_folder} has only {len(images)} images, skipping")
        
        if len(valid_twins) == 2:
            twin_pairs.append((valid_twins[0][1], valid_twins[1][1]))
            processed_folders.extend(valid_twins)
    
    return processed_folders, twin_pairs


def process_custom_format(input_dir, pairs_file, min_images_per_identity):
    """Process dataset with custom pairs file."""
    print("Processing custom format...")
    
    # Load pairs file
    if not pairs_file:
        pairs_file = os.path.join(input_dir, 'pairs.json')
    
    if not os.path.exists(pairs_file):
        raise FileNotFoundError(f"Pairs file not found: {pairs_file}")
    
    with open(pairs_file, 'r') as f:
        twin_pairs = json.load(f)
    
    print(f"Loaded {len(twin_pairs)} twin pairs from {pairs_file}")
    
    # Validate that all folders exist and have enough images
    valid_pairs = []
    processed_folders = []
    folder_counter = 1
    
    for pair in twin_pairs:
        if len(pair) != 2:
            print(f"Warning: Invalid pair format {pair}, skipping")
            continue
        
        folder1, folder2 = pair
        path1 = os.path.join(input_dir, folder1)
        path2 = os.path.join(input_dir, folder2)
        
        if not os.path.exists(path1):
            print(f"Warning: Folder {folder1} not found, skipping pair")
            continue
        
        if not os.path.exists(path2):
            print(f"Warning: Folder {folder2} not found, skipping pair")
            continue
        
        # Count images
        images1 = [f for f in os.listdir(path1) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        images2 = [f for f in os.listdir(path2) 
                  if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        
        if len(images1) < min_images_per_identity:
            print(f"Warning: {folder1} has only {len(images1)} images, skipping pair")
            continue
        
        if len(images2) < min_images_per_identity:
            print(f"Warning: {folder2} has only {len(images2)} images, skipping pair")
            continue
        
        # Create new folder names
        new_folder1 = f"img_folder_{folder_counter}"
        new_folder2 = f"img_folder_{folder_counter + 1}"
        folder_counter += 2
        
        valid_pairs.append((new_folder1, new_folder2))
        processed_folders.append((folder1, new_folder1))
        processed_folders.append((folder2, new_folder2))
    
    return processed_folders, valid_pairs


def copy_dataset(input_dir, output_dir, processed_folders, dry_run=False):
    """Copy the processed dataset to output directory."""
    if dry_run:
        print("DRY RUN: Would copy the following folders:")
        for orig_folder, new_folder in processed_folders:
            orig_path = os.path.join(input_dir, orig_folder)
            new_path = os.path.join(output_dir, new_folder)
            
            images = [f for f in os.listdir(orig_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            
            print(f"  {orig_folder} -> {new_folder} ({len(images)} images)")
        return
    
    print("Copying dataset...")
    os.makedirs(output_dir, exist_ok=True)
    
    for orig_folder, new_folder in processed_folders:
        orig_path = os.path.join(input_dir, orig_folder)
        new_path = os.path.join(output_dir, new_folder)
        
        print(f"Copying {orig_folder} -> {new_folder}")
        
        if os.path.exists(new_path):
            shutil.rmtree(new_path)
        
        shutil.copytree(orig_path, new_path)


def save_pairs_file(output_dir, twin_pairs, dry_run=False):
    """Save the pairs.json file."""
    pairs_path = os.path.join(output_dir, 'pairs.json')
    
    if dry_run:
        print(f"DRY RUN: Would save {len(twin_pairs)} pairs to {pairs_path}")
        print("Sample pairs:")
        for i, pair in enumerate(twin_pairs[:5]):
            print(f"  {pair}")
        if len(twin_pairs) > 5:
            print(f"  ... and {len(twin_pairs) - 5} more")
        return
    
    print(f"Saving {len(twin_pairs)} twin pairs to {pairs_path}")
    
    with open(pairs_path, 'w') as f:
        json.dump(twin_pairs, f, indent=2)


def validate_output_dataset(output_dir):
    """Validate the output dataset structure."""
    print("Validating output dataset...")
    
    # Check pairs.json
    pairs_path = os.path.join(output_dir, 'pairs.json')
    if not os.path.exists(pairs_path):
        print("Error: pairs.json not found in output directory")
        return False
    
    with open(pairs_path, 'r') as f:
        twin_pairs = json.load(f)
    
    # Check folders
    folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
    
    # Validate that all pairs reference existing folders
    all_folders_in_pairs = set()
    for pair in twin_pairs:
        all_folders_in_pairs.update(pair)
    
    missing_folders = all_folders_in_pairs - set(folders)
    if missing_folders:
        print(f"Error: Folders referenced in pairs.json but not found: {missing_folders}")
        return False
    
    # Count images in each folder
    total_images = 0
    for folder in folders:
        folder_path = os.path.join(output_dir, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        total_images += len(images)
    
    print(f"Validation successful:")
    print(f"  {len(twin_pairs)} twin pairs")
    print(f"  {len(folders)} identity folders")
    print(f"  {total_images} total images")
    
    return True


def print_dataset_statistics(input_dir, output_dir=None):
    """Print dataset statistics."""
    print("\nDATASET STATISTICS:")
    print("=" * 50)
    
    # Input statistics
    print(f"Input directory: {input_dir}")
    input_folders = [f for f in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, f))]
    input_images = 0
    
    for folder in input_folders:
        folder_path = os.path.join(input_dir, folder)
        images = [f for f in os.listdir(folder_path) 
                 if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
        input_images += len(images)
    
    print(f"  Input folders: {len(input_folders)}")
    print(f"  Input images: {input_images}")
    
    if output_dir and os.path.exists(output_dir):
        print(f"\nOutput directory: {output_dir}")
        output_folders = [f for f in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, f))]
        output_images = 0
        
        for folder in output_folders:
            folder_path = os.path.join(output_dir, folder)
            images = [f for f in os.listdir(folder_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]
            output_images += len(images)
        
        print(f"  Output folders: {len(output_folders)}")
        print(f"  Output images: {output_images}")
        
        # Load pairs if available
        pairs_path = os.path.join(output_dir, 'pairs.json')
        if os.path.exists(pairs_path):
            with open(pairs_path, 'r') as f:
                twin_pairs = json.load(f)
            print(f"  Twin pairs: {len(twin_pairs)}")


def main():
    """Main function."""
    args = parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # Detect format if auto
    dataset_format = args.format
    if dataset_format == 'auto':
        dataset_format = detect_dataset_format(args.input_dir)
        if dataset_format == 'auto':
            print("Error: Could not detect dataset format")
            return
    
    # Process dataset based on format
    if dataset_format == 'twins_db':
        processed_folders, twin_pairs = process_twins_db_format(
            args.input_dir, args.output_dir, args.min_images_per_identity
        )
    elif dataset_format == 'custom':
        processed_folders, twin_pairs = process_custom_format(
            args.input_dir, args.pairs_file, args.min_images_per_identity
        )
    else:
        print(f"Error: Unknown dataset format: {dataset_format}")
        return
    
    print(f"Processed {len(processed_folders)} folders into {len(twin_pairs)} twin pairs")
    
    # Copy dataset
    copy_dataset(args.input_dir, args.output_dir, processed_folders, args.dry_run)
    
    # Save pairs file
    save_pairs_file(args.output_dir, twin_pairs, args.dry_run)
    
    # Validate output (only if not dry run)
    if not args.dry_run:
        validate_output_dataset(args.output_dir)
    
    # Print statistics
    print_dataset_statistics(args.input_dir, args.output_dir if not args.dry_run else None)
    
    if args.dry_run:
        print("\nDry run completed. Use without --dry_run to actually process the dataset.")
    else:
        print(f"\nDataset preparation completed! Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
