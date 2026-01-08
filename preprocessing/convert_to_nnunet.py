"""
Convert Dataset to nnU-Net Format

Converts CT scans and rPCI region segmentations to the nnU-Net dataset format.

Usage:
    python -m preprocessing.convert_to_nnunet \
        --images_dir /path/to/scans \
        --segmentations_dir /path/to/labels \
        --output_dir /path/to/nnUNet_raw \
        --dataset_name Dataset001_rPCI
"""

import os
import argparse
import shutil


# rPCI label definitions
PCI_LABELS = {
    "background": 0,
    "region0": 1,   # Central
    "region1": 2,   # Right upper
    "region2": 3,   # Epigastrium
    "region3": 4,   # Left upper
    "region4": 5,   # Left flank
    "region5": 6,   # Left lower
    "region6": 7,   # Pelvis
    "region7": 8,   # Right lower
    "region8": 9,   # Right flank
    "region9": 10,  # Upper jejunum
    "region10": 11, # Lower jejunum
    "region11": 12, # Upper ileum
    "region12": 13  # Lower ileum
}


def convert_to_nnunet_format(images_dir, segmentations_dir, output_dir, dataset_name,
                             image_prefix="Scan", seg_prefix="Segmentations"):
    """
    Convert dataset to nnU-Net format.
    
    Args:
        images_dir: Directory containing CT scans
        segmentations_dir: Directory containing segmentation masks
        output_dir: Output directory for nnU-Net dataset
        dataset_name: Name of the dataset (e.g., Dataset001_rPCI)
        image_prefix: Prefix for image files (default: "Scan")
        seg_prefix: Prefix for segmentation files (default: "Segmentations")
    
    Returns:
        tuple: (images_tr_dir, raw_data_dir)
    """
    print(f"Converting dataset to nnU-Net format: {dataset_name}")
    
    # Check if paths exist
    if not os.path.exists(images_dir):
        raise ValueError(f"Images directory not found: {images_dir}")
    if not os.path.exists(segmentations_dir):
        raise ValueError(f"Segmentations directory not found: {segmentations_dir}")
    
    # Create nnUNet directory structure
    raw_data_dir = os.path.join(output_dir, dataset_name)
    images_tr_dir = os.path.join(raw_data_dir, "imagesTr")
    labels_tr_dir = os.path.join(raw_data_dir, "labelsTr")
    images_ts_dir = os.path.join(raw_data_dir, "imagesTs")

    os.makedirs(images_tr_dir, exist_ok=True)
    os.makedirs(labels_tr_dir, exist_ok=True)
    os.makedirs(images_ts_dir, exist_ok=True)

    print(f"Output directory: {raw_data_dir}")
    print(f"Copying images to {images_tr_dir}")
    print(f"Copying segmentations to {labels_tr_dir}")
    
    # Check if images and segmentations are in the same directory
    same_directory = os.path.abspath(images_dir) == os.path.abspath(segmentations_dir)
    
    copied_images = 0
    copied_segs = 0
    
    if same_directory:
        # Process both types of files in a single loop
        for file in os.listdir(images_dir):
            if file.startswith(image_prefix) and file.endswith('.nii.gz'):
                # Extract case identifier
                case_id = file.split('_')[1].split('.')[0]
                src = os.path.join(images_dir, file)
                dst = os.path.join(images_tr_dir, f"{case_id}_0000.nii.gz")
                shutil.copy(src, dst)
                copied_images += 1
                
            elif file.startswith(seg_prefix) and file.endswith('.nii.gz'):
                case_id = file.split('_')[1].split('.')[0]
                src = os.path.join(segmentations_dir, file)
                dst = os.path.join(labels_tr_dir, f"{case_id}.nii.gz")
                shutil.copy(src, dst)
                copied_segs += 1
    else:
        # Process directories separately
        for file in os.listdir(images_dir):
            if file.startswith(image_prefix) and file.endswith('.nii.gz'):
                case_id = file.split('_')[1].split('.')[0]
                src = os.path.join(images_dir, file)
                dst = os.path.join(images_tr_dir, f"{case_id}_0000.nii.gz")
                shutil.copy(src, dst)
                copied_images += 1
            
        for file in os.listdir(segmentations_dir):
            if file.startswith(seg_prefix) and file.endswith('.nii.gz'):
                case_id = file.split('_')[1].split('.')[0]
                src = os.path.join(segmentations_dir, file)
                dst = os.path.join(labels_tr_dir, f"{case_id}.nii.gz")
                shutil.copy(src, dst)
                copied_segs += 1
    
    print(f"Copied {copied_images} images and {copied_segs} segmentations")
    
    return images_tr_dir, raw_data_dir


def generate_dataset_json(output_folder, num_training_cases, labels=None):
    """
    Generate dataset.json for nnU-Net.
    
    Args:
        output_folder: Path to dataset folder
        num_training_cases: Number of training cases
        labels: Label dictionary (defaults to PCI_LABELS)
    """
    import json
    
    if labels is None:
        labels = PCI_LABELS
    
    dataset_json = {
        "channel_names": {
            "0": "CT"
        },
        "labels": labels,
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": os.path.basename(output_folder),
        "description": "rPCI region segmentation from CT scans",
        "reference": "Eindhoven University of Technology",
        "licence": "See LICENSE",
        "release": "1.0",
        "overwrite_image_reader_writer": "SimpleITKIO"
    }
    
    json_path = os.path.join(output_folder, "dataset.json")
    with open(json_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print(f"Generated dataset.json at {json_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert dataset to nnU-Net format")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path to directory containing CT images")
    parser.add_argument("--segmentations_dir", type=str, required=True,
                        help="Path to directory containing segmentations")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Path to nnUNet_raw directory")
    parser.add_argument("--dataset_name", type=str, default="Dataset001_rPCI",
                        help="Name of the dataset")
    parser.add_argument("--image_prefix", type=str, default="Scan",
                        help="Prefix for image files")
    parser.add_argument("--seg_prefix", type=str, default="Segmentations",
                        help="Prefix for segmentation files")
    
    args = parser.parse_args()

    images_tr_dir, raw_data_dir = convert_to_nnunet_format(
        args.images_dir, 
        args.segmentations_dir, 
        args.output_dir, 
        args.dataset_name,
        args.image_prefix,
        args.seg_prefix
    )

    # Generate dataset.json
    num_training = len(os.listdir(images_tr_dir))
    generate_dataset_json(raw_data_dir, num_training)
    
    print("\n✓ Conversion complete!")
    print(f"\nNext steps:")
    print(f"1. Set environment variable: export nnUNet_raw=\"{os.path.dirname(raw_data_dir)}\"")
    print(f"2. Run: nnUNetv2_plan_and_preprocess -d {args.dataset_name.split('_')[0].replace('Dataset', '')} --verify_dataset_integrity")


if __name__ == "__main__":
    main()

