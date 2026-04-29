"""
Dataset Utilities for SwinUNETR

This module provides data loading and preprocessing utilities.
"""

import os
import json
import logging
from glob import glob
from typing import List, Dict, Tuple

import numpy as np
from sklearn.model_selection import KFold


def get_scan_files(input_folder: str, pattern: str = "*.nii.gz") -> List[str]:
    """
    Get all scan files from the input folder.
    
    Args:
        input_folder: Path to folder containing scans
        pattern: Glob pattern for finding scan files
    
    Returns:
        Sorted list of scan file paths
    """
    return sorted(glob(os.path.join(input_folder, pattern)))


def build_dataset(
    scan_files: List[str],
    input_folder: str,
    image_suffix: str = "_0000.nii.gz",
    label_suffix: str = ".nii.gz",
) -> List[Dict[str, str]]:
    """
    Build dataset dictionary from scan files.
    
    Args:
        scan_files: List of scan file paths
        input_folder: Base input folder
        image_suffix: Suffix for image files
        label_suffix: Suffix for label files
    
    Returns:
        List of dictionaries with 'image' and 'label' keys
    """
    dataset = []
    
    for scan_path in scan_files:
        scan_name = os.path.basename(scan_path)
        
        # Try to find corresponding label
        # Adjust this logic based on your naming convention
        if image_suffix in scan_name:
            identifier = scan_name.replace(image_suffix, "")
            label_path = os.path.join(input_folder, "labelsTr", f"{identifier}{label_suffix}")
        else:
            identifier = os.path.splitext(os.path.splitext(scan_name)[0])[0]
            label_path = os.path.join(input_folder, "labelsTr", f"{identifier}{label_suffix}")
        
        if os.path.exists(label_path):
            dataset.append({
                "image": scan_path,
                "label": label_path
            })
        else:
            logging.warning(f"No segmentation found for {scan_path}")
    
    logging.info(f"Total matched samples: {len(dataset)}")
    return dataset


def generate_kfold_datasets(
    dataset: List[Dict[str, str]],
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    """
    Generate k-fold datasets for cross-validation.
    
    Args:
        dataset: Full dataset
        n_splits: Number of folds
        random_state: Random seed for reproducibility
    
    Returns:
        List of (train_data, val_data) tuples for each fold
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = []
    
    for train_index, val_index in kf.split(dataset):
        train_set = [dataset[i] for i in train_index]
        val_set = [dataset[i] for i in val_index]
        folds.append((train_set, val_set))
    
    return folds


def save_folds_to_json(
    folds: List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]],
    output_dir: str,
) -> None:
    """
    Save each fold's training and validation datasets to JSON files.
    
    Args:
        folds: List of (train_set, val_set) tuples
        output_dir: Directory to save JSON files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (train_set, val_set) in enumerate(folds):
        fold_data = {
            "training": train_set,
            "validation": val_set
        }
        fold_json_path = os.path.join(output_dir, f"fold_{i}.json")
        with open(fold_json_path, "w") as f:
            json.dump(fold_data, f, indent=2)


def prepare_kfold_datasets(
    data_dir: str,
    output_dir: str,
    n_splits: int = 5,
    image_pattern: str = "*_0000.nii.gz",
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    """
    Prepare k-fold datasets from a data directory.
    
    Args:
        data_dir: Path to data directory with imagesTr and labelsTr subdirs
        output_dir: Directory to save fold JSON files
        n_splits: Number of folds
        image_pattern: Glob pattern for finding image files
    
    Returns:
        List of (train_data, val_data) tuples
    """
    logging.basicConfig(level=logging.INFO)
    
    images_dir = os.path.join(data_dir, "imagesTr")
    scan_files = get_scan_files(images_dir, image_pattern)
    dataset = build_dataset(scan_files, data_dir)
    
    folds = generate_kfold_datasets(dataset, n_splits)
    save_folds_to_json(folds, output_dir)
    
    return folds


def load_predefined_splits(
    json_path: str,
    data_dir: str,
) -> List[Tuple[List[Dict[str, str]], List[Dict[str, str]]]]:
    """
    Load predefined splits from JSON file and convert to dataset format.
    
    The JSON file should have the format:
    [
        {"train": ["case_001", "case_002", ...], "val": ["case_010", ...]},
        ...
    ]
    
    Args:
        json_path: Path to splits JSON file
        data_dir: Path to data directory
    
    Returns:
        List of (train_data, val_data) tuples for each fold
    """
    with open(json_path, 'r') as f:
        splits = json.load(f)
    
    # Build full dataset mapping
    images_dir = os.path.join(data_dir, "imagesTr") if os.path.exists(os.path.join(data_dir, "imagesTr")) else data_dir
    labels_dir = os.path.join(data_dir, "labelsTr") if os.path.exists(os.path.join(data_dir, "labelsTr")) else data_dir
    
    # Try different patterns to find images
    patterns = ["*_0000.nii.gz", "Scan_*.nii.gz", "*.nii.gz"]
    scan_files = []
    for pattern in patterns:
        scan_files = sorted(glob(os.path.join(images_dir, pattern)))
        if scan_files:
            break
    
    # Build identifier -> paths mapping
    dataset = []
    for scan_path in scan_files:
        scan_name = os.path.basename(scan_path)
        
        # Extract identifier (customize based on naming convention)
        identifier = scan_name
        for suffix in ["_0000.nii.gz", "_TS.nii.gz", ".nii.gz"]:
            identifier = identifier.replace(suffix, "")
        for prefix in ["Scan_"]:
            identifier = identifier.replace(prefix, "")
        
        # Find corresponding label
        label_patterns = [
            os.path.join(labels_dir, f"{identifier}.nii.gz"),
            os.path.join(labels_dir, f"Segmentations_{identifier}_all_expanded.nii.gz"),
            os.path.join(labels_dir, f"{identifier}_seg.nii.gz"),
        ]
        
        label_path = None
        for lp in label_patterns:
            if os.path.exists(lp):
                label_path = lp
                break
        
        if label_path:
            dataset.append({
                "image": scan_path,
                "label": label_path,
                "identifier": identifier
            })
    
    # Convert splits to dataset format
    monai_splits = []
    for split in splits:
        train_ids = set(split["train"])
        val_ids = set(split["val"])
        
        train_data = [
            {"image": item["image"], "label": item["label"]}
            for item in dataset if item["identifier"] in train_ids
        ]
        val_data = [
            {"image": item["image"], "label": item["label"]}
            for item in dataset if item["identifier"] in val_ids
        ]
        
        monai_splits.append((train_data, val_data))
    
    return monai_splits


def generate_dataset_json(
    dataset: List[Dict[str, str]],
    description: str = "Dataset",
    labels: Dict[str, int] = None,
    train_ratio: float = 0.8,
) -> Dict:
    """
    Generate a JSON structure for the dataset.
    
    Args:
        dataset: List of image/label dictionaries
        description: Dataset description
        labels: Label name to ID mapping
        train_ratio: Ratio of data for training
    
    Returns:
        Dataset JSON structure
    """
    if labels is None:
        labels = {
            "background": 0,
            **{f"region{i}": i for i in range(1, 14)}
        }
    
    n_train = int(train_ratio * len(dataset))
    
    return {
        "description": description,
        "labels": labels,
        "licence": "MIT License",
        "modality": {"0": "CT"},
        "name": description,
        "numTest": 0,
        "numTraining": n_train,
        "numVal": len(dataset) - n_train,
        "tensorImageSize": "3D",
        "training": dataset[:n_train],
        "validation": dataset[n_train:]
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Prepare dataset splits")
    parser.add_argument("--data-dir", type=str, required=True, help="Path to data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to output directory")
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds")
    
    args = parser.parse_args()
    prepare_kfold_datasets(args.data_dir, args.output_dir, args.n_splits)

