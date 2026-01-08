"""
Postprocessing for rPCI Segmentations

Removes small connected components from segmentation predictions,
keeping only the largest connected component per class.

Usage:
    python -m postprocessing.postprocess \
        --input_folder /path/to/predictions \
        --output_folder /path/to/postprocessed
"""

import os
import argparse
import logging
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy.ndimage import label
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def find_largest_connected_component(segmentation):
    """
    Find and return the largest connected component in a binary segmentation.
    
    Args:
        segmentation: Binary segmentation mask (numpy array)
        
    Returns:
        Binary mask with only the largest connected component
    """
    if segmentation.sum() == 0:
        logger.debug("Empty segmentation found")
        return segmentation
    
    # Label connected components
    labeled_array, num_features = label(segmentation)
    
    if num_features == 0:
        return segmentation
    
    # Find the largest component (excluding background label 0)
    component_sizes = np.bincount(labeled_array.ravel())
    component_sizes[0] = 0  # Skip background
    largest_component_label = np.argmax(component_sizes)
    
    # Create mask with only the largest component
    largest_component_mask = (labeled_array == largest_component_label).astype(segmentation.dtype)
    
    logger.debug(f"Found {num_features} components, kept largest with {component_sizes[largest_component_label]} voxels")
    
    return largest_component_mask


def process_multi_class_segmentation(segmentation):
    """
    Process multi-class segmentation by applying largest connected component
    filtering to each class separately.
    
    Args:
        segmentation: Multi-class segmentation (numpy array)
        
    Returns:
        Processed segmentation with largest components only
    """
    unique_labels = np.unique(segmentation)
    unique_labels = unique_labels[unique_labels != 0]  # Remove background
    
    if len(unique_labels) == 0:
        return segmentation
    
    processed_segmentation = np.zeros_like(segmentation)
    
    for label_value in unique_labels:
        # Create binary mask for this class
        binary_mask = (segmentation == label_value).astype(np.uint8)
        
        # Find largest connected component
        largest_component = find_largest_connected_component(binary_mask)
        
        # Add to processed segmentation
        processed_segmentation[largest_component == 1] = label_value
    
    return processed_segmentation


def process_segmentation_file(input_path, output_path, multi_class=True):
    """
    Process a single segmentation file.
    
    Args:
        input_path: Path to input segmentation file
        output_path: Path to output processed file
        multi_class: Whether to treat as multi-class segmentation
    """
    try:
        # Load segmentation
        nii_img = nib.load(input_path)
        segmentation = nii_img.get_fdata().astype(np.uint8)
        
        logger.info(f"Processing {input_path}")
        logger.debug(f"Shape: {segmentation.shape}, Labels: {np.unique(segmentation)}")
        
        # Process segmentation
        if multi_class:
            processed_segmentation = process_multi_class_segmentation(segmentation)
        else:
            binary_seg = (segmentation > 0).astype(np.uint8)
            processed_segmentation = find_largest_connected_component(binary_seg)
        
        # Save processed segmentation
        processed_nii = nib.Nifti1Image(processed_segmentation, nii_img.affine, nii_img.header)
        nib.save(processed_nii, output_path)
        
        logger.info(f"Saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error processing {input_path}: {str(e)}")
        raise


def process_segmentation_folder(input_folder, output_folder, multi_class=True, file_pattern="*.nii.gz"):
    """
    Process all segmentation files in a folder.
    
    Args:
        input_folder: Path to folder containing segmentations
        output_folder: Path to output folder
        multi_class: Whether to treat as multi-class segmentation
        file_pattern: File pattern to match
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    seg_files = list(input_path.glob(file_pattern))
    
    if not seg_files:
        logger.warning(f"No files found matching {file_pattern} in {input_folder}")
        return
    
    logger.info(f"Found {len(seg_files)} files to process")
    
    for seg_file in tqdm(seg_files, desc="Postprocessing"):
        output_file = output_path / seg_file.name
        process_segmentation_file(str(seg_file), str(output_file), multi_class)


def main():
    parser = argparse.ArgumentParser(
        description="Remove small connected components from segmentations"
    )
    
    parser.add_argument("--input_folder", type=str, required=True,
                        help="Path to folder containing segmentation files")
    parser.add_argument("--output_folder", type=str, required=True,
                        help="Path to output folder for processed segmentations")
    parser.add_argument("--multi_class", action="store_true", default=True,
                        help="Treat as multi-class segmentation (default)")
    parser.add_argument("--binary", action="store_true",
                        help="Treat as binary segmentation")
    parser.add_argument("--file_pattern", type=str, default="*.nii.gz",
                        help="File pattern to match")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_folder):
        raise ValueError(f"Input folder does not exist: {args.input_folder}")
    
    multi_class = not args.binary
    
    logger.info(f"Processing mode: {'Multi-class' if multi_class else 'Binary'}")
    
    process_segmentation_folder(
        args.input_folder,
        args.output_folder,
        multi_class=multi_class,
        file_pattern=args.file_pattern
    )
    
    logger.info("✓ Postprocessing completed!")


if __name__ == "__main__":
    main()

