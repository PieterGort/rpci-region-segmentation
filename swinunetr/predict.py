"""
SwinUNETR Inference Script

This module provides inference utilities for trained SwinUNETR models.
"""

import os
import json
import argparse
import numpy as np
import torch
import SimpleITK as sitk
from monai.transforms import (
    Compose,
    LoadImaged,
    ScaleIntensityRanged,
    CropForegroundd,
    Orientationd,
    Spacingd,
    EnsureTyped
)
from monai.data import CacheDataset, ThreadDataLoader
from monai.inferers import sliding_window_inference

from .model import load_model


def get_inference_transforms(
    device: str,
    a_min: float = -175,
    a_max: float = 250,
    pixdim: tuple = (1.5, 1.5, 1.5),
) -> Compose:
    """
    Get transforms for inference.
    
    Args:
        device: Device for tensors
        a_min: Minimum intensity for normalization
        a_max: Maximum intensity for normalization
        pixdim: Target voxel spacing
    
    Returns:
        Composed transforms
    """
    return Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min, a_max=a_max,
            b_min=0.0, b_max=1.0,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=pixdim,
            mode=("bilinear", "nearest")
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])


def load_validation_samples(
    fold_json_path: str,
    device: str = 'cuda',
    pixdim: tuple = (1.5, 1.5, 1.5),
):
    """
    Load validation samples from a fold JSON file.
    
    Args:
        fold_json_path: Path to fold JSON file
        device: Device for inference
        pixdim: Target voxel spacing
    
    Returns:
        Tuple of (data_loader, validation_data_list)
    """
    with open(fold_json_path, 'r') as f:
        fold_data = json.load(f)
    
    validation_data = fold_data["validation"]
    transforms = get_inference_transforms(device, pixdim=pixdim)
    
    val_ds = CacheDataset(
        data=validation_data,
        transform=transforms,
        cache_num=len(validation_data),
        cache_rate=1.0,
        num_workers=0
    )
    val_loader = ThreadDataLoader(val_ds, num_workers=0, batch_size=1)
    
    return val_loader, validation_data


def predict_and_save(
    model,
    val_loader,
    validation_data: list,
    output_dir: str,
    device: str,
    roi_size: tuple = (96, 96, 96),
    sw_batch_size: int = 4,
    overlap: float = 0.8,
    pixdim: tuple = (1.5, 1.5, 1.5),
    verbose: bool = False,
) -> None:
    """
    Run inference on validation samples and save predictions.
    
    Args:
        model: Trained model
        val_loader: Validation data loader
        validation_data: List of validation data dictionaries
        output_dir: Directory to save predictions
        device: Device for inference
        roi_size: ROI size for sliding window inference
        sw_batch_size: Batch size for sliding window
        overlap: Overlap ratio for sliding window
        pixdim: Voxel spacing used during preprocessing
        verbose: Print debug information
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        for i, batch_data in enumerate(val_loader):
            val_inputs = batch_data["image"].to(device)
            
            # Run inference
            with torch.no_grad():
                val_outputs = sliding_window_inference(
                    val_inputs, roi_size, sw_batch_size, model, overlap=overlap
                )
            
            if verbose:
                print(f"Output shape: {val_outputs.shape}")
                print(f"Input shape: {val_inputs.shape}")
            
            # Get original image path
            original_image_path = validation_data[i]["image"]
            image_filename = os.path.basename(original_image_path)
            
            # Extract identifier
            identifier = image_filename
            for suffix in ["_0000.nii.gz", "_TS.nii.gz", ".nii.gz"]:
                identifier = identifier.replace(suffix, "")
            for prefix in ["Scan_"]:
                identifier = identifier.replace(prefix, "")
            
            output_path = os.path.join(output_dir, f"{identifier}.nii.gz")
            
            # Convert prediction to segmentation labels
            pred_array = torch.argmax(val_outputs, dim=1).cpu().numpy()[0].astype(np.uint8)
            
            # Handle orientation (may need adjustment based on your data)
            pred_array = np.flip(pred_array, axis=0)
            pred_array = np.flip(pred_array, axis=1)
            pred_array = pred_array.copy()
            pred_array_corrected = np.transpose(pred_array, (2, 1, 0))
            
            if verbose:
                print(f"Prediction shape: {pred_array_corrected.shape}")
            
            # Create SimpleITK image
            prediction_img = sitk.GetImageFromArray(pred_array_corrected)
            original_img = sitk.ReadImage(original_image_path)
            
            # Set spacing and origin
            prediction_img.SetSpacing(pixdim)
            prediction_img.SetOrigin(original_img.GetOrigin())
            
            # Resample to original space
            resampler = sitk.ResampleImageFilter()
            resampler.SetReferenceImage(original_img)
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)
            resampler.SetDefaultPixelValue(0)
            
            resampled_prediction = resampler.Execute(prediction_img)
            sitk.WriteImage(resampled_prediction, output_path)
            
            if verbose:
                print(f"Resampled size: {resampled_prediction.GetSize()}")
                print(f"Original size: {original_img.GetSize()}")
            
            print(f"Saved prediction: {output_path}")


def main():
    """Main inference function."""
    parser = argparse.ArgumentParser(description="SwinUNETR Inference Script")
    parser.add_argument(
        '--model-path', type=str, required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--fold-json-path', type=str, required=True,
        help='Path to fold JSON file with validation samples'
    )
    parser.add_argument(
        '--output-dir', type=str, required=True,
        help='Directory to save predictions'
    )
    parser.add_argument(
        '--num-classes', type=int, default=14,
        help='Number of output classes'
    )
    parser.add_argument(
        '--verbose', action='store_true',
        help='Enable verbose output'
    )
    
    args = parser.parse_args()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model
    model = load_model(
        args.model_path,
        device=device,
        out_channels=args.num_classes
    )
    print(f"Loaded model from: {args.model_path}")
    
    # Load data
    val_loader, validation_data = load_validation_samples(args.fold_json_path, device)
    print(f"Loaded {len(validation_data)} validation samples")
    
    # Run inference
    predict_and_save(
        model, val_loader, validation_data,
        args.output_dir, device,
        verbose=args.verbose
    )
    print(f"✓ Predictions saved to: {args.output_dir}")


if __name__ == "__main__":
    main()

