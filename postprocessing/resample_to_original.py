"""
Resample Predictions to Original Image Space

Resamples segmentation predictions from preprocessed/cropped space
back to the original CT image space.

Usage:
    python -m postprocessing.resample_to_original \
        --original_dir /path/to/original/images \
        --prediction_dir /path/to/predictions \
        --output_dir /path/to/resampled
"""

import os
import argparse
import SimpleITK as sitk
import numpy as np


def resample_prediction_to_original_space(original_img_path, prediction_path, output_path):
    """
    Resample a prediction from preprocessed space back to original image space.
    
    Args:
        original_img_path: Path to original full-size CT image
        prediction_path: Path to prediction segmentation
        output_path: Path to save resampled prediction
    """
    try:
        # Load images
        img_original = sitk.ReadImage(original_img_path)
        prediction = sitk.ReadImage(prediction_path)
        
        print(f"Original image: {img_original.GetSize()}, spacing: {img_original.GetSpacing()}")
        print(f"Prediction: {prediction.GetSize()}, spacing: {prediction.GetSpacing()}")
        
        # Resample prediction to original image space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_original)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Nearest neighbor for segmentation
        resampler.SetTransform(sitk.Transform())  # Identity transform
        resampler.SetDefaultPixelValue(0)
        
        resampled_prediction = resampler.Execute(prediction)
        
        # Save the resampled prediction
        sitk.WriteImage(resampled_prediction, output_path)
        
        print(f"✓ Saved resampled prediction to: {output_path}")
        
        # Verification
        pred_array = sitk.GetArrayFromImage(resampled_prediction)
        non_zero = np.sum(pred_array > 0)
        print(f"  Prediction volume: {non_zero} voxels")
        
        if non_zero == 0:
            print("  WARNING: No prediction found in resampled image!")
            
    except Exception as e:
        print(f"✗ Error resampling {prediction_path}: {e}")
        raise


def batch_resample_predictions(original_dir, prediction_dir, output_dir,
                                original_suffix="_0000.nii.gz"):
    """
    Batch resample multiple predictions to original image space.
    
    Args:
        original_dir: Directory containing original CT images
        prediction_dir: Directory containing predictions
        output_dir: Directory to save resampled predictions
        original_suffix: Suffix for original image files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all prediction files
    prediction_files = sorted([f for f in os.listdir(prediction_dir) if f.endswith(".nii.gz")])
    
    print(f"Found {len(prediction_files)} predictions to process")
    
    for pred_file in prediction_files:
        # Extract case ID
        case_id = pred_file.replace('.nii.gz', '')
        
        # Try different naming conventions for original image
        original_candidates = [
            os.path.join(original_dir, f"{case_id}{original_suffix}"),
            os.path.join(original_dir, f"{case_id}.nii.gz"),
            os.path.join(original_dir, f"Scan_{case_id}_TS.nii.gz"),
        ]
        
        original_path = None
        for candidate in original_candidates:
            if os.path.exists(candidate):
                original_path = candidate
                break
        
        if original_path is None:
            print(f"Skipping {pred_file}: original image not found")
            continue
        
        prediction_path = os.path.join(prediction_dir, pred_file)
        output_path = os.path.join(output_dir, pred_file)
        
        print(f"\nProcessing {case_id}...")
        resample_prediction_to_original_space(original_path, prediction_path, output_path)
    
    print(f"\n✓ Batch resampling complete. Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Resample predictions to original image space")
    
    parser.add_argument("--original_dir", type=str, required=True,
                        help="Directory containing original CT images")
    parser.add_argument("--prediction_dir", type=str, required=True,
                        help="Directory containing predictions")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save resampled predictions")
    parser.add_argument("--original_suffix", type=str, default="_0000.nii.gz",
                        help="Suffix for original image files")
    
    args = parser.parse_args()
    
    batch_resample_predictions(
        args.original_dir,
        args.prediction_dir,
        args.output_dir,
        args.original_suffix
    )


if __name__ == "__main__":
    main()

