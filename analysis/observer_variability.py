"""
Interobserver Variability Analysis for rPCI Region Segmentation

Computes interobserver agreement metrics (Dice, HD95, ASD) from multiple
observer annotations of the same CT scans.

Usage:
    python -m analysis.observer_variability \
        --segmentations-folder /path/to/multi_observer_annotations
"""

import argparse
import SimpleITK as sitk
import os
import gc
import numpy as np
import pandas as pd
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.transforms import AsDiscrete
import torch


# rPCI region names
PCI_REGIONS = [
    "0: Central",
    "1: Right upper",
    "2: Epigastrium",
    "3: Left upper",
    "4: Left flank",
    "5: Left lower",
    "6: Pelvis",
    "7: Right lower",
    "8: Right flank",
    "9: Upper jejunum",
    "10: Lower jejunum",
    "11: Upper ileum",
    "12: Lower ileum",
]


def compute_pairwise_metrics(segA, segL, segM, num_classes=14):
    """
    Compute interobserver metrics from three annotations.
    
    Args:
        segA, segL, segM: Numpy arrays of segmentations from three observers
        num_classes: Number of classes including background
    
    Returns:
        dict: Metrics per class and overall
    """
    # Initialize metrics (CPU only)
    dice = DiceMetric(include_background=False, reduction="mean_batch", 
                      num_classes=num_classes, return_with_label=True)
    hd95 = HausdorffDistanceMetric(include_background=False, percentile=95, 
                                    reduction="mean_batch")
    asd = SurfaceDistanceMetric(include_background=False, reduction="mean_batch")
    to_onehot = AsDiscrete(to_onehot=num_classes)

    # Convert to tensors
    segA_tensor = torch.from_numpy(segA).unsqueeze(0).unsqueeze(0).float()
    segL_tensor = torch.from_numpy(segL).unsqueeze(0).unsqueeze(0).float()
    segM_tensor = torch.from_numpy(segM).unsqueeze(0).unsqueeze(0).float()
    
    # Convert to one-hot
    segA_onehot = to_onehot(segA_tensor)
    segL_onehot = to_onehot(segL_tensor)
    segM_onehot = to_onehot(segM_tensor)
    
    # Compute all pairwise metrics
    dice_results = []
    hd95_results = []
    asd_results = []
    
    # All three pairwise comparisons
    pairs = [
        (segA_tensor, segL_tensor, segA_onehot, segL_onehot, "A-L"),
        (segA_tensor, segM_tensor, segA_onehot, segM_onehot, "A-M"),
        (segL_tensor, segM_tensor, segL_onehot, segM_onehot, "L-M")
    ]
    
    for seg1, seg2, seg1_oh, seg2_oh, pair_name in pairs:
        # Compute metrics for this pair
        dice_pair = dice(seg1, seg2).detach().numpy().squeeze()
        hd95_pair = hd95(seg1_oh, seg2_oh).detach().numpy().squeeze()[1:]
        asd_pair = asd(seg1_oh, seg2_oh).detach().numpy().squeeze()[1:]
        
        dice_results.append(dice_pair)
        hd95_results.append(hd95_pair)
        asd_results.append(asd_pair)
    
    # Stack results: shape [3, num_classes-1]
    dice_all = np.stack(dice_results, axis=0)
    hd95_all = np.stack(hd95_results, axis=0)
    asd_all = np.stack(asd_results, axis=0)
    
    # Compute interobserver metrics (mean across 3 pairwise comparisons)
    dice_per_class = np.mean(dice_all, axis=0)
    hd95_per_class = np.mean(hd95_all, axis=0)
    asd_per_class = np.mean(asd_all, axis=0)
    
    # Standard deviations
    dice_per_class_std = np.std(dice_all, axis=0)
    hd95_per_class_std = np.std(hd95_all, axis=0)
    asd_per_class_std = np.std(asd_all, axis=0)
    
    # Overall means
    dice_mean = np.mean(dice_per_class)
    hd95_mean = np.mean(hd95_per_class)
    asd_mean = np.mean(asd_per_class)
    
    dice_mean_std = np.std(dice_per_class)
    hd95_mean_std = np.std(hd95_per_class)
    asd_mean_std = np.std(asd_per_class)

    # Memory cleanup
    del segA_tensor, segL_tensor, segM_tensor
    del segA_onehot, segL_onehot, segM_onehot
    del dice_results, hd95_results, asd_results
    del dice_all, hd95_all, asd_all
    del dice, hd95, asd, to_onehot
    gc.collect()
    
    return {
        'dice_per_class': dice_per_class,
        'hd95_per_class': hd95_per_class,
        'asd_per_class': asd_per_class,
        'dice_per_class_std': dice_per_class_std,
        'hd95_per_class_std': hd95_per_class_std,
        'asd_per_class_std': asd_per_class_std,
        'dice_mean': dice_mean,
        'hd95_mean': hd95_mean,
        'asd_mean': asd_mean,
        'dice_mean_std': dice_mean_std,
        'hd95_mean_std': hd95_mean_std,
        'asd_mean_std': asd_mean_std
    }


def load_segmentations(segA_path, segL_path, segM_path):
    """Load all three segmentations."""
    try:
        segA = sitk.GetArrayFromImage(sitk.ReadImage(segA_path))
        segL = sitk.GetArrayFromImage(sitk.ReadImage(segL_path))
        segM = sitk.GetArrayFromImage(sitk.ReadImage(segM_path))
        return segA, segL, segM
    except Exception as e:
        raise RuntimeError(f"Failed to load segmentations: {e}")


def process_interobserver_samples(segmentations_folder, output_dir=None):
    """
    Process all cases and compute interobserver variability metrics.
    
    Args:
        segmentations_folder: Folder containing segmentations from multiple observers
        output_dir: Directory to save results (defaults to segmentations_folder)
    
    Expected file naming convention:
        Seg_{case_id}_interob_A_expanded.nii.gz
        Seg_{case_id}_interob_L_expanded.nii.gz
        Seg_{case_id}_interob_M_expanded.nii.gz
    """
    if output_dir is None:
        output_dir = segmentations_folder
    
    # Find all unique case IDs
    case_ids = set()
    for file in os.listdir(segmentations_folder):
        if file.endswith("expanded.nii.gz") and "_interob_" in file:
            parts = file.split("_")
            if len(parts) >= 2:
                case_id = parts[1]
                case_ids.add(case_id)
    
    print(f"Found {len(case_ids)} cases to process")
    print("Using CPU for computations")
    
    all_results = []
    
    for i, case_id in enumerate(sorted(case_ids)):
        print(f"\nProcessing case {case_id} ({i+1}/{len(case_ids)})")
        
        # Build file paths (adjust naming convention as needed)
        segA_path = os.path.join(segmentations_folder, f"Seg_{case_id}_interob_A_expanded.nii.gz")
        segL_path = os.path.join(segmentations_folder, f"Seg_{case_id}_interob_L_expanded.nii.gz")
        segM_path = os.path.join(segmentations_folder, f"Seg_{case_id}_interob_M_expanded.nii.gz")
        
        # Check if all files exist
        if not all(os.path.exists(path) for path in [segA_path, segL_path, segM_path]):
            print(f"  Skipping case {case_id}: missing files")
            continue
        
        try:
            # Load segmentations
            segA, segL, segM = load_segmentations(segA_path, segL_path, segM_path)
            
            # Compute metrics
            results = compute_pairwise_metrics(segA, segL, segM)
            results['case_id'] = case_id
            all_results.append(results)
            
            print(f"  Dice mean: {results['dice_mean']:.4f}")
            print(f"  HD95 mean: {results['hd95_mean']:.4f}")
            print(f"  ASD mean:  {results['asd_mean']:.4f}")
            
        except Exception as e:
            print(f"  Error processing case {case_id}: {str(e)}")
    
    # Print and save summary
    if all_results:
        print(f"\n{'='*60}")
        print("INTEROBSERVER VARIABILITY SUMMARY:")
        print(f"{'='*60}")
        
        dice_means = [r['dice_mean'] for r in all_results]
        hd95_means = [r['hd95_mean'] for r in all_results]
        asd_means = [r['asd_mean'] for r in all_results]
        
        print(f"Overall Dice: {np.mean(dice_means):.4f} ± {np.std(dice_means):.4f}")
        print(f"Overall HD95: {np.mean(hd95_means):.4f} ± {np.std(hd95_means):.4f}")
        print(f"Overall ASD:  {np.mean(asd_means):.4f} ± {np.std(asd_means):.4f}")
        
        # Per-class averages
        dice_per_class_all = np.stack([r['dice_per_class'] for r in all_results], axis=0)
        hd95_per_class_all = np.stack([r['hd95_per_class'] for r in all_results], axis=0)
        asd_per_class_all = np.stack([r['asd_per_class'] for r in all_results], axis=0)
        
        dice_per_class_std_all = np.stack([r['dice_per_class_std'] for r in all_results], axis=0)
        hd95_per_class_std_all = np.stack([r['hd95_per_class_std'] for r in all_results], axis=0)
        asd_per_class_std_all = np.stack([r['asd_per_class_std'] for r in all_results], axis=0)

        dice_per_class_mean = np.mean(dice_per_class_all, axis=0)
        hd95_per_class_mean = np.mean(hd95_per_class_all, axis=0)
        asd_per_class_mean = np.mean(asd_per_class_all, axis=0)
        
        dice_per_class_std_mean = np.mean(dice_per_class_std_all, axis=0)
        hd95_per_class_std_mean = np.mean(hd95_per_class_std_all, axis=0)
        asd_per_class_std_mean = np.mean(asd_per_class_std_all, axis=0)
        
        print(f"\nPer-region Dice:")
        for i, (dice_val, dice_std) in enumerate(zip(dice_per_class_mean, dice_per_class_std_mean)):
            region_name = PCI_REGIONS[i] if i < len(PCI_REGIONS) else f"Class {i}"
            print(f"  {region_name}: {dice_val:.4f} ± {dice_std:.4f}")

        # Create DataFrame
        df_data = {
            'Region': PCI_REGIONS[:len(dice_per_class_mean)],
            'Dice': dice_per_class_mean,
            'Dice_std': dice_per_class_std_mean,
            'HD95': hd95_per_class_mean,
            'HD95_std': hd95_per_class_std_mean,
            'ASD': asd_per_class_mean,
            'ASD_std': asd_per_class_std_mean
        }

        df = pd.DataFrame(df_data)

        # Add overall row
        overall_row = {
            'Region': 'Overall',
            'Dice': np.mean(dice_per_class_mean),
            'Dice_std': np.std(dice_per_class_mean),
            'HD95': np.mean(hd95_per_class_mean),
            'HD95_std': np.std(hd95_per_class_mean),
            'ASD': np.mean(asd_per_class_mean),
            'ASD_std': np.std(asd_per_class_mean)
        }
        df = pd.concat([df, pd.DataFrame([overall_row])], ignore_index=True)

        # Save to CSV
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, "interobserver_metrics.csv")
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to: {csv_path}")
        
        return df


def main():
    parser = argparse.ArgumentParser(description='Compute interobserver variability metrics')
    parser.add_argument("--segmentations-folder", type=str, required=True,
                        help="Folder containing multi-observer segmentations")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save results (defaults to segmentations folder)")
    args = parser.parse_args()
    
    process_interobserver_samples(args.segmentations_folder, args.output_dir)


if __name__ == "__main__":
    main()

