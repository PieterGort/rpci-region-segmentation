"""
Compute Segmentation Metrics for rPCI Region Segmentation

Computes Dice, IoU, HD95 (95th percentile Hausdorff Distance), and ASD (Average Surface Distance)
for comparing ground truth and predicted segmentations.

Usage:
    python -m analysis.compute_metrics \
        --gt-folder /path/to/ground_truth \
        --pred-folder /path/to/predictions \
        --output-dir ./results/metrics
"""

from functools import partial
import numpy as np
import pandas as pd
import argparse
import os
import SimpleITK as sitk
from SimpleITK import GetArrayViewFromImage as ArrayView


# rPCI region names for reporting
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


def compute_dice(gt, pred):
    """
    Compute Dice coefficient between ground truth and prediction.
    
    Args:
        gt: Ground truth binary mask
        pred: Prediction binary mask
    
    Returns:
        float: Dice coefficient [0, 1]
    """
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    
    intersection = np.sum(gt & pred)
    total = np.sum(gt) + np.sum(pred)
    
    if total == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    
    return 2.0 * intersection / total


def compute_iou(gt, pred):
    """
    Compute Intersection over Union (IoU) between ground truth and prediction.
    
    Args:
        gt: Ground truth binary mask
        pred: Prediction binary mask
    
    Returns:
        float: IoU score [0, 1]
    """
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    
    intersection = np.sum(gt & pred)
    union = np.sum(gt | pred)
    
    if union == 0:
        return 1.0 if np.sum(pred) == 0 else 0.0
    
    return intersection / union


def compute_hd95(gt_img, pred_img, label):
    """
    Compute 95th percentile Hausdorff distance using SimpleITK distance maps.
    
    Args:
        gt_img: SimpleITK image of ground truth
        pred_img: SimpleITK image of prediction
        label: Label value to compute HD95 for
        
    Returns:
        float: HD95 distance in mm
    """
    # Create binary masks for the specific label
    gt_binary = gt_img == label
    pred_binary = pred_img == label
    
    # Check if either mask is empty
    gt_array = ArrayView(gt_binary)
    pred_array = ArrayView(pred_binary)
    
    if not np.any(gt_array) or not np.any(pred_array):
        return np.inf
    
    # Get surface contours
    gt_surface = sitk.LabelContour(gt_binary, False)
    pred_surface = sitk.LabelContour(pred_binary, False)
    
    # Check if surfaces exist
    if np.sum(ArrayView(gt_surface)) == 0 or np.sum(ArrayView(pred_surface)) == 0:
        return np.inf
    
    # Create distance maps
    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)
    
    pred_distance_map = sitk.Abs(distance_map(pred_surface))
    gt_distance_map = sitk.Abs(distance_map(gt_surface))
    
    # Find distances to surface points in both directions
    gt_to_pred = ArrayView(pred_distance_map)[ArrayView(gt_surface) == 1]
    pred_to_gt = ArrayView(gt_distance_map)[ArrayView(pred_surface) == 1]
    
    if len(gt_to_pred) == 0 or len(pred_to_gt) == 0:
        return np.inf
    
    # Calculate HD95 as average of 95th percentiles in both directions
    hd95 = (np.percentile(pred_to_gt, 95) + np.percentile(gt_to_pred, 95)) / 2.0
    return hd95


def compute_asd(gt_img, pred_img, label):
    """
    Compute Average Surface Distance using SimpleITK distance maps.
    
    Args:
        gt_img: SimpleITK image of ground truth
        pred_img: SimpleITK image of prediction
        label: Label value to compute ASD for
        
    Returns:
        float: Average surface distance in mm
    """
    # Create binary masks for the specific label
    gt_binary = gt_img == label
    pred_binary = pred_img == label
    
    # Check if either mask is empty
    gt_array = ArrayView(gt_binary)
    pred_array = ArrayView(pred_binary)
    
    if not np.any(gt_array) or not np.any(pred_array):
        return np.inf
    
    # Get surface contours
    gt_surface = sitk.LabelContour(gt_binary, False)
    pred_surface = sitk.LabelContour(pred_binary, False)
    
    # Check if surfaces exist
    if np.sum(ArrayView(gt_surface)) == 0 or np.sum(ArrayView(pred_surface)) == 0:
        return np.inf
    
    # Create distance maps
    distance_map = partial(sitk.SignedMaurerDistanceMap, squaredDistance=False, useImageSpacing=True)
    
    pred_distance_map = sitk.Abs(distance_map(pred_surface))
    gt_distance_map = sitk.Abs(distance_map(gt_surface))
    
    # Find distances to surface points in both directions
    gt_to_pred = ArrayView(pred_distance_map)[ArrayView(gt_surface) == 1]
    pred_to_gt = ArrayView(gt_distance_map)[ArrayView(pred_surface) == 1]
    
    if len(gt_to_pred) == 0 or len(pred_to_gt) == 0:
        return np.inf
    
    # Calculate ASD as average of all distances in both directions
    all_distances = np.concatenate([gt_to_pred, pred_to_gt])
    return np.mean(all_distances)


def compute_confusion_matrix_metrics(gt, pred):
    """
    Compute TP, TN, FP, FN, n_pred, n_ref from ground truth and prediction.
    
    Args:
        gt: Ground truth binary mask
        pred: Prediction binary mask
    
    Returns:
        dict: Dictionary with confusion matrix metrics
    """
    gt = np.asarray(gt, dtype=bool)
    pred = np.asarray(pred, dtype=bool)
    
    tp = np.sum(gt & pred)
    fp = np.sum(~gt & pred)
    fn = np.sum(gt & ~pred)
    tn = np.sum(~gt & ~pred)
    
    n_pred = np.sum(pred)
    n_ref = np.sum(gt)
    
    return {
        'TP': int(tp),
        'TN': int(tn), 
        'FP': int(fp),
        'FN': int(fn),
        'n_pred': int(n_pred),
        'n_ref': int(n_ref)
    }


def process_segmentation_pair(gt_path, pred_path):
    """
    Process a single ground truth and prediction segmentation pair.
    
    Args:
        gt_path: Path to ground truth segmentation
        pred_path: Path to prediction segmentation
    
    Returns:
        dict: Results per label
    """
    # Load images
    gt_img = sitk.ReadImage(gt_path)
    pred_img = sitk.ReadImage(pred_path)
    
    # Ensure images have same properties
    if gt_img.GetSpacing() != pred_img.GetSpacing():
        raise ValueError(f"Spacing mismatch: {gt_img.GetSpacing()} != {pred_img.GetSpacing()}")
    
    if gt_img.GetSize() != pred_img.GetSize():
        raise ValueError(f"Size mismatch: {gt_img.GetSize()} != {pred_img.GetSize()}")
    
    # Convert to numpy arrays
    gt_array = sitk.GetArrayFromImage(gt_img)
    pred_array = sitk.GetArrayFromImage(pred_img)
    
    # Get unique labels (excluding background 0)
    all_labels = np.unique(np.concatenate([gt_array, pred_array]))
    labels = all_labels[all_labels > 0]
    
    results = {}
    
    # Process each label
    for label in labels:
        gt_binary = (gt_array == label)
        pred_binary = (pred_array == label)
        
        # Compute overlap metrics
        dice = compute_dice(gt_binary, pred_binary)
        iou = compute_iou(gt_binary, pred_binary)
        
        # Compute distance metrics
        hd95 = compute_hd95(gt_img, pred_img, label)
        asd = compute_asd(gt_img, pred_img, label)
        
        # Compute confusion matrix metrics
        cm_metrics = compute_confusion_matrix_metrics(gt_binary, pred_binary)
        
        results[label] = {
            'Dice': dice,
            'IoU': iou,
            'HD95': hd95,
            'ASD': asd,
            **cm_metrics
        }
    
    return results


def aggregate_results(all_results):
    """
    Aggregate results across all cases and labels.
    
    Args:
        all_results: Dictionary of results per case
    
    Returns:
        dict: Aggregated statistics per label
    """
    # Collect all metrics per label
    label_metrics = {}
    
    for case_id, case_results in all_results.items():
        for label, metrics in case_results.items():
            if label not in label_metrics:
                label_metrics[label] = {metric: [] for metric in metrics.keys()}
            
            for metric, value in metrics.items():
                # Handle infinite values for distance metrics
                if metric in ['HD95', 'ASD'] and np.isinf(value):
                    continue
                label_metrics[label][metric].append(value)
    
    # Compute statistics per label
    aggregated = {}
    
    for label in sorted(label_metrics.keys()):
        aggregated[label] = {}
        
        for metric in ['Dice', 'IoU', 'HD95', 'ASD', 'TP', 'TN', 'FP', 'FN', 'n_pred', 'n_ref']:
            values = label_metrics[label][metric]
            
            if values:
                aggregated[label][f'{metric}_mean'] = np.mean(values)
                aggregated[label][f'{metric}_std'] = np.std(values)
                aggregated[label][f'{metric}_count'] = len(values)
            else:
                aggregated[label][f'{metric}_mean'] = np.nan
                aggregated[label][f'{metric}_std'] = np.nan
                aggregated[label][f'{metric}_count'] = 0
    
    # Compute overall statistics
    overall_metrics = {}
    
    for metric in ['Dice', 'IoU', 'HD95', 'ASD', 'TP', 'TN', 'FP', 'FN', 'n_pred', 'n_ref']:
        all_values = []
        for label in label_metrics:
            all_values.extend(label_metrics[label][metric])
        
        if all_values:
            overall_metrics[f'{metric}_mean'] = np.mean(all_values)
            overall_metrics[f'{metric}_std'] = np.std(all_values)
            overall_metrics[f'{metric}_count'] = len(all_values)
        else:
            overall_metrics[f'{metric}_mean'] = np.nan
            overall_metrics[f'{metric}_std'] = np.nan
            overall_metrics[f'{metric}_count'] = 0
    
    aggregated['Overall'] = overall_metrics
    
    return aggregated


def compute_segmentation_metrics(gt_folder, pred_folder, output_dir):
    """
    Compute segmentation metrics for all cases in the folders.
    
    Args:
        gt_folder: Path to folder containing ground truth segmentations
        pred_folder: Path to folder containing prediction segmentations  
        output_dir: Path to save results
    """
    # Get all files in ground truth folder
    gt_files = [f for f in os.listdir(gt_folder) if f.endswith('.nii.gz')]
    
    if not gt_files:
        print(f"No .nii.gz files found in {gt_folder}")
        return
    
    print(f"Found {len(gt_files)} cases to process")
    
    # Store results for all cases
    all_results = {}
    
    for gt_file in gt_files:
        case_id = gt_file.replace('.nii.gz', '')
        gt_path = os.path.join(gt_folder, gt_file)
        pred_path = os.path.join(pred_folder, gt_file)
        
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file not found for {case_id}")
            continue
        
        print(f"Processing case {case_id}...")
        
        try:
            case_results = process_segmentation_pair(gt_path, pred_path)
            all_results[case_id] = case_results
        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            continue
    
    # Save per-case results
    os.makedirs(output_dir, exist_ok=True)
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(os.path.join(output_dir, "results_per_case.csv"))
    print(f"Per-case results saved to: {os.path.join(output_dir, 'results_per_case.csv')}")
    
    # Aggregate results
    print("Aggregating results...")
    aggregated_results = aggregate_results(all_results)
    
    # Save aggregated results
    save_results(aggregated_results, output_dir)
    
    return aggregated_results


def save_results(aggregated_results, output_dir):
    """Save aggregated results to CSV and text files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare data for DataFrame
    rows = []
    
    for label, metrics in aggregated_results.items():
        if label == 'Overall':
            region_name = 'Overall'
        else:
            region_name = PCI_REGIONS[label - 1] if label <= len(PCI_REGIONS) else f'Label {label}'
        
        row = {'Label': label, 'Region': region_name}
        row.update(metrics)
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    metric_cols = []
    for metric in ['Dice', 'IoU', 'HD95', 'ASD']:
        metric_cols.extend([f'{metric}_mean', f'{metric}_std', f'{metric}_count'])
    
    column_order = ['Label', 'Region'] + metric_cols
    available_cols = [c for c in column_order if c in df.columns]
    df = df.reindex(columns=available_cols)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, "segmentation_metrics.csv")
    df.to_csv(csv_path, index=False)
    
    # Save summary text
    txt_path = os.path.join(output_dir, "segmentation_metrics_summary.txt")
    with open(txt_path, 'w') as f:
        f.write("=== rPCI Segmentation Metrics Summary ===\n\n")
        f.write(df.to_string(index=False))
    
    print(f"\nResults saved to:")
    print(f"  - {csv_path}")
    print(f"  - {txt_path}")
    
    # Print summary
    print("\n=== Segmentation Metrics Summary ===")
    print(df[['Region', 'Dice_mean', 'Dice_std', 'HD95_mean', 'ASD_mean']].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description='Compute segmentation metrics (Dice, IoU, HD95, ASD)')
    parser.add_argument('--gt-folder', type=str, required=True,
                        help='Path to folder containing ground truth segmentations')
    parser.add_argument('--pred-folder', type=str, required=True,
                        help='Path to folder containing prediction segmentations')
    parser.add_argument('--output-dir', type=str, default="./results/metrics",
                        help='Directory to save results')

    args = parser.parse_args()
    
    compute_segmentation_metrics(
        gt_folder=args.gt_folder,
        pred_folder=args.pred_folder,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()

