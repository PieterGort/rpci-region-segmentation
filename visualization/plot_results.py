"""
Visualization Utilities for rPCI Region Segmentation

Provides functions for visualizing segmentation results,
creating overlay plots, and generating boxplots of metrics.

Usage:
    python -m visualization.plot_results \
        --image /path/to/scan.nii.gz \
        --gt /path/to/ground_truth.nii.gz \
        --pred /path/to/prediction.nii.gz \
        --output ./plots/
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import SimpleITK as sitk


# rPCI region colors and labels
REGION_COLORS = [
    "#000000",  # 0: Background
    "#1f77b4",  # 1: Central
    "#ff7f0e",  # 2: Right upper
    "#2ca02c",  # 3: Epigastrium
    "#d62728",  # 4: Left upper
    "#9467bd",  # 5: Left flank
    "#8c564b",  # 6: Left lower
    "#e377c2",  # 7: Pelvis
    "#7f7f7f",  # 8: Right lower
    "#bcbd22",  # 9: Right flank
    "#17becf",  # 10: Upper jejunum
    "#aec7e8",  # 11: Lower jejunum
    "#ffbb78",  # 12: Upper ileum
    "#98df8a",  # 13: Lower ileum
]

REGION_LABELS = [
    "Background",
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
    "12: Lower ileum"
]


def plot_segmentation_overlay(img_path, gt_path, pred_path, output_dir, 
                               slice_idx=None, plane="XY", case_id="case"):
    """
    Plot ground truth and prediction overlays side by side.
    
    Args:
        img_path: Path to CT image
        gt_path: Path to ground truth segmentation
        pred_path: Path to prediction segmentation
        output_dir: Directory to save plots
        slice_idx: Slice index (default: middle slice)
        plane: Anatomical plane ('XY', 'XZ', 'YZ')
        case_id: Case identifier for filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load images
    img = sitk.ReadImage(img_path)
    gt = sitk.ReadImage(gt_path)
    pred = sitk.ReadImage(pred_path)

    img_arr = sitk.GetArrayFromImage(img)
    gt_arr = sitk.GetArrayFromImage(gt)
    pred_arr = sitk.GetArrayFromImage(pred)

    # Handle shape mismatch (e.g., 4D one-hot)
    if gt_arr.shape != img_arr.shape:
        if len(gt_arr.shape) == 4 and gt_arr.shape[:3] == img_arr.shape:
            gt_arr = np.argmax(gt_arr, axis=-1)
    
    assert img_arr.shape == gt_arr.shape == pred_arr.shape, "Shape mismatch"

    # Select slice
    if plane == "XY":
        if slice_idx is None:
            slice_idx = img_arr.shape[0] // 2
        img_slice = img_arr[slice_idx, :, :]
        gt_slice = gt_arr[slice_idx, :, :]
        pred_slice = pred_arr[slice_idx, :, :]
    elif plane == "XZ":
        if slice_idx is None:
            slice_idx = img_arr.shape[1] // 2
        img_slice = np.flipud(img_arr[:, slice_idx, :])
        gt_slice = np.flipud(gt_arr[:, slice_idx, :])
        pred_slice = np.flipud(pred_arr[:, slice_idx, :])
    elif plane == "YZ":
        if slice_idx is None:
            slice_idx = img_arr.shape[2] // 2
        img_slice = np.flipud(img_arr[:, :, slice_idx])
        gt_slice = np.flipud(gt_arr[:, :, slice_idx])
        pred_slice = np.flipud(pred_arr[:, :, slice_idx])
    else:
        raise ValueError(f"Invalid plane: {plane}")

    # Create colormap
    cmap = mcolors.ListedColormap(REGION_COLORS[:len(REGION_LABELS)])
    norm = mcolors.BoundaryNorm(np.arange(len(REGION_LABELS)+1)-0.5, len(REGION_LABELS))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    # Ground truth
    axes[0].imshow(img_slice, cmap='gray')
    axes[0].imshow(gt_slice, alpha=0.5, cmap=cmap, norm=norm)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')

    # Prediction
    axes[1].imshow(img_slice, cmap='gray')
    axes[1].imshow(pred_slice, alpha=0.5, cmap=cmap, norm=norm)
    axes[1].set_title('Prediction')
    axes[1].axis('off')

    # Legend
    unique_labels = np.unique(np.concatenate([gt_slice.flatten(), pred_slice.flatten()]))
    patches = [mpatches.Patch(color=REGION_COLORS[i], label=REGION_LABELS[i]) 
               for i in unique_labels if i < len(REGION_LABELS)]
    fig.legend(handles=patches, loc='center right', bbox_to_anchor=(1.15, 0.5))

    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"{case_id}_{plane}_overlay.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def plot_dice_boxplot(metrics_csv, output_path, model_name="Model"):
    """
    Create boxplot of Dice scores per region.
    
    Args:
        metrics_csv: Path to CSV with per-case metrics
        output_path: Path to save plot
        model_name: Name for legend
    """
    import pandas as pd
    
    df = pd.read_csv(metrics_csv)
    
    # Extract Dice scores per region
    dice_data = []
    regions = []
    
    for col in df.columns:
        if 'Dice' in col and 'mean' not in col and 'std' not in col:
            dice_data.append(df[col].dropna().values)
            regions.append(col.replace('_Dice', ''))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bp = ax.boxplot(dice_data, labels=regions, patch_artist=True)
    
    # Style boxplot
    for patch in bp['boxes']:
        patch.set_facecolor('#1f77b4')
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Dice Score')
    ax.set_xlabel('rPCI Region')
    ax.set_title(f'{model_name} - Dice Scores per Region')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved boxplot: {output_path}")


def plot_multi_plane(img_path, seg_path, output_path, case_id="case"):
    """
    Create a 3-plane view of segmentation overlay.
    
    Args:
        img_path: Path to CT image
        seg_path: Path to segmentation
        output_path: Path to save plot
        case_id: Case identifier
    """
    img = sitk.ReadImage(img_path)
    seg = sitk.ReadImage(seg_path)
    
    img_arr = sitk.GetArrayFromImage(img)
    seg_arr = sitk.GetArrayFromImage(seg)
    
    # Get middle slices
    z_mid = img_arr.shape[0] // 2
    y_mid = img_arr.shape[1] // 2
    x_mid = img_arr.shape[2] // 2
    
    # Create colormap
    cmap = mcolors.ListedColormap(REGION_COLORS[:len(REGION_LABELS)])
    norm = mcolors.BoundaryNorm(np.arange(len(REGION_LABELS)+1)-0.5, len(REGION_LABELS))
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Axial (XY)
    axes[0].imshow(img_arr[z_mid, :, :], cmap='gray')
    axes[0].imshow(seg_arr[z_mid, :, :], alpha=0.5, cmap=cmap, norm=norm)
    axes[0].set_title(f'Axial (slice {z_mid})')
    axes[0].axis('off')
    
    # Coronal (XZ)
    axes[1].imshow(np.flipud(img_arr[:, y_mid, :]), cmap='gray')
    axes[1].imshow(np.flipud(seg_arr[:, y_mid, :]), alpha=0.5, cmap=cmap, norm=norm)
    axes[1].set_title(f'Coronal (slice {y_mid})')
    axes[1].axis('off')
    
    # Sagittal (YZ)
    axes[2].imshow(np.flipud(img_arr[:, :, x_mid]), cmap='gray')
    axes[2].imshow(np.flipud(seg_arr[:, :, x_mid]), alpha=0.5, cmap=cmap, norm=norm)
    axes[2].set_title(f'Sagittal (slice {x_mid})')
    axes[2].axis('off')
    
    plt.suptitle(f'Case: {case_id}')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize rPCI segmentation results")
    
    subparsers = parser.add_subparsers(dest='command')
    
    # Overlay command
    overlay_parser = subparsers.add_parser('overlay', help='Create overlay plot')
    overlay_parser.add_argument('--image', type=str, required=True)
    overlay_parser.add_argument('--gt', type=str, required=True)
    overlay_parser.add_argument('--pred', type=str, required=True)
    overlay_parser.add_argument('--output', type=str, default='./plots/')
    overlay_parser.add_argument('--plane', type=str, default='XY', choices=['XY', 'XZ', 'YZ'])
    overlay_parser.add_argument('--slice', type=int, default=None)
    
    # Boxplot command
    boxplot_parser = subparsers.add_parser('boxplot', help='Create Dice boxplot')
    boxplot_parser.add_argument('--metrics', type=str, required=True)
    boxplot_parser.add_argument('--output', type=str, required=True)
    boxplot_parser.add_argument('--name', type=str, default='Model')
    
    # Multi-plane command
    multiplane_parser = subparsers.add_parser('multiplane', help='Create 3-plane view')
    multiplane_parser.add_argument('--image', type=str, required=True)
    multiplane_parser.add_argument('--seg', type=str, required=True)
    multiplane_parser.add_argument('--output', type=str, required=True)
    
    args = parser.parse_args()
    
    if args.command == 'overlay':
        case_id = os.path.basename(args.image).replace('.nii.gz', '')
        plot_segmentation_overlay(args.image, args.gt, args.pred, args.output, 
                                   args.slice, args.plane, case_id)
    elif args.command == 'boxplot':
        plot_dice_boxplot(args.metrics, args.output, args.name)
    elif args.command == 'multiplane':
        case_id = os.path.basename(args.image).replace('.nii.gz', '')
        plot_multi_plane(args.image, args.seg, args.output, case_id)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

