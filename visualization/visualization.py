import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import pandas as pd
import scienceplots
from matplotlib.colors import LinearSegmentedColormap

# Use the SciencePlots styles
plt.style.use(['science', 'ieee'])
plt.rcParams['text.usetex'] = False

# Remap Times to available font
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']

# Set global font sizes and line thickness for readability
plt.rcParams.update({
    'font.size': 8,
    'axes.titlesize': 9,
    'axes.labelsize': 8,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.titlesize': 10,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.5,
    'grid.linewidth': 0.5,
})

def plot_gt_and_pred_in_plane(img_path, gt_path, pred_path, case_id, slice_idx, plane="XY", save_eps=False):
    """
    Plot ground truth and prediction overlays for a specific slice in a given plane.
    
    Args:
        img_path (str): Path to the input image
        gt_path (str): Path to the ground truth segmentation
        pred_path (str): Path to the predicted segmentation
        case_id (str): Case identifier for the plot title
        slice_idx (int): Index of the slice to plot
        plane (str): Plane to plot ('XY', 'XZ', or 'YZ')
    """
    # Load images
    img = sitk.ReadImage(img_path)
    gt = sitk.ReadImage(gt_path)
    pred = sitk.ReadImage(pred_path)

    img_arr = sitk.GetArrayFromImage(img)
    gt_arr = sitk.GetArrayFromImage(gt)
    pred_arr = sitk.GetArrayFromImage(pred)

    # Handle 4D GT if necessary
    if gt_arr.shape != img_arr.shape:
        if len(gt_arr.shape) == 4 and gt_arr.shape[:3] == img_arr.shape:
            gt_arr = np.argmax(gt_arr, axis=-1)
        else:
            raise ValueError("GT shape incompatible with image shape")
    
    print(img_arr.shape)
    print(gt_arr.shape)
    print(pred_arr.shape)

    # verify image arrays have same shape
    assert img_arr.shape == gt_arr.shape == pred_arr.shape, "Image arrays must have the same shape"

    if plane == "XY":
        selected_img_arr = img_arr[slice_idx, :, :]
        selected_gt_arr = gt_arr[slice_idx, :, :]
        selected_pred_arr = pred_arr[slice_idx, :, :]
    elif plane == "XZ":
        selected_img_arr = img_arr[:, slice_idx, :]
        selected_gt_arr = gt_arr[:, slice_idx, :]
        selected_pred_arr = pred_arr[:, slice_idx, :]
    elif plane == "YZ":
        selected_img_arr = img_arr[:, :, slice_idx]
        selected_gt_arr = gt_arr[:, :, slice_idx]
        selected_pred_arr = pred_arr[:, :, slice_idx]
    else:
        raise ValueError("Plane must be one of: 'XY', 'XZ', 'YZ'")
    
    # Flip arrays vertically if requested (inverts Z for XZ/YZ planes)
    if plane in ["XZ", "YZ"]:
         selected_img_arr = np.flipud(selected_img_arr)
         selected_gt_arr = np.flipud(selected_gt_arr)
         selected_pred_arr = np.flipud(selected_pred_arr)

    plt.figure(figsize=(4, 3))  # Adjusted width for color bar

    # Define distinct colors for each label, including background
    colors = [
        "#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", 
        "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
    ]
    
    labels = [
        "Background", "0: Central", "1: Right upper", "2: Epigastrium", "3: Left upper", "4: Left flank", 
        "5: Left lower", "6: Pelvis", "7: Right lower", "8: Right flank", 
        "9: Upper jejunum", "10: Lower jejunum", "11: Upper ileum", "12: Lower ileum"
    ]

    # Create a custom colormap
    cmap = mcolors.ListedColormap(colors[:len(labels)])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(labels)+1)-0.5, ncolors=len(labels))

    plt.subplot(1, 2, 1)
    plt.imshow(selected_img_arr, cmap='gray')
    plt.imshow(selected_gt_arr, alpha=0.5, cmap=cmap, norm=norm)
    plt.title(f"Ground Truth")
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(selected_img_arr, cmap='gray')
    pred_im = plt.imshow(selected_pred_arr, alpha=0.5, cmap=cmap, norm=norm)
    plt.title(f"Prediction")
    plt.axis('off')

    # Get unique labels in the prediction
    unique_labels_pred = np.unique(selected_pred_arr)
    unique_labels_gt = np.unique(selected_gt_arr)
    unique_labels = np.unique(np.concatenate((unique_labels_pred, unique_labels_gt)))
    
    # Create custom legend only for present labels
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in unique_labels if i < len(labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.tight_layout()
    plt.savefig(f"plots/val_pred_{case_id}_{plane}plane.png", bbox_inches='tight')
    if save_eps:
        plt.savefig(f"plots/val_pred_{case_id}_{plane}plane.eps", bbox_inches='tight', format='eps')
    # plt.show()

def plot_image_slice(img_path, slice_idx, plane="XY", output_dir="plots"):
    """
    Plots a specific slice of a NIfTI image from a given path.

    Args:
        img_path (str): Path to the NIfTI image file.
        slice_idx (int): Index of the slice to plot.
        plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
        output_dir (str): Directory to save the plot. Defaults to 'plots'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = sitk.ReadImage(img_path)
    img_arr = sitk.GetArrayFromImage(img)

    if plane == "XY":
        selected_img_arr = img_arr[slice_idx, :, :]
    elif plane == "XZ":
        selected_img_arr = img_arr[:, slice_idx, :]
    elif plane == "YZ":
        selected_img_arr = img_arr[:, :, slice_idx]
    else:
        raise ValueError("Plane must be one of: 'XY', 'XZ', 'YZ'")

    plt.figure(figsize=(3, 4))
    plt.imshow(np.flipud(selected_img_arr), cmap='gray', vmin=-100, vmax=300)
    
    # Extract a base name for the title and filename
    base_filename = os.path.basename(img_path).replace('.nii.gz', '')
    
    plt.title(f"Image: {base_filename}\nPlane: {plane}, Slice: {slice_idx}")
    plt.axis('off')
    
    output_filename = os.path.join(output_dir, f"{base_filename}_{plane}_slice{slice_idx}.png")
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    plt.show() # Uncomment to display the plot interactively

def plot_img_and_seg_slice(img_path, seg_path, slice_idx, plane="XY", output_dir="plots"):
    """
    Plots a specific slice of a NIfTI image from a given path.

    Args:
        img_path (str): Path to the NIfTI image file.
        seg_path (str): Path to the NIfTI segmentation file.
        slice_idx (int): Index of the slice to plot.
        plane (str): Plane to slice ('XY', 'XZ', 'YZ'). Defaults to 'XY'.
        output_dir (str): Directory to save the plot. Defaults to 'plots'.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    img = sitk.ReadImage(img_path)
    img = sitk.IntensityWindowing(img, windowMinimum=-250, windowMaximum=300, outputMinimum=0, outputMaximum=255)
    img_arr = sitk.GetArrayFromImage(img)

    seg = sitk.ReadImage(seg_path)
    seg_arr = sitk.GetArrayFromImage(seg)

    if plane == "XY":
        selected_img_arr = img_arr[slice_idx, :, :]
        selected_seg_arr = seg_arr[slice_idx, :, :] 
    elif plane == "XZ":
        selected_img_arr = img_arr[:, slice_idx, :]
        selected_seg_arr = seg_arr[:, slice_idx, :]
    elif plane == "YZ":
        selected_img_arr = img_arr[:, :, slice_idx]
        selected_seg_arr = seg_arr[:, :, slice_idx]
    else:
        raise ValueError("Plane must be one of: 'XY', 'XZ', 'YZ'")

    if plane in ["XZ", "YZ"]:
        selected_img_arr = np.flipud(selected_img_arr)
        selected_seg_arr = np.flipud(selected_seg_arr)
    elif plane == "XY":
        selected_img_arr = np.flipud(selected_img_arr)
        selected_seg_arr = np.flipud(selected_seg_arr)
    

    plt.figure(figsize=(8, 10))

    # Define distinct colors for each label, including background
    colors = [
        "#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", 
        "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
    ]
    
    labels = [
        "Background", "0: Central", "1: Right upper", "2: Epigastrium", "3: Left upper", "4: Left flank", 
        "5: Left lower", "6: Pelvis", "7: Right lower", "8: Right flank", 
        "9: Upper jejunum", "10: Lower jejunum", "11: Upper ileum", "12: Lower ileum"
    ]

    # Create a custom colormap
    cmap = mcolors.ListedColormap(colors[:len(labels)])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(labels)+1)-0.5, ncolors=len(labels))
    
    plt.imshow(np.flipud(selected_img_arr), cmap='gray')
    plt.imshow(np.flipud(selected_seg_arr), alpha=0.5, cmap=cmap, norm=norm)
    # Extract a base name for the title and filename
    base_filename = os.path.basename(img_path).replace('.nii.gz', '')

    # Get unique labels in the segmentation
    unique_labels = np.unique(selected_seg_arr)

    # Create custom legend only for present labels
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in unique_labels if i < len(labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    plt.title(f"Image: {base_filename}\nPlane: {plane}, Slice: {slice_idx}")
    plt.axis('off')
    
    output_filename = os.path.join(output_dir, f"{base_filename}_{plane}_slice{slice_idx}.png")
    plt.tight_layout()
    plt.savefig(output_filename)
    print(f"Saved plot to {output_filename}")
    #plt.show()

def plot_dataset_label_distribution(segmentations_dir, labels):
    """
    Plots the distribution of labels across all segmentations in a dataset.

    Args:
        segmentations_dir (str): Path to directory containing all segmentation files.
        labels (list): List of label names corresponding to the segmentation classes.

    """
    # Initialize dictionary to store total counts for each label
    total_label_counts = {}
    
    # Get all .nii.gz files in the directory
    seg_files = [f for f in os.listdir(segmentations_dir) if f.endswith('.nii.gz')]
    
    print(f"Processing {len(seg_files)} segmentation files...")
    
    # Loop through all segmentation files
    for i, seg_file in enumerate(seg_files):
        seg_path = os.path.join(segmentations_dir, seg_file)
        
        # Read segmentation
        seg = sitk.ReadImage(seg_path)
        segmentation_arr = sitk.GetArrayFromImage(seg)
        
        # Calculate the frequency of each label in this segmentation
        unique, counts = np.unique(segmentation_arr, return_counts=True)
        label_counts = dict(zip(unique, counts))
        
        # Add counts to total
        for label_id, count in label_counts.items():
            if label_id in total_label_counts:
                total_label_counts[label_id] += count
            else:
                total_label_counts[label_id] = count
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(seg_files):
            print(f"Processed {i + 1}/{len(seg_files)} files")
    
    # Exclude the background (assuming it's label 0) and only keep foreground labels
    label_indices = [i for i in total_label_counts.keys() if i != 0]
    label_indices.sort()  # Sort to ensure consistent ordering
    frequencies = [total_label_counts[i] for i in label_indices]
    label_names = [labels[i] for i in label_indices if i < len(labels)]
    
    # Calculate total foreground voxels and percentages
    total_foreground_voxels = sum(frequencies)
    percentages = [(freq / total_foreground_voxels) * 100 for freq in frequencies]
    
    # Print summary statistics
    print(f"\nDataset Label Distribution Summary:")
    print(f"Total files processed: {len(seg_files)}")
    print(f"Labels found: {sorted(total_label_counts.keys())}")
    print(f"Background (label 0): {total_label_counts.get(0, 0):,} voxels")
    print(f"Total foreground voxels: {total_foreground_voxels:,}")
    print(f"\nForeground class distribution:")
    for label_id, freq, pct in zip(label_indices, frequencies, percentages):
        if label_id < len(labels):
            print(f"  {labels[label_id]} (label {label_id}): {pct:.2f}% ({freq:,} voxels)")

    # Define distinct colors for each label, including background
    colors = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", 
        "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
    ]

    # Plot the distribution
    plt.figure(figsize=(8, 4))
    bars = plt.bar(label_names, percentages, color=colors[:len(label_names)])
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('Labels')
    plt.ylabel('Percentage (%)')
    plt.title('Dataset Label Distribution (All Segmentations)')
    
    # Add percentage labels on top of bars
    for bar, pct in zip(bars, percentages):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(percentages)*0.01, 
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig("results/segmentation_analysis/plots/Dataset101_PM_label_distribution.png", dpi=150)
    plt.savefig("results/segmentation_analysis/plots/Dataset101_PM_label_distribution.eps", format='eps', dpi=150)
    return total_label_counts

def print_image_info(image, image_name):
    print(f"Information for {image_name}:")
    print(f"Size: {image.GetSize()}")
    print(f"Spacing: {image.GetSpacing()}")
    print(f"Origin: {image.GetOrigin()}")
    print(f"Direction: {image.GetDirection()}")
    print(f"Pixel ID: {image.GetPixelIDTypeAsString()}")
    print("-" * 50)

def plot_labelwise_prediction_error(img_path, gt_path, pred_path, case_id, slice_idx, plane="XY", output_dir="plots", save_eps=False):
    """
    Plot CT image with overlay showing label-wise disagreement between GT and prediction.
    """
    # Create output dir if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load images
    img = sitk.ReadImage(img_path)
    gt = sitk.ReadImage(gt_path)
    pred = sitk.ReadImage(pred_path)

    img_arr = sitk.GetArrayFromImage(img)
    gt_arr = sitk.GetArrayFromImage(gt)
    pred_arr = sitk.GetArrayFromImage(pred)

    # Handle 4D GT if necessary
    if gt_arr.shape != img_arr.shape:
        if len(gt_arr.shape) == 4 and gt_arr.shape[:3] == img_arr.shape:
            gt_arr = np.argmax(gt_arr, axis=-1)
        else:
            raise ValueError("GT shape incompatible with image shape")

    assert img_arr.shape == gt_arr.shape == pred_arr.shape, "Image arrays must have the same shape"

    # Compute label-wise difference map: keep GT label if disagreement, else 0
    diff_arr = np.where(gt_arr != pred_arr, gt_arr, 0).astype(np.uint8)

    # Select slice
    if plane == "XY":
        img_slice = img_arr[slice_idx, :, :]
        diff_slice = diff_arr[slice_idx, :, :]
    elif plane == "XZ":
        img_slice = img_arr[:, slice_idx, :]
        diff_slice = diff_arr[:, slice_idx, :]
    elif plane == "YZ":
        img_slice = img_arr[:, :, slice_idx]
        diff_slice = diff_arr[:, :, slice_idx]
    else:
        raise ValueError("Plane must be one of: 'XY', 'XZ', 'YZ'")

    # Flip for correct anatomical orientation
    if plane in ["XZ", "YZ"]:
        img_slice = np.flipud(img_slice)
        diff_slice = np.flipud(diff_slice)

    plt.figure(figsize=(4, 4))

    # Define color map for 13-class labels
    colors = [
        "#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", 
        "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
    ]

    labels = [
        "Background", "0: Central", "1: Right upper", "2: Epigastrium", "3: Left upper", "4: Left flank", 
        "5: Left lower", "6: Pelvis", "7: Right lower", "8: Right flank", 
        "9: Upper jejunum", "10: Lower jejunum", "11: Upper ileum", "12: Lower ileum"
    ]

    cmap = mcolors.ListedColormap(colors[:len(labels)])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(labels)+1)-0.5, ncolors=len(labels))

    # Plot CT image
    plt.imshow(img_slice, cmap='gray')

    # Overlay label-wise disagreement map
    plt.imshow(np.ma.masked_where(diff_slice == 0, diff_slice), cmap=cmap, norm=norm, alpha=0.5)

    plt.title(f"Label-wise Error: Case {case_id}")
    plt.axis('off')

    # Dynamic legend for present labels
    unique_labels = np.unique(diff_slice)
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in unique_labels if i < len(labels)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    # Save figure
    plt.tight_layout()
    out_path = os.path.join(output_dir, f"error_labelwise_{case_id}_{plane}_slice{slice_idx}.png")
    plt.savefig(out_path, dpi=300)
    if save_eps:
        plt.savefig(out_path.replace('.png', '.eps'), format='eps')
    # plt.show()

def compute_labelwise_difference_map(gt_path, pred_path, output_path=None):
    """
    Computes a label-wise difference map: 
    - If prediction differs from GT, the voxel is assigned GT label value.
    - If prediction matches GT, voxel is assigned 0.
    """
    gt = sitk.ReadImage(gt_path)
    pred = sitk.ReadImage(pred_path)

    gt_arr = sitk.GetArrayFromImage(gt)
    pred_arr = sitk.GetArrayFromImage(pred)

    # Handle 4D GT if necessary
    if gt_arr.shape != pred_arr.shape:
        if len(gt_arr.shape) == 4 and gt_arr.shape[:3] == pred_arr.shape:
            gt_arr = np.argmax(gt_arr, axis=-1)
        else:
            raise ValueError("GT shape incompatible with prediction shape")

    assert gt_arr.shape == pred_arr.shape, "Shapes must match after correction"

    # Difference map keeps the ground truth label at disagreement voxels
    diff_arr = np.where(gt_arr != pred_arr, gt_arr, 0).astype(np.uint8)

    # Convert back to SimpleITK
    diff_img = sitk.GetImageFromArray(diff_arr)
    diff_img.CopyInformation(gt)

    if output_path:
        sitk.WriteImage(diff_img, output_path)
        print(f"Saved labelwise difference map to: {output_path}")

    return diff_img

def plot_gt_pred_error(img_path, gt_path, pred_path, case_id, slice_idx, plane="XY", output_dir="plots", save_eps=False):
    """
    Plot image with GT, prediction and label-wise error overlay in one figure.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load images
    img = sitk.ReadImage(img_path)
    gt = sitk.ReadImage(gt_path)
    pred = sitk.ReadImage(pred_path)

    img_arr = sitk.GetArrayFromImage(img)
    gt_arr = sitk.GetArrayFromImage(gt)
    pred_arr = sitk.GetArrayFromImage(pred)

    # Handle 4D GT if necessary
    if gt_arr.shape != img_arr.shape:
        if len(gt_arr.shape) == 4 and gt_arr.shape[:3] == img_arr.shape:
            gt_arr = np.argmax(gt_arr, axis=-1)
        else:
            raise ValueError("GT shape incompatible with image shape")

    assert img_arr.shape == gt_arr.shape == pred_arr.shape, "Image arrays must have the same shape"

    # Compute label-wise difference map (keep GT label where prediction fails)
    diff_arr = np.where(gt_arr != pred_arr, gt_arr, 0).astype(np.uint8)

    # Extract slice
    if plane == "XY":
        img_slice = img_arr[slice_idx, :, :]
        gt_slice = gt_arr[slice_idx, :, :]
        pred_slice = pred_arr[slice_idx, :, :]
        diff_slice = diff_arr[slice_idx, :, :]
    elif plane == "XZ":
        img_slice = img_arr[:, slice_idx, :]
        gt_slice = gt_arr[:, slice_idx, :]
        pred_slice = pred_arr[:, slice_idx, :]
        diff_slice = diff_arr[:, slice_idx, :]
    elif plane == "YZ":
        img_slice = img_arr[:, :, slice_idx]
        gt_slice = gt_arr[:, :, slice_idx]
        pred_slice = pred_arr[:, :, slice_idx]
        diff_slice = diff_arr[:, :, slice_idx]
    else:
        raise ValueError("Plane must be one of: 'XY', 'XZ', 'YZ'")

    if plane in ["XZ", "YZ"]:
        img_slice = np.flipud(img_slice)
        gt_slice = np.flipud(gt_slice)
        pred_slice = np.flipud(pred_slice)
        diff_slice = np.flipud(diff_slice)

    # Color definitions
    colors = [
        "#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#ffd700", "#bcbd22", 
        "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
    ]

    labels = [
        "Background", "0: Central", "1: Right upper", "2: Epigastrium", "3: Left upper", "4: Left flank", 
        "5: Left lower", "6: Pelvis", "7: Right lower", "8: Right flank", 
        "9: Upper jejunum", "10: Lower jejunum", "11: Upper ileum", "12: Lower ileum"
    ]

    cmap = mcolors.ListedColormap(colors[:len(labels)])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(labels)+1)-0.5, ncolors=len(labels))

    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(6, 4))

    # 1 - GT
    axs[0].imshow(img_slice, cmap='gray', vmin=-100, vmax=300)
    axs[0].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap=cmap, norm=norm, alpha=0.7)
    axs[0].set_title("Ground Truth")
    axs[0].axis('off')

    # 2 - Prediction
    axs[1].imshow(img_slice, cmap='gray', vmin=-100, vmax=300)
    axs[1].imshow(np.ma.masked_where(pred_slice == 0, pred_slice), cmap=cmap, norm=norm, alpha=0.7)
    axs[1].set_title("Prediction")
    axs[1].axis('off')

    # 3 - Label-wise error
    axs[2].imshow(img_slice, cmap='gray', vmin=-100, vmax=300)
    axs[2].imshow(np.ma.masked_where(diff_slice == 0, diff_slice), cmap=cmap, norm=norm, alpha=0.7)
    axs[2].set_title("Segmentation Error")
    axs[2].axis('off')

    # Generate legend dynamically based on all slices
    all_labels = np.unique(np.concatenate((gt_slice, pred_slice, diff_slice)))
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in all_labels if i < len(labels)]
    fig.legend(handles=patches, bbox_to_anchor=(0.97, 0.5), loc='center left')

    plt.tight_layout()
    filename = os.path.join(output_dir, f"full_segmentation_analysis_{case_id}_{plane}_slice{slice_idx}.png")
    plt.savefig(filename, dpi=150)
    if save_eps:
        plt.savefig(filename.replace('.png', '.eps'), format='eps', dpi=150)

def plot_gt_pred_error_3planes(img_path, gt_path, pred_path, case_id, output_dir="plots", save_eps=False, figsize=(18, 12)):
    """
    Plot image with GT, prediction and label-wise error overlay in one figure.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load images
    img = sitk.ReadImage(img_path)
    gt = sitk.ReadImage(gt_path)
    pred = sitk.ReadImage(pred_path)

    img_arr = sitk.GetArrayFromImage(img)
    gt_arr = sitk.GetArrayFromImage(gt)
    pred_arr = sitk.GetArrayFromImage(pred)

    # Handle 4D GT if necessary
    if gt_arr.shape != img_arr.shape:
        if len(gt_arr.shape) == 4 and gt_arr.shape[:3] == img_arr.shape:
            gt_arr = np.argmax(gt_arr, axis=-1)
        else:
            raise ValueError("GT shape incompatible with image shape")

    assert img_arr.shape == gt_arr.shape == pred_arr.shape, "Image arrays must have the same shape"

    # Compute label-wise difference map (keep GT label where prediction fails)
    diff_arr = np.where(gt_arr != pred_arr, gt_arr, 0).astype(np.uint8)

    # Color definitions
    colors = [
        "#000000", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", 
        "#8c564b", "#e377c2", "#ffd700", "#bcbd22", 
        "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
    ]

    labels = [
        "Background", "0: Central", "1: Right upper", "2: Epigastrium", "3: Left upper", "4: Left flank", 
        "R5: Left lower", "R6: Pelvis", "R7: Right lower", "R8: Right flank", 
        "R9: Upper jejunum", "R10: Lower jejunum", "R11: Upper ileum", "R12: Lower ileum"
    ]

    cmap = mcolors.ListedColormap(colors[:len(labels)])
    norm = mcolors.BoundaryNorm(boundaries=np.arange(len(labels)+1)-0.5, ncolors=len(labels))

    # Plotting with custom subplot sizes
    fig = plt.figure(figsize=figsize)
    
    # Create a gridspec with custom width and height ratios
    # width_ratios: label column + 3 image columns (GT, Prediction, Error)
    # height_ratios: relative heights of the 3 rows (XZ, XY, YZ planes)
    gs = fig.add_gridspec(3, 4, 
                         width_ratios=[0.15, 1, 1, 1],  # Small label column + 3 equal image columns
                         height_ratios=[1.65, 1, 2.07],  # Make XY plane row slightly taller
                         hspace=0.01, wspace=0.01,  # Remove horizontal spacing completely
                         left=0, right=0.85, top=0.99, bottom=0.01)  # Tight margins
    
    # Create subplots using the gridspec
    # Column 0: labels, Columns 1-3: images
    label_axs = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    axs = np.array([[fig.add_subplot(gs[i, j]) for j in range(1, 4)] for i in range(3)])


    for i, plane in enumerate(["XZ", "XY", "YZ"]):

        if plane == "XY":
            img_slice = img_arr[281, :, :]
            gt_slice = gt_arr[281, :, :]
            pred_slice = pred_arr[281, :, :]
            diff_slice = diff_arr[281, :, :]
        elif plane == "XZ":
            img_slice = img_arr[:, 116, :]
            gt_slice = gt_arr[:, 116, :]
            pred_slice = pred_arr[:, 116, :]
            diff_slice = diff_arr[:, 116, :]
        elif plane == "YZ":
            img_slice = img_arr[:, :, 160]
            gt_slice = gt_arr[:, :, 160]
            pred_slice = pred_arr[:, :, 160]
            diff_slice = diff_arr[:, :, 160]

        if plane in ["XZ", "YZ"]:
            img_slice = np.flipud(img_slice)
            gt_slice = np.flipud(gt_slice)
            pred_slice = np.flipud(pred_slice)
            diff_slice = np.flipud(diff_slice)

        # 1 - GT
        axs[i, 0].imshow(img_slice, cmap='gray', vmin=-175, vmax=250)
        axs[i, 0].imshow(np.ma.masked_where(gt_slice == 0, gt_slice), cmap=cmap, norm=norm, alpha=0.7)
        if i == 0:
            axs[i, 0].set_title("Ground Truth", fontsize=14)
        axs[i, 0].axis('off')

        # 2 - Prediction
        axs[i, 1].imshow(img_slice, cmap='gray', vmin=-175, vmax=250)
        axs[i, 1].imshow(np.ma.masked_where(pred_slice == 0, pred_slice), cmap=cmap, norm=norm, alpha=0.7)
        if i == 0:
            axs[i, 1].set_title("Prediction", fontsize=14)
        axs[i, 1].axis('off')

        # 3 - Label-wise error
        axs[i, 2].imshow(img_slice, cmap='gray', vmin=-175, vmax=250)
        axs[i, 2].imshow(np.ma.masked_where(diff_slice == 0, diff_slice), cmap=cmap, norm=norm, alpha=0.7)
        if i == 0:
            axs[i, 2].set_title("Segmentation Error", fontsize=14)
        axs[i, 2].axis('off')

    # Add plane labels in the left column
    plane_labels = ["Coronal", "Axial", "Sagittal"]
    for i, plane_label in enumerate(plane_labels):
        label_axs[i].text(0.5, 0.5, plane_label, 
                         horizontalalignment='center', 
                         verticalalignment='center',
                         fontsize=14,
                         rotation=90, transform=label_axs[i].transAxes)
        label_axs[i].axis('off')

    all_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in all_labels if i < len(labels)]
    fig.legend(handles=patches[1:], bbox_to_anchor=(0.85, 0.5), loc='center left', fontsize=14)

    # plt.tight_layout()  # Commented out since we're using manual positioning
    filename = os.path.join(output_dir, f"full_segmentation_analysis_{case_id}_3planes.png")
    plt.savefig(filename, dpi=150)
    if save_eps:
        plt.savefig(filename.replace('.png', '.eps'), format='eps', dpi=150)

def plot_dice_bar_chart(swin_results_path, nnunet_results_path):
    """
    Plot a bar chart of the Dice scores for the SwinUNETR and nnU-Net models.
    """
    swin_results = pd.read_csv(swin_results_path)
    nnunet_results = pd.read_csv(nnunet_results_path)
    
    region_names = [
        "Central", "Right upper", "Epigastrium", "Left upper", "Left flank",
        "Left lower", "Pelvis", "Right lower", "Right flank",
        "Upper jejunum", "Lower jejunum", "Upper ileum", "Lower ileum"
    ]
    region_labels = [f"{i}: {name}" for i, name in enumerate(region_names)]
    
    # Results loading
    swin_means = swin_results["Dice_Mean"].tolist()
    swin_stds = swin_results["Dice_Std"].tolist()
    nnunet_means = nnunet_results["Dice Mean"][:-1].tolist()
    nnunet_stds = nnunet_results["Dice Std"][:-1].tolist()

    # nnU-Net results
    # nnunet_means = [0.80, 0.94, 0.86, 0.90, 0.77, 0.75, 0.88,
    #                 0.73, 0.68, 0.78, 0.63, 0.64, 0.72]
    # nnunet_stds = [0.027, 0.017, 0.026, 0.022, 0.042, 0.053, 0.013,
    #             0.051, 0.046, 0.0090, 0.068, 0.041, 0.012]

    # Plot settings
    x = np.arange(len(region_labels))
    width = 0.35

    # Create plot
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(x - width/2, swin_means, width, yerr=swin_stds, capsize=3,
        label='SwinUNETR', color='#edf8b1', edgecolor='black')
    ax.bar(x + width/2, nnunet_means, width, yerr=nnunet_stds, capsize=3,
        label='nnU-Net', color='#2c7fb8', edgecolor='black')

    # Axis labels and title
    ax.set_xlabel("Region")
    ax.set_ylabel("Dice Similarity Coefficient")
    # ax.set_title("Per-region Dice Scores with Std. Dev.")
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc='upper right')

    plt.tight_layout()

    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/dice_scores_comparison.png", bbox_inches='tight', dpi=300)
    plt.savefig("plots/dice_scores_comparison.eps", bbox_inches='tight', format='eps')

    plt.show()

def plot_dice_boxplots(model1_results_path, model2_results_path, model1_name, model2_name):
    """
    Plot vertical boxplots of Dice scores for SwinUNETR and nnU-Net models.
    Reads JSON files containing raw data points for each region/case.
    """
    import json
    
    with open(model1_results_path, 'r') as f:
        model1_results = json.load(f)   
    with open(model2_results_path, 'r') as f:
        model2_results = json.load(f)
    
    region_names = [
        "Central", "Right upper", "Epigastrium", "Left upper", "Left flank",
        "Left lower", "Pelvis", "Right lower", "Right flank",
        "Upper jejunum", "Lower jejunum", "Upper ileum", "Lower ileum"
    ]
    region_labels = [f"{i}: {name}" for i, name in enumerate(region_names)]
    
    # Extract data from metric_per_case for each region
    model1_data = []
    model2_data = []
    
    # Extract SwinUNETR data from metric_per_case
    for i in range(len(region_names)):
        region_key = str(i + 1)  # Regions are numbered 1-13 in the JSON
        model1_region_data = []
        
        for case in model1_results["metric_per_case"]:
            if region_key in case["metrics"]:
                dice_score = case["metrics"][region_key]["Dice"]
                # Filter out NaN values
                if not np.isnan(dice_score):
                    model1_region_data.append(dice_score)
        
        model1_data.append(np.array(model1_region_data))
    
    # Extract nnU-Net data from metric_per_case  
    for i in range(len(region_names)):
        region_key = str(i + 1)  # Regions are numbered 1-13 in the JSON
        model2_region_data = []
        
        for case in model2_results["metric_per_case"]:
            if region_key in case["metrics"]:
                dice_score = case["metrics"][region_key]["Dice"]
                # Filter out NaN values
                if not np.isnan(dice_score):
                    model2_region_data.append(dice_score)
        
        model2_data.append(np.array(model2_region_data))
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    # Set up positions for boxplots
    x = np.arange(len(region_labels))
    width = 0.35
    
    # More vibrant colors similar to the reference image
    model1_color = '#FF7F50'  # Coral for SwinUNETR
    model2_color = '#6495ED'  # Cornflowerblue for nnU-Net
    
    # # Plot individual data points first (in background)
    # np.random.seed(42)  # For reproducible jitter
    # for i, (swin_vals, nnunet_vals) in enumerate(zip(swin_data, nnunet_data)):
    #     # Filter out any remaining NaN values and ensure we have valid data
    #     swin_vals_clean = swin_vals[~np.isnan(swin_vals)] if len(swin_vals) > 0 else swin_vals
    #     nnunet_vals_clean = nnunet_vals[~np.isnan(nnunet_vals)] if len(nnunet_vals) > 0 else nnunet_vals
        
    #     if len(swin_vals_clean) > 0:
    #         # Add small random jitter to x positions for better visibility
    #         swin_x = np.random.normal(x[i] - width/2, 0.04, size=len(swin_vals_clean))
    #         ax.scatter(swin_x, swin_vals_clean, alpha=0.7, s=12, color='#FF4500', edgecolors='white', linewidth=0.3)
        
    #     if len(nnunet_vals_clean) > 0:
    #         nnunet_x = np.random.normal(x[i] + width/2, 0.04, size=len(nnunet_vals_clean))
    #         ax.scatter(nnunet_x, nnunet_vals_clean, alpha=0.7, s=12, color='#4169E1', edgecolors='white', linewidth=0.3)
    
    # Create boxplots on top (no outliers)
    bp1 = ax.boxplot([data for data in model1_data], 
                     positions=x - width/2, 
                     widths=width*0.6,
                     patch_artist=True,
                     showfliers=True,
                     flierprops=dict(marker='o', markerfacecolor=model1_color, markeredgecolor='black', markeredgewidth=1, markersize=5, alpha=1))
    
    bp2 = ax.boxplot([data for data in model2_data], 
                     positions=x + width/2, 
                     widths=width*0.6,
                     patch_artist=True,
                     showfliers=True,
                     flierprops=dict(marker='o', markerfacecolor=model2_color, markeredgecolor='black', markeredgewidth=1, markersize=5, alpha=1))  # No outliers
    
    # Set vibrant colors for the boxes with prominent black edges
    for patch in bp1['boxes']:
        patch.set_facecolor(model1_color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)  # Thicker black edges
    
    for patch in bp2['boxes']:
        patch.set_facecolor(model2_color)
        patch.set_edgecolor('black')
        patch.set_linewidth(1.0)  # Thicker black edges
    
    # Set colors for other boxplot elements
    for element in ['whiskers', 'caps', 'medians']:
        for item in bp1[element]:
            item.set_color('black')
            item.set_linewidth(1.0)  # Thicker to match box edges
        for item in bp2[element]:
            item.set_color('black')
            item.set_linewidth(1.0)  # Thicker to match box edges
    
    # Axis labels and formatting
    ax.set_xlabel("Region")
    ax.set_ylabel("Dice Similarity Coefficient")
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)

    # Create custom legend with vibrant colors and black borders
    import matplotlib.patches as mpatches
    model1_patch = mpatches.Patch(color=model1_color, edgecolor='black', linewidth=1, label=model1_name)
    model2_patch = mpatches.Patch(color=model2_color, edgecolor='black', linewidth=1, label=model2_name)
    ax.legend(handles=[model1_patch, model2_patch], loc='lower left', frameon=True, edgecolor='black', framealpha=1)
    
    plt.tight_layout()
    
    # Save plots
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig("plots/dice_scores_boxplots.png", bbox_inches='tight', dpi=300)
    plt.savefig("plots/dice_scores_boxplots.eps", bbox_inches='tight', format='eps')
    
    plt.show()

def visualize_disagreement(img_path, seg_path1, seg_path2, seg_path3, slice_idx, plane="XY", output_dir="plots"):
    """
    Visualize inter-observer disagreement as a heatmap overlay on the original CT image.
    
    Args:
        img_path (str): Path to the original CT image
        seg_path1 (str): Path to observer 1 segmentation
        seg_path2 (str): Path to observer 2 segmentation  
        seg_path3 (str): Path to observer 3 segmentation
        slice_idx (int): Index of the slice to plot
        plane (str): Plane to plot ('XY', 'XZ', or 'YZ')
        output_dir (str): Directory to save the plot
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load images
    img = sitk.ReadImage(img_path)
    img = sitk.IntensityWindowing(img, windowMinimum=-250, windowMaximum=300, outputMinimum=0, outputMaximum=255)
    img_arr = sitk.GetArrayFromImage(img)
    
    # Load segmentations
    seg1 = sitk.ReadImage(seg_path1)
    seg2 = sitk.ReadImage(seg_path2)
    seg3 = sitk.ReadImage(seg_path3)
    
    seg1_arr = sitk.GetArrayFromImage(seg1)
    seg2_arr = sitk.GetArrayFromImage(seg2)
    seg3_arr = sitk.GetArrayFromImage(seg3)
    
    # Calculate disagreement map
    # Count how many observers agree at each voxel
    agreement_count = np.zeros_like(seg1_arr, dtype=np.float32)
    
    # For each voxel, count agreements between pairs of observers
    agreement_12 = (seg1_arr == seg2_arr).astype(float)
    agreement_13 = (seg1_arr == seg3_arr).astype(float) 
    agreement_23 = (seg2_arr == seg3_arr).astype(float)
    
    # Create a more nuanced disagreement score based on label diversity
    disagreement_map = np.zeros_like(seg1_arr, dtype=np.float32)
    
    # Only consider voxels where at least one observer has a non-background label
    foreground_mask = (seg1_arr > 0) | (seg2_arr > 0) | (seg3_arr > 0)
    
    # For each foreground voxel, calculate disagreement based on unique labels
    for z in range(seg1_arr.shape[0]):
        for y in range(seg1_arr.shape[1]):
            for x in range(seg1_arr.shape[2]):
                if foreground_mask[z, y, x]:
                    labels = [seg1_arr[z, y, x], seg2_arr[z, y, x], seg3_arr[z, y, x]]
                    unique_labels = len(set(labels))
                    
                    if unique_labels == 1:
                        # All three observers agree
                        disagreement_map[z, y, x] = 0.0
                    elif unique_labels == 2:
                        # Two observers agree, one disagrees
                        disagreement_map[z, y, x] = 0.5
                    else:
                        # All three observers disagree
                        disagreement_map[z, y, x] = 1.0
    
    # Extract slice
    if plane == "XY":
        img_slice = img_arr[slice_idx, :, :]
        disagreement_slice = disagreement_map[slice_idx, :, :]
    elif plane == "XZ":
        img_slice = img_arr[:, slice_idx, :]
        disagreement_slice = disagreement_map[:, slice_idx, :]
    elif plane == "YZ":
        img_slice = img_arr[:, :, slice_idx]
        disagreement_slice = disagreement_map[:, :, slice_idx]
    else:
        raise ValueError("Plane must be one of: 'XY', 'XZ', 'YZ'")
    
    # Flip for correct anatomical orientation
    if plane in ["XZ", "YZ"]:
        img_slice = np.flipud(img_slice)
        disagreement_slice = np.flipud(disagreement_slice)
    
    # Create the plot
    plt.figure(figsize=(8, 6))
    
    # Display CT image
    plt.imshow(img_slice, cmap='gray', alpha=1.0)
    
    # Create custom colormap for disagreement heatmap (linear color progression)
    disagreement_cmap = plt.cm.get_cmap('hot')  # Black -> red -> yellow -> white progression
    
    # Overlay disagreement heatmap (mask zero values)
    masked_disagreement = np.ma.masked_where(disagreement_slice == 0, disagreement_slice)
    im = plt.imshow(masked_disagreement, cmap=disagreement_cmap, alpha=0.7, vmin=0, vmax=1, 
                   interpolation='bilinear')  # Smooth interpolation for heatmap effect
    
    # Add colorbar with smooth gradient
    cbar = plt.colorbar(im, shrink=0.8, aspect=20)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(['Complete\nAgreement', 'Low\nDisagreement', 'Moderate\nDisagreement', 
                        'High\nDisagreement', 'Complete\nDisagreement'])
    
    # plt.title(f'Inter-observer Disagreement\nPlane: {plane}, Slice: {slice_idx}')
    plt.axis('off')
    
    # Save the plot
    base_filename = os.path.basename(img_path).replace('.nii.gz', '')
    output_filename = os.path.join(output_dir, f"{base_filename}_disagreement_{plane}_slice{slice_idx}.png")
    plt.tight_layout()
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Saved disagreement plot to {output_filename}")
    
    # Also save as EPS
    plt.savefig(output_filename.replace('.png', '.eps'), format='eps', bbox_inches='tight')
    
    # plt.show()

def plot_dice_bar_chart_from_csv(model1_results_path, model2_results_path, IOB_results_path, model1_name, model2_name, metric_name):
    """
    Plot vertical boxplots of Dice scores for SwinUNETR and nnU-Net models.
    Reads JSON files containing raw data points for each region/case.
    """
    
    model1_results = pd.read_csv(model1_results_path)
    model2_results = pd.read_csv(model2_results_path)
    IOB_results = pd.read_csv(IOB_results_path)
    
    region_names = [
        "Central", "Right upper", "Epigastrium", "Left upper", "Left flank",
        "Left lower", "Pelvis", "Right lower", "Right flank",
        "Upper jejunum", "Lower jejunum", "Upper ileum", "Lower ileum"
    ]
    region_labels = [f"{i}: {name}" for i, name in enumerate(region_names)]


    model1_means = model1_results[f"{metric_name}_mean"].values.tolist()
    model2_means = model2_results[f"{metric_name}_mean"].values.tolist()
    model1_stds = model1_results[f"{metric_name}_std"].values.tolist()
    model2_stds = model2_results[f"{metric_name}_std"].values.tolist()
    IOB_means = IOB_results[f"{metric_name}_mean"].values.tolist()
    IOB_stds = IOB_results[f"{metric_name}_std"].values.tolist()

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 3.5))
    
    # Set up positions for boxplots
    x = np.arange(len(region_labels))
    width = 0.25
    
    # make two vibrant random colors that are different for each metric, make them colorblind friendly
    if metric_name == "Dice":
        model1_color = '#377eb8'
        model2_color = '#ff7f00'
    elif metric_name == "HD95":
        model1_color = '#4daf4a'
        model2_color = '#f781bf'
    elif metric_name == "ASD":
        model1_color = '#a65628'
        model2_color = '#984ea3'
    else:
        raise ValueError(f"Invalid metric name: {metric_name}")
    IOB_color = '#808080'
    # Create boxplots on top (no outliers)
    ax.bar(x - width, model1_means[:-1], width, yerr=model1_stds[:-1], capsize=1.5,
        label=model1_name, color=model1_color, edgecolor='black')
    ax.bar(x, model2_means[:-1], width, yerr=model2_stds[:-1], capsize=1.5,
        label=model2_name, color=model2_color, edgecolor='black')
    ax.bar(x + width, IOB_means[:-1], width, yerr=IOB_stds[:-1], capsize=1.5,
        label='IOB', color=IOB_color, edgecolor='black')
    ax.axhline(y=model1_means[-1], color=model1_color, linestyle='--', linewidth=1)
    ax.axhline(y=model2_means[-1], color=model2_color, linestyle='--', linewidth=1)
    ax.axhline(y=IOB_means[-1], color=IOB_color, linestyle='--', linewidth=1)

    # Axis labels and formatting
    ax.set_xlabel("Region")
    ax.set_ylabel(metric_name)
    ax.set_xticks(x)
    ax.set_xticklabels(region_labels, rotation=45, ha="right")
    ax.legend(loc='lower left', frameon=True, edgecolor='black', framealpha=1)
    
    plt.tight_layout()
    
    # Save plots
    if not os.path.exists("plots"):
        os.makedirs("plots")
    plt.savefig(f"results/segmentation_analysis/plots/{metric_name}_scores_bar_chart.png", bbox_inches='tight', dpi=300)
    plt.savefig(f"results/segmentation_analysis/plots/{metric_name}_scores_bar_chart.eps", bbox_inches='tight', format='eps')
    
    # plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Paper plotting utilities for rPCI segmentation.")
    subparsers = parser.add_subparsers(dest="command")

    dice_parser = subparsers.add_parser("dice-boxplots", help="Plot Dice boxplots for two result JSON files.")
    dice_parser.add_argument("--model1-results", required=True)
    dice_parser.add_argument("--model2-results", required=True)
    dice_parser.add_argument("--model1-name", default="SwinUNETR")
    dice_parser.add_argument("--model2-name", default="nnU-Net")

    args = parser.parse_args()

    if args.command == "dice-boxplots":
        plot_dice_boxplots(
            args.model1_results,
            args.model2_results,
            args.model1_name,
            args.model2_name,
        )
    else:
        parser.print_help()