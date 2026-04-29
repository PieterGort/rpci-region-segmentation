"""
Utility Functions for SwinUNETR

This module provides visualization and logging utilities.
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib


def plot_training_validation_metrics(
    epoch_loss_values: list,
    val_loss_values: list,
    metric_values: list,
    eval_num: int,
    save_path: str,
    fold_idx: int = None,
    lr_history: list = None,
) -> None:
    """
    Plot training and validation loss, validation Dice, and learning rate.
    
    Args:
        epoch_loss_values: List of training loss values
        val_loss_values: List of validation loss values
        metric_values: List of validation Dice scores
        eval_num: Evaluation interval
        save_path: Path to save the plot
        fold_idx: Optional fold index for filename
        lr_history: Optional list of learning rates
    """
    fig = plt.figure(figsize=(14, 6))
    
    # Training and Validation Loss
    plt.subplot(1, 3, 1)
    plt.title("Training and Validation Loss")
    x = [eval_num * (i + 1) for i in range(len(epoch_loss_values))]
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.plot(x, epoch_loss_values, label="Training Loss")
    plt.plot(x, val_loss_values, label="Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation Dice
    plt.subplot(1, 3, 2)
    plt.title("Validation Mean Dice")
    x = [eval_num * (i + 1) for i in range(len(metric_values))]
    plt.xlabel("Iteration")
    plt.ylabel("Dice Score")
    plt.plot(x, metric_values, label="Val Dice", color='green')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Learning Rate
    plt.subplot(1, 3, 3)
    plt.title("Learning Rate")
    if lr_history:
        x = [eval_num * (i + 1) for i in range(len(lr_history))]
        plt.plot(x, lr_history, label="Learning Rate", color='orange')
        plt.xlabel("Iteration")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Adjust save path for fold
    if fold_idx is not None:
        base, ext = os.path.splitext(save_path)
        save_path = f"{base}_fold{fold_idx}{ext}"
    
    plt.savefig(save_path, dpi=150)
    plt.close(fig)


def log_fold_results_to_csv(
    results_dir: str,
    fold_idx: int,
    dice_val_best: float,
    global_step_best: int,
) -> None:
    """
    Log fold results to CSV file.
    
    Args:
        results_dir: Directory for results
        fold_idx: Fold index
        dice_val_best: Best validation Dice score
        global_step_best: Step with best Dice score
    """
    log_path = os.path.join(results_dir, "fold_results.csv")
    write_header = not os.path.exists(log_path)
    
    with open(log_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(["fold", "best_dice", "global_step"])
        writer.writerow([fold_idx, round(dice_val_best, 4), global_step_best])


def visualize_segmentations(
    original_image_path: str,
    ground_truth_path: str,
    prediction_path: str,
    output_path: str,
    slice_idx: int = None,
    title: str = "Prediction",
) -> None:
    """
    Visualize ground truth and prediction side by side.
    
    Args:
        original_image_path: Path to original CT image
        ground_truth_path: Path to ground truth segmentation
        prediction_path: Path to prediction segmentation
        output_path: Path to save visualization
        slice_idx: Slice index to visualize (default: middle slice)
        title: Title for the prediction panel
    """
    # Load images
    original_img = nib.load(original_image_path)
    gt_img = nib.load(ground_truth_path)
    pred_img = nib.load(prediction_path)
    
    original_data = original_img.get_fdata()
    gt_data = gt_img.get_fdata()
    pred_data = pred_img.get_fdata()
    
    if slice_idx is None:
        slice_idx = original_data.shape[2] // 2
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Ground truth
    axes[0].imshow(original_data[:, :, slice_idx], cmap='gray')
    gt_overlay = np.ma.masked_where(gt_data[:, :, slice_idx] == 0, gt_data[:, :, slice_idx])
    axes[0].imshow(gt_overlay, cmap='viridis', alpha=0.5)
    axes[0].set_title('Ground Truth')
    axes[0].axis('off')
    
    # Prediction
    axes[1].imshow(original_data[:, :, slice_idx], cmap='gray')
    pred_overlay = np.ma.masked_where(pred_data[:, :, slice_idx] == 0, pred_data[:, :, slice_idx])
    axes[1].imshow(pred_overlay, cmap='viridis', alpha=0.5)
    axes[1].set_title(title)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def visualize_multiple_slices(
    original_image_path: str,
    ground_truth_path: str,
    prediction_path: str,
    output_dir: str,
    num_slices: int = 5,
    title: str = "Prediction",
) -> None:
    """
    Visualize multiple slices from a volume.
    
    Args:
        original_image_path: Path to original CT image
        ground_truth_path: Path to ground truth segmentation
        prediction_path: Path to prediction segmentation
        output_dir: Directory to save visualizations
        num_slices: Number of slices to visualize
        title: Title for prediction panels
    """
    original_img = nib.load(original_image_path)
    original_data = original_img.get_fdata()
    
    total_slices = original_data.shape[2]
    slice_indices = np.linspace(0, total_slices - 1, num_slices, dtype=int)
    
    os.makedirs(output_dir, exist_ok=True)
    
    for slice_idx in slice_indices:
        output_path = os.path.join(output_dir, f'slice_{slice_idx:03d}.png')
        visualize_segmentations(
            original_image_path, ground_truth_path, prediction_path,
            output_path, slice_idx=slice_idx, title=title
        )


def plot_samples(
    train_ds,
    case_idx: int = 0,
    num_samples: int = 4,
    save_dir: str = "./sample_plots",
) -> None:
    """
    Plot training crop samples.
    
    Args:
        train_ds: Training dataset
        case_idx: Case index to visualize
        num_samples: Number of samples to plot
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    train_sample = train_ds[case_idx]
    
    plt.figure(figsize=(4 * num_samples, 4))
    for i in range(min(num_samples, len(train_sample))):
        plt.subplot(1, num_samples, i + 1)
        plt.title(f"Training Crop {i + 1}")
        img = train_sample[i]["image"]
        mid_slice = img.shape[-1] // 2
        plt.imshow(img[0, :, :, mid_slice].detach().cpu(), cmap="gray")
        plt.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"case_{case_idx}_training_crops.png"), dpi=150)
    plt.close()


def calculate_dice(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Prediction array
        target: Target array
        smooth: Smoothing factor
    
    Returns:
        Dice coefficient
    """
    intersection = (pred * target).sum()
    return (2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth)

