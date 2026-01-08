"""
SwinUNETR Training Script for RPCI Region Segmentation

This script provides the main entry point for training SwinUNETR models
on medical imaging segmentation tasks.

Usage:
    python -m swinunetr.main --config configs/swinunetr/default.yaml --data-dir /path/to/data
"""

import os
import argparse
import json
import yaml
import torch
import numpy as np
import wandb

from .dataset import load_predefined_splits
from .train import train
from .model import initialize_model
from .utils import plot_training_validation_metrics, plot_samples

from monai.losses import DiceCELoss
from monai.transforms import (
    AsDiscrete,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotated,
    EnsureTyped,
)
from monai.metrics import DiceMetric
from monai.data import CacheDataset, ThreadDataLoader, pad_list_data_collate


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: dict, path: str) -> None:
    """Save configuration to JSON file."""
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train SwinUNETR model for region segmentation."
    )
    
    # Configuration
    parser.add_argument(
        "--config", type=str, default="configs/swinunetr/default.yaml",
        help="Path to configuration file."
    )
    
    # Data paths
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Path to the folder containing NIfTI images and segmentations."
    )
    parser.add_argument(
        "--json-path", type=str, default="dataset/splits_final.json",
        help="Path to the dataset splits JSON file."
    )
    parser.add_argument(
        "--results-dir", type=str, default="./results/",
        help="Path to directory to save results."
    )
    
    # Model settings
    parser.add_argument(
        "--pretrained-weights", type=str, default=None,
        help="Path to pretrained weights file."
    )
    
    # Training settings
    parser.add_argument(
        "--run-name", type=str, default="SwinUNETR",
        help="Name of the run for logging."
    )
    parser.add_argument(
        "--num-workers", type=int, default=1,
        help="Number of workers for data loading."
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Batch size for training."
    )
    parser.add_argument(
        "--max-iter", type=int, default=25000,
        help="Maximum number of iterations for training."
    )
    parser.add_argument(
        "--eval-num", type=int, default=500,
        help="Number of iterations between evaluations."
    )
    parser.add_argument(
        "--fold", type=int, default=0,
        help="Fold number for cross-validation (0 to 4)."
    )
    parser.add_argument(
        "--num-crop-samples", type=int, default=2,
        help="Number of crops to sample from training data."
    )
    
    # Misc
    parser.add_argument(
        "--visualize-sample", action="store_true", default=False,
        help="Visualize a sample from validation dataset."
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Random seed for reproducibility."
    )
    
    return parser.parse_args()


def setup_transforms(config: dict, device: torch.device):
    """
    Set up training and validation transforms.
    
    Args:
        config: Configuration dictionary with preprocessing parameters
        device: PyTorch device
    
    Returns:
        Tuple of (train_transforms, val_transforms)
    """
    # Get preprocessing parameters from config
    prep = config.get('preprocessing', {})
    aug = config.get('augmentation', {})
    
    a_min = prep.get('intensity_range', {}).get('a_min', -175)
    a_max = prep.get('intensity_range', {}).get('a_max', 250)
    b_min = prep.get('intensity_range', {}).get('b_min', 0)
    b_max = prep.get('intensity_range', {}).get('b_max', 1)
    pixdim = prep.get('spacing', {}).get('pixdim', [1.5, 1.5, 1.5])
    crop_size = prep.get('crop_size', [96, 96, 96])
    
    train_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min, a_max=a_max,
            b_min=b_min, b_max=b_max,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=tuple(pixdim),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], track_meta=False),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=tuple(crop_size),
            pos=1, neg=1,
            num_samples=config.get('training', {}).get('num_crop_samples', 2),
            image_key="image",
            image_threshold=0,
        ),
        RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=aug.get('flip_prob', 0.1)),
        RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=aug.get('flip_prob', 0.1)),
        RandFlipd(keys=["image", "label"], spatial_axis=[2], prob=aug.get('flip_prob', 0.1)),
        RandRotated(
            keys=["image", "label"],
            prob=aug.get('rotation_prob', 0.5),
            range_x=aug.get('rotation_range', [-15, 15]),
            range_y=aug.get('rotation_range', [-15, 15]),
            range_z=aug.get('rotation_range', [-15, 15]),
            mode=["bilinear", "nearest"]
        ),
        RandShiftIntensityd(
            keys=["image"],
            offsets=aug.get('intensity_shift', {}).get('offsets', 0.1),
            prob=aug.get('intensity_shift', {}).get('prob', 0.5),
        ),
    ])
    
    val_transforms = Compose([
        LoadImaged(keys=["image", "label"], ensure_channel_first=True),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=a_min, a_max=a_max,
            b_min=b_min, b_max=b_max,
            clip=True
        ),
        CropForegroundd(keys=["image", "label"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(
            keys=["image", "label"],
            pixdim=tuple(pixdim),
            mode=("bilinear", "nearest"),
        ),
        EnsureTyped(keys=["image", "label"], device=device, track_meta=True),
    ])
    
    return train_transforms, val_transforms


def main():
    """Main training function."""
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # Load configuration
    config = load_config(args.config)
    
    # Get model parameters from config
    model_config = config.get('model', {})
    training_config = config.get('training', {})
    
    # Override with command line arguments
    learning_rate = training_config.get('learning_rate', 1e-4)
    weight_decay = training_config.get('weight_decay', 1e-5)
    max_iter = args.max_iter or training_config.get('max_iter', 25000)
    args.max_iter = max_iter
    
    # Setup output directory
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Save config for reproducibility
    save_config(config, os.path.join(args.results_dir, "config.json"))
    
    # Initialize wandb (optional)
    wandb_config = config.get('wandb', {})
    if wandb_config.get('enabled', True):
        wandb.init(
            project=wandb_config.get('project', 'rpci-segmentation'),
            entity=wandb_config.get('entity'),
            name=f"SwinUNETR_fold{args.fold}_{args.run_name}",
            config=config
        )
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup transforms
    train_transforms, val_transforms = setup_transforms(config, device)
    
    # Load data splits
    splits = load_predefined_splits(args.json_path, args.data_dir)
    fold_idx = args.fold
    print(f"\n--- Starting Fold {fold_idx} ---")
    
    train_data, val_data = splits[fold_idx]
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    
    # Save fold data
    fold_data = {"training": train_data, "validation": val_data}
    with open(os.path.join(args.results_dir, f"fold_{fold_idx}.json"), "w") as f:
        json.dump(fold_data, f, indent=2)
    
    # Create datasets
    train_ds = CacheDataset(
        train_data, train_transforms,
        cache_num=len(train_data), cache_rate=1.0,
        num_workers=args.num_workers
    )
    val_ds = CacheDataset(
        val_data, val_transforms,
        cache_num=len(val_data), cache_rate=1.0,
        num_workers=args.num_workers
    )
    
    # Create data loaders
    train_loader = ThreadDataLoader(
        train_ds, num_workers=args.num_workers,
        batch_size=args.batch_size, shuffle=True,
        collate_fn=pad_list_data_collate
    )
    val_loader = ThreadDataLoader(
        val_ds, num_workers=0, batch_size=args.batch_size,
        collate_fn=pad_list_data_collate
    )
    
    # Visualize samples if requested
    if args.visualize_sample:
        plot_samples(train_ds, num_samples=args.num_crop_samples, save_dir="./training_samples")
    
    # Initialize model
    num_classes = model_config.get('out_channels', 14)
    model = initialize_model(
        pretrained_weights=args.pretrained_weights,
        device=device,
        in_channels=model_config.get('in_channels', 1),
        out_channels=num_classes,
        feature_size=model_config.get('feature_size', 48),
    )
    
    # Setup training components
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    scaler = torch.amp.GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_iter // len(train_loader)
    )
    
    # Setup metrics
    dice_metric = DiceMetric(
        include_background=False,
        num_classes=num_classes,
        reduction="mean"
    )
    
    # Train
    dice_val_best, global_step_best, *_, best_label_metrics = train(
        model=model,
        train_loader=train_loader,
        loss_function=loss_function,
        optimizer=optimizer,
        scaler=scaler,
        num_crop_samples=args.num_crop_samples,
        scheduler=scheduler,
        val_loader=val_loader,
        dice_metric=dice_metric,
        args=args,
        fold_idx=fold_idx,
        num_classes=num_classes,
    )
    
    # Save results
    print(f"Fold {fold_idx} finished. Best Dice: {dice_val_best:.4f} at step {global_step_best}")
    
    # Define label names (customize for your dataset)
    label_names = {i: f"Region{i}" for i in range(num_classes)}
    label_names[0] = "Background"
    
    # Save per-label metrics
    metrics_file = os.path.join(args.results_dir, f"per_label_metrics_fold{fold_idx}.csv")
    with open(metrics_file, 'w') as f:
        headers = [
            "Label", "Dice", "Dice_std",
            "TP", "TP_std", "FP", "FP_std", "FN", "FN_std", "TN", "TN_std",
            "n_pred", "n_ref"
        ]
        f.write(",".join(headers) + "\n")
        
        for label_idx in range(1, num_classes):
            metrics = best_label_metrics[label_idx]
            row = [
                label_names[label_idx],
                f"{metrics['Dice']:.3f}", f"{metrics['Dice_std']:.3f}",
                f"{metrics['TP']:.1f}", f"{metrics['TP_std']:.1f}",
                f"{metrics['FP']:.1f}", f"{metrics['FP_std']:.1f}",
                f"{metrics['FN']:.1f}", f"{metrics['FN_std']:.1f}",
                f"{metrics['TN']:.1f}", f"{metrics['TN_std']:.1f}",
                f"{metrics['n_pred']:.1f}", f"{metrics['n_ref']:.1f}",
            ]
            f.write(",".join(row) + "\n")
        
        # Overall metrics
        all_label_dice = [best_label_metrics[i]['Dice'] for i in range(1, num_classes)]
        overall_mean = np.mean(all_label_dice)
        overall_std = np.std(all_label_dice)
        overall_row = ["Overall", f"{overall_mean:.4f}", f"{overall_std:.4f}"] + [""] * (len(headers) - 3)
        f.write(",".join(overall_row) + "\n")
    
    print(f"Fold {fold_idx} - Overall Mean Dice: {overall_mean:.4f} ± {overall_std:.4f}")
    
    # Finish wandb
    if wandb_config.get('enabled', True):
        wandb.finish()


if __name__ == "__main__":
    main()

