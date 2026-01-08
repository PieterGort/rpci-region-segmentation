"""
SwinUNETR Training Loop

This module implements the training and validation loops for SwinUNETR.
"""

import os
import numpy as np
import torch
import wandb
from tqdm import tqdm
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete

from .utils import plot_training_validation_metrics, log_fold_results_to_csv


def train(
    model,
    train_loader,
    loss_function,
    optimizer,
    scaler,
    scheduler,
    num_crop_samples,
    val_loader,
    dice_metric,
    args,
    fold_idx=None,
    num_classes=14,
):
    """
    Main training loop for SwinUNETR.
    
    Args:
        model: The SwinUNETR model
        train_loader: Training data loader
        loss_function: Loss function
        optimizer: Optimizer
        scaler: Gradient scaler for mixed precision
        scheduler: Learning rate scheduler
        num_crop_samples: Number of crop samples for sliding window inference
        val_loader: Validation data loader
        dice_metric: MONAI Dice metric
        args: Training arguments
        fold_idx: Current fold index
        num_classes: Number of output classes
    
    Returns:
        Tuple of (best_dice, best_step, losses, metrics, val_losses, steps, best_label_metrics)
    """
    # Initialize training state
    global_step = 0
    dice_val_best = 0.0
    global_step_best = 0
    
    epoch_loss_values = []
    metric_values = []
    val_loss_values = []
    steps_validated_at = []
    lr_history = []
    best_label_metrics = None
    
    device = next(model.parameters()).device
    
    while global_step < args.max_iter:
        model.train()
        epoch_loss = 0
        step = 0
        
        epoch_iterator = tqdm(
            train_loader,
            desc=f"Training ({global_step} / {args.max_iter} Steps) (loss=X.X)",
            dynamic_ncols=True
        )
        
        for batch in epoch_iterator:
            step += 1
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            
            with torch.autocast("cuda"):
                logit_map = model(x)
                loss = loss_function(logit_map, y)
            
            scaler.scale(loss).backward()
            epoch_loss += loss.item()
            scaler.unscale_(optimizer)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
            epoch_iterator.set_description(
                f"Training ({global_step} / {args.max_iter} Steps) (loss={loss:2.5f})"
            )
            
            # Log to wandb
            try:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/learning_rate": optimizer.param_groups[0]['lr']
                }, step=global_step)
            except wandb.errors.Error:
                pass  # wandb not initialized
            
            # Validation
            if (global_step % args.eval_num == 0 and global_step != 0) or global_step == args.max_iter:
                epoch_iterator_val = tqdm(
                    val_loader,
                    desc="Validate (X / X Steps) (dice=X.X)",
                    dynamic_ncols=True
                )
                dice_val, val_loss, label_metrics = validation(
                    epoch_iterator_val, model, loss_function,
                    dice_metric, global_step, num_crop_samples, num_classes
                )
                
                avg_epoch_loss = epoch_loss / step if step > 0 else 0.0
                epoch_loss_values.append(avg_epoch_loss)
                metric_values.append(dice_val)
                val_loss_values.append(val_loss)
                steps_validated_at.append(global_step)
                lr_history.append(scheduler.get_last_lr()[0])
                
                # Log validation to wandb
                try:
                    wandb.log({
                        "val/dice": dice_val,
                        "val/loss": val_loss
                    }, step=global_step)
                except wandb.errors.Error:
                    pass
                
                if dice_val > dice_val_best:
                    dice_val_best = dice_val
                    global_step_best = global_step
                    best_label_metrics = label_metrics
                    torch.save(
                        model.state_dict(),
                        os.path.join(args.results_dir, f"best_metric_model_fold_{args.fold}.pth")
                    )
                    print(f"Model saved! Best Dice: {dice_val_best:.4f} Current: {dice_val:.4f}")
                else:
                    print(f"Model not saved. Best: {dice_val_best:.4f} Current: {dice_val:.4f}")
                
                # Plot training curves
                plot_filename = os.path.join(args.results_dir, "loss_plot.png")
                plot_training_validation_metrics(
                    epoch_loss_values, val_loss_values, metric_values,
                    args.eval_num, plot_filename,
                    fold_idx=fold_idx, lr_history=lr_history
                )
            
            global_step += 1
            if global_step >= args.max_iter:
                break
        
        scheduler.step()
    
    log_fold_results_to_csv(args.results_dir, fold_idx, dice_val_best, global_step_best)
    
    if best_label_metrics is None:
        best_label_metrics = label_metrics
    
    return (dice_val_best, global_step_best, epoch_loss_values, 
            metric_values, val_loss_values, steps_validated_at, best_label_metrics)


def validation(
    epoch_iterator_val,
    model,
    loss_function,
    dice_metric,
    global_step,
    num_crop_samples,
    num_classes=14,
):
    """
    Validation loop.
    
    Args:
        epoch_iterator_val: Validation data iterator
        model: The model
        loss_function: Loss function
        dice_metric: MONAI Dice metric
        global_step: Current training step
        num_crop_samples: Number of samples for sliding window inference
        num_classes: Number of output classes
    
    Returns:
        Tuple of (mean_dice, mean_loss, per_label_metrics)
    """
    model.eval()
    total_val_loss = 0
    val_steps = 0
    
    # Track metrics per label (excluding background)
    label_stats = {
        i: {
            "dice": [], "iou": [], "tp": [], "fp": [],
            "fn": [], "tn": [], "n_pred": [], "n_ref": []
        }
        for i in range(1, num_classes)
    }
    
    post_label = AsDiscrete(to_onehot=num_classes)
    post_pred = AsDiscrete(argmax=True, to_onehot=num_classes)
    
    with torch.no_grad():
        for batch in epoch_iterator_val:
            val_steps += 1
            val_inputs = batch["image"].cuda()
            val_labels = batch["label"].cuda()
            
            with torch.autocast("cuda"):
                val_outputs = sliding_window_inference(
                    val_inputs, (96, 96, 96), num_crop_samples, model
                )
                val_loss = loss_function(val_outputs, val_labels)
                total_val_loss += val_loss.item()
            
            val_labels_list = decollate_batch(val_labels)
            val_labels_convert = [post_label(t) for t in val_labels_list]
            
            val_outputs_list = decollate_batch(val_outputs)
            val_output_convert = [post_pred(t) for t in val_outputs_list]
            
            # Calculate per-label metrics
            for pred, label in zip(val_output_convert, val_labels_convert):
                p = pred.float()
                t = label.float()
                
                tp = (p * t).sum(dim=(1, 2, 3))
                fp = (p * (1 - t)).sum(dim=(1, 2, 3))
                fn = ((1 - p) * t).sum(dim=(1, 2, 3))
                tn = ((1 - p) * (1 - t)).sum(dim=(1, 2, 3))
                n_pred = p.sum(dim=(1, 2, 3))
                n_ref = t.sum(dim=(1, 2, 3))
                dice = (2 * tp + 1e-5) / (n_pred + n_ref + 1e-5)
                iou = (tp + 1e-5) / (tp + fp + fn + 1e-5)
                
                for i in range(1, num_classes):
                    label_stats[i]["dice"].append(dice[i].item())
                    label_stats[i]["iou"].append(iou[i].item())
                    label_stats[i]["tp"].append(tp[i].item())
                    label_stats[i]["fp"].append(fp[i].item())
                    label_stats[i]["fn"].append(fn[i].item())
                    label_stats[i]["tn"].append(tn[i].item())
                    label_stats[i]["n_pred"].append(n_pred[i].item())
                    label_stats[i]["n_ref"].append(n_ref[i].item())
            
            dice_metric(y_pred=val_output_convert, y=val_labels_convert)
            epoch_iterator_val.set_description(
                f"Validate ({global_step} / {len(epoch_iterator_val)} Steps)"
            )
        
        mean_dice_val = dice_metric.aggregate().item()
        dice_metric.reset()
        mean_val_loss = total_val_loss / val_steps if val_steps > 0 else 0
        
        # Aggregate per-label metrics
        label_metrics = {}
        for i, stats in label_stats.items():
            label_metrics[i] = {
                "Dice": np.mean(stats["dice"]),
                "Dice_std": np.std(stats["dice"]),
                "TP": np.mean(stats["tp"]),
                "TP_std": np.std(stats["tp"]),
                "FP": np.mean(stats["fp"]),
                "FP_std": np.std(stats["fp"]),
                "FN": np.mean(stats["fn"]),
                "FN_std": np.std(stats["fn"]),
                "TN": np.mean(stats["tn"]),
                "TN_std": np.std(stats["tn"]),
                "n_pred": np.mean(stats["n_pred"]),
                "n_ref": np.mean(stats["n_ref"]),
            }
        
        print(f"Validation Dice: {mean_dice_val:.4f} Loss: {mean_val_loss:.4f}")
    
    return mean_dice_val, mean_val_loss, label_metrics

