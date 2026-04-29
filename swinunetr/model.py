"""
SwinUNETR Model Initialization

This module provides utilities for initializing and loading SwinUNETR models.
"""

import os
import torch
from monai.networks.nets import SwinUNETR


def initialize_model(
    pretrained_weights: str = None,
    device: torch.device = None,
    in_channels: int = 1,
    out_channels: int = 14,
    feature_size: int = 48,
    drop_rate: float = 0.0,
    attn_drop_rate: float = 0.0,
    dropout_path_rate: float = 0.0,
    use_checkpoint: bool = True,
) -> SwinUNETR:
    """
    Initialize a SwinUNETR model with optional pretrained weights.
    
    Args:
        pretrained_weights: Path to pretrained weights file
        device: PyTorch device to load model on
        in_channels: Number of input channels
        out_channels: Number of output classes
        feature_size: Feature size for the model
        drop_rate: Dropout rate
        attn_drop_rate: Attention dropout rate
        dropout_path_rate: Dropout path rate
        use_checkpoint: Whether to use gradient checkpointing
    
    Returns:
        Initialized SwinUNETR model
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = SwinUNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        drop_rate=drop_rate,
        attn_drop_rate=attn_drop_rate,
        dropout_path_rate=dropout_path_rate,
        use_checkpoint=use_checkpoint,
    ).to(device)
    
    if pretrained_weights and os.path.exists(pretrained_weights):
        try:
            ckpt = torch.load(pretrained_weights, map_location="cpu", weights_only=True)
        except TypeError:
            # Older PyTorch versions don't have weights_only parameter
            ckpt = torch.load(pretrained_weights, map_location="cpu")
        
        # Handle different checkpoint formats
        state_dict = ckpt.get("state_dict", ckpt.get("model", ckpt))
        
        # Remove 'module.' prefix if present (from DataParallel)
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
        
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from: {pretrained_weights}")
        print(f"  Missing keys: {len(missing)}, Unexpected keys: {len(unexpected)}")
    else:
        if pretrained_weights:
            print(f"Warning: Pretrained weights not found at {pretrained_weights}")
        print("Initializing model with random weights.")
    
    return model


def load_model(
    model_path: str,
    device: str = None,
    in_channels: int = 1,
    out_channels: int = 14,
    feature_size: int = 48,
) -> SwinUNETR:
    """
    Load a trained SwinUNETR model from checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load model on ('cuda' or 'cpu')
        in_channels: Number of input channels
        out_channels: Number of output classes
        feature_size: Feature size for the model
    
    Returns:
        Loaded SwinUNETR model in evaluation mode
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    try:
        model = SwinUNETR(
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
            spatial_dims=3
        ).to(device)
    except TypeError:
        # Fallback for older MONAI versions
        model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=out_channels,
            feature_size=feature_size,
            use_checkpoint=True,
            spatial_dims=3
        ).to(device)
    
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
    except TypeError:
        model.load_state_dict(torch.load(model_path))
    
    model.eval()
    return model

