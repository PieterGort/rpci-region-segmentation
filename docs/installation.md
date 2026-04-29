# Installation Guide

This guide covers installation for both SwinUNETR and nnU-Net components.

## Prerequisites

- Python 3.10 or higher
- CUDA 11.8 or higher (for GPU support)
- At least 16GB RAM
- NVIDIA GPU with at least 12GB VRAM (recommended: 24GB+)

## Option 1: Conda Environment (Recommended)

```bash
# Create environment
conda create -n rpci python=3.10
conda activate rpci

# Install PyTorch with CUDA support
# For CUDA 11.8:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install project dependencies
pip install -r requirements.txt

# Install the repository in editable mode.
# This exposes commands such as rpci-preprocess, rpci-train, and rpci-predict.
pip install -e .
```

## Option 2: Docker

```bash
# Build the Docker image
docker build -t rpci-segmentation .

# Run with GPU support
docker run --gpus all -it rpci-segmentation bash
```

## Installing nnU-Net

nnU-Net requires additional setup:

```bash
# Install nnU-Net v2
pip install nnunetv2

# Set required environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Create the directories
mkdir -p $nnUNet_raw $nnUNet_preprocessed $nnUNet_results
```

Add these to your `~/.bashrc` or `~/.zshrc` for persistence.

## Verifying Installation

After installing the package, verify that the command line entry points and core modules import correctly:

```bash
rpci-preprocess --help
rpci-train --help
rpci-predict --help

python -m preprocessing.convert_to_nnunet --help
python -m analysis.compute_metrics --help
python -m visualization.plot_results --help
```

You can also check the main deep learning dependencies:

```bash
python - <<'PY'
import torch
import monai

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"MONAI version: {monai.__version__}")
PY
```

If you installed nnU-Net, verify its environment variables separately:

```bash
python - <<'PY'
from nnunetv2.paths import nnUNet_raw, nnUNet_preprocessed, nnUNet_results

print(f"nnUNet_raw: {nnUNet_raw}")
print(f"nnUNet_preprocessed: {nnUNet_preprocessed}")
print(f"nnUNet_results: {nnUNet_results}")
PY
```

## Troubleshooting

### CUDA out of memory
- Reduce batch size in configuration
- Enable mixed precision training (`use_amp: true`)
- Use gradient checkpointing

### nnU-Net environment variables not found
- Ensure you've set the environment variables
- Check they're exported in your shell profile
- Restart your terminal after adding to bashrc

### MONAI version conflicts
- Install MONAI after PyTorch
- Use `pip install monai[all]` for all optional dependencies

