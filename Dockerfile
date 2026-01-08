# RPCI Region Segmentation Docker Image
# Supports both SwinUNETR and nnU-Net

FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install nnU-Net (optional, uncomment if needed)
# RUN pip install nnunetv2

# Copy the code
COPY . .

# Set environment variables for nnU-Net (override with -e flag when running)
ENV nnUNet_raw=/data/nnUNet_raw
ENV nnUNet_preprocessed=/data/nnUNet_preprocessed
ENV nnUNet_results=/data/nnUNet_results

# Default command
CMD ["python", "-m", "swinunetr.main", "--help"]

# Usage examples:
# 
# Build:
#   docker build -t rpci-segmentation .
#
# Train SwinUNETR:
#   docker run --gpus all -v /path/to/data:/data rpci-segmentation \
#       python -m swinunetr.main \
#       --config configs/swinunetr/default.yaml \
#       --data-dir /data/dataset \
#       --results-dir /data/results
#
# Run nnU-Net:
#   docker run --gpus all \
#       -v /path/to/data:/data \
#       -e nnUNet_raw=/data/nnUNet_raw \
#       -e nnUNet_preprocessed=/data/nnUNet_preprocessed \
#       -e nnUNet_results=/data/nnUNet_results \
#       rpci-segmentation \
#       nnUNetv2_train DATASET_ID 3d_fullres 0

