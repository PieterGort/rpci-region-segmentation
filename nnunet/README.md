# nnU-Net Integration

This directory documents how to use [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) with the rPCI data format. The actual nnU-Net framework is installed separately; this repository provides a dataset conversion utility rather than vendoring nnU-Net code.

## Setup

### 1. Install nnU-Net

```bash
pip install nnunetv2
```

### 2. Set Environment Variables

Add these to your `~/.bashrc` or `~/.zshrc`:

```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### 3. Prepare Your Dataset

Convert your data to nnU-Net format:

```bash
python preprocessing/convert_to_nnunet.py \
    --images_dir /path/to/processed_data \
    --segmentations_dir /path/to/processed_data \
    --output_dir "$nnUNet_raw" \
    --dataset_name Dataset001_rPCI
```

## Usage

### Plan and Preprocess

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Training

```bash
# Train one fold. Use the configuration selected during nnU-Net planning.
nnUNetv2_train DATASET_ID 3d_fullres FOLD
```

### Inference

```bash
nnUNetv2_predict \
    -i /path/to/input/images \
    -o /path/to/output \
    -d DATASET_ID \
    -c 3d_fullres \
    -f FOLD
```

## Notes

- The clinical dataset and exact paper splits are confidential and are not distributed in this repository.
- No custom nnU-Net trainer is included here. Use the official nnU-Net v2 trainers unless you add and document your own.
- The conversion utility lives at `preprocessing/convert_to_nnunet.py`.

