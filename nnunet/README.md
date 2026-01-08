# nnU-Net Integration

This directory contains our modifications and custom configurations for [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet).

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
python scripts/convert_to_nnunet.py \
    --input-dir /path/to/your/data \
    --dataset-id 001 \
    --dataset-name RPCI
```

## Usage

### Plan and Preprocess

```bash
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity
```

### Training

```bash
# Train 3D full resolution
nnUNetv2_train DATASET_ID 3d_fullres FOLD

# Train with our custom trainer (if available)
nnUNetv2_train DATASET_ID 3d_fullres FOLD -tr nnUNetTrainerCustom
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

## Custom Trainers

If you have custom trainers, place them in this directory and reference them using:

```bash
export nnUNet_trainer_path="/path/to/rpci-region-segmentation/nnunet"
```

## Our Modifications

- `custom_trainer.py`: Custom training configurations
- `dataset_conversion.py`: Data conversion utilities

