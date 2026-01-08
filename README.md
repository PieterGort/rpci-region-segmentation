# RPCI Region Segmentation

Automated CT segmentation of peritoneal metastases regions using deep learning. This repository contains code for training and evaluating two state-of-the-art 3D medical image segmentation architectures:

- **SwinUNETR** — Transformer-based architecture using shifted windows (MONAI implementation)
- **nnU-Net** — Self-configuring framework for biomedical image segmentation

## Overview

This repository accompanies the paper:

> **[Your Paper Title]**  
> Authors: [Your Names]  
> Published in: [Journal/Conference, Year]

We provide complete training pipelines, configuration files, and evaluation scripts to reproduce our results on peritoneal region segmentation from abdominal CT scans.

## Repository Structure

```
rpci-region-segmentation/
├── swinunetr/              # SwinUNETR implementation
│   ├── main.py             # Training entry point
│   ├── train.py            # Training loop
│   ├── predict.py          # Inference script
│   ├── dataset.py          # Data loading utilities
│   └── utils.py            # Helper functions
│
├── nnunet/                 # nnU-Net modifications
│   └── ...                 # Custom trainers and configurations
│
├── configs/                # Configuration files
│   ├── swinunetr/          # SwinUNETR configs
│   └── nnunet/             # nnU-Net configs
│
├── scripts/                # Utility scripts
│   ├── preprocess.py       # Data preprocessing
│   └── evaluate.py         # Evaluation metrics
│
├── experiments/            # Experiment reproduction
│   └── run_all.py          # Reproduce paper results
│
└── docs/                   # Documentation
    ├── installation.md     # Detailed installation guide
    ├── data_format.md      # Data preparation instructions
    └── experiments.md      # Experiment details
```

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/PieterGort/rpci-region-segmentation.git
cd rpci-region-segmentation

# Create conda environment
conda create -n rpci python=3.10
conda activate rpci

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install dependencies
pip install -r requirements.txt
```

### 2. Data Preparation

Organize your data in the following structure:

```
/path/to/data/
├── imagesTr/
│   ├── case_001_0000.nii.gz
│   ├── case_002_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_001.nii.gz
│   ├── case_002.nii.gz
│   └── ...
└── dataset.json
```

See [docs/data_format.md](docs/data_format.md) for detailed data preparation instructions.

### 3. Training

#### SwinUNETR

```bash
python swinunetr/main.py \
    --config configs/swinunetr/default.yaml \
    --data-dir /path/to/data \
    --output-dir ./results/swinunetr
```

#### nnU-Net

```bash
# Set environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# Plan and preprocess
nnUNetv2_plan_and_preprocess -d DATASET_ID --verify_dataset_integrity

# Train
nnUNetv2_train DATASET_ID 3d_fullres FOLD
```

### 4. Inference

```bash
# SwinUNETR
python swinunetr/predict.py \
    --model-path ./results/swinunetr/best_model.pth \
    --input-dir /path/to/test/images \
    --output-dir ./predictions

# nnU-Net
nnUNetv2_predict -i /path/to/test/images -o ./predictions -d DATASET_ID -c 3d_fullres
```

## Docker

For reproducibility, we provide Docker containers:

```bash
# Build
docker build -t rpci-segmentation .

# Run SwinUNETR training
docker run --gpus all -v /path/to/data:/data rpci-segmentation \
    python swinunetr/main.py --data-dir /data
```

## Results

| Model | Dice Score | HD95 (mm) | Training Time |
|-------|------------|-----------|---------------|
| SwinUNETR | X.XX ± X.XX | X.XX ± X.XX | ~Xh |
| nnU-Net 3D | X.XX ± X.XX | X.XX ± X.XX | ~Xh |

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.0
- CUDA ≥ 11.8
- MONAI ≥ 1.3 (for SwinUNETR)
- nnU-Net v2 (for nnU-Net experiments)

See [requirements.txt](requirements.txt) for complete dependencies.

## Citation

If you use this code in your research, please cite:

```bibtex
@article{yourpaper2026,
  title={Your Paper Title},
  author={Your Names},
  journal={Journal Name},
  year={2026}
}
```

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work builds upon:
- [MONAI](https://monai.io/) - Medical Open Network for Artificial Intelligence
- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - Self-configuring method for biomedical image segmentation
- [SwinUNETR](https://arxiv.org/abs/2201.01266) - Swin Transformers for Semantic Segmentation of Brain Tumors

## Contact

For questions or issues, please open a GitHub issue or contact [your.email@tue.nl].
