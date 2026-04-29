# Deep Learning–Based Segmentation of Radiological Peritoneal Cancer Index Regions

Automated CT segmentation of **radiological Peritoneal Cancer Index (rPCI) regions** using deep learning. This repository contains the code for training and evaluating segmentation models as described in our paper.

> **Deep Learning–Based Segmentation of Radiological Peritoneal Cancer Index Regions in Abdominal Imaging**  
> Pieter C. Gort, Lotte J.S. Ewals, Marion W. Tops-Welten, Cris H.B. Claessens, Joost Nederend, Fons van der Sommen  
> *Department of Electrical Engineering, Eindhoven University of Technology & Catharina Hospital Eindhoven*

## Overview

Peritoneal metastases (PM) are currently assessed using diagnostic laparoscopy to determine Sugarbaker's Peritoneal Cancer Index (PCI), which divides the abdomen into **13 regions** and scores each based on tumor size. A recent [Delphi consensus study](https://doi.org/10.1007/s00330-025-11762-3) defined 3D regions for a **radiological PCI (rPCI)**, enabling standardized imaging-based assessment.

This repository provides:
- **SwinUNETR** implementation using MONAI
- **nnU-Net** training pipeline (following the [official nnU-Net framework](https://github.com/MIC-DKFZ/nnUNet))
- Analysis scripts for computing segmentation metrics and interobserver variability
- Preprocessing and postprocessing utilities

### rPCI Regions (0–12)

| Region | Name | Region | Name |
|--------|------|--------|------|
| 0 | Central | 7 | Right lower |
| 1 | Right upper | 8 | Right flank |
| 2 | Epigastrium | 9 | Upper jejunum |
| 3 | Left upper | 10 | Lower jejunum |
| 4 | Left flank | 11 | Upper ileum |
| 5 | Left lower | 12 | Lower ileum |
| 6 | Pelvis | | |

## Results

Our experiments on 62 CT scans with expert annotations showed:

| Model | Dice Score | HD95 (mm) | ASD (mm) |
|-------|------------|-----------|----------|
| **nnU-Net** | **0.82 ± 0.15** | 13.73 ± 12.03 | 4.10 ± 5.03 |
| SwinUNETR | 0.76 ± 0.18 | 16.82 ± 14.21 | 5.23 ± 6.12 |
| Interobserver | 0.88 ± 0.02 | 11.7 ± 3.3 | 2.5 ± 0.6 |

The source CT scans, annotations, and exact cross-validation splits are not distributed with this repository because they contain confidential clinical data. The code is provided for method transparency and for training/evaluation on locally available data in the documented format. Trained model weights may be published separately in a future release for inference use.

## Repository Structure

```
rpci-region-segmentation/
├── swinunetr/                   # SwinUNETR implementation (MONAI)
│   ├── main.py                  # Training entry point
│   ├── train.py                 # Training loop
│   ├── model.py                 # Model initialization
│   ├── dataset.py               # Data loading utilities
│   ├── predict.py               # Inference script
│   └── utils.py                 # Visualization & logging
│
├── nnunet/                      # nnU-Net utilities
│   └── README.md                # nnU-Net setup guide
│
├── analysis/                    # Evaluation scripts
│   ├── compute_metrics.py       # Dice, HD95, ASD computation
│   └── observer_variability.py  # Interobserver agreement
│
├── preprocessing/               # Data preprocessing
│   ├── convert_to_nnunet.py     # Convert to nnU-Net format
│   ├── dilate_segmentations.py  # Mask dilation
│   └── crop_to_bounds.py        # Crop to segmentation bounds
│
├── scripts/
│   └── run_preprocessing.py     # End-to-end preprocessing wrapper
│
├── postprocessing/              # Post-processing utilities
│   ├── postprocess.py           # Connected component filtering
│   └── resample_to_original.py  # Resample predictions
│
├── visualization/               # Plotting utilities
│   ├── plot_results.py          # Lightweight segmentation overlays
│   └── visualization.py         # Paper plotting utilities
│
├── configs/                     # Configuration files
│   └── swinunetr/default.yaml
│
└── docs/                        # Documentation
    ├── installation.md
    └── data_format.md
```

## Quick Start

### Installation

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

### Training with nnU-Net

For nnU-Net training, follow the [official nnU-Net documentation](https://github.com/MIC-DKFZ/nnUNet):

```bash
# 1. Install nnU-Net
pip install nnunetv2

# 2. Set environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"

# 3. Convert your data to nnU-Net format
python preprocessing/convert_to_nnunet.py \
    --images_dir /path/to/images \
    --segmentations_dir /path/to/labels \
    --output_dir $nnUNet_raw \
    --dataset_name Dataset001_rPCI

# 4. Plan and preprocess
nnUNetv2_plan_and_preprocess -d 001 --verify_dataset_integrity

# 5. Train (5-fold cross-validation)
nnUNetv2_train 001 3d_lowres 0
nnUNetv2_train 001 3d_lowres 1
nnUNetv2_train 001 3d_lowres 2
nnUNetv2_train 001 3d_lowres 3
nnUNetv2_train 001 3d_lowres 4
```

### Training with SwinUNETR

```bash
python -m swinunetr.main \
    --config configs/swinunetr/default.yaml \
    --data-dir /path/to/data \
    --results-dir ./results/swinunetr \
    --fold 0
```

Weights & Biases logging is disabled by default. To enable it, set `wandb.enabled: true` in `configs/swinunetr/default.yaml` and configure your W&B account outside the repository.

### Evaluation

Compute segmentation metrics (Dice, HD95, ASD):

```bash
python analysis/compute_metrics.py \
    --gt-folder /path/to/ground_truth \
    --pred-folder /path/to/predictions \
    --output-dir ./results/metrics
```

Compute interobserver variability:

```bash
python analysis/observer_variability.py \
    --segmentations-folder /path/to/multi_observer_annotations
```

## Data Format

### Input Data Structure

```
/path/to/data/
├── Scan_001_TS.nii.gz              # CT scan (portal venous phase)
├── Scan_002_TS.nii.gz
├── Segmentations_001_all.nii.gz    # rPCI region labels (0-13)
├── Segmentations_002_all.nii.gz
└── ...
```

### Label Encoding

| Label | Region Name |
|-------|-------------|
| 0 | Background |
| 1 | Region 0 (Central) |
| 2 | Region 1 (Right upper) |
| ... | ... |
| 13 | Region 12 (Lower ileum) |

## Preprocessing Pipeline

1. **Expand and combine segmentation masks** by 2mm (compensate for under-segmentation)
2. **Optionally crop to segmentation bounds** with a configurable voxel/slice margin
3. **Convert to nnU-Net format** (or use the processed folder with SwinUNETR)

```bash
# Full preprocessing pipeline
python scripts/run_preprocessing.py \
    --input-folder /path/to/raw_data \
    --output-folder /path/to/processed_data \
    --expansion 2 \
    --crop-margin 5

python preprocessing/convert_to_nnunet.py \
    --images_dir /path/to/processed_data \
    --segmentations_dir /path/to/processed_data \
    --output_dir /path/to/nnUNet_raw \
    --dataset_name Dataset001_rPCI
```

## Citation

If you use this code in your research, please cite:

```bibtex
@article{gort2026rpci,
  title={Deep Learning–Based Segmentation of Radiological Peritoneal Cancer Index Regions in Abdominal Imaging},
  author={Gort, Pieter C. and Ewals, Lotte J.S. and Tops-Welten, Marion W. and Claessens, Cris H.B. and Nederend, Joost and van der Sommen, Fons},
  journal={International Journal of Computer Assisted Radiology and Surgery},
  year={2026}
}
```

Please also cite the following if you use the respective components:

**nnU-Net:**
```bibtex
@article{isensee2021nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and Jaeger, Paul F and Kohl, Simon AA and Petersen, Jens and Maier-Hein, Klaus H},
  journal={Nature methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021}
}
```

**SwinUNETR:**
```bibtex
@article{hatamizadeh2022swinunetr,
  title={Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images},
  author={Hatamizadeh, Ali and Nath, Vishwesh and Tang, Yucheng and Yang, Dong and Roth, Holger R and Xu, Daguang},
  journal={Lecture Notes in Computer Science},
  volume={12962},
  pages={272--284},
  year={2022},
  doi={10.1007/978-3-031-08999-2_22}
}
```

**MONAI:**
```bibtex
@misc{cardoso2022monai,
  title={MONAI: An open-source framework for deep learning in healthcare}, 
  author={M. Jorge Cardoso and Wenqi Li and Richard Brown and others},
  year={2022},
  eprint={2211.02701},
  archivePrefix={arXiv},
  url={https://arxiv.org/abs/2211.02701}
}
```

**rPCI Region Definitions:**
```bibtex
@article{tops2025defining,
  title={Defining region boundaries to assess the peritoneal cancer index on imaging: a Delphi study},
  author={Tops-Welten, Marion W and Ewals, Lotte JS and van Hellemond, Irene EG and others},
  journal={European Radiology},
  year={2025},
  doi={10.1007/s00330-025-11762-3}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) - Self-configuring medical image segmentation
- [MONAI](https://monai.io/) - Medical Open Network for Artificial Intelligence
- [SwinUNETR](https://arxiv.org/abs/2201.01266) - Swin Transformers for medical imaging
- Catharina Cancer Institute, Catharina Hospital Eindhoven
- Hanarth Fund for supporting AI research in oncology
- SURF for access to the Snellius supercomputer

## Contact

For questions or issues, please open a GitHub issue or contact:
- Pieter Gort: p.c.gort@tue.nl
