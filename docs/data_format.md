# Data Format

This guide describes the expected data format for both SwinUNETR and nnU-Net.

## Overview

Both models expect 3D NIfTI files (`.nii.gz`) containing CT scans and corresponding segmentation masks.

The study data and exact paper splits are not included in this repository because they contain confidential clinical data. Use the formats below for your own data, and provide your own split file when training SwinUNETR.

## SwinUNETR Format

```
/path/to/data/
├── imagesTr/
│   ├── case_001_0000.nii.gz    # Training images
│   ├── case_002_0000.nii.gz
│   └── ...
├── labelsTr/
│   ├── case_001.nii.gz         # Training labels
│   ├── case_002.nii.gz
│   └── ...
├── imagesTs/                    # Optional: Test images
│   └── ...
└── dataset/
    └── splits_final.json        # Cross-validation splits, supplied by the user
```

### Splits File Format

```json
[
  {
    "train": ["case_001", "case_002", ...],
    "val": ["case_010", "case_011", ...]
  },
  // ... more folds
]
```

## nnU-Net Format

nnU-Net uses a specific dataset structure:

```
nnUNet_raw/
└── Dataset001_RPCI/
    ├── imagesTr/
    │   ├── RPCI_001_0000.nii.gz
    │   └── ...
    ├── labelsTr/
    │   ├── RPCI_001.nii.gz
    │   └── ...
    └── dataset.json
```

### dataset.json Format

```json
{
    "channel_names": {
        "0": "CT"
    },
    "labels": {
        "background": 0,
        "region_1": 1,
        "region_2": 2,
        // ... more labels
    },
    "numTraining": 50,
    "file_ending": ".nii.gz"
}
```

## Label Encoding

| Label ID | Region Name |
|----------|-------------|
| 0 | Background |
| 1 | Region 1 |
| 2 | Region 2 |
| ... | ... |

## Preprocessing

### Recommended preprocessing steps:

1. **Resampling**: Isotropic spacing (e.g., 1.0mm³)
2. **Intensity normalization**: Window/level for CT (-175 to 250 HU)
3. **Cropping**: Remove empty regions around the body

### Using the provided preprocessing script

```bash
python scripts/run_preprocessing.py \
    --input-folder /path/to/raw_data \
    --output-folder /path/to/processed_data \
    --expansion 2 \
    --crop-margin 5 \
    --target-spacing 1.0 1.0 1.0
```

Raw preprocessing expects pairs named `Scan_{case_id}_TS.nii.gz` and `Segmentations_{case_id}_all.seg.nrrd`. Pass `--fix-names` only if you want the script to move loose files into that naming convention.

## Converting Your Data

If your data is in a different format, use our conversion utilities:

```python
from preprocessing.convert_to_nnunet import convert_to_nnunet_format

convert_to_nnunet_format(
    images_dir="/path/to/processed_data",
    segmentations_dir="/path/to/processed_data",
    output_dir="/path/to/nnUNet_raw",
    dataset_name="Dataset001_RPCI",
)
```

