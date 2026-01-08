# Data Format

This guide describes the expected data format for both SwinUNETR and nnU-Net.

## Overview

Both models expect 3D NIfTI files (`.nii.gz`) containing CT scans and corresponding segmentation masks.

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
    └── splits_final.json        # Cross-validation splits
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

### Using provided preprocessing script:

```bash
python scripts/preprocess.py \
    --input-dir /path/to/raw/data \
    --output-dir /path/to/processed/data \
    --spacing 1.0 1.0 1.0 \
    --intensity-range -175 250
```

## Converting Your Data

If your data is in a different format, use our conversion utilities:

```python
from scripts.convert import convert_to_nnunet_format

convert_to_nnunet_format(
    input_dir="/path/to/your/data",
    output_dir="/path/to/nnUNet_raw/Dataset001_RPCI",
    dataset_name="RPCI"
)
```

