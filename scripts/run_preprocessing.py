import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from preprocessing.dilate_segmentations import (
    preprocess_input_folder,
    process_folder,
    process_folder_parallel,
    test_naming_convention,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run the rPCI preprocessing pipeline on raw scans and .seg.nrrd annotations."
    )
    parser.add_argument(
        "--input-folder",
        required=True,
        help="Folder with Scan_{case_id}_TS.nii.gz and Segmentations_{case_id}_all.seg.nrrd files.",
    )
    parser.add_argument(
        "--output-folder",
        required=True,
        help="Folder to save processed Scan_* and Segmentations_*_expanded.nii.gz files.",
    )
    parser.add_argument("--expansion", type=float, default=2.0, help="Expansion radius in millimeters.")
    parser.add_argument("--chunk-size", type=int, default=100, help="Chunk size for slice-wise processing.")
    parser.add_argument(
        "--target-spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional target voxel spacing. If unset, segmentations are aligned to the image grid.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=5,
        help="Margin in voxels/slices for cropping around segmentation bounds.",
    )
    parser.add_argument("--hu-min", type=int, default=-200, help="Minimum HU value for preprocessing.")
    parser.add_argument("--hu-max", type=int, default=500, help="Maximum HU value for preprocessing.")
    parser.add_argument(
        "--disable-hu-windowing",
        action="store_true",
        help="Disable HU windowing before writing processed scans.",
    )
    parser.add_argument("--parallel-processing", action="store_true", help="Enable parallel processing.")
    parser.add_argument(
        "--fix-names",
        action="store_true",
        help="Move loose files into the expected Scan_/Segmentations_ naming convention.",
    )
    args = parser.parse_args()

    naming_correct, _ = test_naming_convention(args.input_folder)

    if naming_correct:
        print("Naming conventions are correct.")
    elif args.fix_names:
        preprocess_input_folder(args.input_folder)
        print("Naming conventions have been corrected.")
    else:
        raise SystemExit(
            "Input naming does not match the expected convention. "
            "Use --fix-names to move files automatically, or rename them manually."
        )
    
    # check if number of segmentations and images are equal
    num_images = len([file for file in os.listdir(args.input_folder) if file.startswith("Scan_") and file.endswith("_TS.nii.gz")])
    num_segmentations = len([file for file in os.listdir(args.input_folder) if file.startswith("Segmentations_") and file.endswith(".seg.nrrd")])

    if num_images != num_segmentations:
        print(f"Warning: Number of images ({num_images}) and segmentations ({num_segmentations}) do not match. Please check the input folder.")
    else:
        print(f"Number of images and segmentations match ({num_images}).")


    if args.parallel_processing:
        print("Parallel processing enabled.")
        process_folder_parallel(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            expansion_mm=args.expansion,
            chunk_size=args.chunk_size,
            target_spacing=args.target_spacing,
            crop_margin=args.crop_margin,
            max_workers=os.cpu_count(),
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            disable_hu_windowing=args.disable_hu_windowing,
        )
    else:
        print("Parallel processing disabled.")
        process_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder,
            expansion_mm=args.expansion,
            chunk_size=args.chunk_size,
            target_spacing=args.target_spacing,
            crop_margin=args.crop_margin,
            hu_min=args.hu_min,
            hu_max=args.hu_max,
            disable_hu_windowing=args.disable_hu_windowing
        )
    
    print("Processing complete.")

if __name__ == "__main__":
    main()