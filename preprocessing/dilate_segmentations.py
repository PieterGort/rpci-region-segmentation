import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from .crop_to_bounds import (
        crop_to_segmentation_bounds,
        resample_to_reference,
        resample_to_spacing_reference,
    )
except ImportError:
    from crop_to_bounds import (
        crop_to_segmentation_bounds,
        resample_to_reference,
        resample_to_spacing_reference,
    )

import SimpleITK as sitk
import numpy as np

label_filter = sitk.LabelShapeStatisticsImageFilter()

def validate_and_preprocess_image(image, hu_min=-200, hu_max=500):
    """
    Validate and preprocess CT image using SimpleITK IntensityWindowingImageFilter.
    
    Parameters:
    - image: SimpleITK image
    - hu_min: Minimum HU value (-200 for soft tissues)
    - hu_max: Maximum HU value (500 for enhanced tissues/calcifications)
    
    Returns:
    - processed_image: SimpleITK image with validated and windowed HU values
    """
    # Get original statistics for reporting
    stats_filter = sitk.StatisticsImageFilter()
    stats_filter.Execute(image)
    original_min, original_max = stats_filter.GetMinimum(), stats_filter.GetMaximum()
    
    filter = sitk.IntensityWindowingImageFilter()

    filter.SetWindowMinimum(hu_min)
    filter.SetWindowMaximum(hu_max)
    filter.SetOutputMinimum(hu_min)
    filter.SetOutputMaximum(hu_max)
    processed_image = filter.Execute(image)
    
    # Report clipping statistics
    if original_min < hu_min or original_max > hu_max:
        print(f"  Info: Applied HU windowing [{hu_min}, {hu_max}]")
        print(f"  Original HU range: [{original_min:.1f}, {original_max:.1f}]")
    
    return processed_image

def validate_image_properties(image, case_name):
    """
    Validate image properties and log diagnostics.
    """
    # Check spacing
    spacing = image.GetSpacing()
    size = image.GetSize()
    
    print(f"  Image properties for {case_name}:")
    print(f"    Spacing: {spacing}")
    print(f"    Size: {size}")
    print(f"    Total voxels: {np.prod(size):,}")
    
    # Check for unusual spacing
    for i, sp in enumerate(spacing):
        if not (0.5 <= sp <= 5.0):
            print(f"    Warning: Unusual spacing in dimension {i}: {sp}mm")
    
    # Check for very large volumes
    total_voxels = np.prod(size)
    if total_voxels > 500_000_000:  # 500M voxels
        print(f"    Warning: Very large volume ({total_voxels:,} voxels) - may cause memory issues")
    
    return True


def combine_segmentation(segmentation, chunk_size=100):

    # Initialize combined segmentation and combined mask
    combined_segmentation = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
    combined_segmentation.CopyInformation(segmentation)
    combined_mask = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
    combined_mask.CopyInformation(segmentation)

    region_labels = {f"Region {i}": i + 1 for i in range(13)}  # Map region names to labels
    meta_keys = segmentation.GetMetaDataKeys()
    segments = []
    seen_regions = set()
    
    for key in meta_keys:
        if key.startswith("Segment") and key.endswith("Name"):
            segment_id = key.split("_")[0]
            segment_name = segmentation.GetMetaData(key)

            # Check for "Region {index}" format first
            if segment_name.startswith("Region"):
                parts = segment_name.split(" ")
                if len(parts) == 2 and parts[1].isdigit():
                    index = int(parts[1])
                else:
                    index = -1

            # If not found, check for "Segment_1_{index}" format
            elif segment_name.startswith("Segment_1"):
                print(f"Wrong format found in segmentations! -> {segment_name}")
                parts = segment_name.split("_")
                if len(parts) == 3 and parts[2].isdigit():
                    index = int(parts[2])
                else:
                    index = -1

            else:
                index = -1
                # print(f"Warning: Unrecognized segment name format for {segment_name}. Skipping.")

            if index != -1 and f"Region {index}" in region_labels:
                segment_name = f"Region {index}"
                if segment_name in seen_regions:
                    print(f"Warning: Duplicate region name found for {segment_name}. Skipping.")
                    continue

                seen_regions.add(segment_name)

                layer_key = f"{segment_id}_Layer"
                label_key = f"{segment_id}_LabelValue"
                segment_layer = int(segmentation.GetMetaData(layer_key))
                segment_label = int(segmentation.GetMetaData(label_key))

                segments.append({
                    "id": segment_id,
                    "name": segment_name,
                    "layer": segment_layer,
                    "label": segment_label,
                    "new_label": region_labels[segment_name]
                })
    
    # Sort the segments by new_label
    segments = sorted(segments, key=lambda x: x["new_label"])

    # Add the binary masks and distance maps for each label
    has_overlaps = segmentation.GetNumberOfComponentsPerPixel() > 1
    for segment in segments:
        if has_overlaps:
            binary_mask = sitk.Equal(sitk.VectorIndexSelectionCast(segmentation, segment["layer"]), segment["label"])
        else:
            binary_mask = sitk.Equal(segmentation, segment["label"])

        # Only take the largest connected component
        labeled_mask = sitk.ConnectedComponent(binary_mask)
        label_filter.Execute(labeled_mask)

        if label_filter.GetNumberOfLabels() == 0:
            print(f"Warning: No connected components found for {segment['name']}. Skipping.")
            print(f"Setting all voxels to 0 for {segment['name']}.")
            binary_mask = sitk.Image(binary_mask.GetSize(), sitk.sitkUInt8)
            binary_mask.CopyInformation(segmentation)
            distance_map = sitk.SignedMaurerDistanceMap(binary_mask, squaredDistance=False, useImageSpacing=True)
        else:            
            largest_label = max(label_filter.GetLabels(), key=lambda x: label_filter.GetPhysicalSize(x))
            binary_mask = sitk.Equal(labeled_mask, largest_label)
            del labeled_mask  # Release memory

            distance_map = sitk.SignedMaurerDistanceMap(binary_mask, squaredDistance=False, useImageSpacing=True)

        segment["binary_mask"] = binary_mask
        segment["distance_map"] = distance_map
        
    # Directly assign labels if there are no overlaps
    if not has_overlaps:
        for segment in segments:
            combined_segmentation += segment["new_label"] * segment["binary_mask"]
        
        # Set the distance maps to 0 where the combined segmentation > 0
        for segment in segments:
            segment["distance_map"] = sitk.Mask(segment["distance_map"], combined_segmentation == 0)

        return combined_segmentation, [segment["distance_map"] for segment in segments]
    
    # Combine the segmentations chunk-wise to save memory
    for start_idx in range(0, segmentation.GetSize()[-1], chunk_size):
        end_idx = min(start_idx + chunk_size, segmentation.GetSize()[-1])

        # Initialize the combined segmentation array
        combined_array = sitk.GetArrayFromImage(combined_segmentation[..., start_idx:end_idx])

        # Assign non-ambiguous regions directly
        for segment in segments:
            binary_mask_array = sitk.GetArrayFromImage(segment["binary_mask"][..., start_idx:end_idx])
            combined_array += segment["new_label"] * binary_mask_array
            del binary_mask_array  # Release memory
        
        # Identify ambiguous voxels
        ambiguous_mask = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
        ambiguous_mask.CopyInformation(segmentation)
        ambiguous_mask = ambiguous_mask[..., start_idx:end_idx]

        for segment in segments:
            ambiguous_mask = sitk.Add(ambiguous_mask, segment["binary_mask"][..., start_idx:end_idx])
        ambiguous_mask = sitk.Greater(ambiguous_mask, 1)
        ambiguous_array = sitk.GetArrayFromImage(ambiguous_mask)

        # Compute distance maps for ambiguous voxels
        distance_array = np.stack([sitk.GetArrayFromImage(segment["distance_map"][..., start_idx:end_idx]) for segment in segments])
        nearest_labels = np.argmin(distance_array[:, ambiguous_array > 0], axis=0) + 1
        del distance_array  # Release memory

        # Update the combined segmentation
        combined_array[ambiguous_array > 0] = nearest_labels
        del ambiguous_array, nearest_labels  # Release memory

        # Add to the combined segmentation image
        combined_segmentation[..., start_idx:end_idx] = sitk.GetImageFromArray(combined_array.astype(np.uint8))
    
    # Set the distance maps to 0 where the combined segmentation > 0
    for segment in segments:
        segment["distance_map"] = sitk.Mask(segment["distance_map"], combined_segmentation == 0)

    return combined_segmentation, [segment["distance_map"] for segment in segments]


def expand_segmentation(segmentation, distance_maps, expansion_mm, chunk_size=100):

    # Mask the distance maps within the expansion radius
    for idx in range(len(distance_maps)):
        distance_maps[idx] = sitk.Mask(distance_maps[idx], 0 < distance_maps[idx] <= expansion_mm)
    
    # Only update the segmentation where the distance maps are non-zero
    sum_distance_maps = sitk.Image(distance_maps[0].GetSize(), sitk.sitkFloat32)
    sum_distance_maps.CopyInformation(distance_maps[0])
    for idx, distance_map in enumerate(distance_maps):
        sum_distance_maps = sitk.Add(sum_distance_maps, distance_map)

        # set zeros to inf
        distance_maps[idx] = sitk.Mask(distance_map, sitk.Greater(distance_map, 0), np.inf)
    update_mask = sitk.Greater(sum_distance_maps, 0)
    del sum_distance_maps  # Release memory

    # Combine the distance maps chunk-wise to save memory
    expanded_segmentation = sitk.Image(segmentation.GetSize(), sitk.sitkUInt8)
    expanded_segmentation.CopyInformation(segmentation)
    for start_idx in range(0, distance_maps[0].GetSize()[-1], chunk_size):
        end_idx = min(start_idx + chunk_size, distance_maps[0].GetSize()[-1])

        distance_map = np.stack([sitk.GetArrayFromImage(distance_map[..., start_idx:end_idx]) for distance_map in distance_maps])
        nearest_labels = np.argmin(distance_map, axis=0) + 1

        # Update nearest labels where the distance maps are zero with the original segmentation
        nearest_labels = np.where(sitk.GetArrayFromImage(update_mask[..., start_idx:end_idx]), nearest_labels, sitk.GetArrayFromImage(segmentation[..., start_idx:end_idx]))

        # Update the expanded segmentation
        expanded_segmentation[..., start_idx:end_idx] = sitk.GetImageFromArray(nearest_labels.astype(np.uint8))

    return expanded_segmentation

def process_case(base_name, input_folder, output_folder, expansion_mm, chunk_size, target_spacing, crop_margin, hu_min=-200, hu_max=500, disable_hu_windowing=False):
    try:
        input_segmentation_path = os.path.join(input_folder, f"Segmentations_{base_name}_all.seg.nrrd")
        output_segmentation_path = os.path.join(output_folder, f"Segmentations_{base_name}_all_expanded.nii.gz")
        output_image_path = os.path.join(output_folder, f"Scan_{base_name}_TS.nii.gz")
        image_path = os.path.join(input_folder, f"Scan_{base_name}_TS.nii.gz")

        if not os.path.exists(input_segmentation_path):
            return f"Segmentation file for {base_name} not found. Skipping."

        print(f"Processing {base_name}...")

        segmentation = sitk.ReadImage(input_segmentation_path)
        segmentation, distance_maps = combine_segmentation(segmentation, chunk_size=chunk_size)
        segmentation = expand_segmentation(segmentation, distance_maps, expansion_mm, chunk_size=chunk_size)

        image = sitk.ReadImage(image_path)

        # add validation and preprocessing
        validate_image_properties(image, base_name)
        if not disable_hu_windowing:
            image = validate_and_preprocess_image(image, hu_min=hu_min, hu_max=hu_max)
        else:
            print(f"  Info: HU windowing disabled for {base_name}")

        # Flip axes if needed
        flip_axes = [bool(np.sign(dir_image) != np.sign(dir_seg)) for dir_image, dir_seg in zip(
            image.GetDirection()[::4], segmentation.GetDirection()[::4])]
        segmentation = sitk.Flip(segmentation, flip_axes)

        if target_spacing is not None:
            image = resample_to_spacing_reference(image, target_spacing, is_label=False)
            segmentation = resample_to_reference(segmentation, image, is_label=True)
        else:
            segmentation = resample_to_reference(segmentation, image, is_label=True)

        if np.allclose(image.GetOrigin(), segmentation.GetOrigin(), atol=1e-5) and \
           np.allclose(image.GetSpacing(), segmentation.GetSpacing(), atol=1e-5):
            segmentation.CopyInformation(image)

        if crop_margin is not None:
            image, segmentation = crop_to_segmentation_bounds(image, segmentation, margin_slices=crop_margin)
        else:
            print(f"No cropping performed for {base_name}")

        sitk.WriteImage(segmentation, output_segmentation_path)
        sitk.WriteImage(image, output_image_path)

        return f"{base_name} processed successfully."

    except Exception as e:
        return f"Error processing {base_name}: {str(e)}"

def process_folder_parallel(input_folder, output_folder, expansion_mm, chunk_size=10, target_spacing=None, crop_margin=None, max_workers=4, hu_min=-200, hu_max=500, disable_hu_windowing=False):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    base_names = [
        file.replace("Scan_", "").replace("_TS.nii.gz", "")
        for file in os.listdir(input_folder)
        if file.startswith("Scan_") and file.endswith("_TS.nii.gz")
    ]

    print(f"Processing {len(base_names)} cases with {max_workers} workers...")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_case, base_name, input_folder, output_folder,
                expansion_mm, chunk_size, target_spacing, crop_margin, hu_min, hu_max, disable_hu_windowing
            ): base_name for base_name in base_names
        }

        for future in as_completed(futures):
            print(future.result())

def process_folder(input_folder, output_folder, expansion_mm, chunk_size=10, target_spacing=None, crop_margin=None, hu_min=-200, hu_max=500, disable_hu_windowing=False):
    """
    Processes all NIfTI image and segmentation pairs in a folder.

    Steps:
    - Combines multi-component segmentations into label maps
    - Expands the segmentations using distance maps
    - Crops image & segmentation to non-zero bounds (z-axis, with margin)
    - Resamples both to lower resolution (1mm³)
    - Saves both image and segmentation to output_folder
    """
    

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file in os.listdir(input_folder):
        if file.startswith("Scan_") and file.endswith("_TS.nii.gz"):
            # Derive paths
            base_name = file.replace("Scan_", "").replace("_TS.nii.gz", "")
            input_segmentation_path = os.path.join(input_folder, f"Segmentations_{base_name}_all.seg.nrrd")
            output_segmentation_path = os.path.join(output_folder, f"Segmentations_{base_name}_all_expanded.nii.gz")
            output_image_path = os.path.join(output_folder, f"Scan_{base_name}_TS.nii.gz")

            if not os.path.exists(input_segmentation_path):
                print(f"Segmentation file for {base_name} not found. Skipping.")
                continue

            print(f"Processing {base_name}...")

            # read and expand the image
            segmentation = sitk.ReadImage(input_segmentation_path)
            segmentation, distance_maps = combine_segmentation(segmentation, chunk_size=chunk_size)
            segmentation = expand_segmentation(segmentation, distance_maps, expansion_mm, chunk_size=chunk_size)

            # Read the image
            image = sitk.ReadImage(os.path.join(input_folder, file))

            # add validation and preprocessing
            validate_image_properties(image, base_name)
            if not disable_hu_windowing:
                image = validate_and_preprocess_image(image, hu_min=hu_min, hu_max=hu_max)
            else:
                print(f"  Info: HU windowing disabled for {base_name}")

            # Check if we need to flip the segmentation in some direction
            flip_axes = [bool(np.sign(dir_image) != np.sign(dir_seg)) for dir_image, dir_seg in zip(
                image.GetDirection()[::4], segmentation.GetDirection()[::4])]
            segmentation = sitk.Flip(segmentation, flip_axes)

            if target_spacing is not None:
                # Use specified spacing — same for both image & seg
                image = resample_to_spacing_reference(image, target_spacing, is_label=False)
                segmentation = resample_to_reference(segmentation, image, is_label=True)
            else:
                # Just align segmentation to image grid
                segmentation = resample_to_reference(segmentation, image, is_label=True)

            # If differences in origin or spacing are minimal, use the image's origin and spacing
            if np.allclose(image.GetOrigin(), segmentation.GetOrigin(), atol=1e-5) and \
                np.allclose(image.GetSpacing(), segmentation.GetSpacing(), atol=1e-5):
                segmentation.CopyInformation(image)
            else:
                print("Warning: Origin or spacing of image and segmentation differ.")
                
            # Crop both image & segmentation to z-extent of segmentation
            if crop_margin is not None:
                image, segmentation = crop_to_segmentation_bounds(image, segmentation, margin_slices=crop_margin)
            else:
                print(f"No cropping performed for {base_name}")

            # Save the expanded segmentation and image
            sitk.WriteImage(segmentation, output_segmentation_path)
            sitk.WriteImage(image, output_image_path)

def preprocess_input_folder(input_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.nii.gz'):
                # Check if the file is a loose file or in a subdirectory
                if file.startswith("Scan_") and file.endswith("_TS.nii.gz"):
                    # Already in the correct format, skip renaming
                    continue
                
                # Extract the case identifier, ignoring any additional suffixes like '_1mm'
                case_id = file.split('_')[0]
                src = os.path.join(root, file)
                dst = os.path.join(input_folder, f"Scan_{case_id}_TS.nii.gz")
                shutil.move(src, dst)

            elif file.endswith('.seg.nrrd'):
                # Extract the case identifier, ignoring any additional suffixes like '_all'
                case_id = file.split('_')[1].split('.')[0]
                src = os.path.join(root, file)
                dst = os.path.join(input_folder, f"Segmentations_{case_id}_all.seg.nrrd")
                shutil.move(src, dst)
        
    for root, dirs, _ in os.walk(input_folder, topdown=False):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            if not os.listdir(dir_path):
                os.rmdir(dir_path)
    

def debug_metadata(image, segmentation, step):
    print(f"Step: {step}")
    print(f"Image origin: {image.GetOrigin()}")
    print(f"Segmentation origin: {segmentation.GetOrigin()}")
    print(f"Image spacing: {image.GetSpacing()}")
    print(f"Segmentation spacing: {segmentation.GetSpacing()}")
    print(f"Image direction: {image.GetDirection()}")
    print(f"Segmentation direction: {segmentation.GetDirection()}")
    print("-" * 50)

def test_naming_convention(input_folder):
    """
    Tests the naming convention for segmentations and images in a folder.

    Parameters:
    - input_folder: Path to the folder containing NIfTI images and segmentations.

    Returns:
    - bool: True if all naming conventions are correct, False otherwise.
    """
    incorrect_files = []
    all_correct = True
    for file in os.listdir(input_folder):
        if file.startswith("Scan_") and file.endswith("_TS.nii.gz"):
            # Derive the base name
            base_name = file.replace("Scan_", "").replace("_TS.nii.gz", "")
            expected_segmentation_name = f"Segmentations_{base_name}_all.seg.nrrd"

            if not os.path.exists(os.path.join(input_folder, expected_segmentation_name)):
                print(f"Naming inconsistency: Expected {expected_segmentation_name} to exist for {file}")
                all_correct = False

        elif file.startswith("Segmentations_") and file.endswith(".seg.nrrd"):
            # Derive the base name
            base_name = file.replace("Segmentations_", "").replace("_all.seg.nrrd", "")
            expected_image_name = f"Scan_{base_name}_TS.nii.gz"

            if not os.path.exists(os.path.join(input_folder, expected_image_name)):
                print(f"Naming inconsistency: Expected {expected_image_name} to exist for {file}")
                all_correct = False
        
        else:
            print(f"Warning: Unexpected file {file} in the input folder.")
            all_correct = False

    return all_correct, incorrect_files

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Expand rPCI segmentations and optionally crop/resample the paired CT scans."
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
        default=None,
        help="Optional margin in voxels/slices for cropping around segmentation bounds.",
    )
    parser.add_argument("--hu-min", type=int, default=-200, help="Minimum HU value for windowing.")
    parser.add_argument("--hu-max", type=int, default=500, help="Maximum HU value for windowing.")
    parser.add_argument(
        "--disable-hu-windowing",
        action="store_true",
        help="Disable HU windowing before writing processed scans.",
    )
    parser.add_argument(
        "--parallel-processing",
        action="store_true",
        help="Process cases in parallel.",
    )
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

    num_images = len([
        file for file in os.listdir(args.input_folder)
        if file.startswith("Scan_") and file.endswith("_TS.nii.gz")
    ])
    num_segmentations = len([
        file for file in os.listdir(args.input_folder)
        if file.startswith("Segmentations_") and file.endswith(".seg.nrrd")
    ])

    if num_images != num_segmentations:
        print(
            f"Warning: Number of images ({num_images}) and segmentations "
            f"({num_segmentations}) do not match."
        )
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
            disable_hu_windowing=args.disable_hu_windowing,
        )

    print("Processing complete.")


if __name__ == "__main__":
    main()
