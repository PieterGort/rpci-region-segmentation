import SimpleITK as sitk
import numpy as np
import os
# from totalsegmentator.python_api import totalsegmentator

def crop_to_liver_bounds(image, segmentation, margin_slices=15, liver_label=5):
    """
    Crop original image to start slightly above the superior-most liver slice (label==liver_label),
    keeping the rest of the scan (inferior part). This removes most of the thoracic region.

    Parameters:
    - image: SimpleITK.Image (original scan)
    - segmentation: SimpleITK.Image (multi-label segmentation; liver at `liver_label`)
    - margin_slices: int, how many slices above the liver to keep
    - liver_label: int, label index for liver (default=5)

    Returns:
    - cropped_image: SimpleITK.Image
    """
    try:
        seg_array = sitk.GetArrayFromImage(segmentation)  # [z, y, x]
        liver_mask = (seg_array == liver_label)

        if not np.any(liver_mask):
            print("⚠️ Warning: liver label not found in segmentation. Returning original image.")
            return image

        # Find first z-slice where liver appears (superior-most along array order)
        liver_slices = np.any(liver_mask, axis=(1, 2))  # [z]

        # bottom = 0, top = last slice, cro
        z_liver_top = int(np.where(liver_slices)[0][-1])

        # Start crop a bit above liver, so z_liver_top + margin_slices
        total_slices = image.GetSize()[2]
        z_max_cropped = min(total_slices - 1, z_liver_top + margin_slices)
        z_min_cropped = 0

        # keep everything under z_max_cropped
        size = list(image.GetSize())
        index = [0, 0, z_min_cropped]
        size[2] = z_max_cropped - z_min_cropped + 1
        cropped_image = sitk.RegionOfInterest(image, size=list(map(int, size)), index=list(map(int, index)))
        return cropped_image

    except Exception as e:
        print(f"❌ Error during liver cropping: {e}")
        print("Returning original image.")
        return image
    
def crop_liver_from_dirs(original_scans_dir, segs_dir, output_dir, margin_slices=15, liver_label=5):
    os.makedirs(output_dir, exist_ok=True)
    seg_files = sorted([f for f in os.listdir(segs_dir) if f.lower().endswith(".nii.gz")])

    for f in seg_files:
        img_path = os.path.join(original_scans_dir, f)
        seg_path = os.path.join(segs_dir, f)
        if not os.path.exists(img_path):
            print(f"Skipping {f}: original scan not found at {img_path}")
            continue
        try:
            img = sitk.ReadImage(img_path)
            seg = sitk.ReadImage(seg_path)
            cropped = crop_to_liver_bounds(img, seg, margin_slices=margin_slices, liver_label=liver_label)
            out_path = os.path.join(output_dir, f)
            sitk.WriteImage(cropped, out_path)
            print(f"Saved cropped scan: {out_path}")
        except Exception as e:
            print(f"Failed cropping {f}: {e}")

def crop_to_Z_segmentation_bounds(image, segmentation, margin_slices=5):
    """
    Crops both image and segmentation based on the z-axis bounds of non-zero segmentation data.
    Supports both scalar and multi-component segmentations.

    Parameters:
    - image: SimpleITK.Image (original scan)
    - segmentation: SimpleITK.Image (raw or expanded segmentation)
    - margin_slices: int, number of extra slices to include before/after (default = 5)

    Returns:
    - cropped_image: SimpleITK.Image
    - cropped_segmentation: SimpleITK.Image
    """
    try:
        # Step 1: Convert to scalar mask if needed
        if segmentation.GetNumberOfComponentsPerPixel() > 1:
            print(f"Segmentations has {segmentation.GetNumberOfComponentsPerPixel()} components. Converting to scalar mask...")
            seg_array = None
            for i in range(segmentation.GetNumberOfComponentsPerPixel()):
                comp = sitk.VectorIndexSelectionCast(segmentation, i)
                comp_array = sitk.GetArrayFromImage(comp) != 0
                seg_array = comp_array if seg_array is None else np.logical_or(seg_array, comp_array)
        else:
            seg_array = sitk.GetArrayFromImage(segmentation) != 0

        # Step 2: Find z-range of segmentation
        non_zero_slices = np.any(seg_array, axis=(1, 2))
        if not np.any(non_zero_slices):
            print("⚠️ Warning: Segmentation contains no non-zero data. Skipping cropping.")
            return image, segmentation

        z_min = int(np.where(non_zero_slices)[0][0])
        z_max = int(np.where(non_zero_slices)[0][-1])

        # Step 3: Apply margin and clamp within bounds
        total_slices = segmentation.GetSize()[2]
        z_min_cropped = max(0, z_min - margin_slices)
        z_max_cropped = min(total_slices - 1, z_max + margin_slices)

        # Step 4: Prepare cropping parameters
        size = list(segmentation.GetSize())
        index = [0, 0, z_min_cropped]
        size[2] = z_max_cropped - z_min_cropped + 1  # inclusive range

        # Step 5: Ensure size + index do not exceed bounds in any dimension
        for dim in range(3):
            if index[dim] + size[dim] > segmentation.GetSize()[dim]:
                size[dim] = segmentation.GetSize()[dim] - index[dim]

        # Step 6: Crop safely
        cropped_segmentation = sitk.RegionOfInterest(segmentation, size=list(map(int, size)), index=list(map(int, index)))
        cropped_image = sitk.RegionOfInterest(image, size=list(map(int, size)), index=list(map(int, index)))

        return cropped_image, cropped_segmentation

    except Exception as e:
        print(f"❌ Error during cropping: {e}")
        print("Returning original image and segmentation.")
        return image, segmentation
    
def crop_to_segmentation_bounds(image, segmentation, margin_slices=10):
    """
    Crops both image and segmentation based on the bounds of non-zero segmentation data in all three dimensions.

    Parameters:
    - image: SimpleITK.Image (original scan)
    - segmentation: SimpleITK.Image (raw or expanded segmentation)
    - margin_slices: int, number of extra slices to include before/after (default = 5)

    Returns:
    - cropped_image: SimpleITK.Image
    - cropped_segmentation: SimpleITK.Image
    """
    try:
        # Step 1: Convert to scalar mask if needed
        if segmentation.GetNumberOfComponentsPerPixel() > 1:
            seg_array = None
            for i in range(segmentation.GetNumberOfComponentsPerPixel()):
                comp = sitk.VectorIndexSelectionCast(segmentation, i)
                comp_array = sitk.GetArrayFromImage(comp) != 0
                seg_array = comp_array if seg_array is None else np.logical_or(seg_array, comp_array)
        else:
            seg_array = sitk.GetArrayFromImage(segmentation) != 0

        # Step 2: Find bounds of segmentation
        non_zero_indices = np.argwhere(seg_array)
        if non_zero_indices.size == 0:
            print("⚠️ Warning: Segmentation contains no non-zero data. Skipping cropping.")
            return image, segmentation
        
        z_min, y_min, x_min = np.maximum(0, non_zero_indices.min(axis=0) - margin_slices)
        z_max, y_max, x_max = np.minimum(np.array(seg_array.shape) - 1, non_zero_indices.max(axis=0) + margin_slices)

        index = [int(x_min), int(y_min), int(z_min)]
        size = [int(x_max - x_min + 1), int(y_max - y_min + 1), int(z_max - z_min + 1)]

        # Ensure size + index do not exceed bounds in any dimension
        for dim in range(3):
            if index[dim] + size[dim] > segmentation.GetSize()[dim]:
                size[dim] = segmentation.GetSize()[dim] - index[dim]
        
        # Crop safely
        cropped_segmentation = sitk.RegionOfInterest(segmentation, size=list(map(int, size)), index=list(map(int, index)))
        cropped_image = sitk.RegionOfInterest(image, size=list(map(int, size)), index=list(map(int, index)))

        return cropped_image, cropped_segmentation

    except Exception as e:
        print(f"❌ Error during cropping: {e}")
        print("Returning original image and segmentation.")
        return image, segmentation
    
def resample_to_reference(moving_image, reference_image, is_label=False):
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(moving_image)

def resample_to_spacing_reference(image, target_spacing, is_label=False):
    """
    Resamples image to a new voxel spacing. Uses the same origin/direction as input image.
    """
    original_size = image.GetSize()
    original_spacing = image.GetSpacing()
    original_origin = image.GetOrigin()
    original_direction = image.GetDirection()

    physical_size = [s * sp for s, sp in zip(original_size, original_spacing)]
    new_size = [int(np.round(p / ts)) for p, ts in zip(physical_size, target_spacing)]

    resample = sitk.ResampleImageFilter()
    resample.SetSize(new_size)
    resample.SetOutputSpacing(target_spacing)
    resample.SetOutputOrigin(original_origin)
    resample.SetOutputDirection(original_direction)
    resample.SetInterpolator(sitk.sitkNearestNeighbor if is_label else sitk.sitkBSpline)
    resample.SetDefaultPixelValue(0)

    return resample.Execute(image)

def expand_segmentation_to_original_space(cropped_seg, original_img, crop_margin=15):
    """
    Pads cropped_segmentation to match the size of original_img,
    assuming spacing, direction, and origin are already identical.

    Parameters:
    - cropped_seg: SimpleITK.Image (cropped segmentation)
    - original_img: SimpleITK.Image (original image, full size)

    Returns:
    - full_seg: SimpleITK.Image (segmentation matching original size)
    """

    # Create empty (zero) segmentation with same size, spacing, origin, direction as original image
    full_seg = sitk.Image(original_img.GetSize(), cropped_seg.GetPixelID())
    full_seg.SetSpacing(original_img.GetSpacing())
    full_seg.SetOrigin(original_img.GetOrigin())
    full_seg.SetDirection(original_img.GetDirection())

    # Calculate where to paste: use cropped_seg origin relative to original origin, divided by spacing
    start_index = [
        int(round((cropped_seg.GetOrigin()[i] - original_img.GetOrigin()[i]) / original_img.GetSpacing()[i]))
        for i in range(3)
    ]

    # Paste cropped_seg into full_seg at correct index
    full_seg = sitk.Paste(
        destinationImage=full_seg,
        sourceImage=cropped_seg,
        destinationIndex=start_index,
        sourceSize=cropped_seg.GetSize(),
        sourceIndex=[0, 0, 0]
    )

    return full_seg

def resample_segmentation(original_img_path, seg_img_path, output_path):
    img_original = sitk.ReadImage(original_img_path)
    seg = sitk.ReadImage(seg_img_path)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(img_original)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)

    resampled_val_pred = resampler.Execute(seg)
    sitk.WriteImage(resampled_val_pred, output_path)

    print(f"Saved resampled segmentation to: {output_path}")

def resample_prediction_to_original_space(original_img_path, cropped_img_path, prediction_path, output_path):
    """
    Resamples a prediction from cropped image space back to original image space.
    Handles the coordinate system mapping between MONAI preprocessing and SimpleITK.
    
    Parameters:
    - original_img_path: str, path to original full-size image
    - cropped_img_path: str, path to cropped image used for inference
    - prediction_path: str, path to prediction from model (same space as cropped image)
    - output_path: str, path to save the correctly positioned prediction
    """
    try:
        # Load images
        img_original = sitk.ReadImage(original_img_path)
        img_cropped = sitk.ReadImage(cropped_img_path)
        prediction = sitk.ReadImage(prediction_path)
        
        print(f"Original image origin: {img_original.GetOrigin()}")
        print(f"Cropped image origin: {img_cropped.GetOrigin()}")
        print(f"Prediction origin: {prediction.GetOrigin()}")
        
        # Calculate the origin shift from crop to prediction
        origin_shift_x = prediction.GetOrigin()[0] - img_original.GetOrigin()[0]
        origin_shift_y = prediction.GetOrigin()[1] - img_original.GetOrigin()[1]
        origin_shift_z = prediction.GetOrigin()[2] - img_original.GetOrigin()[2]
        
        print(f"Origin shifts during preprocessing:")
        print(f"X shift: {origin_shift_x:.2f} mm")
        print(f"Y shift: {origin_shift_y:.2f} mm")
        print(f"Z shift: {origin_shift_z:.2f} mm")
        
        prediction.SetOrigin((img_original.GetOrigin()[0] + origin_shift_x,
                             img_original.GetOrigin()[1] + origin_shift_y,
                             img_original.GetOrigin()[2] + origin_shift_z))
        
        # Resample prediction back to original image space
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(img_original)
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
        resampler.SetTransform(sitk.Transform())  # Identity transform
        resampler.SetDefaultPixelValue(0)
        
        resampled_prediction = resampler.Execute(prediction)
        
        # Save the correctly positioned prediction
        sitk.WriteImage(resampled_prediction, output_path)
        
        print(f"Successfully resampled prediction to original space: {output_path}")
        
        # Verification: check prediction bounds
        pred_array = sitk.GetArrayFromImage(resampled_prediction)
        non_zero_indices = np.where(pred_array > 0)
        
        if len(non_zero_indices[0]) > 0:
            z_min, z_max = non_zero_indices[0].min(), non_zero_indices[0].max()
            y_min, y_max = non_zero_indices[1].min(), non_zero_indices[1].max()
            x_min, x_max = non_zero_indices[2].min(), non_zero_indices[2].max()
            
            print(f"Prediction bounds in original image:")
            print(f"Z range: {z_min} to {z_max} (slices)")
            print(f"Y range: {y_min} to {y_max} (pixels)")
            print(f"X range: {x_min} to {x_max} (pixels)")
            print(f"Prediction volume: {np.sum(pred_array > 0)} voxels")
        else:
            print("WARNING: No prediction found in resampled image!")
            
    except Exception as e:
        print(f"❌ Error during prediction resampling: {e}")
        raise e

def batch_resample_predictions(original_img_dir, cropped_img_dir, prediction_dir, output_dir):
    """
    Batch process multiple predictions from cropped space back to original space.
    
    Parameters:
    - original_img_dir: str, directory containing original full-size images
    - cropped_img_dir: str, directory containing cropped images used for inference
    - prediction_dir: str, directory containing predictions from model
    - output_dir: str, directory to save correctly positioned predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    
    prediction_files = sorted([f for f in os.listdir(prediction_dir) if f.lower().endswith(".nii.gz")])
    
    for pred_file in prediction_files:
        try:
            # Extract case ID from prediction filename
            case_id = pred_file.split('.')[0]
            
            # Construct file paths
            original_img_path = os.path.join(original_img_dir, f"{case_id}_AX_pvp_TS.nii.gz")
            cropped_img_path = os.path.join(cropped_img_dir, f"{case_id}_0000.nii.gz")
            prediction_path = os.path.join(prediction_dir, pred_file)
            output_path = os.path.join(output_dir, f"{case_id}_resampled.nii.gz")
            
            # Check if required files exist
            if not os.path.exists(original_img_path):
                print(f"Skipping {pred_file}: original image not found at {original_img_path}")
                continue
            if not os.path.exists(cropped_img_path):
                print(f"Skipping {pred_file}: cropped image not found at {cropped_img_path}")
                continue
                
            print(f"\nProcessing {pred_file}...")
            resample_prediction_to_original_space(original_img_path, cropped_img_path, prediction_path, output_path)
            
        except Exception as e:
            print(f"Failed processing {pred_file}: {e}")

def batch_resample(pairs_file):
    with open(pairs_file, 'r') as f:
        for line in f:
            original_img_path, seg_img_path, output_path = line.strip().split(',')
            resample_segmentation(original_img_path, seg_img_path, output_path)

def manual_crop_image(image, x_min, x_max, y_min, y_max, z_min, z_max):
    image_array = sitk.GetArrayFromImage(image)
    image_array = image_array[z_min:z_max, y_min:y_max, x_min:x_max]
    image_cropped = sitk.GetImageFromArray(image_array)
    image_cropped.SetSpacing(image.GetSpacing())
    image_cropped.SetOrigin(image.GetOrigin())
    image_cropped.SetDirection(image.GetDirection())
    return image_cropped

def crop_folder_to_segmentation_bounds(
    input_dir,
    output_dir,
    margin_slices=10,
    image_suffix="_TS.nii.gz",
    segmentation_suffix="_all_expanded.nii.gz",
):
    """
    Crop Scan_* images and matching Segmentations_* masks to segmentation bounds.

    The expected input names are:
    - Scan_{case_id}{image_suffix}
    - Segmentations_{case_id}{segmentation_suffix}
    """
    os.makedirs(output_dir, exist_ok=True)

    image_files = sorted(
        file for file in os.listdir(input_dir)
        if file.startswith("Scan_") and file.endswith(image_suffix)
    )

    if not image_files:
        print(f"No Scan_*{image_suffix} files found in {input_dir}")
        return

    for image_file in image_files:
        case_id = image_file.replace("Scan_", "").replace(image_suffix, "")
        seg_file = f"Segmentations_{case_id}{segmentation_suffix}"
        image_path = os.path.join(input_dir, image_file)
        seg_path = os.path.join(input_dir, seg_file)

        if not os.path.exists(seg_path):
            print(f"Skipping {case_id}: segmentation not found at {seg_path}")
            continue

        image = sitk.ReadImage(image_path)
        segmentation = sitk.ReadImage(seg_path)
        cropped_image, cropped_segmentation = crop_to_segmentation_bounds(
            image, segmentation, margin_slices=margin_slices
        )

        sitk.WriteImage(cropped_image, os.path.join(output_dir, image_file))
        sitk.WriteImage(cropped_segmentation, os.path.join(output_dir, seg_file))
        print(f"Cropped {case_id}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Crop CT images and matching segmentations to non-zero segmentation bounds."
    )
    parser.add_argument("--input-dir", required=True, help="Folder with Scan_* and Segmentations_* files.")
    parser.add_argument("--output-dir", required=True, help="Folder for cropped images and segmentations.")
    parser.add_argument(
        "--margin-slices",
        type=int,
        default=10,
        help="Number of slices/voxels to keep around the segmentation bounds.",
    )
    parser.add_argument(
        "--image-suffix",
        default="_TS.nii.gz",
        help="Suffix after Scan_{case_id}.",
    )
    parser.add_argument(
        "--segmentation-suffix",
        default="_all_expanded.nii.gz",
        help="Suffix after Segmentations_{case_id}.",
    )
    args = parser.parse_args()

    crop_folder_to_segmentation_bounds(
        args.input_dir,
        args.output_dir,
        margin_slices=args.margin_slices,
        image_suffix=args.image_suffix,
        segmentation_suffix=args.segmentation_suffix,
    )


if __name__ == "__main__":
    main()

    




