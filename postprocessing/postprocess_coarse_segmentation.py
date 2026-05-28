import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

try:
    from .config import (
        COARSE_REGION_GROUPS,
        COARSE_SPLIT_STRUCTURES,
        COMBINED_REGION_MAPPING,
        PCI_REGION_MAPPING,
        REGION9_12_HALF_CORONAL_ORDER,
        REGION9_12_CORONAL_ORDER,
        TOTALSEG_STRUCTURES,
    )
except ImportError:
    from config import (
        COARSE_REGION_GROUPS,
        COARSE_SPLIT_STRUCTURES,
        COMBINED_REGION_MAPPING,
        PCI_REGION_MAPPING,
        REGION9_12_HALF_CORONAL_ORDER,
        REGION9_12_CORONAL_ORDER,
        TOTALSEG_STRUCTURES,
    )


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def strip_nii_suffix(path: Path) -> str:
    """Return the filename stem while treating .nii.gz as one suffix."""
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def as_label_values(value: int | list[int]) -> list[int]:
    return value if isinstance(value, list) else [value]


def same_geometry(image: sitk.Image, reference_image: sitk.Image, tolerance: float = 1e-5) -> bool:
    return (
        image.GetSize() == reference_image.GetSize()
        and np.allclose(image.GetSpacing(), reference_image.GetSpacing(), atol=tolerance)
        and np.allclose(image.GetOrigin(), reference_image.GetOrigin(), atol=tolerance)
        and np.allclose(image.GetDirection(), reference_image.GetDirection(), atol=tolerance)
    )


def resample_mask_to_reference(mask_image: sitk.Image, reference_image: sitk.Image) -> sitk.Image:
    mask_image = sitk.Cast(sitk.Greater(mask_image, 0), sitk.sitkUInt8)
    if same_geometry(mask_image, reference_image):
        return mask_image

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(sitk.sitkUInt8)
    return resampler.Execute(mask_image)


def find_structure_path(totalseg_folder: Path, structure_name: str) -> Path:
    for suffix in (".nii.gz", ".nii"):
        path = totalseg_folder / f"{structure_name}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find TotalSegmentator mask for {structure_name} in {totalseg_folder}")


def has_any_totalseg_structure(totalseg_folder: Path) -> bool:
    return any(
        any((totalseg_folder / f"{structure}{suffix}").exists() for suffix in (".nii.gz", ".nii"))
        for structure in TOTALSEG_STRUCTURES
    )


def load_totalseg_structure_mask(
    totalseg_folder: Path,
    structure_name: str,
    reference_image: sitk.Image,
) -> np.ndarray:
    structure_path = find_structure_path(totalseg_folder, structure_name)
    structure_image = sitk.ReadImage(str(structure_path))
    structure_image = resample_mask_to_reference(structure_image, reference_image)
    return sitk.GetArrayFromImage(structure_image).astype(bool)


def array_coordinates_to_physical_points(array_coordinates: tuple[np.ndarray, ...], image: sitk.Image) -> np.ndarray:
    """Convert numpy array coordinates (z, y, x) to physical points (x, y, z)."""
    direction = np.asarray(image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.asarray(image.GetSpacing(), dtype=float)
    origin = np.asarray(image.GetOrigin(), dtype=float)

    indices = np.column_stack((array_coordinates[2], array_coordinates[1], array_coordinates[0])).astype(float)
    return origin + ((indices * spacing) @ direction.T)


def physical_superior_coordinates(array_coordinates: tuple[np.ndarray, ...], image: sitk.Image) -> np.ndarray:
    """Convert numpy array coordinates (z, y, x) to physical superior-inferior coordinates."""
    return array_coordinates_to_physical_points(array_coordinates, image)[:, 2]


def find_superior_extent(mask: np.ndarray, reference_image: sitk.Image, structure_name: str) -> float:
    coordinates = np.where(mask)
    if len(coordinates[0]) == 0:
        raise ValueError(f"TotalSegmentator structure {structure_name} is empty; cannot infer cut plane")
    return float(np.max(physical_superior_coordinates(coordinates, reference_image)))


def split_region_by_hip_plane(
    output_array: np.ndarray,
    coarse_array: np.ndarray,
    coarse_label_values: list[int],
    hip_mask: np.ndarray,
    reference_image: sitk.Image,
    inferior_label: int,
    superior_label: int,
    region_name: str,
    structure_name: str,
) -> None:
    region_mask = np.isin(coarse_array, coarse_label_values)
    if not np.any(region_mask):
        return

    cut_plane = find_superior_extent(hip_mask, reference_image, structure_name)
    region_coordinates = np.where(region_mask)
    superior_coordinates = physical_superior_coordinates(region_coordinates, reference_image)
    superior_side = superior_coordinates > cut_plane

    output_array[region_coordinates] = inferior_label
    output_array[
        region_coordinates[0][superior_side],
        region_coordinates[1][superior_side],
        region_coordinates[2][superior_side],
    ] = superior_label

    logger.info(
        "%s split at superior coordinate %.2f using %s",
        region_name,
        cut_plane,
        structure_name,
    )


def select_balanced_halfplane_angle(
    angles: np.ndarray,
    target_fraction: float = 0.5,
    volume_tolerance: float = 0.01,
) -> tuple[float, float]:
    normalized_angles = angles % (2 * np.pi)
    sorted_angles = np.sort(normalized_angles)
    sample_count = sorted_angles.size
    if sample_count == 0:
        return 0.0, 0.0

    extended = np.concatenate((sorted_angles, sorted_angles + 2 * np.pi))
    half_turn_limits = sorted_angles + np.pi
    half_turn_end_indices = np.searchsorted(extended, half_turn_limits, side="left")
    positive_counts = half_turn_end_indices - np.arange(sample_count)
    positive_fractions = positive_counts / sample_count
    negative_fractions = 1.0 - positive_fractions

    positive_errors = np.abs(positive_fractions - target_fraction)
    negative_errors = np.abs(negative_fractions - target_fraction)
    use_negative_side = negative_errors < positive_errors
    best_errors = np.where(use_negative_side, negative_errors, positive_errors)
    min_error = float(np.min(best_errors))
    candidate_indices = np.flatnonzero(best_errors <= min_error + volume_tolerance)

    best_index = int(candidate_indices[0])
    if use_negative_side[best_index]:
        angle = sorted_angles[best_index] + np.pi
        achieved_fraction = float(negative_fractions[best_index])
    else:
        angle = sorted_angles[best_index]
        achieved_fraction = float(positive_fractions[best_index])

    return float(angle % (2 * np.pi)), achieved_fraction


def optimize_balanced_plane_split(
    indices: np.ndarray,
    region_angles: np.ndarray,
    target_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if indices.size == 0:
        return indices, indices, {
            "angle_radians": 0.0,
            "angle_degrees": 0.0,
            "achieved_fraction": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "status": "fallback_empty",
        }

    if indices.size == 1:
        return indices, np.empty((0,), dtype=indices.dtype), {
            "angle_radians": 0.0,
            "angle_degrees": 0.0,
            "achieved_fraction": 1.0,
            "positive_count": 1,
            "negative_count": 0,
            "status": "fallback_single_voxel",
        }

    subset_angles = region_angles[indices]
    split_angle, achieved_fraction = select_balanced_halfplane_angle(
        subset_angles,
        target_fraction=target_fraction,
    )
    positive_side = ((subset_angles - split_angle) % (2 * np.pi)) < np.pi
    positive_indices = indices[positive_side]
    negative_indices = indices[~positive_side]

    status = "optimized"
    if positive_indices.size == 0 or negative_indices.size == 0:
        order = np.argsort(subset_angles)
        positive_order, negative_order = np.array_split(order, 2)
        positive_indices = indices[positive_order]
        negative_indices = indices[negative_order]
        achieved_fraction = float(positive_indices.size / indices.size)
        status = "fallback_equal_index_split"

    return positive_indices, negative_indices, {
        "angle_radians": float(split_angle),
        "angle_degrees": angle_radians_to_degrees(split_angle),
        "achieved_fraction": achieved_fraction,
        "positive_count": int(positive_indices.size),
        "negative_count": int(negative_indices.size),
        "status": status,
    }


def order_group_ids_by_coronal_angle(group_indices: list[np.ndarray], region_angles: np.ndarray) -> list[int]:
    non_empty_group_ids: list[int] = []
    mean_angles: list[float] = []
    for group_id, indices in enumerate(group_indices):
        if indices.size == 0:
            continue
        group_angles = region_angles[indices]
        circular_mean = float(np.arctan2(np.mean(np.sin(group_angles)), np.mean(np.cos(group_angles))) % (2 * np.pi))
        non_empty_group_ids.append(group_id)
        mean_angles.append(circular_mean)

    if len(non_empty_group_ids) <= 1:
        ordered_non_empty = non_empty_group_ids
    else:
        unwrapped, _, _ = unwrap_angles_around_largest_gap(np.asarray(mean_angles))
        ordered_non_empty = [non_empty_group_ids[index] for index in np.argsort(unwrapped)]

    empty_group_ids = [group_id for group_id, indices in enumerate(group_indices) if indices.size == 0]
    return ordered_non_empty + empty_group_ids


def angle_radians_to_degrees(angle: float) -> float:
    return float(np.degrees(((angle + np.pi) % (2 * np.pi)) - np.pi))


def signed_angle_difference_degrees(angle: float, reference_angle: float) -> float:
    """Directed angle difference angle-reference in degrees, normalized to [-180, 180)."""
    return float(np.degrees(((angle - reference_angle + np.pi) % (2 * np.pi)) - np.pi))


def undirected_angle_difference_degrees(angle: float, reference_angle: float) -> float:
    relative_angle = abs(signed_angle_difference_degrees(angle, reference_angle))
    return float(min(relative_angle, 180.0 - relative_angle))


def unwrap_angles_around_largest_gap(angles: np.ndarray) -> tuple[np.ndarray, float, float]:
    normalized_angles = angles % (2 * np.pi)
    sorted_angles = np.sort(normalized_angles)
    wrapped_next_angles = np.concatenate((sorted_angles[1:], sorted_angles[:1] + 2 * np.pi))
    gaps = wrapped_next_angles - sorted_angles
    largest_gap_index = int(np.argmax(gaps))
    start_angle = float(sorted_angles[(largest_gap_index + 1) % sorted_angles.size])
    largest_gap = float(gaps[largest_gap_index])
    return (normalized_angles - start_angle) % (2 * np.pi), start_angle, largest_gap


def cutline_angle_to_direction_xz(cutline_angle: float) -> np.ndarray:
    """Convert a coronal cutline angle to its x/z direction vector."""
    return np.asarray([np.cos(cutline_angle), np.sin(cutline_angle)], dtype=float)


def cutline_angle_to_normal_xz(cutline_angle: float) -> np.ndarray:
    """Return a side-test normal for a coronal cutline angle."""
    return np.asarray([-np.sin(cutline_angle), np.cos(cutline_angle)], dtype=float)


def compute_coronal_angles_from_anchor(points: np.ndarray, anchor: np.ndarray) -> np.ndarray:
    """Polar angles in the coronal x/z plane from a physical anchor point."""
    return np.arctan2(
        points[:, 2] - anchor[2],
        points[:, 0] - anchor[0],
    ) % (2 * np.pi)


def infer_coronal_voxel_half_diagonal(reference_image: sitk.Image) -> float:
    """Approximate how far a voxel footprint can extend from its center in coronal x/z."""
    spacing = np.asarray(reference_image.GetSpacing(), dtype=float)
    return float(0.5 * np.hypot(spacing[0], spacing[2]))


def infer_first_cut_anchor_from_duodenum(
    duodenum_mask: np.ndarray,
    reference_image: sitk.Image,
) -> tuple[np.ndarray, dict]:
    """Find the first-cut anchor from the anterior-most duodenum slice and its inferior-most voxel."""
    duodenum_coordinates = np.where(duodenum_mask)
    if len(duodenum_coordinates[0]) == 0:
        raise ValueError("TotalSegmentator duodenum mask is empty; cannot infer first cut anchor")

    duodenum_points = array_coordinates_to_physical_points(duodenum_coordinates, reference_image)
    y_indices = duodenum_coordinates[1]
    direction = np.asarray(reference_image.GetDirection(), dtype=float).reshape(3, 3)
    spacing = np.asarray(reference_image.GetSpacing(), dtype=float)
    ap_step = float((direction[:, 1] * spacing[1])[1])

    if abs(ap_step) > 1e-6:
        anterior_is_lower_index = ap_step > 0
        anterior_slice_index = int(np.min(y_indices) if anterior_is_lower_index else np.max(y_indices))
        anterior_selection_strategy = "extreme_y_index_using_ap_direction"
    else:
        anterior_slice_index = int(y_indices[int(np.argmin(duodenum_points[:, 1]))])
        anterior_selection_strategy = "fallback_min_physical_y"

    anterior_slice_mask = y_indices == anterior_slice_index
    anterior_slice_points = duodenum_points[anterior_slice_mask]
    # Physical z is used as superior-inferior after array (z,y,x) -> physical (x,y,z) conversion.
    inferior_offset = int(np.argmin(anterior_slice_points[:, 2]))
    first_cut_anchor = np.asarray(anterior_slice_points[inferior_offset], dtype=float)
    metadata = {
        "first_cut_anchor_strategy": "anterior_most_duodenum_slice_then_inferior_most_voxel",
        "first_cut_anterior_slice_index": anterior_slice_index,
        "first_cut_ap_step_mm": ap_step,
        "first_cut_anterior_selection_strategy": anterior_selection_strategy,
        "duodenum_voxel_count": int(duodenum_points.shape[0]),
        "anterior_slice_voxel_count": int(anterior_slice_points.shape[0]),
    }
    return first_cut_anchor, metadata


def infer_first_cutline_region_projection_range(
    region_points: np.ndarray,
    first_cut_anchor: np.ndarray,
    first_split_angle: float,
    cutline_intersection_tolerance_mm: float,
) -> tuple[tuple[float, float], dict]:
    """Find the segment where the first cutline intersects the region9_12 coronal footprint."""
    line_direction_xz = cutline_angle_to_direction_xz(first_split_angle)
    normal_direction_xz = cutline_angle_to_normal_xz(first_split_angle)
    offsets_xz = np.column_stack(
        (
            region_points[:, 0] - first_cut_anchor[0],
            region_points[:, 2] - first_cut_anchor[2],
        )
    )
    projections = offsets_xz @ line_direction_xz
    normal_distances = offsets_xz @ normal_direction_xz

    cutline_intersection_mask = np.abs(normal_distances) <= cutline_intersection_tolerance_mm
    if np.any(cutline_intersection_mask):
        candidate_indices = np.flatnonzero(cutline_intersection_mask)
        region_constraint_status = "cutline_intersects_region_footprint"
    else:
        closest_count = min(max(1, int(np.ceil(region_points.shape[0] * 0.01))), 500)
        candidate_indices = np.argpartition(np.abs(normal_distances), closest_count - 1)[:closest_count]
        region_constraint_status = "fallback_closest_region_footprint_to_cutline"

    candidate_projections = projections[candidate_indices]
    projection_range = (float(np.min(candidate_projections)), float(np.max(candidate_projections)))
    metadata = {
        "first_cutline_region_constraint_status": region_constraint_status,
        "first_cutline_region_candidate_count": int(candidate_indices.size),
        "first_cutline_region_projection_range_mm": [projection_range[0], projection_range[1]],
        "first_cutline_intersection_tolerance_mm": float(cutline_intersection_tolerance_mm),
    }
    return projection_range, metadata


def infer_second_cutline_orientation_constraint(
    label_pair: list[int],
    first_cutline_angle: float,
) -> dict:
    """Anatomical direction prior for the two 25/25 cutlines."""
    label_set = set(label_pair)
    first_angle_degrees = angle_radians_to_degrees(first_cutline_angle)

    if label_set == {10, 11}:  # R9/R10: rotate counter-clockwise toward horizontal.
        preferred_delta = -first_angle_degrees
        return {
            "constraint_name": "R9_R10_counter_clockwise_from_50_50",
            "relative_angle_range_degrees": [15.0, 90.0],
            "preferred_relative_angle_degrees": float(np.clip(preferred_delta, 15.0, 90.0)),
        }

    if label_set == {12, 13}:  # R11/R12: absolute cutline angle should point down-right.
        return {
            "constraint_name": "R11_R12_absolute_clockwise_cutline",
            "absolute_angle_range_degrees": [-90.0, -60.0],
            "preferred_absolute_angle_degrees": -75.0,
        }

    return {
        "constraint_name": "unconstrained",
        "relative_angle_range_degrees": [-180.0, 180.0],
        "preferred_relative_angle_degrees": 45.0,
    }


def optimize_second_cutline_from_first_cutline(
    subset_indices: np.ndarray,
    region_points: np.ndarray,
    first_cut_anchor: np.ndarray,
    first_split_angle: float,
    origin_projection_range_mm: tuple[float, float],
    orientation_constraint: dict,
    angle_sample_count: int = 360,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Split one 50/50 half with a cutline whose origin lies on the first cutline."""
    if subset_indices.size == 0:
        return subset_indices, subset_indices, np.asarray(first_cut_anchor, dtype=float), {
            "angle_radians": 0.0,
            "angle_degrees": 0.0,
            "achieved_fraction": 0.0,
            "positive_count": 0,
            "negative_count": 0,
            "status": "fallback_empty",
        }
    if subset_indices.size == 1:
        return subset_indices, np.empty((0,), dtype=subset_indices.dtype), np.asarray(first_cut_anchor, dtype=float), {
            "angle_radians": 0.0,
            "angle_degrees": 0.0,
            "achieved_fraction": 1.0,
            "positive_count": 1,
            "negative_count": 0,
            "status": "fallback_single_voxel",
        }

    first_cutline_direction_xz = cutline_angle_to_direction_xz(first_split_angle)
    subset_points = region_points[subset_indices]
    offsets_xz = np.column_stack(
        (
            subset_points[:, 0] - first_cut_anchor[0],
            subset_points[:, 2] - first_cut_anchor[2],
        )
    )
    origin_projection_min, origin_projection_max = origin_projection_range_mm
    origin_projection_center = float((origin_projection_min + origin_projection_max) / 2.0)
    relative_angle_range = orientation_constraint.get("relative_angle_range_degrees")
    preferred_relative_angle = orientation_constraint.get("preferred_relative_angle_degrees")
    absolute_angle_range = orientation_constraint.get("absolute_angle_range_degrees")
    preferred_absolute_angle = orientation_constraint.get("preferred_absolute_angle_degrees")

    best_candidate: dict | None = None
    for split_angle in np.linspace(0.0, 2.0 * np.pi, angle_sample_count, endpoint=False):
        relative_angle = signed_angle_difference_degrees(float(split_angle), first_split_angle)
        angle_degrees = angle_radians_to_degrees(float(split_angle))
        angle_preference_error = 0.0
        if relative_angle_range is not None:
            relative_angle_min, relative_angle_max = relative_angle_range
            if not (relative_angle_min <= relative_angle <= relative_angle_max):
                continue
            angle_preference_error = abs(relative_angle - float(preferred_relative_angle))
        if absolute_angle_range is not None:
            absolute_angle_min, absolute_angle_max = absolute_angle_range
            if not (absolute_angle_min <= angle_degrees <= absolute_angle_max):
                continue
            angle_preference_error = abs(angle_degrees - float(preferred_absolute_angle))

        normal_direction_xz = cutline_angle_to_normal_xz(float(split_angle))
        denominator = float(first_cutline_direction_xz @ normal_direction_xz)
        if abs(denominator) < 1e-6:
            continue

        signed_offsets = offsets_xz @ normal_direction_xz
        desired_origin_projection = float(np.median(signed_offsets / denominator))
        origin_projection = float(
            np.clip(desired_origin_projection, origin_projection_min, origin_projection_max)
        )
        signed_distances = signed_offsets - origin_projection * denominator
        positive_side = signed_distances >= 0
        positive_count = int(np.count_nonzero(positive_side))
        negative_count = int(subset_indices.size - positive_count)
        if positive_count == 0 or negative_count == 0:
            continue

        achieved_fraction = float(positive_count / subset_indices.size)
        balance_error = abs(achieved_fraction - 0.5)
        origin_center_distance = abs(origin_projection - origin_projection_center)
        candidate_key = (balance_error, angle_preference_error, origin_center_distance)
        if best_candidate is None or candidate_key < best_candidate["key"]:
            best_candidate = {
                "key": candidate_key,
                "split_angle": float(split_angle),
                "origin_projection": origin_projection,
                "positive_side": positive_side,
                "achieved_fraction": achieved_fraction,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "relative_angle": relative_angle,
            }

    if best_candidate is None:
        fallback_angles = compute_coronal_angles_from_anchor(subset_points, first_cut_anchor)
        positive_indices, negative_indices, split_metadata = optimize_balanced_plane_split(
            np.arange(subset_indices.size, dtype=np.int64),
            fallback_angles,
        )
        positive_subset_indices = subset_indices[positive_indices]
        negative_subset_indices = subset_indices[negative_indices]
        split_metadata.update(
            {
                "status": "fallback_first_anchor_no_valid_first_cutline_origin",
                "second_cut_anchor_projection_mm": 0.0,
                "second_cutline_angle_separation_degrees": undirected_angle_difference_degrees(
                    split_metadata["angle_radians"],
                    first_split_angle,
                ),
                "second_cutline_angle_degrees": angle_radians_to_degrees(float(split_metadata["angle_radians"])),
                "second_cutline_relative_angle_from_first_degrees": signed_angle_difference_degrees(
                    split_metadata["angle_radians"],
                    first_split_angle,
                ),
                "second_cutline_orientation_constraint": orientation_constraint,
            }
        )
        return positive_subset_indices, negative_subset_indices, np.asarray(first_cut_anchor, dtype=float), split_metadata

    positive_indices = subset_indices[best_candidate["positive_side"]]
    negative_indices = subset_indices[~best_candidate["positive_side"]]
    anchor_offset_mm = float(best_candidate["origin_projection"])
    second_cut_anchor = np.asarray(
        [
            first_cut_anchor[0] + anchor_offset_mm * first_cutline_direction_xz[0],
            first_cut_anchor[1],
            first_cut_anchor[2] + anchor_offset_mm * first_cutline_direction_xz[1],
        ],
        dtype=float,
    )
    metadata = {
        "angle_radians": float(best_candidate["split_angle"]),
        "angle_degrees": angle_radians_to_degrees(float(best_candidate["split_angle"])),
        "achieved_fraction": float(best_candidate["achieved_fraction"]),
        "positive_count": int(best_candidate["positive_count"]),
        "negative_count": int(best_candidate["negative_count"]),
        "status": "optimized_first_cutline_origin",
        "second_cut_anchor_projection_mm": anchor_offset_mm,
        "second_cutline_angle_separation_degrees": undirected_angle_difference_degrees(
            float(best_candidate["split_angle"]),
            first_split_angle,
        ),
        "second_cutline_relative_angle_from_first_degrees": float(best_candidate["relative_angle"]),
        "second_cutline_orientation_constraint": orientation_constraint,
    }
    return positive_indices, negative_indices, second_cut_anchor, metadata

def split_region9_12(
    output_array: np.ndarray,
    coarse_array: np.ndarray,
    reference_image: sitk.Image,
    totalseg_folder: Path,
) -> dict | None:
    """
    Split region9_12 using two coronal-view cutlines:
    1) first cutline: 50/50 split through the superior mesenteric artery (branches off "aorta" at "vertebrae_L1")
    2) each 50 half is split into 25/25 of total volume by a cutline whose origin is constrained to the first 50/50 cutline

    """
    region_mask = np.isin(coarse_array, COARSE_REGION_GROUPS["region9_12"])
    if not np.any(region_mask):
        return None

    aorta_mask = load_totalseg_structure_mask(totalseg_folder, "aorta", reference_image)
    vertebrae_L1_mask = load_totalseg_structure_mask(totalseg_folder, "vertebrae_L1", reference_image)

    vertebrae_L1_coordinates = np.where(vertebrae_L1_mask)
    if len(vertebrae_L1_coordinates[0]) == 0:
        raise ValueError("TotalSegmentator vertebrae_L1 mask is empty; cannot infer SMA anchor")

    aorta_coordinates = np.where(aorta_mask)
    if len(aorta_coordinates[0]) == 0:
        raise ValueError("TotalSegmentator aorta mask is empty; cannot infer SMA anchor")

    vertebrae_L1_z_half_height_index = float(np.median(vertebrae_L1_coordinates[0]))
    aorta_z_indices = aorta_coordinates[0]
    selected_aorta_z_index = int(
        aorta_z_indices[np.argmin(np.abs(aorta_z_indices - vertebrae_L1_z_half_height_index))]
    )
    aorta_slice_mask = aorta_z_indices == selected_aorta_z_index
    aorta_points = array_coordinates_to_physical_points(aorta_coordinates, reference_image)
    aorta_slice_points = aorta_points[aorta_slice_mask]
    first_cut_anchor = np.asarray(np.median(aorta_slice_points, axis=0), dtype=float)
    first_anchor_metadata = {
        "first_cut_anchor_strategy": "median_aorta_at_vertebrae_L1_half_height",
        "vertebrae_L1_voxel_count": int(vertebrae_L1_coordinates[0].size),
        "aorta_voxel_count": int(aorta_points.shape[0]),
        "aorta_slice_voxel_count": int(aorta_slice_points.shape[0]),
        "vertebrae_L1_z_half_height_index": vertebrae_L1_z_half_height_index,
        "selected_aorta_z_index": selected_aorta_z_index,
    }

    region_coordinates = np.where(region_mask)
    region_points = array_coordinates_to_physical_points(region_coordinates, reference_image)
    first_angles = compute_coronal_angles_from_anchor(region_points, first_cut_anchor)
    all_indices = np.arange(first_angles.size, dtype=np.int64)
    cutline_intersection_tolerance_mm = infer_coronal_voxel_half_diagonal(reference_image)

    first_positive, first_negative, first_split = optimize_balanced_plane_split(
        all_indices,
        first_angles,
    )

    first_cutline_origin_range, first_cutline_origin_metadata = infer_first_cutline_region_projection_range(
        region_points=region_points,
        first_cut_anchor=first_cut_anchor,
        first_split_angle=first_split["angle_radians"],
        cutline_intersection_tolerance_mm=cutline_intersection_tolerance_mm,
    )

    half_groups = [first_positive, first_negative]
    ordered_half_ids = order_group_ids_by_coronal_angle(half_groups, first_angles)
    half_label_pairs = [
        [PCI_REGION_MAPPING[region_name] for region_name in label_pair]
        for label_pair in REGION9_12_HALF_CORONAL_ORDER
    ]
    label_order = [PCI_REGION_MAPPING[region_name] for region_name in REGION9_12_CORONAL_ORDER]

    assigned_labels = np.full(first_angles.shape, label_order[0], dtype=np.uint8)
    first_cutline_direction_xz = cutline_angle_to_direction_xz(first_split["angle_radians"])
    first_cutline_segment_points = [
        [
            float(first_cut_anchor[0] + projection * first_cutline_direction_xz[0]),
            float(first_cut_anchor[1]),
            float(first_cut_anchor[2] + projection * first_cutline_direction_xz[1]),
        ]
        for projection in first_cutline_origin_range
    ]

    cutlines = {
        "50_50_region9_12": {
            "description": "Splits the combined region9_12 volume into two equal halves.",
            "splits_labels": [int(label) for label in label_order],
            "origin_point_physical": [float(value) for value in first_cut_anchor],
            "in_region_segment_endpoints_physical": first_cutline_segment_points,
            "in_region_projection_range_mm": [
                float(first_cutline_origin_range[0]),
                float(first_cutline_origin_range[1]),
            ],
            "cutline_angle_degrees": first_split["angle_degrees"],
            "volume_ratio": first_split["achieved_fraction"],
            "status": first_split["status"],
        }
    }

    for half_id, label_pair in zip(ordered_half_ids, half_label_pairs):
        label_set = set(label_pair)
        if label_set == {10, 11}:  # R9/R10: counter-clockwise from the first cutline.
            orientation_constraint = {
                "constraint_name": "R9_R10_constraint",
                "relative_angle_range_degrees": [0.0, 45.0],
                "preferred_relative_angle_degrees": 45.0,
            }
        elif label_set == {12, 13}:  # R11/R12: clockwise from the first cutline.
            orientation_constraint = {
                "constraint_name": "R11_R12_constraint",
                "relative_angle_range_degrees": [-45.0, 0.0],
                "preferred_relative_angle_degrees": -45.0,
            }
        else:
            orientation_constraint = infer_second_cutline_orientation_constraint(label_pair, first_split["angle_radians"])

        positive_indices, negative_indices, second_anchor, second_split = optimize_second_cutline_from_first_cutline(
            subset_indices=half_groups[half_id],
            region_points=region_points,
            first_cut_anchor=first_cut_anchor,
            first_split_angle=first_split["angle_radians"],
            origin_projection_range_mm=first_cutline_origin_range,
            orientation_constraint=orientation_constraint,
        )
        split_groups = [positive_indices, negative_indices]
        split_angles = compute_coronal_angles_from_anchor(region_points, second_anchor)
        ordered_split_ids = order_group_ids_by_coronal_angle(split_groups, split_angles)
        for split_id, label_value in zip(ordered_split_ids, label_pair):
            assigned_labels[split_groups[split_id]] = label_value

        label_pair_key = "_".join(f"R{int(label) - 1}" for label in sorted(label_pair))
        cutlines[f"25_25_{label_pair_key}"] = {
            "description": f"Splits {label_pair_key} into two equal volumes.",
            "labels": [int(label) for label in label_pair],
            "origin_point_physical": [float(value) for value in second_anchor],
            "origin_projection_on_first_cutline_mm": float(second_split["second_cut_anchor_projection_mm"]),
            "cutline_angle_degrees": second_split["angle_degrees"],
            "undirected_angle_separation_from_first_cutline_degrees": second_split[
                "second_cutline_angle_separation_degrees"
            ],
            "directed_relative_angle_from_first_cutline_degrees": second_split[
                "second_cutline_relative_angle_from_first_degrees"
            ],
            "orientation_constraint": second_split["second_cutline_orientation_constraint"],
            "origin_projection_range_on_first_cutline_mm": [
                float(first_cutline_origin_range[0]),
                float(first_cutline_origin_range[1]),
            ],
            "volume_ratio": second_split["achieved_fraction"],
            "status": second_split["status"],
        }

    output_array[region_coordinates] = assigned_labels
    label_values, label_counts = np.unique(assigned_labels, return_counts=True)
    counts = {int(label): int(count) for label, count in zip(label_values, label_counts)}
    total_count = int(first_angles.size)
    target_fraction = 0.25
    tolerance = 0.10 * target_fraction
    volume_fractions = {
        int(label): float(counts.get(int(label), 0) / total_count) for label in label_order
    }
    within_tolerance = all(
        abs(volume_fractions[int(label)] - target_fraction) <= tolerance for label in label_order
    )

    split_metadata = {
        "strategy": "two_cutlines_from_aorta_at_vertebrae_L1_anchor",
        "label_order": [int(label) for label in label_order],
        "counts": counts,
        "region9_12_volume_fractions": volume_fractions,
        "within_10_percent_tolerance": bool(within_tolerance),
        "cutlines": cutlines,
    }
    split_metadata.update(first_anchor_metadata)
    split_metadata.update(first_cutline_origin_metadata)

    logger.info(
        "region9_12 split with aorta/L1 two-cutline method: %s (within tolerance=%s)",
        counts,
        within_tolerance,
    )
    return split_metadata

    
def run_totalsegmentator_for_constraints(
    scan_path: Path,
    output_folder: Path,
    fast: bool = False,
    device: str = "gpu",
) -> Path:
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as error:
        raise ImportError(
            "TotalSegmentator is required when --run_totalsegmentator is used. "
            "Install it or provide --totalseg_folder with the required constraint masks."
        ) from error

    output_folder.mkdir(parents=True, exist_ok=True)
    logger.info("Running TotalSegmentator constraints for %s", scan_path)
    totalsegmentator(
        input=str(scan_path),
        output=str(output_folder),
        fast=fast,
        ml=False,
        task="total",
        device=device,
        roi_subset=sorted(TOTALSEG_STRUCTURES),
    )
    return output_folder


def get_or_create_totalseg_folder(
    scan_path: Optional[Path],
    output_path: Path,
    totalseg_folder: Optional[Path],
    run_totalsegmentator: bool,
    fast: bool,
    device: str,
) -> Path:
    if totalseg_folder is not None:
        return totalseg_folder

    if not run_totalsegmentator:
        raise ValueError(
            "TotalSegmentator masks are required for coarse-region splitting. "
            "Provide --totalseg_folder or use --run_totalsegmentator with --scan."
        )

    if scan_path is None:
        raise ValueError("--scan is required when --run_totalsegmentator is used")

    case_name = strip_nii_suffix(scan_path)
    generated_folder = output_path.parent / "totalsegmentator_output" / case_name
    return run_totalsegmentator_for_constraints(scan_path, generated_folder, fast=fast, device=device)


def copy_direct_regions(output_array: np.ndarray, coarse_array: np.ndarray) -> None:
    for region_name, coarse_label_values in COMBINED_REGION_MAPPING.items():
        if region_name in COARSE_REGION_GROUPS or region_name == "background":
            continue

        output_label = PCI_REGION_MAPPING[region_name]
        region_mask = np.isin(coarse_array, as_label_values(coarse_label_values))
        output_array[region_mask] = output_label


def postprocess_coarse_segmentation_file(
    coarse_segmentation_path: str | Path,
    output_path: str | Path,
    scan_path: str | Path | None = None,
    totalseg_folder: str | Path | None = None,
    run_totalsegmentator: bool = False,
    fast: bool = False,
    device: str = "gpu",
    debug_json_path: str | Path | None = None,
) -> None:
    coarse_segmentation_path = Path(coarse_segmentation_path)
    output_path = Path(output_path)
    scan_path = Path(scan_path) if scan_path is not None else None
    totalseg_folder = Path(totalseg_folder) if totalseg_folder is not None else None
    debug_json_path = Path(debug_json_path) if debug_json_path is not None else None

    coarse_image = sitk.ReadImage(str(coarse_segmentation_path))
    coarse_array = sitk.GetArrayFromImage(coarse_image).astype(np.int16)
    output_array = np.zeros_like(coarse_array, dtype=np.uint8)

    logger.info("Processing %s", coarse_segmentation_path)
    logger.info("Coarse labels: %s", np.unique(coarse_array).tolist())

    copy_direct_regions(output_array, coarse_array)

    needs_hip_split = any(
        np.any(np.isin(coarse_array, COARSE_REGION_GROUPS[region_name]))
        for region_name in COARSE_SPLIT_STRUCTURES
    )
    needs_region9_12_split = np.any(np.isin(coarse_array, COARSE_REGION_GROUPS["region9_12"]))
    if needs_hip_split or needs_region9_12_split:
        totalseg_folder = get_or_create_totalseg_folder(
            scan_path=scan_path,
            output_path=output_path,
            totalseg_folder=totalseg_folder,
            run_totalsegmentator=run_totalsegmentator,
            fast=fast,
            device=device,
        )

    if needs_hip_split:
        hip_left = load_totalseg_structure_mask(totalseg_folder, COARSE_SPLIT_STRUCTURES["region45"], coarse_image)
        split_region_by_hip_plane(
            output_array=output_array,
            coarse_array=coarse_array,
            coarse_label_values=COARSE_REGION_GROUPS["region45"],
            hip_mask=hip_left,
            reference_image=coarse_image,
            inferior_label=PCI_REGION_MAPPING["region5"],
            superior_label=PCI_REGION_MAPPING["region4"],
            region_name="region45",
            structure_name=COARSE_SPLIT_STRUCTURES["region45"],
        )

        hip_right = load_totalseg_structure_mask(totalseg_folder, COARSE_SPLIT_STRUCTURES["region78"], coarse_image)
        split_region_by_hip_plane(
            output_array=output_array,
            coarse_array=coarse_array,
            coarse_label_values=COARSE_REGION_GROUPS["region78"],
            hip_mask=hip_right,
            reference_image=coarse_image,
            inferior_label=PCI_REGION_MAPPING["region7"],
            superior_label=PCI_REGION_MAPPING["region8"],
            region_name="region78",
            structure_name=COARSE_SPLIT_STRUCTURES["region78"],
        )

    split_metadata = None
    if needs_region9_12_split:
        split_metadata = split_region9_12(output_array, coarse_array, coarse_image, totalseg_folder)

    output_image = sitk.GetImageFromArray(output_array)
    output_image.CopyInformation(coarse_image)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(output_image, str(output_path), useCompression=True)
    logger.info("Saved %s with labels %s", output_path, np.unique(output_array).tolist())

    if debug_json_path is not None and split_metadata is not None:
        debug_json_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_json_path.open("w") as debug_file:
            json.dump(split_metadata, debug_file, indent=2)
        logger.info("Saved region9_12 debug metadata to %s", debug_json_path)


def find_scan_for_segmentation(segmentation_path: Path, scan_folder: Path, image_suffix: str = "_0000") -> Path:
    case_name = strip_nii_suffix(segmentation_path)
    candidates = [
        scan_folder / f"{case_name}{image_suffix}.nii.gz",
        scan_folder / f"{case_name}{image_suffix}.nii",
        scan_folder / segmentation_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not find scan for {segmentation_path.name} in {scan_folder}")


def find_case_totalseg_folder(segmentation_path: Path, totalseg_root: Optional[Path]) -> Optional[Path]:
    if totalseg_root is None:
        return None

    if has_any_totalseg_structure(totalseg_root):
        return totalseg_root

    case_name = strip_nii_suffix(segmentation_path)
    candidates = [
        totalseg_root / case_name,
        totalseg_root / f"{case_name}_totalsegmentator",
        totalseg_root / f"{case_name}_totalsegmentator_output",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return totalseg_root / case_name


def postprocess_coarse_segmentation(
    input_folder: str | Path,
    output_folder: str | Path,
    scan_folder: str | Path | None = None,
    totalseg_folder: str | Path | None = None,
    run_totalsegmentator: bool = False,
    fast: bool = False,
    device: str = "gpu",
    file_pattern: str = "*.nii.gz",
    image_suffix: str = "_0000",
    debug_folder: str | Path | None = None,
) -> None:
    """
    Postprocess coarsely segmented regions back to the original thirteen PCI regions using TotalSegmentator.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    scan_folder = Path(scan_folder) if scan_folder is not None else None
    totalseg_folder = Path(totalseg_folder) if totalseg_folder is not None else None
    debug_folder = Path(debug_folder) if debug_folder is not None else None
    output_folder.mkdir(parents=True, exist_ok=True)
    if debug_folder is not None:
        debug_folder.mkdir(parents=True, exist_ok=True)

    segmentation_files = sorted(input_folder.glob(file_pattern))
    if not segmentation_files:
        logger.warning("No coarse segmentations found in %s matching %s", input_folder, file_pattern)
        return

    for segmentation_path in tqdm(segmentation_files, desc="Postprocessing coarse segmentations"):
        output_path = output_folder / segmentation_path.name
        scan_path = find_scan_for_segmentation(segmentation_path, scan_folder, image_suffix) if scan_folder else None
        case_totalseg_folder = find_case_totalseg_folder(segmentation_path, totalseg_folder)
        debug_json_path = debug_folder / f"{strip_nii_suffix(segmentation_path)}.json" if debug_folder else None
        postprocess_coarse_segmentation_file(
            coarse_segmentation_path=segmentation_path,
            output_path=output_path,
            scan_path=scan_path,
            totalseg_folder=case_totalseg_folder,
            run_totalsegmentator=run_totalsegmentator,
            fast=fast,
            device=device,
            debug_json_path=debug_json_path,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Postprocess coarse PCI segmentations back to the original thirteen PCI regions."
    )
    parser.add_argument("--coarse_segmentation", type=str, help="Path to one coarse segmentation NIfTI file.")
    parser.add_argument("--scan", type=str, help="Path to the matching original scan, used when running TotalSegmentator.")
    parser.add_argument("--output", type=str, help="Path to write one postprocessed segmentation NIfTI file.")
    parser.add_argument("--input_folder", type=str, help="Path to a folder containing coarse segmentations.")
    parser.add_argument("--scan_folder", type=str, help="Path to a folder containing original scans for batch mode.")
    parser.add_argument("--output_folder", type=str, help="Path to save postprocessed segmentations in batch mode.")
    parser.add_argument("--totalseg_folder", type=str, help="Existing TotalSegmentator output folder.")
    parser.add_argument("--run_totalsegmentator", action="store_true", help="Run TotalSegmentator if hip masks are not supplied.")
    parser.add_argument("--fast", action="store_true", help="Use TotalSegmentator fast mode.")
    parser.add_argument("--device", type=str, default="gpu", help="TotalSegmentator device, e.g. cpu, gpu, or gpu:0.")
    parser.add_argument("--file_pattern", type=str, default="*.nii.gz", help="Batch segmentation file pattern.")
    parser.add_argument("--image_suffix", type=str, default="_0000", help="Scan suffix used in batch mode.")
    parser.add_argument("--debug_json", type=str, help="Optional JSON path for single-scan region9_12 split metadata.")
    parser.add_argument("--debug_folder", type=str, help="Optional folder for batch region9_12 split metadata JSON files.")
    args = parser.parse_args()

    if args.coarse_segmentation:
        if not args.output:
            parser.error("--output is required with --coarse_segmentation")
        postprocess_coarse_segmentation_file(
            coarse_segmentation_path=args.coarse_segmentation,
            output_path=args.output,
            scan_path=args.scan,
            totalseg_folder=args.totalseg_folder,
            run_totalsegmentator=args.run_totalsegmentator,
            fast=args.fast,
            device=args.device,
            debug_json_path=args.debug_json,
        )
        return

    if not args.input_folder or not args.output_folder:
        parser.error("Provide either --coarse_segmentation/--output or --input_folder/--output_folder")

    postprocess_coarse_segmentation(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        scan_folder=args.scan_folder,
        totalseg_folder=args.totalseg_folder,
        run_totalsegmentator=args.run_totalsegmentator,
        fast=args.fast,
        device=args.device,
        file_pattern=args.file_pattern,
        image_suffix=args.image_suffix,
        debug_folder=args.debug_folder,
    )


if __name__ == "__main__":
    main()