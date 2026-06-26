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
        REGION9_12_CORONAL_ORDER,
        TOTALSEG_STRUCTURES,
    )
except ImportError:
    from config import (
        COARSE_REGION_GROUPS,
        COARSE_SPLIT_STRUCTURES,
        COMBINED_REGION_MAPPING,
        PCI_REGION_MAPPING,
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


def infer_mesenteric_root_anchor(
    totalseg_folder: Path,
    reference_image: sitk.Image,
) -> tuple[np.ndarray, dict]:
    """
    Estimate the mesenteric-root attachment (~ligament of Treitz / duodenojejunal flexure).

    The duodenojejunal flexure is the proximal attachment of the small-bowel mesenteric
    root, anatomically defined as the duodenum->small_bowel transition. We estimate it as
    the centroid of the contact zone between the (dilated) duodenum and the small_bowel
    masks.
    """
    from scipy.ndimage import binary_dilation

    duodenum_mask = load_totalseg_structure_mask(totalseg_folder, "duodenum", reference_image)
    small_bowel_mask = load_totalseg_structure_mask(totalseg_folder, "small_bowel", reference_image)
    if not duodenum_mask.any():
        raise ValueError("TotalSegmentator duodenum mask is empty; cannot infer mesenteric-root anchor")
    if not small_bowel_mask.any():
        raise ValueError("TotalSegmentator small_bowel mask is empty; cannot infer mesenteric-root anchor")

    for iterations in (2, 4, 8):
        contact_mask = binary_dilation(duodenum_mask, iterations=iterations) & small_bowel_mask
        if contact_mask.any():
            contact_coordinates = np.where(contact_mask)
            contact_points = array_coordinates_to_physical_points(contact_coordinates, reference_image)
            anchor = np.asarray(np.median(contact_points, axis=0), dtype=float)
            metadata = {
                "mesenteric_root_anchor_strategy": "duodenojejunal_contact_centroid",
                "mesenteric_root_contact_dilation_iterations": int(iterations),
                "mesenteric_root_contact_voxel_count": int(contact_coordinates[0].size),
                "duodenum_voxel_count": int(duodenum_mask.sum()),
                "small_bowel_voxel_count": int(small_bowel_mask.sum()),
            }
            return anchor, metadata

    raise ValueError("Could not find a duodenum-small_bowel contact zone for mesenteric-root anchor")


def split_region9_12_equal_volume_fan(
    output_array: np.ndarray,
    coarse_array: np.ndarray,
    reference_image: sitk.Image,
    totalseg_folder: Path,
) -> dict | None:
    """
    Split region9_12 with anteroposterior planes radiating from a single origin
    (the mesenteric root at the ligament of Treitz), dividing the small bowel into
    four equal-volume wedges, per the rPCI Delphi consensus definition.

    Implementation: every region voxel is assigned a coronal (x/z) polar angle around
    the anchor; the angles are unwrapped around their largest empty gap, and three
    boundaries are placed at the 25/50/75% volume quantiles. Because all voxels share
    one origin, the boundaries form a true radial fan, and equal voxel count yields
    equal volume by construction. Assignment depends only on (x, z), so it is invariant
    along the anteroposterior (y) axis.

    The fan origin is inferred from the TotalSegmentator duodenum and small_bowel masks
    at their transition/contact zone.
    """
    region_mask = np.isin(coarse_array, COARSE_REGION_GROUPS["region9_12"])
    if not np.any(region_mask):
        return None

    anchor, anchor_metadata = infer_mesenteric_root_anchor(totalseg_folder, reference_image)

    region_coordinates = np.where(region_mask)
    region_points = array_coordinates_to_physical_points(region_coordinates, reference_image)
    angles = compute_coronal_angles_from_anchor(region_points, anchor)
    voxel_count = int(angles.size)

    label_order = [PCI_REGION_MAPPING[region_name] for region_name in REGION9_12_CORONAL_ORDER]
    assigned_labels = np.full(angles.shape, label_order[0], dtype=np.uint8)

    unwrapped_angles, sweep_start_angle, largest_gap = unwrap_angles_around_largest_gap(angles)
    sweep_order = np.argsort(unwrapped_angles, kind="stable")
    ranks = np.empty(voxel_count, dtype=np.int64)
    ranks[sweep_order] = np.arange(voxel_count, dtype=np.int64)
    quantile_fractions = (ranks + 0.5) / max(voxel_count, 1)
    bin_index = np.clip((quantile_fractions * 4.0).astype(np.int64), 0, 3)

    wedge_groups = [np.flatnonzero(bin_index == wedge_id) for wedge_id in range(4)]
    ordered_wedge_ids = order_group_ids_by_coronal_angle(wedge_groups, angles)
    group_label = [label_order[0]] * 4
    for sweep_position, wedge_id in enumerate(ordered_wedge_ids):
        label_value = label_order[sweep_position]
        group_label[wedge_id] = label_value
        assigned_labels[wedge_groups[wedge_id]] = label_value

    sorted_unwrapped_angles = unwrapped_angles[sweep_order]
    boundary_split_indices = [int(round(boundary * voxel_count / 4.0)) for boundary in (1, 2, 3)]
    boundary_unwrapped_angles = [
        float(sorted_unwrapped_angles[min(max(split_index, 0), voxel_count - 1)])
        for split_index in boundary_split_indices
    ]
    boundary_absolute_angles = [
        float((sweep_start_angle + unwrapped_angle) % (2 * np.pi))
        for unwrapped_angle in boundary_unwrapped_angles
    ]

    cutline_intersection_tolerance_mm = infer_coronal_voxel_half_diagonal(reference_image)
    cutlines = {}
    for boundary_id, boundary_angle in enumerate(boundary_absolute_angles):
        lower_label = group_label[boundary_id]
        upper_label = group_label[boundary_id + 1]
        projection_range, projection_metadata = infer_first_cutline_region_projection_range(
            region_points=region_points,
            first_cut_anchor=anchor,
            first_split_angle=boundary_angle,
            cutline_intersection_tolerance_mm=cutline_intersection_tolerance_mm,
        )
        direction_xz = cutline_angle_to_direction_xz(boundary_angle)
        segment_endpoints = [
            [
                float(anchor[0] + projection * direction_xz[0]),
                float(anchor[1]),
                float(anchor[2] + projection * direction_xz[1]),
            ]
            for projection in projection_range
        ]
        label_pair_key = "_".join(f"R{int(label) - 1}" for label in sorted((lower_label, upper_label)))
        cutlines[f"boundary_{boundary_id + 1}_{label_pair_key}"] = {
            "description": f"Equal-volume radial boundary between {label_pair_key}.",
            "separates_labels": [int(lower_label), int(upper_label)],
            "origin_point_physical": [float(value) for value in anchor],
            "cutline_angle_degrees": angle_radians_to_degrees(boundary_angle),
            "in_region_segment_endpoints_physical": segment_endpoints,
            "in_region_projection_range_mm": [float(projection_range[0]), float(projection_range[1])],
            "region_constraint_status": projection_metadata["first_cutline_region_constraint_status"],
        }

    output_array[region_coordinates] = assigned_labels
    label_values, label_counts = np.unique(assigned_labels, return_counts=True)
    counts = {int(label): int(count) for label, count in zip(label_values, label_counts)}
    target_fraction = 0.25
    tolerance = 0.10 * target_fraction
    volume_fractions = {
        int(label): float(counts.get(int(label), 0) / max(voxel_count, 1)) for label in label_order
    }
    within_tolerance = all(
        abs(volume_fractions[int(label)] - target_fraction) <= tolerance for label in label_order
    )

    split_metadata = {
        "strategy": "equal_volume_angular_fan_from_mesenteric_root",
        "label_order": [int(label) for label in label_order],
        "counts": counts,
        "region9_12_volume_fractions": volume_fractions,
        "within_10_percent_tolerance": bool(within_tolerance),
        "anchor_point_physical": [float(value) for value in anchor],
        "sweep_start_angle_degrees": angle_radians_to_degrees(float(sweep_start_angle)),
        "sweep_span_degrees": float(np.degrees(2 * np.pi - largest_gap)),
        "largest_angular_gap_degrees": float(np.degrees(largest_gap)),
        "boundary_angles_degrees": [angle_radians_to_degrees(angle) for angle in boundary_absolute_angles],
        "cutlines": cutlines,
    }
    split_metadata.update(anchor_metadata)

    logger.info(
        "region9_12 split with equal-volume angular fan: %s (within tolerance=%s)",
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
    case_name: Optional[str] = None,
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

    if case_name is None:
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

    case_name = strip_nii_suffix(coarse_segmentation_path)
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
            case_name=case_name,
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
        split_metadata = split_region9_12_equal_volume_fan(output_array, coarse_array, coarse_image, totalseg_folder)

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