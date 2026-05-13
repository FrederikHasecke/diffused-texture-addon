from collections.abc import Mapping, Sequence

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

import numpy as np
from numpy.typing import NDArray

from ..blender_operations import UVPassAssets
from .process_operations import _require_cv2, smooth_alpha_map
from .uv_pass import _prepare_base_texture, _seam_link_arrays_are_valid

RGB_CHANNELS = 3
SEAM_STITCH_MISMATCH_THRESHOLD = 18.0
SEAM_STITCH_CONFIDENCE_MARGIN = 0.05
SEAM_STITCH_MAX_CORRECTION = 96.0
CORRECTION_WEIGHT_EPSILON = 1e-6


def _extract_facing_weight(facing_image: NDArray) -> NDArray[np.float32]:
    facing = facing_image[..., 0] if facing_image.ndim == RGB_CHANNELS else facing_image
    facing = facing.astype(np.float32)
    if facing.size == 0:
        return facing
    if float(np.nanmax(facing)) > 1.0:
        facing = facing / 255.0
    return np.clip(facing, 0.0, 1.0)


def _compute_uv_flat_indices(
    uv_image: NDArray,
    texture_resolution: int,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    uv = uv_image[..., :2].astype(np.float32)
    valid_mask = np.sum(uv, axis=-1) > 0
    if not np.any(valid_mask):
        return np.empty((0,), dtype=np.int64), valid_mask

    coords = uv[valid_mask]
    uv_x = np.mod(
        np.rint(coords[:, 0] * float(texture_resolution - 1)).astype(np.int64),
        texture_resolution,
    )
    uv_y = np.mod(
        (texture_resolution - 1)
        - np.rint(coords[:, 1] * float(texture_resolution - 1)).astype(np.int64),
        texture_resolution,
    )
    return ((uv_y * texture_resolution) + uv_x).astype(np.int64), valid_mask


def accumulate_stitch_confidence(
    multiview_images: Mapping[str, Sequence[NDArray]] | None,
    texture_resolution: int,
) -> NDArray[np.float32]:
    confidence = np.zeros((texture_resolution, texture_resolution), dtype=np.float32)
    if multiview_images is None:
        return confidence

    uv_images = multiview_images.get("uv", [])
    facing_images = multiview_images.get("facing", [])
    coverage_count = np.zeros_like(confidence)
    best_facing = np.zeros_like(confidence)

    for uv_image, facing_image in zip(uv_images, facing_images, strict=False):
        flat_indices, valid_mask = _compute_uv_flat_indices(
            uv_image,
            texture_resolution,
        )
        if flat_indices.size == 0:
            continue

        facing_weight = _extract_facing_weight(facing_image)[valid_mask].reshape(-1)
        np.add.at(coverage_count.reshape(-1), flat_indices, 1.0)
        np.maximum.at(best_facing.reshape(-1), flat_indices, facing_weight)

    coverage_boost = np.log1p(coverage_count) / np.log(5.0)
    return np.clip(best_facing + (0.15 * coverage_boost), 0.0, 1.0).astype(
        np.float32,
        copy=False,
    )


def _unique_link_pairs(
    seam_link_source_yx: NDArray[np.int32],
    seam_link_target_yx: NDArray[np.int32],
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    seen: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for source, target in zip(seam_link_source_yx, seam_link_target_yx, strict=False):
        source_coord = (int(source[0]), int(source[1]))
        target_coord = (int(target[0]), int(target[1]))
        if source_coord == target_coord:
            continue
        key = (
            (source_coord, target_coord)
            if source_coord <= target_coord
            else (target_coord, source_coord)
        )
        if key in seen:
            continue
        seen.add(key)
        pairs.append((source_coord, target_coord))
    return pairs


def _clamp_delta(delta: NDArray[np.float32]) -> NDArray[np.float32]:
    magnitude = float(np.linalg.norm(delta))
    if magnitude <= SEAM_STITCH_MAX_CORRECTION:
        return delta
    return (delta * (SEAM_STITCH_MAX_CORRECTION / magnitude)).astype(np.float32)


def _accumulate_pairwise_corrections(
    texture: NDArray[np.uint8],
    link_pairs: list[tuple[tuple[int, int], tuple[int, int]]],
    confidence: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    correction_sum = np.zeros(texture.shape, dtype=np.float32)
    correction_weight = np.zeros(texture.shape[:2], dtype=np.float32)

    for source_coord, target_coord in link_pairs:
        source_y, source_x = source_coord
        target_y, target_x = target_coord
        source_color = texture[source_y, source_x].astype(np.float32)
        target_color = texture[target_y, target_x].astype(np.float32)
        color_delta = source_color - target_color
        if float(np.linalg.norm(color_delta)) < SEAM_STITCH_MISMATCH_THRESHOLD:
            continue

        source_confidence = float(confidence[source_y, source_x])
        target_confidence = float(confidence[target_y, target_x])
        if source_confidence > target_confidence + SEAM_STITCH_CONFIDENCE_MARGIN:
            corrections = [(target_coord, _clamp_delta(color_delta))]
        elif target_confidence > source_confidence + SEAM_STITCH_CONFIDENCE_MARGIN:
            corrections = [(source_coord, _clamp_delta(-color_delta))]
        else:
            midpoint = 0.5 * (source_color + target_color)
            corrections = [
                (source_coord, _clamp_delta(midpoint - source_color)),
                (target_coord, _clamp_delta(midpoint - target_color)),
            ]

        for (coord_y, coord_x), delta in corrections:
            correction_sum[coord_y, coord_x] += delta
            correction_weight[coord_y, coord_x] += 1.0

    return correction_sum, correction_weight


def _apply_local_correction_field(
    texture: NDArray[np.uint8],
    correction_sum: NDArray[np.float32],
    correction_weight: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    _require_cv2()

    seed_mask = correction_weight > 0
    if not np.any(seed_mask):
        return texture

    seed_correction = np.divide(
        correction_sum,
        correction_weight[..., None],
        out=np.zeros_like(correction_sum),
        where=correction_weight[..., None] > CORRECTION_WEIGHT_EPSILON,
    )
    local_mask = cv2.dilate(
        seed_mask.astype(np.uint8) * 255,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    local_mask[surface_mask == 0] = 0

    weights = cv2.GaussianBlur(seed_mask.astype(np.float32), (3, 3), 0)
    correction_field = np.zeros_like(seed_correction, dtype=np.float32)
    for channel in range(RGB_CHANNELS):
        blurred_delta = cv2.GaussianBlur(seed_correction[..., channel], (3, 3), 0)
        correction_field[..., channel] = np.divide(
            blurred_delta,
            weights,
            out=np.zeros_like(seed_correction[..., channel]),
            where=weights > CORRECTION_WEIGHT_EPSILON,
        )

    alpha = smooth_alpha_map(local_mask.astype(np.float32) / 255.0)
    alpha *= (local_mask > 0).astype(np.float32)
    alpha *= (surface_mask > 0).astype(np.float32)

    corrected = texture.astype(np.float32) + (correction_field * alpha[..., None])
    stitched = np.clip(np.rint(corrected), 0, 255).astype(np.uint8)
    stitched[surface_mask == 0] = texture[surface_mask == 0]
    return stitched


def uv_stitch_pass(
    texture: NDArray[np.uint8] | NDArray[np.float32],
    uv_assets: UVPassAssets,
    multiview_images: Mapping[str, Sequence[NDArray]] | None = None,
) -> NDArray[np.uint8]:
    """Narrow deterministic seam stitch for image-mode texture generation."""
    _require_cv2()

    texture_resolution = int(texture.shape[0])
    surface_mask = uv_assets.surface_mask.astype(np.uint8)
    base_texture = _prepare_base_texture(texture, texture_resolution)
    base_texture[surface_mask == 0] = 255

    seam_link_source_yx = getattr(uv_assets, "seam_link_source_yx", None)
    seam_link_target_yx = getattr(uv_assets, "seam_link_target_yx", None)
    if not _seam_link_arrays_are_valid(seam_link_source_yx, seam_link_target_yx):
        return base_texture

    source_yx = seam_link_source_yx.astype(np.int32, copy=False)
    target_yx = seam_link_target_yx.astype(np.int32, copy=False)
    link_pairs = _unique_link_pairs(source_yx, target_yx)
    if not link_pairs:
        return base_texture

    confidence = accumulate_stitch_confidence(multiview_images, texture_resolution)
    correction_sum, correction_weight = _accumulate_pairwise_corrections(
        base_texture,
        link_pairs,
        confidence,
    )
    stitched = _apply_local_correction_field(
        base_texture,
        correction_sum,
        correction_weight,
        surface_mask,
    )
    stitched[surface_mask == 0] = base_texture[surface_mask == 0]
    return stitched
