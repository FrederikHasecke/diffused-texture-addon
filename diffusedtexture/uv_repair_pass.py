from collections.abc import Callable
from typing import Any, NamedTuple, cast

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

import numpy as np
from numpy.typing import NDArray

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from ..blender_operations import ProcessParameter, UVPassAssets
from ..model_support import get_default_sd_resolution
from .pipeline.pipeline_builder import create_diffusion_pipeline

try:
    from .pipeline.pipeline_runner import TextureGenerationCancelledError, run_pipeline
except ImportError:
    from .pipeline.pipeline_runner import run_pipeline

    class TextureGenerationCancelledError(RuntimeError):
        """Fallback cancellation error for tests with a stubbed runner."""


from .process_operations import _require_cv2, blend_texture_with_alpha, smooth_alpha_map
from .uv_pass import (
    SEAM_SEARCH_RADIUS,
    _build_chart_frame,
    _decode_baked_normal_vectors,
    _expand_seam_mask,
    _make_seam_boundary_mask,
    _prepare_base_texture,
    _seam_link_arrays_are_valid,
    _split_texture_frequencies,
    apply_position_seam_stitching,
    canonicalize_uv_normal_map,
    find_position_seam_groups,
)
from .uv_seams import seam_link_components, seam_link_mask

AUTO_REPAIR_MAX_STRENGTH = 0.35
LOW_CONFIDENCE_FACING_FLOOR = 0.2
LOW_CONFIDENCE_FACING_CEILING = 0.35
RGB_CHANNELS = 3
TEXTURE_EDGE_LOWER = 64
TEXTURE_EDGE_UPPER = 128
GEOMETRY_EDGE_LOWER = 48
GEOMETRY_EDGE_UPPER = 96
SEAM_MISMATCH_THRESHOLD = 18.0


class _SeamRepairData(NamedTuple):
    seam_groups: list[NDArray[np.int32]]
    topology_groups: list[NDArray[np.int32]]
    fallback_groups: list[NDArray[np.int32]]
    seam_boundary: NDArray[np.uint8]
    seam_band: NDArray[np.uint8]
    seam_link_source_yx: NDArray[np.int32] | None
    seam_link_target_yx: NDArray[np.int32] | None


def _extract_facing_weight(facing_image: NDArray[np.float32]) -> NDArray[np.float32]:
    facing = facing_image[..., 0] if facing_image.ndim == RGB_CHANNELS else facing_image
    facing = facing.astype(np.float32)
    if facing.size == 0:
        return facing
    if float(np.nanmax(facing)) > 1.0:
        facing = facing / 255.0
    return np.clip(facing, 0.0, 1.0)


def _compute_uv_flat_indices(
    uv_image: NDArray[np.float32],
    texture_resolution: int,
) -> tuple[NDArray[np.int64], NDArray[np.bool_]]:
    uv = uv_image[..., :2].astype(np.float32)
    valid_mask = np.all(np.isfinite(uv), axis=-1)
    if uv_image.ndim == RGB_CHANNELS and uv_image.shape[-1] >= 4:  # noqa: PLR2004
        # Rendered UV passes include alpha, which distinguishes atlas-origin
        # texels from untouched background pixels that also decode to (0, 0).
        valid_mask &= np.isfinite(uv_image[..., 3]) & (uv_image[..., 3] > 0)
    else:
        valid_mask &= np.sum(uv, axis=-1) > 0
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


def accumulate_uv_repair_diagnostics(
    multiview_images: dict[str, list[NDArray[Any]]],
    texture_resolution: int,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    coverage_count = np.zeros(
        (texture_resolution, texture_resolution),
        dtype=np.float32,
    )
    best_facing = np.zeros((texture_resolution, texture_resolution), dtype=np.float32)
    weight_sum = np.zeros((texture_resolution, texture_resolution), dtype=np.float32)

    for uv_image, facing_image in zip(
        multiview_images["uv"],
        multiview_images["facing"],
        strict=False,
    ):
        flat_indices, valid_mask = _compute_uv_flat_indices(
            uv_image,
            texture_resolution,
        )
        if flat_indices.size == 0:
            continue

        facing_weight = _extract_facing_weight(facing_image)[valid_mask].reshape(-1)
        np.add.at(coverage_count.reshape(-1), flat_indices, 1.0)
        np.add.at(weight_sum.reshape(-1), flat_indices, facing_weight)
        np.maximum.at(best_facing.reshape(-1), flat_indices, facing_weight)

    return coverage_count, best_facing, weight_sum


def _build_low_confidence_mask(
    surface_mask: NDArray[np.uint8],
    coverage_count: NDArray[np.float32],
    best_facing: NDArray[np.float32],
) -> NDArray[np.uint8]:
    repair_mask = np.zeros(surface_mask.shape, dtype=np.uint8)
    valid_surface = surface_mask > 0

    covered_surface = valid_surface & (coverage_count > 0)
    repair_mask[valid_surface & ~covered_surface] = 255

    if np.any(covered_surface):
        covered_best_facing = best_facing[covered_surface]
        threshold = min(
            LOW_CONFIDENCE_FACING_CEILING,
            max(
                LOW_CONFIDENCE_FACING_FLOOR,
                float(np.percentile(covered_best_facing, 25)),
            ),
        )
        repair_mask[covered_surface & (best_facing <= threshold)] = 255

    return repair_mask.astype(np.uint8, copy=False)


def _compute_seam_repair_data(
    uv_assets: UVPassAssets,
) -> _SeamRepairData:
    surface_mask = uv_assets.surface_mask.astype(np.uint8)
    texture_resolution = int(surface_mask.shape[0])
    seam_link_source_yx = getattr(uv_assets, "seam_link_source_yx", None)
    seam_link_target_yx = getattr(uv_assets, "seam_link_target_yx", None)

    if _seam_link_arrays_are_valid(seam_link_source_yx, seam_link_target_yx):
        source_yx = cast("NDArray[np.int32]", seam_link_source_yx)
        target_yx = cast("NDArray[np.int32]", seam_link_target_yx)
        link_mask = seam_link_mask(
            source_yx,
            target_yx,
            surface_mask.shape,
        )
        seam_line_mask = getattr(uv_assets, "seam_line_mask", None)
        if seam_line_mask is None:
            seam_boundary = link_mask
        else:
            seam_boundary = np.maximum(seam_line_mask.astype(np.uint8), link_mask)
        seam_boundary[surface_mask == 0] = 0

        topology_groups = seam_link_components(
            source_yx.astype(np.int32, copy=False),
            target_yx.astype(np.int32, copy=False),
        )
        fallback_groups: list[NDArray[np.int32]] = []
        seam_band = seam_boundary.copy()
        unresolved_mask = getattr(uv_assets, "seam_unresolved_link_mask", None)
        if unresolved_mask is not None and np.any(unresolved_mask):
            fallback_band = _expand_seam_mask(
                unresolved_mask.astype(np.uint8),
                surface_mask,
                radius=SEAM_SEARCH_RADIUS,
            )
            fallback_groups = find_position_seam_groups(
                position_map=uv_assets.position_map,
                normal_map=uv_assets.normal_map,
                seam_band=fallback_band,
                surface_mask=surface_mask,
                texture_resolution=texture_resolution,
            )
            seam_band = np.maximum(seam_band, fallback_band)

        return _SeamRepairData(
            seam_groups=topology_groups + fallback_groups,
            topology_groups=topology_groups,
            fallback_groups=fallback_groups,
            seam_boundary=seam_boundary.astype(np.uint8, copy=False),
            seam_band=seam_band.astype(np.uint8, copy=False),
            seam_link_source_yx=source_yx.astype(np.int32, copy=False),
            seam_link_target_yx=target_yx.astype(np.int32, copy=False),
        )

    seam_boundary = _make_seam_boundary_mask(uv_assets.uv_layout, surface_mask)
    seam_search_mask = _expand_seam_mask(
        seam_boundary,
        surface_mask,
        radius=SEAM_SEARCH_RADIUS,
    )
    seam_groups = find_position_seam_groups(
        position_map=uv_assets.position_map,
        normal_map=uv_assets.normal_map,
        seam_band=seam_search_mask,
        surface_mask=surface_mask,
        texture_resolution=texture_resolution,
    )
    return _SeamRepairData(
        seam_groups=seam_groups,
        topology_groups=[],
        fallback_groups=seam_groups,
        seam_boundary=seam_boundary,
        seam_band=seam_search_mask,
        seam_link_source_yx=None,
        seam_link_target_yx=None,
    )


def _build_seam_mismatch_mask(
    texture: NDArray[np.uint8],
    seam_groups: list[NDArray[np.int32]],
    seam_band: NDArray[np.uint8],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    mismatch_mask = np.zeros(surface_mask.shape, dtype=np.uint8)
    low_frequency, _high_frequency = _split_texture_frequencies(texture)

    for seam_group in seam_groups:
        group_colors = low_frequency[seam_group[:, 0], seam_group[:, 1]].astype(
            np.float32,
        )
        mean_color = np.mean(group_colors, axis=0)
        color_error = np.linalg.norm(group_colors - mean_color, axis=1)
        if float(np.max(color_error)) >= SEAM_MISMATCH_THRESHOLD:
            mismatch_mask[seam_group[:, 0], seam_group[:, 1]] = 255

    mismatch_mask[(seam_band == 0) | (surface_mask == 0)] = 0
    return mismatch_mask.astype(np.uint8, copy=False)


def _build_topology_seam_mismatch_mask(
    texture: NDArray[np.uint8],
    seam_link_source_yx: NDArray[np.int32],
    seam_link_target_yx: NDArray[np.int32],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    mismatch_mask = np.zeros(surface_mask.shape, dtype=np.uint8)
    low_frequency, _high_frequency = _split_texture_frequencies(texture)

    source_colors = low_frequency[
        seam_link_source_yx[:, 0],
        seam_link_source_yx[:, 1],
    ].astype(np.float32)
    target_colors = low_frequency[
        seam_link_target_yx[:, 0],
        seam_link_target_yx[:, 1],
    ].astype(np.float32)
    color_error = np.linalg.norm(source_colors - target_colors, axis=1)
    mismatch_links = color_error >= SEAM_MISMATCH_THRESHOLD
    if np.any(mismatch_links):
        mismatch_sources = seam_link_source_yx[mismatch_links]
        mismatch_targets = seam_link_target_yx[mismatch_links]
        mismatch_mask[mismatch_sources[:, 0], mismatch_sources[:, 1]] = 255
        mismatch_mask[mismatch_targets[:, 0], mismatch_targets[:, 1]] = 255

    mismatch_mask[surface_mask == 0] = 0
    return mismatch_mask.astype(np.uint8, copy=False)


def _couple_mask_across_seams(
    repair_mask: NDArray[np.uint8],
    seam_groups: list[NDArray[np.int32]],
) -> NDArray[np.uint8]:
    coupled_mask = repair_mask.copy()
    for seam_group in seam_groups:
        if np.any(coupled_mask[seam_group[:, 0], seam_group[:, 1]] > 0):
            coupled_mask[seam_group[:, 0], seam_group[:, 1]] = 255
    return coupled_mask.astype(np.uint8, copy=False)


def _couple_mask_across_topology_links(
    repair_mask: NDArray[np.uint8],
    seam_link_source_yx: NDArray[np.int32],
    seam_link_target_yx: NDArray[np.int32],
) -> NDArray[np.uint8]:
    coupled_mask = repair_mask.copy()
    source_active = coupled_mask[
        seam_link_source_yx[:, 0],
        seam_link_source_yx[:, 1],
    ] > 0
    target_active = coupled_mask[
        seam_link_target_yx[:, 0],
        seam_link_target_yx[:, 1],
    ] > 0
    active_links = source_active | target_active
    if np.any(active_links):
        active_sources = seam_link_source_yx[active_links]
        active_targets = seam_link_target_yx[active_links]
        coupled_mask[active_sources[:, 0], active_sources[:, 1]] = 255
        coupled_mask[active_targets[:, 0], active_targets[:, 1]] = 255
    return coupled_mask.astype(np.uint8, copy=False)


def build_uv_repair_mask(
    texture: NDArray[np.uint8],
    uv_assets: UVPassAssets,
    multiview_images: dict[str, list[NDArray[Any]]],
) -> tuple[NDArray[np.uint8], list[NDArray[np.int32]], NDArray[np.uint8]]:
    texture_resolution = int(texture.shape[0])
    surface_mask = uv_assets.surface_mask.astype(np.uint8)
    coverage_count, best_facing, _weight_sum = accumulate_uv_repair_diagnostics(
        multiview_images,
        texture_resolution,
    )
    seam_data = _compute_seam_repair_data(uv_assets)

    repair_mask = _build_low_confidence_mask(surface_mask, coverage_count, best_facing)
    if (
        seam_data.seam_link_source_yx is not None
        and seam_data.seam_link_target_yx is not None
        and len(seam_data.seam_link_source_yx) > 0
    ):
        seam_mismatch_mask = _build_topology_seam_mismatch_mask(
            texture,
            seam_data.seam_link_source_yx,
            seam_data.seam_link_target_yx,
            surface_mask,
        )
        repair_mask = np.maximum(repair_mask, seam_mismatch_mask)
        repair_mask = _couple_mask_across_topology_links(
            repair_mask,
            seam_data.seam_link_source_yx,
            seam_data.seam_link_target_yx,
        )

    if seam_data.fallback_groups:
        seam_mismatch_mask = _build_seam_mismatch_mask(
            texture,
            seam_data.fallback_groups,
            seam_data.seam_band,
            surface_mask,
        )
        repair_mask = np.maximum(repair_mask, seam_mismatch_mask)
        repair_mask = _couple_mask_across_seams(repair_mask, seam_data.fallback_groups)

    repair_mask = _expand_seam_mask(
        repair_mask,
        surface_mask,
        radius=max(1, texture_resolution // 512),
    )
    repair_mask[surface_mask == 0] = 0
    repair_mask = cv2.dilate(
        repair_mask,
        np.ones((3, 3), dtype=np.uint8),
        iterations=1,
    )
    repair_mask[surface_mask == 0] = 0
    return (
        repair_mask.astype(np.uint8, copy=False),
        seam_data.seam_groups,
        seam_data.seam_boundary,
    )


def build_chart_height_map(
    normal_map: NDArray[np.float32],
    position_map: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.float32]:
    normals = _decode_baked_normal_vectors(normal_map)
    position = position_map[..., :3].astype(np.float32)
    valid_mask = (
        (surface_mask > 0)
        & np.all(np.isfinite(position), axis=-1)
        & np.all(np.isfinite(normals), axis=-1)
    )
    height_map = np.full(surface_mask.shape, 0.5, dtype=np.float32)
    if not np.any(valid_mask):
        return height_map

    num_labels, labels = cv2.connectedComponents(
        valid_mask.astype(np.uint8),
        connectivity=4,
    )
    signed_height = np.zeros(surface_mask.shape, dtype=np.float32)

    for label in range(1, num_labels):
        label_mask = labels == label
        if not np.any(label_mask):
            continue

        _x_axis, _y_axis, z_axis = _build_chart_frame(label_mask, position, normals)
        chart_height = position[label_mask] @ z_axis
        chart_height -= float(np.mean(chart_height))
        signed_height[label_mask] = chart_height.astype(np.float32)

    valid_height = np.abs(signed_height[valid_mask])
    scale = float(np.percentile(valid_height, 95)) if valid_height.size > 0 else 0.0
    if scale > 1e-6:  # noqa: PLR2004
        height_map[valid_mask] = np.clip(
            0.5 + (0.5 * signed_height[valid_mask] / scale),
            0.0,
            1.0,
        )

    return height_map.astype(np.float32, copy=False)


def _build_repair_canny_guidance(
    texture: NDArray[np.uint8],
    canonical_normal_map: NDArray[np.float32],
    height_map: NDArray[np.float32],
    seam_boundary: NDArray[np.uint8],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    texture_gray = cv2.cvtColor(texture, cv2.COLOR_RGB2GRAY)
    texture_edges = cv2.Canny(texture_gray, TEXTURE_EDGE_LOWER, TEXTURE_EDGE_UPPER)

    normal_uint8 = np.clip(np.rint(canonical_normal_map * 255.0), 0, 255).astype(
        np.uint8,
    )
    normal_gray = cv2.cvtColor(normal_uint8, cv2.COLOR_RGB2GRAY)
    normal_edges = cv2.Canny(normal_gray, GEOMETRY_EDGE_LOWER, GEOMETRY_EDGE_UPPER)

    height_uint8 = np.clip(np.rint(height_map * 255.0), 0, 255).astype(np.uint8)
    height_edges = cv2.Canny(height_uint8, GEOMETRY_EDGE_LOWER, GEOMETRY_EDGE_UPPER)

    combined = np.maximum(texture_edges, normal_edges)
    combined = np.maximum(combined, height_edges)
    combined = np.maximum(combined, seam_boundary)
    combined[surface_mask == 0] = 0
    return np.stack((combined, combined, combined), axis=-1)


def uv_repair_pass(  # noqa: PLR0913
    texture: NDArray[np.uint8] | NDArray[np.float32],
    uv_assets: UVPassAssets,
    multiview_images: dict[str, list[NDArray[Any]]],
    process_parameter: ProcessParameter,
    progress_callback: Callable[[int], None],
    should_cancel: Callable[[], bool] = lambda: False,
) -> NDArray[np.uint8]:
    """Run an automatic UV-space repair pass after image-based generation."""
    _require_cv2()

    if Image is None:
        msg = "Pillow is not available."
        raise RuntimeError(msg)

    if should_cancel():
        msg = "Cancelled by user."
        raise TextureGenerationCancelledError(msg)

    texture_resolution = int(process_parameter.texture_resolution)
    sd_resolution = get_default_sd_resolution(
        process_parameter.sd_version,
        process_parameter.custom_sd_resolution,
    )
    surface_mask = uv_assets.surface_mask.astype(np.uint8)
    base_texture = _prepare_base_texture(texture, texture_resolution)
    base_texture[surface_mask == 0] = 255

    repair_mask, seam_groups, seam_boundary = build_uv_repair_mask(
        texture=base_texture,
        uv_assets=uv_assets,
        multiview_images=multiview_images,
    )
    if not np.any(repair_mask):
        progress_callback(100)
        return base_texture

    working_texture = base_texture
    if seam_groups:
        working_texture = apply_position_seam_stitching(
            texture=base_texture,
            position_map=uv_assets.position_map,
            normal_map=uv_assets.normal_map,
            uv_layout=uv_assets.uv_layout,
            surface_mask=surface_mask,
            seam_line_mask=getattr(uv_assets, "seam_line_mask", None),
            seam_link_source_yx=getattr(uv_assets, "seam_link_source_yx", None),
            seam_link_target_yx=getattr(uv_assets, "seam_link_target_yx", None),
            seam_unresolved_link_mask=getattr(
                uv_assets,
                "seam_unresolved_link_mask",
                None,
            ),
        )

    canonical_normal_map = canonicalize_uv_normal_map(
        normal_map=uv_assets.normal_map,
        position_map=uv_assets.position_map,
        surface_mask=surface_mask,
    )
    height_map = build_chart_height_map(
        normal_map=uv_assets.normal_map,
        position_map=uv_assets.position_map,
        surface_mask=surface_mask,
    )
    canny_map = _build_repair_canny_guidance(
        texture=working_texture,
        canonical_normal_map=canonical_normal_map,
        height_map=height_map,
        seam_boundary=seam_boundary,
        surface_mask=surface_mask,
    )

    input_small = cv2.resize(
        working_texture,
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_LANCZOS4,
    )
    mask_small = cv2.resize(
        repair_mask,
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_NEAREST,
    )
    canny_small = cv2.resize(
        canny_map,
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_NEAREST,
    )
    normal_small = cv2.resize(
        np.clip(np.rint(canonical_normal_map * 255.0), 0, 255).astype(np.uint8),
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_LINEAR,
    )
    height_rgb = np.stack((height_map, height_map, height_map), axis=-1)
    depth_small = cv2.resize(
        np.clip(np.rint(height_rgb * 255.0), 0, 255).astype(np.uint8),
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_LINEAR,
    )

    pipeline = create_diffusion_pipeline(process_parameter)

    def repair_progress_callback(percent: int) -> None:
        progress_callback(percent)

    result = run_pipeline(
        pipe=pipeline,
        process_parameter=process_parameter,
        input_img=Image.fromarray(input_small),
        uv_mask=Image.fromarray(mask_small),
        canny_img=Image.fromarray(canny_small),
        normal_img=Image.fromarray(normal_small),
        depth_img=Image.fromarray(depth_small),
        progress_callback=repair_progress_callback,
        should_cancel=should_cancel,
        strength=min(process_parameter.denoise_strength, AUTO_REPAIR_MAX_STRENGTH),
        guidance_scale=process_parameter.guidance_scale or 7.5,
        num_inference_steps=process_parameter.num_inference_steps,
    )

    if should_cancel():
        msg = "Cancelled by user."
        raise TextureGenerationCancelledError(msg)

    repaired_small = np.array(result)[..., :3]
    repaired_texture = cv2.resize(
        repaired_small,
        (texture_resolution, texture_resolution),
        interpolation=cv2.INTER_LANCZOS4,
    ).astype(np.uint8)
    repaired_texture[surface_mask == 0] = base_texture[surface_mask == 0]

    repair_alpha = smooth_alpha_map(repair_mask.astype(np.float32) / 255.0)
    repair_alpha *= (repair_mask > 0).astype(np.float32)
    repair_alpha *= (surface_mask > 0).astype(np.float32)

    repaired = blend_texture_with_alpha(base_texture, repaired_texture, repair_alpha)
    if seam_groups:
        repaired = apply_position_seam_stitching(
            texture=repaired,
            position_map=uv_assets.position_map,
            normal_map=uv_assets.normal_map,
            uv_layout=uv_assets.uv_layout,
            surface_mask=surface_mask,
            seam_line_mask=getattr(uv_assets, "seam_line_mask", None),
            seam_link_source_yx=getattr(uv_assets, "seam_link_source_yx", None),
            seam_link_target_yx=getattr(uv_assets, "seam_link_target_yx", None),
            seam_unresolved_link_mask=getattr(
                uv_assets,
                "seam_unresolved_link_mask",
                None,
            ),
        )

    repaired[surface_mask == 0] = base_texture[surface_mask == 0]
    progress_callback(100)
    return repaired.astype(np.uint8, copy=False)
