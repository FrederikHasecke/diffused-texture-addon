from collections.abc import Callable, Iterable

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


from .process_operations import (
    _require_cv2,
    blend_texture_with_alpha,
    smooth_alpha_map,
)

NORMAL_EPSILON = 1e-8
UV_LAYOUT_BOUNDARY_THRESHOLD = 192
SEAM_SEARCH_RADIUS = 1
LOW_FREQUENCY_MIN_KERNEL = 3
FALLBACK_AXIS_ALIGNMENT_LIMIT = 0.9


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        parent = self.parents[index]
        while parent != self.parents[parent]:
            parent = self.parents[parent]
        while index != parent:
            next_index = self.parents[index]
            self.parents[index] = parent
            index = next_index
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


def _normalize_uv_layout_uint8(uv_layout: NDArray[np.float32]) -> NDArray[np.uint8]:
    return np.clip(255.0 * uv_layout, 0, 255).astype(np.uint8)


def _normalize_baked_normal_map(normal_map: NDArray[np.float32]) -> NDArray[np.float32]:
    return _encode_controlnet_normal_map(_decode_baked_normal_vectors(normal_map))


def _decode_baked_normal_vectors(
    normal_map: NDArray[np.float32],
) -> NDArray[np.float32]:
    normal = normal_map[..., :3].astype(np.float32)
    if normal.size == 0:
        return normal

    if float(np.nanmin(normal)) >= 0.0 and float(np.nanmax(normal)) <= 1.0:
        normal = normal * 2.0 - 1.0

    norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    return np.divide(
        normal,
        norm,
        out=np.zeros_like(normal),
        where=norm > NORMAL_EPSILON,
    )


def _encode_controlnet_normal_map(
    normal_vectors: NDArray[np.float32],
) -> NDArray[np.float32]:
    if normal_vectors.size == 0:
        return normal_vectors.astype(np.float32, copy=False)

    encoded = np.empty_like(normal_vectors, dtype=np.float32)
    encoded[..., 0] = np.clip((normal_vectors[..., 0] + 1.0) / 2.0, 0.0, 1.0)
    encoded[..., 1] = np.clip((1.0 - normal_vectors[..., 1]) / 2.0, 0.0, 1.0)
    encoded[..., 2] = np.clip((1.0 - normal_vectors[..., 2]) / 2.0, 0.0, 1.0)
    return encoded.astype(np.float32, copy=False)


def _normalize_vectors(vectors: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(
        vectors,
        norm,
        out=np.zeros_like(vectors),
        where=norm > NORMAL_EPSILON,
    )


def _normalize_vector(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    return _normalize_vectors(vector[np.newaxis, :])[0]


def _choose_fallback_axis(direction: NDArray[np.float32]) -> NDArray[np.float32]:
    if abs(float(direction[0])) < FALLBACK_AXIS_ALIGNMENT_LIMIT:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return np.array([0.0, 1.0, 0.0], dtype=np.float32)


def _build_chart_frame(  # noqa: C901
    label_mask: NDArray[np.bool_],
    position: NDArray[np.float32],
    normals: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    horizontal_pairs = label_mask[:, 1:] & label_mask[:, :-1]
    vertical_pairs = label_mask[1:, :] & label_mask[:-1, :]

    horizontal_deltas = position[:, 1:, :] - position[:, :-1, :]
    vertical_deltas = position[1:, :, :] - position[:-1, :, :]

    x_raw = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    y_raw = np.array([0.0, 0.0, 0.0], dtype=np.float32)

    if np.any(horizontal_pairs):
        x_raw = -np.mean(horizontal_deltas[horizontal_pairs], axis=0).astype(np.float32)
    if np.any(vertical_pairs):
        y_raw = np.mean(vertical_deltas[vertical_pairs], axis=0).astype(np.float32)

    mean_normal = _normalize_vector(
        np.mean(normals[label_mask], axis=0).astype(np.float32),
    )
    if not np.any(mean_normal):
        mean_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    if not np.any(x_raw):
        fallback_axis = _choose_fallback_axis(mean_normal)
        x_raw = np.cross(fallback_axis, -mean_normal).astype(np.float32)
    x_axis = _normalize_vector(x_raw.astype(np.float32))
    if not np.any(x_axis):
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    if not np.any(y_raw):
        y_raw = np.cross(-mean_normal, x_axis).astype(np.float32)

    z_axis = _normalize_vector(np.cross(x_axis, y_raw).astype(np.float32))
    if not np.any(z_axis):
        z_axis = -mean_normal
    if float(np.dot(z_axis, mean_normal)) > 0.0:
        z_axis = -z_axis

    x_axis = _normalize_vector(
        (x_axis - (float(np.dot(x_axis, z_axis)) * z_axis)).astype(np.float32),
    )
    if not np.any(x_axis):
        fallback_axis = _choose_fallback_axis(z_axis)
        x_axis = _normalize_vector(np.cross(fallback_axis, z_axis).astype(np.float32))

    y_axis = _normalize_vector(np.cross(z_axis, x_axis).astype(np.float32))
    if float(np.dot(y_axis, y_raw)) < 0.0:
        x_axis = -x_axis
        y_axis = -y_axis

    return x_axis, y_axis, z_axis


def canonicalize_uv_normal_map(
    normal_map: NDArray[np.float32],
    position_map: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.float32]:
    normals = _decode_baked_normal_vectors(normal_map)
    position = position_map[..., :3].astype(np.float32)
    encoded = _normalize_baked_normal_map(normal_map)

    valid_mask = (
        (surface_mask > 0)
        & np.all(np.isfinite(position), axis=-1)
        & np.all(np.isfinite(normals), axis=-1)
    )
    if not np.any(valid_mask):
        return encoded

    num_labels, labels = cv2.connectedComponents(
        valid_mask.astype(np.uint8),
        connectivity=4,
    )
    canonical = np.zeros_like(normals, dtype=np.float32)

    for label in range(1, num_labels):
        label_mask = labels == label
        if not np.any(label_mask):
            continue

        x_axis, y_axis, z_axis = _build_chart_frame(label_mask, position, normals)
        chart_normals = normals[label_mask]
        canonical[label_mask, 0] = chart_normals @ x_axis
        canonical[label_mask, 1] = chart_normals @ y_axis
        canonical[label_mask, 2] = chart_normals @ z_axis

    encoded[valid_mask] = _encode_controlnet_normal_map(
        _normalize_vectors(canonical[valid_mask]),
    )
    return encoded.astype(np.float32, copy=False)


def _create_uv_edge_guidance(
    uv_layout: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    layout_uint8 = _normalize_uv_layout_uint8(uv_layout)
    canny = np.full(surface_mask.shape, 255, dtype=np.uint8)
    alpha_mask = layout_uint8[..., 3] > 0
    canny[alpha_mask] = layout_uint8[..., 0][alpha_mask]
    canny[canny < 64] = 0  # noqa: PLR2004
    canny[canny >= 192] = 255  # noqa: PLR2004
    canny = 255 - canny
    canny[surface_mask == 0] = 255
    return np.stack((canny, canny, canny), axis=-1)


def _prepare_base_texture(
    texture: NDArray[np.float32] | None,
    texture_resolution: int,
) -> NDArray[np.uint8]:
    if texture is None:
        return np.full((texture_resolution, texture_resolution, 3), 255, dtype=np.uint8)

    texture_rgb = texture[..., :3]
    if texture_rgb.dtype != np.uint8:
        texture_rgb = np.clip(texture_rgb, 0.0, 1.0)
        texture_rgb = (255.0 * texture_rgb).astype(np.uint8)

    if texture_rgb.shape[:2] != (texture_resolution, texture_resolution):
        _require_cv2()
        texture_rgb = cv2.resize(
            texture_rgb,
            (texture_resolution, texture_resolution),
            interpolation=cv2.INTER_LANCZOS4,
        )

    return texture_rgb.astype(np.uint8, copy=False)


def _make_seam_boundary_mask(
    uv_layout: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    layout_uint8 = _normalize_uv_layout_uint8(uv_layout)
    boundary = np.zeros(surface_mask.shape, dtype=np.uint8)
    boundary[
        (surface_mask > 0)
        & (layout_uint8[..., 3] > 0)
        & (layout_uint8[..., 0] < UV_LAYOUT_BOUNDARY_THRESHOLD)
    ] = 255

    if not np.any(boundary):
        boundary = cv2.morphologyEx(
            surface_mask,
            cv2.MORPH_GRADIENT,
            np.ones((3, 3), dtype=np.uint8),
        )

    boundary[surface_mask == 0] = 0
    return boundary.astype(np.uint8, copy=False)


def _expand_seam_mask(
    seam_mask: NDArray[np.uint8],
    surface_mask: NDArray[np.uint8],
    radius: int,
) -> NDArray[np.uint8]:
    if radius <= 0:
        expanded_mask = seam_mask.copy()
    else:
        kernel_size = (2 * radius) + 1
        expanded_mask = cv2.dilate(
            seam_mask,
            np.ones((kernel_size, kernel_size), dtype=np.uint8),
            iterations=1,
        )

    expanded_mask[surface_mask == 0] = 0
    return expanded_mask.astype(np.uint8, copy=False)


def _low_frequency_kernel_size(texture_resolution: int) -> int:
    base_radius = max(1, texture_resolution // 512)
    return max(LOW_FREQUENCY_MIN_KERNEL, (2 * base_radius) + 1)


def _split_texture_frequencies(
    texture: NDArray[np.uint8],
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    kernel_size = _low_frequency_kernel_size(int(texture.shape[0]))
    texture_float = texture.astype(np.float32)
    low_frequency = cv2.GaussianBlur(
        texture_float,
        (kernel_size, kernel_size),
        0,
    ).astype(np.float32)
    high_frequency = (texture_float - low_frequency).astype(np.float32)
    return low_frequency, high_frequency


def _iter_neighbor_voxel_keys(
    voxel_key: tuple[int, int, int],
) -> Iterable[tuple[int, int, int]]:
    base_x, base_y, base_z = voxel_key
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                yield (base_x + dx, base_y + dy, base_z + dz)


def compute_adaptive_position_threshold(
    position_map: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> float:
    position = position_map[..., :3].astype(np.float32)
    valid = (surface_mask > 0) & np.all(np.isfinite(position), axis=-1)

    deltas: list[NDArray[np.float32]] = []

    vertical_pairs = valid[:-1, :] & valid[1:, :]
    if np.any(vertical_pairs):
        vertical_delta = np.linalg.norm(position[1:, :] - position[:-1, :], axis=-1)
        deltas.append(vertical_delta[vertical_pairs].astype(np.float32))

    horizontal_pairs = valid[:, :-1] & valid[:, 1:]
    if np.any(horizontal_pairs):
        horizontal_delta = np.linalg.norm(
            position[:, 1:] - position[:, :-1],
            axis=-1,
        )
        deltas.append(horizontal_delta[horizontal_pairs].astype(np.float32))

    if deltas:
        valid_deltas = np.concatenate(deltas)
        valid_deltas = valid_deltas[np.isfinite(valid_deltas) & (valid_deltas > 0)]
        if valid_deltas.size > 0:
            return float(np.median(valid_deltas) * 2.5)

    valid_positions = position[valid]
    if valid_positions.size == 0:
        return 1e-4

    fallback_scale = np.linalg.norm(valid_positions, axis=-1)
    fallback_mean = float(np.nanmean(fallback_scale))
    return max(fallback_mean * 1e-4, 1e-4)


def find_position_seam_groups(  # noqa: C901, PLR0913
    position_map: NDArray[np.float32],
    normal_map: NDArray[np.float32],
    seam_band: NDArray[np.uint8],
    surface_mask: NDArray[np.uint8],
    texture_resolution: int,
    min_normal_cosine: float = 0.85,
) -> list[NDArray[np.int32]]:
    position = position_map[..., :3].astype(np.float32)
    normals = _decode_baked_normal_vectors(normal_map)
    valid = (
        (seam_band > 0)
        & (surface_mask > 0)
        & np.all(np.isfinite(position), axis=-1)
        & np.all(np.isfinite(normals), axis=-1)
    )

    coords = np.argwhere(valid)
    if len(coords) < 2:  # noqa: PLR2004
        return []

    match_threshold = compute_adaptive_position_threshold(position_map, surface_mask)
    match_threshold_squared = match_threshold * match_threshold
    min_uv_distance = max(8, texture_resolution // 128)
    min_uv_distance_squared = min_uv_distance * min_uv_distance
    positions = position[coords[:, 0], coords[:, 1]]
    normal_vectors = normals[coords[:, 0], coords[:, 1]]
    voxel_coords = np.floor(positions / match_threshold).astype(np.int32)

    uf = _UnionFind(len(coords))
    voxel_map: dict[tuple[int, int, int], list[int]] = {}

    for idx, coord in enumerate(coords):
        position_value = positions[idx]
        normal_value = normal_vectors[idx]
        coord_y = int(coord[0])
        coord_x = int(coord[1])
        voxel_x, voxel_y, voxel_z = (int(value) for value in voxel_coords[idx])
        voxel_key = (voxel_x, voxel_y, voxel_z)

        for neighbor_key in _iter_neighbor_voxel_keys(voxel_key):
            for other_idx in voxel_map.get(neighbor_key, []):
                other_coord = coords[other_idx]
                uv_delta_x = coord_x - int(other_coord[1])
                uv_delta_y = coord_y - int(other_coord[0])
                uv_distance_squared = (uv_delta_x * uv_delta_x) + (
                    uv_delta_y * uv_delta_y
                )
                if uv_distance_squared < min_uv_distance_squared:
                    continue

                normal_similarity = float(
                    np.dot(normal_value, normal_vectors[other_idx]),
                )
                if normal_similarity < min_normal_cosine:
                    continue

                position_delta = position_value - positions[other_idx]
                position_distance_squared = float(
                    np.dot(position_delta, position_delta),
                )
                if position_distance_squared > match_threshold_squared:
                    continue

                uf.union(idx, other_idx)

        bucket = voxel_map.get(voxel_key)
        if bucket is None:
            bucket = []
            voxel_map[voxel_key] = bucket
        bucket.append(idx)

    grouped_indices: dict[int, list[int]] = {}
    for idx in range(len(coords)):
        grouped_indices.setdefault(uf.find(idx), []).append(idx)

    seam_groups: list[NDArray[np.int32]] = []
    for members in grouped_indices.values():
        if len(members) < 2:  # noqa: PLR2004
            continue
        seam_groups.append(coords[members].astype(np.int32))

    return seam_groups


def apply_position_seam_stitching(
    texture: NDArray[np.uint8],
    position_map: NDArray[np.float32],
    normal_map: NDArray[np.float32],
    uv_layout: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    _require_cv2()

    texture_resolution = int(texture.shape[0])
    seam_boundary = _make_seam_boundary_mask(uv_layout, surface_mask)
    seam_search_mask = _expand_seam_mask(
        seam_boundary,
        surface_mask,
        radius=SEAM_SEARCH_RADIUS,
    )
    seam_blend_mask = _expand_seam_mask(
        seam_boundary,
        surface_mask,
        radius=max(1, texture_resolution // 512),
    )
    seam_groups = find_position_seam_groups(
        position_map=position_map,
        normal_map=normal_map,
        seam_band=seam_search_mask,
        surface_mask=surface_mask,
        texture_resolution=texture_resolution,
    )
    if not seam_groups:
        return texture

    low_frequency, high_frequency = _split_texture_frequencies(texture)
    anchor_mask = np.zeros(surface_mask.shape, dtype=np.float32)
    anchor_deltas = np.zeros(texture.shape, dtype=np.float32)

    for seam_group in seam_groups:
        group_colors = low_frequency[seam_group[:, 0], seam_group[:, 1]].astype(
            np.float32,
        )
        mean_color = np.mean(group_colors, axis=0)
        anchor_mask[seam_group[:, 0], seam_group[:, 1]] = 1.0
        anchor_deltas[seam_group[:, 0], seam_group[:, 1]] = mean_color - group_colors

    band_radius = max(1, texture_resolution // 512)
    kernel_size = (2 * band_radius) + 1
    expanded_mask = cv2.dilate(
        (255 * anchor_mask).astype(np.uint8),
        np.ones((kernel_size, kernel_size), dtype=np.uint8),
        iterations=1,
    )
    expanded_mask[(seam_blend_mask == 0) | (surface_mask == 0)] = 0

    weights = cv2.GaussianBlur(
        anchor_mask.astype(np.float32),
        (kernel_size, kernel_size),
        0,
    )
    correction_field = np.zeros_like(low_frequency, dtype=np.float32)
    for channel in range(3):
        blurred_delta = cv2.GaussianBlur(
            (anchor_deltas[..., channel] * anchor_mask).astype(np.float32),
            (kernel_size, kernel_size),
            0,
        )
        correction_field[..., channel] = np.divide(
            blurred_delta,
            weights,
            out=np.zeros_like(low_frequency[..., channel]),
            where=weights > 1e-6,  # noqa: PLR2004
        )

    alpha_map = smooth_alpha_map(expanded_mask.astype(np.float32) / 255.0)
    alpha_map *= (expanded_mask > 0).astype(np.float32)
    alpha_map *= (surface_mask > 0).astype(np.float32)

    corrected_low = np.clip(low_frequency + correction_field, 0.0, 255.0)
    corrected_texture = np.clip(
        np.rint(corrected_low + high_frequency),
        0,
        255,
    ).astype(np.uint8)

    stitched = blend_texture_with_alpha(
        texture,
        corrected_texture,
        alpha_map,
    )
    stitched[surface_mask == 0] = texture[surface_mask == 0]
    return stitched


def uv_pass(
    uv_assets: UVPassAssets,
    process_parameter: ProcessParameter,
    progress_callback: Callable[[int], None],
    should_cancel: Callable[[], bool] = lambda: False,
    texture: NDArray[np.float32] | None = None,
) -> NDArray[np.uint8]:
    """Run the dedicated UV-space generation mode."""
    _require_cv2()

    if Image is None:
        msg = "Pillow is not available."
        raise RuntimeError(msg)

    texture_resolution = int(process_parameter.texture_resolution)
    sd_resolution = get_default_sd_resolution(
        process_parameter.sd_version,
        process_parameter.custom_sd_resolution,
    )

    surface_mask = uv_assets.surface_mask.astype(np.uint8)
    mask = cv2.dilate(surface_mask, np.ones((3, 3), np.uint8), iterations=2)
    canny = _create_uv_edge_guidance(uv_assets.uv_layout, surface_mask)
    normal_map = canonicalize_uv_normal_map(
        normal_map=uv_assets.normal_map,
        position_map=uv_assets.position_map,
        surface_mask=surface_mask,
    )
    base_texture = _prepare_base_texture(texture, texture_resolution)
    base_texture[surface_mask == 0] = 255

    progress_callback(5)

    resize_shape = (sd_resolution, sd_resolution)
    input_small = cv2.resize(
        base_texture,
        resize_shape,
        interpolation=cv2.INTER_LANCZOS4,
    )
    mask_small = cv2.resize(mask, resize_shape, interpolation=cv2.INTER_NEAREST)
    canny_small = cv2.resize(canny, resize_shape, interpolation=cv2.INTER_NEAREST)
    normal_small = cv2.resize(
        (255.0 * normal_map).astype(np.uint8),
        resize_shape,
        interpolation=cv2.INTER_LINEAR,
    )
    depth_small = np.zeros_like(canny_small, dtype=np.uint8)

    progress_callback(10)

    if should_cancel():
        msg = "Cancelled by user."
        raise TextureGenerationCancelledError(msg)

    pipeline = create_diffusion_pipeline(process_parameter)
    output_image = run_pipeline(
        pipe=pipeline,
        process_parameter=process_parameter,
        input_img=Image.fromarray(input_small),
        uv_mask=Image.fromarray(mask_small),
        canny_img=Image.fromarray(canny_small),
        normal_img=Image.fromarray(normal_small),
        depth_img=Image.fromarray(depth_small),
        progress_callback=progress_callback,
        should_cancel=should_cancel,
        strength=process_parameter.denoise_strength,
        guidance_scale=process_parameter.guidance_scale or 7.5,
        num_inference_steps=process_parameter.num_inference_steps,
    )

    if should_cancel():
        msg = "Cancelled by user."
        raise TextureGenerationCancelledError(msg)

    output_texture = cv2.resize(
        np.array(output_image)[..., :3],
        (texture_resolution, texture_resolution),
        interpolation=cv2.INTER_LANCZOS4,
    ).astype(np.uint8)
    output_texture[surface_mask == 0] = base_texture[surface_mask == 0]

    stitched = apply_position_seam_stitching(
        texture=output_texture,
        position_map=uv_assets.position_map,
        normal_map=uv_assets.normal_map,
        uv_layout=uv_assets.uv_layout,
        surface_mask=surface_mask,
    )
    stitched[surface_mask == 0] = base_texture[surface_mask == 0]
    return stitched
