import math

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from collections.abc import Callable
from typing import cast

import numpy as np
from numpy.typing import NDArray

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from ..blender_operations import ProcessParameter
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
    assemble_multiview_subgrid,
    blend_texture_with_alpha,
    inpaint_missing,
    process_uv_texture,
    smooth_alpha_map,
)


def img_parasequential(  # noqa: PLR0913, PLR0915
    multiview_images: dict[str, list[NDArray]],
    process_parameter: ProcessParameter,
    progress_callback: Callable,
    should_cancel: Callable[[], bool] = lambda: False,
    texture: NDArray[np.float32] | None = None,
    subgrid_rows: int = 2,
    subgrid_cols: int = 2,
    facing_percentile: float = 0.5,
) -> NDArray[np.uint8]:
    """Hybrid multiview processing.

    - Split the full set of views into r x c sub-grids (parallel within a sub-grid).
    - After each sub-grid inference, project only its visible UVs back to the texture
        (sequential across sub-grids).

    Args:
        multiview_images: Dict with lists for keys ["depth","normal","facing","uv"].
        process_parameter: Processing parameters.
        progress_callback: Function to track progress (expects int 0..100).
        should_cancel: Returns True when generation should stop.
        texture: Optional starting texture (float32 in [0,1] or uint8 in [0,255]).
        subgrid_rows: Rows per sub-grid (e.g., 1, 2, 3).
        subgrid_cols: Cols per sub-grid (e.g., 2, 2, 3).
        facing_percentile: Passed through to process_uv_texture (weight shaping).

    Returns:
        Final full-resolution texture as uint8 (H=texture_resolution,
                                                W=texture_resolution,
                                                C=3).
    """
    _require_cv2()
    sd_tile_res = get_default_sd_resolution(
        process_parameter.sd_version,
        process_parameter.custom_sd_resolution,
    )

    render_res = int(process_parameter.render_resolution)
    tex_res = int(process_parameter.texture_resolution)

    n_views = int(process_parameter.num_cameras)

    views_per_subgrid = max(1, subgrid_rows * subgrid_cols)
    n_subgrids = math.ceil(n_views / views_per_subgrid)

    if texture is None:
        previous_texture = np.full((tex_res, tex_res, 3), 255, dtype=np.uint8)
    else:
        if texture.dtype != np.uint8:
            previous_texture = (255 * np.clip(texture, 0, 1)).astype(np.uint8)
        else:
            previous_texture = texture.copy()
        if previous_texture.shape[:2] != (tex_res, tex_res):
            previous_texture = cv2.resize(
                previous_texture,
                (tex_res, tex_res),
                interpolation=cv2.INTER_LANCZOS4,
            )

    previous_texture = previous_texture[..., :3]  # Ensure RGB only

    painted_area_texture = np.zeros((tex_res, tex_res, 3), dtype=np.uint8)

    pipeline = create_diffusion_pipeline(process_parameter)

    all_texture_array = np.zeros(
        shape=(
            n_subgrids,
            int(process_parameter.texture_resolution),
            int(process_parameter.texture_resolution),
            3,
        ),
    )
    all_texture_weight_array = np.zeros(
        shape=(
            n_subgrids,
            int(process_parameter.texture_resolution),
            int(process_parameter.texture_resolution),
        ),
    )

    for s in range(n_subgrids):
        if should_cancel():
            msg = "Cancelled by user."
            raise TextureGenerationCancelledError(msg)

        start_idx = s * views_per_subgrid
        count = min(views_per_subgrid, n_views - start_idx)
        if count <= 0:
            break

        # Sub-grid slice
        mv_subset = slice_multiview_dict(multiview_images, start_idx, count)

        # Re-assemble sub-grid using the CURRENT texture (parallel within this group)
        _, subgrids_resized = assemble_multiview_subgrid(
            texture=(previous_texture.astype(np.float32) / 255.0),
            painted_area_texture=painted_area_texture,
            multiview_images=mv_subset,
            render_resolution=render_res,
            sd_resolution=sd_tile_res,
            n_subgrids=1,
            n_rows_subgrid=subgrid_rows,
            n_cols_subgrid=subgrid_cols,
        )
        grids_resized = subgrids_resized[0]

        # SD inference for this sub-grid
        def sub_progress_callback(sub_percent: int) -> None:
            # Map sub-grid progress into overall [0..100]
            overall = int(((s + sub_percent / 100.0) / n_subgrids) * 100)  # noqa: B023
            progress_callback(overall)

        output_grid = run_pipeline(
            pipe=pipeline,
            process_parameter=process_parameter,
            input_img=Image.fromarray(grids_resized["input_grid"].astype(np.uint8)),
            uv_mask=Image.fromarray(grids_resized["content_grid"]),
            canny_img=Image.fromarray(grids_resized["canny_grid"]),
            normal_img=Image.fromarray(grids_resized["normal_grid"]),
            depth_img=Image.fromarray(grids_resized["depth_grid"]),
            progress_callback=sub_progress_callback,
            should_cancel=should_cancel,
            strength=process_parameter.denoise_strength,
            guidance_scale=process_parameter.guidance_scale,
            num_inference_steps=process_parameter.num_inference_steps,
        )
        output_grid_np = np.array(output_grid)

        if should_cancel():
            msg = "Cancelled by user."
            raise TextureGenerationCancelledError(msg)

        # Project only sub-grid output to UV space (full-res) using existing weighting
        subgrid_tex, subgrid_valid_texture = process_uv_texture(
            process_parameter=process_parameter,
            uv_images=mv_subset["uv"],
            facing_images=mv_subset["facing"],
            output_grid=output_grid_np,
            target_resolution=tex_res,
            render_resolution=render_res,
            facing_percentile=facing_percentile,
            grid_cols=subgrid_cols,
        )

        # Add the new painted area to painted_area_texture
        painted_area_texture[subgrid_valid_texture > 0] = 255

        all_texture_array[s] = subgrid_tex
        all_texture_weight_array[s] = subgrid_valid_texture / 255.0

        subgrid_alpha = smooth_alpha_map(
            subgrid_valid_texture.astype(np.float32) / 255.0,
        )
        subgrid_alpha *= (subgrid_valid_texture > 0).astype(np.float32)
        previous_texture = blend_texture_with_alpha(
            cast("NDArray[np.uint8]", previous_texture),
            subgrid_tex,
            subgrid_alpha,
        )

    textured_area = np.sum(all_texture_weight_array, axis=0)
    textured_area = np.clip(textured_area, 0, 1)
    textured_area = (textured_area * 255).astype(np.uint8)

    final_texture = inpaint_missing(
        int(process_parameter.texture_resolution),
        all_texture_array,
        all_texture_weight_array,
    ).astype(np.uint8)

    progress_callback(100)
    return final_texture


def slice_multiview_dict(
    d: dict[str, list[NDArray]],
    start: int,
    count: int,
) -> dict[str, list[NDArray]]:
    """Slice a multi-view dictionary.

    Return a dict with the same keys, slicing each list to [start, start+count),
    wrapping around if needed.

    Example:
        n = 9, start = 7, count = 4
        indices = [7, 8, 0, 1]
    """
    n = len(d["depth"])
    return {k: [d[k][(start + i) % n] for i in range(count)] for k in d}
