import math

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..blender_operations import ProcessParameter
from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline
from .process_operations import (
    assemble_multiview_subgrid,
    process_uv_texture,
)


def img_parasequential(  # noqa: PLR0913
    multiview_images: dict[str, list[NDArray]],
    process_parameter: ProcessParameter,
    progress_callback: callable,
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
        texture: Optional starting texture (float32 in [0,1] or uint8 in [0,255]).
        subgrid_rows: Rows per sub-grid (e.g., 1, 2, 3).
        subgrid_cols: Cols per sub-grid (e.g., 2, 2, 3).
        facing_percentile: Passed through to process_uv_texture (weight shaping).

    Returns:
        Final full-resolution texture as uint8 (H=texture_resolution,
                                                W=texture_resolution,
                                                C=3).
    """
    if process_parameter.custom_sd_resolution:
        sd_tile_res = int(process_parameter.custom_sd_resolution)
    else:
        sd_tile_res = 512 if process_parameter.sd_version == "sd15" else 1024

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
    for s in range(n_subgrids):
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
            strength=process_parameter.denoise_strength,
            guidance_scale=process_parameter.guidance_scale,
            num_inference_steps=process_parameter.num_inference_steps,
        )
        output_grid_np = np.array(output_grid)

        # Project only sub-grid output to UV space (full-res) using existing weighting
        subgrid_tex, subgrid_valid_texture = process_uv_texture(
            process_parameter=process_parameter,
            uv_images=mv_subset["uv"],
            facing_images=mv_subset["facing"],
            output_grid=output_grid_np,
            target_resolution=tex_res,
            render_resolution=render_res,
        )

        # Add the new painted area to painted_area_texture
        painted_area_texture[subgrid_valid_texture == 255] = 255  # noqa: PLR2004

        # DEBUG: Output all relevant images
        cv2.imwrite(
            f"{process_parameter.output_path}/subgrid_{s}_output.png",
            subgrid_tex,
        )
        Image.fromarray(grids_resized["input_grid"].astype(np.uint8)).save(
            f"{process_parameter.output_path}/input_view_{s}.png",
        )
        Image.fromarray(grids_resized["content_grid"]).save(
            f"{process_parameter.output_path}/uv_mask_{s}.png",
        )
        output_grid.save(f"{process_parameter.output_path}/output_grid_{s}.png")

        # Paste only where the sub-grid is actually visible
        coverage_mask = build_uv_coverage_mask(
            mv_subset["uv"],
            mv_subset["facing"],
            tex_res,
        )
        keep = coverage_mask > (facing_percentile * 255)
        previous_texture[keep] = subgrid_tex[keep]

    progress_callback(100)
    return previous_texture


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


def build_uv_coverage_mask(
    uv_list: list[NDArray],
    facing_list: list[NDArray],
    tex_res: int,
) -> NDArray[np.uint8]:
    """Compute a binary mask of texels covered by this sub-grid's views."""
    mask = np.zeros((tex_res, tex_res), dtype=np.uint8)
    for uv_img, facing_img in zip(uv_list, facing_list, strict=False):
        # UV to texture space
        uv = uv_img[..., :2] * (tex_res - 1)
        uv = uv.astype(np.int32)
        uv_x = np.clip(uv[..., 0], 0, tex_res - 1)
        uv_y = np.clip((tex_res - 1) - uv[..., 1], 0, tex_res - 1)  # flip V

        # "Visible" proxy from facing (any >0 considered visible)
        f = (255 * np.clip(facing_img[..., 0], 0, 1)).astype(np.uint8)
        visible = (f > 0).astype(np.uint8)

        m = np.zeros((tex_res, tex_res), dtype=np.uint8)
        m[uv_y.flatten(), uv_x.flatten()] = visible.flatten() * 255
        mask = np.maximum(mask, m)

    # Small dilation to avoid pinholes
    return cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
