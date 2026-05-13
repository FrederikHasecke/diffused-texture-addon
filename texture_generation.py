from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .blender_operations import ProcessParameter, UVPassAssets, load_img_to_numpy
from .diffusedtexture.img_parallel import img_parallel
from .diffusedtexture.img_parasequential import img_parasequential
from .diffusedtexture.img_sequential import img_sequential
from .diffusedtexture.uv_pass import uv_pass
from .diffusedtexture.uv_stitch_pass import uv_stitch_pass

MultiviewImages = dict[str, list[NDArray[Any]]]


def _as_path(path_like: NDArray | str | Path) -> Path | None:
    if isinstance(path_like, Path):
        return path_like
    if isinstance(path_like, str):
        return Path(path_like)
    return None


def _target_render_bucket(file_name: str) -> str | None:
    if "depth" in file_name:
        return "depth"
    if "normal" in file_name:
        return "normal"
    if "uv" in file_name:
        return "uv"
    if "facing" in file_name:
        return "facing"
    return None


def load_multiview_images(
    render_img_folders: dict[str, NDArray | str],
) -> MultiviewImages:
    multiview_images = {"depth": [], "normal": [], "uv": [], "facing": []}

    for folder_path in render_img_folders.values():
        folder = _as_path(folder_path)
        if folder is None:
            continue

        for camera_folder_name in sorted(folder.iterdir()):
            if not camera_folder_name.is_dir():
                continue

            for file_path in sorted(camera_folder_name.iterdir()):
                if not file_path.is_file():
                    continue

                image = load_img_to_numpy(file_path)

                target_bucket = _target_render_bucket(file_path.name)
                if target_bucket is None:
                    continue

                multiview_images[target_bucket].append(image)

    return multiview_images


def run_texture_generation(  # noqa: C901, PLR0913
    process_parameter: ProcessParameter,
    generation_inputs: MultiviewImages | UVPassAssets,
    progress_callback: Callable,
    should_cancel: Callable[[], bool],
    mark_done: Callable,
    return_texture_bucket: list,
    texture: NDArray[np.float32] | None = None,
    uv_assets: UVPassAssets | None = None,
) -> None:
    """Run the texture generation in a separate thread.

    Args:
        process_parameter: parameter for the process.
        generation_inputs: Preloaded generation inputs for the selected mode.
        progress_callback: Function accepting an int (0-100) to report progress.
        should_cancel: Function returning True when generation should stop.
        mark_done: Function to call when the process is done.
        return_texture_bucket: Optional bucket to store the resulting texture.
        texture: Optional input texture.
        uv_assets: Optional UV-space assets for automatic image-mode stitching.
    """
    if process_parameter.operation_mode == "UV_PASS":
        if not isinstance(generation_inputs, UVPassAssets):
            msg = "UV mode requires UV pass assets."
            raise TypeError(msg)
        output_texture = uv_pass(
            uv_assets=generation_inputs,
            process_parameter=process_parameter,
            progress_callback=progress_callback,
            should_cancel=should_cancel,
            texture=texture,
        )
    else:
        if isinstance(generation_inputs, UVPassAssets):
            msg = "Image-based modes require preloaded multiview images."
            raise TypeError(msg)
        multiview_images = generation_inputs

        def base_progress_callback(percent: int) -> None:
            if uv_assets is None:
                progress_callback(percent)
                return
            progress_callback(int(percent * 0.75))

        def stitch_progress_callback(percent: int) -> None:
            progress_callback(75 + int(percent * 0.25))

        if process_parameter.operation_mode == "PARALLEL_IMG":
            output_texture = img_parallel(
                multiview_images=multiview_images,
                process_parameter=process_parameter,
                progress_callback=base_progress_callback,
                should_cancel=should_cancel,
                texture=texture,
            )
        elif process_parameter.operation_mode == "SEQUENTIAL_IMG":
            output_texture = img_sequential(
                multiview_images=multiview_images,
                process_parameter=process_parameter,
                progress_callback=base_progress_callback,
                should_cancel=should_cancel,
                texture=texture,
            )
        elif process_parameter.operation_mode == "PARA_SEQUENTIAL_IMG":
            output_texture = img_parasequential(
                multiview_images=multiview_images,
                process_parameter=process_parameter,
                progress_callback=base_progress_callback,
                should_cancel=should_cancel,
                texture=texture,
                subgrid_rows=process_parameter.subgrid_rows,
                subgrid_cols=process_parameter.subgrid_cols,
            )
        else:
            msg = f"Unknown operation mode: {process_parameter.operation_mode}"
            raise ValueError(msg)

        if uv_assets is not None:
            output_texture = uv_stitch_pass(
                texture=output_texture,
                uv_assets=uv_assets,
                multiview_images=multiview_images,
            )
            stitch_progress_callback(100)

    return_texture_bucket.append(output_texture)

    mark_done(success=True) if mark_done else None
