from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .blender_operations import ProcessParameter, load_img_to_numpy
from .diffusedtexture.img_parallel import img_parallel
from .diffusedtexture.img_parasequential import img_parasequential
from .diffusedtexture.img_sequential import img_sequential


def load_multiview_images(
    render_img_folders: dict[str, NDArray | str],
) -> dict[str, list[NDArray[Any]]]:
    multiview_images = {"depth": [], "normal": [], "uv": [], "facing": []}

    for folder_path in render_img_folders.values():
        for camera_folder_name in Path(folder_path).iterdir():
            for file_path in camera_folder_name.iterdir():
                image = load_img_to_numpy(file_path)

                if "depth" in file_path.name:
                    multiview_images["depth"].append(image)
                elif "normal" in file_path.name:
                    multiview_images["normal"].append(image)
                elif "uv" in file_path.name:
                    multiview_images["uv"].append(image)
                elif "facing" in file_path.name:
                    multiview_images["facing"].append(image)
                else:
                    continue  # Skip files that do not match any category

    return multiview_images


def run_texture_generation(  # noqa: PLR0913
    process_parameter: ProcessParameter,
    render_img_folders: dict[str, NDArray | str],
    progress_callback: Callable,
    mark_done: Callable,
    return_texture_bucket: list,
    texture: NDArray[np.float32] | None = None,
) -> None:
    """Run the texture generation in a separate thread.

    Args:
        process_parameter: parameter for the process.
        render_img_folders: Rendered image folders.
        progress_callback: Function accepting an int (0-100) to report progress.
        mark_done: Function to call when the process is done.
        return_texture_bucket: Optional bucket to store the resulting texture.
        texture: Optional input texture.
    """
    if process_parameter.operation_mode == "UV_PASS":
        msg = "UV Pass mode is currently not implemented."
        raise NotImplementedError(msg)
        # output_texture: NDArray[np.uint8] = uv_pass(
        #     baked_texture_dict=render_img_folders,  # noqa: ERA001
        #     process_parameter=process_parameter,  # noqa: ERA001
        #     progress_callback=progress_callback,  # noqa: ERA001
        #     texture=texture,  # noqa: ERA001
        # )  # noqa: ERA001, RUF100
    else:  # noqa: RET506
        # Assemble grids from rendered images
        multiview_images = load_multiview_images(render_img_folders)

        if process_parameter.operation_mode == "PARALLEL_IMG":
            output_texture: NDArray[np.uint8] = img_parallel(
                multiview_images=multiview_images,
                process_parameter=process_parameter,
                progress_callback=progress_callback,
                texture=texture,
            )
        elif process_parameter.operation_mode == "SEQUENTIAL_IMG":
            output_texture: NDArray[np.uint8] = img_sequential(
                multiview_images=multiview_images,
                process_parameter=process_parameter,
                progress_callback=progress_callback,
                texture=texture,
            )
        elif process_parameter.operation_mode == "PARA_SEQUENTIAL_IMG":
            output_texture: NDArray[np.uint8] = img_parasequential(
                multiview_images=multiview_images,
                process_parameter=process_parameter,
                progress_callback=progress_callback,
                texture=texture,
                subgrid_rows=process_parameter.subgrid_rows,
                subgrid_cols=process_parameter.subgrid_cols,
            )
        else:
            msg = f"Unknown operation mode: {process_parameter.operation_mode}"
            raise ValueError(msg)

    return_texture_bucket.append(output_texture)

    mark_done(success=True) if mark_done else None
