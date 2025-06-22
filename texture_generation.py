from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .blender_operations import ProcessParameters
from .diffusedtexture.img_parallel import img_parallel

# from .diffusedtexture.img_sequential import img_sequential  # noqa: ERA001
from .diffusedtexture.uv_pass import uv_pass


def load_multiview_images(render_img_folders: str) -> dict[str, list[NDArray[Any]]]:
    multiview_images = {"depth": [], "normal": [], "uv": [], "facing": []}

    for folder in render_img_folders:
        for file_name in Path(folder).iterdir:
            file_path = Path(folder) / file_name
            image = np.array(Image.open(file_path))
            if "depth" in file_name:
                multiview_images["depth"].append(image)
            elif "normal" in file_name:
                multiview_images["normal"].append(image)
            elif "uv" in file_name:
                multiview_images["uv"].append(image)
            elif "facing" in file_name:
                multiview_images["facing"].append(image)
            else:
                continue  # Skip files that do not match any category

    return multiview_images


def run_texture_generation(
    process_parameter: ProcessParameters,
    render_img_folders: dict[str, NDArray | str],
    texture: NDArray[Any] | None = None,
) -> None:
    """Run the texture generation in a separate thread."""
    if process_parameter.operation_mode == "UV_PASS":
        output_texture: NDArray[np.uint8] = uv_pass(
            process_parameter,
            render_img_folders,
            texture,
        )

    else:
        # Assemble grids from rendered images
        multiview_images = load_multiview_images(render_img_folders)

        if process_parameter.operation_mode == "PARALLEL_IMG":
            output_texture: NDArray[np.uint8] = img_parallel(
                multiview_images,
                process_parameter,
                texture,
            )
        elif process_parameter.operation_mode == "SEQUENTIAL_IMG":
            msg = "SEQUENTIAL_IMG mode not yet available"
            raise NotImplementedError(msg)
            # output_texture: NDArray[np.uint8] = img_sequential(
            #     multiview_images,
            #     process_parameter,
            #     texture,
            # )  # noqa: ERA001, RUF100
        elif process_parameter.operation_mode == "PARA_SEQUENTIAL_IMG":
            msg = "PARA_SEQUENTIAL_IMG mode not yet available"
            raise NotImplementedError(msg)

    # Save the resulting texture
    output_path = Path(process_parameter.output_path) / "final_texture.png"
    Image.fromarray(output_texture).save(output_path)
