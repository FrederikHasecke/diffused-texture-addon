import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..blender_operations import ProcessParameters
from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline
from .process_operations import (
    assemble_multiview_grid,
    process_uv_texture,
)


def img_parallel(
    multiview_images: dict[str, list[NDArray]],
    process_parameters: ProcessParameters,
    texture: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    # Assemble grids
    grids, resized_grids = assemble_multiview_grid(
        texture=texture,
        multiview_images=multiview_images,
        render_resolution=int(process_parameters.render_resolution),
    )

    # Create the diffusion pipeline
    pipeline = create_diffusion_pipeline(process_parameters)

    # Run pipeline
    output_grid = run_pipeline(
        pipeline,
        process_parameters,
        Image.fromarray(resized_grids["input_grid"].astype(np.uint8)),
        Image.fromarray(resized_grids["content_grid"]),
        Image.fromarray(resized_grids["canny_grid"]),
        Image.fromarray(resized_grids["normal_grid"]),
        Image.fromarray(resized_grids["depth_grid"]),
        strength=process_parameters.denoise_strength,
        guidance_scale=process_parameters.guidance_scale,
    )

    # Process UV texture
    return process_uv_texture(
        process_parameters=process_parameters,
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=np.array(output_grid),
        target_resolution=int(process_parameters.texture_resolution),
    )
