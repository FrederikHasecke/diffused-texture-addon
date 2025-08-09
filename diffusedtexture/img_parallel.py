import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..blender_operations import ProcessParameter
from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline
from .process_operations import (
    assemble_multiview_grid,
    process_uv_texture,
)


def img_parallel(
    multiview_images: dict[str, list[NDArray]],
    process_parameter: ProcessParameter,
    progress_callback: callable,
    texture: NDArray[np.float32] | None = None,
    facing_percentile: float = 0.5,
) -> NDArray[np.float32]:
    """Process multiview images in parallel.

    Args:
        multiview_images (dict[str, list[NDArray]]): Multiview images to process.
        process_parameter (ProcessParameter): parameter for the processing.
        progress_callback (callable): Callback to report progress.
        texture (NDArray[np.float32] | None, optional): Input texture. Defaults to None.

    Returns:
        NDArray[np.float32]: Processed texture.
    """
    # Assemble grids
    _, resized_grids = assemble_multiview_grid(
        texture=texture,
        multiview_images=multiview_images,
        render_resolution=int(process_parameter.render_resolution),
    )

    # Create the diffusion pipeline
    pipeline = create_diffusion_pipeline(process_parameter)

    # Run pipeline
    output_grid = run_pipeline(
        pipe=pipeline,
        process_parameter=process_parameter,
        input_img=Image.fromarray(resized_grids["input_grid"].astype(np.uint8)),
        uv_mask=Image.fromarray(resized_grids["content_grid"]),
        canny_img=Image.fromarray(resized_grids["canny_grid"]),
        normal_img=Image.fromarray(resized_grids["normal_grid"]),
        depth_img=Image.fromarray(resized_grids["depth_grid"]),
        progress_callback=progress_callback,
        strength=process_parameter.denoise_strength,
        guidance_scale=process_parameter.guidance_scale,
        num_inference_steps=process_parameter.num_inference_steps,
    )

    # Process UV texture
    return process_uv_texture(
        process_parameter=process_parameter,
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=np.array(output_grid),
        target_resolution=int(process_parameter.texture_resolution),
        facing_percentile=facing_percentile,
    )
