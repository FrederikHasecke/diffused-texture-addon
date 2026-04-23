from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from ..blender_operations import ProcessParameter
from .pipeline.pipeline_builder import create_diffusion_pipeline

try:
    from .pipeline.pipeline_runner import TextureGenerationCancelledError, run_pipeline
except ImportError:
    from .pipeline.pipeline_runner import run_pipeline

    class TextureGenerationCancelledError(RuntimeError):
        """Fallback cancellation error for tests with a stubbed runner."""


from .process_operations import (
    assemble_multiview_grid,
    process_uv_texture,
)


def img_parallel(  # noqa: PLR0913
    multiview_images: dict[str, list[NDArray[Any]]],
    process_parameter: ProcessParameter,
    progress_callback: Callable,
    should_cancel: Callable[[], bool] = lambda: False,
    texture: NDArray[np.float32] | None = None,
    facing_percentile: float = 0.5,
) -> NDArray[np.uint8]:
    """Process multiview images in parallel.

    Args:
        multiview_images (dict[str, list[NDArray]]): Multiview images to process.
        process_parameter (ProcessParameter): parameter for the processing.
        progress_callback (Callable): Callback to report progress.
        texture (NDArray[np.float32] | None, optional): Input texture. Defaults to None.
        should_cancel (Callable[[], bool]): Returns True when generation should stop.
        facing_percentile (float, optional): Facing percentile. Defaults to 0.5.

    Returns:
        NDArray[np.uint8]: Processed texture.
    """
    if Image is None:
        msg = "Pillow is not available."
        raise RuntimeError(msg)

    # Assemble grids
    _, resized_grids = assemble_multiview_grid(
        texture=texture,
        multiview_images=multiview_images,
        render_resolution=int(process_parameter.render_resolution),
    )

    # Create the diffusion pipeline
    pipeline = create_diffusion_pipeline(process_parameter)

    if should_cancel():
        msg = "Cancelled by user."
        raise TextureGenerationCancelledError(msg)

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
        should_cancel=should_cancel,
        strength=process_parameter.denoise_strength,
        guidance_scale=process_parameter.guidance_scale or 7.5,
        num_inference_steps=process_parameter.num_inference_steps,
    )

    if should_cancel():
        msg = "Cancelled by user."
        raise TextureGenerationCancelledError(msg)

    # Process UV texture
    output_texture, _ = process_uv_texture(
        process_parameter=process_parameter,
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=np.array(output_grid),
        render_resolution=int(process_parameter.render_resolution),
        target_resolution=int(process_parameter.texture_resolution),
        facing_percentile=facing_percentile,
    )

    return output_texture
