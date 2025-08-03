import numpy as np
from numpy.typing import NDArray
from PIL import Image

from ..blender_operations import ProcessParameter
from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline
from .process_operations import (
    assemble_multiview_list,
    process_uv_texture,
)


def img_sequential(
    multiview_images: dict[str, list[NDArray]],
    process_parameter: ProcessParameter,
    progress_callback: callable,
    texture: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Process multiview images sequentially view-by-view with diffusion pipeline.

    Args:
        multiview_images: Dict of multiview input images per view.
        process_parameter: Processing parameters.
        progress_callback: Function to track progress.
        texture: Optional texture to project.

    Returns:
        Processed full-resolution texture.
    """
    # Assemble per-view processed/resized maps
    _, resized_list = assemble_multiview_list(
        texture=texture,
        multiview_images=multiview_images,
        sd_resolution=int(process_parameter.sd_resolution),
    )

    n_views = len(resized_list["input"])
    results = []

    # Create the diffusion pipeline once
    pipeline = create_diffusion_pipeline(process_parameter)

    # create a sub callback to report progress
    def sub_progress_callback(view_index: int, total_views: int, step_index: int, total_steps: int) -> None:
        progress = int((view_index + step_index / total_steps) / total_views * 100)
        progress_callback(progress)

    keep_mask = None
    for i in range(n_views):
        if i == 0:
            input_img = Image.fromarray(resized_list["input"][i].astype(np.uint8))
            mask_img = Image.fromarray(resized_list["content"][i])
            previous_texture = texture if texture is not None else 255*np.ones_like(resized_list["input"][i], dtype=np.uint8)
        else:
            # TODO: if i>0 project the texture to the current view
            # TODO: To build parts of the view point input.
            # TODO: if we have an input texture, we overwrite the input_img with the projected texture
            # TODO: In the projected areas and remove the projected view from the input image inpainting area
            input_img, keep_mask = create_new_view_input(
                previous_texture,
                resized_list["input"][i],   
                resized_list["uv"][i],
                resized_list["facing"][i],
                resized_list["content"][i],
            )
            mask_img = resized_list["content"][i] * keep_mask[..., None]
            mask_img = Image.fromarray(mask_img)


        result = run_pipeline(
            pipe=pipeline,
            process_parameter=process_parameter,
            input_img=input_img,
            uv_mask=mask_img,
            canny_img=Image.fromarray(resized_list["canny"][i]),
            normal_img=Image.fromarray(resized_list["normal"][i]),
            depth_img=Image.fromarray(resized_list["depth"][i]),
            progress_callback=sub_progress_callback,
            strength=process_parameter.denoise_strength,
            guidance_scale=process_parameter.guidance_scale,
            num_inference_steps=process_parameter.num_inference_steps,
        )

        # TODO: Project the current result back to the texture
        previous_texture = project_view_to_texture(
            sd_result=result,
            uv_view=resized_list["uv"][i],
            facing_view=resized_list["facing"][i],
            texture_resolution=int(process_parameter.texture_resolution),
            texture=previous_texture,
        )

        results.append(np.array(result))

    # Stitch back to texture using UVs and facing info
    return process_uv_texture(
        process_parameter=process_parameter,
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=np.stack(results),
        target_resolution=int(process_parameter.texture_resolution),
    )

def project_view_to_texture(
            sd_result: Image,
            uv_view: NDArray[np.uint8],
            facing_view: NDArray[np.uint8],
            texture_resolution: int,
            texture: NDArray[np.uint8],
        ) -> NDArray[np.uint8]:
    """Project the output of the diffusion model back onto the texture."""
    
    sd_array = np.array(sd_result)
    sd_array = sd_array[..., :3]  # Ensure we only take RGB channels

    uv_view = uv_view[..., :2]  # Ensure UVs are 2D
    facing_view = facing_view[..., 0]  # Use the first channel for facing

    # Create a new texture to hold the projected result
    new_texture = np.zeros(
        (texture_resolution, texture_resolution, 3), dtype=np.uint8
    )
    facing_texture = np.zeros(
        (texture_resolution, texture_resolution), dtype=np.uint8
    )

    uv_view = uv_view * (texture_resolution - 1)  # Scale UVs to texture size

    new_texture[uv_view[..., 1].astype(int), uv_view[..., 0].astype(int)] = sd_array
    facing_texture[uv_view[..., 1].astype(int), uv_view[..., 0].astype(int)] = facing_view

    # remove areas where facing is less than 0.5
    mask = facing_texture < 0.5

    new_texture[mask] = 0
    facing_texture[mask] = 0

    if texture is not None:
        # Blend the new texture with the existing texture
        new_texture = np.where(
            new_texture == 0, texture, new_texture
        )

    return new_texture

def create_new_view_input(
    output_texture: NDArray[np.float32],
    input_view: NDArray[np.uint8],
    uv_view: NDArray[np.uint8],
    facing_view: NDArray[np.uint8],
    content_view: NDArray[np.uint8],
) -> Image:
    """Create a new input image for the current view by projecting the output texture."""

    # TODO: Implement the projection logic

    # TODO: keep_mask should be a binary mask where only the area that we want to keep is 0

    return Image.fromarray(new_view_input), keep_mask