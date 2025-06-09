import bpy
import numpy as np
from PIL import Image

from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline
from .process_operations import (
    assemble_multiview_grid,
    create_input_image_grid,
    delete_render_folders,
    generate_multiple_views,
    process_uv_texture,
)


def img_parallel(context: bpy.context, max_size: float, texture=None):
    """Run the first pass for texture generation."""
    multiview_images, render_img_folders = generate_multiple_views(
        context=context.scene,
        max_size=max_size,
        suffix="img_parallel",
        render_resolution=int(context.scene.render_resolution),
    )

    # sd_resolution = 512 if context.scene.sd_version == "sd15" else 1024

    if context.scene.custom_sd_resolution:
        sd_resolution = int(
            int(context.scene.custom_sd_resolution)
            // np.sqrt(int(context.scene.num_cameras))
        )
    else:
        sd_resolution = 512 if context.scene.sd_version == "sd15" else 1024

    _, resized_multiview_grids = assemble_multiview_grid(
        multiview_images,
        render_resolution=int(context.scene.render_resolution),
        sd_resolution=sd_resolution,
    )

    if texture is not None:
        # Flip texture vertically (blender y 0 is down, opencv y 0 is up)
        texture = texture[::-1]

        input_image_sd = create_input_image_grid(
            texture,
            resized_multiview_grids["uv_grid"],
            resized_multiview_grids["uv_grid"],
        )
    else:
        input_image_sd = (
            255 * np.ones_like(resized_multiview_grids["canny_grid"])
        ).astype(np.uint8)

    pipe = create_diffusion_pipeline(context.scene)
    output_grid = run_pipeline(
        pipe,
        context.scene,
        Image.fromarray(input_image_sd),
        resized_multiview_grids["content_mask"],
        resized_multiview_grids["canny_grid"],
        resized_multiview_grids["normal_grid"],
        resized_multiview_grids["depth_grid"],
        strength=context.scene.denoise_strength,
        guidance_scale=context.scene.guidance_scale,
    )[0]

    output_grid = np.array(output_grid)

    filled_uv_texture = process_uv_texture(
        context=context,
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=output_grid,
        target_resolution=int(context.scene.texture_resolution),
        render_resolution=int(context.scene.render_resolution),
        facing_percentile=0.5,
    )

    # delete all rendering folders
    delete_render_folders(render_img_folders)

    return filled_uv_texture
