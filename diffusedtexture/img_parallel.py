import numpy as np
from PIL import Image

# from .diffusers_utils import (
#     create_pipeline,
#     infer_pipeline,
# )
from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline

from .process_operations import (
    process_uv_texture,
    generate_multiple_views,
    assemble_multiview_grid,
    create_input_image_grid,
    delete_render_folders,
)


def img_parallel(scene, max_size, texture=None):
    """Run the first pass for texture generation."""

    multiview_images, render_img_folders = generate_multiple_views(
        scene=scene,
        max_size=max_size,
        suffix="img_parallel",
        render_resolution=int(scene.render_resolution),
    )

    # sd_resolution = 512 if scene.sd_version == "sd15" else 1024

    if scene.custom_sd_resolution:
        sd_resolution = int(
            int(scene.custom_sd_resolution) // np.sqrt(int(scene.num_cameras))
        )
    else:
        sd_resolution = 512 if scene.sd_version == "sd15" else 1024

    _, resized_multiview_grids = assemble_multiview_grid(
        multiview_images,
        render_resolution=int(scene.render_resolution),
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

    pipe = create_diffusion_pipeline(scene)
    output_grid = run_pipeline(
        pipe,
        scene,
        Image.fromarray(input_image_sd),
        resized_multiview_grids["content_mask"],
        resized_multiview_grids["canny_grid"],
        resized_multiview_grids["normal_grid"],
        resized_multiview_grids["depth_grid"],
        strength=scene.denoise_strength,
        guidance_scale=scene.guidance_scale,
    )[0]

    output_grid = np.array(output_grid)

    filled_uv_texture = process_uv_texture(
        scene=scene,
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=output_grid,
        target_resolution=int(scene.texture_resolution),
        render_resolution=int(scene.render_resolution),
        facing_percentile=0.5,
    )

    # delete all rendering folders
    delete_render_folders(render_img_folders)

    return filled_uv_texture
