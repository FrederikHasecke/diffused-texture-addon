import numpy as np
import cv2

from .diffusers_utils import (
    create_first_pass_pipeline,
    infer_first_pass_pipeline,
)
from .process_operations import (
    process_uv_texture,
    generate_multiple_views,
    assemble_multiview_grid,
    create_input_image_grid,
    delete_render_folders,
)


def second_pass(scene, max_size, texture):
    """Run the second pass for texture refinement."""

    multiview_images, render_img_folders = generate_multiple_views(
        scene=scene,
        max_size=max_size,
        suffix="second_pass",
        render_resolution=int(scene.render_resolution),
        offset_additional=5,
    )

    multiview_grids, resized_multiview_grids = assemble_multiview_grid(
        multiview_images,
        render_resolution=int(scene.render_resolution),
        sd_resolution=512,
    )

    # Flip texture vertically (blender y 0 is down, opencv y 0 is up)
    # texture_flipped = np.transpose(texture[::-1], (1, 0, 2))
    texture = texture[::-1]

    projected_texture_grid = create_input_image_grid(
        texture, multiview_grids["uv_grid"], resized_multiview_grids["uv_grid"]
    )

    # TODO: Create the pipe with the optional addition of a new checkpoint and additional loras
    pipe = create_first_pass_pipeline(scene)
    output_grid = infer_first_pass_pipeline(
        pipe,
        scene,
        projected_texture_grid,
        resized_multiview_grids["content_mask"],
        resized_multiview_grids["canny_grid"],
        resized_multiview_grids["normal_grid"],
        resized_multiview_grids["depth_grid"],
        strength=scene.denoise_strength,
        guidance_scale=scene.guidance_scale,
    )

    output_grid = np.array(output_grid)

    filled_uv_texture_first_pass = process_uv_texture(
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=output_grid,
        target_resolution=int(scene.texture_resolution),
        render_resolution=int(scene.render_resolution),
        facing_percentile=0.5,
    )

    # delete all rendering folders
    delete_render_folders(render_img_folders)

    return filled_uv_texture_first_pass
