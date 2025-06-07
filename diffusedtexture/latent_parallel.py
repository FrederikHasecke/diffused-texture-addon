"""
This module contains the latent_parallel function, which is used to generate a texture from a scene using the latent parallel method.
The latent parallel method uses a pipeline to denoise the input image at multiple stages, mixing latents inbetween each stage.
The final output is a texture that is generated from the mixed latents. The texture is then returned as a numpy array.

It does not work with the current version of the Diffuser's Library, as pipelines with denoising_start and denoising_end parameters
do not support ControlNets yet. I will need to update the pipeline to support ControlNets in order to use this method.
"""

import numpy as np

from .diffusers_utils import (
    create_pipeline,
    infer_pipeline,
)
from .process_operations import (
    latent_mixing_parallel,
    process_uv_texture,
    generate_multiple_views,
    assemble_multiview_grid,
    create_input_image_grid,
    delete_render_folders,
)


def latent_parallel(scene, max_size, texture=None):
    multiview_images, render_img_folders = generate_multiple_views(
        scene=scene,
        max_size=max_size,
        suffix="img_parallel",
        render_resolution=int(scene.render_resolution),
    )

    if scene.custom_sd_resolution:
        sd_resolution = int(
            int(scene.custom_sd_resolution) // np.sqrt(int(scene.num_cameras))
        )
    else:
        sd_resolution = 512 if scene.sd_version == "sd15" else 1024

    latent_resolution = sd_resolution // 16

    _, resized_multiview_grids = assemble_multiview_grid(
        multiview_images,
        render_resolution=int(scene.render_resolution),
        sd_resolution=sd_resolution,
        latent_resolution=latent_resolution,
    )

    if texture is not None:
        # Flip texture vertically (blender y 0 is down, opencv y 0 is up)
        texture = texture[::-1]

        input_image_sd = create_input_image_grid(
            texture, multiview_images["uv_grid"], resized_multiview_grids["uv_grid"]
        )
    else:
        input_image_sd = (
            255 * np.ones_like(resized_multiview_grids["canny_grid"])
        ).astype(np.uint8)

    latents = None
    pipe = create_pipeline(scene)

    # add latent mixing inbetween at multiple times throughout the denoising process
    for denoising_start, denoising_end in [
        (0.0, 0.25),
        (0.25, 0.5),
        (0.5, 0.75),
        (0.75, 1.0),
    ]:
        if latents is None:
            latents = infer_pipeline(
                pipe,
                scene,
                input_image=input_image_sd,
                uv_mask=resized_multiview_grids["content_mask"],
                canny_img=resized_multiview_grids["canny_grid"],
                normal_img=resized_multiview_grids["normal_grid"],
                depth_img=resized_multiview_grids["depth_grid"],
                strength=scene.denoise_strength,
                num_inference_steps=int(0.25 * scene.num_inference_steps),
                guidance_scale=scene.guidance_scale,
                denoising_start=denoising_start,
                denoising_end=denoising_end,
                output_type="latent",
            )

        else:
            # Mix latents via normal face direction
            latents = latent_mixing_parallel(
                scene=scene,
                uv_list=multiview_images["uv"],
                facing_list=multiview_images["facing"],
                latents=latents,
            )

            if denoising_end == 1.0:
                output_grid = infer_pipeline(
                    pipe,
                    scene,
                    input_image=latents,
                    uv_mask=resized_multiview_grids["content_mask"],
                    canny_img=resized_multiview_grids["canny_grid"],
                    normal_img=resized_multiview_grids["normal_grid"],
                    depth_img=resized_multiview_grids["depth_grid"],
                    strength=scene.denoise_strength,
                    num_inference_steps=int(0.25 * scene.num_inference_steps),
                    guidance_scale=scene.guidance_scale,
                    denoising_start=denoising_start,
                    denoising_end=denoising_end,
                    output_type="PIL",
                )[0]
                output_grid = np.array(output_grid)

            else:
                latents = infer_pipeline(
                    pipe,
                    scene,
                    input_image=latents,
                    uv_mask=resized_multiview_grids["content_mask"],
                    canny_img=resized_multiview_grids["canny_grid"],
                    normal_img=resized_multiview_grids["normal_grid"],
                    depth_img=resized_multiview_grids["depth_grid"],
                    strength=scene.denoise_strength,
                    num_inference_steps=int(0.25 * scene.num_inference_steps),
                    guidance_scale=scene.guidance_scale,
                    denoising_start=denoising_start,
                    denoising_end=denoising_end,
                    output_type="latent",
                )

    # Save output_grid as image
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
