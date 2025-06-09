import os
from pathlib import Path


import bpy
import numpy as np
from PIL import Image

from .diffusedtexture.pipeline.pipeline_builder import create_diffusion_pipeline
from .diffusedtexture.pipeline.pipeline_runner import run_pipeline
from .diffusedtexture.process_operations import (
    assemble_multiview_grid,
    create_input_image_grid,
    process_uv_texture,
)


def run_texture_generation(context: bpy.context, render_img_folders: list) -> None:
    """Run the texture generation in a separate thread."""
    # Assemble grids from rendered images
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

    # Assemble grids
    grids, resized_grids = assemble_multiview_grid(
        multiview_images, render_resolution=int(context.scene.render_resolution)
    )

    # Create the diffusion pipeline
    pipeline = create_diffusion_pipeline(context)

    # Input grid
    input_image_grid = create_input_image_grid(
        np.ones_like(resized_grids["uv_grid"]),
        resized_grids["uv_grid"],
        resized_grids["uv_grid"],
    )

    # Run pipeline
    output_grid = run_pipeline(
        pipeline,
        context,
        Image.fromarray(input_image_grid),
        Image.fromarray(resized_grids["content_mask"]),
        Image.fromarray(resized_grids["canny_grid"]),
        Image.fromarray(resized_grids["normal_grid"]),
        Image.fromarray(resized_grids["depth_grid"]),
        strength=context.scene.denoise_strength,
        guidance_scale=context.scene.guidance_scale,
    )[0]

    # Process UV texture
    filled_uv_texture = process_uv_texture(
        context=context,
        uv_images=multiview_images["uv"],
        facing_images=multiview_images["facing"],
        output_grid=np.array(output_grid),
        target_resolution=int(context.scene.texture_resolution),
    )

    # Save the resulting texture
    output_path = Path(context.scene.output_path) / "final_texture.png"
    Image.fromarray(filled_uv_texture).save(output_path)
