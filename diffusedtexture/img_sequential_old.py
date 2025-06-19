import numpy as np
from PIL import Image

import cv2

from .pipeline.pipeline_builder import create_diffusion_pipeline
from .pipeline.pipeline_runner import run_pipeline
from .process_operations import (
    create_input_image,
    create_content_mask,
    delete_render_folders,
)


def prepare_texture(texture, texture_resolution):
    """Prepare the texture by flipping and scaling."""
    if texture is None:
        return (
            np.ones((texture_resolution, texture_resolution, 3), dtype=np.uint8) * 255
        )
    texture = texture[::-1]
    if texture.shape[0] != texture_resolution:
        texture = cv2.resize(
            texture[..., :3],
            (texture_resolution, texture_resolution),
            interpolation=cv2.INTER_LANCZOS4,
        )
    return np.copy(texture)[..., :3]


def prepare_uv_coordinates(uv_image, texture_resolution, target_resolution=None):
    """Prepare UV coordinates scaled to the texture resolution."""

    if target_resolution is not None:
        uv_image = cv2.resize(
            uv_image,
            (target_resolution, target_resolution),
            interpolation=cv2.INTER_NEAREST,
        )

    uv_image = uv_image[..., :2].reshape(-1, 2)

    uv_coords = np.round(uv_image * (texture_resolution - 1)).astype(np.int32)
    uv_coords[:, 1] = texture_resolution - 1 - uv_coords[:, 1]  # Flip v-axis
    uv_coords %= texture_resolution
    return uv_coords


def update_content_mask(
    content_mask_texture, pixel_status, facing_texture, max_angle_status
):
    """Update the content mask based on pixel status and facing angles."""
    content_mask_texture[pixel_status == 2] = 0  # Exclude fixed pixels
    content_mask_texture[(pixel_status == 1) & (max_angle_status > facing_texture)] = 0
    return content_mask_texture


def generate_inputs_for_inference(
    input_render_resolution,
    content_mask_texture,
    multiview_images,
    uv_coordinates_sd,
    scene,
    i,
):
    """Generate inputs for Stable Diffusion inference."""

    # sd_resolution = 512 if scene.sd_version == "sd15" else 1024

    if scene.custom_sd_resolution:
        sd_resolution = int(
            int(scene.custom_sd_resolution) // np.sqrt(int(scene.num_cameras))
        )
    else:
        sd_resolution = 512 if scene.sd_version == "sd15" else 1024

    input_image_sd = cv2.resize(
        input_render_resolution,
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_LINEAR,
    )
    canny_img = cv2.Canny(
        cv2.resize(
            multiview_images["normal"][i].astype(np.uint8),
            (sd_resolution, sd_resolution),
        ),
        100,
        200,
    )
    canny_img = np.stack([canny_img] * 3, axis=-1)
    normal_img = cv2.resize(
        multiview_images["normal"][i],
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_LINEAR,
    )
    depth_img = cv2.resize(
        multiview_images["depth"][i],
        (sd_resolution, sd_resolution),
        interpolation=cv2.INTER_LINEAR,
    )

    content_mask_sd = content_mask_texture_to_render_sd(
        content_mask_texture, uv_coordinates_sd, scene, sd_resolution
    )

    return input_image_sd, content_mask_sd, canny_img, normal_img, depth_img


def blend_output(
    input_render_resolution, output, content_mask_render, render_resolution
):
    """Blend output with input using feathered overlay."""
    overlay_mask = cv2.resize(
        content_mask_render,
        (render_resolution, render_resolution),
        interpolation=cv2.INTER_LINEAR,
    )
    # overlay_mask = cv2.erode(overlay_mask, np.ones((3, 3)), iterations=2)
    # overlay_mask = cv2.blur(overlay_mask, (9, 9))
    overlay_mask = cv2.blur(overlay_mask, (3, 3))
    overlay_mask = np.stack([overlay_mask] * 3, axis=-1).astype(np.float32) / 255

    # blended = (1 - overlay_mask) * input_render_resolution + overlay_mask * output
    blended = (1 - overlay_mask) * input_render_resolution + overlay_mask * output

    return np.clip(blended, 0, 255).astype(np.uint8)


def content_mask_texture_to_render_sd(
    content_mask_texture, uv_coordinates_sd, scene, target_resolution
):
    """Convert content mask from texture resolution to SD resolution."""

    content_mask_sd = content_mask_texture[
        uv_coordinates_sd[:, 1], uv_coordinates_sd[:, 0]
    ]

    input_resolution = np.sqrt(len(uv_coordinates_sd)).astype(np.int32)

    content_mask_sd = cv2.resize(
        content_mask_sd.reshape(input_resolution, input_resolution),
        (target_resolution, target_resolution),
        interpolation=cv2.INTER_NEAREST,
    )

    content_mask_sd = content_mask_sd.reshape(
        target_resolution, target_resolution
    ).astype(np.uint8)

    return content_mask_sd


# Main Function
def img_sequential(scene, max_size, texture):
    """Run the third pass for sequential texture refinement."""
    texture_resolution = int(scene.texture_resolution)
    render_resolution = int(scene.render_resolution)
    num_cameras = int(scene.num_cameras)

    final_texture = prepare_texture(texture, texture_resolution)
    # multiview_images, render_img_folders = generate_multiple_views(
    #     scene, max_size, suffix="img_sequential", render_resolution=render_resolution
    # )

    pixel_status = np.zeros((texture_resolution, texture_resolution))
    max_angle_status = np.zeros((texture_resolution, texture_resolution))

    pipe = create_diffusion_pipeline(scene)

    for i in range(num_cameras):
        _, input_render_resolution, _ = create_input_image(
            final_texture,
            multiview_images["uv"][i],
            render_resolution,
            texture_resolution,
        )

        content_mask_fullsize = create_content_mask(multiview_images["uv"][i])
        uv_image = multiview_images["uv"][i][..., :2]
        uv_image[content_mask_fullsize == 0] = 0

        uv_coordinates_tex = prepare_uv_coordinates(
            uv_image, texture_resolution, texture_resolution
        )
        facing_render = cv2.resize(
            multiview_images["facing"][i],
            (texture_resolution, texture_resolution),
            interpolation=cv2.INTER_LINEAR,
        ).flatten()

        facing_texture = np.zeros((texture_resolution, texture_resolution))

        facing_texture[uv_coordinates_tex[:, 1], uv_coordinates_tex[:, 0]] = (
            facing_render
        )

        content_mask_tex_size = cv2.resize(
            content_mask_fullsize,
            (texture_resolution, texture_resolution),
            interpolation=cv2.INTER_NEAREST,
        )

        content_mask_texture = np.zeros((texture_resolution, texture_resolution))
        content_mask_texture[uv_coordinates_tex[:, 1], uv_coordinates_tex[:, 0]] = (
            content_mask_tex_size.reshape(-1)
        )

        content_mask_texture = cv2.resize(
            content_mask_texture,
            (texture_resolution, texture_resolution),
            interpolation=cv2.INTER_NEAREST,
        )

        content_mask_texture = update_content_mask(
            content_mask_texture, pixel_status, facing_texture, max_angle_status
        )

        max_angle_status[content_mask_texture > 0] = facing_texture[
            content_mask_texture > 0
        ]
        pixel_status[content_mask_texture > 0] = np.clip(
            pixel_status[content_mask_texture > 0] + 1, 0, 2
        )

        uv_coordinates_sd = prepare_uv_coordinates(
            uv_image, texture_resolution, 512 if scene.sd_version == "sd15" else 1024
        )

        input_image_sd, content_mask_render_sd, canny_img, normal_img, depth_img = (
            generate_inputs_for_inference(
                input_render_resolution,
                content_mask_texture,
                multiview_images,
                uv_coordinates_sd,
                scene,
                i,
            )
        )

        output = run_pipeline(
            pipe,
            scene,
            Image.fromarray(input_image_sd),
            content_mask_render_sd,
            canny_img,
            normal_img,
            depth_img,
            strength=scene.denoise_strength,
            guidance_scale=scene.guidance_scale,
        )[0]
        output = cv2.resize(
            np.array(output)[..., :3],
            (render_resolution, render_resolution),
            interpolation=cv2.INTER_LINEAR,
        )

        # save_debug_images(
        #     scene,
        #     i,
        #     input_image_sd,
        #     content_mask_render_sd,
        #     content_mask_texture,
        #     canny_img,
        #     normal_img,
        #     depth_img,
        #     output,
        # )

        uv_coordinates_render = prepare_uv_coordinates(
            uv_image, texture_resolution, render_resolution
        )
        content_mask_render = content_mask_texture_to_render_sd(
            content_mask_texture, uv_coordinates_render, scene, render_resolution
        )

        if i > 0:
            output = blend_output(
                input_render_resolution, output, content_mask_render, render_resolution
            )

        uv_coordinates_render = prepare_uv_coordinates(uv_image, texture_resolution)
        final_texture[uv_coordinates_render[:, 1], uv_coordinates_render[:, 0], :] = (
            output.reshape(-1, 3)
        )

    delete_render_folders(render_img_folders)
    return final_texture
