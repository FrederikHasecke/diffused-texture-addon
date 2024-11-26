import os
import numpy as np
import cv2

from .diffusers_utils import (
    create_first_pass_pipeline,
    infer_first_pass_pipeline,
)
from .process_operations import (
    process_uv_texture,
    generate_multiple_views,
    create_input_image,
    create_content_mask,
    delete_render_folders,
)


def third_pass(scene, max_size, texture):
    """Run the third pass for sequential texture refinement."""

    # Flip texture vertically (blender y 0 is down, opencv y 0 is up)
    # texture_flipped = np.transpose(texture[::-1], (1, 0, 2))
    texture = texture[::-1]

    multiview_images, render_img_folders = generate_multiple_views(
        scene=scene,
        max_size=max_size,
        suffix="third_pass",
        render_resolution=int(scene.render_resolution),
        offset_additional=15,
    )

    # Scale the texture to the new texture res (if it isnt)
    if int(scene.texture_resolution) != texture.shape[0]:
        final_texture = cv2.resize(
            texture[..., :3],
            (int(scene.texture_resolution), int(scene.texture_resolution)),
            interpolation=cv2.INTER_LANCZOS4,
        )
    else:
        final_texture = np.copy(texture)[..., :3]

    # lookup table to store info on already set and fixed pixels (2), set and variable pixels (1), and untouched ones (0)
    pixel_status = np.zeros(
        (int(scene.texture_resolution), int(scene.texture_resolution))
    )
    max_angle_status = np.zeros(
        (int(scene.texture_resolution), int(scene.texture_resolution))
    )

    # TODO: Create the pipe with the optional addition of a new checkpoint and additional loras
    pipe = create_first_pass_pipeline(scene)

    for i in range(int(scene.num_cameras)):

        (
            _,
            input_render_resolution,
            _,
        ) = create_input_image(
            final_texture,
            multiview_images["uv"][i],
            int(scene.render_resolution),
            int(scene.texture_resolution),
        )

        # Get the render view mask
        content_mask_fullsize = create_content_mask(multiview_images["uv"][i])

        # get the UV coordinates of the current iteration
        uv_image = multiview_images["uv"][i][..., :2]  # Keep only u and v
        uv_image[content_mask_fullsize == 0] = 0

        # copy of uv_image at texture scale
        uv_image_texture_scale = cv2.resize(
            np.copy(multiview_images["uv"][i]),
            (int(scene.texture_resolution), int(scene.texture_resolution)),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        content_mask_texture_scale = cv2.resize(
            content_mask_fullsize,
            (int(scene.texture_resolution), int(scene.texture_resolution)),
            interpolation=cv2.INTER_NEAREST_EXACT,
        )
        uv_image_texture_scale = uv_image_texture_scale[..., :2]
        uv_image_texture_scale[content_mask_texture_scale == 0] = 0

        # resize the uv values to 0-2047
        uv_coordinates_tex = (
            (uv_image_texture_scale * (int(scene.texture_resolution) - 1))
            .astype(np.int32)
            .reshape(-1, 2)
        )

        # the uvs are meant to start from the bottom left corner, so we flip the y axis (v axis)
        uv_coordinates_tex[:, 1] = (
            int(scene.texture_resolution) - 1
        ) - uv_coordinates_tex[:, 1]

        # in case we have uv coordinates beyond the texture
        uv_coordinates_tex = uv_coordinates_tex % int(scene.texture_resolution)
        uv_coordinates_tex = uv_coordinates_tex.astype(np.int32)

        # Coordiates in render size
        uv_coordinates_render = uv_image
        uv_coordinates_render = (
            (uv_coordinates_render * (int(scene.texture_resolution) - 1))
            .astype(np.int32)
            .reshape(-1, 2)
        )
        uv_coordinates_render[:, 1] = (
            int(scene.texture_resolution) - 1
        ) - uv_coordinates_render[:, 1]
        uv_coordinates_render = uv_coordinates_render % int(scene.texture_resolution)

        content_mask_texture = np.zeros(
            (int(scene.texture_resolution), int(scene.texture_resolution))
        )

        # add ALL the UV content mask to content_mask_texture
        content_mask_texture[uv_coordinates_tex[:, 1], uv_coordinates_tex[:, 0]] = 1

        # Remove the fixed parts of the texture from the content mask before resizing
        content_mask_texture[pixel_status == 2] = 0

        # Remove the variable parts of the texture from the content mask if the current facing is less than the previous
        facing_texture = np.zeros(
            (int(scene.texture_resolution), int(scene.texture_resolution))
        )

        # facing image to texture scale
        facing_render = cv2.resize(
            np.copy(multiview_images["facing"][i]),
            (int(scene.texture_resolution), int(scene.texture_resolution)),
            interpolation=cv2.INTER_NEAREST_EXACT,
        ).flatten()

        facing_texture[uv_coordinates_tex[:, 1], uv_coordinates_tex[:, 0]] = (
            facing_render
        )

        content_mask_texture[
            np.logical_and(pixel_status == 1, max_angle_status > facing_texture)
        ] = 0

        # if something is facing too little toward the camera, remove it
        content_mask_texture[facing_texture < 0.75] = 0

        # Add the current values to the maps
        max_angle_status[content_mask_texture > 0] = facing_texture[
            content_mask_texture > 0
        ]
        pixel_status[content_mask_texture > 0] = (
            pixel_status[content_mask_texture > 0] + 1
        )
        pixel_status = np.clip(pixel_status, a_min=0, a_max=2)

        # Project the content_mask_texture back to render view
        content_mask_render = content_mask_texture[
            uv_coordinates_tex[:, 1], uv_coordinates_tex[:, 0]
        ]

        # point array to image
        content_mask_render = content_mask_render.reshape(
            (int(scene.texture_resolution), int(scene.texture_resolution))
        )

        content_mask_render_sd = cv2.resize(
            content_mask_render,
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        )

        input_image_sd = cv2.resize(
            input_render_resolution,
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        )

        canny_img = cv2.resize(
            multiview_images["normal"][i].astype(np.uint8),
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        )
        canny_img = cv2.Canny(canny_img, 100, 200)
        canny_img = np.stack(
            (canny_img, canny_img, canny_img),
            axis=-1,
        ).astype(np.uint8)
        normal_img = cv2.resize(
            multiview_images["normal"][i],
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        )
        depth_img = cv2.resize(
            multiview_images["depth"][i],
            (512, 512),
            interpolation=cv2.INTER_LINEAR,
        )

        output = infer_first_pass_pipeline(
            pipe,
            scene,
            input_image_sd,
            content_mask_render_sd,
            canny_img,
            normal_img,
            depth_img,
            strength=scene.denoise_strength,
            guidance_scale=scene.guidance_scale,
        )

        # upscale output to texture size
        output = cv2.resize(
            np.array(output)[..., :3],
            (int(scene.render_resolution), int(scene.render_resolution)),
            interpolation=cv2.INTER_LINEAR,
        )

        # overlay the output on the input with feathered blend
        overlay_mask = np.copy(content_mask_render)

        overlay_mask = cv2.resize(
            overlay_mask,
            (int(scene.render_resolution), int(scene.render_resolution)),
            interpolation=cv2.INTER_LINEAR,
        )

        # shrink the mask by half the blur distance
        overlay_mask = cv2.erode(overlay_mask, np.ones((3, 3)), iterations=2)
        overlay_mask = cv2.blur(overlay_mask, (9, 9))
        overlay_mask = np.stack((overlay_mask, overlay_mask, overlay_mask), axis=-1)

        output = (1 - overlay_mask) * input_render_resolution.astype(
            np.float32
        ) + overlay_mask * output.astype(np.float32)
        # Convert back to uint8
        output = np.clip(output, 0, 255).astype(np.uint8)

        iteration_texture = np.zeros(
            (int(scene.texture_resolution), int(scene.texture_resolution), 3)
        )
        iteration_texture[
            uv_coordinates_render[:, 1], uv_coordinates_render[:, 0], ...
        ] = output.reshape(-1, 3)

        # add the new parts to the texture
        final_texture[content_mask_texture > 0] = iteration_texture[
            content_mask_texture > 0
        ]

    # delete all rendering folders
    delete_render_folders(render_img_folders)

    return final_texture
