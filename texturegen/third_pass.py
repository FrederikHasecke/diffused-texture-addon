import os
import bpy
import cv2
import numpy as np
from pathlib import Path


from render_setup import (
    setup_render_settings,
    create_cameras_on_sphere,
    create_cameras_on_two_rings,
    create_cameras_on_one_ring,
)
from condition_setup import (
    bpy_img_to_numpy,
    create_depth_condition,
    create_normal_condition,
)
from texturegen.diffusers_utils import (
    create_first_pass_pipeline,
    infer_first_pass_pipeline,
)


def third_pass(scene, max_size, texture):
    """Run the third pass in image space."""

    num_cameras = int(scene.num_cameras)

    if num_cameras == 4:
        cameras = create_cameras_on_one_ring(
            num_cameras=num_cameras,
            max_size=max_size,  # we dont want to intersect with the object
            name_prefix="Camera_third_pass",
        )
    elif num_cameras == 9:
        cameras = create_cameras_on_sphere(
            num_cameras=num_cameras,
            max_size=max_size,  # we dont want to intersect with the object
            name_prefix="Camera_third_pass",
            offset=False,
        )
    elif num_cameras == 16:
        cameras = create_cameras_on_two_rings(
            num_cameras=num_cameras, max_size=max_size, name_prefix="Camera_third_pass"
        )
    else:
        raise ValueError("Only 4, 9 or 16 cameras are supported for first pass.")

    output_path = Path(scene.output_path)  # Use Path for OS-independent path handling

    output_nodes = setup_render_settings(scene, resolution=(512, 512))

    # Set base paths for outputs
    output_nodes["depth"].base_path = str(output_path / "third_pass_depth")
    output_nodes["normal"].base_path = str(output_path / "third_pass_normal")
    output_nodes["uv"].base_path = str(output_path / "third_pass_uv")
    output_nodes["position"].base_path = str(output_path / "third_pass_position")
    output_nodes["img"].base_path = str(output_path / "third_pass_img")

    # Create directories if they don't exist
    for key in output_nodes:
        os.makedirs(output_nodes[key].base_path, exist_ok=True)

        # Render with each of the cameras in the list
    for i, camera in enumerate(cameras):
        # Set the active camera to the current camera
        scene.camera = camera

        # Set the output file path for each pass with the corresponding camera index
        output_nodes["depth"].file_slots[0].path = f"depth_camera_{i:02d}_"
        output_nodes["normal"].file_slots[0].path = f"normal_camera_{i:02d}_"
        output_nodes["uv"].file_slots[0].path = f"uv_camera_{i:02d}_"
        output_nodes["position"].file_slots[0].path = f"position_camera_{i:02d}_"
        output_nodes["img"].file_slots[0].path = f"img_camera_{i:02d}_"

        # Render the scene
        bpy.ops.render.render(write_still=True)

    # Load the rendered images
    depth_images = []
    normal_images = []
    uv_images = []
    img_images = []

    # get the current blender frame
    frame_number = bpy.context.scene.frame_current

    for i, camera in enumerate(cameras):

        # Load UV images
        uv_image_path = (
            Path(output_nodes["uv"].base_path)
            / f"uv_camera_{i:02d}_{frame_number:04d}.exr"
        )

        uv_image = bpy_img_to_numpy(str(uv_image_path))

        uv_images.append(uv_image)

        # Load depth images
        depth_image_path = (
            Path(output_nodes["depth"].base_path)
            / f"depth_camera_{i:02d}_{frame_number:04d}.exr"
        )

        depth_image = create_depth_condition(str(depth_image_path))

        depth_images.append(depth_image)

        # Load position images
        position_image_path = (
            Path(output_nodes["position"].base_path)
            / f"position_camera_{i:02d}_{frame_number:04d}.exr"
        )
        # Load normal images
        normal_image_path = (
            Path(output_nodes["normal"].base_path)
            / f"normal_camera_{i:02d}_{frame_number:04d}.exr"
        )
        normal_image = create_normal_condition(
            str(normal_image_path), str(position_image_path), camera
        )
        normal_images.append(normal_image)

        # Load render images
        img_image_path = (
            Path(output_nodes["img"].base_path)
            / f"img_camera_{i:02d}_{frame_number:04d}.exr"
        )

        img_image = bpy_img_to_numpy(str(img_image_path))

        img_images.append(img_image)

    # final_texture = cv2.GaussianBlur(texture, (5, 5), 0)

    # transpose texture
    texture = texture[..., :3]
    texture = texture[::-1]  # np.transpose(texture[::-1], (1, 0, 2))

    # create the base of the final texture
    final_texture = cv2.resize(
        texture,
        (int(scene.texture_resolution), int(scene.texture_resolution)),
        interpolation=cv2.INTER_LANCZOS4,
    )

    final_texture = final_texture[..., :3]

    # create an unpainted mask for the inpainting
    unpainted_mask = np.zeros(
        (int(scene.texture_resolution), int(scene.texture_resolution)), dtype=np.uint8
    )

    store_z = np.zeros(
        (int(scene.texture_resolution), int(scene.texture_resolution))
    ).astype(np.float32)

    pipe = create_first_pass_pipeline(scene)

    for i, camera in enumerate(cameras):

        # create an empty texture for the current iteration
        iteration_refined_texture = np.nan * np.ones(np.shape(final_texture))

        uv_image_orig = uv_images[i]
        depth_image = depth_images[i]
        normal_image = normal_images[i]
        img_image = img_images[i]

        # get the texture content projected
        uv_image = np.copy(uv_image_orig)
        uv_image = uv_image[..., :2]  # Keep only u and v

        # resize the uv values to 0-1023
        uv_coordinates = (
            (uv_image * (int(scene.texture_resolution) - 1))
            .astype(np.uint16)
            .reshape(-1, 2)
        )

        # the uvs are meant to start from the bottom left corner, so we flip the y axis (v axis)
        uv_coordinates[:, 1] = (int(scene.texture_resolution) - 1) - uv_coordinates[
            :, 1
        ]

        # in case we have uv coordinates beyond the texture
        uv_coordinates = uv_coordinates % int(scene.texture_resolution)

        # remove values where the normal is pointing away from the camera
        normal_z = np.copy(normal_image).astype(np.float32)
        normal_z = normal_z[..., 2]

        normal_z_uv = np.zeros(
            (int(scene.texture_resolution), int(scene.texture_resolution))
        )

        normal_z_uv[uv_coordinates[:, 1], uv_coordinates[:, 0]] = normal_z.reshape(
            -1,
        )

        # compare the normal_z to the store_z
        add_mask = normal_z_uv > store_z

        # get an input image from the original texture
        input_image = final_texture[uv_coordinates[:, 1], uv_coordinates[:, 0], ...]
        input_image = input_image[..., :3]  # cut off alpha
        input_image = input_image.reshape((512, 512, 3))

        img_image = (img_image[..., :3] / np.max(img_image[..., :3]) * 255).astype(
            np.uint8
        )
        canny_image = cv2.Canny(img_image, 100, 200)
        canny_image = np.stack((canny_image, canny_image, canny_image), axis=-1)

        # TODO: add_content_mask
        add_content_mask = add_mask[uv_coordinates[:, 1], uv_coordinates[:, 0]].reshape(
            (512, 512)
        )
        add_content_mask = (255 * add_content_mask).astype(np.uint8)
        add_content_mask = cv2.erode(
            add_content_mask, np.ones((3, 3), np.uint8), iterations=1
        )

        # add_content_mask = np.stack(
        #     (add_content_mask, add_content_mask, add_content_mask), axis=-1
        # )

        output_image = infer_first_pass_pipeline(
            pipe,
            scene,
            input_image,
            add_content_mask,
            canny_image,
            normal_image,
            depth_image,
            strength=scene.denoise_strength,
        )

        # remove alpha
        output_image = np.array(output_image)[..., :3]

        cv2.imwrite(
            os.path.join(scene.output_path, f"input_image_third_pass_{i}.png"),
            cv2.cvtColor(input_image, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(scene.output_path, f"add_content_mask_third_pass_{i}.png"),
            cv2.cvtColor(add_content_mask, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(scene.output_path, f"canny_third_pass_{i}.png"),
            cv2.cvtColor(canny_image, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(scene.output_path, f"normal_third_pass_{i}.png"),
            cv2.cvtColor(normal_image, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(scene.output_path, f"depth_third_pass_{i}.png"),
            cv2.cvtColor(depth_image, cv2.COLOR_RGB2BGR),
        )

        # TODO: REMOVE ME, just for debugging
        # output the image as png
        cv2.imwrite(
            os.path.join(scene.output_path, f"output_third_pass_{i}.png"),
            cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR),
        )

        # project the output image back to the uv map
        iteration_refined_texture[uv_coordinates[:, 1], uv_coordinates[:, 0], :] = (
            np.array(output_image).reshape(-1, 3)
        )

        # save the normal_z values for the next iteration
        store_z[normal_z_uv > store_z] = normal_z_uv[normal_z_uv > store_z]

        # block all worked on set pixels
        store_z[np.logical_and(add_mask, normal_z_uv > 150)] = 255

        # set the values to nan if the normal is pointing away from the camera
        iteration_refined_texture[normal_z_uv < 150] = np.nan

        # set the values to nan if the normal was better in a previous iteration
        iteration_refined_texture[~add_mask] = np.nan

        final_texture[~np.isnan(iteration_refined_texture)] = iteration_refined_texture[
            ~np.isnan(iteration_refined_texture)
        ]

        unpainted_mask[uv_coordinates[:, 1], uv_coordinates[:, 0]] = 255

    iteration_refined_texture = iteration_refined_texture.astype(np.uint8)

    unpainted_mask = 255 - unpainted_mask
    unpainted_mask = unpainted_mask.astype(np.uint8)

    # inpaint all the missing pixels
    final_texture = cv2.inpaint(
        final_texture,
        unpainted_mask,
        inpaintRadius=3,
        flags=cv2.INPAINT_TELEA,
    )

    return final_texture
