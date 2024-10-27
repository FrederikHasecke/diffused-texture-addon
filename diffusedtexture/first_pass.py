import os
import math
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
    create_similar_angle_image,
)
from diffusedtexture.diffusers_utils import (
    create_first_pass_pipeline,
    infer_first_pass_pipeline,
)
from diffusedtexture.process_operations import (
    process_uv_texture,
    generate_multiple_views,
    assemble_multiview_grid,
    save_multiview_grids,
)


def first_pass(scene, max_size):
    """Run the first pass for texture generation."""

    # num_cameras = int(scene.num_cameras)

    # # target_resolution = int(2 * end_resolution)
    # target_resolution = int(scene.texture_resolution)

    # render_resolution = 2048  # TODO: Test if this prevents the dittering effect

    # if num_cameras == 4:
    #     cameras = create_cameras_on_one_ring(
    #         num_cameras=num_cameras,
    #         max_size=max_size,
    #         name_prefix="Camera_first_pass",
    #     )
    # elif num_cameras == 9:
    #     cameras = create_cameras_on_sphere(
    #         num_cameras=num_cameras,
    #         max_size=max_size,
    #         name_prefix="Camera_first_pass",
    #         offset=False,
    #     )
    # elif num_cameras == 16:
    #     cameras = create_cameras_on_sphere(
    #         num_cameras=num_cameras, max_size=max_size, name_prefix="Camera_first_pass"
    #     )

    # else:
    #     raise ValueError("Only 4, 9 or 16 cameras are supported for first pass.")

    # output_path = Path(scene.output_path)  # Use Path for OS-independent path handling

    # output_nodes = setup_render_settings(
    #     scene, resolution=(render_resolution, render_resolution)
    # )

    # # Set base paths for outputs
    # output_nodes["depth"].base_path = str(output_path / "first_pass_depth")
    # output_nodes["normal"].base_path = str(output_path / "first_pass_normal")
    # output_nodes["uv"].base_path = str(output_path / "first_pass_uv")
    # output_nodes["position"].base_path = str(output_path / "first_pass_position")
    # output_nodes["img"].base_path = str(output_path / "first_pass_img")

    # # Create directories if they don't exist
    # for key in output_nodes:
    #     os.makedirs(output_nodes[key].base_path, exist_ok=True)

    # # Update to have all new cameras available
    # bpy.context.view_layer.update()

    # # Load the rendered images
    # depth_images = []
    # normal_images = []
    # position_images = []
    # uv_images = []
    # img_images = []
    # facing_images = []

    # # get the current blender frame
    # frame_number = bpy.context.scene.frame_current

    # # Render with each of the cameras in the list
    # for i, camera in enumerate(cameras):
    #     # Set the active camera to the current camera
    #     scene.camera = camera

    #     # Set the output file path for each pass with the corresponding camera index
    #     output_nodes["depth"].file_slots[0].path = f"depth_camera_{i:02d}_"
    #     output_nodes["normal"].file_slots[0].path = f"normal_camera_{i:02d}_"
    #     output_nodes["uv"].file_slots[0].path = f"uv_camera_{i:02d}_"
    #     output_nodes["position"].file_slots[0].path = f"position_camera_{i:02d}_"
    #     output_nodes["img"].file_slots[0].path = f"img_camera_{i:02d}_"

    #     # Update to have all new cameras available
    #     bpy.context.view_layer.update()

    #     # Render the scene
    #     bpy.ops.render.render(write_still=True)

    #     # Update to have all new cameras available
    #     bpy.context.view_layer.update()

    #     # Load UV images
    #     uv_image_path = (
    #         Path(output_nodes["uv"].base_path)
    #         / f"uv_camera_{i:02d}_{frame_number:04d}.exr"
    #     )

    #     uv_image = bpy_img_to_numpy(str(uv_image_path))

    #     uv_images.append(uv_image[..., :3])

    #     # Load depth images
    #     depth_image_path = (
    #         Path(output_nodes["depth"].base_path)
    #         / f"depth_camera_{i:02d}_{frame_number:04d}.exr"
    #     )

    #     depth_image = create_depth_condition(str(depth_image_path))

    #     depth_images.append(depth_image[..., :3])

    #     # Load position images
    #     position_image_path = (
    #         Path(output_nodes["position"].base_path)
    #         / f"position_camera_{i:02d}_{frame_number:04d}.exr"
    #     )

    #     position_image = bpy_img_to_numpy(str(position_image_path))
    #     position_images.append(position_image[..., :3])

    #     # Load normal images
    #     normal_image_path = (
    #         Path(output_nodes["normal"].base_path)
    #         / f"normal_camera_{i:02d}_{frame_number:04d}.exr"
    #     )
    #     normal_image = create_normal_condition(str(normal_image_path), camera)
    #     normal_images.append(normal_image[..., :3])

    #     # Load render images
    #     img_image_path = (
    #         Path(output_nodes["img"].base_path)
    #         / f"img_camera_{i:02d}_{frame_number:04d}.exr"
    #     )

    #     img_image = bpy_img_to_numpy(str(img_image_path))

    #     img_images.append(img_image[..., :3])

    #     # create the facing images
    #     facing_img = create_similar_angle_image(
    #         bpy_img_to_numpy(str(normal_image_path))[..., :3],
    #         position_image[..., :3],
    #         camera,
    #     )

    #     facing_images.append(facing_img)

    # TODO: Parallel stuff here (quad creation and such)

    multiview_images = generate_multiple_views(scene=scene, max_size=max_size)

    # # Set parameters
    # num_cameras = int(scene.num_cameras)
    # target_resolution = int(scene.texture_resolution)
    # render_resolution = 2048

    # # create the empty grid images
    # depth_quad = np.zeros(
    #     (
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         3,
    #     ),
    #     dtype=np.uint8,
    # )
    # normal_quad = np.zeros(
    #     (
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         3,
    #     ),
    #     dtype=np.uint8,
    # )

    # # TODO: Just for debugging
    # facing_quad = np.zeros(
    #     (
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         int(math.sqrt(num_cameras) * render_resolution),
    #     ),
    #     dtype=np.uint8,
    # )

    # uv_quad = np.zeros(
    #     (
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         3,
    #     ),
    #     dtype=np.uint8,
    # )

    # uv_quad_vis = np.zeros(
    #     (
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         3,
    #     ),
    #     dtype=np.uint8,
    # )

    # # Combine the 16 images into a 4x4 grid structure
    # # TODO: CHECK IF THIS FIXES THE ASSIGNMENT!
    # for i, (depth_img, normal_img, face_img, uv_image) in enumerate(
    #     zip(
    #         multiview_images["depth"],
    #         multiview_images["normal"],
    #         multiview_images["facing"],
    #         multiview_images["uv"],
    #     )
    # ):
    #     # Calculate the position in the grid
    #     row = int((i // math.sqrt(num_cameras)) * render_resolution)
    #     col = int((i % math.sqrt(num_cameras)) * render_resolution)

    #     # Add the images to the grid
    #     depth_quad[row : row + render_resolution, col : col + render_resolution] = (
    #         depth_img
    #     )
    #     normal_quad[row : row + render_resolution, col : col + render_resolution] = (
    #         normal_img
    #     )

    #     # TODO: Just for debugging
    #     facing_quad[row : row + render_resolution, col : col + render_resolution] = (
    #         255 * face_img
    #     ).astype(np.uint8)

    #     uv_quad[row : row + render_resolution, col : col + render_resolution] = uv_image

    #     uv_quad_vis[row : row + render_resolution, col : col + render_resolution] = (
    #         255 * uv_image
    #     ).astype(np.uint8)

    # # Input image full white image
    # content_mask = np.zeros(
    #     (
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         int(math.sqrt(num_cameras) * render_resolution),
    #     )
    # )
    # content_mask[np.sum(uv_quad, axis=-1) > 0] = 255
    # content_mask = content_mask.astype(np.uint8)

    # input_image = 255 * np.ones(
    #     (
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         int(math.sqrt(num_cameras) * render_resolution),
    #         3,
    #     )
    # )
    # # input_image[content_mask == 0] = 255
    # # input_image = np.clip(input_image, 0, 255)
    # input_image = input_image.astype(np.uint8)

    # # Resize the images to a 512 square content size
    # input_image_sd = cv2.resize(
    #     input_image,
    #     (
    #         int(input_image.shape[0] / render_resolution * 512),
    #         int(input_image.shape[0] / render_resolution * 512),
    #     ),
    #     interpolation=cv2.INTER_NEAREST,
    # )
    # content_mask_sd = cv2.resize(
    #     content_mask,
    #     (
    #         int(content_mask.shape[0] / render_resolution * 512),
    #         int(content_mask.shape[0] / render_resolution * 512),
    #     ),
    #     interpolation=cv2.INTER_NEAREST,
    # )

    # content_mask_sd = cv2.dilate(
    #     content_mask_sd, np.ones((5, 5), np.uint8), iterations=3
    # )

    # normal_quad_sd = cv2.resize(
    #     normal_quad,
    #     (
    #         int(normal_quad.shape[0] / render_resolution * 512),
    #         int(normal_quad.shape[0] / render_resolution * 512),
    #     ),
    #     interpolation=cv2.INTER_NEAREST,
    # )
    # depth_quad_sd = cv2.resize(
    #     depth_quad,
    #     (
    #         int(depth_quad.shape[0] / render_resolution * 512),
    #         int(depth_quad.shape[0] / render_resolution * 512),
    #     ),
    #     interpolation=cv2.INTER_NEAREST,
    # )

    multiview_grids, resized_multiview_grids = assemble_multiview_grid(
        multiview_images, sd_resolution=int(scene.texture_resolution)
    )

    save_multiview_grids(
        multiview_grids=multiview_grids, output_folder=str(scene.output_path)
    )

    save_multiview_grids(
        multiview_grids=resized_multiview_grids, output_folder=str(scene.output_path)
    )

    input_image_sd = (255 * np.ones_like(resized_multiview_grids["canny_grid"])).astype(
        np.uint8
    )

    # TODO: Create the pipe with the optional addition of a new checkpoint and additional loras
    pipe = create_first_pass_pipeline(scene)
    output_grid = infer_first_pass_pipeline(
        pipe,
        scene,
        input_image_sd,
        resized_multiview_grids["content_mask"],
        resized_multiview_grids["canny_grid"],
        resized_multiview_grids["normal_grid"],
        resized_multiview_grids["depth_grid"],
        strength=scene.denoise_strength,
    )

    # # TODO: for quick debug
    # output_quad = input_image_sd

    # cv2.imwrite(
    #     os.path.join(scene.output_path, "input_image.png"),
    #     cv2.cvtColor(input_image_sd, cv2.COLOR_RGB2BGR),
    # )
    # cv2.imwrite(
    #     os.path.join(scene.output_path, "content_mask_sd.png"),
    #     cv2.cvtColor(content_mask_sd, cv2.COLOR_RGB2BGR),
    # )
    # cv2.imwrite(
    #     os.path.join(scene.output_path, "canny_quad.png"),
    #     cv2.cvtColor(canny_quad_sd, cv2.COLOR_RGB2BGR),
    # )
    # cv2.imwrite(
    #     os.path.join(scene.output_path, "normal_quad.png"),
    #     cv2.cvtColor(normal_quad_sd, cv2.COLOR_RGB2BGR),
    # )
    # cv2.imwrite(
    #     os.path.join(scene.output_path, "depth_quad.png"),
    #     cv2.cvtColor(depth_quad_sd, cv2.COLOR_RGB2BGR),
    # )
    # cv2.imwrite(
    #     os.path.join(scene.output_path, "facing_quad.png"),
    #     cv2.cvtColor(facing_quad, cv2.COLOR_RGB2BGR),
    # )

    # cv2.imwrite(
    #     os.path.join(scene.output_path, "uv_quad_vis.png"),
    #     cv2.cvtColor(uv_quad_vis, cv2.COLOR_RGB2BGR),
    # )

    output_grid = np.array(output_grid)

    # TODO: REMOVE ME, just for debugging
    # output the image as png
    # cv2.imwrite(
    #     os.path.join(scene.output_path, "output_quad.png"),
    #     cv2.cvtColor(output_quad, cv2.COLOR_RGB2BGR),
    # )

    filled_uv_texture_first_pass = process_uv_texture(
        uv_images=multiview_grids["uv_grid"],
        facing_images=multiview_grids["facing_grid"],
        output_grid=output_grid,
        target_resolution=int(scene.texture_resolution),
        render_resolution=2048,
    )

    return filled_uv_texture_first_pass
