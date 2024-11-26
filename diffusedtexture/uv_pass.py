import os
import bpy
import cv2
import numpy as np
from pathlib import Path

from ..condition_setup import (
    bpy_img_to_numpy,
)
from .diffusers_utils import (
    create_uv_pass_pipeline,
    infer_uv_pass_pipeline,
)


def export_uv_layout(obj_name, export_path, uv_map_name=None, size=(1024, 1024)):
    """
    Export the UV layout of a given mesh object and specified UV map to an image file.

    :param obj_name: Name of the object in Blender.
    :param export_path: File path where the UV layout should be saved.
    :param uv_map_name: Name of the UV map to use (default is the active UV map).
    :param size: Resolution of the UV layout image (default is 1024x1024).
    """
    # Get the object
    obj = bpy.data.objects.get(obj_name)
    if obj is None or obj.type != "MESH":
        print(f"Object {obj_name} not found or is not a mesh.")
        return

    # Set the object as active
    bpy.context.view_layer.objects.active = obj

    # Ensure it is in object mode
    bpy.ops.object.mode_set(mode="OBJECT")

    # Get the UV map layers
    uv_layers = obj.data.uv_layers
    if not uv_layers:
        print(f"No UV maps found for object {obj_name}.")
        return

    # Find or set the active UV map
    if uv_map_name:
        uv_layer = uv_layers.get(uv_map_name)
        if uv_layer is None:
            print(
                f"UV map {uv_map_name} not found on object {obj_name}. Using active UV map."
            )
            uv_layer = obj.data.uv_layers.active
        else:
            uv_layers.active = uv_layer  # Set the selected UV map as active
    else:
        uv_layer = uv_layers.active  # Use active UV map if none is provided

    if uv_layer is None:
        print(f"No active UV map found for object {obj_name}.")
        return

    # Set UV export settings
    bpy.ops.uv.export_layout(
        filepath=export_path,
        size=size,
        opacity=1.0,  # Adjust opacity of the lines (0.0 to 1.0)
        export_all=False,  # Export only active UV map
    )


def uv_pass(scene, texture_input):
    """Run the UV space refinement pass."""

    obj_name = scene.my_mesh_object
    uv_map_name = scene.my_uv_map

    # uv output path
    uv_map_path = Path(scene.output_path)  # Use Path for OS-independent path handling
    uv_map_path = str(uv_map_path / "uv_map_layout.png")

    export_uv_layout(
        obj_name,
        uv_map_path,
        uv_map_name=uv_map_name,
        size=(int(scene.texture_resolution), int(scene.texture_resolution)),
    )

    # load the uv layout
    uv_layout_image = bpy_img_to_numpy(str(uv_map_path))

    uv_layout_image = uv_layout_image / np.max(uv_layout_image)
    uv_layout_image = uv_layout_image * 255
    uv_layout_image = uv_layout_image.astype(np.uint8)

    # Texture mask based on the alpha channel (3rd channel in RGBA image)
    mask = uv_layout_image[..., 3]

    # Apply dilation to the mask
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    # Stack the mask into 3 channels (to match the RGB format)
    mask = np.stack((mask, mask, mask), axis=-1)
    mask = mask.astype(np.uint8)

    # Create the Canny edge detection mask
    canny = 255 * np.ones(
        (int(scene.texture_resolution), int(scene.texture_resolution))
    )

    # Only process areas where the alpha channel is 255
    canny[uv_layout_image[..., 3] >= 1] = uv_layout_image[..., 0][
        uv_layout_image[..., 3] >= 1
    ]

    canny[canny < 64] = 0
    canny[canny >= 192] = 255
    canny = 255 - canny

    # Stack the canny result into 3 channels (RGB)
    canny = np.stack((canny, canny, canny), axis=-1)
    canny = canny.astype(np.uint8)

    # create the pipe
    pipe = create_uv_pass_pipeline(scene)

    texture_input = texture_input[..., :3]

    # create the base of the final texture
    texture_input = cv2.resize(
        texture_input,
        (int(scene.texture_resolution), int(scene.texture_resolution)),
        interpolation=cv2.INTER_LANCZOS4,
    )

    output_image = infer_uv_pass_pipeline(
        pipe,
        scene,
        texture_input,
        mask,
        canny,
        strength=scene.denoise_strength,
    )

    # remove alpha
    output_image = np.array(output_image)[..., :3]

    cv2.imwrite(
        os.path.join(scene.output_path, f"texture_input_uv_pass.png"),
        cv2.cvtColor(texture_input, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, f"mask_uv_pass.png"),
        cv2.cvtColor(mask, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, f"canny_uv_pass.png"),
        cv2.cvtColor(canny, cv2.COLOR_RGB2BGR),
    )
    cv2.imwrite(
        os.path.join(scene.output_path, f"output_image_uv_pass.png"),
        cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR),
    )

    return output_image
