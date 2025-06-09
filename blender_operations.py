from enum import Enum
from typing import Any

import bpy

from .render_setup import (
    create_cameras_on_one_ring,
    create_cameras_on_sphere,
    setup_render_settings,
)
from .utils import isolate_object


def prepare_scene(obj: bpy.data.objects) -> dict[str, Any]:
    """Backup and isolate the object to work with."""
    backup_data = isolate_object(obj)
    bpy.context.view_layer.objects.active = obj
    return backup_data


def render_views(context: bpy.context, obj: bpy.data.object) -> dict:
    """Render views and save to folders.

    Args:
        context (bpy.context): _description_
        obj (bpy.data.object): _description_

    Raises:
        ValueError: _description_

    Returns:
        dict: _description_
    """
    # Set up cameras
    num_cameras = int(context.scene.num_cameras)
    max_size = max(obj.dimensions)

    # Set parameters
    num_cameras = int(context.scene.num_cameras)

    # Create cameras based on the number specified in the scene
    if num_cameras == NumCameras.four:
        cameras = create_cameras_on_one_ring(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="RenderCam",
        )
    elif num_cameras in [NumCameras.nine, NumCameras.sixteen]:
        cameras = create_cameras_on_sphere(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="RenderCam",
        )
    else:
        msg = "Only 4, 9, or 16 cameras are supported."
        raise ValueError(msg)

    # Set up render nodes
    render_img_folders = setup_render_settings(context)

    # Render for each camera
    for camera in cameras:
        context.scene.camera = camera
        bpy.ops.render.render(write_still=True)

    return render_img_folders
