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


def render_views(scene: bpy.scene.context, obj: bpy.data.object) -> dict:
    """Render views and save to folders.

    Args:
        scene (bpy.scene.context): _description_
        obj (bpy.data.object): _description_

    Raises:
        ValueError: _description_

    Returns:
        dict: _description_
    """
    # Set up cameras
    num_cameras = int(scene.num_cameras)
    max_size = max(obj.dimensions)

    # Set parameters
    num_cameras = int(scene.num_cameras)

    # Create cameras based on the number specified in the scene
    if num_cameras == 4:
        cameras = create_cameras_on_one_ring(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="RenderCam",
        )
    elif num_cameras in [9, 16]:
        cameras = create_cameras_on_sphere(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="RenderCam",
        )
    else:
        msg = "Only 4, 9, or 16 cameras are supported."
        raise ValueError(msg)

    # Set up render nodes
    render_img_folders = setup_render_settings(scene)

    # Render for each camera
    for camera in cameras:
        scene.camera = camera
        bpy.ops.render.render(write_still=True)

    return render_img_folders
