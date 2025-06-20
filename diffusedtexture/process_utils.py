import bpy
import numpy as np
from numpy.typing import NDArray


def blendercs_to_ccs(
    points_bcs: np.ndarray,
    camera: bpy.types.Camera,
    rotation_only: bool = False,
) -> NDArray[np.float32]:
    """Converts 3D points from the Blender coordinate system to camera coordinates."""
    # Extract camera rotation in world space
    camera_rotation = np.array(camera.matrix_world.to_quaternion().to_matrix()).T

    # Apply the rotation to align normals with the cameras view
    if rotation_only:
        point_3d_cam = np.dot(camera_rotation, points_bcs.T).T
    else:
        # Translate points to the camera's coordinate system
        camera_position = np.array(camera.matrix_world.to_translation()).reshape((3,))
        points_bcs = points_bcs - camera_position
        point_3d_cam = np.dot(camera_rotation, points_bcs.T).T

    # Convert to camera coordinate system by inverting the Z-axis
    R_blender_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    point_3d_cam = np.dot(R_blender_to_cv, point_3d_cam.T).T

    return point_3d_cam
