import bpy
import numpy as np
from scipy.spatial.transform import Rotation


def blendercs_to_ccs(points_bcs, camera, rotation_only=False):
    """
    Converts a point cloud from the Blender coordinate system to the camera coordinate system.

    :param points_bcs: Points in the Blender coordinate system (nx3).
    :param camera: Blender camera object.
    :param rotation_only: If True, only apply rotation without translation.

    :return: Points in the camera coordinate system.
    """

    # Extract camera extrinsics from the Blender camera object
    camera_extrinsic_position = np.array(camera.location).reshape((3, 1))
    camera_extrinsic_euler = np.array(camera.rotation_euler).reshape((3, 1))

    # Convert Euler angles to a rotation matrix
    camera_extrinsic_rotation = Rotation.from_euler(
        "XYZ", camera_extrinsic_euler.T[0], degrees=False
    ).as_matrix()

    # Invert the rotation matrix (Blender's transformation to camera coordinate system)
    camera_extrinsic_rotation = np.linalg.inv(camera_extrinsic_rotation)

    if not rotation_only:
        # Translate the points to the camera's coordinate system
        point_3d_cam = points_bcs - camera_extrinsic_position.T
    else:
        # If rotation only, use the original points
        point_3d_cam = points_bcs

    # Rotate the point cloud to align with the camera's rotation
    point_3d_cam = np.dot(camera_extrinsic_rotation, point_3d_cam.T).T

    # Convert from Blender's coordinate system to the camera coordinate system used in computer vision
    R_blender_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    point_3d_cam = np.dot(R_blender_to_cv, point_3d_cam.T).T

    return point_3d_cam
