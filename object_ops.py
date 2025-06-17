import bpy
import mathutils


def move_object_to_origin(obj: bpy.types.Object) -> None:
    """Move the object to the world origin."""
    obj.location = (0, 0, 0)


def calculate_mesh_midpoint(obj: bpy.types.Object) -> float:
    """Calculate the midpoint of the mesh.

    Move the object's origin to midpoint and return the maximum size
    of the mesh in any dimension.

    Args:
        obj (bpy.types.Object): The object whose mesh midpoint and size are calculated.

    Returns:
        float: The maximum size of the mesh.

    """
    # Calculate local coordinates in world space
    local_coords = [obj.matrix_world @ vert.co for vert in obj.data.vertices]

    # Determine the minimum and maximum coordinates
    min_coord = mathutils.Vector(map(min, zip(*local_coords, strict=False)))
    max_coord = mathutils.Vector(map(max, zip(*local_coords, strict=False)))

    # Calculate the midpoint
    midpoint = (min_coord + max_coord) / 2

    # Set the origin to the calculated midpoint
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY", center="BOUNDS")

    # Move the object to align its origin with the world origin
    obj.location -= midpoint

    # Calculate the maximum size in any dimension
    size_vector = max_coord - min_coord

    return max(size_vector)
