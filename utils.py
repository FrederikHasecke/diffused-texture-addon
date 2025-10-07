import bpy
import numpy as np
from mathutils import Vector


def update_uv_maps(
    self: bpy.types.Scene,
    context: bpy.types.Context,  # noqa: ARG001
) -> list[tuple[str, str, str]]:
    """Update the list of available UV maps for the selected mesh object.

    This function is used in the panel to populate the UV map dropdown.
    """
    obj = bpy.data.objects.get(self.my_mesh_object)
    if obj and obj.type == "MESH":
        uv_layers = obj.data.uv_layers.keys()
        return [(uv, uv, "") for uv in uv_layers]
    return [("None", "None", "")]


def get_mesh_objects(
    self: bpy.types.Scene,  # noqa: ARG001
    context: bpy.types.Context,  # noqa: ARG001
) -> list[tuple[str, str, str]]:
    """Get all mesh objects in the current Blender scene.

    This function is used to populate the mesh object dropdown in the panel.
    """
    return [(obj.name, obj.name, "") for obj in bpy.data.objects if obj.type == "MESH"]


def update_image_list(
    self: bpy.types.Scene,  # noqa: ARG001
    context: bpy.types.Context,  # noqa: ARG001
) -> list[tuple[str, str, str]]:
    """Update the list of available images in the Blender file.

    This function is used to populate the image selection dropdown in the panel.
    """
    images = [(img.name, img.name, "") for img in bpy.data.images]
    if not images:
        images.append(("None", "None", "No images available"))
    return images


def apply_texture_to_uv_map(
    obj: bpy.types.Object,
    uv_map_name: str,
    image_name: str,
) -> None:
    """Apply the selected image to the specified UV map of the selected object.

    This is used when the user selects an image in the panel and it needs to be
    applied to the object's UV map.
    """
    image = bpy.data.images.get(image_name)
    if image:
        for uv_layer in obj.data.uv_layers:
            if uv_layer.name == uv_map_name:
                # Assuming the object has a material and texture slots
                material = obj.material_slots[0].material
                texture_slot = material.texture_paint_slots[0]
                texture_slot.texture.image = image
                break


def image_to_numpy(image: bpy.types.Image) -> np.ndarray:
    """Convert a Blender image object to a NumPy array."""
    # Get image dimensions
    width, height = image.size

    # Get pixel data (Blender stores pixels in a flat array as RGBA values in [0, 1])
    pixels = np.array(image.pixels[:], dtype=np.float32)

    # Reshape to (height, width, 4) since Blender stores data in RGBA format
    pixels = pixels.reshape((height, width, 4))

    # Discard the alpha channel
    pixels = pixels[:, :, :3]  # Keep only RGB channels

    # Convert the pixel values from [0, 1] to [0, 255] for typical image use
    return np.clip((pixels * 255).astype(np.uint8), 0, 255)


def isolate_object(obj: bpy.types.Object) -> dict:
    """Isolate the target object.

    Temporarily hide all other objects and translate the object so that
    its *mesh geometry center* (not the origin) moves to world (0,0,0).
    The object's origin is not moved.

    """
    hidden_objects = []

    # Hide everything else
    for other in bpy.data.objects:
        if other != obj:
            if not other.hide_get():
                other.hide_set(state=True)
                hidden_objects.append(other)
            if not other.hide_render:
                other.hide_render = True

    # Save originals to restore later if needed
    original_location = obj.location.copy()
    original_matrix_world = obj.matrix_world.copy()

    # Prefer evaluated mesh so modifiers are respected
    if obj.type == "MESH":
        depsgraph = bpy.context.evaluated_depsgraph_get()
        obj_eval = obj.evaluated_get(depsgraph)
        me = obj_eval.to_mesh(preserve_all_data_layers=False, depsgraph=depsgraph)

        if me and len(me.vertices):
            acc = Vector((0.0, 0.0, 0.0))
            mw_mesh = obj_eval.matrix_world
            for v in me.vertices:
                acc += mw_mesh @ v.co
            center_world = acc / len(me.vertices)
        else:
            # Fallback: use current world translation if mesh is empty
            center_world = obj.matrix_world.translation.copy()

        # Cleanup evaluated mesh
        obj_eval.to_mesh_clear()
    else:
        # Non-mesh fallback: use bounding box center
        bb_local_center = sum((Vector(c) for c in obj.bound_box), Vector()) / 8.0
        center_world = obj.matrix_world @ bb_local_center

    # --- Move object so the mesh center lands at the world origin ---
    mw = obj.matrix_world.copy()
    mw.translation -= center_world
    obj.matrix_world = mw

    return {
        "hidden_objects": hidden_objects,
        "target_object": obj,
        "original_location": original_location,
        "original_matrix_world": original_matrix_world,
        "moved_by_world": (-center_world.x, -center_world.y, -center_world.z),
    }
