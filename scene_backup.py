# scene_backup.py
import bpy
import copy


class SceneBackup:
    def __init__(self):
        self.original_scene_data = None

    def save_scene_state(self):
        """Save the current state of the scene."""
        self.original_scene_data = copy.deepcopy(bpy.context.scene)

    def restore_scene_state(self):
        """Restore the scene to its saved state."""
        if not self.original_scene_data:
            print("No scene data to restore.")
            return

        # Clear current scene objects
        for obj in bpy.context.scene.objects:
            bpy.data.objects.remove(obj, do_unlink=True)

        # Restore original scene objects
        for obj in self.original_scene_data.objects:
            bpy.context.scene.collection.objects.link(obj)

        print("Scene restored to its original state.")


def clean_scene(scene):
    """
    Remove all objects from the scene except the selected mesh object.
    """
    selected_object_name = scene.my_mesh_object
    selected_object = bpy.data.objects.get(selected_object_name)

    if not selected_object:
        raise ValueError(f"Object '{selected_object_name}' not found in the scene.")

    # List of objects to remove
    objects_to_remove = []

    # Collect all objects in the scene and collections except the selected one
    for obj in bpy.context.scene.objects:
        if obj.name != selected_object_name:
            objects_to_remove.append(obj)

    # Also check other collections
    for collection in bpy.data.collections:
        for obj in collection.objects:
            if obj.name != selected_object_name and obj not in objects_to_remove:
                objects_to_remove.append(obj)

    # Remove the collected objects
    for obj in objects_to_remove:
        bpy.data.objects.remove(obj, do_unlink=True)

    print(
        f"Scene cleaned: All objects except '{selected_object_name}' have been removed."
    )


def clean_object(scene):
    """
    Remove all materials from the selected mesh object.
    """
    # Get the selected object's name and retrieve the object
    selected_object_name = scene.my_mesh_object
    selected_object = bpy.data.objects.get(selected_object_name)
    selected_object.data.materials.clear()
