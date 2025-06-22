from dataclasses import dataclass
from pathlib import Path
from typing import Any

import bpy
import numpy as np
from numpy.typing import NDArray
from PIL import Image

from .diffusedtexture.camera_parameters import NumCameras
from .render_setup import (
    create_cameras_on_one_ring,
    create_cameras_on_sphere,
    setup_render_settings,
)
from .utils import isolate_object


@dataclass
class ProcessParameters:
    """Dataclass of the Process Parameters."""

    output_path: str
    sd_version: str
    mesh_complexity: str
    checkpoint_path: str
    num_loras: int
    lora_models: list[Any]
    use_ipadapter: bool
    ipadapter_strength: float
    controlnet_union_path: str | None = None
    union_controlnet_strength: float | None = None
    depth_controlnet_path: str | None = None
    depth_controlnet_strength: float | None = None
    canny_controlnet_path: str | None = None
    canny_controlnet_strength: float | None = None
    normal_controlnet_path: str | None = None
    normal_controlnet_strength: float | None = None


def extract_process_parameters_from_context(
    context: bpy.types.Context,
) -> ProcessParameters:
    scene = context.scene
    return ProcessParameters(
        output_path=getattr(scene, "output_path", None),
        sd_version=getattr(scene, "sd_version", None),
        mesh_complexity=getattr(scene, "mesh_complexity", None),
        checkpoint_path=getattr(scene, "checkpoint_path", None),
        num_loras=getattr(scene, "num_loras", 0),
        lora_models=getattr(scene, "lora_models", []),
        use_ipadapter=getattr(scene, "use_ipadapter", False),
        ipadapter_strength=getattr(scene, "ipadapter_strength", 0.0),
        controlnet_union_path=getattr(scene, "controlnet_union_path", None),
        union_controlnet_strength=getattr(scene, "union_controlnet_strength", None),
        depth_controlnet_path=getattr(scene, "depth_controlnet_path", None),
        depth_controlnet_strength=getattr(scene, "depth_controlnet_strength", None),
        canny_controlnet_path=getattr(scene, "canny_controlnet_path", None),
        canny_controlnet_strength=getattr(scene, "canny_controlnet_strength", None),
        normal_controlnet_path=getattr(scene, "normal_controlnet_path", None),
        normal_controlnet_strength=getattr(scene, "normal_controlnet_strength", None),
    )


def create_similar_angle_image(
    normal_array: NDArray,
    position_array: NDArray,
    camera_obj: bpy.types.Camera,
) -> NDArray:
    """Create the similarity angle image.

    Create an image where each pixel's intensity represents how aligned the normal
    vector at that point is with the direction vector from the point to the camera.

    Args:
        normal_array (NDArray): NumPy array of shape (height, width, 3) containing
                                normal vectors.
        position_array (NDArray):   NumPy array of shape (height, width, 3) containing
                                    positions in global coordinates.
        camera_obj (bpy.types.Camera):  Blender camera object to get the camera position
                                        in global coordinates.

    Returns:
        NDArray: A NumPy array (height, width) with values ranging from 0 to 1,
                where 1 means perfect alignment.

    """
    # Extract camera position in global coordinates
    camera_position = np.array(camera_obj.matrix_world.to_translation())

    # Calculate direction vectors from each point to the camera
    direction_to_camera = position_array - camera_position[None, None, :]

    # Normalize the normal vectors and direction vectors
    normal_array_normalized = normal_array / np.linalg.norm(
        normal_array,
        axis=2,
        keepdims=True,
    )
    direction_to_camera_normalized = direction_to_camera / np.linalg.norm(
        direction_to_camera,
        axis=2,
        keepdims=True,
    )

    # Compute the dot product between the normalized vectors
    alignment = np.sum(normal_array_normalized * direction_to_camera_normalized, axis=2)

    # Ensure values are in range -1 to 1;
    # clip them just in case due to floating-point errors
    alignment = np.clip(alignment, -1.0, 1.0)

    # and invert
    similar_angle_image = -1 * alignment

    similar_angle_image[np.isnan(similar_angle_image)] = 0

    # Return the final similarity image
    return similar_angle_image


def load_img_to_numpy(img_path: str) -> NDArray:
    """Load an image as a Blender image and converts it to a NumPy array.

    Args:
        img_path (str): The path to the image.

    Returns:
        np.ndarray:A NumPy array representation of the image.

    """
    # Load image using Blender's bpy
    bpy.data.images.load(img_path)

    # Get the file name from the path
    img_file_name = Path(img_path).name

    # Access the image by name after loading
    img_bpy = bpy.data.images.get(img_file_name)

    return bpy_img_to_numpy(img_bpy)


def bpy_img_to_numpy(img_bpy: bpy.types.Image) -> NDArray:
    """Turn a bpy image to a numpy array.

    Args:
        img_bpy (bpy.types.Image): _description_

    Returns:
        NDArray: _description_
    """
    # Get image dimensions
    width, height = img_bpy.size

    # Determine the number of channels
    num_channels = len(img_bpy.pixels) // (width * height)

    # Convert the flat pixel array to a NumPy array
    pixels = np.array(img_bpy.pixels[:], dtype=np.float32)

    # Reshape the array to match the image's dimensions and channels
    image_array = pixels.reshape((height, width, num_channels))

    return np.flipud(image_array)


def prepare_scene(obj: bpy.types.Object) -> dict[str, Any]:
    """Backup all other objects and isolate the target object to work with."""
    backup_data = isolate_object(obj)
    bpy.context.view_layer.objects.active = obj
    return backup_data


def bake_uv_views(context: bpy.types.Context, obj: bpy.types.Object) -> dict:
    return {
        "normal": bake_geometry_channel_to_array(
            obj,
            "Normal",
            resolution=int(context.scene.texture_resolution),
        ),
        "position": bake_geometry_channel_to_array(
            obj,
            "Position",
            resolution=int(context.scene.texture_resolution),
        ),
    }


def render_views(context: bpy.types.Context, obj: bpy.types.Object) -> dict:
    """Render views and save to folders.

    Args:
        context (bpy.context): _description_
        obj (bpy.types.Object): _description_

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
    if num_cameras == NumCameras.four.value:
        cameras = create_cameras_on_one_ring(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="RenderCam",
        )
    elif num_cameras in [NumCameras.nine.value, NumCameras.sixteen.value]:
        cameras = create_cameras_on_sphere(
            num_cameras=num_cameras,
            max_size=max_size,
            name_prefix="RenderCam",
        )
    else:
        msg = "Only 4, 9, or 16 cameras are supported."
        raise ValueError(msg)

    # Set up render nodes
    output_nodes = setup_render_settings(context)

    render_img_folders = {
        "depth": output_nodes["depth"].base_path,
        "normal": output_nodes["normal"].base_path,
        "uv": output_nodes["uv"].base_path,
        "position": output_nodes["position"].base_path,
        # Facing images are in the folder "facing" which is not rendered but created
        "facing": Path(output_nodes["uv"].base_path).parent / "render_facing",
    }

    # Create the facing images folder if it does not exist
    Path(render_img_folders["facing"]).mkdir(parents=True, exist_ok=True)

    # Render for each camera
    for cam_idx, camera in enumerate(cameras):
        for output_node in output_nodes:
            new_path = (
                Path(output_nodes[output_node].base_path) / f"camera_{cam_idx:02d}"
            )

            # Create the new path if it does not exist
            new_path.mkdir(parents=True, exist_ok=True)

            # Set the output path for the output node
            output_nodes[output_node].base_path = str(new_path)

        context.scene.camera = camera
        bpy.ops.render.render(write_still=True)
        # Create the facing images
        save_facing_images(
            output_nodes=output_nodes,
            camera=camera,
            cam_idx=cam_idx,
            context=context,
        )

    return render_img_folders


def save_facing_images(
    output_nodes: dict[str, bpy.types.CompositorNodeOutputFile],
    camera: bpy.types.Object,
    cam_idx: int = 0,
    context: bpy.types.Context = bpy.context,
) -> None:
    """Save facing images for the camera."""

    frame_index = context.scene.frame_current

    # TODO: Replace Image.open with openexr_numpy or something that supports EXR

    normal_path = Path(output_nodes["normal"].base_path) / (
        str(output_nodes["normal"].file_slots[0].path) + f"{frame_index:04d}.exr"
    )
    normal_array = np.array(Image.open(normal_path))

    position_path = Path(output_nodes["position"].base_path) / (
        str(output_nodes["position"].file_slots[0].path) + f"{frame_index:04d}.png"
    )
    position_array = np.array(Image.open(position_path))

    facing_image_array = create_similar_angle_image(
        normal_array=normal_array,
        position_array=position_array,
        camera=camera,
    )
    # save the facing image to the output folder
    facing_image = Image.fromarray(facing_image_array)

    new_folder_path = Path(output_nodes["Facing"].base_path) / f"camera_{cam_idx:02d}"
    new_file_path = new_folder_path / f"facing_{frame_index:04d}.exr"

    # save the image
    new_folder_path.mkdir(parents=True, exist_ok=True)
    facing_image.save(new_file_path)


def bake_geometry_channel_to_array(
    obj: bpy.types.Object,
    channel: str = "Position",
    resolution: int = 1024,
) -> NDArray[np.float32]:
    """Bake a geometry channel ('Position' or 'Normal') to a NumPy array image.

    Args:
        obj (bpy.types.Object): The mesh object to bake.
        channel (str, optional): 'Position' or 'Normal' from the Geometry node.
                                 Defaults to "Position".
        resolution (int, optional): Texture resolution. Defaults to 1024.

    Raises:
        ValueError: _description_

    Returns:
        NDArray[np.float32]:  A float32 NumPy array of shape (height, width, 4).

    """
    if obj is None or obj.type != "MESH":
        msg = "Input object must be a mesh."
        raise ValueError(msg)

    # Create float32 image
    img = bpy.data.images.new(
        name="__bake_temp",
        width=resolution,
        height=resolution,
        alpha=True,
        float_buffer=True,
    )
    img.colorspace_settings.name = "Non-Color"

    # Create temp material
    mat = bpy.data.materials.new(name="__bake_mat")
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    links = mat.node_tree.links
    nodes.clear()

    # Create shader nodes
    geo = nodes.new("ShaderNodeNewGeometry")
    geo.location = (0, 0)

    emission = nodes.new("ShaderNodeEmission")
    emission.location = (200, 0)

    out = nodes.new("ShaderNodeOutputMaterial")
    out.location = (400, 0)

    links.new(geo.outputs[channel], emission.inputs["Color"])
    links.new(emission.outputs["Emission"], out.inputs["Surface"])

    # Add and activate image texture node for baking
    img_node = nodes.new("ShaderNodeTexImage")
    img_node.image = img
    img_node.select = True
    nodes.active = img_node

    # Assign material
    original_materials = list(obj.data.materials)
    obj.data.materials.clear()
    obj.data.materials.append(mat)
    obj.active_material_index = 0

    # Use Cycles and bake
    bpy.context.scene.render.engine = "CYCLES"
    bpy.context.view_layer.objects.active = obj
    bpy.context.scene.cycles.bake_type = "EMIT"
    bpy.ops.object.bake(type="EMIT", use_clear=True)

    # Extract pixels as numpy array
    img_pixels = np.array(img.pixels[:], dtype=np.float32)
    img_pixels = img_pixels.reshape((img.size[1], img.size[0], 4))  # RGBA

    # Cleanup temporary data
    bpy.data.images.remove(img, do_unlink=True)
    bpy.data.materials.remove(mat, do_unlink=True)
    obj.data.materials.clear()
    for m in original_materials:
        obj.data.materials.append(m)

    return img_pixels
