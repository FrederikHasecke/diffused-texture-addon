from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import bpy
import numpy as np
from numpy.typing import NDArray

from .process_utils import blendercs_to_ccs
from .render_setup import (
    create_cameras_on_one_ring,
    create_cameras_on_sphere,
    setup_render_settings,
)
from .utils import isolate_object


@dataclass
class ProcessParameter:
    """Dataclass of the Process parameter."""

    # Blender specific parameter
    my_mesh_object: str
    my_uv_map: str

    # Stable Diffusion Settings
    my_prompt: str
    my_negative_prompt: str | None
    denoise_strength: float
    num_inference_steps: int
    guidance_scale: float | None

    # Texture Generation Settings
    operation_mode: Literal[
        "PARALLEL_IMG",
        "SEQUENTIAL_IMG",
        "PARA_SEQUENTIAL_IMG",
        "UV_PASS",
    ]
    subgrid_rows: int
    subgrid_cols: int
    mesh_complexity: Literal[
        "LOW",
        "MEDIUM",
        "HIGH",
    ]
    num_cameras: Literal[4, 9, 16]
    texture_resolution: Literal[
        "512",
        "1024",
        "2048",
        "4096",
    ]
    render_resolution: Literal[
        "1024",
        "2048",
        "4096",
        "8192",
    ]
    output_path: str
    texture_seed: int
    input_texture: bpy.types.Image | NDArray | None

    # Advanced Settings
    sd_version: Literal["sd15", "sdxl"] | None
    checkpoint_path: str
    custom_sd_resolution: int
    controlnet_union_path: str | None
    union_controlnet_strength: float | None
    depth_controlnet_path: str | None
    depth_controlnet_strength: float | None
    canny_controlnet_path: str | None
    canny_controlnet_strength: float | None
    normal_controlnet_path: str | None
    normal_controlnet_strength: float | None

    # IPAdapter Settings
    use_ipadapter: bool
    ipadapter_strength: float
    ipadapter_image: bpy.types.Image | NDArray | None

    # LoRA Settings
    num_loras: int
    lora_models: list[dict[str, str | float]]


def apply_texture(
    context: bpy.types.Context,
    output_path: str,
) -> None:
    """Apply the generated texture to the selected object.

    Args:
        context (bpy.types.Context): The Blender context.
        texture (NDArray[np.float32]): The texture to apply.
        output_path (str): The output path for the texture.
    """
    # Get the selected object
    selected_obj = bpy.data.objects.get(context.scene.my_mesh_object)

    # Apply the texture to the object
    apply_texture_to_object(selected_obj, output_path)


def apply_texture_to_object(obj: bpy.types.Object, output_path: Path) -> None:
    """Apply the texture to the given object.

    Args:
        obj (bpy.types.Object): The Blender object to apply the texture to.
        output_path (Path): The path to the texture file.
    """
    # Load the texture image
    img = bpy.data.images.load(str(output_path))

    # Create a new material if the object does not have one
    if not obj.data.materials:
        mat = bpy.data.materials.new(name="GeneratedTextureMaterial")
        obj.data.materials.append(mat)
    else:
        mat = obj.data.materials[0]

    # Enable 'Use Nodes' for the material
    mat.use_nodes = True
    nodes = mat.node_tree.nodes

    # Create an image texture node
    tex_image_node = nodes.new(type="ShaderNodeTexImage")
    tex_image_node.image = img

    # Link the image texture node to the material output node
    mat.node_tree.links.new(
        tex_image_node.outputs["Color"],
        mat.node_tree.nodes.get("Material Output").inputs["Surface"],
    )


def create_depth_condition(
    depth_image_path: str,
    invalid_depth: int = 1e10,
) -> NDArray[np.float32]:
    depth_array = load_img_to_numpy(depth_image_path)[..., 0]

    # Replace large invalid values with NaN
    depth_array[depth_array >= invalid_depth] = np.nan

    # Invert the depth values so that closer objects have higher values
    depth_array = np.nanmax(depth_array) - depth_array

    # Normalize the depth array to range [0, 1]
    depth_array -= np.nanmin(depth_array)
    depth_array /= np.nanmax(depth_array)

    # Add a small margin to the background
    depth_array += 10 / 255.0  # Approximately 0.039

    # normalize
    depth_array[np.isnan(depth_array)] = 0
    depth_array /= np.nanmax(depth_array)
    depth_array = np.clip(depth_array, 0, 1)

    return depth_array.astype(np.float32)[..., np.newaxis]  # Add channel dimension


def create_normal_condition(
    normal_img_path: str,
    camera_obj: bpy.types.Object,
) -> NDArray[np.float32]:
    normal_array = load_img_to_numpy(normal_img_path)

    normal_array = normal_array[..., :3]

    # Get image dimensions
    image_size = normal_array.shape[:2]

    # Flatten the normal array for transformation
    normal_pc = normal_array.reshape((-1, 3))

    # Rotate the normal vectors to the camera space without translating
    normal_pc = blendercs_to_ccs(
        points_bcs=normal_pc,
        camera=camera_obj,
        rotation_only=True,
    )

    # Map normalized values to the [0, 1] range for RGB display
    red_channel = ((normal_pc[:, 0] + 1) / 2).reshape(image_size)  # Normal X
    green_channel = ((normal_pc[:, 1] + 1) / 2).reshape(image_size)  # Normal Y
    blue_channel = ((normal_pc[:, 2] + 1) / 2).reshape(image_size)  # Normal Z

    # Adjust to shapenet colors
    blue_channel = 1 - blue_channel
    green_channel = 1 - green_channel

    # Stack channels into a single image
    normal_image = np.stack((red_channel, green_channel, blue_channel), axis=-1)
    normal_image = np.clip(normal_image, 0, 1)
    return normal_image.astype(np.float32)


def extract_process_parameter_from_context(
    context: bpy.types.Context,
) -> ProcessParameter:
    scene = context.scene

    # extract LoRA models from the scene
    lora_models = []
    for i in range(scene.num_loras):
        lora_model = scene.lora_models[i]
        if lora_model:
            lora_models.append(
                {
                    "path": lora_model.path,
                    "strength": lora_model.strength,
                },
            )

    return ProcessParameter(
        my_mesh_object=getattr(scene, "my_mesh_object", ""),
        my_uv_map=getattr(scene, "my_uv_map", ""),
        my_prompt=getattr(scene, "my_prompt", ""),
        my_negative_prompt=getattr(scene, "my_negative_prompt", None),
        denoise_strength=getattr(scene, "denoise_strength", 0.0),
        num_inference_steps=getattr(scene, "num_inference_steps", 50),
        guidance_scale=getattr(scene, "guidance_scale", None),
        operation_mode=getattr(scene, "operation_mode", "PARALLEL_IMG"),
        subgrid_rows=getattr(scene, "subgrid_rows", 2),
        subgrid_cols=getattr(scene, "subgrid_cols", 2),
        mesh_complexity=getattr(scene, "mesh_complexity", "MEDIUM"),
        num_cameras=getattr(scene, "num_cameras", 4),
        texture_resolution=getattr(scene, "texture_resolution", "1024"),
        render_resolution=getattr(scene, "render_resolution", "2048"),
        output_path=getattr(scene, "output_path", ""),
        texture_seed=getattr(scene, "texture_seed", 0),
        input_texture=getattr(scene, "input_texture", None),
        sd_version=getattr(scene, "sd_version", None),
        checkpoint_path=getattr(scene, "checkpoint_path", ""),
        custom_sd_resolution=getattr(scene, "custom_sd_resolution", 0),
        controlnet_union_path=getattr(scene, "controlnet_union_path", None),
        union_controlnet_strength=getattr(scene, "union_controlnet_strength", None),
        depth_controlnet_path=getattr(scene, "depth_controlnet_path", None),
        depth_controlnet_strength=getattr(scene, "depth_controlnet_strength", None),
        canny_controlnet_path=getattr(scene, "canny_controlnet_path", None),
        canny_controlnet_strength=getattr(scene, "canny_controlnet_strength", None),
        normal_controlnet_path=getattr(scene, "normal_controlnet_path", None),
        normal_controlnet_strength=getattr(scene, "normal_controlnet_strength", None),
        use_ipadapter=getattr(scene, "use_ipadapter", False),
        ipadapter_strength=getattr(scene, "ipadapter_strength", 0.0),
        ipadapter_image=getattr(scene, "ipadapter_image", None),
        num_loras=getattr(scene, "num_loras", 0),
        lora_models=getattr(scene, "lora_models", []),
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

    # Ensure the normal and position arrays are 3D
    if normal_array.ndim != 3 or position_array.ndim != 3:  # noqa: PLR2004
        msg = "Both normal_array and position_array must be 3D arrays."
        raise ValueError(msg)

    # Calculate direction vectors from each point to the camera
    direction_to_camera = position_array[..., :3] - camera_position[None, None, :]

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
    alignment[np.isnan(alignment)] = 0

    # and invert
    similar_angle_image = -1 * alignment

    similar_angle_image[np.isnan(similar_angle_image)] = 0

    return similar_angle_image.astype(np.float32)


def load_img_to_numpy(img_path: str | Path) -> NDArray:
    """Load an image as a Blender image and converts it to a float32 NumPy array.

    Args:
        img_path (str | Path): The path to the image.

    Returns:
        np.ndarray: A NumPy array representation of the image.
    """
    img_file_name = Path(img_path).name
    if img_file_name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[img_file_name])
    bpy.data.images.load(str(img_path))

    img_bpy = bpy.data.images.get(img_file_name)

    if img_bpy is None:
        msg = f"Image '{img_file_name}' could not be loaded into Blender."
        raise FileNotFoundError(msg)

    return bpy_img_to_numpy(img_bpy)


def bpy_img_to_numpy(img_bpy: bpy.types.Image) -> NDArray[np.float32]:
    """Turn a bpy image to a numpy array.

    Args:
        img_bpy (bpy.types.Image): Blender image.

    Returns:
        NDArray: NumPy array with shape (H, W, C), float32.
    """
    width, height = img_bpy.size
    num_channels = img_bpy.channels

    pixels = np.array(img_bpy.pixels[:], dtype=np.float32)
    image_array = pixels.reshape((height, width, num_channels))

    return np.flipud(image_array)


def numpy_to_bpy_img(img_np: np.ndarray, name: str = "TempImage") -> bpy.types.Image:
    """Converts a NumPy array to a Blender image.

    Args:
        img_np (np.ndarray): A NumPy array with shape (H, W, C) and dtype float32.
        name (str): Name of the image in Blender's data.

    Returns:
        bpy.types.Image: The Blender image object.
    """
    if img_np.dtype != np.float32:
        msg = "Input image must be a float32 NumPy array."
        raise ValueError(msg)

    if img_np.ndim != 3:  # noqa: PLR2004
        # Check if the input is a 2D array and convert it to 3D
        if img_np.ndim == 2:  # noqa: PLR2004
            img_np = img_np[:, :, np.newaxis]
        # If it is still not 3D, raise an error
        elif img_np.ndim > 3:  # noqa: PLR2004
            msg = "Input image must have 2 or 3 dimensions (H, W) or (H, W, C)."
            raise ValueError(msg)

    if img_np.shape[2] not in [1, 3, 4]:
        msg = "Input image must have 1, 3, or 4 channels (C)."
        raise ValueError(msg)

    height, width, channels = img_np.shape

    # Convert to RGBA if necessary
    if channels == 1:
        img_rgba = np.concatenate(
            [img_np] * 3 + [np.ones((height, width, 1), dtype=np.float32)],
            axis=2,
        )
    elif channels == 3:  # noqa: PLR2004
        img_rgba = np.concatenate(
            [img_np, np.ones((height, width, 1), dtype=np.float32)],
            axis=2,
        )
    else:
        img_rgba = img_np

    # Flatten pixels in Blenders top-down order
    pixels = np.flipud(img_rgba).reshape(-1).tolist()

    # Remove existing image if needed
    if name in bpy.data.images:
        bpy.data.images.remove(bpy.data.images[name])

    # Create image
    image = bpy.data.images.new(
        name=name,
        width=width,
        height=height,
        alpha=True,
        float_buffer=True,
    )
    image.pixels = pixels

    return image


def save_numpy_to_exr(
    img_np: np.ndarray,
    filepath: str,
    name: str = "TempImage",
) -> None:
    """Saves a NumPy image array as an EXR file via Blender.

    Args:
        img_np (np.ndarray): A float32 NumPy array of shape (H, W, C).
        filepath (str): Output path ending in `.exr`.
        name (str): Internal name for the temporary Blender image.
    """
    img_bpy = numpy_to_bpy_img(img_np, name=name)
    img_bpy.filepath_raw = filepath
    img_bpy.file_format = "OPEN_EXR"
    img_bpy.save()
    bpy.data.images.remove(img_bpy)


def prepare_scene(obj: bpy.types.Object) -> dict[str, Any]:
    """Backup all other objects and isolate the target object to work with."""
    backup_data = isolate_object(obj)
    bpy.context.view_layer.objects.active = obj
    return backup_data


def restore_scene(backup_data: dict, cameras: list[bpy.types.Object]) -> None:
    """Restore object position, unhide other objects, delete process cameras."""
    obj = backup_data["target_object"]
    obj.location = backup_data["original_location"]

    for o in backup_data["hidden_objects"]:
        o.hide_set(state=False)
        o.hide_render = False

    for cam in cameras:
        # delete the cameras created for rendering and their data
        bpy.data.objects.remove(cam, do_unlink=True)

    # update the scene to reflect the changes
    bpy.context.view_layer.update()


def bake_uv_views(
    context: bpy.types.Context,
    obj: bpy.types.Object,
) -> tuple[dict, None]:
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
    }, None


def render_views(
    context: bpy.types.Context,
    obj: bpy.types.Object,
) -> tuple[dict, list[bpy.types.Object]]:
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

    # Set parameter
    num_cameras = int(context.scene.num_cameras)

    # Create cameras based on the number specified in the scene
    if num_cameras == 4:  # noqa: PLR2004
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
    output_nodes = setup_render_settings(context, context.scene.render_resolution)

    render_img_folders = {
        "depth": output_nodes["depth"].base_path,
        "normal": output_nodes["normal"].base_path,
        "uv": output_nodes["uv"].base_path,
        "position": output_nodes["position"].base_path,
        # Facing images are in the folder "facing" which is not rendered but created
        "facing": str(Path(output_nodes["uv"].base_path).parent / "render_facing"),
    }

    # Create the facing images folder if it does not exist
    Path(render_img_folders["facing"]).mkdir(parents=True, exist_ok=True)

    # Render for each camera
    for cam_idx, camera in enumerate(cameras):
        for output_node in output_nodes:
            if cam_idx == 0:
                new_path = (
                    Path(output_nodes[output_node].base_path) / f"camera_{cam_idx:02d}"
                )
            else:
                new_path = (
                    Path(output_nodes[output_node].base_path).parent
                    / f"camera_{cam_idx:02d}"
                )

            # Create the new path if it does not exist
            new_path.mkdir(parents=True, exist_ok=True)

            # Set the output path for the output node
            output_nodes[output_node].base_path = str(new_path)

        context.scene.camera = camera

        # update the scene to reflect the camera change
        bpy.context.view_layer.update()

        bpy.ops.render.render(write_still=True)

        save_normals_in_camera_coordinates(output_nodes=output_nodes, camera=camera)

        save_depth_condition(output_nodes=output_nodes)

        # Create the facing images
        save_facing_images(
            output_nodes=output_nodes,
            cam_idx=cam_idx,
            context=context,
        )

    return render_img_folders, cameras


def save_normals_in_camera_coordinates(
    output_nodes: dict[str, bpy.types.CompositorNodeOutputFile],
    camera: bpy.types.Object,
) -> None:
    image_path = Path(output_nodes["normal"].base_path) / (
        str(output_nodes["normal"].file_slots[0].path)
        + f"{bpy.context.scene.frame_current:04d}.exr"
    )

    normal_ccs = create_normal_condition(
        normal_img_path=str(image_path),
        camera_obj=camera,
    )

    # overwrite the normal image with the camera coordinates
    normal_path = Path(output_nodes["normal"].base_path) / (
        str(output_nodes["normal"].file_slots[0].path)
        + f"{bpy.context.scene.frame_current:04d}.exr"
    )

    save_numpy_to_exr(
        img_np=normal_ccs,
        filepath=str(normal_path),
        name="normal_camera_coordinates",
    )


def save_depth_condition(
    output_nodes: dict[str, bpy.types.CompositorNodeOutputFile],
) -> None:
    """Save the depth condition as an image as stable diffusion uses in Controlnet."""
    image_path = Path(output_nodes["depth"].base_path) / (
        str(output_nodes["depth"].file_slots[0].path)
        + f"{bpy.context.scene.frame_current:04d}.exr"
    )

    depth_sd = create_depth_condition(
        depth_image_path=str(image_path),
    )

    # overwrite the normal image with the camera coordinates
    depth_path = Path(output_nodes["depth"].base_path) / (
        str(output_nodes["depth"].file_slots[0].path)
        + f"{bpy.context.scene.frame_current:04d}.exr"
    )

    save_numpy_to_exr(
        img_np=depth_sd,
        filepath=str(depth_path),
        name="depth_sd_like",
    )


def save_facing_images(
    output_nodes: dict[str, bpy.types.CompositorNodeOutputFile],
    cam_idx: int,
    context: bpy.types.Context = bpy.context,
) -> None:
    """Save facing images for the camera."""
    frame_index = context.scene.frame_current

    normal_path = (
        Path(output_nodes["normal"].base_path).parent
        / f"camera_{cam_idx:02d}"
        / (str(output_nodes["normal"].file_slots[0].path) + f"{frame_index:04d}.exr")
    )

    normal_array = load_img_to_numpy(str(normal_path))

    facing_image_array = normal_array[..., 2]
    facing_image_array = 2 * facing_image_array
    facing_image_array -= 1  # Normalize to [-1, 1]
    facing_image_array = np.clip(facing_image_array, 0, 1)  # remove negative values

    new_folder_path = (
        Path(
            str(Path(output_nodes["normal"].base_path).parent).replace(
                "render_normal",
                "render_facing",
            ),
        )
        / f"camera_{cam_idx:02d}"
    )
    new_file_path = new_folder_path / f"facing_{frame_index:04d}.exr"

    new_folder_path.mkdir(parents=True, exist_ok=True)

    save_numpy_to_exr(
        img_np=facing_image_array,
        filepath=str(new_file_path),
        name=f"facing_{cam_idx:02d}_{frame_index:04d}",
    )


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
