from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import bpy
import numpy as np
from numpy.typing import NDArray
from PIL import Image as PILImage
from PIL import ImageDraw

from .diagnostics import get_logger
from .diffusedtexture.uv_seams import (
    UV_EPSILON,
    SeamTopologyAssets,
    UVSeamCandidate,
    empty_float_array,
    empty_int_array,
    empty_yx_array,
    normalize_uv_vector,
)
from .diffusedtexture.uv_seams import (
    build_uv_seam_topology_assets as rasterize_uv_seam_topology_assets,
)
from .operation_mode import OperationMode, validate_operation_mode
from .render_setup import (
    clear_render_output_paths,
    create_cameras_on_sphere,
    create_cameras_on_two_rings,
    find_output_node_image_path,
    get_output_node_directory,
    set_output_node_directory,
    setup_render_settings,
)
from .utils import isolate_object

_logger = get_logger("blender_operations")

_MANAGED_MATERIAL_NAME_PREFIX = "GeneratedTextureMaterial"
_MANAGED_MATERIAL_TAG = "diffused_texture_managed_material"
_MANAGED_MATERIAL_OWNER = "diffused_texture_managed_material_owner"
_MANAGED_NODE_ROLE = "diffused_texture_managed_node_role"
_TEXTURE_NODE_ROLE = "texture"
_BSDF_NODE_ROLE = "bsdf"
_OUTPUT_NODE_ROLE = "output"


@dataclass
class UVPassAssets:
    """Assets baked in UV space for the dedicated UV generation mode."""

    normal_map: NDArray[np.float32]
    position_map: NDArray[np.float32]
    uv_layout: NDArray[np.float32]
    surface_mask: NDArray[np.uint8]
    seam_line_mask: NDArray[np.uint8] | None = None
    seam_link_source_yx: NDArray[np.int32] = field(default_factory=empty_yx_array)
    seam_link_target_yx: NDArray[np.int32] = field(default_factory=empty_yx_array)
    seam_link_weight: NDArray[np.float32] = field(default_factory=empty_float_array)
    seam_link_edge_id: NDArray[np.int32] = field(default_factory=empty_int_array)
    seam_link_t: NDArray[np.float32] = field(default_factory=empty_float_array)
    seam_unresolved_link_mask: NDArray[np.uint8] | None = None


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
    operation_mode: OperationMode
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
    dtype: Literal["float16", "bfloat16"] | None
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


def apply_texture_to_object(obj: bpy.types.Object, output_path: Path | str) -> None:
    """Apply the texture to the given object.

    Args:
        obj (bpy.types.Object): The Blender object to apply the texture to.
        output_path (Path): The path to the texture file.
    """
    if obj is None or obj.type != "MESH":
        msg = "Input object must be a mesh."
        raise ValueError(msg)

    img = bpy.data.images.load(str(output_path), check_existing=True)
    mat = _get_or_create_managed_material(obj)
    tex_image_node = _ensure_managed_material_node_tree(mat)
    tex_image_node.image = img
    _assign_managed_material_to_slots(obj, mat)


def _managed_material_name(obj: bpy.types.Object) -> str:
    return f"{_MANAGED_MATERIAL_NAME_PREFIX}_{obj.name}"


def _mark_material_as_managed(
    mat: bpy.types.Material,
    obj: bpy.types.Object,
) -> bpy.types.Material:
    mat[_MANAGED_MATERIAL_TAG] = True
    mat[_MANAGED_MATERIAL_OWNER] = obj.name
    return mat


def _is_managed_material_for_object(
    mat: bpy.types.Material | None,
    obj: bpy.types.Object,
) -> bool:
    if mat is None:
        return False
    return bool(mat.get(_MANAGED_MATERIAL_TAG)) and (
        mat.get(_MANAGED_MATERIAL_OWNER) == obj.name
    )


def _find_managed_material(obj: bpy.types.Object) -> bpy.types.Material | None:
    for mat in obj.data.materials:
        if _is_managed_material_for_object(mat, obj):
            return mat

    for mat in bpy.data.materials:
        if _is_managed_material_for_object(mat, obj):
            return mat

    return None


def _get_or_create_managed_material(obj: bpy.types.Object) -> bpy.types.Material:
    mat = _find_managed_material(obj)
    if mat is None:
        mat = bpy.data.materials.new(name=_managed_material_name(obj))
    return _mark_material_as_managed(mat, obj)


def _clear_links(links: bpy.types.bpy_prop_collection) -> None:
    for link in list(links):
        links.remove(link)


def _ensure_managed_node(
    nodes: bpy.types.Nodes,
    node_type: str,
    *,
    role: str,
    name: str,
    location: tuple[int, int],
) -> bpy.types.Node:
    matches = [node for node in nodes if node.get(_MANAGED_NODE_ROLE) == role]
    valid_matches = [node for node in matches if node.bl_idname == node_type]

    for node in matches:
        if node not in valid_matches:
            nodes.remove(node)

    if valid_matches:
        primary = valid_matches[0]
        for duplicate in valid_matches[1:]:
            nodes.remove(duplicate)
    else:
        primary = nodes.new(type=node_type)

    primary[_MANAGED_NODE_ROLE] = role
    primary.name = name
    primary.label = name
    primary.location = location
    return primary


def _ensure_managed_material_node_tree(
    mat: bpy.types.Material,
) -> bpy.types.ShaderNodeTexImage:
    mat.use_nodes = True
    node_tree = mat.node_tree
    if node_tree is None:
        msg = "Managed material must have a node tree."
        raise RuntimeError(msg)

    nodes = node_tree.nodes
    links = node_tree.links

    if not any(node.get(_MANAGED_NODE_ROLE) for node in nodes):
        nodes.clear()

    tex_image_node = _ensure_managed_node(
        nodes,
        "ShaderNodeTexImage",
        role=_TEXTURE_NODE_ROLE,
        name="DiffusedTexture_Texture",
        location=(-400, 0),
    )
    principled_node = _ensure_managed_node(
        nodes,
        "ShaderNodeBsdfPrincipled",
        role=_BSDF_NODE_ROLE,
        name="DiffusedTexture_Principled",
        location=(-80, 0),
    )
    output_node = _ensure_managed_node(
        nodes,
        "ShaderNodeOutputMaterial",
        role=_OUTPUT_NODE_ROLE,
        name="DiffusedTexture_Output",
        location=(220, 0),
    )

    _clear_links(links)
    links.new(
        tex_image_node.outputs["Color"],
        principled_node.inputs["Base Color"],
    )
    links.new(
        principled_node.outputs["BSDF"],
        output_node.inputs["Surface"],
    )

    return tex_image_node


def _assign_managed_material_to_slots(
    obj: bpy.types.Object,
    mat: bpy.types.Material,
) -> None:
    if not obj.data.materials:
        obj.data.materials.append(mat)
        return

    for index in range(len(obj.data.materials)):
        obj.data.materials[index] = mat


def blendercs_to_ccs(
    points_bcs: np.ndarray,
    camera: bpy.types.Camera,
    rotation_only: bool = False,  # noqa: FBT001, FBT002
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
    R_blender_to_cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])  # noqa: N806
    return np.dot(R_blender_to_cv, point_3d_cam.T).T


def create_depth_condition(
    depth_image_path: str,
    invalid_depth: float = 1e10,
) -> NDArray[np.float32]:
    depth_array = load_img_to_numpy(depth_image_path)[..., 0]

    # Replace large invalid values with NaN
    depth_array[depth_array >= invalid_depth] = np.nan

    # If all values are invalid, return a zero image
    if np.all(np.isnan(depth_array)):
        return np.zeros_like(depth_array, dtype=np.float32)[..., np.newaxis]

    # Invert the depth values so that closer objects have higher values
    depth_array = np.nanmax(depth_array) - depth_array

    # Normalize the depth array to range [0, 1]
    depth_array -= np.nanmin(depth_array)
    depth_range = np.nanmax(depth_array)
    if depth_range > 0:
        depth_array /= depth_range

    # Add a small margin to the background
    depth_array += 10 / 255.0  # Approximately 0.039

    # normalize
    depth_array[np.isnan(depth_array)] = 0
    max_val = np.nanmax(depth_array)
    if max_val > 0:
        depth_array /= max_val
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
    operation_mode = validate_operation_mode(
        getattr(scene, "operation_mode", "PARALLEL_IMG"),
    )

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

    ipadapter_image = getattr(scene, "ipadapter_image", None)
    if ipadapter_image is not None:
        ipadapter_image = bpy_img_to_numpy(ipadapter_image)

    return ProcessParameter(
        my_mesh_object=getattr(scene, "my_mesh_object", ""),
        my_uv_map=getattr(scene, "my_uv_map", ""),
        my_prompt=getattr(scene, "my_prompt", ""),
        my_negative_prompt=getattr(scene, "my_negative_prompt", None),
        denoise_strength=getattr(scene, "denoise_strength", 0.0),
        num_inference_steps=getattr(scene, "num_inference_steps", 50),
        guidance_scale=getattr(scene, "guidance_scale", None),
        operation_mode=operation_mode,
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
        dtype=getattr(scene, "dtype", None),
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
        ipadapter_image=ipadapter_image,
        num_loras=getattr(scene, "num_loras", 0),
        lora_models=lora_models,
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


def export_uv_layout(
    obj: bpy.types.Object,
    export_path: str | Path,
    uv_map_name: str | None = None,
    size: tuple[int, int] = (1024, 1024),
) -> None:
    """Export the UV layout for the given mesh object."""
    if obj is None or obj.type != "MESH":
        msg = "Input object must be a mesh."
        raise ValueError(msg)

    bpy.context.view_layer.objects.active = obj
    if obj.mode != "OBJECT":
        bpy.ops.object.mode_set(mode="OBJECT")

    _resolve_uv_layout_layer(obj, uv_map_name)
    export_path = Path(export_path)
    export_path.parent.mkdir(parents=True, exist_ok=True)

    if getattr(getattr(bpy, "app", None), "background", False):
        if export_path.suffix.lower() != ".png":
            msg = "Background UV layout export currently supports PNG output only."
            raise ValueError(msg)
        _save_uv_layout_png(
            _rasterize_uv_layout_array(
                obj,
                uv_map_name=uv_map_name,
                size=size,
            ),
            export_path,
        )
        return

    bpy.ops.uv.export_layout(
        filepath=str(export_path),
        size=size,
        opacity=1.0,
        export_all=False,
    )


def _resolve_uv_layout_layer(
    obj: bpy.types.Object,
    uv_map_name: str | None = None,
) -> bpy.types.MeshUVLoopLayer:
    uv_layers = obj.data.uv_layers
    if not uv_layers:
        msg = f"No UV maps found for object {obj.name}."
        raise ValueError(msg)

    if uv_map_name:
        uv_layer = uv_layers.get(uv_map_name)
        if uv_layer is None:
            msg = f"UV map {uv_map_name} not found on object {obj.name}."
            raise ValueError(msg)
        uv_layers.active = uv_layer
        return uv_layer

    uv_layer = uv_layers.active
    if uv_layer is None:
        msg = f"No active UV map found for object {obj.name}."
        raise ValueError(msg)
    return uv_layer


def _uv_to_layout_xy(
    uv: Sequence[float],
    width: int,
    height: int,
) -> tuple[float, float]:
    return (
        float(uv[0]) * float(max(width - 1, 1)),
        (1.0 - float(uv[1])) * float(max(height - 1, 1)),
    )


def _rasterize_uv_layout_array(
    obj: bpy.types.Object,
    uv_map_name: str | None = None,
    size: tuple[int, int] = (1024, 1024),
) -> NDArray[np.float32]:
    width, height = size
    uv_layer = _resolve_uv_layout_layer(obj, uv_map_name)
    mesh = obj.data
    image = PILImage.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)

    polygons: list[list[tuple[float, float]]] = []
    for polygon in mesh.polygons:
        loop_indices = list(polygon.loop_indices)
        if len(loop_indices) < 3:  # noqa: PLR2004
            continue

        points = [
            _uv_to_layout_xy(uv_layer.data[loop_index].uv, width, height)
            for loop_index in loop_indices
        ]
        polygons.append(points)
        draw.polygon(points, fill=(255, 255, 255, 255))

    for points in polygons:
        draw.line(
            [*points, points[0]],
            fill=(0, 0, 0, 255),
            width=1,
        )

    return (np.asarray(image, dtype=np.float32) / 255.0).astype(np.float32)


def _save_uv_layout_png(
    uv_layout: NDArray[np.float32],
    export_path: str | Path,
) -> None:
    image = PILImage.fromarray(
        np.clip(np.rint(uv_layout * 255.0), 0, 255).astype(np.uint8),
        mode="RGBA",
    )
    image.save(Path(export_path))


def _surface_mask_from_uv_layout(uv_layout: NDArray[np.float32]) -> NDArray[np.uint8]:
    if uv_layout.ndim != 3 or uv_layout.shape[2] < 4:  # noqa: PLR2004
        msg = "UV layout image must be RGBA."
        raise ValueError(msg)

    return (uv_layout[..., 3] > 0).astype(np.uint8) * 255


def load_img_to_numpy(img_path: str | Path) -> NDArray:
    """Load an image as a Blender image and converts it to a float32 NumPy array.

    Args:
        img_path (str | Path): The path to the image.

    Returns:
        np.ndarray: A NumPy array representation of the image.
    """
    img_bpy = bpy.data.images.load(str(img_path), check_existing=False)

    try:
        return bpy_img_to_numpy(img_bpy)
    finally:
        if img_bpy.name in bpy.data.images:
            bpy.data.images.remove(img_bpy)


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
    original_object_visibility = {
        scene_obj.name: {
            "hide_viewport": scene_obj.hide_get(),
            "hide_render": scene_obj.hide_render,
        }
        for scene_obj in bpy.data.objects
    }

    backup_data = isolate_object(obj)
    backup_data["original_object_visibility"] = original_object_visibility
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    backup_data["original_scene_camera"] = getattr(scene, "camera", None)
    backup_data["original_render_settings"] = {
        "engine": getattr(scene.render, "engine", None),
        "resolution_x": getattr(scene.render, "resolution_x", None),
        "resolution_y": getattr(scene.render, "resolution_y", None),
        "resolution_percentage": getattr(scene.render, "resolution_percentage", None),
        "filepath": getattr(scene.render, "filepath", None),
        "filter_size": getattr(scene.render, "filter_size", None),
        "film_transparent": getattr(scene.render, "film_transparent", None),
    }

    image_settings = getattr(scene.render, "image_settings", None)
    backup_data["original_image_settings"] = {
        "file_format": (
            getattr(image_settings, "file_format", None)
            if image_settings is not None
            else None
        ),
        "color_depth": (
            getattr(image_settings, "color_depth", None)
            if image_settings is not None
            else None
        ),
    }

    cycles_settings = getattr(scene, "cycles", None)
    if cycles_settings is not None:
        backup_data["original_cycles_settings"] = {
            "samples": getattr(cycles_settings, "samples", None),
            "use_denoising": getattr(cycles_settings, "use_denoising", None),
            "use_light_tree": getattr(cycles_settings, "use_light_tree", None),
            "max_bounces": getattr(cycles_settings, "max_bounces", None),
            "diffuse_bounces": getattr(cycles_settings, "diffuse_bounces", None),
            "glossy_bounces": getattr(cycles_settings, "glossy_bounces", None),
            "transmission_bounces": getattr(
                cycles_settings,
                "transmission_bounces",
                None,
            ),
            "volume_bounces": getattr(cycles_settings, "volume_bounces", None),
            "transparent_max_bounces": getattr(
                cycles_settings,
                "transparent_max_bounces",
                None,
            ),
        }

    backup_data["original_view_layer_passes"] = {
        "use_pass_z": getattr(view_layer, "use_pass_z", None),
        "use_pass_normal": getattr(view_layer, "use_pass_normal", None),
        "use_pass_uv": getattr(view_layer, "use_pass_uv", None),
        "use_pass_position": getattr(view_layer, "use_pass_position", None),
    }

    # Snapshot compositor state
    backup_data["original_use_nodes"] = getattr(scene, "use_nodes", None)
    backup_data["original_use_compositing"] = getattr(
        scene.render,
        "use_compositing",
        None,
    )

    view_layer.objects.active = obj
    return backup_data


def restore_scene(  # noqa: C901, PLR0912, PLR0915
    backup_data: dict,
    cameras: list[bpy.types.Object] | None,
) -> None:
    """Restore the original Scene.

    Restore object transform (matrix_world if available), unhide others,
    delete temp cameras, TODO: restore original camera/rendering settings.

    """
    obj = backup_data["target_object"]

    # Restore transform (matches isolate_object which edited matrix_world.translation)
    try:
        if (
            "original_matrix_world" in backup_data
            and backup_data["original_matrix_world"] is not None
        ):
            obj.matrix_world = backup_data["original_matrix_world"].copy()
        # Back-compat: if only location was stored
        elif "original_location" in backup_data:
            obj.location = backup_data["original_location"]
    except ReferenceError:
        _logger.debug(
            (
                "Skipping transform restore because the target object reference is "
                "no longer valid."
            ),
            exc_info=True,
        )

    object_visibility = backup_data.get("original_object_visibility")
    if isinstance(object_visibility, dict):
        for object_name, visibility in object_visibility.items():
            scene_obj = bpy.data.objects.get(object_name)
            if scene_obj is None or not isinstance(visibility, dict):
                continue

            hide_viewport = visibility.get("hide_viewport")
            if hide_viewport is not None:
                scene_obj.hide_set(hide_viewport)

            hide_render = visibility.get("hide_render")
            if hide_render is not None:
                scene_obj.hide_render = hide_render
    else:
        # Back-compat for backups created before visibility snapshots were added.
        for o in backup_data.get("hidden_objects", []):
            try:
                object_name = o.name if o else None
            except ReferenceError:
                continue

            if object_name and object_name in bpy.data.objects:
                o.hide_set(False)  # noqa: FBT003
                o.hide_render = False

    scene = bpy.context.scene
    view_layer = bpy.context.view_layer

    if "original_scene_camera" in backup_data:
        scene.camera = backup_data["original_scene_camera"]

    render_settings = backup_data.get("original_render_settings", {})
    for key, value in render_settings.items():
        if value is not None and hasattr(scene.render, key):
            setattr(scene.render, key, value)

    image_settings = getattr(scene.render, "image_settings", None)
    image_backup = backup_data.get("original_image_settings", {})
    if image_settings is not None:
        for key, value in image_backup.items():
            if value is not None and hasattr(image_settings, key):
                setattr(image_settings, key, value)

    cycles_settings = getattr(scene, "cycles", None)
    cycles_backup = backup_data.get("original_cycles_settings", {})
    if cycles_settings is not None:
        for key, value in cycles_backup.items():
            if value is not None and hasattr(cycles_settings, key):
                setattr(cycles_settings, key, value)

    view_layer_passes = backup_data.get("original_view_layer_passes", {})
    for key, value in view_layer_passes.items():
        if value is not None and hasattr(view_layer, key):
            setattr(view_layer, key, value)

    # Restore compositor state
    original_use_nodes = backup_data.get("original_use_nodes")
    if original_use_nodes is not None and hasattr(scene, "use_nodes"):
        scene.use_nodes = original_use_nodes

    original_use_compositing = backup_data.get("original_use_compositing")
    if original_use_compositing is not None and hasattr(
        scene.render,
        "use_compositing",
    ):
        scene.render.use_compositing = original_use_compositing

    # Remove compositor output nodes created during rendering
    try:
        from .render_setup import get_scene_compositor_node_tree

        node_tree = get_scene_compositor_node_tree(scene)
        addon_node_names = {
            "depth_output",
            "normal_output",
            "uv_output",
            "position_output",
            "DiffusedTexture_RenderLayers",
        }
        for node in list(node_tree.nodes):
            if node.name in addon_node_names:
                node_tree.nodes.remove(node)
    except Exception:  # noqa: BLE001
        _logger.debug(
            "Failed to clean up compositor nodes during scene restore.",
            exc_info=True,
        )

    # Delete the temporary cameras used during processing
    for cam in cameras or []:
        if cam and cam.name in bpy.data.objects:
            bpy.data.objects.remove(cam, do_unlink=True)

    # Make sure depsgraph reflects the changes
    view_layer.update()


def _activate_selected_uv_map(
    context: bpy.types.Context,
    obj: bpy.types.Object,
) -> None:
    uv_map_name = getattr(context.scene, "my_uv_map", "")
    if uv_map_name and obj.data.uv_layers.get(uv_map_name):
        obj.data.uv_layers.active = obj.data.uv_layers[uv_map_name]


@dataclass(frozen=True)
class _UVEdgeSide:
    edge_index: int
    vertex_start: int
    vertex_end: int
    uv_start: NDArray[np.float32]
    uv_end: NDArray[np.float32]
    interior_ref: NDArray[np.float32]


def _uv_layer_vector(
    uv_layer: bpy.types.MeshUVLoopLayer,
    loop_index: int,
) -> NDArray[np.float32]:
    return np.array(uv_layer.data[loop_index].uv, dtype=np.float32)


def _compute_uv_inward_direction(
    uv_start: NDArray[np.float32],
    uv_end: NDArray[np.float32],
    interior_ref: NDArray[np.float32],
) -> NDArray[np.float32]:
    segment = uv_end - uv_start
    if float(np.linalg.norm(segment)) <= UV_EPSILON:
        return np.zeros(2, dtype=np.float32)

    perp = normalize_uv_vector(np.array([-segment[1], segment[0]], dtype=np.float32))
    midpoint = 0.5 * (uv_start + uv_end)
    if float(np.dot(perp, interior_ref - midpoint)) < 0.0:
        perp = -perp
    return perp.astype(np.float32, copy=False)


def _orient_edge_side_to_mesh_edge(
    side: _UVEdgeSide,
    edge_vertices: tuple[int, int],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]] | None:
    edge_start, edge_end = edge_vertices
    if side.vertex_start == edge_start and side.vertex_end == edge_end:
        uv_start = side.uv_start
        uv_end = side.uv_end
    elif side.vertex_start == edge_end and side.vertex_end == edge_start:
        uv_start = side.uv_end
        uv_end = side.uv_start
    else:
        return None

    inward = _compute_uv_inward_direction(uv_start, uv_end, side.interior_ref)
    if float(np.linalg.norm(inward)) <= UV_EPSILON:
        return None
    return uv_start, uv_end, inward


def _edge_sides_are_uv_split(
    first_start: NDArray[np.float32],
    first_end: NDArray[np.float32],
    second_start: NDArray[np.float32],
    second_end: NDArray[np.float32],
    texture_resolution: int,
) -> bool:
    uv_epsilon = 0.5 / float(max(texture_resolution - 1, 1))
    endpoint_delta = max(
        float(np.linalg.norm(first_start - second_start)),
        float(np.linalg.norm(first_end - second_end)),
    )
    return endpoint_delta > uv_epsilon


def _selected_uv_layer_name(
    context: bpy.types.Context,
    obj: bpy.types.Object,
) -> str | None:
    uv_map_name = getattr(context.scene, "my_uv_map", "") or None
    if uv_map_name:
        return str(uv_map_name)
    active_uv = obj.data.uv_layers.active
    return active_uv.name if active_uv is not None else None


def _collect_uv_seam_candidates(  # noqa: C901
    context: bpy.types.Context,
    obj: bpy.types.Object,
    texture_resolution: int,
) -> list[UVSeamCandidate]:
    depsgraph = context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh = obj_eval.to_mesh(preserve_all_data_layers=True, depsgraph=depsgraph)

    try:
        if mesh is None or not mesh.uv_layers:
            return []

        uv_layer_name = _selected_uv_layer_name(context, obj)
        uv_layer = mesh.uv_layers.get(uv_layer_name) if uv_layer_name else None
        if uv_layer is None:
            uv_layer = mesh.uv_layers.active
        if uv_layer is None and len(mesh.uv_layers) > 0:
            uv_layer = mesh.uv_layers[0]
        if uv_layer is None:
            return []

        sides_by_edge: dict[int, list[_UVEdgeSide]] = {}
        for polygon in mesh.polygons:
            loop_indices = list(polygon.loop_indices)
            loop_total = len(loop_indices)
            if loop_total < 3:  # noqa: PLR2004
                continue

            for local_index, loop_index in enumerate(loop_indices):
                next_loop_index = loop_indices[(local_index + 1) % loop_total]
                prev_loop_index = loop_indices[local_index - 1]
                after_next_loop_index = loop_indices[(local_index + 2) % loop_total]
                loop = mesh.loops[loop_index]
                next_loop = mesh.loops[next_loop_index]
                uv_prev = _uv_layer_vector(uv_layer, prev_loop_index)
                uv_next = _uv_layer_vector(uv_layer, after_next_loop_index)
                side = _UVEdgeSide(
                    edge_index=int(loop.edge_index),
                    vertex_start=int(loop.vertex_index),
                    vertex_end=int(next_loop.vertex_index),
                    uv_start=_uv_layer_vector(uv_layer, loop_index),
                    uv_end=_uv_layer_vector(uv_layer, next_loop_index),
                    interior_ref=(0.5 * (uv_prev + uv_next)).astype(np.float32),
                )
                sides_by_edge.setdefault(side.edge_index, []).append(side)

        candidates: list[UVSeamCandidate] = []
        for edge_index, edge_sides in sides_by_edge.items():
            if len(edge_sides) != 2:  # noqa: PLR2004
                continue

            mesh_edge = mesh.edges[edge_index]
            edge_vertices = (int(mesh_edge.vertices[0]), int(mesh_edge.vertices[1]))
            first = _orient_edge_side_to_mesh_edge(edge_sides[0], edge_vertices)
            second = _orient_edge_side_to_mesh_edge(edge_sides[1], edge_vertices)
            if first is None or second is None:
                continue

            first_start, first_end, first_inward = first
            second_start, second_end, second_inward = second
            if not _edge_sides_are_uv_split(
                first_start,
                first_end,
                second_start,
                second_end,
                texture_resolution,
            ):
                continue

            candidates.append(
                UVSeamCandidate(
                    edge_id=edge_index,
                    uv_a_start=first_start.astype(np.float32, copy=False),
                    uv_a_end=first_end.astype(np.float32, copy=False),
                    uv_b_start=second_start.astype(np.float32, copy=False),
                    uv_b_end=second_end.astype(np.float32, copy=False),
                    inward_a=first_inward.astype(np.float32, copy=False),
                    inward_b=second_inward.astype(np.float32, copy=False),
                ),
            )

        return candidates
    finally:
        obj_eval.to_mesh_clear()


def _build_uv_seam_topology_assets(
    context: bpy.types.Context,
    obj: bpy.types.Object,
    surface_mask: NDArray[np.uint8],
    texture_resolution: int,
) -> SeamTopologyAssets:
    try:
        candidates = _collect_uv_seam_candidates(context, obj, texture_resolution)
        return rasterize_uv_seam_topology_assets(
            candidates,
            surface_mask,
            texture_resolution,
        )
    except Exception:  # noqa: BLE001
        _logger.debug("Failed to build topology UV seam assets.", exc_info=True)
        return rasterize_uv_seam_topology_assets([], surface_mask, texture_resolution)


def build_uv_pass_assets(
    context: bpy.types.Context,
    obj: bpy.types.Object,
) -> UVPassAssets:
    """Build UV-space assets for the dedicated UV generation path."""
    texture_resolution = int(context.scene.texture_resolution)
    _activate_selected_uv_map(context, obj)

    uv_layout_path = (
        Path(context.scene.output_path) / "RenderOutput" / "uv_mode_layout.png"
    )
    uv_layout_path.parent.mkdir(parents=True, exist_ok=True)

    export_uv_layout(
        obj,
        uv_layout_path,
        uv_map_name=getattr(context.scene, "my_uv_map", "") or None,
        size=(texture_resolution, texture_resolution),
    )

    try:
        uv_layout = load_img_to_numpy(uv_layout_path)
    finally:
        uv_layout_path.unlink(missing_ok=True)

    surface_mask = _surface_mask_from_uv_layout(uv_layout)
    seam_topology_assets = _build_uv_seam_topology_assets(
        context,
        obj,
        surface_mask,
        texture_resolution,
    )

    return UVPassAssets(
        normal_map=bake_geometry_channel_to_array(
            obj,
            "Normal",
            resolution=texture_resolution,
        ),
        position_map=bake_geometry_channel_to_array(
            obj,
            "Position",
            resolution=texture_resolution,
        ),
        uv_layout=uv_layout,
        surface_mask=surface_mask,
        seam_line_mask=seam_topology_assets.seam_line_mask,
        seam_link_source_yx=seam_topology_assets.seam_link_source_yx,
        seam_link_target_yx=seam_topology_assets.seam_link_target_yx,
        seam_link_weight=seam_topology_assets.seam_link_weight,
        seam_link_edge_id=seam_topology_assets.seam_link_edge_id,
        seam_link_t=seam_topology_assets.seam_link_t,
        seam_unresolved_link_mask=seam_topology_assets.seam_unresolved_link_mask,
    )


def render_views(
    context: bpy.types.Context,
    obj: bpy.types.Object,
) -> tuple[dict, list[bpy.types.Object]]:
    """Render views and save to folders.

    Args:
        context (bpy.context): Blender Context
        obj (bpy.types.Object): Blender Object to be rendered

    Raises:
        ValueError: If the number of cameras is not supported.

    Returns:
        dict: A dictionary containing the rendered image paths.
    """
    original_camera = context.scene.camera
    clear_render_output_paths(context.scene.output_path)

    # Set up cameras
    num_cameras = int(context.scene.num_cameras)
    max_size = max(obj.dimensions)

    # Set parameter
    num_cameras = int(context.scene.num_cameras)

    cameras: list[bpy.types.Object] = []

    # Create cameras based on the number specified in the scene
    if num_cameras == 4:  # noqa: PLR2004
        cameras = create_cameras_on_two_rings(
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

    try:
        _activate_selected_uv_map(context, obj)

        # Set up render nodes
        output_nodes = setup_render_settings(context, context.scene.render_resolution)

        render_img_folders = {
            "depth": get_output_node_directory(output_nodes["depth"]),
            "normal": get_output_node_directory(output_nodes["normal"]),
            "uv": get_output_node_directory(output_nodes["uv"]),
            # Facing images are in the folder "facing" which is not rendered but created
            "facing": str(
                Path(get_output_node_directory(output_nodes["uv"])).parent
                / "render_facing",
            ),
        }

        # Create the facing images folder if it does not exist
        Path(render_img_folders["facing"]).mkdir(parents=True, exist_ok=True)

        # Render for each camera
        for cam_idx, camera in enumerate(cameras):
            for output_node in output_nodes:
                new_path = (
                    Path(render_img_folders[output_node]) / f"camera_{cam_idx:02d}"
                )

                # Create the new path if it does not exist
                new_path.mkdir(parents=True, exist_ok=True)

                # Set the output path for the output node
                set_output_node_directory(output_nodes[output_node], new_path)

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

    except Exception:
        _logger.exception("Rendering views failed; cleaning up temporary cameras.")
        clear_render_output_paths(context.scene.output_path)
        context.scene.camera = original_camera
        for cam in cameras:
            if cam and cam.name in bpy.data.objects:
                bpy.data.objects.remove(cam, do_unlink=True)
        raise
    else:
        return render_img_folders, cameras


def save_normals_in_camera_coordinates(
    output_nodes: dict[str, bpy.types.CompositorNodeOutputFile],
    camera: bpy.types.Object,
) -> None:
    image_path = find_output_node_image_path(
        output_nodes["normal"],
        bpy.context.scene.frame_current,
    )

    normal_ccs = create_normal_condition(
        normal_img_path=str(image_path),
        camera_obj=camera,
    )

    save_numpy_to_exr(
        img_np=normal_ccs,
        filepath=str(image_path),
        name="normal_camera_coordinates",
    )


def save_depth_condition(
    output_nodes: dict[str, bpy.types.CompositorNodeOutputFile],
) -> None:
    """Save the depth condition as an image as stable diffusion uses in Controlnet."""
    image_path = find_output_node_image_path(
        output_nodes["depth"],
        bpy.context.scene.frame_current,
    )

    depth_sd = create_depth_condition(
        depth_image_path=str(image_path),
    )

    save_numpy_to_exr(
        img_np=depth_sd,
        filepath=str(image_path),
        name="depth_sd_like",
    )


def save_facing_images(
    output_nodes: dict[str, bpy.types.CompositorNodeOutputFile],
    cam_idx: int,
    context: bpy.types.Context = bpy.context,
) -> None:
    """Save facing images for the camera."""
    frame_index = context.scene.frame_current

    normal_path = find_output_node_image_path(
        output_nodes["normal"],
        frame_index,
    )

    normal_array = load_img_to_numpy(str(normal_path))

    facing_image_array = normal_array[..., 2]
    facing_image_array = 2 * facing_image_array
    facing_image_array -= 1  # Normalize to [-1, 1]
    facing_image_array = np.clip(facing_image_array, 0, 1)  # remove negative values

    new_folder_path = (
        Path(
            str(Path(get_output_node_directory(output_nodes["normal"])).parent).replace(
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
