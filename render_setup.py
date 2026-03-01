import math
from pathlib import Path

import bpy
import mathutils


def farthest_point_ordering(positions: list[mathutils.Vector]) -> list[int]:
    """Sample the order of farthest points.

    Given a list of positions (Vectors), return an index ordering such that
    consecutive items are far apart. Uses greedy farthest-point traversal.
    """
    if not positions:
        return []

    ordered = []
    remaining = set(range(len(positions)))
    current = 0
    ordered.append(current)
    remaining.remove(current)

    while remaining:
        best_idx, best_dist = None, -1
        for idx in remaining:
            d = min((positions[idx] - positions[j]).length for j in ordered)
            if d > best_dist:
                best_idx, best_dist = idx, d

        if isinstance(best_idx, int):
            ordered.append(best_idx)
            remaining.remove(best_idx)

    return ordered


def create_cameras_on_one_ring(
    num_cameras: int,
    max_size: float,
    name_prefix: str = "Camera",
    fov: float = 22.5,
    elevation_ratio: float = 0.25,
) -> list[bpy.types.Object]:
    """Create single ring of cameras.

    Evenly distribute cameras on one horizontal ring, then reorder them
    with farthest-point ordering so consecutive cameras are far apart.
    """
    fov_rad = math.radians(fov)
    radius = 1.1 * (max_size * math.sqrt(2)) / math.tan(fov_rad)
    elevation = radius * elevation_ratio

    # Generate evenly spaced ring positions
    positions = []
    for i in range(num_cameras):
        theta = (2 * math.pi / num_cameras) * i
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        positions.append(mathutils.Vector((x, y, elevation)))

    # Reorder
    order = farthest_point_ordering(positions)

    cameras = []
    for k, idx in enumerate(order):
        loc = positions[idx]
        cam_data = bpy.data.cameras.new(f"{name_prefix}_{k + 1}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = fov_rad

        cam_obj = bpy.data.objects.new(f"{name_prefix}_{k + 1}", cam_data)
        cam_obj.location = loc
        cam_obj.rotation_euler = loc.to_track_quat("Z", "Y").to_euler()
        bpy.context.collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    return cameras


def create_cameras_on_two_rings(
    num_cameras: int = 16,
    max_size: float = 1,
    name_prefix: str = "Camera",
    fov: float = 22.5,
    elevation_ratio: float = 0.5,
) -> list[bpy.types.Object]:
    """Create cameras on two rings (upper/lower).

    Create cameras on two rings (upper/lower), then reorder them so that
    consecutive cameras are maximally separated.
    """
    fov_rad = math.radians(fov)
    radius = (max_size * 0.5) / math.tan(fov_rad * 0.5)

    num_cameras_per_ring = num_cameras // 2
    elevation_upper = radius * elevation_ratio
    elevation_lower = -radius * elevation_ratio

    positions = []

    # Lower ring
    for i in range(num_cameras_per_ring):
        theta = (2 * math.pi / num_cameras_per_ring) * i
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        positions.append(mathutils.Vector((x, y, elevation_lower)))

    # Upper ring
    for i in range(num_cameras_per_ring):
        theta = (
            2 * math.pi / num_cameras_per_ring
        ) * i + math.pi / num_cameras_per_ring
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        positions.append(mathutils.Vector((x, y, elevation_upper)))

    # Reorder
    order = farthest_point_ordering(positions)

    cameras = []
    for k, idx in enumerate(order):
        loc = positions[idx]
        cam_data = bpy.data.cameras.new(f"{name_prefix}_{k + 1}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = fov_rad

        cam_obj = bpy.data.objects.new(f"{name_prefix}_{k + 1}", cam_data)
        cam_obj.location = loc
        cam_obj.rotation_euler = loc.to_track_quat("Z", "Y").to_euler()
        bpy.context.collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    return cameras


def create_cameras_on_sphere(
    num_cameras: int = 16,
    max_size: float = 1,
    name_prefix: str = "Camera",
    fov: float = 22.5,
) -> list[bpy.types.Camera]:
    """Create cameras on a Fibonacci sphere.

    Places cameras on a Fibonacci sphere, then reorders them so that
    consecutive cameras are maximally separated (farthest-point ordering).
    """
    phi = math.pi * (3.0 - math.sqrt(5.0))
    fov_rad = math.radians(fov)
    radius = (max_size * 0.5) / math.tan(fov_rad * 0.5)

    positions = []
    for i in range(num_cameras):
        y = 1 - (i / float(num_cameras - 1)) * 2
        radius_at_y = math.sqrt(1 - y * y)
        theta = phi * i
        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        loc = mathutils.Vector((x, y, z)) * radius
        positions.append(loc)

    ordered = farthest_point_ordering(positions)

    cameras = []
    for k, idx in enumerate(ordered):
        loc = positions[idx]
        cam_data = bpy.data.cameras.new(f"{name_prefix}_{k + 1}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = fov_rad

        cam_obj = bpy.data.objects.new(f"{name_prefix}_{k + 1}", cam_data)
        cam_obj.location = loc
        cam_obj.rotation_euler = loc.to_track_quat("Z", "Y").to_euler()

        bpy.context.collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    return cameras


def setup_cycles_setting(context: bpy.types.Context) -> None:
    # Enable Cycles (Eevee does not offer UV output)
    context.scene.render.engine = "CYCLES"

    # Attempt to enable GPU support with preference order: OPTIX, CUDA, OPENCL, CPU
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = "OPTIX"
    except Exception:  # noqa: BLE001
        try:
            preferences.compute_device_type = "CUDA"
        except Exception as e_cuda:
            msg = "An NVIDIA GPU is required for this addon."
            raise SystemError(msg) from e_cuda

    # Set rendering samples and noise threshold
    context.scene.cycles.samples = (
        1  # Reduce to 1 sample for no anti-aliasing in Cycles
    )
    context.scene.cycles.use_denoising = False
    context.scene.cycles.use_light_tree = False
    context.scene.cycles.max_bounces = 1
    context.scene.cycles.diffuse_bounces = 1
    context.scene.cycles.glossy_bounces = 0
    context.scene.cycles.transmission_bounces = 0
    context.scene.cycles.volume_bounces = 0
    context.scene.cycles.transparent_max_bounces = 0


def setup_render_settings(
    context: bpy.types.Context,
    resolution: int,
) -> dict[str, bpy.types.CompositorNodeOutputFile]:
    """Configure render settings.

    Include enabling specific passes and setting up the node tree.

    :param scene:
    :param resolution:
    :return:

    Args:
        context (bpy.types.Context): The scene to configure.
        resolution (tuple, optional): Tuple specifying the render resolution (w, h).
                                        Defaults to (512, 512).

    Raises:
        SystemError: _description_

    Returns:
        dict[str, bpy.types.CompositorNodeOutputFile]:  A dictionary containing
                                                        references to the output
                                                        nodes for each pass.
    """
    setup_cycles_setting(context)

    # Set filter size to minimum (0.01 to disable most filtering)
    context.scene.render.filter_size = 0.01

    # Enable transparent background
    context.scene.render.film_transparent = True

    # Set the render resolution
    context.scene.render.resolution_x = int(resolution)
    context.scene.render.resolution_y = int(resolution)

    # put render resolution scale to 100%
    context.scene.render.resolution_percentage = 100

    # Prevent interpolation for the UV, depth, and normal outputs
    context.scene.render.image_settings.file_format = "OPEN_EXR"
    context.scene.render.image_settings.color_depth = "32"  # Ensure high precision

    return setup_output_nodes(context)


def setup_output_nodes(
    context: bpy.types.Context,
) -> dict[str, bpy.types.CompositorNodeOutputFile]:
    """Create the Node tree.

    Args:
        context (bpy.types.Context): _description_

    Returns:
        dict[str, bpy.types.CompositorNodeOutputFile]: _description_

    """
    scene = context.scene

    # Blender <5 uses scene.use_nodes + scene.node_tree.
    if hasattr(scene, "node_tree"):
        scene.use_nodes = True

    # Blender 5+ uses render.use_compositing instead of scene.use_nodes.
    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True

    node_tree = get_scene_compositor_node_tree(scene)

    # Clear existing nodes
    node_tree.nodes.clear()

    # Enable necessary passes
    view_layer = context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_normal = True
    view_layer.use_pass_uv = True
    view_layer.use_pass_position = True

    # Create render layers node after enabling passes so sockets exist.
    render_layers = node_tree.nodes.new("CompositorNodeRLayers")

    # output path for the render
    render_output_dir = Path(scene.output_path) / "RenderOutput"
    render_output_dir.mkdir(parents=True, exist_ok=True)
    scene.render.filepath = str(render_output_dir / "render_")

    # Create output nodes for each pass
    output_nodes = {}

    for name in ["Depth", "Normal", "UV", "Position"]:
        output_nodes[name.lower()] = set_node_path(name, node_tree, render_layers)
        output_dir = Path(scene.output_path) / f"render_{name.lower()}"
        set_output_node_directory(output_nodes[name.lower()], output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_nodes


def get_scene_compositor_node_tree(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    """Return the compositor node tree across Blender API versions."""
    node_tree = getattr(scene, "node_tree", None)
    if node_tree is not None:
        return node_tree

    if hasattr(scene, "compositing_node_group"):
        node_tree = scene.compositing_node_group
        if node_tree is None:
            node_tree = bpy.data.node_groups.new(
                f"{scene.name}_Compositor",
                "CompositorNodeTree",
            )
            scene.compositing_node_group = node_tree
        return node_tree

    if hasattr(scene, "use_nodes"):
        scene.use_nodes = True
        node_tree = getattr(scene, "node_tree", None)
        if node_tree is not None:
            return node_tree

    msg = "Unable to access scene compositor node tree."
    raise AttributeError(msg)


def set_output_node_directory(
    output_node: bpy.types.CompositorNodeOutputFile,
    output_dir: Path | str,
) -> None:
    """Set output directory across Blender compositor API versions."""
    output_dir_str = str(output_dir)

    if hasattr(output_node, "base_path"):
        output_node.base_path = output_dir_str
        return

    if hasattr(output_node, "directory"):
        output_node.directory = output_dir_str
        return

    msg = "Unable to set output node directory."
    raise AttributeError(msg)


def get_output_node_directory(
    output_node: bpy.types.CompositorNodeOutputFile,
) -> str:
    """Return output directory across Blender compositor API versions."""
    if hasattr(output_node, "base_path"):
        return str(output_node.base_path)

    if hasattr(output_node, "directory"):
        return str(output_node.directory)

    msg = "Unable to read output node directory."
    raise AttributeError(msg)


def get_output_node_file_prefix(
    output_node: bpy.types.CompositorNodeOutputFile,
) -> str:
    """Return the configured output file prefix for this node."""
    if hasattr(output_node, "file_slots") and len(output_node.file_slots) > 0:
        return str(output_node.file_slots[0].path)

    if (
        hasattr(output_node, "file_output_items")
        and len(output_node.file_output_items) > 0
    ):
        return str(output_node.file_output_items[0].name)

    if hasattr(output_node, "file_name") and output_node.file_name:
        return str(output_node.file_name)

    return ""


def find_output_node_image_path(
    output_node: bpy.types.CompositorNodeOutputFile,
    frame_index: int,
) -> Path:
    """Locate the rendered EXR path for a compositor output node."""
    output_dir = Path(get_output_node_directory(output_node))
    prefix = get_output_node_file_prefix(output_node)

    if prefix:
        direct_candidates = [
            output_dir / f"{prefix}{frame_index:04d}.exr",
            output_dir / f"{prefix}.exr",
        ]
        for path in direct_candidates:
            if path.exists():
                return path

        pattern = f"{prefix}*.exr"
    else:
        pattern = "*.exr"

    matches = sorted(output_dir.glob(pattern))
    if matches:
        return matches[0]

    msg = f"No EXR output found in '{output_dir}' for node '{output_node.name}'."
    raise FileNotFoundError(msg)


def map_socket_type_to_file_output_item(socket_type: str) -> str:
    """Map compositor socket type to NodeCompositorFileOutputItem enum."""
    return {
        "VALUE": "FLOAT",
        "INT": "INT",
        "BOOLEAN": "BOOLEAN",
        "VECTOR": "VECTOR",
        "RGBA": "RGBA",
        "ROTATION": "ROTATION",
        "MATRIX": "MATRIX",
        "STRING": "STRING",
        "MENU": "MENU",
        "SHADER": "SHADER",
        "OBJECT": "OBJECT",
        "IMAGE": "IMAGE",
        "GEOMETRY": "GEOMETRY",
        "COLLECTION": "COLLECTION",
        "TEXTURE": "TEXTURE",
        "MATERIAL": "MATERIAL",
        "BUNDLE": "BUNDLE",
        "CLOSURE": "CLOSURE",
    }.get(socket_type, "RGBA")


def get_node_socket_by_name(
    sockets: bpy.types.NodeOutputs | bpy.types.NodeInputs,
    socket_name: str,
) -> bpy.types.NodeSocket:
    """Get a node socket by name with an iteration fallback."""
    socket = sockets.get(socket_name)
    if socket is not None:
        return socket

    for sock in sockets:
        if sock.name == socket_name:
            return sock

    msg = f"Socket '{socket_name}' not found."
    raise KeyError(msg)


def set_node_path(
    name: str,
    node_tree: bpy.types.NodeTree,
    render_layers: bpy.types.CompositorNodeRLayers,
) -> bpy.types.CompositorNodeOutputFile:
    output_node = node_tree.nodes.new("CompositorNodeOutputFile")
    output_node.label = f"{name.lower()}_output"
    output_node.name = f"{name.lower()}_output"

    render_socket = get_node_socket_by_name(render_layers.outputs, name)

    if hasattr(output_node, "file_slots"):
        output_node.base_path = ""
        output_node.file_slots[0].path = f"{name.lower()}_"
        input_socket = output_node.inputs[0]
    elif hasattr(output_node, "file_output_items"):
        output_node.file_output_items.clear()
        socket_type = map_socket_type_to_file_output_item(render_socket.type)
        item = output_node.file_output_items.new(socket_type, f"{name.lower()}_")
        output_node.file_name = ""
        try:
            input_socket = get_node_socket_by_name(output_node.inputs, item.name)
        except KeyError:
            input_socket = output_node.inputs[0]
    else:
        msg = "Unsupported compositor output node API."
        raise AttributeError(msg)

    if hasattr(output_node.format, "media_type"):
        output_node.format.media_type = "IMAGE"

    try:
        output_node.format.file_format = "OPEN_EXR"
    except TypeError:
        output_node.format.file_format = "OPEN_EXR_MULTILAYER"

    output_node.format.color_depth = "32"
    output_node.format.color_mode = "RGBA"

    node_tree.links.new(
        render_socket,
        input_socket,
    )
    return output_node
