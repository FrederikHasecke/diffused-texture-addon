import math
from pathlib import Path

import bpy
import mathutils


def create_cameras_on_one_ring(
    num_cameras: int,
    max_size: float,
    name_prefix: str = "Camera",
    fov: float = 22.5,
) -> list[bpy.types.Camera]:
    cameras = []
    fov_rad = math.radians(fov)
    radius = 1.1 * (max_size * math.sqrt(2)) / math.tan(fov_rad)
    angle_offset = math.pi / num_cameras
    elevation = radius * 0.25

    for i in range(num_cameras):
        theta = (2 * math.pi / num_cameras) * i + angle_offset
        x = radius * math.cos(theta)
        y = radius * math.sin(theta)
        location = mathutils.Vector((x, y, elevation))

        cam_data = bpy.data.cameras.new(f"{name_prefix}_{i + 1}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = fov_rad

        cam_obj = bpy.data.objects.new(f"{name_prefix}_{i + 1}", cam_data)
        cam_obj.location = location

        direction = location - mathutils.Vector((0, 0, 0))
        cam_obj.rotation_euler = direction.to_track_quat("Z", "Y").to_euler()

        bpy.context.collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    return cameras


def create_cameras_on_two_rings(
    num_cameras: int = 16,
    max_size: float = 1,
    name_prefix: str = "Camera",
    fov: float = 22.5,
) -> list[bpy.types.Camera]:
    cameras = []
    fov_rad = math.radians(fov)
    radius = (max_size * 0.5) / math.tan(fov_rad * 0.5)

    num_cameras_per_ring = num_cameras // 2
    angle_offset = math.pi / num_cameras_per_ring
    elevation_upper = radius * 0.5
    elevation_lower = -radius * 0.5

    for i in range(num_cameras_per_ring):
        theta = (2 * math.pi / num_cameras_per_ring) * i
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        loc = mathutils.Vector((x, y, elevation_lower))

        cam_data = bpy.data.cameras.new(f"{name_prefix}_LowerRing_{i + 1}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = fov_rad

        cam_obj = bpy.data.objects.new(f"{name_prefix}_LowerRing_{i + 1}", cam_data)
        cam_obj.location = loc
        cam_obj.rotation_euler = loc.to_track_quat("Z", "Y").to_euler()

        bpy.context.collection.objects.link(cam_obj)
        cameras.append(cam_obj)

    for i in range(num_cameras_per_ring):
        theta = (2 * math.pi / num_cameras_per_ring) * i + angle_offset
        x, y = radius * math.cos(theta), radius * math.sin(theta)
        loc = mathutils.Vector((x, y, elevation_upper))

        cam_data = bpy.data.cameras.new(f"{name_prefix}_UpperRing_{i + 1}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = fov_rad

        cam_obj = bpy.data.objects.new(f"{name_prefix}_UpperRing_{i + 1}", cam_data)
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
    cameras = []
    phi = math.pi * (3.0 - math.sqrt(5.0))
    fov_rad = math.radians(fov)
    radius = (max_size * 0.5) / math.tan(fov_rad * 0.5)

    for i in range(num_cameras):
        y = 1 - (i / float(num_cameras - 1)) * 2
        radius_at_y = math.sqrt(1 - y * y)
        theta = phi * i

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        loc = mathutils.Vector((x, y, z)) * radius

        cam_data = bpy.data.cameras.new(f"{name_prefix}_{i + 1}")
        cam_data.lens_unit = "FOV"
        cam_data.angle = fov_rad

        cam_obj = bpy.data.objects.new(f"{name_prefix}_{i + 1}", cam_data)
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
    # Ensure the scene uses nodes
    context.scene.use_nodes = True

    # Clear existing nodes
    if context.scene.node_tree:
        context.scene.node_tree.nodes.clear()

    # Create render layers node
    render_layers = context.scene.node_tree.nodes.new("CompositorNodeRLayers")

    # Enable necessary passes
    context.scene.view_layers["ViewLayer"].use_pass_z = True
    context.scene.view_layers["ViewLayer"].use_pass_normal = True
    context.scene.view_layers["ViewLayer"].use_pass_uv = True
    context.scene.view_layers["ViewLayer"].use_pass_position = True

    # output path for the render
    context.scene.render.filepath = context.scene.output_path + "RenderOutput/render_"

    # Create output nodes for each pass
    output_nodes = {}

    for name in ["Depth", "Normal", "UV", "Position"]:
        output_nodes[name.lower()] = set_node_path(name, context, render_layers)
        output_nodes[name.lower()].base_path = str(
            Path(context.scene.output_path) / f"render_{name.lower()}",
        )
        Path(output_nodes[name.lower()].base_path).mkdir(parents=True, exist_ok=True)

    return output_nodes


def set_node_path(
    name: str,
    context: bpy.types.Context,
    render_layers: bpy.types.CompositorNodeRLayers,
) -> bpy.types.CompositorNodeOutputFile:
    output_node = context.scene.node_tree.nodes.new("CompositorNodeOutputFile")
    output_node.label = f"{name.lower()}_output"
    output_node.name = f"{name.lower()}_output"
    output_node.base_path = ""  # Set the base path in the calling function if needed
    output_node.file_slots[0].path = f"{name.lower()}_"

    output_node.format.file_format = "OPEN_EXR"
    output_node.format.color_depth = "32"
    output_node.format.color_mode = "RGBA"

    context.scene.node_tree.links.new(
        render_layers.outputs[name],
        output_node.inputs[0],
    )
    return output_node
