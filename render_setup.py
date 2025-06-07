import bpy
import math
import mathutils


def create_cameras_on_one_ring(
    num_cameras=16, max_size=1, name_prefix="Camera", fov=22.5
):
    cameras = []
    fov_rad = math.radians(fov)
    radius = (max_size * 0.5) / math.tan(fov_rad * 0.5)
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
    num_cameras=16, max_size=1, name_prefix="Camera", fov=22.5
):
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
    num_cameras=16, max_size=1, name_prefix="Camera", fov=22.5
):
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


def setup_render_settings(scene, resolution=(512, 512)):
    """
    Configure render settings, including enabling specific passes and setting up the node tree.

    :param scene: The scene to configure.
    :param resolution: Tuple specifying the render resolution (width, height).
    :return: A dictionary containing references to the output nodes for each pass.
    """

    # Enable Cycles (Eevee does not offer UV output)
    scene.render.engine = "CYCLES"

    # Attempt to enable GPU support with preference order: OPTIX, CUDA, OPENCL, CPU
    preferences = bpy.context.preferences.addons["cycles"].preferences
    try:
        preferences.compute_device_type = "OPTIX"
        print("Using OPTIX for rendering.")
    except Exception as e_optix:
        print(f"OPTIX failed: {e_optix}")
        try:
            preferences.compute_device_type = "CUDA"
            print("Using CUDA for rendering.")
        except:
            raise SystemError("You need an NVidia GPU for this Addon!")

    # Set rendering samples and noise threshold
    scene.cycles.samples = 1  # Reduce to 1 sample for no anti-aliasing in Cycles
    scene.cycles.use_denoising = False
    scene.cycles.use_light_tree = False
    scene.cycles.max_bounces = 1
    scene.cycles.diffuse_bounces = 1
    scene.cycles.glossy_bounces = 0
    scene.cycles.transmission_bounces = 0
    scene.cycles.volume_bounces = 0
    scene.cycles.transparent_max_bounces = 0

    # Set filter size to minimum (0.01 to disable most filtering)
    scene.render.filter_size = 0.01

    # Enable transparent background
    scene.render.film_transparent = True

    # Set the render resolution
    scene.render.resolution_x, scene.render.resolution_y = resolution

    # put render resolution scale to 100%
    scene.render.resolution_percentage = 100

    # Prevent interpolation for the UV, depth, and normal outputs
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"  # Ensure high precision

    # Ensure the scene uses nodes
    scene.use_nodes = True

    # Clear existing nodes
    if scene.node_tree:
        scene.node_tree.nodes.clear()

    # Create a new node tree
    tree = scene.node_tree
    links = tree.links

    # Create render layers node
    render_layers = tree.nodes.new("CompositorNodeRLayers")

    # Enable necessary passes
    scene.view_layers["ViewLayer"].use_pass_z = True
    scene.view_layers["ViewLayer"].use_pass_normal = True
    scene.view_layers["ViewLayer"].use_pass_uv = True
    scene.view_layers["ViewLayer"].use_pass_position = True

    # scene.world.light_settings.use_ambient_occlusion = True  # turn AO on
    # scene.world.light_settings.ao_factor = 1.0
    scene.view_layers["ViewLayer"].use_pass_ambient_occlusion = True  # Enable AO pass

    # output path for the render
    scene.render.filepath = scene.output_path + "RenderOutput/render_"

    # Create output nodes for each pass
    output_nodes = {}

    # Depth pass
    depth_output = tree.nodes.new("CompositorNodeOutputFile")
    depth_output.label = "Depth Output"
    depth_output.name = "DepthOutput"
    depth_output.base_path = ""  # Set the base path in the calling function if needed
    depth_output.file_slots[0].path = "depth_"
    links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    output_nodes["depth"] = depth_output

    # Normal pass
    normal_output = tree.nodes.new("CompositorNodeOutputFile")
    normal_output.label = "Normal Output"
    normal_output.name = "NormalOutput"
    normal_output.base_path = ""
    normal_output.file_slots[0].path = "normal_"
    links.new(render_layers.outputs["Normal"], normal_output.inputs[0])
    output_nodes["normal"] = normal_output

    # UV pass
    uv_output = tree.nodes.new("CompositorNodeOutputFile")
    uv_output.label = "UV Output"
    uv_output.name = "UVOutput"
    uv_output.base_path = ""
    uv_output.file_slots[0].path = "uv_"
    links.new(render_layers.outputs["UV"], uv_output.inputs[0])
    output_nodes["uv"] = uv_output

    # Position pass
    position_output = tree.nodes.new("CompositorNodeOutputFile")
    position_output.label = "Position Output"
    position_output.name = "PositionOutput"
    position_output.base_path = ""
    position_output.file_slots[0].path = "position_"
    links.new(render_layers.outputs["Position"], position_output.inputs[0])
    output_nodes["position"] = position_output

    # Ambient Occlusion pass
    img_output = tree.nodes.new("CompositorNodeOutputFile")
    img_output.label = "Image Output"
    img_output.name = "ImageOutput"
    img_output.base_path = ""
    img_output.file_slots[0].path = "img_"
    links.new(render_layers.outputs["Image"], img_output.inputs[0])
    output_nodes["img"] = img_output

    return output_nodes
