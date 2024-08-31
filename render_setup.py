import bpy
import math
import mathutils


def create_cameras_on_sphere(
    num_cameras=16, max_size=1, name_prefix="Camera", offset=False, fov=22.5
):
    """
    Create cameras evenly distributed on a sphere around the world origin,
    with each camera positioned such that it perfectly frames an object of size
    max_size using a field of view of fov.

    :param num_cameras: Number of cameras to create.
    :param max_size: The maximum size of the object to be captured by the cameras.
    :param name_prefix: Prefix for naming the cameras.
    :param offset: If True, swap the coordinates (x -> y, y -> z, z -> x).
    :param fov: Field of view for the cameras in degrees.
    :return: List of created camera objects.
    """

    cameras = []
    phi = math.pi * (3.0 - math.sqrt(5.0))  # Golden angle in radians

    # Convert FOV from degrees to radians
    fov_rad = math.radians(fov)

    # Calculate the distance from the origin such that the FOV covers max_size
    radius = (max_size * 0.5) / math.tan(fov_rad * 0.5)

    # Add a 2x margin in case the object is very weirdly shaped
    radius *= 2

    for i in range(num_cameras):
        y = 1 - (i / float(num_cameras - 1)) * 2  # y goes from 1 to -1
        radius_at_y = math.sqrt(1 - y * y)  # Radius at y
        theta = phi * i  # Golden angle increment

        x = math.cos(theta) * radius_at_y
        z = math.sin(theta) * radius_at_y
        location = mathutils.Vector((x, y, z)) * radius

        if offset:
            # Swap coordinates: x -> y, y -> z, z -> x
            location = mathutils.Vector((location.y, location.z, location.x))

        # Create camera
        bpy.ops.object.camera_add(location=location)
        camera = bpy.context.object
        camera.name = f"{name_prefix}_{i+1}"

        # Set the camera's FOV
        camera.data.lens_unit = "FOV"
        camera.data.angle = fov_rad

        # Point the camera at the origin
        direction = camera.location - mathutils.Vector((0, 0, 0))
        rot_quat = direction.to_track_quat("Z", "Y")
        camera.rotation_euler = rot_quat.to_euler()

        cameras.append(camera)

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
    scene.cycles.samples = 1024
    scene.cycles.adaptive_threshold = 0.05
    scene.cycles.use_denoising = True

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

    # Set the render resolution
    scene.render.resolution_x, scene.render.resolution_y = resolution

    # Prevent interpolation for the UV, depth, and normal outputs
    scene.render.image_settings.file_format = "OPEN_EXR"
    scene.render.image_settings.color_depth = "32"  # Ensure high precision

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

    # Position pass (using Position pass)
    position_output = tree.nodes.new("CompositorNodeOutputFile")
    position_output.label = "Position Output"
    position_output.name = "PositionOutput"
    position_output.base_path = ""
    position_output.file_slots[0].path = "position_"
    links.new(render_layers.outputs["Position"], position_output.inputs[0])
    output_nodes["position"] = position_output

    return output_nodes
