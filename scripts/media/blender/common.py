# ruff: noqa: T201  (progress and results go to stdout / the Blender log)
"""Shared Blender helpers for the README media scripts.

Every script here runs inside Blender (``blender -b ... --python <script> --
<spec.json>``). ``media.py`` writes the spec with resolved absolute paths, so
these scripts never read ``jobs.json`` or guess workspace locations.
"""

import json
import math
import sys
from pathlib import Path
from typing import Any

import bpy
from mathutils import Vector


def load_spec() -> dict[str, Any]:
    """Return the JSON spec passed after ``--`` on the Blender command line."""
    return json.loads(Path(sys.argv[sys.argv.index("--") + 1]).read_text())


def use_optix(scene: bpy.types.Scene) -> None:
    """Render on every OptiX device, matching the recorded README renders."""
    scene.render.engine = "CYCLES"
    scene.cycles.device = "GPU"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for device in prefs.devices:
        device.use = device.type == "OPTIX"


def bounds(obj: bpy.types.Object) -> tuple[Vector, float, float]:
    """Return the world-space bounding-box center, radius and lowest Z."""
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector()) / 8
    radius = max((c - center).length for c in corners)
    return center, radius, min(c.z for c in corners)


def orbit(
    camera: bpy.types.Object,
    center: Vector,
    distance: float,
    azimuth: float,
    elevation: float,
) -> None:
    """Place a camera on a sphere around ``center``, looking at it."""
    az, el = math.radians(azimuth), math.radians(elevation)
    offset = Vector(
        (math.sin(az) * math.cos(el), -math.cos(az) * math.cos(el), math.sin(el)),
    )
    camera.location = center + offset * distance
    camera.rotation_euler = (
        (center - camera.location).to_track_quat("-Z", "Y").to_euler()
    )


def setup_beauty(obj: bpy.types.Object, spec: dict[str, Any]) -> None:
    """Add HDRI lighting, a shadow catcher and the comparison camera.

    The same setup is used for the original-albedo, clay and generated renders,
    so their framing and lighting match exactly.
    """
    scene = bpy.context.scene
    render = spec["render"]
    world = bpy.data.worlds.new("hero_world")
    scene.world = world
    world.use_nodes = True
    nodes = world.node_tree.nodes
    env = nodes.new("ShaderNodeTexEnvironment")
    env.image = bpy.data.images.load(spec["hdri"])
    world.node_tree.links.new(
        env.outputs["Color"],
        nodes["Background"].inputs["Color"],
    )
    nodes["Background"].inputs["Strength"].default_value = 1.0

    center, radius, zmin = bounds(obj)
    bpy.ops.mesh.primitive_plane_add(
        size=radius * 20,
        location=(center.x, center.y, zmin),
    )
    bpy.context.active_object.is_shadow_catcher = True

    camera_data = bpy.data.cameras.new("hero_cam")
    camera_data.lens = render["lens"]
    camera = bpy.data.objects.new("hero_cam", camera_data)
    scene.collection.objects.link(camera)
    fov = 2 * math.atan(18 / camera_data.lens)
    distance = radius / math.sin(fov / 2) * spec.get("zoom", 1.0)
    orbit(camera, center, distance, spec["azimuth"], spec["elevation"])
    scene.camera = camera

    scene.compositing_node_group = None
    use_optix(scene)
    scene.cycles.samples = render["samples"]
    scene.cycles.use_denoising = True
    scene.render.film_transparent = True
    scene.render.resolution_x = render["size"]
    scene.render.resolution_y = render["size"]
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.view_settings.view_transform = render["view_transform"]


def render_still(path: Path) -> None:
    """Render the active camera to ``path``."""
    bpy.context.scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)
    print("RENDERED", path, flush=True)
