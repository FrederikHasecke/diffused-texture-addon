"""Re-render a saved ``<job>.blend`` from other angles, or as a camera sweep.

``views`` entries render one still each (e.g. the reverse axe view). ``sweep``
renders the frames for ``showcase.mp4``: the camera swings around the saved
hero azimuth at the same distance.
"""

import math
import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from common import bounds, load_spec, orbit, render_still, use_optix

spec = load_spec()
scene = bpy.context.scene
camera = scene.camera
obj = bpy.data.objects[spec["object"]]
center, _, _ = bounds(obj)
distance = (camera.location - center).length
use_optix(scene)
renders = Path(spec["renders_dir"])

for label, view in spec.get("views", {}).items():
    orbit(camera, center, distance, view["azimuth"], view["elevation"])
    render_still(renders / f"{spec['name']}_{label}.png")
    bpy.ops.wm.save_as_mainfile(filepath=str(renders / f"{spec['name']}_{label}.blend"))

if spec.get("sweep"):
    sweep = spec["sweep"]
    scene.render.resolution_x = sweep["size"]
    scene.render.resolution_y = sweep["size"]
    scene.cycles.samples = sweep["samples"]
    scene.cycles.use_denoising = True
    scene.render.use_persistent_data = True
    scene.compositing_node_group = None
    folder = Path(spec["frames_dir"])
    folder.mkdir(parents=True, exist_ok=True)
    frames = sweep["frames"]
    for frame in range(frames):
        swing = sweep["swing_degrees"] * math.sin(2 * math.pi * frame / frames)
        orbit(camera, center, distance, spec["azimuth"] + swing, spec["elevation"])
        render_still(folder / f"{frame:04}.png")
