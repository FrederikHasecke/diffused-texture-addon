# ruff: noqa: T201  (progress and results go to stdout / the Blender log)
"""Generate one texture with the installed add-on, then render it.

Run with ``<scene>_prepared.blend`` open. Drives the add-on through its normal
operator, records every diffusion call when ``capture_dir`` is set, and saves
the textured scene as ``<job>.blend`` for the camera sweep and extra views.
"""

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from common import load_spec, render_still, setup_beauty

spec = load_spec()
renders = Path(spec["renders_dir"])
gen_dir = renders / f"{spec['name']}_gen"
gen_dir.mkdir(parents=True, exist_ok=True)
module = spec["addon_module"]
bpy.ops.preferences.addon_enable(module=module)

obj = next(o for o in bpy.data.objects if o.type == "MESH")
for o in list(bpy.data.objects):
    if o is not obj:
        bpy.data.objects.remove(o)
for slot in obj.material_slots:
    slot.material = None
bpy.context.view_layer.objects.active = obj
obj.select_set(True)  # noqa: FBT003

sc: Any = bpy.context.scene
sc.my_mesh_object = obj.name
sc.my_uv_map = obj.data.uv_layers.active.name
sc.my_prompt = spec["prompt"]
sc.my_negative_prompt = spec.get("negative", "")
sc.sd_version = spec["sd_version"]
sc.operation_mode = spec["mode"]
sc.num_cameras = str(spec["cameras"])
sc.mesh_complexity = spec["complexity"]
sc.texture_resolution = str(spec["texture_res"])
sc.render_resolution = str(spec["render_res"])
sc.num_inference_steps = spec["steps"]
sc.guidance_scale = spec["guidance"]
sc.denoise_strength = spec["denoise"]
sc.texture_seed = spec["seed"]
sc.custom_sd_resolution = spec["custom_sd_res"]
sc.output_path = str(gen_dir) + "/"
if "union_strength" in spec:
    sc.union_controlnet_strength = spec["union_strength"]
sc.input_texture = None
if spec.get("input_texture"):
    sc.input_texture = bpy.data.images.load(spec["input_texture"], check_existing=True)
if spec.get("subgrid"):
    sc.subgrid_rows, sc.subgrid_cols = spec["subgrid"]

if spec.get("capture_dir"):
    # Record the exact images passed to and returned from every diffusion call.
    name = (
        "img_parasequential"
        if spec["mode"] == "PARA_SEQUENTIAL_IMG"
        else "img_parallel"
    )
    workflow = importlib.import_module(f"{module}.diffusedtexture.{name}")
    original_run = workflow.run_pipeline
    calls = []

    def capture(*args: object, **kwargs: Any) -> object:  # noqa: ANN401
        folder = Path(spec["capture_dir"]) / f"{len(calls):02}"
        folder.mkdir(parents=True, exist_ok=True)
        for kind, key in (
            ("input", "input_img"),
            ("mask", "uv_mask"),
            ("depth", "depth_img"),
            ("normal", "normal_img"),
            ("canny", "canny_img"),
        ):
            kwargs[key].save(folder / f"{kind}.png")
        call = {
            "index": len(calls),
            "input_size": list(kwargs["input_img"].size),
            "denoise": kwargs["strength"],
            "steps": kwargs["num_inference_steps"],
            "guidance": kwargs["guidance_scale"],
        }
        calls.append(call)
        (folder / "call.json").write_text(json.dumps(call, indent=2))
        result = original_run(*args, **kwargs)
        result.save(folder / "output.png")
        return result

    workflow.run_pipeline = capture

start = time.time()
result = bpy.ops.object.generate_texture()
seconds = time.time() - start
error = str(getattr(sc, "diffused_texture_operator_error", ""))
print("GEN_RESULT", spec["name"], result, f"seconds={seconds:.1f}", f"error={error!r}")
if error or "FINISHED" not in result:
    sys.exit(1)
texture = sorted(gen_dir.glob("output_texture_*.png"))[-1]
try:
    import torch

    gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU"
except ImportError:
    gpu = "unknown"
(gen_dir / "result.json").write_text(
    json.dumps(
        {"texture": texture.name, "seconds": round(seconds, 1), "gpu": gpu},
        indent=2,
    ),
)
print("GEN_TEXTURE", texture, flush=True)

setup_beauty(obj, spec)
render_still(renders / f"{spec['name']}.png")
bpy.ops.wm.save_as_mainfile(filepath=str(renders / f"{spec['name']}.blend"))
