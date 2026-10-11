# ruff: noqa: T201  (progress and results go to stdout / the Blender log)
"""Prepare one Poly Haven asset: one joined mesh, plus original and clay renders.

Run with the downloaded asset blend open. Writes ``<scene>_prepared.blend``
for generation, and ``<scene>_original.png`` / ``<scene>_clay.png`` with the
same camera and lighting as the generated renders.
"""

import sys
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).parent))
from common import load_spec, render_still, setup_beauty

spec = load_spec()
renders = Path(spec["renders_dir"])
renders.mkdir(parents=True, exist_ok=True)

meshes = [o for o in bpy.data.objects if o.type == "MESH"]
if spec.get("mesh") == "largest":
    # Poly Haven LOD sets: keep only the highest-detail LOD0 mesh.
    meshes = [max(meshes, key=lambda o: len(o.data.polygons))]
for o in list(bpy.data.objects):
    if o not in meshes:
        bpy.data.objects.remove(o, do_unlink=True)
bpy.ops.object.select_all(action="DESELECT")
for o in meshes:
    o.hide_set(False)  # noqa: FBT003
    o.hide_render = False
    o.select_set(True)  # noqa: FBT003
bpy.context.view_layer.objects.active = meshes[0]
bpy.ops.object.convert(target="MESH")
if len(meshes) > 1:
    bpy.ops.object.join()
obj = bpy.context.active_object
obj.name = spec["name"]
obj.data.calc_loop_triangles()
print(
    "MESH",
    obj.name,
    len(obj.data.vertices),
    len(obj.data.loop_triangles),
    flush=True,
)
bpy.ops.wm.save_as_mainfile(filepath=spec["prepared_blend"])

# Identical diffuse-only shading for the source and generated comparisons:
# original albedo only, no normal, roughness or metal maps.
material = bpy.data.materials.new("Original albedo comparison")
material.use_nodes = True
tree = material.node_tree
bsdf = tree.nodes["Principled BSDF"]
image_node = tree.nodes.new("ShaderNodeTexImage")
image_node.image = bpy.data.images.load(spec["albedo"], check_existing=True)
tree.links.new(image_node.outputs["Color"], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.5
for slot in obj.material_slots:
    slot.material = material

setup_beauty(obj, spec)
render_still(renders / f"{spec['name']}_original.png")
bpy.ops.wm.save_as_mainfile(filepath=str(renders / f"{spec['name']}_original.blend"))

tree.links.remove(bsdf.inputs["Base Color"].links[0])
bsdf.inputs["Base Color"].default_value = (0.45, 0.45, 0.45, 1)
render_still(renders / f"{spec['name']}_clay.png")
