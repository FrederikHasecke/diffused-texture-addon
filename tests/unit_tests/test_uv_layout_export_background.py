# ruff: noqa: PLR2004, S101

from pathlib import Path
from uuid import uuid4

import bpy
import numpy as np
from diffused_texture_addon.blender_operations import (
    export_uv_layout,
    load_img_to_numpy,
)


def _make_uv_test_object() -> tuple[bpy.types.Object, bpy.types.Mesh]:
    mesh = bpy.data.meshes.new(f"uv_export_mesh_{uuid4().hex}")
    obj = bpy.data.objects.new(f"uv_export_obj_{uuid4().hex}", mesh)
    bpy.context.scene.collection.objects.link(obj)

    mesh.from_pydata(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)],
        [],
        [(0, 1, 2, 3)],
    )
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_coords = [
        (0.2, 0.2),
        (0.8, 0.2),
        (0.8, 0.8),
        (0.2, 0.8),
    ]
    for loop_index, uv_coord in zip(
        mesh.polygons[0].loop_indices,
        uv_coords,
        strict=True,
    ):
        uv_layer.data[loop_index].uv = uv_coord

    return obj, mesh


def _cleanup_uv_test_object(obj: bpy.types.Object, mesh: bpy.types.Mesh) -> None:
    if obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.name in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)


def test_export_uv_layout_writes_png_in_background_mode(tmp_path: Path) -> None:
    obj, mesh = _make_uv_test_object()
    export_path = tmp_path / "uv_layout.png"

    try:
        export_uv_layout(obj, export_path, uv_map_name="UVMap", size=(64, 64))
        uv_layout = load_img_to_numpy(export_path)
    finally:
        _cleanup_uv_test_object(obj, mesh)

    assert export_path.exists()
    assert uv_layout.shape == (64, 64, 4)
    assert uv_layout.dtype == np.float32

    opaque_pixels = uv_layout[..., 3] > 0.9
    assert np.any(opaque_pixels)
    assert np.any(uv_layout[..., 0][opaque_pixels] < 0.1)
    assert np.any(uv_layout[..., 0][opaque_pixels] > 0.9)
