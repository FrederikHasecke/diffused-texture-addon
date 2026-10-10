# ruff: noqa: PLR2004, S101

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import bpy
import diffused_texture_addon.blender_operations as blender_operations
import numpy as np
import pytest
from diffused_texture_addon.blender_operations import (
    build_uv_pass_assets,
    export_uv_layout,
    load_img_to_numpy,
    rasterize_uv_layout,
)

# Two separate UV islands: left quad in u 0.1..0.4, right quad in u 0.6..0.9.
_UV_ISLANDS = [
    [(0.1, 0.2), (0.4, 0.2), (0.4, 0.8), (0.1, 0.8)],
    [(0.6, 0.2), (0.9, 0.2), (0.9, 0.8), (0.6, 0.8)],
]


def _make_uv_test_object(
    *,
    with_faces: bool = True,
) -> tuple[bpy.types.Object, bpy.types.Mesh]:
    mesh = bpy.data.meshes.new(f"uv_export_mesh_{uuid4().hex}")
    obj = bpy.data.objects.new(f"uv_export_obj_{uuid4().hex}", mesh)
    bpy.context.scene.collection.objects.link(obj)

    vertices = [
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 1.0, 0.0),
        (2.0, 0.0, 0.0),
        (3.0, 0.0, 0.0),
        (3.0, 1.0, 0.0),
        (2.0, 1.0, 0.0),
    ]
    faces = [(0, 1, 2, 3), (4, 5, 6, 7)] if with_faces else []
    mesh.from_pydata(vertices, [], faces)
    mesh.update()

    uv_layer = mesh.uv_layers.new(name="UVMap")
    for polygon, uv_coords in zip(mesh.polygons, _UV_ISLANDS, strict=False):
        for loop_index, uv_coord in zip(
            polygon.loop_indices,
            uv_coords,
            strict=True,
        ):
            uv_layer.data[loop_index].uv = uv_coord

    return obj, mesh


def _deselect_all(mesh: bpy.types.Mesh) -> None:
    """Mimic an imported model: nothing selected, one face even hidden."""
    for polygon in mesh.polygons:
        polygon.select = False
    for edge in mesh.edges:
        edge.select = False
    for vertex in mesh.vertices:
        vertex.select = False
    if mesh.polygons:
        mesh.polygons[-1].hide = True


def _cleanup_uv_test_object(obj: bpy.types.Object, mesh: bpy.types.Mesh) -> None:
    if obj.name in bpy.data.objects:
        bpy.data.objects.remove(obj, do_unlink=True)
    if mesh.name in bpy.data.meshes:
        bpy.data.meshes.remove(mesh, do_unlink=True)


def _assert_both_islands_covered(mask: np.ndarray) -> None:
    height, width = mask.shape
    row = height // 2
    left_center = int(0.25 * (width - 1))
    right_center = int(0.75 * (width - 1))
    gap_center = int(0.5 * (width - 1))

    assert mask[row, left_center]
    assert mask[row, right_center]
    assert not mask[row, gap_center]


def test_export_uv_layout_writes_png(tmp_path: Path) -> None:
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


def test_export_uv_layout_ignores_face_selection(tmp_path: Path) -> None:
    obj, mesh = _make_uv_test_object()
    _deselect_all(mesh)
    export_path = tmp_path / "uv_layout.png"

    try:
        export_uv_layout(obj, export_path, uv_map_name="UVMap", size=(64, 64))
        uv_layout = load_img_to_numpy(export_path)
    finally:
        _cleanup_uv_test_object(obj, mesh)

    _assert_both_islands_covered(uv_layout[..., 3] > 0)


def test_export_uv_layout_rejects_non_png(tmp_path: Path) -> None:
    obj, mesh = _make_uv_test_object()

    try:
        with pytest.raises(ValueError, match="PNG"):
            export_uv_layout(obj, tmp_path / "uv_layout.svg", uv_map_name="UVMap")
    finally:
        _cleanup_uv_test_object(obj, mesh)


def test_rasterize_uv_layout_ignores_face_selection() -> None:
    obj, mesh = _make_uv_test_object()
    _deselect_all(mesh)

    try:
        uv_layout = rasterize_uv_layout(obj, uv_map_name="UVMap", size=(64, 64))
    finally:
        _cleanup_uv_test_object(obj, mesh)

    assert uv_layout.shape == (64, 64, 4)
    _assert_both_islands_covered(uv_layout[..., 3] > 0)


def _uv_pass_context(tmp_path: Path, texture_resolution: int) -> SimpleNamespace:
    return SimpleNamespace(
        scene=SimpleNamespace(
            texture_resolution=texture_resolution,
            my_uv_map="UVMap",
            output_path=str(tmp_path),
        ),
    )


class _GuiBpy:
    """Proxy for ``bpy`` that looks like an interactive (non ``-b``) session.

    ``bpy.ops.uv.export_layout`` only exports selected faces there, so it must
    never be used to build the surface mask.
    """

    def __init__(self) -> None:
        self.app = SimpleNamespace(background=False)
        self.ops = SimpleNamespace(
            object=bpy.ops.object,
            uv=SimpleNamespace(export_layout=self._forbidden_export_layout),
        )

    @staticmethod
    def _forbidden_export_layout(**_kwargs: object) -> None:
        msg = "bpy.ops.uv.export_layout depends on the edit-mode face selection."
        raise AssertionError(msg)

    def __getattr__(self, name: str) -> object:
        return getattr(bpy, name)


def _fake_bake(
    obj: bpy.types.Object,  # noqa: ARG001
    channel: str,  # noqa: ARG001
    resolution: int,
) -> np.ndarray:
    return np.zeros((resolution, resolution, 4), dtype=np.float32)


@pytest.mark.parametrize("gui_mode", [False, True], ids=["background", "gui"])
def test_build_uv_pass_assets_surface_mask_ignores_face_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gui_mode: bool,  # noqa: FBT001
) -> None:
    monkeypatch.setattr(
        blender_operations,
        "bake_geometry_channel_to_array",
        _fake_bake,
    )
    if gui_mode:
        monkeypatch.setattr(blender_operations, "bpy", _GuiBpy())
    obj, mesh = _make_uv_test_object()
    _deselect_all(mesh)

    try:
        assets = build_uv_pass_assets(_uv_pass_context(tmp_path, 64), obj)
    finally:
        _cleanup_uv_test_object(obj, mesh)

    assert assets.surface_mask.shape == (64, 64)
    _assert_both_islands_covered(assets.surface_mask > 0)
    assert not (tmp_path / "RenderOutput" / "uv_mode_layout.png").exists()


def test_build_uv_pass_assets_raises_on_empty_surface_mask(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        blender_operations,
        "bake_geometry_channel_to_array",
        _fake_bake,
    )
    obj, mesh = _make_uv_test_object(with_faces=False)

    try:
        with pytest.raises(ValueError, match="does not cover any texture pixels"):
            build_uv_pass_assets(_uv_pass_context(tmp_path, 32), obj)
    finally:
        _cleanup_uv_test_object(obj, mesh)
