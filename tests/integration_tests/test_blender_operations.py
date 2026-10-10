# ruff: noqa: PLR2004, S101

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import numpy as np
from PIL import Image

bpy = pytest.importorskip("bpy")
from diffused_texture_addon import blender_operations, render_setup


pytestmark = pytest.mark.exclusive
MANAGED_MATERIAL_TAG = "diffused_texture_managed_material"
MANAGED_MATERIAL_OWNER = "diffused_texture_managed_material_owner"
MANAGED_NODE_ROLE = "diffused_texture_managed_node_role"
TEXTURE_NODE_ROLE = "texture"
BSDF_NODE_ROLE = "bsdf"
OUTPUT_NODE_ROLE = "output"
EXPECTED_MANAGED_NODE_TYPES = [
    "ShaderNodeBsdfPrincipled",
    "ShaderNodeOutputMaterial",
    "ShaderNodeTexImage",
]
EXPECTED_MANAGED_NODE_COUNT = 3
EXPECTED_MANAGED_LINK_COUNT = 2


@pytest.fixture(autouse=True)
def _reset_blender_state() -> Iterator[None]:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    yield
    bpy.ops.wm.read_factory_settings(use_empty=True)


def _write_texture(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (4, 4), color).save(path)
    return path


def _create_mesh_object() -> bpy.types.Object:
    bpy.ops.mesh.primitive_cube_add(size=2, location=(0, 0, 0))
    bpy.ops.object.mode_set(mode="OBJECT")
    obj = bpy.context.active_object
    if obj is None:
        msg = "Expected an active mesh object"
        raise AssertionError(msg)
    return obj


def test_prepare_scene_ignores_objects_outside_active_view_layer() -> None:
    target = _create_mesh_object()
    target.name = "TextureTarget"
    bpy.ops.mesh.primitive_cube_add(size=1, location=(3, 0, 0))
    other_in_view_layer = bpy.context.active_object
    assert other_in_view_layer is not None
    other_in_view_layer.name = "OtherInViewLayer"

    other_scene = bpy.data.scenes.new("OtherScene")
    outside_mesh = bpy.data.meshes.new("OutsideMesh")
    outside_view_layer = bpy.data.objects.new("Cube", outside_mesh)
    other_scene.collection.objects.link(outside_view_layer)

    assert outside_view_layer.name == "Cube"
    assert outside_view_layer.name in bpy.data.objects
    assert outside_view_layer.name not in bpy.context.view_layer.objects

    backup = blender_operations.prepare_scene(target)
    try:
        assert outside_view_layer.name not in backup["original_object_visibility"]
        assert outside_view_layer.hide_render is False
        assert other_in_view_layer.hide_get() is True
        assert other_in_view_layer.hide_render is True
    finally:
        blender_operations.restore_scene(backup, cameras=[])

    assert other_in_view_layer.hide_get() is False
    assert other_in_view_layer.hide_render is False


def _simulate_render_compositor_setup(scene: bpy.types.Scene) -> bpy.types.NodeTree:
    node_tree = render_setup.get_scene_compositor_node_tree(scene)
    render_layers = node_tree.nodes.new("CompositorNodeRLayers")
    render_layers.name = "DiffusedTexture_RenderLayers"
    for name in ("depth_output", "normal_output", "uv_output"):
        output_node = node_tree.nodes.new("CompositorNodeOutputFile")
        output_node.name = name
    return node_tree


def test_restore_scene_removes_compositor_group_created_during_generation() -> None:
    scene = bpy.context.scene
    if not hasattr(scene, "compositing_node_group"):
        pytest.skip("Requires the Blender 5+ compositing_node_group API")
    target = _create_mesh_object()
    scene.compositing_node_group = None
    groups_before = set(bpy.data.node_groups.keys())

    backup = blender_operations.prepare_scene(target)
    try:
        created_tree = _simulate_render_compositor_setup(scene)
        created_name = created_tree.name
        assert scene.compositing_node_group is created_tree
    finally:
        blender_operations.restore_scene(backup, cameras=[])

    assert scene.compositing_node_group is None
    assert created_name not in bpy.data.node_groups
    assert set(bpy.data.node_groups.keys()) == groups_before


def test_restore_scene_keeps_existing_compositor_group_and_user_nodes() -> None:
    scene = bpy.context.scene
    if not hasattr(scene, "compositing_node_group"):
        pytest.skip("Requires the Blender 5+ compositing_node_group API")
    target = _create_mesh_object()
    user_tree = bpy.data.node_groups.new("UserCompositor", "CompositorNodeTree")
    user_node = user_tree.nodes.new("CompositorNodeRLayers")
    user_node.name = "UserRenderLayers"
    scene.compositing_node_group = user_tree
    groups_before = set(bpy.data.node_groups.keys())

    backup = blender_operations.prepare_scene(target)
    try:
        assert _simulate_render_compositor_setup(scene) is user_tree
    finally:
        blender_operations.restore_scene(backup, cameras=[])

    assert scene.compositing_node_group is user_tree
    assert [node.name for node in user_tree.nodes] == ["UserRenderLayers"]
    assert set(bpy.data.node_groups.keys()) == groups_before


def _create_two_quad_uv_seam_object() -> bpy.types.Object:
    mesh = bpy.data.meshes.new("TwoQuadSeamMesh")
    mesh.from_pydata(
        [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (2.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 1.0, 0.0),
            (2.0, 1.0, 0.0),
        ],
        [],
        [
            (0, 1, 4, 3),
            (1, 2, 5, 4),
        ],
    )
    mesh.update()
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_values = [
        (0.10, 0.10),
        (0.40, 0.10),
        (0.40, 0.90),
        (0.10, 0.90),
        (0.60, 0.10),
        (0.90, 0.10),
        (0.90, 0.90),
        (0.60, 0.90),
    ]
    for loop_index, uv in enumerate(uv_values):
        uv_layer.data[loop_index].uv = uv
    obj = bpy.data.objects.new("TwoQuadSeam", mesh)
    bpy.context.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    return obj


def _managed_nodes(material: bpy.types.Material) -> dict[str, Any]:
    return {
        node[MANAGED_NODE_ROLE]: node
        for node in material.node_tree.nodes
        if MANAGED_NODE_ROLE in node
    }


def _assert_managed_material_layout(
    material: bpy.types.Material,
    texture_path: Path,
) -> None:
    managed_nodes = _managed_nodes(material)
    texture_node = managed_nodes[TEXTURE_NODE_ROLE]
    principled_node = managed_nodes[BSDF_NODE_ROLE]
    output_node = managed_nodes[OUTPUT_NODE_ROLE]

    assert bool(material.get(MANAGED_MATERIAL_TAG))
    assert len(material.node_tree.nodes) == EXPECTED_MANAGED_NODE_COUNT
    assert sorted(node.bl_idname for node in material.node_tree.nodes) == (
        EXPECTED_MANAGED_NODE_TYPES
    )
    assert texture_node.image is not None
    assert Path(texture_node.image.filepath_from_user()) == texture_path
    assert len(material.node_tree.links) == EXPECTED_MANAGED_LINK_COUNT
    assert principled_node.inputs["Base Color"].links[0].from_node == texture_node
    assert output_node.inputs["Surface"].links[0].from_node == principled_node


def _material_signature(
    material: bpy.types.Material,
) -> tuple[bool, list[str], list[tuple[str, str]]]:
    if not material.use_nodes or material.node_tree is None:
        return material.use_nodes, [], []

    node_types = sorted(node.bl_idname for node in material.node_tree.nodes)
    links = sorted(
        (
            link.from_node.bl_idname,
            link.to_node.bl_idname,
        )
        for link in material.node_tree.links
    )
    return material.use_nodes, node_types, links


def test_build_uv_seam_topology_assets_extracts_known_mesh_seam() -> None:
    obj = _create_two_quad_uv_seam_object()
    surface_mask = np.full((32, 32), 255, dtype=np.uint8)

    assets = blender_operations._build_uv_seam_topology_assets(
        bpy.context,
        obj,
        surface_mask,
        32,
    )

    assert np.any(assets.seam_line_mask)
    assert len(assets.seam_link_source_yx) > 0
    source_x = assets.seam_link_source_yx[:, 1]
    target_x = assets.seam_link_target_yx[:, 1]
    assert np.any(source_x < 16)
    assert np.any(target_x > 16)


def test_apply_texture_to_object_creates_managed_material(
    tmp_path: Path,
) -> None:
    obj = _create_mesh_object()
    texture_path = _write_texture(tmp_path / "texture.png", (255, 0, 0))

    blender_operations.apply_texture_to_object(obj, texture_path)

    assert len(obj.data.materials) == 1
    material = obj.data.materials[0]
    assert material is not None
    assert material.get(blender_operations._MANAGED_MATERIAL_OWNER) == obj.name
    _assert_managed_material_layout(material, texture_path)


def test_apply_texture_to_object_reuses_managed_material_and_node(
    tmp_path: Path,
) -> None:
    obj = _create_mesh_object()
    first_path = _write_texture(tmp_path / "first.png", (255, 0, 0))
    second_path = _write_texture(tmp_path / "second.png", (0, 255, 0))

    blender_operations.apply_texture_to_object(obj, first_path)
    original_material = obj.data.materials[0]
    assert original_material is not None
    original_pointer = original_material.as_pointer()

    blender_operations.apply_texture_to_object(obj, second_path)

    updated_material = obj.data.materials[0]
    assert updated_material is not None
    assert updated_material.as_pointer() == original_pointer
    _assert_managed_material_layout(updated_material, second_path)


def test_apply_texture_to_object_does_not_mutate_existing_material_datablock(
    tmp_path: Path,
) -> None:
    obj = _create_mesh_object()
    existing_material = bpy.data.materials.new(name="ExistingMaterial")
    existing_material.use_nodes = True
    existing_material.node_tree.nodes.new(type="ShaderNodeValue")
    existing_signature = _material_signature(existing_material)
    obj.data.materials.append(existing_material)
    texture_path = _write_texture(tmp_path / "texture.png", (0, 0, 255))

    blender_operations.apply_texture_to_object(obj, texture_path)

    assert _material_signature(existing_material) == existing_signature
    assert bpy.data.materials.get(existing_material.name) is not None
    managed_material = obj.data.materials[0]
    assert managed_material is not None
    assert managed_material.as_pointer() != existing_material.as_pointer()
    _assert_managed_material_layout(managed_material, texture_path)


def test_apply_texture_to_object_replaces_each_material_slot_but_preserves_indices(
    tmp_path: Path,
) -> None:
    obj = _create_mesh_object()
    first_material = bpy.data.materials.new(name="SlotOne")
    second_material = bpy.data.materials.new(name="SlotTwo")
    obj.data.materials.append(first_material)
    obj.data.materials.append(second_material)

    for index, polygon in enumerate(obj.data.polygons):
        polygon.material_index = index % 2

    original_indices = [polygon.material_index for polygon in obj.data.polygons]
    texture_path = _write_texture(tmp_path / "texture.png", (255, 255, 0))

    blender_operations.apply_texture_to_object(obj, texture_path)

    assert len(obj.data.materials) == 2
    managed_material = obj.data.materials[0]
    assert managed_material is not None
    assert all(
        slot is not None and slot.as_pointer() == managed_material.as_pointer()
        for slot in obj.data.materials
    )
    assert [polygon.material_index for polygon in obj.data.polygons] == original_indices
    _assert_managed_material_layout(managed_material, texture_path)


def test_apply_texture_to_object_recovers_when_managed_material_needs_repair(
    tmp_path: Path,
) -> None:
    obj = _create_mesh_object()
    first_path = _write_texture(tmp_path / "first.png", (255, 0, 255))
    second_path = _write_texture(tmp_path / "second.png", (0, 255, 255))

    blender_operations.apply_texture_to_object(obj, first_path)
    managed_material = obj.data.materials[0]
    assert managed_material is not None

    managed_material.use_nodes = False
    blender_operations.apply_texture_to_object(obj, second_path)
    _assert_managed_material_layout(managed_material, second_path)

    output_nodes = [
        node
        for node in managed_material.node_tree.nodes
        if node.get(MANAGED_NODE_ROLE) == OUTPUT_NODE_ROLE
    ]
    managed_material.node_tree.nodes.remove(output_nodes[0])

    blender_operations.apply_texture_to_object(obj, first_path)
    _assert_managed_material_layout(managed_material, first_path)


class _BakeSpyBpy:
    """Proxy for ``bpy`` that records the selection when ``object.bake`` runs."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.selected_during_bake: list[str] | None = None
        self.active_during_bake: str | None = None
        self.ops = SimpleNamespace(object=SimpleNamespace(bake=self._bake))

    def __getattr__(self, name: str) -> Any:
        return getattr(bpy, name)

    def _bake(self, **kwargs: Any) -> set[str]:
        view_layer = bpy.context.view_layer
        self.selected_during_bake = sorted(
            o.name for o in view_layer.objects if o.select_get()
        )
        active = view_layer.objects.active
        self.active_during_bake = active.name if active is not None else None
        if self.fail:
            msg = "bake failed"
            raise RuntimeError(msg)
        return bpy.ops.object.bake(**kwargs)


def _selection_state() -> tuple[list[str], str | None]:
    view_layer = bpy.context.view_layer
    active = view_layer.objects.active
    return (
        sorted(o.name for o in view_layer.objects if o.select_get()),
        active.name if active is not None else None,
    )


def _create_bake_scene() -> tuple[bpy.types.Object, bpy.types.Object]:
    target = _create_mesh_object()
    target.name = "BakeTarget"
    bpy.ops.mesh.primitive_plane_add(size=1, location=(3, 0, 0))
    other = bpy.context.active_object
    assert other is not None
    other.name = "OtherMesh"
    bpy.context.scene.cycles.device = "CPU"
    bpy.context.scene.cycles.samples = 1
    return target, other


@pytest.mark.parametrize("selection", ["nothing", "other"])
def test_bake_geometry_channel_selects_only_target_and_restores_selection(
    monkeypatch: pytest.MonkeyPatch,
    selection: str,
) -> None:
    target, other = _create_bake_scene()
    view_layer = bpy.context.view_layer
    for obj in view_layer.objects:
        obj.select_set(state=False)
    if selection == "nothing":
        view_layer.objects.active = None
    else:
        other.select_set(state=True)
        view_layer.objects.active = other
    original_state = _selection_state()

    spy = _BakeSpyBpy()
    monkeypatch.setattr(blender_operations, "bpy", spy)

    pixels = blender_operations.bake_geometry_channel_to_array(
        target,
        "Normal",
        resolution=8,
    )

    assert pixels.shape == (8, 8, 4)
    assert spy.selected_during_bake == [target.name]
    assert spy.active_during_bake == target.name
    assert _selection_state() == original_state


def test_bake_geometry_channel_restores_selection_when_bake_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target, other = _create_bake_scene()
    original_material_count = len(target.data.materials)
    view_layer = bpy.context.view_layer
    target.select_set(state=False)
    other.select_set(state=True)
    view_layer.objects.active = other

    spy = _BakeSpyBpy(fail=True)
    monkeypatch.setattr(blender_operations, "bpy", spy)

    with pytest.raises(RuntimeError, match="bake failed"):
        blender_operations.bake_geometry_channel_to_array(target, resolution=8)

    assert spy.selected_during_bake == [target.name]
    assert _selection_state() == ([other.name], other.name)
    assert len(target.data.materials) == original_material_count
    assert "__bake_mat" not in bpy.data.materials
    assert "__bake_temp" not in bpy.data.images


def test_prepare_scene_selects_target_and_restore_scene_restores_selection() -> (
    None
):
    target, other = _create_bake_scene()
    view_layer = bpy.context.view_layer
    target.select_set(state=False)
    other.select_set(state=True)
    view_layer.objects.active = other

    backup = blender_operations.prepare_scene(target)
    try:
        assert _selection_state() == ([target.name], target.name)
    finally:
        blender_operations.restore_scene(backup, cameras=[])

    assert _selection_state() == ([other.name], other.name)
