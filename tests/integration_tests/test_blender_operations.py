# ruff: noqa: PLR2004, S101

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

bpy = pytest.importorskip("bpy")
from diffused_texture_addon import blender_operations


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
