from pathlib import Path
from uuid import uuid4

import pytest

import bpy
import render_setup
find_output_node_image_path = render_setup.find_output_node_image_path
get_output_node_directory = render_setup.get_output_node_directory
get_output_node_file_prefix = render_setup.get_output_node_file_prefix
get_scene_compositor_node_tree = render_setup.get_scene_compositor_node_tree
set_node_path = render_setup.set_node_path
set_output_node_directory = render_setup.set_output_node_directory
setup_output_nodes = render_setup.setup_output_nodes


def _new_compositor_node_tree() -> bpy.types.NodeTree:
    return bpy.data.node_groups.new(
        f"test_comp_tree_{uuid4().hex}",
        "CompositorNodeTree",
    )


def _cleanup_compositor_node_tree(node_tree: bpy.types.NodeTree) -> None:
    if node_tree.name in bpy.data.node_groups:
        bpy.data.node_groups.remove(node_tree)


def test_get_scene_compositor_node_tree_returns_tree() -> None:
    scene = bpy.context.scene

    if hasattr(scene, "node_tree"):
        scene.use_nodes = True
        assert get_scene_compositor_node_tree(scene) is scene.node_tree
        return

    original = scene.compositing_node_group
    scene.compositing_node_group = None

    created_tree = get_scene_compositor_node_tree(scene)

    assert created_tree is not None
    assert scene.compositing_node_group is created_tree

    scene.compositing_node_group = original
    if original is None and created_tree.users == 0:
        bpy.data.node_groups.remove(created_tree)


def test_output_node_directory_roundtrip(tmp_path: Path) -> None:
    node_tree = _new_compositor_node_tree()

    try:
        output_node = node_tree.nodes.new("CompositorNodeOutputFile")
        output_dir = tmp_path / "render_depth"

        set_output_node_directory(output_node, output_dir)

        assert Path(get_output_node_directory(output_node)) == output_dir
    finally:
        _cleanup_compositor_node_tree(node_tree)


def test_set_node_path_handles_uv_socket_name_lookup() -> None:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    old_engine = scene.render.engine
    old_use_pass_uv = view_layer.use_pass_uv
    render_setup.setup_cycles_setting(bpy.context)
    view_layer.use_pass_uv = True

    node_tree = _new_compositor_node_tree()

    try:
        render_layers = node_tree.nodes.new("CompositorNodeRLayers")
        output_node = set_node_path("UV", node_tree, render_layers)

        assert output_node is not None
        assert get_output_node_file_prefix(output_node) == "uv_"
        assert any(
            link.from_socket.name == "UV" and link.to_node == output_node
            for link in node_tree.links
        )
    finally:
        _cleanup_compositor_node_tree(node_tree)
        view_layer.use_pass_uv = old_use_pass_uv
        scene.render.engine = old_engine


def test_find_output_node_image_path_resolves_frame_file(tmp_path: Path) -> None:
    scene = bpy.context.scene
    view_layer = bpy.context.view_layer
    old_engine = scene.render.engine
    old_use_pass_z = view_layer.use_pass_z
    render_setup.setup_cycles_setting(bpy.context)
    view_layer.use_pass_z = True

    node_tree = _new_compositor_node_tree()

    try:
        render_layers = node_tree.nodes.new("CompositorNodeRLayers")
        output_node = set_node_path("Depth", node_tree, render_layers)
        set_output_node_directory(output_node, tmp_path)

        expected = tmp_path / "depth_0001.exr"
        expected.touch()

        found = find_output_node_image_path(output_node, 1)

        assert found == expected
    finally:
        _cleanup_compositor_node_tree(node_tree)
        view_layer.use_pass_z = old_use_pass_z
        scene.render.engine = old_engine


def test_setup_output_nodes_returns_expected_passes(tmp_path: Path) -> None:
    scene = bpy.context.scene
    old_engine = scene.render.engine
    created_property = False

    if not hasattr(bpy.types.Scene, "output_path"):
        bpy.types.Scene.output_path = bpy.props.StringProperty(default="")
        created_property = True

    previous_output_path = scene.output_path
    scene.output_path = str(tmp_path) + "/"
    render_setup.setup_cycles_setting(bpy.context)

    try:
        output_nodes = setup_output_nodes(bpy.context)
    finally:
        scene.output_path = previous_output_path
        scene.render.engine = old_engine
        if created_property and hasattr(bpy.types.Scene, "output_path"):
            del bpy.types.Scene.output_path

    assert sorted(output_nodes.keys()) == ["depth", "normal", "uv"]
