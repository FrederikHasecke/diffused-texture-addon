import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np


def _load_mesh_settings_module() -> tuple[ModuleType, type]:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "properties" / "mesh_settings.py"
    package_name = f"addon_under_test_{uuid4().hex}"
    module_name = f"{package_name}.properties.mesh_settings"

    addon_pkg = ModuleType(package_name)
    addon_pkg.__path__ = [str(repo_root)]

    properties_pkg = ModuleType(f"{package_name}.properties")
    properties_pkg.__path__ = [str(repo_root / "properties")]

    utils_module = ModuleType(f"{package_name}.utils")
    utils_module.get_mesh_objects = lambda *args, **kwargs: []  # noqa: ARG005
    utils_module.update_uv_maps = lambda *args, **kwargs: []  # noqa: ARG005

    scene_type = type("Scene", (), {})

    bpy_module = ModuleType("bpy")
    bpy_module.types = SimpleNamespace(Scene=scene_type, Context=object, Image=object)
    bpy_module.props = SimpleNamespace(PointerProperty=lambda **kwargs: kwargs)
    bpy_module.path = SimpleNamespace(abspath=lambda value: value)

    props_module = ModuleType("bpy.props")
    props_module.EnumProperty = lambda **kwargs: kwargs
    props_module.FloatProperty = lambda **kwargs: kwargs
    props_module.IntProperty = lambda **kwargs: kwargs
    props_module.StringProperty = lambda **kwargs: kwargs

    with patch.dict(
        sys.modules,
        {
            package_name: addon_pkg,
            f"{package_name}.properties": properties_pkg,
            f"{package_name}.utils": utils_module,
            "bpy": bpy_module,
            "bpy.props": props_module,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load mesh_settings module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module, scene_type


def _load_blender_operations_module() -> ModuleType:
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "blender_operations.py"
    package_name = f"addon_under_test_{uuid4().hex}"
    module_name = f"{package_name}.blender_operations"

    addon_pkg = ModuleType(package_name)
    addon_pkg.__path__ = [str(repo_root)]

    class _BpyTypes:
        Context = object
        Scene = type("Scene", (), {})
        Image = object
        Object = object
        Camera = object

        def __getattr__(self, name: str):
            del name
            return object

    bpy_module = ModuleType("bpy")
    bpy_module.types = _BpyTypes()
    bpy_module.context = SimpleNamespace()

    diagnostics_module = ModuleType(f"{package_name}.diagnostics")
    diagnostics_module.get_logger = lambda name: SimpleNamespace(  # noqa: ARG005
        debug=lambda *args, **kwargs: None,
        exception=lambda *args, **kwargs: None,
    )

    uv_seams_module = ModuleType(f"{package_name}.diffusedtexture.uv_seams")
    uv_seams_module.UV_EPSILON = 1e-6
    uv_seams_module.SeamTopologyAssets = object
    uv_seams_module.UVSeamCandidate = object
    uv_seams_module.empty_float_array = lambda: np.array([], dtype=np.float32)
    uv_seams_module.empty_int_array = lambda: np.array([], dtype=np.int32)
    uv_seams_module.empty_yx_array = lambda: np.zeros((0, 2), dtype=np.int32)
    uv_seams_module.normalize_uv_vector = lambda value: value
    uv_seams_module.build_uv_seam_topology_assets = (
        lambda *args, **kwargs: SimpleNamespace(
            seam_line_mask=None,
            seam_link_source_yx=np.zeros((0, 2), dtype=np.int32),
            seam_link_target_yx=np.zeros((0, 2), dtype=np.int32),
            seam_link_weight=np.array([], dtype=np.float32),
            seam_link_edge_id=np.array([], dtype=np.int32),
            seam_link_t=np.array([], dtype=np.float32),
            seam_unresolved_link_mask=None,
        )
    )

    render_setup_module = ModuleType(f"{package_name}.render_setup")
    render_setup_module.create_cameras_on_sphere = lambda *args, **kwargs: []
    render_setup_module.create_cameras_on_two_rings = lambda *args, **kwargs: []
    render_setup_module.clear_render_output_paths = lambda *args, **kwargs: None
    render_setup_module.find_output_node_image_path = lambda *args, **kwargs: ""
    render_setup_module.get_output_node_directory = lambda *args, **kwargs: ""
    render_setup_module.set_output_node_directory = lambda *args, **kwargs: None
    render_setup_module.setup_render_settings = lambda *args, **kwargs: {}

    utils_module = ModuleType(f"{package_name}.utils")
    utils_module.isolate_object = lambda *args, **kwargs: None

    with patch.dict(
        sys.modules,
        {
            package_name: addon_pkg,
            "bpy": bpy_module,
            f"{package_name}.diagnostics": diagnostics_module,
            f"{package_name}.diffusedtexture.uv_seams": uv_seams_module,
            f"{package_name}.render_setup": render_setup_module,
            f"{package_name}.utils": utils_module,
        },
    ):
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None or spec.loader is None:
            msg = "Could not load blender_operations module spec."
            raise RuntimeError(msg)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module


def test_mesh_settings_expose_only_image_operation_modes() -> None:
    mesh_settings, scene_type = _load_mesh_settings_module()

    mesh_settings.register_mesh_properties()
    try:
        operation_mode_items = scene_type.operation_mode["items"]
        assert [item[0] for item in operation_mode_items] == [  # noqa: S101
            "PARALLEL_IMG",
            "SEQUENTIAL_IMG",
            "PARA_SEQUENTIAL_IMG",
        ]
    finally:
        mesh_settings.unregister_mesh_properties()


def test_extract_process_parameter_rejects_disabled_uv_mode() -> None:
    blender_operations = _load_blender_operations_module()

    context = SimpleNamespace(
        scene=SimpleNamespace(
            operation_mode="UV_PASS",
            num_loras=0,
        ),
    )

    try:
        blender_operations.extract_process_parameter_from_context(context)
    except ValueError as exc:
        assert "Dedicated UV mode is currently disabled." in str(exc)  # noqa: S101
    else:
        raise AssertionError("Expected UV_PASS to be rejected during extraction.")
