import importlib
import sys
import types
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import numpy as np
import pytest


def _load_addon_submodule(submodule: str):
    root = Path(__file__).resolve().parents[2]
    package_name = f"addon_under_test_{uuid4().hex}"

    package_module = ModuleType(package_name)
    package_module.__path__ = [str(root)]

    class _Vector:
        def __init__(self, coords) -> None:
            self.coords = tuple(coords)

        def __sub__(self, other):
            return _Vector(
                a - b for a, b in zip(self.coords, other.coords, strict=False)
            )

        @property
        def length(self) -> float:
            return 0.0

        def to_track_quat(self, *args):  # noqa: ANN002, ARG002
            return SimpleNamespace(to_euler=lambda: (0.0, 0.0, 0.0))

    mathutils = ModuleType("mathutils")
    mathutils.Vector = _Vector

    class _BpyTypes:
        Operator = type("Operator", (), {})
        Context = object
        Scene = type("Scene", (), {})
        Object = object
        Camera = object
        Image = object

        def __getattr__(self, name: str):
            del name
            return object

    bpy_module = ModuleType("bpy")
    bpy_module.types = _BpyTypes()
    bpy_module.context = SimpleNamespace()

    texture_generation = ModuleType(f"{package_name}.texture_generation")
    texture_generation.load_multiview_images = lambda *args, **kwargs: {}  # noqa: ARG005
    texture_generation.run_texture_generation = lambda *args, **kwargs: None  # noqa: ARG005
    runtime_capability = ModuleType(f"{package_name}.runtime_capability")
    runtime_capability.get_runtime_capability = lambda context: SimpleNamespace(  # noqa: ARG005
        can_generate=True,
        message="Generation ready.",
    )

    modules = {
        package_name: package_module,
        "bpy": bpy_module,
        "mathutils": mathutils,
    }
    if submodule == "operators":
        modules[f"{package_name}.texture_generation"] = texture_generation
        modules[f"{package_name}.runtime_capability"] = runtime_capability

    with patch.dict(sys.modules, modules):
        return importlib.import_module(f"{package_name}.{submodule}")


def test_execute_restores_scene_when_render_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operators = _load_addon_submodule("operators")

    selected_obj = object()
    scene_backup = {"target_object": selected_obj}
    restore_calls = []

    def _raise_render_error(context, obj):  # noqa: ARG001
        msg = "forced render failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(operators, "prepare_scene", lambda obj: scene_backup)
    monkeypatch.setattr(operators, "render_views", _raise_render_error)
    monkeypatch.setattr(
        operators,
        "extract_process_parameter_from_context",
        lambda context: SimpleNamespace(sd_version="sd15", output_path="C:/tmp"),
    )
    monkeypatch.setattr(operators, "require_supported_sd_version", lambda version: None)
    monkeypatch.setattr(
        operators,
        "restore_scene",
        lambda backup, cameras: restore_calls.append((backup, cameras)),
    )

    monkeypatch.setattr(
        operators,
        "bpy",
        SimpleNamespace(
            data=SimpleNamespace(
                objects=SimpleNamespace(get=lambda name: selected_obj),
            ),
        ),
    )

    class _WindowManager:
        def progress_begin(self, start: int, end: int) -> None:  # noqa: ARG002
            return

        def progress_update(self, value: int) -> None:  # noqa: ARG002
            return

        def progress_end(self) -> None:
            return

        def event_timer_add(self, interval: float, window=None):  # noqa: ARG002
            return object()

        def modal_handler_add(self, operator) -> None:  # noqa: ARG002
            return

    class _Window:
        def cursor_set(self, value: str) -> None:  # noqa: ARG002
            return

    scene = SimpleNamespace(
        my_mesh_object="Cube",
        operation_mode="PARALLEL_IMG",
        input_texture=None,
        diffused_texture_operator_running=False,
        diffused_texture_operator_done=False,
        diffused_texture_operator_error="",
        diffused_texture_operator_cancel_requested=False,
    )
    context = SimpleNamespace(
        scene=scene,
        window_manager=_WindowManager(),
        window=_Window(),
    )

    class _FakeOperator:
        def report(self, levels: set[str], message: str) -> None:  # noqa: ARG002
            return

        def _set_scene_status(
            self,
            scene_obj,
            *,
            running: bool,
            done: bool,
            error: str = "",
        ) -> None:
            scene_obj.diffused_texture_operator_running = running
            scene_obj.diffused_texture_operator_done = done
            scene_obj.diffused_texture_operator_error = error

    operator = _FakeOperator()
    operator._should_cancel = types.MethodType(  # type: ignore[attr-defined]
        operators.OBJECT_OT_GenerateTexture._should_cancel,
        operator,
    )
    result = operators.OBJECT_OT_GenerateTexture.execute(operator, context)

    assert result == {"CANCELLED"}
    assert restore_calls == [(scene_backup, [])]


def test_execute_blocks_before_render_when_runtime_capability_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operators = _load_addon_submodule("operators")

    selected_obj = object()
    render_called = False
    reported: list[tuple[set[str], str]] = []

    monkeypatch.setattr(
        operators,
        "extract_process_parameter_from_context",
        lambda context: SimpleNamespace(sd_version="sd15", output_path="C:/tmp"),
    )
    monkeypatch.setattr(operators, "require_supported_sd_version", lambda version: None)
    monkeypatch.setattr(
        operators,
        "get_runtime_capability",
        lambda context: SimpleNamespace(
            can_generate=False,
            message=(
                "Dependency backend: rocm6.3. Cycles render: CPU. "
                "Diffusion dependencies are not importable."
            ),
        ),
    )

    def _render_views(context, obj):  # noqa: ARG001
        nonlocal render_called
        render_called = True
        return {}, []

    monkeypatch.setattr(operators, "render_views", _render_views)
    monkeypatch.setattr(
        operators,
        "bpy",
        SimpleNamespace(
            data=SimpleNamespace(
                objects=SimpleNamespace(get=lambda name: selected_obj),
            ),
        ),
    )

    class _WindowManager:
        def progress_begin(self, start: int, end: int) -> None:  # noqa: ARG002
            return

        def progress_update(self, value: int) -> None:  # noqa: ARG002
            return

        def progress_end(self) -> None:
            return

    class _Window:
        def cursor_set(self, value: str) -> None:  # noqa: ARG002
            return

    scene = SimpleNamespace(
        my_mesh_object="Cube",
        operation_mode="PARALLEL_IMG",
        input_texture=None,
        diffused_texture_operator_running=False,
        diffused_texture_operator_done=False,
        diffused_texture_operator_error="",
        diffused_texture_operator_cancel_requested=False,
    )
    context = SimpleNamespace(
        scene=scene,
        window_manager=_WindowManager(),
        window=_Window(),
    )

    class _FakeOperator:
        def report(self, levels: set[str], message: str) -> None:
            reported.append((levels, message))

        def _set_scene_status(
            self,
            scene_obj,
            *,
            running: bool,
            done: bool,
            error: str = "",
        ) -> None:
            scene_obj.diffused_texture_operator_running = running
            scene_obj.diffused_texture_operator_done = done
            scene_obj.diffused_texture_operator_error = error

    operator = _FakeOperator()
    operator._should_cancel = types.MethodType(  # type: ignore[attr-defined]
        operators.OBJECT_OT_GenerateTexture._should_cancel,
        operator,
    )
    result = operators.OBJECT_OT_GenerateTexture.execute(operator, context)

    assert result == {"CANCELLED"}
    assert not render_called
    assert scene.diffused_texture_operator_error.startswith(
        "Dependency backend: rocm6.3."
    )
    assert reported == [
        (
            {"ERROR"},
            (
                "Dependency backend: rocm6.3. Cycles render: CPU. "
                "Diffusion dependencies are not importable."
            ),
        ),
    ]


def test_execute_uses_uv_pass_assets_and_skips_multiview_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operators = _load_addon_submodule("operators")

    selected_obj = object()
    scene_backup = {"target_object": selected_obj}
    restore_calls = []
    built_assets = operators.UVPassAssets(
        normal_map=np.zeros((2, 2, 4), dtype=np.float32),
        position_map=np.zeros((2, 2, 4), dtype=np.float32),
        uv_layout=np.zeros((2, 2, 4), dtype=np.float32),
        surface_mask=np.zeros((2, 2), dtype=np.uint8),
    )
    thread_state: dict[str, object] = {}

    monkeypatch.setattr(operators, "prepare_scene", lambda obj: scene_backup)
    monkeypatch.setattr(
        operators,
        "build_uv_pass_assets",
        lambda context, obj: built_assets,  # noqa: ARG005
    )
    monkeypatch.setattr(
        operators,
        "render_views",
        lambda context, obj: (_ for _ in ()).throw(AssertionError("render_views")),  # noqa: ARG005, B023
    )
    monkeypatch.setattr(
        operators,
        "load_multiview_images",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("load_multiview_images")
        ),
    )
    monkeypatch.setattr(
        operators,
        "extract_process_parameter_from_context",
        lambda context: SimpleNamespace(
            sd_version="sd15",
            output_path="C:/tmp",
            operation_mode="UV_PASS",
        ),
    )
    monkeypatch.setattr(operators, "require_supported_sd_version", lambda version: None)
    monkeypatch.setattr(
        operators,
        "get_runtime_capability",
        lambda context: SimpleNamespace(can_generate=True, message="Generation ready."),
    )
    monkeypatch.setattr(
        operators,
        "restore_scene",
        lambda backup, cameras: restore_calls.append((backup, cameras)),
    )

    class _FakeThread:
        def __init__(self, target, args, daemon: bool) -> None:
            thread_state["target"] = target
            thread_state["args"] = args
            thread_state["daemon"] = daemon

        def start(self) -> None:
            thread_state["started"] = True

        def is_alive(self) -> bool:
            return False

    monkeypatch.setattr(operators.threading, "Thread", _FakeThread)
    monkeypatch.setattr(
        operators,
        "bpy",
        SimpleNamespace(
            data=SimpleNamespace(
                objects=SimpleNamespace(get=lambda name: selected_obj),
            ),
            app=SimpleNamespace(background=False),
        ),
    )

    class _WindowManager:
        def progress_begin(self, start: int, end: int) -> None:  # noqa: ARG002
            return

        def progress_update(self, value: int) -> None:  # noqa: ARG002
            return

        def progress_end(self) -> None:
            return

        def event_timer_add(self, interval: float, window=None):  # noqa: ARG002
            return object()

        def modal_handler_add(self, operator) -> None:  # noqa: ARG002
            return

    class _Window:
        def cursor_set(self, value: str) -> None:  # noqa: ARG002
            return

    scene = SimpleNamespace(
        my_mesh_object="Cube",
        operation_mode="UV_PASS",
        input_texture=None,
        diffused_texture_operator_running=False,
        diffused_texture_operator_done=False,
        diffused_texture_operator_error="",
        diffused_texture_operator_cancel_requested=True,
    )
    context = SimpleNamespace(
        scene=scene,
        window_manager=_WindowManager(),
        window=_Window(),
    )

    class _FakeOperator:
        def report(self, levels: set[str], message: str) -> None:  # noqa: ARG002
            return

        def _set_scene_status(
            self,
            scene_obj,
            *,
            running: bool,
            done: bool,
            error: str = "",
        ) -> None:
            scene_obj.diffused_texture_operator_running = running
            scene_obj.diffused_texture_operator_done = done
            scene_obj.diffused_texture_operator_error = error

    operator = _FakeOperator()
    operator._should_cancel = types.MethodType(  # type: ignore[attr-defined]
        operators.OBJECT_OT_GenerateTexture._should_cancel,
        operator,
    )
    operator._run_texture_generation_thread = types.MethodType(  # type: ignore[attr-defined]
        operators.OBJECT_OT_GenerateTexture._run_texture_generation_thread,
        operator,
    )
    result = operators.OBJECT_OT_GenerateTexture.execute(operator, context)

    assert result == {"RUNNING_MODAL"}
    assert restore_calls == [(scene_backup, [])]
    assert operator.render_img_folders is None
    assert thread_state["started"] is True
    args = thread_state["args"]
    assert isinstance(args, tuple)
    assert callable(args[2])
    assert callable(args[3])
    assert args[4] is built_assets


def test_cancel_operator_marks_scene_request(monkeypatch: pytest.MonkeyPatch) -> None:
    operators = _load_addon_submodule("operators")

    reported: list[tuple[set[str], str]] = []
    scene = SimpleNamespace(
        diffused_texture_operator_running=True,
        diffused_texture_operator_cancel_requested=False,
    )
    context = SimpleNamespace(scene=scene)

    class _FakeOperator:
        def report(self, levels: set[str], message: str) -> None:
            reported.append((levels, message))

    result = operators.OBJECT_OT_CancelTextureGeneration.execute(
        _FakeOperator(), context
    )

    assert result == {"FINISHED"}
    assert scene.diffused_texture_operator_cancel_requested is True
    assert reported == [({"INFO"}, "Cancelling texture generation...")]


def test_render_views_cleans_up_cameras_when_render_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blender_operations = _load_addon_submodule("blender_operations")

    class _ObjectStore(dict):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.removed = []

        def remove(self, obj, do_unlink: bool = True) -> None:  # noqa: FBT001, FBT002
            self.removed.append((obj.name, do_unlink))
            self.pop(obj.name, None)

    cam_a = SimpleNamespace(name="RenderCam_A")
    cam_b = SimpleNamespace(name="RenderCam_B")
    object_store = _ObjectStore({cam_a.name: cam_a, cam_b.name: cam_b})

    monkeypatch.setattr(
        blender_operations,
        "create_cameras_on_two_rings",
        lambda **kwargs: [cam_a, cam_b],
    )

    output_nodes = {
        "depth": SimpleNamespace(directory="C:/tmp/render_depth"),
        "normal": SimpleNamespace(directory="C:/tmp/render_normal"),
        "uv": SimpleNamespace(directory="C:/tmp/render_uv"),
    }
    monkeypatch.setattr(
        blender_operations,
        "setup_render_settings",
        lambda context, resolution: output_nodes,  # noqa: ARG005
    )
    monkeypatch.setattr(
        blender_operations,
        "get_output_node_directory",
        lambda output_node: output_node.directory,
    )
    monkeypatch.setattr(
        blender_operations,
        "set_output_node_directory",
        lambda output_node, output_dir: setattr(
            output_node, "directory", str(output_dir)
        ),
    )
    monkeypatch.setattr(
        blender_operations,
        "save_normals_in_camera_coordinates",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        blender_operations, "save_depth_condition", lambda **kwargs: None
    )
    monkeypatch.setattr(blender_operations, "save_facing_images", lambda **kwargs: None)

    def _raise_render_error(write_still: bool = True) -> None:  # noqa: ARG001
        msg = "forced render failure"
        raise RuntimeError(msg)

    monkeypatch.setattr(
        blender_operations,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(
                view_layer=SimpleNamespace(update=lambda: None),
            ),
            ops=SimpleNamespace(
                render=SimpleNamespace(render=_raise_render_error),
            ),
            data=SimpleNamespace(objects=object_store),
        ),
    )

    scene = SimpleNamespace(
        num_cameras="4",
        render_resolution="1024",
        camera="OriginalCamera",
    )
    context = SimpleNamespace(scene=scene)
    obj = SimpleNamespace(dimensions=(1.0, 2.0, 3.0))

    with pytest.raises(RuntimeError, match="forced render failure"):
        blender_operations.render_views(context, obj)

    assert scene.camera == "OriginalCamera"
    assert object_store.removed == [
        ("RenderCam_A", True),
        ("RenderCam_B", True),
    ]


def test_restore_scene_restores_camera_and_render_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blender_operations = _load_addon_submodule("blender_operations")

    class _ObjectStore(dict):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.removed = []

        def remove(self, obj, do_unlink: bool = True) -> None:  # noqa: FBT001, FBT002
            self.removed.append((obj.name, do_unlink))
            self.pop(obj.name, None)

    class _HiddenObject:
        def __init__(self, name: str) -> None:
            self.name = name
            self.hidden = True
            self.hide_render = True

        def hide_set(self, state: bool) -> None:  # noqa: FBT001
            self.hidden = state

    hidden = _HiddenObject("HiddenObj")
    temp_camera = SimpleNamespace(name="TempRenderCam")
    object_store = _ObjectStore({hidden.name: hidden, temp_camera.name: temp_camera})

    scene = SimpleNamespace(
        camera="current_camera",
        render=SimpleNamespace(
            engine="CYCLES",
            resolution_x=2048,
            resolution_y=2048,
            resolution_percentage=75,
            filepath="C:/tmp/current",
            filter_size=1.0,
            film_transparent=True,
            image_settings=SimpleNamespace(
                file_format="PNG",
                color_depth="8",
            ),
        ),
        cycles=SimpleNamespace(
            samples=64,
            use_denoising=True,
            use_light_tree=True,
            max_bounces=12,
            diffuse_bounces=4,
            glossy_bounces=4,
            transmission_bounces=4,
            volume_bounces=4,
            transparent_max_bounces=4,
        ),
    )

    updated = {"called": False}
    view_layer = SimpleNamespace(
        use_pass_z=False,
        use_pass_normal=False,
        use_pass_uv=False,
        use_pass_position=False,
        update=lambda: updated.__setitem__("called", True),
    )

    monkeypatch.setattr(
        blender_operations,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(scene=scene, view_layer=view_layer),
            data=SimpleNamespace(objects=object_store),
        ),
    )

    target_obj = SimpleNamespace(location=(99, 99, 99))
    backup_data = {
        "target_object": target_obj,
        "original_location": (1, 2, 3),
        "hidden_objects": [hidden],
        "original_scene_camera": "orig_camera",
        "original_render_settings": {
            "engine": "BLENDER_EEVEE",
            "resolution_x": 512,
            "resolution_y": 768,
            "resolution_percentage": 100,
            "filepath": "C:/tmp/original",
            "filter_size": 0.01,
            "film_transparent": False,
        },
        "original_image_settings": {
            "file_format": "OPEN_EXR",
            "color_depth": "32",
        },
        "original_cycles_settings": {
            "samples": 1,
            "use_denoising": False,
            "use_light_tree": False,
            "max_bounces": 1,
            "diffuse_bounces": 1,
            "glossy_bounces": 0,
            "transmission_bounces": 0,
            "volume_bounces": 0,
            "transparent_max_bounces": 0,
        },
        "original_view_layer_passes": {
            "use_pass_z": True,
            "use_pass_normal": True,
            "use_pass_uv": True,
            "use_pass_position": True,
        },
    }

    blender_operations.restore_scene(backup_data, [temp_camera])

    assert target_obj.location == (1, 2, 3)
    assert hidden.hidden is False
    assert hidden.hide_render is False
    assert scene.camera == "orig_camera"
    assert scene.render.engine == "BLENDER_EEVEE"
    assert scene.render.resolution_x == 512
    assert scene.render.resolution_y == 768
    assert scene.render.resolution_percentage == 100
    assert scene.render.filepath == "C:/tmp/original"
    assert scene.render.filter_size == 0.01
    assert scene.render.film_transparent is False
    assert scene.render.image_settings.file_format == "OPEN_EXR"
    assert scene.render.image_settings.color_depth == "32"
    assert scene.cycles.samples == 1
    assert scene.cycles.use_denoising is False
    assert scene.cycles.use_light_tree is False
    assert scene.cycles.max_bounces == 1
    assert scene.cycles.diffuse_bounces == 1
    assert scene.cycles.glossy_bounces == 0
    assert scene.cycles.transmission_bounces == 0
    assert scene.cycles.volume_bounces == 0
    assert scene.cycles.transparent_max_bounces == 0
    assert view_layer.use_pass_z is True
    assert view_layer.use_pass_normal is True
    assert view_layer.use_pass_uv is True
    assert view_layer.use_pass_position is True
    assert object_store.removed == [("TempRenderCam", True)]
    assert updated["called"] is True


def test_prepare_scene_snapshots_object_visibility(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blender_operations = _load_addon_submodule("blender_operations")

    class _VisibilityObject:
        def __init__(
            self,
            name: str,
            *,
            hidden: bool,
            hide_render: bool,
        ) -> None:
            self.name = name
            self.hidden = hidden
            self.hide_render = hide_render

        def hide_get(self) -> bool:
            return self.hidden

        def hide_set(self, state: bool) -> None:  # noqa: FBT001
            self.hidden = state

    target_obj = _VisibilityObject(
        "Target",
        hidden=False,
        hide_render=False,
    )
    render_hidden = _VisibilityObject(
        "RenderHidden",
        hidden=False,
        hide_render=True,
    )
    viewport_hidden = _VisibilityObject(
        "ViewportHidden",
        hidden=True,
        hide_render=False,
    )

    monkeypatch.setattr(
        blender_operations,
        "isolate_object",
        lambda obj: {
            "target_object": obj,
            "hidden_objects": [],
            "original_location": (0, 0, 0),
        },
    )

    scene = SimpleNamespace(
        camera="orig_camera",
        render=SimpleNamespace(
            engine="CYCLES",
            resolution_x=1024,
            resolution_y=1024,
            resolution_percentage=100,
            filepath="C:/tmp/output",
            filter_size=1.5,
            film_transparent=True,
            image_settings=SimpleNamespace(file_format="PNG", color_depth="8"),
        ),
        cycles=None,
    )
    view_layer = SimpleNamespace(
        use_pass_z=False,
        use_pass_normal=False,
        use_pass_uv=False,
        use_pass_position=False,
        objects=SimpleNamespace(active=None),
    )

    monkeypatch.setattr(
        blender_operations,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(scene=scene, view_layer=view_layer),
            data=SimpleNamespace(
                objects=[target_obj, render_hidden, viewport_hidden],
            ),
        ),
    )

    backup_data = blender_operations.prepare_scene(target_obj)

    assert backup_data["original_object_visibility"] == {
        "Target": {"hide_viewport": False, "hide_render": False},
        "RenderHidden": {"hide_viewport": False, "hide_render": True},
        "ViewportHidden": {"hide_viewport": True, "hide_render": False},
    }


def test_restore_scene_restores_exact_visibility_and_skips_deleted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    blender_operations = _load_addon_submodule("blender_operations")

    class _ObjectStore(dict):
        def remove(self, obj, do_unlink: bool = True) -> None:  # noqa: FBT001, FBT002, ARG002
            self.pop(obj.name, None)

    class _VisibilityObject:
        def __init__(
            self,
            name: str,
            *,
            hidden: bool,
            hide_render: bool,
        ) -> None:
            self.name = name
            self.hidden = hidden
            self.hide_render = hide_render

        def hide_get(self) -> bool:
            return self.hidden

        def hide_set(self, state: bool) -> None:  # noqa: FBT001
            self.hidden = state

    target_obj = SimpleNamespace(location=(99, 99, 99))
    keep_hidden = _VisibilityObject("KeepHidden", hidden=False, hide_render=True)
    keep_render_hidden = _VisibilityObject(
        "KeepRenderHidden", hidden=True, hide_render=False
    )

    object_store = _ObjectStore(
        {
            keep_hidden.name: keep_hidden,
            keep_render_hidden.name: keep_render_hidden,
        },
    )

    updated = {"called": False}
    view_layer = SimpleNamespace(update=lambda: updated.__setitem__("called", True))
    scene = SimpleNamespace(
        render=SimpleNamespace(image_settings=SimpleNamespace()),
        cycles=None,
    )

    monkeypatch.setattr(
        blender_operations,
        "bpy",
        SimpleNamespace(
            context=SimpleNamespace(scene=scene, view_layer=view_layer),
            data=SimpleNamespace(objects=object_store),
        ),
    )

    backup_data = {
        "target_object": target_obj,
        "original_location": (1, 2, 3),
        "original_object_visibility": {
            "KeepHidden": {"hide_viewport": True, "hide_render": False},
            "KeepRenderHidden": {"hide_viewport": False, "hide_render": True},
            "DeletedObj": {"hide_viewport": False, "hide_render": False},
        },
    }

    blender_operations.restore_scene(backup_data, cameras=[])

    assert target_obj.location == (1, 2, 3)
    assert keep_hidden.hide_get() is True
    assert keep_hidden.hide_render is False
    assert keep_render_hidden.hide_get() is False
    assert keep_render_hidden.hide_render is True
    assert updated["called"] is True
