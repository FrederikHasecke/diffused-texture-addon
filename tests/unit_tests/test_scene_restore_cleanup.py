import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

bpy = pytest.importorskip("bpy")


def _load_addon_submodule(submodule: str):
    root = Path(__file__).resolve().parents[2]
    package_name = f"addon_under_test_{uuid4().hex}"
    init_path = root / "__init__.py"

    spec = importlib.util.spec_from_file_location(
        package_name,
        init_path,
        submodule_search_locations=[str(root)],
    )
    if spec is None or spec.loader is None:
        msg = "Failed to create module spec for addon package."
        raise RuntimeError(msg)

    package_module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = package_module
    spec.loader.exec_module(package_module)

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

    result = operators.OBJECT_OT_GenerateTexture.execute(_FakeOperator(), context)

    assert result == {"CANCELLED"}
    assert restore_calls == [(scene_backup, [])]


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
        "position": SimpleNamespace(directory="C:/tmp/render_position"),
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
