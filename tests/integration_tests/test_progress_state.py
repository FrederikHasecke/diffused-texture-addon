import importlib
from pathlib import Path

import bpy


def _progress_module():
    return importlib.import_module("diffused_texture_addon.properties.progress")


def _reset_handlers(progress) -> list:
    return [
        handler
        for handler in bpy.app.handlers.load_post
        if getattr(handler, "__name__", "")
        == progress.reset_stale_progress_state.__name__
    ]


def test_stale_running_flag_is_cleared_when_file_is_reopened(tmp_path: Path) -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    progress = _progress_module()
    progress.register_progress_properties()
    try:
        scene = bpy.context.scene
        scene.diffused_texture_operator_running = True
        scene.diffused_texture_operator_cancel_requested = True

        blend_path = tmp_path / "saved_mid_run.blend"
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        bpy.ops.wm.open_mainfile(filepath=str(blend_path))

        scene = bpy.context.scene
        assert scene.diffused_texture_operator_running is False
        assert scene.diffused_texture_operator_cancel_requested is False
        assert (
            scene.diffused_texture_operator_error
            == progress.INTERRUPTED_RUN_MESSAGE
        )
    finally:
        progress.unregister_progress_properties()


def test_reset_handler_registers_once_and_unregisters() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    progress = _progress_module()

    progress.register_progress_properties()
    progress.register_progress_properties()
    assert len(_reset_handlers(progress)) == 1

    progress.unregister_progress_properties()
    assert _reset_handlers(progress) == []


def test_reset_leaves_idle_scene_untouched() -> None:
    bpy.ops.wm.read_factory_settings(use_empty=True)
    progress = _progress_module()
    progress.register_progress_properties()
    try:
        scene = bpy.context.scene
        scene.diffused_texture_operator_error = "earlier failure"

        progress.reset_stale_progress_state("")

        assert scene.diffused_texture_operator_running is False
        assert scene.diffused_texture_operator_error == "earlier failure"
    finally:
        progress.unregister_progress_properties()
