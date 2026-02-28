import importlib
import tempfile
import time
from pathlib import Path


ADDON_MODULE_CANDIDATES = (
    "diffused_texture_addon",
    "bl_ext.user_default.diffused_texture_addon",
)


def _enable_addon() -> str:
    import bpy

    errors: list[str] = []
    for module_name in ADDON_MODULE_CANDIDATES:
        try:
            result = bpy.ops.preferences.addon_enable(module=module_name)
            if "FINISHED" in result:
                return module_name
        except RuntimeError as exc:
            errors.append(f"{module_name}: {exc}")

    try:
        addon_module = importlib.import_module("diffused_texture_addon")
        addon_module.register()
        return "diffused_texture_addon"
    except Exception as exc:  # noqa: BLE001
        errors.append(f"manual register failed: {exc}")

    error_text = " | ".join(errors)
    raise AssertionError(f"Could not enable addon. Errors: {error_text}")


def _wait_until_finished(timeout_sec: int = 900) -> None:
    import bpy

    scene = bpy.context.scene
    start = time.time()
    while bool(getattr(scene, "diffused_texture_operator_running", False)):
        if time.time() - start > timeout_sec:
            raise TimeoutError(
                f"Texture generation did not finish within {timeout_sec} seconds."
            )
        time.sleep(1)


def test_run_parallel() -> None:
    """Test the parallel image-based texture generation end to end."""
    try:
        import bpy
    except ImportError as exc:
        raise AssertionError("bpy module is required for this e2e test") from exc

    _enable_addon()

    if not bpy.app.online_access:
        bpy.context.preferences.system.use_online_access = True
    assert bpy.app.online_access, (
        "Failed to enable online access in Blender preferences"
    )

    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    bpy.ops.mesh.primitive_monkey_add(size=2, location=(0, 0, 0))
    bpy.ops.object.mode_set(mode="OBJECT")

    obj = bpy.context.selected_objects[0]
    if not obj.data.uv_layers:
        obj.data.uv_layers.new(name="UVMap")

    output_path = Path(tempfile.gettempdir())
    existing = {p.name for p in output_path.glob("output_texture_*.png")}

    scene = bpy.context.scene
    scene.my_prompt = "Monkey head"
    scene.my_negative_prompt = "CGI, unrealistic, deformed, blurry"
    scene.denoise_strength = 1.0
    scene.num_inference_steps = 1
    scene.guidance_scale = 7.5
    scene.operation_mode = "PARALLEL_IMG"
    scene.mesh_complexity = "LOW"
    scene.num_cameras = "4"
    scene.texture_resolution = "512"
    scene.render_resolution = "1024"
    scene.output_path = str(output_path)
    scene.my_mesh_object = obj.name
    scene.my_uv_map = obj.data.uv_layers.active.name
    scene.custom_sd_resolution = 64

    op_result = bpy.ops.object.generate_texture()
    assert "RUNNING_MODAL" in op_result or "FINISHED" in op_result

    _wait_until_finished()

    error_text = str(getattr(scene, "diffused_texture_operator_error", ""))
    assert not error_text, f"Texture generation failed: {error_text}"

    generated = {p.name for p in output_path.glob("output_texture_*.png")}
    assert generated - existing, "No output texture file was generated"
