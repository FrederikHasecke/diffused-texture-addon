import bpy
from bpy.app.handlers import persistent
from bpy.props import BoolProperty, StringProperty

INTERRUPTED_RUN_MESSAGE = (
    "The previous texture generation was interrupted before it finished."
)


@persistent
def reset_stale_progress_state(_file_path: str = "") -> None:
    """Clear run flags that were saved while a generation was in flight.

    The flags live on the Scene, so a file saved (or auto-saved) mid-run, or
    recovered after a crash, would otherwise reopen with generation marked as
    running and the Start button permanently disabled.
    """
    for scene in bpy.data.scenes:
        if getattr(scene, "diffused_texture_operator_running", False):
            scene.diffused_texture_operator_running = False
            scene.diffused_texture_operator_error = INTERRUPTED_RUN_MESSAGE
        if getattr(scene, "diffused_texture_operator_cancel_requested", False):
            scene.diffused_texture_operator_cancel_requested = False


def _remove_reset_handlers() -> None:
    # Match by name so handlers from a previously loaded copy of the addon
    # (e.g. after a reload) are removed as well.
    for handler in list(bpy.app.handlers.load_post):
        if getattr(handler, "__name__", "") == reset_stale_progress_state.__name__:
            bpy.app.handlers.load_post.remove(handler)


def register_progress_properties() -> None:
    """Register properties to track the progress of the texture generation operator."""
    bpy.types.Scene.diffused_texture_operator_running = BoolProperty(
        name="Running",
        description="Is the texture generation operator running?",
    )
    bpy.types.Scene.diffused_texture_operator_done = BoolProperty(
        name="Done",
        description="Did the texture generation operator finish successfully?",
    )
    bpy.types.Scene.diffused_texture_operator_error = StringProperty(
        name="Error",
        description="Error message if the texture generation operator failed.",
    )
    bpy.types.Scene.diffused_texture_operator_cancel_requested = BoolProperty(
        name="Cancel Requested",
        description="Was cancellation requested for the texture generation operator?",
        default=False,
    )
    _remove_reset_handlers()
    bpy.app.handlers.load_post.append(reset_stale_progress_state)


def unregister_progress_properties() -> None:
    """Unregister properties that track the progress of the operator."""
    _remove_reset_handlers()
    for prop_name in (
        "diffused_texture_operator_running",
        "diffused_texture_operator_done",
        "diffused_texture_operator_error",
        "diffused_texture_operator_cancel_requested",
    ):
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
