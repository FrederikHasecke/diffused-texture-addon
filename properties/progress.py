import bpy
from bpy.props import BoolProperty, StringProperty


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


def unregister_progress_properties() -> None:
    """Unregister properties that track the progress of the operator."""
    for prop_name in (
        "diffused_texture_operator_running",
        "diffused_texture_operator_done",
        "diffused_texture_operator_error",
        "diffused_texture_operator_cancel_requested",
    ):
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
