import bpy  # noqa: I001
from bpy.props import BoolProperty, PointerProperty, FloatProperty


def update_ipadapter_image(self: bpy.types.Scene, context: bpy.types.Context) -> None:  # noqa: ARG001
    image = context.scene.ipadapter_image
    if image:
        image_data = bpy.data.images.get(image.name)
        if image_data != context.scene.ipadapter_image:
            context.scene.ipadapter_image = image_data


def register_ipadapter_properties() -> None:
    """Register all IPAdapter Properties."""
    bpy.types.Scene.use_ipadapter = BoolProperty(
        name="Use IPAdapter",
        description="Enable IPAdapter for image-based conditioning",
        default=False,
    )
    bpy.types.Scene.ipadapter_image = PointerProperty(
        type=bpy.types.Image,
        name="IPAdapter Image",
        description="Reference image for IPAdapter",
        update=update_ipadapter_image,
    )
    bpy.types.Scene.ipadapter_strength = FloatProperty(
        name="IPAdapter Strength",
        description="Controls blend between text and image prompts",
        default=0.5,
        min=0.0,
        soft_max=1.0,
    )


def unregister_ipadapter_properties() -> None:
    for prop_name in ("use_ipadapter", "ipadapter_image", "ipadapter_strength"):
        if hasattr(bpy.types.Scene, prop_name):
            delattr(bpy.types.Scene, prop_name)
