import bpy
from bpy.props import BoolProperty, PointerProperty, FloatProperty


def update_ipadapter_image(self, context):
    image = context.scene.ipadapter_image
    if image:
        image_data = bpy.data.images.get(image.name)
        if image_data != context.scene.ipadapter_image:
            context.scene.ipadapter_image = image_data


def register_ipadapter_properties():
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


def unregister_ipadapter_properties():
    del bpy.types.Scene.use_ipadapter
    del bpy.types.Scene.ipadapter_image
    del bpy.types.Scene.ipadapter_strength
