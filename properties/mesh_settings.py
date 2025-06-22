import bpy
from bpy.props import (  # type: ignore  # noqa: PGH003
    EnumProperty,
    FloatProperty,
    IntProperty,
    StringProperty,
)

from ..utils import get_mesh_objects, update_uv_maps


def update_input_image(self: bpy.types.Scene, context: bpy.types.Context) -> None:
    """Ensure the selected image from the preview window is set in scene.input_image."""
    image = context.scene.input_texture
    if image:
        image_data = bpy.data.images.get(image.name)

        # Only set the image if it's not already correctly set to prevent recursion
        if image_data != context.scene.input_texture:
            context.scene.input_texture = image_data


def update_output_path(self: bpy.types.Scene, context: bpy.types.Context) -> None:
    if context.scene.output_path.startswith("//"):
        context.scene.output_path = bpy.path.abspath(context.scene.output_path)


def register_mesh_properties() -> None:
    bpy.types.Scene.my_mesh_object = EnumProperty(
        name="Mesh Object",
        items=get_mesh_objects,
        description="Select the mesh object to texture",
    )

    bpy.types.Scene.my_uv_map = EnumProperty(
        name="UV Map",
        items=update_uv_maps,
        description="Select UV map for texture baking",
    )

    bpy.types.Scene.texture_seed = IntProperty(
        name="Seed",
        description="Random seed",
        default=0,
        min=0,
    )

    bpy.types.Scene.input_texture = bpy.props.PointerProperty(
        type=bpy.types.Image,
        name="Input Texture",
        description="Select an image to use as input texture",
        update=update_input_image,
    )

    bpy.types.Scene.mesh_complexity = EnumProperty(
        name="Mesh Complexity",
        description="Set mesh complexity level",
        items=[("LOW", "Low", ""), ("MEDIUM", "Medium", ""), ("HIGH", "High", "")],
    )

    bpy.types.Scene.num_cameras = EnumProperty(
        name="Cameras",
        description="Number of viewpoints for rendering",
        items=[("4", "4", ""), ("9", "9", ""), ("16", "16", "")],
    )

    bpy.types.Scene.texture_resolution = EnumProperty(
        name="Texture Resolution",
        items=[
            ("512", "512x512", ""),
            ("1024", "1024x1024", ""),
            ("2048", "2048x2048", ""),
            ("4096", "4096x4096", ""),
        ],
        default="1024",
    )

    bpy.types.Scene.render_resolution = EnumProperty(
        name="Render Resolution",
        items=[
            ("1024", "1024x1024", ""),
            ("2048", "2048x2048", ""),
            ("4096", "4096x4096", ""),
            ("8192", "8192x8192", ""),
        ],
        default="2048",
    )

    bpy.types.Scene.output_path = StringProperty(
        name="Output Path",
        description="Directory to store the resulting texture and temporary files",
        subtype="DIR_PATH",
        default="",
        update=update_output_path,
    )

    bpy.types.Scene.operation_mode = EnumProperty(
        name="Operation Mode",
        items=[
            ("PARALLEL_IMG", "Parallel", "Run views in parallel"),
            ("SEQUENTIAL_IMG", "Sequential", "Run views one by one"),
            ("PARA_SEQUENTIAL_IMG", "Para-Sequential", "Run subsets parallel"),
            ("UV_PASS", "Texture Pass", "Run on a flat texture"),
        ],
        default="PARALLEL_IMG",
    )

    bpy.types.Scene.num_inference_steps = IntProperty(
        name="Steps",
        default=50,
        min=1,
        max=100,
    )

    bpy.types.Scene.denoise_strength = FloatProperty(
        name="Denoise Strength",
        default=1.0,
        min=0.0,
        max=1.0,
    )


def unregister_mesh_properties() -> None:
    props = [
        "my_mesh_object",
        "my_uv_map",
        "texture_seed",
        "mesh_complexity",
        "num_cameras",
        "texture_resolution",
        "render_resolution",
        "output_path",
        "operation_mode",
        "num_inference_steps",
        "denoise_strength",
    ]
    for p in props:
        delattr(bpy.types.Scene, p)
