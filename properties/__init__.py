from .controlnet import register_controlnet_properties, unregister_controlnet_properties
from .ipadapter import register_ipadapter_properties, unregister_ipadapter_properties
from .lora import register_lora_properties, unregister_lora_properties
from .mesh_settings import register_mesh_properties, unregister_mesh_properties
from .prompts import register_prompt_properties, unregister_prompt_properties
from .stable_diffusion import (
    register_stable_diffusion_properties,
    unregister_stable_diffusion_properties,
)


def register_properties() -> None:
    register_prompt_properties()
    register_mesh_properties()
    register_ipadapter_properties()
    register_lora_properties()
    register_stable_diffusion_properties()
    register_controlnet_properties()


def unregister_properties() -> None:
    unregister_controlnet_properties()
    unregister_stable_diffusion_properties()
    unregister_lora_properties()
    unregister_ipadapter_properties()
    unregister_mesh_properties()
    unregister_prompt_properties()
