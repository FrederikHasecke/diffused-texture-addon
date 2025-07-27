class MockScene:
    """Mockup Scene."""

    def __init__(self) -> None:
        """Initialize a Mock-up Scene."""
        self.num_loras = 0
        self.use_ipadapter = True
        self.ipadapter_strength = 0.5
        self.mesh_complexity = "HIGH"
        self.depth_controlnet_strength = 1.0
        self.canny_controlnet_strength = 1.0
        self.normal_controlnet_strength = 1.0
        self.sd_version = "sd15"
        self.checkpoint_path = "runwayml/stable-diffusion-v1-5"
        self.canny_controlnet_path = "lllyasviel/sd-controlnet-canny"
        self.normal_controlnet_path = "lllyasviel/sd-controlnet-normal"
        self.depth_controlnet_path = "lllyasviel/control_v11f1p_sd15_depth"


class MockUpContext:
    """Mockup Context used for default model download."""

    def __init__(self) -> None:
        """Initialize a Mock-up Context."""
        self.scene = MockScene()
