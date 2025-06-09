class MockUpScene:
    def __init__(self):
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
        self.depth_controlnet_path = "lllyasviel/sd-controlnet-depth"
