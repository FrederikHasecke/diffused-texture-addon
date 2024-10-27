# DiffuseTex: AI-Powered Texture Generation for Blender

DiffuseTex is a Blender add-on that uses advanced image generation techniques to create textures directly on 3D meshes. Using Stable Diffusion and LoRA conditioning, DiffuseTex can texture models from scratch or enhance existing ones. By integrating texture generation directly into Blender, DiffuseTex streamlines workflows and expands creative possibilities, making it easier than ever to texture models with high detail and artistic control.

## Table of Contents
- [DiffuseTex: AI-Powered Texture Generation for Blender](#diffusetex-ai-powered-texture-generation-for-blender)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Setup](#setup)
    - [Blender Setup](#blender-setup)
  - [Usage](#usage)
    - [Main Workflow](#main-workflow)
    - [Additional Options](#additional-options)
    - [Outputs](#outputs)
  - [Configuration](#configuration)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [License](#license)

## Features
- **Direct Texture Generation:** Textures are generated directly on the 3D model within Blender, enabling WYSIWYG (what you see is what you get) results.
- **Modes for Different Workflows:**
  - **Text2Image Parallel**: Generates textures from text prompts.
  - **Image2Image Parallel**: Generates textures from input images, applied parallelly across views.
  - **Image2Image Sequential**: Sequentially applies textures across views, great for refinement.
  - **Texture2Texture Enhancement**: Enhances existing textures with subtle yet powerful updates.
- **Camera-Based UV Mapping**: Automatically arranges viewpoints around the object and projects UV maps accordingly.
- **LoRA Integration**: Uses LoRA conditioning to further refine texture outputs.
- **IPAdapter Integration**: Customize textures with conditioning images for enhanced flexibility and control.
- **In-Painting and Weighting for Detail**: Fill in gaps and enhance detail based on camera-facing angle and UV mapping.

## Installation
1. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/yourusername/DiffuseTex.git
   ```

2. Open Blender as an administrator (required on Windows for package installations).
3. Open the Blender scripting editor and load install_requirements.py from the _DiffuseTex_ folder.
4. Run install_requirements.py to install necessary Python packages.

## Setup
### Blender Setup
Ensure your Blender preferences are configured for CUDA (for GPU rendering). If you’re using an NVIDIA GPU, make cycles is enabled and set up with either CUDA or OPTIX.

## Usage

### Main Workflow
1. **Open a 3D Model**: Open the `.blend` file containing the 3D model to texture.
2. **Open the DiffuseTex Panel**: The add-on will appear in the right-hand panel under `DiffuseTex`.
3. **Select Operation Mode**: Choose one of the four operation modes:
   - `Text2Image Parallel`
   - `Image2Image Parallel`
   - `Image2Image Sequential`
   - `Texture2Texture Enhancement`
4. **Configure Parameters**:
   - **Viewpoints**: Choose the number of viewpoints for texture projection.
   - **Denoise Strength**: Set the denoise level (auto-set to 1.0 for Text2Image Parallel).
   - **Input Texture**: For image-based or texture-enhancement modes, select an input image.
5. **Start Texture Generation**: Click `Start Texture Generation` to begin.

### Additional Options
- **LoRA Models**: Add multiple LoRA models to refine results.
- **IPAdapter**: Customize texture using conditioning images.

### Outputs
Generated textures will be saved in the specified output path. 

## Configuration
DiffuseTex has customizable settings in the Preferences panel:

- **Mesh Complexity**: Adjusts ControlNets based on object polycount.
  -  **Low Complexity**: Depth ControlNet Only
  -  **Mid Complexity**: Depth and Canny ControlNets
  -  **High Complexity**: Depth, Canny and Normalmap ControlNets
- **Texture Resolution**: Options range from 256x256 to 4096x4096.
- **Output Path**: Specify where the generated textures will be saved.
- **Checkpoint Path**: For specifying the Stable Diffusion checkpoint file.

## Troubleshooting
- **Add-On Not Showing Up**: Ensure the add-on is enabled in Blender's preferences.
- **CUDA/OPTIX Issues**: Verify GPU support is enabled in Blender and that the correct drivers are installed.
- **Slow Rendering**: Higher resolutions and camera counts can increase memory usage and render times.

## Contributing
We welcome contributions! To get started:
1. Fork the repository.
2. Create a new branch for your feature or bugfix.
3. Submit a pull request, explaining your changes.

## License
Distributed under the MIT License. See `LICENSE` for more information.
