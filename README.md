# DiffusedTexture: AI-Powered Texture Generation for Blender

DiffusedTexture is a Blender add-on that uses Stable Diffusion to create textures directly on 3D meshes. 

## Table of Contents
- [DiffusedTexture: AI-Powered Texture Generation for Blender](#diffusedtexture-ai-powered-texture-generation-for-blender)
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

## Features
- **Direct Texture Generation:** Diffuse Textures are generated directly on the 3D model within Blender, enabling WYSIWYG (what you see is what you get) results.
- **Modes for Different Workflows:**
  - **Text2Image Parallel**: Generates textures from text prompts.
  - **Image2Image Parallel**: Generates textures from input images, applied parallel across all views.
  - **Image2Image Sequential**: Sequentially applies textures across views, great for refinement.
- **LoRA Integration**: Uses LoRA conditioning for specific styles.
- **IPAdapter Integration**: Fit specific styles or objects with images for enhanced flexibility and control.

## Installation
Download the [latest release](https://github.com/FrederikHasecke/diffused-texture-addon/releases/latest) and install it as an Add-On.

## Setup
### Blender Setup
Ensure your Blender preferences are configured for CUDA (for GPU rendering). If you’re using an NVIDIA GPU, enable cycles and set up with either CUDA or OPTIX.

## Usage

### Main Workflow
1. **Open a 3D Model**: Open the `.blend` file containing the 3D model to texture.
2. **Open the DiffusedTexture Panel**: The add-on will appear in the right-hand panel under `DiffusedTexture`.
3. **Select Operation Mode**: Choose one of the three operation modes:
   - `Text2Image Parallel`
   - `Image2Image Parallel`
   - `Image2Image Sequential`

### Additional Options
- **LoRA Models**: Add one or multiple LoRA models to refine results.
- **IPAdapter**: Customize texture using conditioning images.

### Outputs
Generated textures will be saved in the specified output path. 

## Configuration
DiffusedTexture has customizable settings in the Preferences panel:

- **Mesh Complexity**: Adjusts ControlNets based on object polycount.
  -  **Low Complexity**: Depth ControlNet Only
  -  **Mid Complexity**: Depth and Canny ControlNets
  -  **High Complexity**: Depth, Canny and Normalmap ControlNets
- **Texture Resolution**: Options range from 256x256 to 4096x4096.
- **Output Path**: Specify where the generated textures will be saved.
- **Checkpoint Path**: For specifying the Stable Diffusion checkpoint file.

## Troubleshooting
- **Freezes**: Open up the Terminal before executing the Addon to see the progress bar.
- **Add-On Not Showing Up**: Ensure the add-on is enabled in Blender's preferences.
- **CUDA/OPTIX Issues**: Verify GPU support is enabled in Blender and that the correct drivers are installed.
- **Slow Rendering**: Higher resolutions and camera counts can increase memory usage and render times.
- **RuntimeError: Error: Cannot open file [...]: Permission denied**: Create a new Folder and select that one.
- **torch.OutOfMemoryError: CUDA out of memory.**: Choose less cameras in the parallel tasks and close all other processes that might use GPU Memory.
