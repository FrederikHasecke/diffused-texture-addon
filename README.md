# DiffusedTexture: AI-Powered Texture Generation for Blender

DiffusedTexture is a Blender add-on that uses Stable Diffusion to create textures directly on 3D meshes. 

![https://www.cgtrader.com/free-3d-print-models/miniatures/other/elephant-natural-history-museum-1](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/elephant.gif)
![https://graphics.stanford.edu/data/3Dscanrep/](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/bunny.gif)

## Table of Contents
- [DiffusedTexture: AI-Powered Texture Generation for Blender](#diffusedtexture-ai-powered-texture-generation-for-blender)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation (Windows)](#installation-windows)
  - [Installation (Linux)](#installation-linux)
  - [Setup](#setup)
    - [Blender Configuration](#blender-configuration)
  - [Usage](#usage)
    - [Additional Options](#additional-options)
  - [Troubleshooting](#troubleshooting)
  - [**Roadmap**](#roadmap)
  - [**Acknowledgements**](#acknowledgements)

## Features
- **AI-Driven Texture Creation:** Generate diffuse textures directly on 3D models
- **Modes for Different Workflows:**
  - **Text2Image Parallel**: Create textures from text prompts, ensuring global consistency.
  - **Image2Image Parallel**: Generates textures from input textures, applied parallel across all views.
  - **Image2Image Sequential**: Sequentially adjusts textures across views, great for refinement.
- **LoRA Integration**: Uses LoRA conditioning for specific styles.
- **IPAdapter Integration**: Fit specific styles or objects with images for enhanced flexibility and control.

## Installation (Windows)
0. Download [7-Zip](https://7-zip.de/download.html) 
1. Download all .tar files of the [latest release](https://github.com/FrederikHasecke/diffused-texture-addon/releases/latest)
2. Untar the file `diffused_texture_addon.7z.001` this will automatically untar the other `.7z` files
    >**WARNING:**    _DO NOT_ unzip the resulting `diffused_texture_addon.zip`
3.  Install the `diffused_texture_addon.zip` file in Blender as an Add-On.
    1.  `Edit` -> `Preferences...` -> Sidebar `Add-ons` -> Top right corner dropdown menu -> `Install from Disk...`

## Installation (Linux)
- TODO: Test on Linux Instructions and test if it works

## Setup
### Blender Configuration
1. Enable CUDA or OPTIX in Blender if using an NVIDIA GPU.
     - Go to `Edit > Preferences > System` and configure GPU settings.
     - **Note:** Requires a modern NVIDIA GPU with at least 12GB (TODO: Maybe 8?) VRAM.
2. Download necessary models (~10.6 GB total):
     - Optionally specify a custom cache directory via the `HuggingFace Cache Path` setting during installation.
     - **Tip:** Open Blender's system console (`Window > Toggle System Console`) to monitor download progress.
## Usage

![General Usage](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/usage.gif)

1. **Load a 3D Model**:
   - Import or create a `.blend` file containing the 3D model.
2. **UV Unwrap the Model**:
   - Apply a UV map (`Smart UV Project` works well).
3. **Access the Add-On**:
   - Open the `DiffusedTexture` panel in the N-panel (right-hand sidebar).
4. **Set Up Texture Generation**:
   - **Prompt & Negative Prompt**: Describe the desired texture/object and what to avoid.
   - **Guidance Scale**: Adjust creativity vs. fidelity.
   - **Denoise Strength**: Default to `1.0` for `Text2Image`.
5. **Adjust Advanced Options**:
   - **Mesh Complexity**:
     - `Low`: Depth ControlNet only.
     - `Medium`: Adds Canny ControlNet.
     - `High`: Adds Normalmap ControlNet for maximum detail.
   - **Cameras**: Use more viewpoints for better texture blending.
   - **Texture & Render Resolution**: Ensure render resolution is at least 2x texture resolution.
6. **Generate Texture**:
   - Click `Start Texture Generation`. Monitor progress in the system console.


### Additional Options
- **LoRA Models**: Add one or multiple LoRA models to match specific results.
- **IPAdapter**: Supply the desired "look" as an image instead of a text prompt.

## Troubleshooting
- **Add-On Not Visible**: Ensure it’s enabled in `Edit > Preferences > Add-ons`.
- **Blender Freezes**: Open the system console to track progress during long tasks.
- **Permission Issues**: Specify a valid output path.
- **Out of GPU Memory**:
  - Reduce camera count.
  - Close other GPU-intensive applications.
- **Crashes**: Restart Blender or your PC if crashes persist.

## **Roadmap**
- **Performance Enhancements**:
  - Multi-threaded execution to prevent UI freezing.
- **Checkpoint Flexibility**:
  - Allow external Stable Diffusion checkpoints for varied outputs.
- **Masking Support**:
  - Apply textures to specific areas of the mesh.
- **Multi-Mesh Workflow**:
  - Simultaneously texture multiple objects.

## **Acknowledgements**
- Inspired by [Dream Textures](https://github.com/carson-katri/dream-textures) by [Carson Katri](https://github.com/carson-katri).
- Powered by:
  - [Stable Diffusion](https://arxiv.org/pdf/2112.10752)
  - [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/index)
  - [ControlNet](https://arxiv.org/pdf/2302.05543)
  - [IPAdapter](https://arxiv.org/pdf/2308.06721)
- Influenced by research in [TEXTure](https://arxiv.org/pdf/2302.01721), [Text2Tex](https://arxiv.org/pdf/2303.11396), [Paint3D](https://arxiv.org/pdf/2312.13913), [MatAtlas](https://arxiv.org/pdf/2404.02899) and [EucliDreamer](https://arxiv.org/pdf/2404.10279).