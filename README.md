# DiffusedTexture: AI-Powered Texture Generation for Blender
[![Latest Release](https://flat.badgen.net/github/release/FrederikHasecke/diffused-texture-addon/latest)](https://github.com/FrederikHasecke/diffused-texture-addon/releases/latest)
[![Total Downloads](https://img.shields.io/github/downloads/FrederikHasecke/diffused-texture-addon/total?style=flat-square)](https://github.com/FrederikHasecke/diffused-texture-addon/releases/latest)

[![Python 3.11](https://img.shields.io/badge/python-3.11-orange.svg)](https://www.python.org/downloads/release/python-311/)
[![Blender 4.2](https://img.shields.io/badge/Blender-4.2-blue?style=flat-square)](https://download.blender.org/release/Blender4.2/)
[![Blender 4.3](https://img.shields.io/badge/Blender-4.3-blue?style=flat-square)](https://download.blender.org/release/Blender4.3/)
[![Blender 4.4](https://img.shields.io/badge/Blender-4.4-blue?style=flat-square)](https://download.blender.org/release/Blender4.4/)
[![Blender 4.5](https://img.shields.io/badge/Blender-4.5-blue?style=flat-square)](https://download.blender.org/release/Blender4.5/)
[![Blender 5.0](https://img.shields.io/badge/Blender-5.0-blue?style=flat-square)](https://download.blender.org/release/Blender5.0/)

[![Python 3.13](https://img.shields.io/badge/python-3.13-orange.svg)](https://www.python.org/downloads/release/python-313/)
[![Blender 5.1+](https://img.shields.io/badge/Blender-5.1%2B-blue?style=flat-square)](https://www.blender.org/download/)

[![Lint](https://github.com/FrederikHasecke/diffused-texture-addon/actions/workflows/lint.yml/badge.svg?branch=master)](https://github.com/FrederikHasecke/diffused-texture-addon/actions/workflows/lint.yml)
[![Release](https://github.com/FrederikHasecke/diffused-texture-addon/actions/workflows/release-please.yml/badge.svg?branch=master)](https://github.com/FrederikHasecke/diffused-texture-addon/actions/workflows/release-please.yml)


DiffusedTexture is a Blender add-on that uses Stable Diffusion to create textures directly on 3D meshes.

![General Usage](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/usage.gif)

## Table of Contents
- [DiffusedTexture: AI-Powered Texture Generation for Blender](#diffusedtexture-ai-powered-texture-generation-for-blender)
  - [Table of Contents](#table-of-contents)
  - [Examples](#examples)
  - [Features](#features)
  - [Installation](#installation)
  - [Platform Runtime Notes](#platform-runtime-notes)
  - [Usage](#usage)
    - [Additional Options](#additional-options)
  - [Troubleshooting](#troubleshooting)
  - [Development](#development)
  - [**Acknowledgements**](#acknowledgements)

## Examples
![https://www.cgtrader.com/free-3d-print-models/miniatures/other/elephant-natural-history-museum-1](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/elephant.gif)
![https://graphics.stanford.edu/data/3Dscanrep/](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/rabbit.gif)


## Features
- **AI-Driven Texture Creation:** Generate diffuse textures directly on 3D models
- **Modes for Different Workflows:**
  - **Text2Image Parallel**: Create textures from text prompts, ensuring global consistency.
  - **Image2Image Parallel**: Generates textures from input textures, applied parallel across all views.
  - **Image2Image Sequential**: Sequentially adjusts textures across views, great for refinement.
- **LoRA Integration**: Uses LoRA conditioning for specific styles.
- **IPAdapter Integration**: Fit specific styles or objects with images for enhanced flexibility and control.

## Installation 

1. Download the .zip file of the [latest release](https://github.com/FrederikHasecke/diffused-texture-addon/releases/latest)
2. Install the downloaded `diffused_texture_addon-<version>.zip` file in Blender as an Add-On.
    -  `Edit` -> `Preferences...` -> Sidebar `Add-ons` -> Top right corner dropdown menu -> `Install from Disk...`
3. Install the dependencies by clicking the `Install Dependencies` button in the Add-On panel.
    - This will take a while as it installs all necessary packages, including PyTorch and diffusers.
    - The `Dependency backend` setting chooses which PyTorch wheels to install for the diffusion stack. It does not by itself guarantee that Blender can use the same backend for the Cycles render pass.
    ![Installation](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/install.png)

4. **RESTART BLENDER!** This is important to ensure all packages are correctly loaded.
5. (Optional) Download the necessary models by clicking the `Download Models` button in the Add-On panel. 
    - If wanted, provide a custom `HuggingFace Cache Path` to install and/or load the checkpoints, else the default path is choosen.
    - Download necessary models (~10.6 GB total)
    - You can also run the add-on without downloading the models, but it will take longer to generate textures as they will be downloaded on-the-fly.
    - The models will be stored in the specified `HuggingFace Cache Path` or the default cache path.


## Platform Runtime Notes

- DiffusedTexture now reports two separate platform decisions:
  - **Dependency backend**: which PyTorch wheels are installed for diffusion (`cpu`, CUDA, ROCm, or auto-detect).
  - **Runtime capability**: which Cycles render backend Blender can actually use for the render pass, plus which diffusion device PyTorch/diffusers can use at runtime.
- For Cycles renders, the add-on prefers `OPTIX`, `CUDA`, `HIP`, `ONEAPI`, and `METAL`, then falls back to `CPU` when no GPU backend is usable.
- This means a machine can install a non-NVIDIA diffusion backend and still render through Cycles on CPU if that is the only render path Blender exposes locally.


## Usage

1. **Load or create a 3D Model**:
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
- **SDXL**: Use the more advanced SDXL model for higher quality textures.

## Troubleshooting
- **Add-On Not Visible**: Ensure it’s enabled in `Edit > Preferences > Add-ons`.
- **Blender Freezes**: Open the system console to track progress during long tasks.
- **Permission Issues**: Specify a valid output path.
- **Out of GPU Memory**:
  - Reduce camera count.
  - Close other GPU-intensive applications.
- **Platform Confusion**: Check the `Current Runtime Capability` box in the add-on preferences. It shows the installed diffusion backend, the active Cycles render path, and whether texture generation is currently ready.
- **Crashes**: Restart Blender or your PC if crashes persist.

## Development

- Default local dev environment: Python `3.13` with Blender `5.1+`, installed via `uv sync`.
- The local `uv` configuration overrides the published `bpy` NumPy cap so the dev environment aligns with the CY2026 NumPy `2.3.x` lane.
- Legacy Blender `<5.1` / Python `3.11` compatibility is still covered by the installer matrix tests.
- Commit and hook setup: [CONTRIBUTING.md](CONTRIBUTING.md)
- Testing workflow and local check commands: [documentation/TESTING.md](documentation/TESTING.md)
- Release automation and bootstrap instructions: [documentation/RELEASING.md](documentation/RELEASING.md)


## **Acknowledgements**
- Inspired by [Dream Textures](https://github.com/carson-katri/dream-textures) by [Carson Katri](https://github.com/carson-katri).
- Powered by:
  - [Stable Diffusion](https://arxiv.org/pdf/2112.10752)
  - [HuggingFace Diffusers](https://huggingface.co/docs/diffusers/index)
  - [ControlNet](https://arxiv.org/pdf/2302.05543)
  - [IPAdapter](https://arxiv.org/pdf/2308.06721)
- Influenced by research in [TEXTure](https://arxiv.org/pdf/2302.01721), [Text2Tex](https://arxiv.org/pdf/2303.11396), [Paint3D](https://arxiv.org/pdf/2312.13913), [MatAtlas](https://arxiv.org/pdf/2404.02899) and [EucliDreamer](https://arxiv.org/pdf/2404.10279).
