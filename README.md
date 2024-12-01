# DiffusedTexture: AI-Powered Texture Generation for Blender

DiffusedTexture is a Blender add-on that uses Stable Diffusion to create textures directly on 3D meshes. 

![Before](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/elephant_before.gif) ![After](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/elephant_after.gif)

## Table of Contents
- [DiffusedTexture: AI-Powered Texture Generation for Blender](#diffusedtexture-ai-powered-texture-generation-for-blender)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation (Windows)](#installation-windows)
  - [Installation (Linux)](#installation-linux)
  - [Setup](#setup)
    - [Blender Setup](#blender-setup)
  - [Usage](#usage)
    - [Main Workflow](#main-workflow)
    - [Additional Options](#additional-options)
  - [Troubleshooting](#troubleshooting)
  - [TODOs](#todos)
  - [Acknowledgement](#acknowledgement)

## Features
- **Direct Texture Generation:** Diffuse Textures are generated directly on the 3D model within Blender, enabling WYSIWYG (what you see is what you get) results.
- **Modes for Different Workflows:**
  - **Text2Image Parallel**: Generates textures from text prompts.
  - **Image2Image Parallel**: Generates textures from input images, applied parallel across all views.
  - **Image2Image Sequential**: Sequentially applies textures across views, great for refinement.
- **LoRA Integration**: Uses LoRA conditioning for specific styles.
- **IPAdapter Integration**: Fit specific styles or objects with images for enhanced flexibility and control.

## Installation (Windows)
0. Download [7-Zip](https://7-zip.de/download.html) 
1. Download all .tar files of the [latest release](https://github.com/FrederikHasecke/diffused-texture-addon/releases/latest)
2. Untar the file `diffused_texture_addon.7z.001` this will automatically untar the other `.7z` files
    >[!WARNING] WARNING
    DO NOT unzip the resulting `diffused_texture_addon.zip`
3.  Install the `diffused_texture_addon.zip` file in Blender as an Add-On.

## Installation (Linux)
- TODO: Test on Linux Instructions and test if it works

## Setup
### Blender Setup
Ensure your Blender preferences are configured for CUDA (for GPU rendering). If you’re using an NVIDIA GPU, enable cycles and set up with either CUDA or OPTIX.

When you first install the Add-On you will need to download the required Stable Diffusion, ControlNet and IPAdapter Models. This will take a considerable time (10.6 GB in total). If you want to download them to a different drive than `C://`, change the `HuggingFace Cache Path` in the Add-On installation window.

> [!TIP] Open The Console
Open up the terminal (Window -> Toggle System Console) before you press "Install Models" so you can see the progress, the Blender UI will freeze in the meantime.

## Usage

![General Usage]([http://url/to/img.png](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/usage.gif))

### Main Workflow
1. **Open a 3D Model**: Open the `.blend` file containing the 3D model to texture (or create one from scratch).
2. **UV Unwrap the Model**: The add-on requires a UV Map, 
      > [!TIP] Tipp
      `Smart UV Project` will work fine.
3. **Open the DiffusedTexture Panel**: The add-on is in the right-hand panel (n-panel) under `DiffusedTexture`.
4. **Select your Mesh and UV Map**: You need to provide the targets to the add-on.
5. **Set the Stable Diffusion Options**: Prompt, Negative Prompt, Guidance Scale and Denoise (fixed to 1.0 for `Text2Image Parallel`)
6. **Set DiffusedTexture Options**: Special settings:
   - **Operation Model**:
      - `Text2Image Parallel`
      Global consistent texture from your text prompt. This operation mode does not use an input texture, even if it is supplied to the process. Depending on the provided
      - `Image2Image Parallel`
      For global adjustments and improvements to `Text2Image Parallel` textures. This operation mode needs an input texture. Depending on the provided
      - `Image2Image Sequential`
      This operation mode needs an input texture. Depending on the provided
   - **Mesh Complexity**: Adjusts ControlNets based on object polycount.
      -  `Low Complexity`: Depth ControlNet Only
      -  `Mid Complexity`: Depth and Canny ControlNets
      -  `High Complexity`: Depth, Canny and Normalmap ControlNets
   -  **Cameras**: Number of Cameras used to create the texture.
      -  Rule of thumb: The more viewpoints, the better. 
   -  **Texture Resolution**: Size of the resulting Texture.
   -  **Render Resolution**: This is not the image size given to Stable Diffusion, but used for the texture projection. Keep the Render Resolution at least to 2x the Texture Resolution, else the texture will have artifacts.
   -  **Output Path**: Generated textures will be saved in the specified output path. 
   -  **Input Texture**: The modes `Image2Image Parallel` and `Image2Image Sequential` require an input texture to work.
7. Press the `Start Texture Generation` Button
      > [!TIP] Open The Console
      Open up the terminal (Window -> Toggle System Console) before you press `Start Texture Generation` so you can see the progress, the Blender UI will freeze in the meantime.

### Additional Options
- **LoRA Models**: Add one or multiple LoRA models to match specific results.
- **IPAdapter**: Supply the desired "look" as an image instead of a text prompt.

## Troubleshooting
- **Freezes**: Open up the Terminal before executing the Addon to see the progress bar.
- **Add-On Not Showing Up**: Ensure the add-on is enabled in Blender's preferences.
- **CUDA/OPTIX Issues**: Verify GPU support is enabled in Blender and that the correct drivers are installed.
- **Slow Rendering**: Higher resolutions and camera counts can increase memory usage and render times.
- **RuntimeError: Error: Cannot open file [...]: Permission denied**: Create a new Folder and select that one.
- **torch.OutOfMemoryError: CUDA out of memory.**: Choose less cameras in the parallel tasks and close all other processes that might use GPU Memory.

## TODOs
- **Threading/Timing**: Remove the process from the main thread to not freeze Blender.
- **Checkpoints**: Add an option to use external checkpoints finetuned for specific looks.
- **Masking**: Add an option to only apply texture changes to specific parts of the mesh.
- **Multi-Mesh**: Add an option to apply textures in one go to multiple objects with multiple UV Maps at once.

## Acknowledgement
I'd like to thank [carson-katri](https://github.com/carson-katri) and all other contributors to [dream-textures](https://github.com/carson-katri/dream-textures). I took inspiration from the project to create this this one.

Then of course [Stable Diffusion](https://arxiv.org/pdf/2112.10752), [HuggingFace's Diffusers](https://huggingface.co/docs/diffusers/index), [IPAdapter](https://arxiv.org/pdf/2308.06721) and [ControlNet](https://arxiv.org/pdf/2302.05543) which are the main parts of this repo.

Furthermore the following papers which influenced the creation of this add-on:
[TEXTure](https://arxiv.org/pdf/2302.01721), [Text2Tex](https://arxiv.org/pdf/2303.11396), [Paint3D](https://arxiv.org/pdf/2312.13913), [MatAtlas](https://arxiv.org/pdf/2404.02899) and [EucliDreamer](https://arxiv.org/pdf/2404.10279).