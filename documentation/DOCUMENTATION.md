# How Does It Work?

## Overview
DiffusedTexture generates textures for 3D models by leveraging Stable Diffusion with ControlNets. The process involves generating viewpoints from multiple cameras, creating control images, and blending results into a cohesive texture. The workflow adapts to different modes: Text2Image, Image2Image Parallel, and Image2Image Sequential.

---

## Cameras
The number of cameras determines the viewpoints used to generate the texture. These viewpoints ensure adequate coverage of the model's surface, capturing details from various angles.

### **4 Cameras:** Minimal coverage for quick texture generation; best for simple models or initial textures which are further refined.
![4 Cameras](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/process/cameras_4.png)

### **9 Cameras:** Balanced detail and coverage; suitable for moderately complex models.
![9 Cameras](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/process/cameras_9.png)

### **16 Cameras:** High detail and coverage; ideal for highly detailed models, especially in a second or third pass of an already textures object.
![16 Cameras](https://github.com/FrederikHasecke/diffused-texture-addon/blob/master/images/process/cameras_16.png)

---

## Processes

### **Parallel Processing on Images**
This mode generates textures by running Stable Diffusion on multiple viewpoints in parallel.

#### Steps:
1. **Multiple Viewpoints**:
   - The camera viewpoints are laid out in a grid (e.g., 2x2, 3x3, or 4x4).
2. **Grid Input**:
   - The input to Stable Diffusion is either a supplied texture or a fully white image, when generating a texture from text only.
3. **Control Images**:
   - ControlNets use depth, canny (derived from normals), and surface normal images in the same grid structure to guide the texture generation.
4. **Resolution Setup**:
   - *Render Resolution*: The resolution used for camera views and projections. Should be at least 2x the Texture Resolution to avoid striping artifacts.
   - *Texture Resolution*: Final resolution of the generated texture.
   - *Stable Diffusion Resolution*: If you want more or less resolution in the Stable Diffusion process, you can adjust this setting. Can easily lead to out of memory errors.
5. **Projection**:
   - The texture contributions are weighted by the perpendicularity of the surface to the corresponing camera, ensuring smooth blending between viewpoints.

---

### **Sequential Processing on Images**
This mode processes each viewpoint individually, refining the texture step-by-step for greater control over the final result.

#### Steps:
1. **Single View at a Time**:
   - Instead of processing all viewpoints in parallel, this mode handles one camera view at a time.
2. **Input Masking**:
   - Only faces close to perpendicular to the camera are provided as a mask to Stable Diffusion.
3. **Reprojection**:
   - After each view is processed, the resulting texture is reprojected onto the model, contributing to the next iteration.

#### Advantages:
- Greater control over texture alignment and details.
- Allows finer adjustments compared to parallel modes.

#### Disadvantages:
- Slower than parallel processing.
- Requires more manual intervention.
- May lead to visible seams if not handled carefully.

---

## Parameter and Considerations

### Cameras
- **Impact**: The number of cameras affects texture quality but also the runtime.
- **Recommendation**: Use more cameras for complex models, but balance against VRAM limitations.

### Resolutions
- **Render Resolution**: Should be at least 2x the Texture Resolution to reduce artifacts.
- **Texture Resolution**: Determines the final quality of the generated texture.

### ControlNets
- Control images (depth, canny, normal) guide the texture generation process.
- **Complexity Settings**:
  - *Low*: Depth ControlNet only (creative but less accurate).
  - *Medium*: Depth and Canny ControlNets (balanced results).
  - *High*: Depth, Canny, and Normal ControlNets (precise but less creative).

