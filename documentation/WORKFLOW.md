# Recommended Workflow

These steps work with both supported models, SD 1.5 and SDXL. Each pass feeds its output texture back in as the **Input Texture** for the next.

## Prompting tips
- Describe the surface, not just the object: colors, materials, wear and small details. For example, *antique bronze bust, dark aged bronze with turquoise verdigris patina in the crevices, worn golden bronze on the nose* instead of *bronze bust*.
- Match the prompt to what the mesh can show. A smooth statue won't turn into convincing fur; ask for paint, glaze or patina instead.
- Use the negative prompt against baked-in lighting, for example *harsh shadows, strong highlights, reflections*. The texture should hold color, not lighting.
- Raise **Mesh Complexity** and the ControlNet strengths for detailed meshes such as scans and faces, so details land on the geometry. Lower them for simple shapes, where the model can be more creative.

## 1. First pass: global consistency
- **Operation Mode** `Parallel`, 4 or 9 **Camera Views**, **Denoise Strength** `1.0`, no input texture.
- Generate a few variations with different seeds and keep the best one.
- If the prompt alone doesn't get the look you want, add a reference image in the **IPAdapter** panel or a style LoRA.

## 2. Second pass: more coverage
- Set the first result as the **Input Texture**.
- Stay on `Parallel` and raise **Camera Views** to cover surfaces the first pass missed.
- Lower **Denoise Strength** (around `0.4`–`0.6`) to keep the overall look while sharpening details and fixing small inconsistencies. Around `0.4` usually works well.
- Raise **Mesh Complexity** if details drift away from the geometry; lower it for more creative results.

## 3. Third pass: local refinement
- Switch to `Sequential` or `Para-Sequential` with a low denoise strength.
- Each view (or group of views) now refines the texture the previous one left behind.
- Repeat until you are happy with the result.
