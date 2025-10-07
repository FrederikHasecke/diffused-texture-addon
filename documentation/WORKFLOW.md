# Recommended Workflow

## First Pass: Global Consistency
- Use the `Parallel Processing on Images` mode with 4 or 9 cameras to create one or more variations of what you want to achieve as a texture.
- If your description does not yield the results that you want to reach, use the provided IPAdapter and supply an image of what you want.
  
## Second Pass: Local Refinement
- Use the `Parallel Processing on Images` mode to refine the texture with more cameras.
- Use the `ControlNets` to guide the texture generation process, more complex settings can be used for better results if the model is complex, if you want more creative results, use the `Low` setting.
- Repeat the process on the resulting texture to refine it further if needed.

## Third Pass: Fine-Tuning
- Use the `Sequential Processing on Images` mode to fine-tune the texture. This mode allows each viewpoint to be processed individually, refining the texture step-by-step for greater control over the final result.
- Repetition of the process on the resulting texture to refine it further if needed.

