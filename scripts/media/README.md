# README media

Every image and video in [`images/`](../../images) that shows a generated texture
is rebuilt by `media.py` from the definitions in this folder. The renders,
textures and diffusion captures they are laid out from live in a separate
**media workspace** outside git. Only the finished figures are committed.

| Figure | Built from |
| --- | --- |
| `social_preview.png` | renders of the `lion`, `chest` and `boulder` jobs |
| `showcase.mp4` | 72-frame camera sweeps of the same three jobs |
| `hero.webp` | looping animated preview of `showcase.mp4`, displayed inline in the README |
| `texture-to-texture.webp` | `boulder_original`, `boulder` and `boulder_lichen` renders |
| `controlnet-guidance.webp` | depth / canny / normal grids captured from the `boulder` job |
| `how-it-works.webp` | the `boulder` job's captured inputs, SDXL output and UV texture |
| `axe-progressive.webp`, `axe-parallel-finish.webp` | reverse-view renders of `axe_blood_progressive_1` to `_4` |

`usage.webp`/`usage.mp4` (a screen recording), `preferences.png` (a UI
screenshot) and the older elephant/rabbit examples are not generated here.

## Quick start

```bash
export DT_MEDIA_WORKDIR=/path/to/media-workspace
python3 scripts/media/media.py check
python3 scripts/media/media.py all
python3 scripts/media/media.py verify
```

Each stage can run alone and be limited to some names, for example
`media.py generate boulder boulder_lichen` or `media.py compose hero`.

`compose hero` rebuilds the animated hero from the committed `images/showcase.mp4`
without needing the original renders. A full `compose` rebuilds the sweep before
encoding the hero. `compose social-preview` rebuilds the separate static social
preview from the three hero renders.

| Stage | Names | What it does | Needs |
| --- | --- | --- | --- |
| `check` | — | Reports Blender, the installed add-on and its commit, the HDRI checksum, fonts and ffmpeg | — |
| `fetch` | asset ids | Downloads the CC0 Poly Haven blends and textures listed in `jobs.json`, md5-checked. Existing matching files are skipped | network |
| `prepare` | scenes | Joins each scene's mesh into `assets/<scene>_prepared.blend` and renders `<scene>_original.png` (original albedo) and `<scene>_clay.png` | Blender, GPU |
| `generate` | jobs | Runs the add-on's normal operator for each job, capturing every diffusion call, then renders `<job>.png` and saves `<job>.blend`. Chained jobs run after the job they refine | Blender, GPU, add-on with dependencies, models |
| `views` | jobs | Renders extra stills (`<job>_reverse.png`) and camera-sweep frames from `<job>.blend` | Blender, GPU |
| `compose` | figures | Lays out the figures into `images/`, skipping any whose inputs are missing | Pillow, Ubuntu fonts, ffmpeg |
| `verify` | — | Compares `images/` with the checksums recorded in `documentation/*_recipe.json` | — |

`compose` and `verify` take seconds and need no GPU, so layout changes can be
made against an existing workspace without regenerating anything.

## Environment

The Blender stages run `blender -b --online-mode <blend> --python
blender/<stage>.py -- <spec.json>`. `media.py` writes each resolved spec to
`<workspace>/specs/` and each Blender log to `<workspace>/logs/`.

- `--blender` or `$BLENDER` selects the Blender executable (default
  `blender`). The recorded run used Blender 5.2.0 LTS.
- The add-on must be installed as `bl_ext.user_default.diffused_texture_addon`
  in `$BLENDER_USER_EXTENSIONS`, with its Python dependencies installed. Install
  it from the commit you want to show. `check` prints the installed commit.
- Set `HF_HOME` to the Hugging Face cache holding SDXL and ControlNet Union.
- Cycles renders on OptiX, as in the recorded run. On some NVIDIA machines with
  an Intel GPU, `ONEAPI_DEVICE_SELECTOR='opencl:*'` avoids a driver crash
  during device discovery.
- `--hdri` selects the studio HDRI (default `<workspace>/assets/studio.hdr`).
  `check` compares it with the checksum in `jobs.json`. Its original source
  was not recorded. Any studio HDRI works, but renders will not match the
  committed images.

## Workspace layout

```text
assets/<asset id>/          downloaded Poly Haven blend and textures
assets/<scene>_prepared.blend
renders/<scene>_original.png, <scene>_clay.png
renders/<job>.png, <job>.blend, <job>_reverse.png
renders/<job>_gen/          add-on output, including output_texture_*.png and result.json
pipeline_capture/<job>/NN/  input, mask, depth, normal, canny, output and call.json per diffusion call
frames/<scene>/NNNN.png     camera-sweep frames; frames/combined/ holds the composed video frames
specs/, logs/               resolved Blender specs and logs
```

## Changing the media

- **Settings, prompts, cameras:** edit `jobs.json`. `defaults` applies to
  every job; `scenes` holds each asset's camera; `input_from_job` chains a job
  onto the latest texture of another; `views` and `sweep` request extra renders.
- **Layout:** edit `figures.py`. Shared colors, fonts and cut-out helpers are in
  `style.py`. Every figure is built only from actual renders and captures.
- After an intentional change, update the checksums and settings in the
  matching `documentation/*_recipe.json`, so `verify` passes again and the
  README's settings tables stay accurate.

On the same machine, Blender version and add-on commit, a re-run of
`axe_blood_progressive_1` reproduced the recorded captures, SDXL output and UV
texture byte for byte; its Cycles renders differed only by denoiser noise
(at most 9/255 per channel). Other GPUs, drivers or library versions can change
the diffusion result.
