"""README figure layouts. Each function reads workspace renders and writes images/.

Inputs come only from the media workspace (renders, pipeline captures, sweep
frames). Nothing is drawn or mocked up: every object, grid and texture shown
is an actual Cycles render or an image passed to or from the diffusion call.
"""

import json
import re
import shutil
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw
from style import (
    ACCENT,
    BOLD,
    DIVIDER,
    FRAME,
    LIGHT,
    MEDIUM,
    MUTED,
    REGULAR,
    clean_shadow,
    fit,
    font,
    gradient,
    object_box,
    place,
    rounded,
    text_center,
)

HERO = (
    ("lion", "Decorated bronze & gold", "47,214 triangles"),
    ("chest", "Weathered turquoise & brass", "103,330 triangles"),
    ("boulder", "Fractured volcanic basalt", "66,122 triangles"),
)
HERO_FOOTER = "GENERATED TEXTURES  ·  SDXL + DEPTH / CANNY / NORMAL CONTROL"
AXE_CREDIT = "Wooden Axe · Ulan Cabanilla · Poly Haven (CC0)"
BLADE_CROP = (480, 20, 720, 335)  # identical blade inset in every axe panel
RECIPE = Path(__file__).resolve().parents[2] / "documentation" / "media_recipe.json"


@dataclass(frozen=True)
class Workspace:
    """Paths inside a media workspace."""

    root: Path

    @property
    def renders(self) -> Path:
        """Cycles renders, saved scenes and generated textures."""
        return self.root / "renders"

    @property
    def frames(self) -> Path:
        """Camera-sweep frames, one folder per scene."""
        return self.root / "frames"

    def capture(self, job: str, call: int = 0) -> Path:
        """Images passed to and returned from one diffusion call."""
        return self.root / "pipeline_capture" / job / f"{call:02}"

    def result(self, job: str) -> dict:
        """Return texture, seconds and GPU recorded for a job's latest generation.

        Runs from before ``result.json`` existed fall back to the generation
        log and the GPU recorded in documentation/media_recipe.json.
        """
        result = self.renders / f"{job}_gen" / "result.json"
        if result.exists():
            return json.loads(result.read_text())
        for log in (
            self.root / "logs" / f"generate-{job}.log",
            self.root / f"{job}_generate.log",
        ):
            match = log.exists() and re.search(
                rf"GEN_RESULT {job} .*seconds=([\d.]+)",
                log.read_text(),
            )
            if match:
                recipe = json.loads(RECIPE.read_text())
                return {"seconds": float(match[1]), "gpu": recipe["gpu"]}
        return {}

    def texture(self, job: str) -> Path:
        """Return the texture of a job's latest generation."""
        gen = self.renders / f"{job}_gen"
        if "texture" in self.result(job):
            return gen / self.result(job)["texture"]
        textures = sorted(gen.glob("output_texture_*.png"))
        if not textures:
            msg = f"No generated texture for job {job!r} in {gen}"
            raise FileNotFoundError(msg)
        return textures[-1]


def hero_panels(canvas: Image.Image, paste: Callable[[str, tuple], None]) -> None:
    draw = ImageDraw.Draw(canvas)
    for i, (name, title, detail) in enumerate(HERO):
        x = 30 + i * 590
        text_center(canvas, title, 45, font(BOLD, 29), left=x, width=560)
        paste(name, (x + 18, 115, 524, 490))
        text_center(canvas, detail, 640, font(REGULAR, 24), MUTED, x, 560)
        if i:
            draw.line((x - 15, 110, x - 15, 680), fill=DIVIDER, width=2)
    text_center(canvas, HERO_FOOTER, 710, font(REGULAR, 22), ACCENT)


def hero(work: Workspace, out: Path) -> list[Path]:
    canvas = gradient(1800, 760)
    hero_panels(
        canvas,
        lambda name, box: place(canvas, work.renders / f"{name}.png", box),
    )
    canvas.save(out / "hero.webp", quality=90, method=6)

    social = gradient(1280, 640)
    draw = ImageDraw.Draw(social)
    draw.text((44, 30), "DiffusedTexture", font=font(BOLD, 54), fill=LIGHT)
    draw.text(
        (46, 98),
        "AI textures for Blender · SDXL + ControlNet",
        font=font(REGULAR, 26),
        fill=MUTED,
    )
    for i, (name, title, _) in enumerate(HERO):
        x = 20 + i * 420
        place(social, work.renders / f"{name}.png", (x + 10, 160, 380, 380))
        text_center(social, title, 570, font(REGULAR, 22), LIGHT, x, 400)
    social.save(out / "social_preview.png", optimize=True)
    return [out / "hero.webp", out / "social_preview.png"]


def showcase(work: Workspace, out: Path) -> list[Path]:
    """Camera-sweep video of the hero layout; one fixed crop per model."""
    frames = 72
    boxes = {}
    for name, _, _ in HERO:
        sampled = [
            object_box(Image.open(work.frames / name / f"{n:04}.png").convert("RGBA"))
            for n in (0, 18, 36, 54)
        ]
        boxes[name] = (
            max(0, min(b[0] for b in sampled) - 20),
            max(0, min(b[1] for b in sampled) - 20),
            min(720, max(b[2] for b in sampled) + 20),
            min(720, max(b[3] for b in sampled) + 20),
        )
    combined = work.frames / "combined"
    combined.mkdir(exist_ok=True)
    background = gradient(1800, 760)
    for frame in range(frames):
        canvas = background.copy()

        def paste(
            name: str,
            box: tuple,
            canvas: Image.Image = canvas,
            frame: int = frame,
        ) -> None:
            raw = Image.open(work.frames / name / f"{frame:04}.png").convert("RGBA")
            obj = raw.crop(boxes[name])
            clean_shadow(obj)
            obj = fit(obj, box[2], box[3])
            canvas.paste(
                obj,
                (
                    int(box[0] + (box[2] - obj.width) / 2),
                    int(box[1] + (box[3] - obj.height) / 2),
                ),
                obj,
            )

        hero_panels(canvas, paste)
        canvas.save(combined / f"{frame:04}.png")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        msg = "ffmpeg is required to encode showcase.mp4"
        raise RuntimeError(msg)
    target = out / "showcase.mp4"
    subprocess.run(  # noqa: S603
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            "18",
            "-i",
            str(combined / "%04d.png"),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(target),
        ],
        check=True,
    )
    return [target]


def texture_to_texture(work: Workspace, out: Path) -> list[Path]:
    canvas = gradient(1800, 820)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (54, 38),
        "One original texture. Two different finishes.",
        font=font(BOLD, 42),
        fill=LIGHT,
    )
    source = Image.open(work.renders / "boulder_original.png").convert("RGBA")
    box = object_box(source)
    box = (
        max(0, box[0] - 30),
        max(0, box[1] - 30),
        min(source.width, box[2] + 30),
        min(source.height, box[3] + 30),
    )
    for i, (name, title) in enumerate(
        (
            ("boulder_original", "Original Poly Haven albedo"),
            ("boulder", "Glowing lava rock"),
            ("boulder_lichen", "Lichen-covered granite"),
        ),
    ):
        x = 30 + i * 590
        text_center(canvas, title, 120, font(BOLD, 28), LIGHT, x, 560)
        obj = Image.open(work.renders / f"{name}.png").convert("RGBA").crop(box)
        clean_shadow(obj)
        obj = fit(obj, 540, 500)
        canvas.paste(
            obj,
            (int(x + 10 + (540 - obj.width) / 2), int(180 + (500 - obj.height) / 2)),
            obj,
        )
        if i:
            draw.line((x - 15, 120, x - 15, 700), fill=FRAME, width=2)
    text_center(
        canvas,
        "ORIGINAL ALBEDO INPUT   ·   LAVA DENOISE 0.95 / LICHEN 0.85   ·   "
        "SDXL + CONTROLNET",
        754,
        font(REGULAR, 24),
        ACCENT,
    )
    canvas.save(out / "texture-to-texture.webp", quality=92, method=6)
    return [out / "texture-to-texture.webp"]


def controlnet_guidance(work: Workspace, out: Path) -> list[Path]:
    canvas = gradient(1800, 720)
    capture = work.capture("boulder")
    for i, (kind, label) in enumerate(
        (("depth", "Depth"), ("canny", "Canny"), ("normal", "Normals")),
    ):
        x = 30 + i * 590
        text_center(canvas, label, 35, font(BOLD, 32), LIGHT, x, 560)
        grid = Image.open(capture / f"{kind}.png").convert("RGB")
        canvas.paste(grid.resize((520, 520), Image.Resampling.LANCZOS), (x + 20, 100))
    text_center(
        canvas,
        "ACTUAL FOUR-VIEW BOULDER INPUTS TO CONTROLNET UNION",
        660,
        font(REGULAR, 24),
        ACCENT,
    )
    canvas.save(out / "controlnet-guidance.webp", quality=92, method=6)
    return [out / "controlnet-guidance.webp"]


def axe_comparison(
    work: Workspace,
    target: Path,
    title: str,
    stages: tuple[tuple[str, str, str], ...],
    footer: str,
) -> list[Path]:
    """Three reverse-view axe renders, each with the same blade inset."""
    canvas = gradient(1800, 890)
    draw = ImageDraw.Draw(canvas)
    draw.text((54, 38), title, font=font(BOLD, 42), fill=LIGHT)
    source = Image.open(work.renders / f"{stages[0][0]}_reverse.png").convert("RGBA")
    box = object_box(source)
    box = (
        max(0, box[0] - 30),
        max(0, box[1] - 30),
        min(source.width, box[2] + 30),
        min(source.height, box[3] + 30),
    )
    for i, (job, heading, subtitle) in enumerate(stages):
        x = 30 + i * 590
        text_center(canvas, heading, 120, font(BOLD, 28), LIGHT, x, 560)
        text_center(canvas, subtitle, 160, font(REGULAR, 23), ACCENT, x, 560)
        img = Image.open(work.renders / f"{job}_reverse.png").convert("RGBA")
        obj = img.crop(box)
        clean_shadow(obj)
        obj = fit(obj, 185, 500)
        canvas.paste(
            obj,
            (int(x + 10 + (185 - obj.width) / 2), int(180 + (500 - obj.height) / 2)),
            obj,
        )
        detail = img.crop(BLADE_CROP)
        clean_shadow(detail)
        detail = detail.resize((320, 420), Image.Resampling.LANCZOS)
        canvas.paste(detail, (x + 220, 220), detail)
        draw.rectangle((x + 219, 219, x + 540, 640), outline=FRAME, width=1)
        text_center(canvas, "Blade detail", 659, font(REGULAR, 22), MUTED, x + 220, 320)
        if i:
            draw.line((x - 15, 120, x - 15, 700), fill=FRAME, width=2)
    text_center(canvas, footer, 775, font(REGULAR, 24), ACCENT)
    text_center(canvas, AXE_CREDIT, 835, font(REGULAR, 23), MUTED)
    canvas.save(target, quality=92, method=6)
    return [target]


def axe_progressive(work: Workspace, out: Path) -> list[Path]:
    return axe_comparison(
        work,
        out / "axe-progressive.webp",
        "Blood-stained axe · three-pass refinement.",
        (
            (
                "axe_blood_progressive_1",
                "1 · 4 views / 512² texture",
                "Parallel · Denoise 1.00",
            ),
            (
                "axe_blood_progressive_2",
                "2 · 9 views / 1024² texture",
                "Parallel · Denoise 0.50",
            ),
            (
                "axe_blood_progressive_3",
                "3 · 16 views / 2048² texture",
                "Para-Sequential · Denoise 0.20",
            ),
        ),
        "EACH OUTPUT FEEDS THE NEXT PASS   ·   SDXL + CONTROLNET UNION   ·   SEED 157",
    )


def axe_parallel_finish(work: Workspace, out: Path) -> list[Path]:
    return axe_comparison(
        work,
        out / "axe-parallel-finish.webp",
        "A Parallel finish after Para-Sequential.",
        (
            (
                "axe_blood_progressive_2",
                "2 · 9-view Parallel",
                "1024² texture · Denoise 0.50",
            ),
            (
                "axe_blood_progressive_3",
                "3 · 16-view Para-Sequential",
                "2048² texture · Denoise 0.20",
            ),
            (
                "axe_blood_progressive_4",
                "4 · 16-view Parallel",
                "2048² texture · Denoise 0.20",
            ),
        ),
        "PASS 4 INPUT = PASS 3 TEXTURE   ·   16 VIEWS IN ONE SDXL GRID   ·   SEED 157",
    )


def best_square_view(mask_path: Path) -> tuple[int, int, int, int]:
    """Pick the 2x2-grid view whose object fills a square crop best."""
    mask = Image.open(mask_path)
    half = mask.width // 2
    best: tuple[float, tuple[int, int, int, int]] | None = None
    for qy in (0, 1):
        for qx in (0, 1):
            quad = (qx * half, qy * half, (qx + 1) * half, (qy + 1) * half)
            box = mask.crop(quad).getbbox()
            if box is None:
                continue
            x0, y0, x1, y1 = box
            side = min(half, int(max(x1 - x0, y1 - y0) * 1.08))
            left = min(max((x0 + x1) // 2 - side // 2, 0), half - side) + quad[0]
            top = min(max((y0 + y1) // 2 - side // 2, 0), half - side) + quad[1]
            fill = (x1 - x0) * (y1 - y0) / side**2
            if best is None or fill > best[0]:
                best = (fill, (left, top, left + side, top + side))
    if best is None:
        msg = f"Empty generation mask: {mask_path}"
        raise ValueError(msg)
    return best[1]


def how_it_works(work: Workspace, out: Path) -> list[Path]:
    """Five pipeline stages, using the lava boulder's actual intermediates."""
    width, panel, gap, top = 1800, 300, 50, 190
    canvas = gradient(width, 600)
    draw = ImageDraw.Draw(canvas)
    draw.text((54, 38), "How it works.", font=font(BOLD, 42), fill=LIGHT)
    left0 = (width - (5 * panel + 4 * gap)) // 2
    stages = (
        ("1 · Your mesh", "UV map + optional texture"),
        ("2 · Render guidance", "per view: color, depth, canny, normal"),
        ("3 · Diffuse", "all views in one SDXL grid"),
        ("4 · Project to UV", "blended by facing angle"),
        ("5 · Result", "texture applied as a material"),
    )
    for i, (heading, subtitle) in enumerate(stages):
        x = left0 + i * (panel + gap)
        text_center(canvas, heading, top - 72, font(BOLD, 26), LIGHT, x, panel)
        text_center(canvas, subtitle, top - 36, font(REGULAR, 18), ACCENT, x, panel)
        if i:
            ax, ay = x - gap // 2 + 6, top + panel // 2
            draw.line(
                (ax - 12, ay - 12, ax + 2, ay, ax - 12, ay + 12),
                fill=MUTED,
                width=4,
                joint="curve",
            )

    def at(i: int) -> int:
        return left0 + i * (panel + gap)

    for i, name in ((0, "boulder_original"), (4, "boulder")):
        place(canvas, work.renders / f"{name}.png", (at(i), top, panel, panel))

    capture = work.capture("boulder")
    crop = best_square_view(capture / "mask.png")
    tile = (panel - 6) // 2
    label_font = font(MEDIUM, 15)
    labels = ImageDraw.Draw(canvas, "RGBA")
    for k, (kind, label) in enumerate(
        (
            ("input", "color"),
            ("depth", "depth"),
            ("canny", "canny"),
            ("normal", "normal"),
        ),
    ):
        img = Image.open(capture / f"{kind}.png").convert("RGB").crop(crop)
        img = rounded(img.resize((tile, tile), Image.Resampling.LANCZOS), 10)
        tx, ty = at(1) + (k % 2) * (tile + 6), top + (k // 2) * (tile + 6)
        canvas.paste(img, (tx, ty), img)
        lw = labels.textlength(label, font=label_font)
        lx, ly = tx + 7, ty + tile - 29
        labels.rounded_rectangle(
            (lx, ly, lx + lw + 14, ly + 22),
            6,
            fill=(20, 21, 24, 200),
        )
        labels.text((lx + 7, ly + 3), label, font=label_font, fill=LIGHT)

    for i, path in ((2, capture / "output.png"), (3, work.texture("boulder"))):
        img = (
            Image.open(path)
            .convert("RGB")
            .resize((panel, panel), Image.Resampling.LANCZOS)
        )
        img = rounded(img)
        canvas.paste(img, (at(i), top), img)

    footer = (
        "REAL INTERMEDIATES FROM THE LAVA BOULDER ABOVE  ·  4 VIEWS  ·  "
        "SDXL + CONTROLNET UNION"
    )
    timing = work.result("boulder")
    if "seconds" in timing:
        gpu = re.sub(r"^NVIDIA (GeForce )?| \d+ GB$", "", timing["gpu"]).upper()
        footer += f"  ·  IN {timing['seconds']:.0f} SECONDS ON AN {gpu}"
    text_center(canvas, footer, top + panel + 46, font(REGULAR, 20), ACCENT)
    text_center(
        canvas,
        "Boulder 01 · Rico Cilliers · Poly Haven (CC0)",
        top + panel + 82,
        font(REGULAR, 19),
        MUTED,
    )
    canvas.save(out / "how-it-works.webp", quality=88, method=6)
    return [out / "how-it-works.webp"]


# figure -> (layout, workspace files it needs, relative to the workspace root)
FIGURES: dict[str, tuple[Callable[[Workspace, Path], list[Path]], tuple[str, ...]]] = {
    "hero": (hero, ("renders/lion.png", "renders/chest.png", "renders/boulder.png")),
    "showcase": (
        showcase,
        ("frames/lion/0071.png", "frames/chest/0071.png", "frames/boulder/0071.png"),
    ),
    "texture-to-texture": (
        texture_to_texture,
        (
            "renders/boulder_original.png",
            "renders/boulder.png",
            "renders/boulder_lichen.png",
        ),
    ),
    "controlnet-guidance": (
        controlnet_guidance,
        ("pipeline_capture/boulder/00/normal.png",),
    ),
    "axe-progressive": (
        axe_progressive,
        tuple(f"renders/axe_blood_progressive_{i}_reverse.png" for i in (1, 2, 3)),
    ),
    "axe-parallel-finish": (
        axe_parallel_finish,
        tuple(f"renders/axe_blood_progressive_{i}_reverse.png" for i in (2, 3, 4)),
    ),
    "how-it-works": (
        how_it_works,
        (
            "renders/boulder_original.png",
            "renders/boulder.png",
            "pipeline_capture/boulder/00/output.png",
        ),
    ),
}
