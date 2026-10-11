"""Shared look for the README figures: background, fonts, colors and cut-outs.

Changing anything here changes every figure. ``media.py verify`` compares the
results with the checksums recorded in ``documentation/*_recipe.json``.
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path("/usr/share/fonts/truetype/ubuntu")
REGULAR = str(FONT_DIR / "Ubuntu-R.ttf")
MEDIUM = str(FONT_DIR / "Ubuntu-M.ttf")
BOLD = str(FONT_DIR / "Ubuntu-B.ttf")
FONTS = (REGULAR, MEDIUM, BOLD)

LIGHT = (236, 239, 243)
MUTED = (172, 183, 194)
ACCENT = (142, 215, 190)
DIVIDER = (66, 72, 80)
FRAME = (74, 80, 88)
BG_TOP = (52, 54, 60)
BG_BOTTOM = (28, 29, 33)

OPAQUE = 220  # alpha above which a render pixel belongs to the object
FLOOR = 40  # alpha below which the shadow catcher is dropped


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def gradient(width: int, height: int) -> Image.Image:
    """Vertical background gradient, truncated per row like the original loop."""
    t = np.arange(height, dtype=np.float64) / max(height - 1, 1)
    rows = [
        np.trunc(top + (bottom - top) * t)
        for top, bottom in zip(BG_TOP, BG_BOTTOM, strict=True)
    ]
    column = np.stack(rows, axis=-1).astype(np.uint8)
    return Image.fromarray(np.repeat(column[:, None, :], width, axis=1), "RGB")


def object_box(img: Image.Image) -> tuple[int, int, int, int]:
    box = img.getchannel("A").point(lambda a: 255 if a > OPAQUE else 0).getbbox()
    if box is None:
        msg = "Render has no opaque object pixels."
        raise ValueError(msg)
    return box


def trimmed(path: Path) -> Image.Image:
    """Crop a transparent Cycles render to its object and nearby shadow."""
    img = Image.open(path).convert("RGBA")
    # Drop the faint shadow-catcher floor so no rectangle shows; keep real shadows.
    alpha = img.getchannel("A").point(
        lambda a: 0 if a < FLOOR else int((a - FLOOR) * 255 / (255 - FLOOR)),
    )
    img.putalpha(alpha)
    obj_box = alpha.point(lambda a: 255 if a > OPAQUE else 0).getbbox()
    shadow_box = alpha.point(lambda a: 255 if a > 10 else 0).getbbox()  # noqa: PLR2004
    if not obj_box or not shadow_box:
        return img
    # Object bounds decide the size; allow a little shadow below and to the sides.
    pad = int(0.06 * (obj_box[3] - obj_box[1]))
    img = img.crop(
        (
            max(obj_box[0] - pad, shadow_box[0]),
            obj_box[1],
            min(obj_box[2] + pad, shadow_box[2]),
            min(obj_box[3] + pad, shadow_box[3]),
        ),
    )
    # Feather the crop edges so cut-off shadows fade out instead of ending hard.
    a = np.asarray(img.getchannel("A")).astype(np.float32)
    h, w = a.shape
    feather = max(8, int(0.08 * min(w, h)))
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum.reduce([xx, w - 1 - xx, h - 1 - yy]).astype(np.float32)
    ramp = np.clip(dist / feather, 0, 1) * 255
    # Keep only shadows close to the object: fade alpha with distance from it.
    obj_mask = Image.fromarray(((a > OPAQUE) * 255).astype(np.uint8))
    radius = max(6, int(0.05 * min(w, h)))
    envelope = np.asarray(obj_mask.filter(ImageFilter.GaussianBlur(radius))).astype(
        np.float32,
    )
    envelope = np.clip(envelope * 4, 0, 255)
    faded = np.where(a > OPAQUE, a, np.minimum(np.minimum(a, ramp), envelope))
    img.putalpha(Image.fromarray(faded.astype(np.uint8)))
    return img


def clean_shadow(img: Image.Image) -> None:
    """Fade the shadow of an already cropped render in place."""
    alpha = np.asarray(img.getchannel("A")).astype(np.float32)
    alpha = np.clip((alpha - FLOOR) * 255 / (255 - FLOOR), 0, 255)
    h, w = alpha.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum.reduce([xx, w - 1 - xx, h - 1 - yy]).astype(np.float32)
    ramp = np.clip(dist / max(8, 0.08 * min(w, h)), 0, 1) * 255
    mask = Image.fromarray(((alpha > OPAQUE) * 255).astype(np.uint8))
    envelope = np.asarray(
        mask.filter(ImageFilter.GaussianBlur(max(6, 0.05 * min(w, h)))),
    )
    envelope = np.clip(envelope.astype(np.float32) * 4, 0, 255)
    alpha = np.where(
        alpha > OPAQUE,
        alpha,
        np.minimum(np.minimum(alpha, ramp), envelope),
    )
    img.putalpha(Image.fromarray(alpha.astype(np.uint8)))


def fit(img: Image.Image, width: int, height: int) -> Image.Image:
    scale = min(width / img.width, height / img.height)
    size = (int(img.width * scale), int(img.height * scale))
    return img.resize(size, Image.Resampling.LANCZOS)


def place(canvas: Image.Image, path: Path, box: tuple[int, int, int, int]) -> None:
    """Paste a trimmed render centered in ``box`` (x, y, width, height)."""
    x, y, w, h = box
    obj = fit(trimmed(path), w, h)
    canvas.paste(
        obj,
        (int(x + (w - obj.width) / 2), int(y + (h - obj.height) / 2)),
        obj,
    )


def text_center(  # noqa: PLR0913
    canvas: Image.Image,
    text: str,
    y: float,
    face: ImageFont.FreeTypeFont,
    color: tuple[int, int, int] = LIGHT,
    left: float = 0,
    width: float | None = None,
) -> None:
    draw = ImageDraw.Draw(canvas)
    width = canvas.width if width is None else width
    x = left + (width - draw.textlength(text, font=face)) / 2
    draw.text((x, y), text, font=face, fill=color)


def rounded(img: Image.Image, radius: int = 14) -> Image.Image:
    img = img.convert("RGBA")
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, *img.size), radius, fill=255)
    img.putalpha(mask)
    return img
