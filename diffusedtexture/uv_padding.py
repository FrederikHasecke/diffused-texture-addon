"""UV island edge padding ("bleed") for generated textures.

Texels outside the UV islands are never sampled directly, but linear filtering
and mipmapping blend them into the island borders. Extending the island colors
outward by a margin keeps that blend on-surface and prevents visible seams.
"""

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

import numpy as np
from numpy.typing import NDArray

DEFAULT_TEXTURE_MARGIN = 16
TEXTURE_MARGIN_REFERENCE_RESOLUTION = 2048


def scale_texture_margin(
    margin_px: int,
    texture_resolution: int,
    reference_resolution: int = TEXTURE_MARGIN_REFERENCE_RESOLUTION,
) -> int:
    """Scale a margin given at ``reference_resolution`` to ``texture_resolution``.

    Keeps the padding constant in UV space, so it survives the same number of
    mip levels at every texture resolution. A positive margin never rounds to 0.
    """
    if margin_px <= 0:
        return 0
    scaled = round(margin_px * texture_resolution / reference_resolution)
    return max(1, int(scaled))


def pad_uv_islands(
    texture: NDArray[np.uint8],
    coverage_mask: NDArray,
    margin_px: int,
) -> NDArray[np.uint8]:
    """Extend UV island colors into uncovered texels within ``margin_px``.

    Every uncovered texel whose distance to the nearest covered texel is at
    most ``margin_px`` takes the color of that nearest covered texel (Blender's
    "Extend" bake margin). Covered texels are returned byte-identical, and
    uncovered texels farther than ``margin_px`` from any island keep their
    original value.

    Args:
        texture: ``(H, W, C)`` or ``(H, W)`` uint8 texture.
        coverage_mask: ``(H, W)`` mask, non-zero where the UV islands are.
        margin_px: Padding width in texels. ``<= 0`` disables padding.

    Returns:
        A padded copy of ``texture``.
    """
    padded = np.array(texture, dtype=np.uint8, copy=True)
    if margin_px <= 0:
        return padded

    if cv2 is None:
        msg = "OpenCV not available; cannot pad UV island edges."
        raise RuntimeError(msg)

    height, width = padded.shape[:2]
    covered = np.asarray(coverage_mask) > 0
    if covered.ndim > 2:  # noqa: PLR2004
        covered = np.any(covered, axis=-1)
    if covered.shape != (height, width):
        covered = (
            cv2.resize(
                covered.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            )
            > 0
        )

    if not np.any(covered) or np.all(covered):
        return padded

    # distanceTransformWithLabels measures, for every non-zero texel, the
    # distance to the nearest zero texel and labels it with that texel's id.
    uncovered = (~covered).astype(np.uint8)
    distance, raw_labels = cv2.distanceTransformWithLabels(
        uncovered,
        cv2.DIST_L2,
        5,
        labelType=cv2.DIST_LABEL_PIXEL,
    )
    labels = np.asarray(raw_labels, dtype=np.int64)

    flat_indices = np.arange(height * width, dtype=np.int64).reshape(height, width)
    label_to_index = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    label_to_index[labels[covered]] = flat_indices[covered]

    fill = ~covered & (distance <= float(margin_px))
    if not np.any(fill):
        return padded

    source_indices = label_to_index[labels[fill]]
    flat_texture = padded.reshape(height * width, -1)
    flat_texture[flat_indices[fill]] = flat_texture[source_indices]
    return padded
