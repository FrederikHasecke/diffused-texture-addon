# ruff: noqa: S101

import numpy as np
import pytest

from diffused_texture_addon.diffusedtexture.uv_padding import (
    pad_uv_islands,
    scale_texture_margin,
)

WHITE = 255
RED = (200, 10, 10)
BLUE = (10, 10, 200)


def _two_island_texture() -> tuple[np.ndarray, np.ndarray]:
    """64x64 white texture with a red island (left) and a blue island (right)."""
    texture = np.full((64, 64, 3), WHITE, dtype=np.uint8)
    mask = np.zeros((64, 64), dtype=np.uint8)

    mask[20:40, 8:20] = 255
    mask[20:40, 44:56] = 255
    texture[20:40, 8:20] = RED
    texture[20:40, 44:56] = BLUE
    # Per-texel detail so "nearest island color" is checked texel by texel.
    texture[20:40, 8:20, 1] = np.arange(20, dtype=np.uint8)[:, None] + 50
    texture[20:40, 44:56, 1] = np.arange(12, dtype=np.uint8)[None, :] + 100
    return texture, mask


def test_pad_uv_islands_keeps_island_texels_byte_identical() -> None:
    texture, mask = _two_island_texture()

    padded = pad_uv_islands(texture, mask, margin_px=4)

    covered = mask > 0
    assert padded.dtype == np.uint8
    assert np.array_equal(padded[covered], texture[covered])


def test_pad_uv_islands_fills_margin_with_nearest_island_texel() -> None:
    texture, mask = _two_island_texture()

    padded = pad_uv_islands(texture, mask, margin_px=4)

    # Straight out of an edge: copies the adjacent border texel.
    assert np.array_equal(padded[25, 7], texture[25, 8])
    assert np.array_equal(padded[25, 4], texture[25, 8])
    assert np.array_equal(padded[19, 12], texture[20, 12])
    assert np.array_equal(padded[42, 50], texture[39, 50])
    assert np.array_equal(padded[30, 58], texture[30, 55])
    # Diagonal off a corner: copies the corner texel.
    assert np.array_equal(padded[18, 6], texture[20, 8])
    # No background color is left directly around an island.
    ring = np.zeros_like(mask, dtype=bool)
    ring[18:42, 6:22] = True
    ring[mask > 0] = False
    assert not np.any(np.all(padded[ring] == WHITE, axis=-1))


def test_pad_uv_islands_leaves_texels_beyond_margin_untouched() -> None:
    texture, mask = _two_island_texture()

    padded = pad_uv_islands(texture, mask, margin_px=4)

    # Background far from both islands (gap center, image corners) stays white.
    assert np.array_equal(padded[30, 32], [WHITE] * 3)
    assert np.array_equal(padded[0, 0], [WHITE] * 3)
    assert np.array_equal(padded[63, 63], [WHITE] * 3)
    assert np.array_equal(padded[25, 3], [WHITE] * 3)  # 5 px left of red island

    changed = np.any(padded != texture, axis=-1)
    assert not np.any(changed[mask > 0])
    assert np.any(changed)


def test_pad_uv_islands_zero_margin_is_noop_copy() -> None:
    texture, mask = _two_island_texture()

    padded = pad_uv_islands(texture, mask, margin_px=0)

    assert np.array_equal(padded, texture)
    assert padded is not texture


@pytest.mark.parametrize("fill", [0, 255])
def test_pad_uv_islands_handles_empty_and_full_masks(fill: int) -> None:
    texture, _mask = _two_island_texture()
    mask = np.full((64, 64), fill, dtype=np.uint8)

    assert np.array_equal(pad_uv_islands(texture, mask, margin_px=8), texture)


def test_pad_uv_islands_does_not_modify_input() -> None:
    texture, mask = _two_island_texture()
    original = texture.copy()

    pad_uv_islands(texture, mask, margin_px=4)

    assert np.array_equal(texture, original)


def test_pad_uv_islands_resizes_mismatched_mask() -> None:
    texture, mask = _two_island_texture()
    small_mask = mask[::2, ::2]

    padded = pad_uv_islands(texture, small_mask, margin_px=2)

    assert np.array_equal(padded[mask > 0], texture[mask > 0])
    assert np.array_equal(padded[25, 7], texture[25, 8])


@pytest.mark.parametrize(
    ("margin", "resolution", "expected"),
    [
        (16, 2048, 16),
        (16, 4096, 32),
        (16, 1024, 8),
        (16, 512, 4),
        (1, 512, 1),
        (0, 4096, 0),
    ],
)
def test_scale_texture_margin(margin: int, resolution: int, expected: int) -> None:
    assert scale_texture_margin(margin, resolution) == expected
