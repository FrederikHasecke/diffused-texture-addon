# ruff: noqa: S101

import numpy as np

from diffused_texture_addon.diffusedtexture.uv_seams import (
    SeamLinkVote,
    UVSeamCandidate,
    build_uv_seam_topology_assets,
    resolve_seam_link_votes,
    seam_link_components,
    uv_to_image_yx,
)


def test_uv_to_image_yx_uses_array_row_orientation() -> None:
    assert np.allclose(
        uv_to_image_yx(np.array([0.0, 0.0], dtype=np.float32), 8),
        np.array([7.0, 0.0], dtype=np.float32),
    )
    assert np.allclose(
        uv_to_image_yx(np.array([1.0, 1.0], dtype=np.float32), 8),
        np.array([0.0, 7.0], dtype=np.float32),
    )


def test_resolve_seam_link_votes_drops_ambiguous_targets() -> None:
    votes = [
        SeamLinkVote((2, 2), (2, 6), 1, 0.5, 0.1),
        SeamLinkVote((2, 2), (3, 6), 1, 0.5, 0.1),
    ]

    resolved = resolve_seam_link_votes(votes)

    assert resolved.source_yx.shape == (0, 2)
    assert resolved.target_yx.shape == (0, 2)


def test_build_uv_seam_topology_assets_links_opposite_sides() -> None:
    surface_mask = np.full((16, 16), 255, dtype=np.uint8)
    candidate = UVSeamCandidate(
        edge_id=7,
        uv_a_start=np.array([0.25, 0.25], dtype=np.float32),
        uv_a_end=np.array([0.25, 0.75], dtype=np.float32),
        uv_b_start=np.array([0.75, 0.25], dtype=np.float32),
        uv_b_end=np.array([0.75, 0.75], dtype=np.float32),
        inward_a=np.array([1.0, 0.0], dtype=np.float32),
        inward_b=np.array([-1.0, 0.0], dtype=np.float32),
    )

    assets = build_uv_seam_topology_assets([candidate], surface_mask, 16)

    assert np.any(assets.seam_line_mask)
    assert assets.seam_link_source_yx.shape[1:] == (2,)
    assert len(assets.seam_link_source_yx) > 0
    source_x = assets.seam_link_source_yx[:, 1]
    target_x = assets.seam_link_target_yx[:, 1]
    assert np.any(source_x < 8)
    assert np.any(target_x > 8)


def test_seam_link_components_groups_connected_links() -> None:
    components = seam_link_components(
        np.array([[1, 1], [1, 6], [8, 1]], dtype=np.int32),
        np.array([[1, 6], [1, 1], [8, 6]], dtype=np.int32),
    )

    component_sets = [
        {tuple(coord.tolist()) for coord in component}
        for component in components
    ]

    assert {(1, 1), (1, 6)} in component_sets
    assert {(8, 1), (8, 6)} in component_sets
