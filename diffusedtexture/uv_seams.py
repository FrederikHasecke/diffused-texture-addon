from dataclasses import dataclass
from math import ceil
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

UV_EPSILON = 1e-8
INWARD_OFFSETS_TEXELS = (0.5, 0.75, 1.25)
AMBIGUOUS_ERROR_MARGIN = 0.15


@dataclass(frozen=True)
class UVSeamCandidate:
    """Topology seam sides that correspond to the same mesh edge."""

    edge_id: int
    uv_a_start: NDArray[np.float32]
    uv_a_end: NDArray[np.float32]
    uv_b_start: NDArray[np.float32]
    uv_b_end: NDArray[np.float32]
    inward_a: NDArray[np.float32]
    inward_b: NDArray[np.float32]


@dataclass(frozen=True)
class SeamLinkVote:
    """A raster sample vote from one seam-side texel to its topology partner."""

    source_yx: tuple[int, int]
    target_yx: tuple[int, int]
    edge_id: int
    sample_t: float
    error: float


class ResolvedSeamLinks(NamedTuple):
    """Resolved directed seam links after vote aggregation."""

    source_yx: NDArray[np.int32]
    target_yx: NDArray[np.int32]
    weight: NDArray[np.float32]
    edge_id: NDArray[np.int32]
    sample_t: NDArray[np.float32]


class SeamTopologyAssets(NamedTuple):
    """Sparse topology seam assets consumed by UV stitching and repair."""

    seam_line_mask: NDArray[np.uint8]
    seam_link_source_yx: NDArray[np.int32]
    seam_link_target_yx: NDArray[np.int32]
    seam_link_weight: NDArray[np.float32]
    seam_link_edge_id: NDArray[np.int32]
    seam_link_t: NDArray[np.float32]
    seam_unresolved_link_mask: NDArray[np.uint8]


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parents = list(range(size))

    def find(self, index: int) -> int:
        parent = self.parents[index]
        while parent != self.parents[parent]:
            parent = self.parents[parent]
        while index != parent:
            next_index = self.parents[index]
            self.parents[index] = parent
            index = next_index
        return parent

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parents[right_root] = left_root


ResolvedCandidate = tuple[tuple[int, int], int, float, float, float]


def empty_yx_array() -> NDArray[np.int32]:
    return np.empty((0, 2), dtype=np.int32)


def empty_float_array() -> NDArray[np.float32]:
    return np.empty((0,), dtype=np.float32)


def empty_int_array() -> NDArray[np.int32]:
    return np.empty((0,), dtype=np.int32)


def normalize_uv_vector(vector: NDArray[np.float32]) -> NDArray[np.float32]:
    norm = float(np.linalg.norm(vector))
    if norm <= UV_EPSILON:
        return np.zeros(2, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def uv_to_image_xy(
    uv: NDArray[np.float32],
    texture_resolution: int,
) -> NDArray[np.float32]:
    """Convert Blender UV coordinates to array image x/y coordinates."""
    resolution_span = float(max(texture_resolution - 1, 1))
    uv = uv.astype(np.float32)
    return np.array(
        [
            uv[0] * resolution_span,
            (1.0 - uv[1]) * resolution_span,
        ],
        dtype=np.float32,
    )


def uv_to_image_yx(
    uv: NDArray[np.float32],
    texture_resolution: int,
) -> NDArray[np.float32]:
    xy = uv_to_image_xy(uv, texture_resolution)
    return np.array([xy[1], xy[0]], dtype=np.float32)


def _quantize_image_xy(
    xy: NDArray[np.float32],
    surface_mask: NDArray[np.uint8],
) -> tuple[int, int] | None:
    height, width = surface_mask.shape
    x = int(np.mod(np.rint(float(xy[0])), width))
    y = int(np.mod(np.rint(float(xy[1])), height))
    if int(surface_mask[y, x]) == 0:
        return None
    return y, x


def _mark_line_samples(
    line_mask: NDArray[np.uint8],
    start_uv: NDArray[np.float32],
    end_uv: NDArray[np.float32],
    texture_resolution: int,
) -> None:
    start_xy = uv_to_image_xy(start_uv, texture_resolution)
    end_xy = uv_to_image_xy(end_uv, texture_resolution)
    sample_count = max(2, ceil(float(np.linalg.norm(end_xy - start_xy)) * 2.0))
    for sample_index in range(sample_count + 1):
        t = float(sample_index) / float(sample_count)
        uv = ((1.0 - t) * start_uv) + (t * end_uv)
        xy = uv_to_image_xy(uv, texture_resolution)
        x = int(np.mod(np.rint(float(xy[0])), line_mask.shape[1]))
        y = int(np.mod(np.rint(float(xy[1])), line_mask.shape[0]))
        line_mask[y, x] = 255


def _offset_texel_for_side(
    uv: NDArray[np.float32],
    inward: NDArray[np.float32],
    line_yx: tuple[int, int] | None,
    surface_mask: NDArray[np.uint8],
    texture_resolution: int,
) -> tuple[tuple[int, int], float] | None:
    texel_uv = 1.0 / float(max(texture_resolution - 1, 1))
    for offset_texels in INWARD_OFFSETS_TEXELS:
        offset_uv = uv + (inward * offset_texels * texel_uv)
        offset_xy = uv_to_image_xy(offset_uv, texture_resolution)
        quantized = _quantize_image_xy(offset_xy, surface_mask)
        if quantized is None:
            continue
        if line_yx is not None and quantized == line_yx:
            continue

        center_xy = np.array([quantized[1], quantized[0]], dtype=np.float32)
        error = float(np.linalg.norm(offset_xy - center_xy))
        return quantized, error
    return None


def _candidate_votes(
    candidate: UVSeamCandidate,
    surface_mask: NDArray[np.uint8],
    texture_resolution: int,
) -> tuple[list[SeamLinkVote], bool]:
    a_start_xy = uv_to_image_xy(candidate.uv_a_start, texture_resolution)
    a_end_xy = uv_to_image_xy(candidate.uv_a_end, texture_resolution)
    b_start_xy = uv_to_image_xy(candidate.uv_b_start, texture_resolution)
    b_end_xy = uv_to_image_xy(candidate.uv_b_end, texture_resolution)
    edge_len_px = max(
        float(np.linalg.norm(a_end_xy - a_start_xy)),
        float(np.linalg.norm(b_end_xy - b_start_xy)),
    )
    if edge_len_px <= UV_EPSILON:
        return [], True

    sample_count = max(2, ceil(edge_len_px * 2.0))
    votes: list[SeamLinkVote] = []
    unresolved = False

    for sample_index in range(sample_count):
        t = (float(sample_index) + 0.5) / float(sample_count)
        uv_a = ((1.0 - t) * candidate.uv_a_start) + (t * candidate.uv_a_end)
        uv_b = ((1.0 - t) * candidate.uv_b_start) + (t * candidate.uv_b_end)

        line_a = _quantize_image_xy(
            uv_to_image_xy(uv_a, texture_resolution),
            surface_mask,
        )
        line_b = _quantize_image_xy(
            uv_to_image_xy(uv_b, texture_resolution),
            surface_mask,
        )
        a_result = _offset_texel_for_side(
            uv_a,
            candidate.inward_a,
            line_a,
            surface_mask,
            texture_resolution,
        )
        b_result = _offset_texel_for_side(
            uv_b,
            candidate.inward_b,
            line_b,
            surface_mask,
            texture_resolution,
        )
        if a_result is None or b_result is None:
            unresolved = True
            continue

        source_a, error_a = a_result
        source_b, error_b = b_result
        if source_a == source_b:
            unresolved = True
            continue

        votes.append(
            SeamLinkVote(
                source_yx=source_a,
                target_yx=source_b,
                edge_id=candidate.edge_id,
                sample_t=t,
                error=error_a + error_b,
            ),
        )
        votes.append(
            SeamLinkVote(
                source_yx=source_b,
                target_yx=source_a,
                edge_id=candidate.edge_id,
                sample_t=t,
                error=error_a + error_b,
            ),
        )

    return votes, unresolved


def resolve_seam_link_votes(votes: list[SeamLinkVote]) -> ResolvedSeamLinks:
    vote_stats: dict[
        tuple[tuple[int, int], tuple[int, int], int],
        list[float],
    ] = {}
    for vote in votes:
        key = (vote.source_yx, vote.target_yx, vote.edge_id)
        stats = vote_stats.setdefault(key, [0.0, 0.0, 0.0])
        stats[0] += 1.0
        stats[1] += vote.error
        stats[2] += vote.sample_t

    per_source: dict[tuple[int, int], list[ResolvedCandidate]] = {}
    for (source_yx, target_yx, edge_id), stats in vote_stats.items():
        count = stats[0]
        per_source.setdefault(source_yx, []).append(
            (
                target_yx,
                edge_id,
                count,
                stats[1] / count,
                stats[2] / count,
            ),
        )

    resolved_sources: list[tuple[int, int]] = []
    resolved_targets: list[tuple[int, int]] = []
    resolved_weights: list[float] = []
    resolved_edge_ids: list[int] = []
    resolved_sample_t: list[float] = []

    for source_yx, candidates in per_source.items():
        candidates.sort(key=lambda item: (-item[2], item[3]))
        target_yx, edge_id, count, average_error, sample_t = candidates[0]
        if len(candidates) > 1:
            _other_target, _other_edge_id, other_count, other_error, _other_t = (
                candidates[1]
            )
            same_count = count == other_count
            similar_error = average_error >= other_error - AMBIGUOUS_ERROR_MARGIN
            if same_count and similar_error:
                continue

        resolved_sources.append(source_yx)
        resolved_targets.append(target_yx)
        resolved_weights.append(float(count / (1.0 + average_error)))
        resolved_edge_ids.append(edge_id)
        resolved_sample_t.append(sample_t)

    if not resolved_sources:
        return ResolvedSeamLinks(
            source_yx=empty_yx_array(),
            target_yx=empty_yx_array(),
            weight=empty_float_array(),
            edge_id=empty_int_array(),
            sample_t=empty_float_array(),
        )

    return ResolvedSeamLinks(
        source_yx=np.array(resolved_sources, dtype=np.int32),
        target_yx=np.array(resolved_targets, dtype=np.int32),
        weight=np.array(resolved_weights, dtype=np.float32),
        edge_id=np.array(resolved_edge_ids, dtype=np.int32),
        sample_t=np.array(resolved_sample_t, dtype=np.float32),
    )


def build_uv_seam_topology_assets(
    seam_candidates: list[UVSeamCandidate],
    surface_mask: NDArray[np.uint8],
    texture_resolution: int,
) -> SeamTopologyAssets:
    seam_line_mask = np.zeros(surface_mask.shape, dtype=np.uint8)
    unresolved_mask = np.zeros(surface_mask.shape, dtype=np.uint8)
    votes: list[SeamLinkVote] = []

    for candidate in seam_candidates:
        _mark_line_samples(
            seam_line_mask,
            candidate.uv_a_start,
            candidate.uv_a_end,
            texture_resolution,
        )
        _mark_line_samples(
            seam_line_mask,
            candidate.uv_b_start,
            candidate.uv_b_end,
            texture_resolution,
        )
        candidate_votes, unresolved = _candidate_votes(
            candidate,
            surface_mask,
            texture_resolution,
        )
        votes.extend(candidate_votes)
        if unresolved:
            _mark_line_samples(
                unresolved_mask,
                candidate.uv_a_start,
                candidate.uv_a_end,
                texture_resolution,
            )
            _mark_line_samples(
                unresolved_mask,
                candidate.uv_b_start,
                candidate.uv_b_end,
                texture_resolution,
            )

    seam_line_mask[surface_mask == 0] = 0
    unresolved_mask[surface_mask == 0] = 0
    resolved_links = resolve_seam_link_votes(votes)
    if resolved_links.source_yx.size:
        source_surface = surface_mask[
            resolved_links.source_yx[:, 0],
            resolved_links.source_yx[:, 1],
        ]
        target_surface = surface_mask[
            resolved_links.target_yx[:, 0],
            resolved_links.target_yx[:, 1],
        ]
        valid_links = (source_surface > 0) & (target_surface > 0)
        source_yx = resolved_links.source_yx[valid_links]
        target_yx = resolved_links.target_yx[valid_links]
        weight = resolved_links.weight[valid_links]
        edge_id = resolved_links.edge_id[valid_links]
        sample_t = resolved_links.sample_t[valid_links]
    else:
        source_yx = empty_yx_array()
        target_yx = empty_yx_array()
        weight = empty_float_array()
        edge_id = empty_int_array()
        sample_t = empty_float_array()

    return SeamTopologyAssets(
        seam_line_mask=seam_line_mask.astype(np.uint8, copy=False),
        seam_link_source_yx=source_yx.astype(np.int32, copy=False),
        seam_link_target_yx=target_yx.astype(np.int32, copy=False),
        seam_link_weight=weight.astype(np.float32, copy=False),
        seam_link_edge_id=edge_id.astype(np.int32, copy=False),
        seam_link_t=sample_t.astype(np.float32, copy=False),
        seam_unresolved_link_mask=unresolved_mask.astype(np.uint8, copy=False),
    )


def seam_link_mask(
    source_yx: NDArray[np.int32],
    target_yx: NDArray[np.int32],
    shape: tuple[int, int],
) -> NDArray[np.uint8]:
    mask = np.zeros(shape, dtype=np.uint8)
    if source_yx.size:
        mask[source_yx[:, 0], source_yx[:, 1]] = 255
    if target_yx.size:
        mask[target_yx[:, 0], target_yx[:, 1]] = 255
    return mask


def seam_link_components(
    source_yx: NDArray[np.int32],
    target_yx: NDArray[np.int32],
) -> list[NDArray[np.int32]]:
    if source_yx.size == 0 or target_yx.size == 0:
        return []

    coord_to_index: dict[tuple[int, int], int] = {}
    coords: list[tuple[int, int]] = []

    def index_for(coord: tuple[int, int]) -> int:
        index = coord_to_index.get(coord)
        if index is None:
            index = len(coords)
            coord_to_index[coord] = index
            coords.append(coord)
        return index

    edge_indices: list[tuple[int, int]] = []
    for source, target in zip(source_yx, target_yx, strict=False):
        source_index = index_for((int(source[0]), int(source[1])))
        target_index = index_for((int(target[0]), int(target[1])))
        edge_indices.append((source_index, target_index))

    uf = _UnionFind(len(coords))
    for source_index, target_index in edge_indices:
        uf.union(source_index, target_index)

    grouped: dict[int, list[tuple[int, int]]] = {}
    for index, coord in enumerate(coords):
        grouped.setdefault(uf.find(index), []).append(coord)

    return [
        np.array(group_coords, dtype=np.int32)
        for group_coords in grouped.values()
        if len(group_coords) > 1
    ]
