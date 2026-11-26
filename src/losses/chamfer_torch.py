"""Differentiable Chamfer distance implemented with native PyTorch ops.

The public ``chamfer_loss`` function accepts both point-cloud layouts used by
this project: ``[B, N, 3]`` and ``[B, 3, N]``.  Pairwise distances are
evaluated in chunks so the fallback does not allocate a full ``N x M`` matrix.
"""

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn


_DEFAULT_CHUNK_SIZE = 1024


def _as_points_last(points: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(points, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    if points.ndim != 3:
        raise ValueError(
            f"{name} must have shape [B, N, 3] or [B, 3, N], "
            f"but received {tuple(points.shape)}."
        )
    if points.shape[-1] == 3:
        normalized = points
    elif points.shape[1] == 3:
        normalized = points.transpose(1, 2)
    else:
        raise ValueError(
            f"{name} must have a coordinate dimension of size 3, "
            f"but received {tuple(points.shape)}."
        )
    if normalized.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one point.")
    if not normalized.is_floating_point():
        raise TypeError(f"{name} must use a floating-point dtype.")
    return normalized.contiguous()


def _resolve_chunk_size(chunk_size: Optional[int]) -> int:
    if chunk_size is None:
        chunk_size = int(
            os.getenv("ZVISION3_CHAMFER_CHUNK_SIZE", _DEFAULT_CHUNK_SIZE)
        )
    if chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return chunk_size


def _cdist_input(points: torch.Tensor) -> torch.Tensor:
    # torch.cdist does not support every low-precision dtype/device pair.
    # Computing distances in float32 keeps the fallback portable while the
    # cast remains differentiable with respect to the original tensor.
    if points.dtype in (torch.float16, torch.bfloat16):
        return points.float()
    return points


def _nearest_distances(
    query: torch.Tensor,
    reference: torch.Tensor,
    chunk_size: int,
) -> torch.Tensor:
    nearest_chunks = []
    for start in range(0, query.shape[1], chunk_size):
        query_chunk = query[:, start : start + chunk_size]
        # Nearest-neighbour assignment is discrete.  Determine it without
        # building an autograd graph for the pairwise matrix, then calculate
        # the selected distances with regular differentiable tensor ops.
        with torch.no_grad():
            pairwise = torch.cdist(
                _cdist_input(query_chunk.detach()),
                _cdist_input(reference.detach()),
                p=2,
            )
            nearest_indices = pairwise.argmin(dim=-1)
        nearest_points = torch.gather(
            reference,
            dim=1,
            index=nearest_indices.unsqueeze(-1).expand(-1, -1, 3),
        )
        nearest_chunks.append(
            torch.linalg.vector_norm(query_chunk - nearest_points, dim=-1)
        )
    return torch.cat(nearest_chunks, dim=1)


def _bidirectional_nearest_distances(
    template: torch.Tensor,
    source: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    template = _as_points_last(template, "template")
    source = _as_points_last(source, "source")
    if template.shape[0] != source.shape[0]:
        raise ValueError(
            "template and source must have the same batch size, "
            f"but received {template.shape[0]} and {source.shape[0]}."
        )
    if template.device != source.device:
        raise ValueError("template and source must be on the same device.")
    if template.dtype != source.dtype:
        raise ValueError("template and source must have the same dtype.")

    chunk_size = _resolve_chunk_size(chunk_size)
    template_to_source = _nearest_distances(
        template,
        source,
        chunk_size,
    )
    source_to_template = _nearest_distances(
        source,
        template,
        chunk_size,
    )
    return template_to_source, source_to_template


def chamfer_distance_raw_torch(
    template: torch.Tensor,
    source: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return bidirectional squared nearest-neighbour distances.

    The return shapes and squared-distance semantics match the project's
    compiled ``ChamferDistance`` extension: ``[B, N]`` and ``[B, M]``.
    """

    template_to_source, source_to_template = (
        _bidirectional_nearest_distances(
            template,
            source,
            chunk_size=chunk_size,
        )
    )
    return template_to_source.square(), source_to_template.square()


def chamfer_loss(
    template: torch.Tensor,
    source: torch.Tensor,
    reduce: str = "mean",
    *,
    reduction: Optional[str] = None,
    chunk_size: Optional[int] = None,
    direction_weights: Tuple[float, float] = (0.5, 0.5),
) -> torch.Tensor:
    """Compute a weighted Euclidean Chamfer distance.

    ``reduce`` mirrors the former ``neuralnet_pytorch.chamfer_loss`` call
    sites.  ``"none"`` returns one value per batch item.

    """

    if reduction is not None:
        if reduce != "mean" and reduce != reduction:
            raise ValueError("reduce and reduction specify different values.")
        reduce = reduction
    if reduce not in {"none", "mean", "sum"}:
        raise ValueError(f"Unsupported reduction: {reduce}.")
    if len(direction_weights) != 2:
        raise ValueError("direction_weights must contain exactly two values.")

    dist_template, dist_source = _bidirectional_nearest_distances(
        template,
        source,
        chunk_size=chunk_size,
    )
    per_batch = (
        direction_weights[0] * dist_template.mean(dim=-1)
        + direction_weights[1] * dist_source.mean(dim=-1)
    )
    if reduce == "none":
        return per_batch
    if reduce == "mean":
        return per_batch.mean()
    return per_batch.sum()


class TorchChamferDistance(nn.Module):
    """Drop-in native-PyTorch replacement for the compiled raw operator."""

    def __init__(self, chunk_size: Optional[int] = None):
        super().__init__()
        self.chunk_size = chunk_size

    def forward(
        self,
        template: torch.Tensor,
        source: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return chamfer_distance_raw_torch(
            template,
            source,
            chunk_size=self.chunk_size,
        )


class ChamferDistanceLoss(nn.Module):
    def __init__(
        self,
        reduction: str = "mean",
        chunk_size: Optional[int] = None,
        direction_weights: Tuple[float, float] = (0.5, 0.5),
    ):
        super().__init__()
        if reduction not in {"none", "mean", "sum"}:
            raise ValueError(f"Unsupported reduction: {reduction}.")
        self.reduction = reduction
        self.chunk_size = chunk_size
        self.direction_weights = direction_weights

    def forward(
        self,
        template: torch.Tensor,
        source: torch.Tensor,
    ) -> torch.Tensor:
        return chamfer_loss(
            template,
            source,
            reduce=self.reduction,
            chunk_size=self.chunk_size,
            direction_weights=self.direction_weights,
        )
