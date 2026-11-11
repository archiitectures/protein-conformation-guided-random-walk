"""Deterministic explicit sharding for cost-balanced BioEmu tasks."""

from __future__ import annotations

from dataclasses import replace

from .manifests import PairRunSpec, ShardAssignment


def assign_cost_balanced_shards(
    pairs: list[PairRunSpec],
    *,
    run_id: str,
    num_shards: int,
) -> list[ShardAssignment]:
    """Assign tasks deterministically using stable LPT-style greedy balancing."""
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    ordered = sorted(
        pairs,
        key=lambda item: (
            -item.estimated_cost,
            item.protein_id,
            item.pair_id,
            item.n_samples,
        ),
    )

    buckets: list[list[PairRunSpec]] = [[] for _ in range(num_shards)]
    costs = [0.0 for _ in range(num_shards)]

    for pair in ordered:
        shard_index = min(range(num_shards), key=lambda idx: (costs[idx], idx))
        buckets[shard_index].append(pair)
        costs[shard_index] += pair.estimated_cost

    assignments: list[ShardAssignment] = []
    for index, bucket in enumerate(buckets):
        assignments.append(
            ShardAssignment(
                run_id=run_id,
                shard_id=f"shard_{index:03d}",
                shard_index=index,
                total_shards=num_shards,
                tasks=tuple(bucket),
                total_estimated_cost=costs[index],
            )
        )
    return assignments
