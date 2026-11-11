from __future__ import annotations

from parallel_bioemu_tools.manifests import PairRunSpec
from parallel_bioemu_tools.sharding import assign_cost_balanced_shards


def _pair(protein_id: str, pair_id: str, cost: float) -> PairRunSpec:
    return PairRunSpec(
        protein_id=protein_id,
        pair_id=pair_id,
        fasta_path=f"/tmp/{protein_id}_{pair_id}.fasta",
        dataset_pair_dir=f"/tmp/{protein_id}/{pair_id}",
        n_samples=200,
        estimated_cost=cost,
    )


def test_cost_balanced_sharding_is_deterministic() -> None:
    pairs = [
        _pair("p2", "b", 3.0),
        _pair("p1", "a", 10.0),
        _pair("p3", "c", 4.0),
        _pair("p4", "d", 9.0),
    ]
    first = assign_cost_balanced_shards(pairs, run_id="run1", num_shards=2)
    second = assign_cost_balanced_shards(list(reversed(pairs)), run_id="run1", num_shards=2)
    assert [[task.storage_key for task in shard.tasks] for shard in first] == [
        [task.storage_key for task in shard.tasks] for shard in second
    ]
    costs = [shard.total_estimated_cost for shard in first]
    assert costs == [13.0, 13.0]
