"""Local assembly of verified BioEmu chunks."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from guided_conformational_path.bioemu_io import BIOEMU_BUNDLE_SCHEMA, load_bioemu_manifest
from .manifests import ChunkCompletionMarker


def merge_chunk_bundles(
    chunk_dirs: Iterable[str | Path], output_dir: str | Path, *, overwrite: bool = False
) -> Path:
    """Validate and merge chunks without copying their ``raw_bioemu`` trees."""
    sources = [Path(value) for value in chunk_dirs]
    if not sources:
        raise ValueError("at least one chunk is required")
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Refusing to merge into non-empty directory: {output}")
        if output.resolve() in {source.resolve() for source in sources}:
            raise ValueError("merge output must not be a chunk directory")
        shutil.rmtree(output)
    manifests = [load_bioemu_manifest(path) for path in sources]
    first = manifests[0]
    sequence = first.get("sequence")
    if not isinstance(sequence, dict) or not sequence.get("sequence"):
        raise ValueError("chunk has no valid sequence identity")
    expected_schema = first.get("schema_version")
    if expected_schema != BIOEMU_BUNDLE_SCHEMA:
        raise ValueError("unsupported BioEmu bundle schema")
    all_samples: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    seen: set[str] = set()
    for chunk_index, (source, manifest) in enumerate(zip(sources, manifests)):
        if manifest.get("schema_version") != expected_schema or manifest.get("sequence") != sequence:
            raise ValueError("chunk schema or sequence identity mismatch")
        samples = manifest.get("samples")
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"chunk has no samples: {source}")
        chunk_ids: list[str] = []
        source_ids: set[str] = set()
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, dict) or not isinstance(sample.get("pdb_path"), str):
                raise ValueError("chunk sample has no pdb_path")
            source_pdb = (source / sample["pdb_path"]).resolve()
            if source.resolve() not in source_pdb.parents or not source_pdb.is_file():
                raise ValueError(f"missing or unsafe chunk sample: {source_pdb}")
            source_id = str(sample.get("node_id") or f"sample_{sample_index:04d}")
            if source_id in source_ids:
                raise ValueError(f"duplicate source node id within chunk: {source_id}")
            source_ids.add(source_id)
            node_id = f"chunk_{chunk_index:03d}_sample_{sample_index:04d}"
            destination = output / "nodes" / f"{node_id}.pdb"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_pdb, destination)
            all_samples.append({"pdb_path": str(destination.relative_to(output)), "node_id": node_id,
                                "metadata": {"chunk_index": chunk_index, "chunk_sample_index": sample_index,
                                              "source_node_id": source_id, "source_pdb_path": sample["pdb_path"]}})
            chunk_ids.append(node_id)
            if node_id in seen:
                raise ValueError("duplicate merged node id")
            seen.add(node_id)
        provenance.append({"chunk_index": chunk_index, "source": str(source), "node_ids": chunk_ids})
    merged = dict(first)
    merged["n_samples_requested"] = sum(int(m.get("n_samples_requested", 0)) for m in manifests)
    merged["samples"] = all_samples
    merged["metadata"] = {**(first.get("metadata") or {}), "chunk_provenance": provenance}
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps(merged, indent=2) + "\n", encoding="utf-8")
    return output


assemble_merged_bundle = merge_chunk_bundles


def assemble_durable_chunks(markers: Iterable[ChunkCompletionMarker], output_dir: str | Path) -> Path:
    """Assemble local chunk directories after requiring every marker to be durable."""
    ordered = sorted(markers, key=lambda marker: marker.chunk_index)
    if not ordered or {marker.chunk_index for marker in ordered} != set(range(len(ordered))):
        raise ValueError("chunk markers are incomplete or non-contiguous")
    return merge_chunk_bundles((marker.bundle_path for marker in ordered), output_dir)


merge_chunks = merge_chunk_bundles
