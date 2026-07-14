"""Colab-facing BioEmu runner and portable bundle writer.

This module intentionally has no import-time BioEmu dependency. BioEmu is only
imported inside ``run_bioemu_from_fasta`` when the caller does not provide an
explicit sampler callable.
"""

from __future__ import annotations

import importlib
import inspect
import json
import shutil
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from guided_conformational_path.bioemu_io import BIOEMU_BUNDLE_SCHEMA
from guided_conformational_path.io import write_pdb_ca


class BioEmuRunnerError(RuntimeError):
    """Raised when BioEmu sampling or output normalization cannot proceed."""


def read_fasta(path: str | Path) -> tuple[str, str]:
    """Read the first FASTA record from ``path``.

    Args:
        path: FASTA file containing at least one sequence record.

    Returns:
        ``(record_name, sequence)`` with whitespace removed from sequence lines.

    Raises:
        ValueError: If the file is empty, lacks a FASTA header, or has no
            sequence characters.
    """
    fasta_path = Path(path)
    name: str | None = None
    sequence_parts: list[str] = []

    with fasta_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None and sequence_parts:
                    break
                name = line[1:].strip() or fasta_path.stem
                continue
            if name is None:
                raise ValueError(f"FASTA file must start with a header line: {fasta_path}")
            sequence_parts.append(line)

    if name is None:
        raise ValueError(f"No FASTA record found in {fasta_path}")
    sequence = "".join(sequence_parts).replace(" ", "").upper()
    if not sequence:
        raise ValueError(f"FASTA record has no sequence characters: {fasta_path}")
    return name, sequence


def run_bioemu_from_fasta(
    fasta_path: str | Path,
    output_dir: str | Path,
    n_samples: int = 200,
    seed: int | None = None,
    sampler: Callable[..., Any] | None = None,
    **bioemu_kwargs: Any,
) -> dict[str, Any]:
    """Run BioEmu on one FASTA sequence and write a portable node bundle.

    Args:
        fasta_path: Input FASTA path. The first FASTA record is used.
        output_dir: Bundle directory to create or update. The runner writes
            ``manifest.json``, ``nodes/*.pdb``, and ``raw_bioemu/``.
        n_samples: Number of BioEmu samples to request. Defaults to 200.
        seed: Optional random seed forwarded to the sampler when supported.
        sampler: Optional adapter callable. This is the clearest integration
            point when a BioEmu deployment uses a different API. The callable may
            return PDB paths, PDB text strings, coordinate arrays with shape
            ``(L, 3)`` or ``(N, L, 3)``, dictionaries containing those values, or
            ``None`` after writing PDB files under the provided output directory.
        **bioemu_kwargs: Extra sampler-specific keyword arguments.

    Returns:
        The manifest dictionary that was also written to
        ``output_dir/manifest.json``.

    Raises:
        BioEmuRunnerError: If BioEmu is unavailable, no compatible API can be
            found, sampling fails, or outputs cannot be normalized to PDB nodes.
        ValueError: If arguments are invalid.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    fasta = Path(fasta_path).expanduser().resolve()
    bundle_dir = Path(output_dir).expanduser().resolve()
    nodes_dir = bundle_dir / "nodes"
    raw_output_dir = bundle_dir / "raw_bioemu"
    nodes_dir.mkdir(parents=True, exist_ok=True)
    raw_output_dir.mkdir(parents=True, exist_ok=True)

    sequence_name, sequence = read_fasta(fasta)

    if sampler is None:
        # BioEmu imports are deliberately lazy and confined to this runner.
        try:
            bioemu = importlib.import_module("bioemu")
        except ImportError as exc:
            raise BioEmuRunnerError(
                "BioEmu is not installed in this environment. In Colab, run "
                '`pip install "bioemu[cuda]"`, or pass a sampler callable to '
                "run_bioemu_from_fasta(..., sampler=...)."
            ) from exc
        bioemu_modules = [bioemu]
        for module_name in ("bioemu.sample", "bioemu.api", "bioemu.inference"):
            try:
                bioemu_modules.append(importlib.import_module(module_name))
            except ImportError:
                continue
        sampler = _resolve_bioemu_sampler(bioemu_modules)

    try:
        result = _call_sampler(
            sampler=sampler,
            fasta_path=fasta,
            sequence=sequence,
            output_dir=raw_output_dir,
            n_samples=n_samples,
            seed=seed,
            bioemu_kwargs=bioemu_kwargs,
        )
    except BioEmuRunnerError:
        raise
    except Exception as exc:  # pragma: no cover - depends on external BioEmu.
        raise BioEmuRunnerError(f"BioEmu sampler failed: {exc}") from exc

    node_records = _normalize_outputs_to_nodes(
        result=result,
        raw_output_dir=raw_output_dir,
        nodes_dir=nodes_dir,
    )
    if not node_records:
        raise BioEmuRunnerError(
            "BioEmu finished, but no PDB nodes could be found or normalized. "
            "Pass a sampler callable that returns PDB paths, PDB text strings, "
            "or coordinate arrays, or make the BioEmu API write .pdb files into "
            f"{raw_output_dir}."
        )

    manifest = {
        "schema_version": BIOEMU_BUNDLE_SCHEMA,
        "generator": "BioEmu",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sequence": {
            "id": sequence_name,
            "sequence": sequence,
            "length": len(sequence),
            "fasta_path": str(fasta),
        },
        "n_samples_requested": n_samples,
        "inputs": {
            "endpoint_a_path": None,
            "endpoint_b_path": None,
        },
        "samples": node_records,
        "run": {
            "n_samples_requested": n_samples,
            "n_samples_written": len(node_records),
            "seed": seed,
            "raw_output_dir": "raw_bioemu",
        },
        "metadata": {
            "generator": "BioEmu",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "bioemu_kwargs": bioemu_kwargs,
        },
    }
    manifest_path = bundle_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _resolve_bioemu_sampler(bioemu_modules: Iterable[Any]) -> Callable[..., Any]:
    """Find a callable sampler on an imported BioEmu module."""
    candidate_names = (
        "sample",
        "sample_ensemble",
        "generate",
        "generate_ensemble",
        "run",
    )
    for module in bioemu_modules:
        for name in candidate_names:
            candidate = getattr(module, name, None)
            if callable(candidate):
                return candidate

    raise BioEmuRunnerError(
        "BioEmu is installed, but this runner could not find a supported "
        "sampling entry point. Pass an adapter callable with "
        "run_bioemu_from_fasta(..., sampler=your_sampler). The callable should "
        "accept some subset of fasta_path, sequence, output_dir, n_samples, "
        "num_samples, and seed, then return PDB paths/text or coordinate arrays."
    )


def _call_sampler(
    sampler: Callable[..., Any],
    fasta_path: Path,
    sequence: str,
    output_dir: Path,
    n_samples: int,
    seed: int | None,
    bioemu_kwargs: dict[str, Any],
) -> Any:
    """Call a BioEmu sampler while adapting common parameter names."""
    aliases: dict[str, Any] = {
        "fasta_path": str(fasta_path),
        "fasta": str(fasta_path),
        "sequence": sequence,
        "output_dir": str(output_dir),
        "out_dir": str(output_dir),
        "n_samples": n_samples,
        "num_samples": n_samples,
        "samples": n_samples,
        "seed": seed,
    }
    if seed is None:
        aliases.pop("seed")

    try:
        signature = inspect.signature(sampler)
    except (TypeError, ValueError):
        kwargs = {
            "fasta_path": str(fasta_path),
            "output_dir": str(output_dir),
            "n_samples": n_samples,
            **({"seed": seed} if seed is not None else {}),
            **bioemu_kwargs,
        }
        return sampler(**kwargs)

    accepts_var_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    if accepts_var_kwargs:
        kwargs = {
            "fasta_path": str(fasta_path),
            "sequence": sequence,
            "output_dir": str(output_dir),
            "n_samples": n_samples,
            **({"seed": seed} if seed is not None else {}),
            **bioemu_kwargs,
        }
        return sampler(**kwargs)

    kwargs = {
        name: value
        for name, value in aliases.items()
        if name in signature.parameters and value is not None
    }
    kwargs.update(
        {
            name: value
            for name, value in bioemu_kwargs.items()
            if name in signature.parameters
        }
    )
    if not kwargs:
        raise BioEmuRunnerError(
            "The provided BioEmu sampler has no recognized parameters. Expected "
            "some subset of fasta_path, sequence, output_dir, n_samples, "
            "num_samples, or seed."
        )
    return sampler(**kwargs)


def _normalize_outputs_to_nodes(
    result: Any,
    raw_output_dir: Path,
    nodes_dir: Path,
) -> list[dict[str, Any]]:
    """Convert supported BioEmu outputs into ``nodes/*.pdb`` records."""
    records: list[dict[str, Any]] = []
    seen_paths: set[Path] = set()
    for item in _iter_output_items(result, raw_output_dir):
        node_id = f"bioemu_{len(records):06d}"
        destination = nodes_dir / f"{node_id}.pdb"
        metadata: dict[str, Any] = {"generator": "BioEmu"}

        if isinstance(item, Path):
            source_path = item.resolve()
            if source_path in seen_paths or source_path.suffix.lower() != ".pdb":
                continue
            seen_paths.add(source_path)
            shutil.copyfile(source_path, destination)
            metadata["source_path"] = str(source_path)
        elif isinstance(item, str) and _looks_like_pdb_text(item):
            destination.write_text(_ensure_pdb_end(item), encoding="utf-8")
            metadata["source_format"] = "pdb_text"
        else:
            coords = _as_coordinate_array(item)
            if coords is None:
                continue
            write_pdb_ca(destination, coords)
            metadata["source_format"] = "ca_coordinates"

        records.append(
            {
                "node_id": node_id,
                "source": "external",
                "pdb_path": str(destination.relative_to(nodes_dir.parent)),
                "metadata": metadata,
            }
        )
    return records


def _iter_output_items(result: Any, raw_output_dir: Path) -> Iterable[Any]:
    """Yield possible PDB paths, PDB text blobs, or coordinate arrays."""
    if result is None:
        yield from sorted(raw_output_dir.rglob("*.pdb"))
        return

    if isinstance(result, (str, Path)):
        path = Path(result)
        if isinstance(result, str) and _looks_like_pdb_text(result):
            yield result
        elif not path.is_absolute() and (raw_output_dir / path).exists():
            yield raw_output_dir / path
        elif path.exists() and path.is_dir():
            yield from sorted(path.rglob("*.pdb"))
        elif path.exists():
            yield path
        return

    array = _as_coordinate_array(result)
    if array is not None:
        if array.ndim == 3:
            yield from array
        else:
            yield array
        return

    if isinstance(result, dict):
        for key in (
            "pdb_paths",
            "paths",
            "files",
            "pdbs",
            "structures",
            "coords",
            "coordinates",
            "samples",
        ):
            if key in result:
                yield from _iter_output_items(result[key], raw_output_dir)
        return

    for attr in (
        "pdb_paths",
        "paths",
        "files",
        "pdbs",
        "structures",
        "coords",
        "coordinates",
        "samples",
    ):
        if hasattr(result, attr):
            yield from _iter_output_items(getattr(result, attr), raw_output_dir)
            return

    if isinstance(result, Iterable):
        for item in result:
            yield from _iter_output_items(item, raw_output_dir)


def _as_coordinate_array(value: Any) -> np.ndarray | None:
    """Return ``value`` as a float coordinate array if it has a supported shape."""
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    if array.ndim == 2 and array.shape[1] == 3:
        return array
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    return None


def _looks_like_pdb_text(value: str) -> bool:
    """Return true when ``value`` appears to contain PDB records."""
    stripped = value.lstrip()
    return stripped.startswith(("ATOM", "HETATM", "MODEL")) or "\nATOM" in value


def _ensure_pdb_end(value: str) -> str:
    """Ensure PDB text ends with a newline and END record."""
    text = value if value.endswith("\n") else f"{value}\n"
    if not text.rstrip().endswith("END"):
        text += "END\n"
    return text
