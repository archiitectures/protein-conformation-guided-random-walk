"""Simple PDB-style C-alpha coordinate input and output.

The parser and writer are intentionally minimal. They support enough fixed-column
PDB syntax to run demos and tests, but they are not replacements for BioPython,
MDAnalysis, or a full mmCIF/PDB parser.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_pdb_ca(path: str | Path) -> np.ndarray:
    """Read C-alpha coordinates from a PDB file.

    This intentionally supports a minimal PDB subset for the MVP. It reads
    ``ATOM``/``HETATM`` records whose atom name is ``CA`` and returns an array of
    shape ``(L, 3)``. Residue identity, chain breaks, alternate locations, and
    insertion codes are not interpreted yet.
    """
    pdb_path = Path(path)
    coords: list[list[float]] = []
    with pdb_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atom_name = line[12:16].strip()
            if atom_name != "CA":
                continue
            try:
                coords.append(
                    [
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54]),
                    ]
                )
            except ValueError as exc:
                raise ValueError(f"Could not parse coordinate line in {pdb_path}: {line!r}") from exc

    if not coords:
        raise ValueError(f"No C-alpha coordinates found in {pdb_path}")
    return np.asarray(coords, dtype=float)


def write_pdb_ca(
    path: str | Path,
    coords: np.ndarray,
    residue_name: str = "ALA",
    chain_id: str = "A",
) -> None:
    """Write C-alpha coordinates to a minimal PDB-like file.

    The output is suitable for inspecting generated C-alpha traces and for this
    package's own ``read_pdb_ca`` function. It is not intended to be a complete
    physically meaningful all-atom PDB.
    """
    pdb_path = Path(path)
    pdb_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(coords, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"Coordinates must have shape (L, 3); got {array.shape}")

    with pdb_path.open("w", encoding="utf-8") as handle:
        for idx, (x_coord, y_coord, z_coord) in enumerate(array, start=1):
            handle.write(
                f"ATOM  {idx:5d}  CA  {residue_name:>3s} {chain_id}{idx:4d}    "
                f"{x_coord:8.3f}{y_coord:8.3f}{z_coord:8.3f}"
                f"  1.00  0.00           C\n"
            )
        handle.write("END\n")
