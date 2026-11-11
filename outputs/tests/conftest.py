"""Synthetic mmCIF fixtures for the dataset-generator test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import pytest


@pytest.fixture
def mmcif_factory(tmp_path: Path) -> Callable[..., Path]:
    """Return a writer for small standards-oriented mmCIF coordinate fixtures."""
    def write(
        name: str,
        *,
        length: int = 8,
        missing: set[int] | None = None,
        insertion_codes: dict[int, str] | None = None,
        altloc_position: int | None = None,
        models: int = 1,
        ligand: str | None = None,
        partner_chain: bool = False,
        deformation: float = 0.0,
    ) -> Path:
        missing = missing or set()
        insertion_codes = insertion_codes or {}
        path = tmp_path / name
        lines = [
            "data_TEST",
            "_entry.id TEST",
            "#",
            "loop_",
            "_entity.id",
            "_entity.type",
            "1 polymer",
        ]
        if partner_chain:
            lines.append("2 polymer")
        if ligand:
            lines.append("3 non-polymer")
        lines.extend(
            [
                "#",
                "loop_",
                "_entity_poly.entity_id",
                "_entity_poly.type",
                "_entity_poly.pdbx_seq_one_letter_code_can",
                f"1 'polypeptide(L)' {'A' * length}",
            ]
        )
        if partner_chain:
            lines.append("2 'polypeptide(L)' GG")
        lines.extend(["#", "loop_", "_struct_asym.id", "_struct_asym.entity_id", "A 1"])
        if partner_chain:
            lines.append("B 2")
        if ligand:
            lines.append("L 3")
        lines.extend(["#", "loop_", "_entity_poly_seq.entity_id", "_entity_poly_seq.num", "_entity_poly_seq.mon_id", "_entity_poly_seq.hetero"])
        for position in range(1, length + 1):
            lines.append(f"1 {position} ALA n")
        if partner_chain:
            lines.extend(["2 1 GLY n", "2 2 GLY n"])
        lines.extend(
            [
                "#", "loop_", "_atom_site.group_PDB", "_atom_site.id", "_atom_site.type_symbol",
                "_atom_site.label_atom_id", "_atom_site.label_alt_id", "_atom_site.label_comp_id",
                "_atom_site.label_asym_id", "_atom_site.label_entity_id", "_atom_site.label_seq_id",
                "_atom_site.pdbx_PDB_ins_code", "_atom_site.Cartn_x", "_atom_site.Cartn_y",
                "_atom_site.Cartn_z", "_atom_site.occupancy", "_atom_site.B_iso_or_equiv",
                "_atom_site.auth_seq_id", "_atom_site.auth_comp_id", "_atom_site.auth_asym_id",
                "_atom_site.auth_atom_id", "_atom_site.pdbx_PDB_model_num",
            ]
        )
        atom_id = 1
        for model in range(1, models + 1):
            for position in range(1, length + 1):
                if position in missing:
                    continue
                x_coord = (position - 1) * 3.8
                y_coord = deformation * np.sin(position * 0.7)
                insertion = insertion_codes.get(position, "?")
                variants = [(".", 1.0, 0.0)]
                if altloc_position == position:
                    variants = [("A", 0.4, 4.0), ("B", 0.8, 1.0)]
                for altloc, occupancy, extra_y in variants:
                    lines.append(
                        f"ATOM {atom_id} C CA {altloc} ALA A 1 {position} {insertion} "
                        f"{x_coord:.3f} {y_coord + extra_y:.3f} 0.000 {occupancy:.2f} 10.00 "
                        f"{position} ALA A CA {model}"
                    )
                    atom_id += 1
            if partner_chain:
                for position in (1, 2):
                    lines.append(
                        f"ATOM {atom_id} C CA . GLY B 2 {position} ? "
                        f"{position * 3.8:.3f} 20.000 0.000 1.00 10.00 {position} GLY B CA {model}"
                    )
                    atom_id += 1
            if ligand:
                lines.append(
                    f"HETATM {atom_id} P P . {ligand} L 3 . ? 1.000 1.000 1.000 "
                    f"1.00 10.00 501 {ligand} A P {model}"
                )
                atom_id += 1
        lines.extend(["#", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    return write
