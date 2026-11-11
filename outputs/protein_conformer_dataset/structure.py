"""Processed coordinate structure output helpers."""

from __future__ import annotations

from pathlib import Path

from protein_conformer_dataset.models import ParsedChain


def write_protein_only_cif(path: str | Path, chain: ParsedChain) -> None:
    """Write a minimal protein-only C-alpha mmCIF for the selected chain.

    The original all-atom mmCIF is preserved separately. This processed file is
    deliberately C-alpha-only because matched C-alpha coordinates are the
    downstream graph interface in the current research prototype.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"data_{chain.record.pdb_id}_protein_only",
        "#",
        "loop_",
        "_atom_site.group_PDB",
        "_atom_site.id",
        "_atom_site.type_symbol",
        "_atom_site.label_atom_id",
        "_atom_site.label_alt_id",
        "_atom_site.label_comp_id",
        "_atom_site.label_asym_id",
        "_atom_site.label_entity_id",
        "_atom_site.label_seq_id",
        "_atom_site.pdbx_PDB_ins_code",
        "_atom_site.Cartn_x",
        "_atom_site.Cartn_y",
        "_atom_site.Cartn_z",
        "_atom_site.occupancy",
        "_atom_site.B_iso_or_equiv",
        "_atom_site.auth_seq_id",
        "_atom_site.auth_comp_id",
        "_atom_site.auth_asym_id",
        "_atom_site.auth_atom_id",
        "_atom_site.pdbx_PDB_model_num",
    ]
    atom_id = 1
    for residue in chain.residues:
        if residue.ca_coord is None:
            continue
        x_coord, y_coord, z_coord = residue.ca_coord
        label_seq = residue.label_seq_id or residue.index + 1
        auth_seq = residue.auth_seq_id if residue.auth_seq_id is not None else label_seq
        insertion = residue.insertion_code or "?"
        label_chain = chain.record.label_chain_id or chain.record.auth_chain_id
        entity = chain.record.entity_id or "1"
        lines.append(
            " ".join(
                [
                    "ATOM", str(atom_id), "C", "CA", ".", residue.residue_name,
                    label_chain, entity, str(label_seq), insertion,
                    f"{x_coord:.3f}", f"{y_coord:.3f}", f"{z_coord:.3f}",
                    f"{residue.occupancy if residue.occupancy is not None else 1.0:.2f}",
                    "0.00", str(auth_seq), residue.residue_name,
                    chain.record.auth_chain_id, "CA", "1",
                ]
            )
        )
        atom_id += 1
    lines.extend(["#", ""])
    destination.write_text("\n".join(lines), encoding="utf-8")
