from __future__ import annotations

import pytest

from parallel_bioemu_tools.manifests import build_scoped_output_path


def test_scoped_output_paths_are_unique_and_idempotent(tmp_path) -> None:
    first = build_scoped_output_path(
        tmp_path,
        run_id="run_a",
        protein_id="protein_1",
        pair_id="pair_x",
        n_samples=64,
    )
    second = build_scoped_output_path(
        tmp_path,
        run_id="run_a",
        protein_id="protein_1",
        pair_id="pair_x",
        n_samples=64,
    )
    different = build_scoped_output_path(
        tmp_path,
        run_id="run_a",
        protein_id="protein_1",
        pair_id="pair_x",
        n_samples=65,
    )
    assert first == second
    assert first != different
    assert str(first).endswith("run_a/protein_1/pair_x/n_64")


def test_scoped_output_paths_reject_unsafe_components(tmp_path) -> None:
    with pytest.raises(ValueError):
        build_scoped_output_path(
            tmp_path,
            run_id="../bad",
            protein_id="protein_1",
            pair_id="pair_x",
            n_samples=64,
        )
