"""JSON and self-contained HTML reports for candidate graph paths."""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Iterable

import networkx as nx

from guided_conformational_path.path_search import CandidatePathCollection, PathCandidate


def write_candidate_path_reports(
    output_dir: str | Path,
    graph: nx.Graph,
    candidates: CandidatePathCollection,
) -> tuple[Path, Path]:
    """Write the versioned candidate JSON and a sortable HTML comparison."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    json_path = destination / "candidate_paths.json"
    html_path = destination / "candidate_paths.html"
    json_path.write_text(
        json.dumps(candidates.to_dict(), indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    html_path.write_text(_candidate_report_html(graph, candidates), encoding="utf-8")
    return json_path, html_path


def _candidate_report_html(graph: nx.Graph, candidates: CandidatePathCollection) -> str:
    """Return a deterministic, sortable candidate-path report document."""
    positions = nx.spring_layout(graph, seed=7, weight=None)
    exact_rows = _table_rows(graph, candidates.exact_paths, positions)
    diverse_rows = _table_rows(graph, candidates.diverse_paths, positions)
    settings = candidates.settings
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Candidate conformational paths</title>
<style>
:root {{ color-scheme: light; --ink:#17221d; --muted:#637068; --line:#d8ded9; --paper:#f5f3eb; --accent:#b63b2b; }}
body {{ margin:0; padding:2rem; color:var(--ink); background:linear-gradient(135deg,#f5f3eb,#e8efe8); font:15px/1.45 Georgia,serif; }}
main {{ max-width:1500px; margin:auto; background:#fffdf7; padding:2rem; box-shadow:0 18px 60px #23342b22; }}
h1,h2 {{ font-family:"Avenir Next",Avenir,sans-serif; letter-spacing:.02em; }}
.meta {{ color:var(--muted); }} table {{ width:100%; border-collapse:collapse; margin-bottom:2rem; }}
th,td {{ padding:.55rem; border-bottom:1px solid var(--line); text-align:left; vertical-align:middle; }}
th {{ cursor:pointer; background:#edf1eb; font-family:"Avenir Next",Avenir,sans-serif; position:sticky; top:0; }}
tr:hover {{ background:#f7f0df; }} .num {{ font-variant-numeric:tabular-nums; white-space:nowrap; }}
.nodes {{ min-width:18rem; max-width:32rem; overflow-wrap:anywhere; font-family:ui-monospace,monospace; font-size:.8rem; }}
svg {{ width:180px; height:110px; display:block; }} .empty {{ color:var(--muted); font-style:italic; }}
</style>
</head>
<body><main>
<h1>Candidate conformational paths</h1>
<p class="meta">Lowest-cost simple paths and alternatives with at least {_format_percent(float(settings['diversity_threshold']))} edge diversity from the Dijkstra path. Costs are heuristic structural scores, not physical energies or kinetic probabilities. Scanned {candidates.scan_count} candidate paths.</p>
<h2>Exact top paths</h2>
{_table(exact_rows)}
<h2>Diverse alternatives</h2>
{_table(diverse_rows)}
</main>
<script>
document.querySelectorAll('th').forEach((th) => th.addEventListener('click', () => {{
  const table=th.closest('table'), body=table.tBodies[0], index=[...th.parentNode.children].indexOf(th);
  const asc=th.dataset.order!=='asc'; th.dataset.order=asc?'asc':'desc';
  [...body.rows].sort((a,b) => {{
    const av=a.cells[index].dataset.value ?? a.cells[index].textContent;
    const bv=b.cells[index].dataset.value ?? b.cells[index].textContent;
    const an=Number(av), bn=Number(bv); const cmp=Number.isNaN(an)||Number.isNaN(bn)?av.localeCompare(bv):an-bn;
    return asc?cmp:-cmp;
  }}).forEach((row) => body.appendChild(row));
}}));
</script></body></html>
"""


def _table(rows: list[str]) -> str:
    headings = (
        "Set rank", "Global rank", "Cost", "Delta", "Delta %", "Edges",
        "Edge diversity", "Intermediate similarity", "Node sequence", "Graph preview",
    )
    header = "".join(f"<th>{html.escape(label)}</th>" for label in headings)
    body = "".join(rows) if rows else '<tr><td class="empty" colspan="10">No qualifying paths found.</td></tr>'
    return f"<table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table>"


def _table_rows(
    graph: nx.Graph,
    paths: Iterable[PathCandidate],
    positions: dict[object, object],
) -> list[str]:
    rows: list[str] = []
    for set_rank, candidate in enumerate(paths, start=1):
        relative = candidate.relative_delta
        relative_text = "n/a" if relative is None else f"{100.0 * relative:.2f}%"
        relative_value = "" if relative is None else str(relative)
        values = [
            _cell(set_rank, set_rank),
            _cell(candidate.global_cost_rank, candidate.global_cost_rank),
            _cell(f"{candidate.total_cost:.6g}", candidate.total_cost),
            _cell(f"{candidate.absolute_delta:.6g}", candidate.absolute_delta),
            _cell(relative_text, relative_value),
            _cell(candidate.metadata["n_path_edges"], candidate.metadata["n_path_edges"]),
            _cell(_format_percent(candidate.edge_diversity), candidate.edge_diversity),
            _cell(_format_percent(candidate.intermediate_node_jaccard), candidate.intermediate_node_jaccard),
            _cell(" -> ".join(candidate.node_ids), " ".join(candidate.node_ids), css="nodes"),
            f"<td>{_path_thumbnail(graph, candidate.node_ids, positions)}</td>",
        ]
        rows.append("<tr>" + "".join(values) + "</tr>")
    return rows


def _cell(value: object, sort_value: object, css: str = "num") -> str:
    return (
        f'<td class="{html.escape(css, quote=True)}" '
        f'data-value="{html.escape(str(sort_value), quote=True)}">{html.escape(str(value))}</td>'
    )


def _format_percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _path_thumbnail(
    graph: nx.Graph,
    node_ids: list[str],
    positions: dict[object, object],
) -> str:
    """Render a small inline SVG using one shared deterministic graph layout."""
    width, height, pad = 180.0, 110.0, 7.0
    xs = [float(point[0]) for point in positions.values()]
    ys = [float(point[1]) for point in positions.values()]
    x_span = max(max(xs) - min(xs), 1e-12)
    y_span = max(max(ys) - min(ys), 1e-12)

    def point(node_id: object) -> tuple[float, float]:
        raw = positions[node_id]
        x_coord = pad + (float(raw[0]) - min(xs)) * (width - 2 * pad) / x_span
        y_coord = height - pad - (float(raw[1]) - min(ys)) * (height - 2 * pad) / y_span
        return x_coord, y_coord

    background = []
    background_edges = sorted(graph.edges, key=lambda edge: (str(edge[0]), str(edge[1])))[:300]
    for left, right in background_edges:
        x1, y1 = point(left)
        x2, y2 = point(right)
        background.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#8c9990" stroke-opacity=".12"/>')
    foreground = []
    for left, right in zip(node_ids[:-1], node_ids[1:]):
        x1, y1 = point(left)
        x2, y2 = point(right)
        foreground.append(f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" stroke="#b63b2b" stroke-width="2.6"/>')
    nodes = []
    for node_id in graph.nodes:
        x_coord, y_coord = point(node_id)
        on_path = node_id in node_ids
        radius = 3.2 if on_path else 1.25
        fill = "#b63b2b" if on_path else "#65766d"
        nodes.append(f'<circle cx="{x_coord:.2f}" cy="{y_coord:.2f}" r="{radius}" fill="{fill}"/>')
    return (
        f'<svg viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="Path graph thumbnail">'
        + "".join(background + foreground + nodes)
        + "</svg>"
    )
