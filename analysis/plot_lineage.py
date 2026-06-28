#!/usr/bin/env python3
"""
Godelion Analysis: Lineage Tree Visualization

Generates a DOT/Graphviz visualization of the evolutionary lineage.
"""

import argparse
import json
import os
import sys


def load_metadata(output_dir: str) -> list[dict]:
    metadata_path = os.path.join(output_dir, "dgm_metadata.jsonl")
    if not os.path.exists(metadata_path):
        print(f"Metadata not found: {metadata_path}")
        sys.exit(1)

    entries = []
    with open(metadata_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_lineage(output_dir: str, entries: list[dict]) -> dict:
    """Build a lineage tree from metadata."""
    nodes = {"initial": {"generation": -1, "score": None, "parent": None, "children": []}}

    for entry in entries:
        gen = entry.get("generation")
        for child_id in entry.get("children_compiled", []):
            meta_path = os.path.join(output_dir, child_id, "metadata.json")
            if not os.path.exists(meta_path):
                continue
            with open(meta_path) as f:
                meta = json.load(f)

            parent = meta.get("parent_commit", "initial")
            score = meta.get("overall_performance", {}).get("accuracy_score", None)

            if child_id not in nodes:
                nodes[child_id] = {"generation": gen, "score": score, "parent": parent, "children": []}
            if parent in nodes:
                nodes[parent]["children"].append(child_id)

    return nodes


def generate_dot(nodes: dict, output_path: str):
    """Generate Graphviz DOT file."""
    lines = [
        "digraph GodelionLineage {",
        "  rankdir=LR;",
        '  graph [fontname="monospace", fontsize=12];',
        '  node [fontname="monospace", fontsize=10, shape=box, style=filled, fillcolor=lightyellow];',
        '  edge [fontname="monospace", fontsize=9];',
        "",
    ]

    for node_id, info in nodes.items():
        label = node_id[:12] + "..." if len(node_id) > 12 else node_id
        if info["score"] is not None:
            label += f"\\n{info['score']:.2%}"
        if node_id == "initial":
            lines.append(f'  "{node_id}" [label="{label}", fillcolor=lightgreen];')
        else:
            color = "lightblue" if info["score"] and info["score"] > 0.5 else "lightyellow"
            lines.append(f'  "{node_id}" [label="{label}", fillcolor={color}];')

    lines.append("")

    for node_id, info in nodes.items():
        if info["parent"] and info["parent"] in nodes:
            lines.append(f'  "{info["parent"]}" -> "{node_id}";')

    lines.append("}")
    lines.append("")

    dot_path = output_path + ".dot"
    with open(dot_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {dot_path}")

    # Try to render with graphviz
    try:
        import graphviz

        src = graphviz.Source("\n".join(lines))
        src.render(output_path, format="png", cleanup=True)
        print(f"Saved: {output_path}.png")
    except ImportError:
        print("Install graphviz Python package for PNG rendering: pip install graphviz")
        print(f"Or render manually: dot -Tpng {dot_path} -o {output_path}.png")


def main():
    parser = argparse.ArgumentParser(description="Godelion Lineage Visualization")
    parser.add_argument("--output-dir", "-o", required=True, help="Godelion output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Output directory not found: {args.output_dir}")
        sys.exit(1)

    entries = load_metadata(args.output_dir)
    nodes = build_lineage(args.output_dir, entries)
    print(f"Built lineage with {len(nodes)} nodes")

    out_path = os.path.join(args.output_dir, "lineage")
    generate_dot(nodes, out_path)


if __name__ == "__main__":
    main()
