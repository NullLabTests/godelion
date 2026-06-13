#!/usr/bin/env python3
"""
Godelion Analysis: Evolutionary Performance Tracking

Generates:
- Accuracy vs generation plot
- Archive size over time
- Performance distribution histogram
- CSV export of all metrics
"""
import argparse
import json
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAVE_MPL = True
except ImportError:
    HAVE_MPL = False


def load_metadata(output_dir: str) -> list[dict]:
    """Load all metadata from a Godelion run."""
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


def load_run_scores(output_dir: str, run_id: str) -> dict:
    """Load performance metrics for a single run."""
    meta_path = os.path.join(output_dir, run_id, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        return json.load(f)


def collect_all_scores(output_dir: str, metadata_entries: list[dict]) -> dict:
    """Collect all run scores across generations."""
    runs = {}
    for entry in metadata_entries:
        gen = entry.get("generation")
        for child_id in entry.get("children_compiled", []):
            meta = load_run_scores(output_dir, child_id)
            if meta and "overall_performance" in meta:
                runs[child_id] = {
                    "generation": gen,
                    "parent": meta.get("parent_commit", "unknown"),
                    "score": meta["overall_performance"].get("accuracy_score", 0),
                    "resolved": meta["overall_performance"].get("total_resolved_instances", 0),
                    "submitted": meta["overall_performance"].get("total_submitted_instances", 0),
                }
    return runs


def plot_performance(output_dir: str, runs: dict, metadata_entries: list[dict]):
    """Generate performance plots."""
    if not HAVE_MPL:
        print("matplotlib not installed. Install with: pip install matplotlib")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Accuracy by generation
    ax = axes[0, 0]
    gens = sorted(set(e["generation"] for e in metadata_entries))
    gen_scores = {g: [] for g in gens}
    for run_id, info in runs.items():
        gen_scores[info["generation"]].append(info["score"])

    gen_means = [np.mean(gen_scores[g]) if gen_scores[g] else 0 for g in gens]
    gen_maxs = [max(gen_scores[g]) if gen_scores[g] else 0 for g in gens]
    gen_mins = [min(gen_scores[g]) if gen_scores[g] else 0 for g in gens]

    ax.plot(gens, gen_means, "b-", label="Mean", linewidth=2)
    ax.plot(gens, gen_maxs, "g--", label="Best", linewidth=2)
    ax.fill_between(gens, gen_mins, gen_maxs, alpha=0.2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Accuracy")
    ax.set_title("Performance Over Generations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Archive size
    ax = axes[0, 1]
    archive_sizes = [len(e.get("archive", [])) for e in metadata_entries]
    ax.plot(gens, archive_sizes, "r-", linewidth=2)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Archive Size")
    ax.set_title("Archive Growth")
    ax.grid(True, alpha=0.3)

    # 3. Score histogram
    ax = axes[1, 0]
    scores = [info["score"] for info in runs.values()]
    if scores:
        ax.hist(scores, bins=20, alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_xlabel("Accuracy")
        ax.set_ylabel("Count")
        ax.set_title(f"Score Distribution ({len(scores)} runs)")
        ax.grid(True, alpha=0.3)

    # 4. Children per generation
    ax = axes[1, 1]
    children_counts = [len(e.get("children_compiled", [])) for e in metadata_entries]
    ax.bar(gens, children_counts, color="green", alpha=0.7)
    ax.set_xlabel("Generation")
    ax.set_ylabel("Children (compiled)")
    ax.set_title("Children per Generation")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "analysis_performance.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")
    plt.close()


def export_csv(output_dir: str, runs: dict):
    """Export run data to CSV."""
    csv_path = os.path.join(output_dir, "analysis_runs.csv")
    with open(csv_path, "w") as f:
        f.write("run_id,generation,parent,score,resolved,submitted\n")
        for run_id, info in sorted(runs.items()):
            f.write(f"{run_id},{info['generation']},{info['parent']},{info['score']:.4f},{info['resolved']},{info['submitted']}\n")
    print(f"Saved: {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="Godelion Performance Analysis")
    parser.add_argument("--output-dir", "-o", required=True, help="Godelion output directory")
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Output directory not found: {args.output_dir}")
        sys.exit(1)

    entries = load_metadata(args.output_dir)
    print(f"Loaded {len(entries)} generation entries")

    runs = collect_all_scores(args.output_dir, entries)
    print(f"Collected {len(runs)} run metrics")

    plot_performance(args.output_dir, runs, entries)
    export_csv(args.output_dir, runs)

    print("Analysis complete.")


if __name__ == "__main__":
    main()
