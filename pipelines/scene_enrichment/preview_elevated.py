#!/usr/bin/env python3
"""Render a source-labelled CPU diagnostic of the enrichment proposal."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

from extract_elevated import read_json


def preview(output: Path) -> Path:
    model = read_json(output/"elevated_structure.json")
    proposal = read_json(output/"placement_proposal.json")
    evidence = np.load(output/"prediction_evidence_grid.npz")
    z = evidence["height_m"]
    origin = model["grid"]["origin_xy_m"]
    cell = model["grid"]["cell_size_m"]
    extent = [origin[0], origin[0]+z.shape[1]*cell, origin[1], origin[1]+z.shape[0]*cell]
    fig, matrix = plt.subplots(2, 2, figsize=(16, 14), layout="constrained", facecolor="#fbfcfd")
    axes = matrix.ravel()
    for ax in axes:
        ax.set_facecolor("#eef1f4")
        ax.set_xlabel("UE X (m)")
        ax.set_ylabel("UE Y (m)")
        ax.set_aspect("equal")
    image = axes[0].imshow(np.ma.masked_invalid(z), origin="lower", extent=extent, vmin=0, vmax=25,
                           cmap="viridis", interpolation="nearest")
    fig.colorbar(image, ax=axes[0], shrink=.55, label="Predicted world Z (m); display clipped at 0 / 25")
    axes[0].set_title("Observed predicted elevated semantics + depth", loc="left", fontsize=12, weight="bold")
    support = evidence["elevated_view_count"]
    im2 = axes[1].imshow(np.ma.masked_where(support == 0, support), origin="lower", extent=extent,
                         cmap="cividis", vmin=1, vmax=4, interpolation="nearest")
    fig.colorbar(im2, ax=axes[1], shrink=.65, label="Independent views of selected height layer (4+ clipped)")
    axes[1].set_title("Class-4 support; one vote per view and XY cell", loc="left", fontsize=12, weight="bold")
    conflict = evidence["compatible_conflict_view_count"]
    quality = np.minimum(support/2., 1.) * np.exp(-evidence["height_mad_m"]/1.5) * support / np.maximum(1, support+conflict)
    im3 = axes[2].imshow(np.ma.masked_where(support == 0, quality), origin="lower", extent=extent,
                         cmap="RdYlGn", vmin=0, vmax=1, interpolation="nearest")
    fig.colorbar(im3, ax=axes[2], shrink=.65, label="Support x height agreement x compatible semantic agreement")
    axes[2].set_title("Evidence quality diagnostic; not calibrated probability", loc="left", fontsize=12, weight="bold")
    axes[3].imshow(np.isfinite(z), origin="lower", extent=extent, cmap="Greys", vmin=0, vmax=5, interpolation="nearest")
    lines = [np.asarray([p["position_m"][:2] for p in r["profile"]]) for r in model["segments"]]
    axes[3].add_collection(LineCollection(lines, colors="#64748b", linewidths=.65, alpha=.7))
    for key, color, label in [("withheld_placements", "#e59889", "withheld: uncertain evidence"),
                              ("placements", "#00845b", "eligible for real asset binding")]:
        rows = proposal[key]
        a = [np.asarray([p["start_m"][:2], p["end_m"][:2]]) for p in rows]
        if a:
            axes[3].add_collection(LineCollection(a, colors=color, linewidths=2.5 if key == "placements" else 1.2, label=f"{label} ({len(rows)})"))
    axes[3].set_title("Numerical placement proposal; scene unchanged", loc="left", fontsize=12, weight="bold")
    axes[3].legend(loc="upper left", framealpha=.97, fontsize=9)
    counts = Counter(reason for p in proposal["withheld_placements"] for reason in p["uncertainty_reasons"])
    detail = "; ".join(f"{reason.replace('_', ' ')}: {n}" for reason, n in counts.most_common(3))
    fig.suptitle(f"Prediction-only E09 replay  |  {len(model['segments'])} candidate segments  |  {proposal['placement_count']} abstract modules eligible\n"
                 "Historical calibrated predictions; explicit scale priors; no completed GT-supported heights or GT bridge transforms", fontsize=13, weight="bold")
    fig.supxlabel("Most frequent withholding reasons: " + detail, fontsize=9)
    target = output/"elevated_prediction_diagnostic.png"
    fig.savefig(target, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)
    return target


def preview_regularized(output: Path) -> Path:
    model = read_json(output/"elevated_structure.json")
    strict = read_json(output/"placement_proposal.json")
    proposal = read_json(output/"regularized_placement_proposal.json")
    fig, axes = plt.subplots(1, 2, figsize=(17, 8), layout="constrained", facecolor="#fbfcfd")
    for ax in axes:
        ax.set_facecolor("#eef1f4"); ax.set_aspect("equal")
        ax.set_xlabel("UE X (m)"); ax.set_ylabel("UE Y (m)")
        all_lines = [np.asarray([p["position_m"][:2] for p in r["profile"]]) for r in model["segments"]]
        ax.add_collection(LineCollection(all_lines, colors="#b3bdc8", linewidths=1.0))
    strict_lines = [np.asarray([p["start_m"][:2], p["end_m"][:2]]) for p in strict["placements"]]
    axes[0].add_collection(LineCollection(strict_lines, colors="#00845b", linewidths=3))
    axes[0].set_title(f"Strict observed baseline: {strict['placement_count']} eligible modules", loc="left", weight="bold")
    lines = [np.asarray([p["start_m"][:2], p["end_m"][:2]]) for p in proposal["placements"]]
    heights = [p["position_m"][2] for p in proposal["placements"]]
    collection = LineCollection(lines, array=np.asarray(heights), cmap="viridis", linewidths=4, clim=(3, 20))
    axes[1].add_collection(collection)
    axes[1].set_title(f"Explicit rule inference: {proposal['continuous_segment_count']} spans / {proposal['continuous_length_m']/1000:.2f} km", loc="left", weight="bold")
    fig.colorbar(collection, ax=axes[1], shrink=.55, label="Regularized predicted deck Z (m)")
    for ax in axes:
        ax.set_xlim(-610, 620); ax.set_ylim(-450, 220)
        ax.grid(color="#cbd5e1", linewidth=.4, alpha=.5)
    fig.suptitle("Elevated roads from extracted predictions: observed baseline and declared design inference\n"
                 "Repeated high-Z anchors; bounded short-gap fitting and grade; unresolved deck layers remain withheld", fontsize=13, weight="bold")
    fig.supxlabel("Alternative proposals, not additive layers. Native UE mesh joins and support pillars still require scene QA. Ground-level northern class-4 masks are not lifted.", fontsize=10)
    target = output/"elevated_regularization_comparison.png"
    fig.savefig(target, dpi=150, facecolor=fig.get_facecolor()); plt.close(fig)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    output = parser.parse_args().output
    print(preview(output))
    if (output/"regularized_placement_proposal.json").exists():
        print(preview_regularized(output))
