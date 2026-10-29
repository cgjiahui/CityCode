#!/usr/bin/env python3
"""Validate frozen outputs and their numerical/evidence contracts on CPU."""
import argparse
from pathlib import Path

import numpy as np

from extract_elevated import digest, read_json, write_json


def validate(output: Path) -> dict:
    run = read_json(output / "RUN.json")
    if run.get("uses_ground_truth") is not False or run.get("scene_modified") is not False:
        raise ValueError("Output lineage or scene mutation contract failed")
    for name, expected in run["outputs"].items():
        path = (output/name).resolve()
        if not path.is_relative_to(output.resolve()) or digest(path) != expected["sha256"]:
            raise ValueError("Output checksum mismatch: " + name)
    model = read_json(output/"elevated_structure.json")
    audit = read_json(output/"source_audit.json")
    proposal = read_json(output/"placement_proposal.json")
    centerlines = read_json(output/"elevated_centerlines.geojson")
    uncertain = read_json(output/"uncertain_regions.geojson")
    c = model["config"]
    if len(centerlines["features"]) != run["segment_count"]:
        raise ValueError("Centerline feature inventory differs")
    if len(uncertain["features"]) != proposal["withheld_count"]:
        raise ValueError("Uncertainty inventory differs")
    maximum_grade = 0.
    profile_count = 0
    for segment in model["segments"]:
        p = segment["profile"]
        xyz = np.asarray([n["position_m"] for n in p])
        if not np.isfinite(xyz).all():
            raise ValueError("Nonfinite structure coordinate")
        ds = np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1)
        grades = np.abs(np.diff(xyz[:, 2]) / np.maximum(1e-9, ds))
        grade = float(grades.max())
        maximum_grade = max(maximum_grade, grade)
        if grade > c["max_grade"] + 2e-6:
            raise ValueError("Height curve exceeds grade bound")
        for node in p:
            views = node["support_sample_indices"]
            if len(set(views)) != node["support_views"] or not all(0 <= v < audit["sample_count"] for v in views):
                raise ValueError("Support sample inventory differs from unique-view counts")
        profile_count += len(p)
    ids = set()
    for row in proposal["placements"] + proposal["withheld_placements"]:
        if row["placement_id"] in ids:
            raise ValueError("Duplicate placement identifier")
        ids.add(row["placement_id"])
        if row.get("placement_source") != "predicted_elevated_structure" or not row.get("reference_asset_required"):
            raise ValueError("Placement lineage or abstract asset state changed")
        a, b = np.asarray(row["start_m"]), np.asarray(row["end_m"])
        if not np.isfinite([*a, *b, *row["scale"]]).all() or not all(v > 0 for v in row["scale"]):
            raise ValueError("Nonfinite or nonpositive module transform")
        if not np.allclose(row["position_m"], (a+b)*.5, atol=1.1e-6):
            raise ValueError("Deck top-center transform differs from endpoints")
        if row["status"] == "requires_asset_binding":
            if row["uncertainty_reasons"] or row["support"]["minimum_views"] < c["min_support_views"]:
                raise ValueError("Uncertain module entered bindable proposal")
        elif row["status"] != "withheld_uncertain" or not row["uncertainty_reasons"]:
            raise ValueError("Invalid uncertainty state")
    if proposal["placement_count"] != len(proposal["placements"]) or proposal["withheld_count"] != len(proposal["withheld_placements"]):
        raise ValueError("Placement totals differ")
    regularized_path = output/"regularized_placement_proposal.json"
    regularized_count = 0
    regularized_length = 0.
    if regularized_path.exists():
        regularized = read_json(regularized_path)
        rules = regularized["inference_rules"]
        source_segments = {segment["segment_id"]: segment for segment in model["segments"]}
        if regularized.get("uses_ground_truth") is not False or regularized.get("replaces_not_appends") != "placement_proposal.json":
            raise ValueError("Regularized proposal must remain explicit prediction-only replacement")
        for segment in regularized["continuous_segments"]:
            source = source_segments[segment["source_segment_id"]]["profile"]
            profile = segment["profile"]
            xyz = np.asarray([node["position_m"] for node in profile])
            distance = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xyz[:, :2], axis=0), axis=1))]
            if not np.isfinite(xyz).all() or distance[-1] < rules["minimum_continuous_length_m"]-1e-5:
                raise ValueError("Regularized span is nonfinite or too short")
            if np.max(np.abs(np.diff(xyz[:, 2])/np.diff(distance))) > rules["maximum_grade"]+2e-6:
                raise ValueError("Regularized span exceeds bounded grade")
            anchors = []
            indices = []
            for i, node in enumerate(profile):
                original = source[node["source_profile_index"]]
                indices.append(node["source_profile_index"])
                if not np.allclose(node["position_m"][:2], original["position_m"][:2], atol=1e-7):
                    raise ValueError("Regularization moved XY away from its prediction source")
                if original["supported_height_layers"] > 1:
                    raise ValueError("Regularization crossed an unresolved supported deck layer")
                if node["width_m"] > min(original["width_m"], rules["maximum_design_deck_width_m"])+1e-6:
                    raise ValueError("Design width exceeds semantic support")
                if node["height_anchor"]:
                    anchors.append(i)
                    if original["support_views"] < rules["minimum_anchor_views"] or original["height_mad_m"] > rules["maximum_anchor_height_mad_m"]:
                        raise ValueError("Regularized height anchor lacks independent support")
                    if original["observed_z_m"] < rules["ground_reference_z_m"]+rules["minimum_anchor_height_above_calibration_plane_m"]:
                        raise ValueError("Regularization lifted a ground-level mask without high-Z evidence")
                    if abs(node["position_m"][2]-original["observed_z_m"]) > rules["maximum_anchor_correction_m"]+1e-6:
                        raise ValueError("Regularization changed a trusted elevation anchor too far")
            if np.any(np.diff(indices) != 1) or len(anchors) < 3:
                raise ValueError("Regularized run skips source topology or lacks anchors")
            if np.max(np.diff(distance[anchors])) > rules["maximum_anchor_gap_m"]+1e-5:
                raise ValueError("Regularized run fills too long an unsupported elevation span")
            if max(distance[anchors[0]], distance[-1]-distance[anchors[-1]]) > rules["maximum_end_extrapolation_m"]+1e-5:
                raise ValueError("Regularized endpoint exceeds bounded extrapolation")
            regularized_length += distance[-1]
        regularized_count = len(regularized["placements"])
        if regularized_count != regularized["placement_count"]:
            raise ValueError("Regularized placement inventory mismatch")
        if abs(regularized_length-regularized["continuous_length_m"]) > .001:
            raise ValueError("Regularized continuous length inventory mismatch")
    return {"schema": "citycode.predicted_elevated_validation.v1", "status": "PASS",
            "uses_ground_truth": False, "output_hashes_verified": len(run["outputs"]),
            "consumed_source_hashes_verified_by_extractor": len(audit["consumed_files"]),
            "segment_count": run["segment_count"], "profile_samples": profile_count,
            "maximum_abs_grade": maximum_grade, "grade_limit": c["max_grade"],
            "bindable_abstract_modules": proposal["placement_count"],
            "withheld_modules": proposal["withheld_count"],
            "regularized_bindable_abstract_modules": regularized_count,
            "regularized_continuous_length_m": regularized_length,
            "actual_ue_assets_bound": False, "scene_modified": False,
            "scope": "Numerical consistency and prediction lineage; no external accuracy or GT evaluation claim"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = validate(args.output)
    write_json(args.output/"VALIDATION.json", result)
    print(result)
