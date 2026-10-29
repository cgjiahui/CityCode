"""Building asset preparation, inference adapters, and portable delivery checks."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path, PurePosixPath
import re

import numpy as np


ACCEPTED = {"accepted_reuse", "accepted_generated", "accepted_composite",
            "accepted_structure", "accepted_repaired", "accepted_scene_completion"}
POSTPROCESSED = {"accepted_structure", "accepted_repaired", "resolved_fragment"}
EXCLUDED = {"excluded_incomplete", "excluded_noise", "excluded_existing_building"}
PENDING = {"pending_generation", "pending_review", "pending_lod_export",
           "pending_registration", "pending_appearance", "failed"}
LOD_TARGETS = {"0.25M": 250000, "0.1M": 100000}
SUPPORTED_LOD_TARGETS = {**LOD_TARGETS, "1M": 1000000}


def _matrix(value, size, name):
    matrix = np.asarray(value, dtype=float)
    if matrix.shape != (size, size) or not np.isfinite(matrix).all():
        raise ValueError(f"{name} must be a finite {size} x {size} matrix")
    if size == 4:
        if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-10, rtol=0):
            raise ValueError(f"{name} must be an affine column-vector transform")
        if abs(float(np.linalg.det(matrix[:3, :3]))) <= 1e-12:
            raise ValueError(f"{name} is singular")
    return matrix


def _asset_path(root, value):
    if not isinstance(value, str) or not value or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError("Asset paths must be relative POSIX paths")
    relative = PurePosixPath(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Asset paths must remain inside the manifest directory")
    resolved = (root / value).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Asset symlink escapes the manifest directory")
    if not resolved.is_file():
        raise FileNotFoundError(value)
    return resolved


def _policy(manifest):
    supplied = manifest.get("validation_policy", {})
    result = {"minimum_footprint_iou": .70, "minimum_target_coverage": .80,
              "minimum_asset_coverage": .80, "desired_raster_cell_m": .10,
              "maximum_raster_dimension": 2048, "vertical_relative_tolerance": .01,
              "vertical_absolute_tolerance_m": .05, "vertical_maximum_tolerance_m": .20}
    for key, value in supplied.items():
        if key in result:
            number = float(value)
            if not math.isfinite(number) or number <= 0:
                raise ValueError(f"Invalid validation policy: {key}")
            if key.startswith("minimum_"):
                if number < result[key] or number > 1:
                    raise ValueError(f"Validation policy cannot lower the standard gate: {key}")
            elif key == "desired_raster_cell_m":
                if number > .10:
                    raise ValueError("Validation raster cannot be coarser than 0.10 metres")
            elif key in ("vertical_relative_tolerance", "vertical_absolute_tolerance_m", "vertical_maximum_tolerance_m"):
                if number > result[key]:
                    raise ValueError(f"Validation policy cannot relax vertical tolerance: {key}")
            elif key == "maximum_raster_dimension" and (number < 2048 or number != int(number)):
                raise ValueError("Validation raster dimension must be an integer of at least 2048")
            result[key] = int(number) if key == "maximum_raster_dimension" else number
    return result


def _validate_groups(manifest, rows):
    """Shared physical placements require explicit, complete group membership."""
    from shapely.geometry import shape
    from shapely.ops import unary_union

    groups = manifest.get("asset_instance_groups", [])
    if isinstance(groups, dict):
        groups = [{"id": key, **value} for key, value in groups.items()]
    if not isinstance(groups, list):
        raise ValueError("Asset instance groups must be a list or ID-keyed object")
    declared, owners = {}, {}
    for group in groups:
        gid, members = group.get("id"), group.get("building_ids")
        if not isinstance(gid, str) or not gid or gid in declared:
            raise ValueError("Asset group IDs must be nonempty and unique")
        if not isinstance(members, list) or len(members) < 2 or len(members) != len(set(members)):
            raise ValueError("An asset group needs at least two unique member IDs")
        if any(sid in owners for sid in members):
            raise ValueError("A source building cannot belong to multiple physical asset groups")
        declared[gid] = group
        owners.update({sid: gid for sid in members})
    by_id = {row["building_id"]: row for row in rows}
    for gid, group in declared.items():
        members = group["building_ids"]
        if set(members) - set(by_id):
            raise ValueError(f"Asset group {gid} has missing members")
        grouped_rows = [by_id[sid] for sid in members]
        if any(row.get("asset_instance_group_id") != gid for row in grouped_rows):
            raise ValueError(f"Asset group {gid} membership does not match its building records")
        accepted = [row for row in grouped_rows if row.get("status") in ACCEPTED]
        if accepted:
            if len(accepted) != len(grouped_rows):
                raise ValueError(f"Asset group {gid} mixes accepted and unfinished members")
            world = _matrix(group["world_from_local_4x4"], 4, "group world_from_local_4x4")
            union = unary_union([shape(row["source_target_geometry"]) for row in grouped_rows])
            if union.is_empty or not union.is_valid or union.area <= 0:
                raise ValueError(f"Asset group {gid} has invalid source footprints")
            for row in grouped_rows:
                if row.get("lods") != group.get("lods") or not np.allclose(row["world_from_local_4x4"], world, atol=1e-8, rtol=0):
                    raise ValueError(f"Asset group {gid} members disagree on whole-asset LODs or placement")
                if shape(row["target_geometry"]).symmetric_difference(union).area > max(.01, union.area * 1e-8):
                    raise ValueError(f"Asset group {gid} target is not the source footprint union")
    placements = {}
    for row in rows:
        sid, gid = row["building_id"], row.get("asset_instance_group_id")
        if gid is not None and owners.get(sid) != gid:
            raise ValueError(f"Building {sid} refers to an undeclared or inconsistent asset group")
        if row.get("status") in ACCEPTED:
            world = _matrix(row["world_from_local_4x4"], 4, "world_from_local_4x4")
            if not set(LOD_TARGETS).issubset(row.get("lods", {})):
                continue  # The per-asset check reports missing whole-building LODs.
            # Content identity survives copied files and relative path aliases.
            # Reusing a template at a different placement remains legitimate.
            lod_hashes = tuple(row["lods"][label]["sha256"] for label in LOD_TARGETS)
            for other_gid, other_world in placements.get(lod_hashes, []):
                if np.allclose(world, other_world, atol=1e-8, rtol=0) and (gid is None or gid != other_gid):
                    raise ValueError("Duplicate physical asset placement requires one explicit shared group")
            placements.setdefault(lod_hashes, []).append((gid, world))
    return len(declared)


def _validate_asset_instances(manifest, rows):
    """The optional assembly table instantiates each accepted physical asset once."""
    if "asset_instances" not in manifest:
        return None
    instances = manifest["asset_instances"]
    if not isinstance(instances, list):
        raise ValueError("Asset instances must be a list")
    accepted = {}
    for row in rows:
        if row.get("status") in ACCEPTED:
            identity = row.get("asset_instance_id")
            if not isinstance(identity, str) or not identity:
                raise ValueError("Accepted building needs an asset_instance_id when assembly table is present")
            group = row.get("asset_instance_group_id")
            if group is not None and identity != group:
                raise ValueError("Shared group members must use their group ID as the physical asset_instance_id")
            accepted.setdefault(identity, []).append(row)
    seen = set()
    for instance in instances:
        identity = instance.get("id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Physical asset instance IDs must be nonempty and unique")
        seen.add(identity)
        if identity not in accepted or instance.get("status") not in ACCEPTED:
            raise ValueError("Assembly table contains an asset without accepted building membership")
        members = accepted[identity]
        sources = instance.get("source_building_ids")
        if (not isinstance(sources, list) or len(sources) != len(set(sources))
                or set(sources) != {row["building_id"] for row in members}):
            raise ValueError("Physical asset sources do not exactly cover its accepted building rows")
        if len(members) > 1:
            groups = {row.get("asset_instance_group_id") for row in members}
            if len(groups) != 1 or None in groups:
                raise ValueError("A physical asset with multiple source IDs requires an explicit shared group")
        world = _matrix(instance["world_from_local_4x4"], 4, "asset instance world_from_local_4x4")
        if any(instance.get("lods") != row.get("lods")
               or not np.array_equal(world, np.asarray(row["world_from_local_4x4"])) for row in members):
            raise ValueError("Assembly instance LODs or placement differ from accepted building rows")
    if seen != set(accepted):
        raise ValueError("Assembly table does not contain every accepted physical asset exactly once")
    return len(instances)


def _validate_building_entities(manifest):
    """Every active mesh part belongs to exactly one logical building entity."""
    if "building_entities" not in manifest:
        return None
    entities = manifest["building_entities"]
    if not isinstance(entities, list):
        raise ValueError("Building entities must be a list")
    active = {r["id"]: r for r in manifest.get("asset_instances", [])}
    seen, identities = set(), set()
    for entity in entities:
        identity, parts = entity.get("id"), entity.get("asset_instance_ids")
        if not isinstance(identity, str) or not identity or identity in identities:
            raise ValueError("Building entity IDs must be nonempty and unique")
        identities.add(identity)
        if not isinstance(parts, list) or not parts or len(parts) != len(set(parts)):
            raise ValueError("Building entity parts must be nonempty and unique")
        if set(parts) - set(active) or set(parts) & seen:
            raise ValueError("Building entity has inactive or multiply owned mesh parts")
        sources = entity.get("source_building_ids")
        expected = {sid for aid in parts for sid in active[aid]["source_building_ids"]}
        if not isinstance(sources, list) or len(sources) != len(set(sources)) or set(sources) != expected:
            raise ValueError("Building entity source membership differs from its mesh parts")
        if any(active[aid].get("building_entity_id") != identity for aid in parts):
            raise ValueError("Mesh part building_entity_id differs from entity membership")
        seen.update(parts)
    if seen != set(active):
        raise ValueError("Building entities must cover every active mesh part exactly once")
    return len(entities)


def _validate_accepted(root, record, policy, *, geometry):
    visual_completion = record.get('status') == 'accepted_scene_completion'
    if visual_completion:
        from .scene_completion import validate_scene_completion_review
        validate_scene_completion_review(root, record)
    return _validate_candidate_lods(root,record,policy,geometry=geometry,visual_completion=visual_completion)


def _validate_candidate_lods(root, record, policy, *, geometry, visual_completion=False):
    """Check actual files and geometry; this function never approves inclusion.

    A visual candidate still requires its separate source and scene review in
    _validate_accepted before it can enter a complete delivery.
    """
    from shapely.geometry import shape
    from .lod import inspect_glb
    from .registration import read_glb_scene_geometry, raster_triangle_footprint, overlap_metrics

    if record.get("local_coordinate_system") != "glTF Y-up":
        raise ValueError("Accepted assets must declare local_coordinate_system='glTF Y-up'")
    world = _matrix(record["world_from_local_4x4"], 4, "world_from_local_4x4")
    local = _matrix(record["building_local_from_asset_4x4"], 4, "building_local_from_asset_4x4")
    building = _matrix(record["world_from_building_local_4x4"], 4, "world_from_building_local_4x4")
    if not np.allclose(building @ local, world, atol=1e-7, rtol=1e-9):
        raise ValueError("Building-to-world and asset-to-building transforms do not compose")
    normals = _matrix(record["normal_transform_world_3x3"], 3, "normal_transform_world_3x3")
    if not np.allclose(normals, np.linalg.inv(world[:3, :3]).T, atol=1e-8, rtol=1e-8):
        raise ValueError("World normal transform is not the inverse transpose")
    target = shape(record["target_geometry"])
    if target.geom_type not in ("Polygon", "MultiPolygon") or not target.is_valid or target.area <= 0:
        raise ValueError("Accepted asset needs a valid current target footprint")
    base, top = float(record["base_z_m"]), float(record["top_z_m"])
    if not all(math.isfinite(v) for v in (base, top)) or top <= base:
        raise ValueError("Accepted asset needs finite positive target height")
    lods = record.get("lods", {})
    if not set(LOD_TARGETS).issubset(lods) or set(lods)-set(SUPPORTED_LOD_TARGETS):
        raise ValueError("Accepted asset requires whole-building 0.25M and 0.1M GLBs, with optional 1M")
    results, bounds = {}, []
    for label, target_faces in SUPPORTED_LOD_TARGETS.items():
        if label not in lods:
            continue
        row = lods[label]
        path = _asset_path(root, row["path"])
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError(f"Invalid SHA256 for {label}")
        inspection = inspect_glb(path)
        if inspection["sha256"] != row["sha256"] or inspection["bytes"] != row.get("bytes"):
            raise ValueError(f"File bytes or SHA256 mismatch for {label}")
        scene = read_glb_scene_geometry(path)
        scene_faces = len(scene["faces"])
        if scene_faces != row.get("faces") or scene_faces > target_faces * 1.02:
            raise ValueError(f"Actual scene triangle count mismatch or target exceeded for {label}")
        overlap, raster = None, None
        if geometry:
            footprint, raster = raster_triangle_footprint(scene["vertices"], scene["faces"], world,
                cell_m=policy["desired_raster_cell_m"], max_dimension=policy["maximum_raster_dimension"])
            overlap = overlap_metrics(footprint, target)
            gates = {"iou": policy["minimum_footprint_iou"],
                     "target_coverage": policy["minimum_target_coverage"],
                     "asset_coverage": policy["minimum_asset_coverage"]}
            if any(abs(overlap[key] - gate) <= .02 for key, gate in gates.items()):
                footprint, raster = raster_triangle_footprint(scene["vertices"], scene["faces"], world,
                    cell_m=min(.05, policy["desired_raster_cell_m"]),
                    max_dimension=max(4096, policy["maximum_raster_dimension"]))
                overlap = overlap_metrics(footprint, target)
            # A reviewed visual completion may fill previously unobserved
            # wings. Report source agreement without treating a partial source
            # outline as an exact whole-building template. Coarse association
            # still needs substantial support in both directions.
            if visual_completion:
                failures = [key for key in ('target_coverage', 'asset_coverage') if overlap[key] < .20]
            else:
                failures = [key for key, gate in gates.items() if overlap[key] < gate]
            if failures:
                raise ValueError(f"Actual {label} mesh footprint fails {', '.join(failures)}")
        vertices = scene["vertices"] @ world[:3, :3].T + world[:3, 3]
        tolerance = min(policy["vertical_maximum_tolerance_m"],
                        max(policy["vertical_absolute_tolerance_m"],
                            policy["vertical_relative_tolerance"] * (top - base)))
        vertical = {"actual_base_z_m": float(vertices[:, 2].min()),
                    "actual_top_z_m": float(vertices[:, 2].max()), "tolerance_m": tolerance}
        if abs(vertical["actual_base_z_m"] - base) > tolerance or abs(vertical["actual_top_z_m"] - top) > tolerance:
            raise ValueError(f"Actual {label} mesh does not fit the predicted base/top")
        bounds.append([scene["vertices"].min(axis=0), scene["vertices"].max(axis=0)])
        results[label] = {"faces": scene_faces, "encoded_mesh_faces": inspection["faces"],
                          "sha256": inspection["sha256"],
                          "actual_mesh_overlap": overlap, "projection": raster, "vertical": vertical}
    bounds = np.asarray(bounds)
    span = np.maximum(bounds[0, 1] - bounds[0, 0], 1e-8)
    if np.any(np.abs(bounds[1:] - bounds[0]) > span * .025):
        raise ValueError("LOD geometry changed the shared local frame by more than 2.5 percent")
    return results


def validate_assets(manifest_path: Path, *, require_complete=False, geometry=False) -> dict:
    """Validate actual GLBs against the unchanged current structure.

    Provenance may reference a separately delivered structure package. Only
    owned asset paths are opened, and every GLB must stay inside this package.
    Drafts can be inspected; completeness is an explicit stricter requirement.
    """
    manifest_path = Path(manifest_path).resolve()
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "citycode.building_asset_bindings.v1":
        raise ValueError("Unsupported building asset binding schema")
    if manifest.get("path_base") != "manifest_directory":
        raise ValueError("Binding paths must be relative to the manifest directory")
    source = manifest["source_structure"]
    count = source["instance_count"]
    if not isinstance(count, int) or count < 0:
        raise ValueError("Source instance_count must be a nonnegative integer")
    if not re.fullmatch(r"[0-9a-f]{64}", source.get("manifest_sha256", "")):
        raise ValueError("Source structure manifest SHA256 is required")
    expected_ids = source.get("building_ids")
    if expected_ids is not None:
        if (not isinstance(expected_ids, list) or any(not isinstance(s, str) or not s for s in expected_ids)
                or len(set(expected_ids)) != len(expected_ids) or len(expected_ids) != count):
            raise ValueError("Source building_ids must be a unique full instance list")
    rows = manifest.get("buildings")
    if not isinstance(rows, list):
        raise ValueError("Binding buildings must be a list")
    ids = [row.get("building_id") for row in rows]
    if any(not isinstance(s, str) or not s for s in ids) or len(ids) != len(set(ids)):
        raise ValueError("Binding building IDs must be nonempty and unique")
    if expected_ids is not None and set(ids) - set(expected_ids):
        raise ValueError("Bindings include IDs absent from the source structure")
    group_count = _validate_groups(manifest, rows)
    physical_count = _validate_asset_instances(manifest, rows)
    entity_count = _validate_building_entities(manifest)
    from .massing_completion import validate_massing
    massing_check=validate_massing(manifest_path.parent,manifest,geometry=geometry)
    if any(row.get('status') == 'accepted_scene_completion' for row in rows):
        from .scene_completion import validate_scene_completion_context
        validate_scene_completion_context(manifest)
    from .postprocess_validation import validate_surface_instances
    surface_count = validate_surface_instances(manifest, rows)
    missing = sorted(set(expected_ids or []) - set(ids))
    errors, pending, validated, surface_records, statuses = [], [], [], [], Counter()
    policy = _policy(manifest)
    for row in rows:
        sid, status = row["building_id"], row.get("status")
        statuses[status] += 1
        try:
            postprocess_check = None
            if status in POSTPROCESSED:
                from .postprocess_validation import validate_postprocess_record
                postprocess_check = validate_postprocess_record(manifest_path.parent, row, geometry=geometry)
            if status in ACCEPTED:
                value = {"building_id": sid, "lods": _validate_accepted(manifest_path.parent, row, policy, geometry=geometry)}
                if postprocess_check is not None:
                    value["postprocess"] = postprocess_check
                validated.append(value)
            elif status == "resolved_fragment":
                surface_records.append({"building_id": sid, "postprocess": postprocess_check,
                    "lods": _validate_accepted(manifest_path.parent, row, policy, geometry=geometry)})
            elif status in EXCLUDED:
                if not isinstance(row.get("reason"), str) or not row["reason"].strip() or not row.get("evidence"):
                    raise ValueError("Exclusion needs an explicit reason and observed evidence")
                if row.get("lods"):
                    raise ValueError("An excluded instance cannot carry accepted asset LODs")
                if row.get("postprocess", {}).get("method") == "retain_fragment_evidence":
                    from .postprocess_validation import validate_evidence_only_record
                    validate_evidence_only_record(manifest_path.parent, row)
            elif status in PENDING:
                pending.append(sid)
            else:
                raise ValueError(f"Unknown asset state: {status}")
        except (ValueError, KeyError, TypeError, IndexError, FileNotFoundError, OSError) as exc:
            errors.append({"building_id": sid, "message": str(exc)})
    complete = not errors and not pending and len(ids) == count and not missing and expected_ids is not None
    if require_complete:
        if expected_ids is None:
            errors.append({"building_id": None, "message": "Complete delivery requires source building_ids"})
        if len(ids) != count or missing:
            errors.append({"building_id": None, "message": "Binding list does not cover every source instance", "missing_ids": missing})
        if pending:
            errors.append({"building_id": None, "message": "Delivery contains pending or failed instances", "pending_ids": pending})
    return {"schema": "citycode.asset_validation.v1", "status": "passed" if not errors else "failed",
            "complete": complete, "source_instances": count, "binding_instances": len(ids),
            "status_counts": dict(statuses), "pending_ids": pending, "missing_ids": missing,
            "asset_instance_groups": group_count,
            "physical_asset_instances": physical_count,
            "massing_completion": massing_check,
            "building_entities": entity_count,
            "surface_attachment_instances": surface_count,
            "validated_surface_sources": len(surface_records), "surface_records": surface_records,
            "validated_assets": len(validated), "records": validated, "errors": errors,
            "actual_glb_geometry_checked": bool(geometry), "asset_paths_checked": True,
            "structure_modified": False, "reference_geometry_read": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    validate = commands.add_parser("validate-assets", help="Check portable binding data and both actual GLBs")
    validate.add_argument("--manifest", type=Path, required=True)
    validate.add_argument("--require-complete", action="store_true")
    validate.add_argument("--geometry", action="store_true", help="Recompute actual footprint overlap for both LODs")
    validate.add_argument("--output", type=Path)
    validate.add_argument("--verbose", action="store_true", help="Print full per-building validation details")
    prepare = commands.add_parser("prepare-appearance", help="Prepare four references locally")
    prepare.add_argument("--building", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--provider-config", type=Path, help="Non-secret native appearance service configuration")
    complete = commands.add_parser("complete-appearance", help="Send the four prepared references to the configured HTTPS service")
    complete.add_argument("--appearance", type=Path, required=True)
    complete.add_argument("--provider-config", type=Path, help="Same non-secret configuration used during preparation")
    complete.add_argument("--timeout-seconds", type=int)
    preflight = commands.add_parser("generation-preflight", help="Check configured local model resources on CPU")
    preflight.add_argument("--config", type=Path, required=True)
    infer = commands.add_parser("generate", help="Generate a decoded mesh on an already allocated CUDA worker")
    infer.add_argument("--config", type=Path, required=True)
    infer.add_argument("--building", type=Path, required=True)
    infer.add_argument("--appearance", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    export = commands.add_parser("export-lods", help="Bake the requested LODs from a decoded mesh on a CUDA worker")
    export.add_argument("--raw-cache", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--master-glb", type=Path)
    export.add_argument("--texture-size", type=int, default=2048)
    export.add_argument("--targets", type=int, nargs="+", choices=[100000, 250000, 1000000],
                        default=[250000, 100000], help="Triangle budgets; add 1000000 for the 1M version")
    sanitize = commands.add_parser("sanitize-metadata", help="Remove private paths from completed LOD delivery sidecars")
    sanitize.add_argument("--assets", type=Path, required=True, help="Asset package containing the assets/ directory")
    sanitize.add_argument("--output", type=Path)
    postprocess = commands.add_parser("postprocess-assets", help="Apply reviewed CPU repairs, source structural replacements and observed surface attachments")
    postprocess.add_argument("--manifest", type=Path, required=True)
    postprocess.add_argument("--decisions", type=Path, required=True)
    postprocess.add_argument("--structure-root", type=Path, required=True)
    postprocess.add_argument("--output", type=Path, required=True)
    postprocess.add_argument("--report", type=Path)
    review = commands.add_parser("apply-scene-review", help="Apply recorded inclusion, deferral and building ownership decisions")
    review.add_argument("--manifest", type=Path, required=True)
    review.add_argument("--decisions", type=Path, required=True)
    review.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay-scene-completion", help="Replay frozen Agent-reviewed completion decisions using published cached models, without API/GPU")
    replay.add_argument("--assets-root", type=Path, required=True)
    replay.add_argument("--decisions", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    replay.add_argument("--report", type=Path)
    args = parser.parse_args(argv)
    if args.command == "replay-scene-completion":
        from .completion_replay import replay_scene_completion
        try:
            report = replay_scene_completion(args.assets_root, args.decisions, args.output)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            report = {"status": "failed", "message": str(exc)}
        if args.report:
            from .appearance import write_json
            write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    if args.command == "apply-scene-review":
        from .scene_selection import apply_scene_review
        try:
            report = apply_scene_review(args.manifest, args.decisions, args.output)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            print(json.dumps({"status": "failed", "message": str(exc)}, ensure_ascii=False))
            return 1
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    if args.command == "postprocess-assets":
        from .postprocess import postprocess_assets
        try:
            report = postprocess_assets(args.manifest, args.decisions, args.structure_root, args.output)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            report = {"schema": "citycode.asset_postprocess_failure.v1", "status": "failed", "message": str(exc)}
            if args.report:
                from .appearance import write_json
                write_json(args.report, report)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 1
        if args.report:
            from .appearance import write_json
            write_json(args.report, report)
        print(json.dumps({key: value for key, value in report.items() if key != "records"}, ensure_ascii=False, indent=2))
        return 0
    if args.command == "sanitize-metadata":
        from .metadata import sanitize_completed_sidecars
        report = sanitize_completed_sidecars(args.assets)
        if args.output:
            from .appearance import write_json
            write_json(args.output, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    if args.command == "validate-assets":
        try:
            report = validate_assets(args.manifest, require_complete=args.require_complete, geometry=args.geometry)
        except (ValueError, TypeError, KeyError, OSError) as exc:
            report = {"schema": "citycode.asset_validation.v1", "status": "failed", "complete": False,
                      "errors": [{"building_id": None, "message": str(exc)}],
                      "pending_ids": [], "missing_ids": [], "records": []}
        if args.output:
            from .appearance import write_json
            write_json(args.output, report)
        summary = report if args.verbose else {key: value for key, value in report.items()
                   if key not in ("records", "surface_records", "pending_ids", "missing_ids")}
        if not args.verbose:
            summary.update(pending_instances=len(report["pending_ids"]), missing_instances=len(report["missing_ids"]))
            summary["error_count"] = len(report["errors"])
            summary["errors"] = []
            for error in report["errors"][:10]:
                preview = {key: value for key, value in error.items() if key in ("building_id", "message")}
                for field in ("pending_ids", "missing_ids"):
                    if field in error:
                        preview[field.replace("_ids", "_instances")] = len(error[field])
                summary["errors"].append(preview)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if report["status"] == "passed" else 1
    if args.command == "prepare-appearance":
        from .appearance import prepare_building
        config = json.loads(args.provider_config.read_text()) if args.provider_config else None
        report = prepare_building(args.building, args.output, provider_config=config)
        print(json.dumps({"building_id": report["building_id"], "status": report["status"],
                          "input_identity_sha256": report["input_identity_sha256"]}))
    elif args.command == "complete-appearance":
        from .appearance import complete_building
        try:
            config = json.loads(args.provider_config.read_text()) if args.provider_config else None
            report = complete_building(args.appearance, provider_config=config, timeout_seconds=args.timeout_seconds)
            print(json.dumps({key: report[key] for key in ("building_id", "status", "model", "quality", "output_sha256")}))
            return 0
        except (ValueError, RuntimeError, OSError) as exc:
            print(json.dumps({"status": "failed", "exception": type(exc).__name__, "message": str(exc)}))
            return 1
    elif args.command == "generation-preflight":
        from .generation import preflight_resources
        report = preflight_resources(json.loads(args.config.read_text()))
        print(json.dumps({key: value for key, value in report.items() if key != "files"}, indent=2))
    elif args.command == "generate":
        from .generation import GenerationRuntime
        runtime = GenerationRuntime(json.loads(args.config.read_text()))
        report = runtime.generate(args.building, args.appearance, args.output)
        print(json.dumps(report, indent=2))
    elif args.command == "export-lods":
        from .lod import export_decoded_lods
        report = export_decoded_lods(args.raw_cache, args.output, master_glb=args.master_glb,
                                     texture_size=args.texture_size, targets=tuple(args.targets))
        print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
