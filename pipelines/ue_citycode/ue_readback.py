"""Independent persisted-map audit, run in a NEW UE Editor process.

UE_CITYCODE_JOB JSON: assembly_manifest, assembly_result, output_json. Checks the
saved map against CPU geometry/proposal expectations, not spawn-call return values.
It never loads or reads the original SmallCity level or its bridge actors.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import traceback

import unreal

CONTENT_ROOT = "/Game/Research/CityCodeOpenSource"


def vector(value):
    return [float(value.x), float(value.y), float(value.z)]


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def inspect_saved_lighting(actors):
    """Read persisted environment properties; never create or change actors."""
    issues = []
    def select_actor(class_name, label):
        actor_class = getattr(unreal, class_name, ())
        candidates = [actor for actor in actors if isinstance(actor, actor_class)]
        return next((actor for actor in candidates if actor.get_actor_label() == label), candidates[0] if candidates else None)
    def component_values(actor, component_class, properties, key):
        if actor is None:
            issues.append(key + ":missing_actor")
            return {"present": False}
        component = actor.get_component_by_class(getattr(unreal, component_class))
        result = {"present": True, "actor_path": actor.get_path_name(), "actor_label": actor.get_actor_label(),
                  "component_present": component is not None}
        if component is None:
            issues.append(key + ":missing_component")
            return result
        mobility = component.get_editor_property("mobility")
        result["mobility"] = str(mobility)
        result["movable"] = mobility == unreal.ComponentMobility.MOVABLE
        if not result["movable"]:
            issues.append(key + ":mobility_not_movable")
        for name, expected, tolerance in properties:
            value = component.get_editor_property(name)
            result[name] = value
            matches = (abs(float(value)-expected) <= tolerance) if tolerance is not None else value == expected
            if not matches:
                issues.append(key + ":" + name + "_mismatch")
        return result
    atmosphere_class = getattr(unreal, "SkyAtmosphere", ())
    atmosphere = [actor.get_path_name() for actor in actors if isinstance(actor, atmosphere_class)]
    if not atmosphere:
        issues.append("sky_atmosphere:missing_actor")
    sun = select_actor("DirectionalLight", "CityCode_InspectionSun")
    sky = select_actor("SkyLight", "CityCode_InspectionSkyLight")
    environment = {"sky_atmosphere_present": bool(atmosphere), "sky_atmosphere_actor_paths": atmosphere,
        "sun": component_values(sun, "DirectionalLightComponent", [
            ("intensity", 50000., .01), ("atmosphere_sun_light", True, None),
            ("atmosphere_sun_light_index", 0, None)], "sun"),
        "skylight": component_values(sky, "SkyLightComponent", [
            ("intensity", 1., 1e-5), ("real_time_capture", True, None),
            ("lower_hemisphere_is_black", False, None)], "skylight")}
    return environment, issues


def inspect_generated_normal_requirement(mesh, requirement, subsystem):
    """Read saved build settings; never rebuild or save during verification."""
    path = mesh.get_path_name()
    result = {"mesh_path": path, **requirement}
    issues = []
    if not path.startswith(CONTENT_ROOT + "/Imported/"):
        issues.append("normal_policy_mesh_outside_generated_namespace")
    if requirement["recompute_normals_required"]:
        actual = bool(subsystem.get_lod_build_settings(mesh, 0).get_editor_property("recompute_normals"))
        result["actual_lod0_recompute_normals"] = actual
        if not actual:
            issues.append("missing_source_normals_not_recomputed")
    return result, issues


def readback(job):
    manifest_path = Path(job["assembly_manifest"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assembled = json.loads(Path(job["assembly_result"]).read_text(encoding="utf-8-sig"))
    if assembled.get("status") != "assembled_pending_independent_readback" or not assembled.get("ue_executed"):
        raise ValueError("No successful assembly result to inspect")
    if assembled["assembly_manifest_sha256"] != digest_file(manifest_path):
        raise ValueError("Assembly manifest changed after import")
    level = assembled["new_level_path"]
    if not level.startswith(CONTENT_ROOT + "/Maps/"):
        raise ValueError("Refusing to inspect a level outside the dedicated research root")
    if not unreal.EditorLevelLibrary.load_level(level):
        raise RuntimeError("Cannot load saved assembled level")
    by_id, foreign, errors, details, entity_roots = {}, [], [], [], {}
    all_actors = list(unreal.EditorLevelLibrary.get_all_level_actors())
    lighting_checked, environment = False, None
    if job.get("verify_inspection_lighting", False):
        environment, lighting_issues = inspect_saved_lighting(all_actors)
        lighting_checked = True
        if lighting_issues:
            errors.append({"inspection_lighting": lighting_issues})
    for actor in all_actors:
        tags = {}
        for tag in actor.get_editor_property("tags"):
            key, sep, value = str(tag).partition("=")
            if sep:
                tags[key] = value
        if "CityCodeEntityRoot" in tags:
            entity_roots.setdefault(tags["CityCodeEntityRoot"], []).append(actor)
        if "CityCodeInput" not in tags:
            continue
        if tags.get("CityCodeScene") != manifest["scene_id"]:
            foreign.append(actor.get_path_name())
        by_id.setdefault(tags["CityCodeInput"], []).append((actor, tags))
    expected = {r["id"]: r for r in manifest["meshes"] + manifest["placements"]}
    for identity in sorted(set(by_id) - set(expected)):
        errors.append("Unexpected tagged input: " + identity)
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    normals_checked = bool(job.get("verify_generated_mesh_normals", False))
    normal_checks, normal_requirements, policy = [], {}, None
    if normals_checked:
        spec = importlib.util.spec_from_file_location("citycode_readback_normal_policy", str(Path(__file__).with_name("normal_policy.py")))
        policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(policy)
        for record in manifest["meshes"]:
            if record["kind"] != "background":
                continue
            parts = record.get("import_parts") or [{"file": record["import_file"], "sha256": record["prepared"]["interchange_sha256"]}]
            for part in parts:
                path = manifest_path.parent / part["file"]
                if digest_file(path) != part["sha256"]:
                    raise ValueError("Background normal-policy source changed after import")
                normal_requirements[part["sha256"]] = policy.read_requirements(path)
    for identity, record in expected.items():
        found = by_id.get(identity, [])
        is_import = "prepared" in record
        expected_count = record["prepared"]["mesh_count"] if is_import else 1
        issues, low, high, meshes, part_counts = [], [float("inf")]*3, [-float("inf")]*3, [], {}
        if len(found) != expected_count:
            issues.append("actor_count:%d!=%d" % (len(found), expected_count))
        for actor, tags in found:
            if tags.get("CityCodePlacementSource") != record["placement_source"]:
                issues.append("placement_source_mismatch")
            if record.get("group_id") and tags.get("CityCodePropGroup") != record["group_id"]:
                issues.append("native_prop_group_mismatch")
            if record.get("building_entity_id"):
                parent = actor.get_attach_parent_actor()
                parent_tags = [str(t) for t in parent.get_editor_property("tags")] if parent else []
                if "CityCodeEntityRoot=" + record["building_entity_id"] not in parent_tags:
                    issues.append("building_entity_parent_mismatch")
            if record["kind"] in ("bridge_deck", "elevated_deck", "predicted_bridge_span", "bridge_support", "bridge_railing") and tags.get("CityCodePlacementSource") != "predicted_elevated_structure":
                issues.append("bridge_does_not_have_prediction_provenance")
            component = actor.get_component_by_class(unreal.StaticMeshComponent)
            mesh = component.get_editor_property("static_mesh") if component else None
            if not mesh:
                issues.append("missing_static_mesh"); continue
            meshes.append(mesh.get_path_name())
            slots = mesh.get_editor_property("static_materials")
            if not slots or any(slot.get_editor_property("material_interface") is None for slot in slots):
                issues.append("missing_material_slot")
            if is_import:
                source_hash = unreal.EditorAssetLibrary.get_metadata_tag(mesh, "CityCodeInterchangeSHA256")
                expected_parts = record.get("import_parts") or [{"sha256": record["prepared"]["interchange_sha256"], "mesh_count": record["prepared"]["mesh_count"]}]
                if source_hash not in {part["sha256"] for part in expected_parts}:
                    issues.append("interchange_source_hash_mismatch")
                if normals_checked and record["kind"] == "background" and source_hash in normal_requirements:
                    requirement = policy.mesh_requirement(normal_requirements[source_hash], mesh.get_path_name())
                    normal_state, normal_issues = inspect_generated_normal_requirement(mesh, requirement, subsystem)
                    normal_checks.append({"input_id": identity, "source_interchange_sha256": source_hash, **normal_state})
                    issues.extend(normal_issues)
                part_counts[source_hash] = part_counts.get(source_hash, 0)+1
                if record.get("nanite") and not subsystem.get_nanite_settings(mesh).get_editor_property("enabled"):
                    issues.append("building_nanite_disabled")
                origin, extent = actor.get_actor_bounds(False, False)
                origin, extent = vector(origin), vector(extent)
                for i in range(3):
                    low[i] = min(low[i], (origin[i]-extent[i])/100)
                    high[i] = max(high[i], (origin[i]+extent[i])/100)
            else:
                if mesh.get_path_name() != record["mesh_path"]:
                    issues.append("proposal_mesh_path_mismatch")
                transform = actor.get_actor_transform()
                position, scale = vector(transform.translation), vector(transform.scale3d)
                if max(abs(a-b) for a, b in zip(position, record["position_cm"])) > .1:
                    issues.append("position_error_over_0.1cm")
                if max(abs(a-b) for a, b in zip(scale, record["scale"])) > 1e-4:
                    issues.append("scale_mismatch")
                q = transform.rotation; actual_q = [q.x, q.y, q.z, q.w]
                dot = abs(sum(a*b for a, b in zip(actual_q, record["rotation_quat_xyzw"])))
                if abs(dot-1) > 1e-5:
                    issues.append("rotation_mismatch")
                if record.get("prepared_world_geometry"):
                    actual_materials = [slot.get_editor_property("material_interface").get_path_name()
                                        if slot.get_editor_property("material_interface") else None for slot in slots]
                    if actual_materials != record["native_material_paths"]:
                        issues.append("native_bridge_material_slots_mismatch")
                    if unreal.EditorAssetLibrary.get_metadata_tag(mesh, "CityCodeInterchangeSHA256") != record["template_sha256"]:
                        issues.append("native_bridge_template_hash_mismatch")
                    box = mesh.get_bounding_box(); native_bounds = [vector(box.min), vector(box.max)]
                    if max(abs(native_bounds[a][i]-record["expected_native_bounds_m"][a][i]*100) for a in range(2) for i in range(3)) > .2:
                        issues.append("native_bridge_deformed_geometry_bounds_mismatch")
        bound_error = None
        actual_bounds = None
        if is_import and found and all(math.isfinite(v) for v in low+high):
            actual_bounds = [low, high]
            bound_error = max(abs(actual_bounds[a][i]-record["prepared"]["world_bounds_m"][a][i]) for a in range(2) for i in range(3))
            if bound_error > float(job.get("bounds_tolerance_m", .025)):
                issues.append("world_bounds_error_over_tolerance")
            expected_parts = record.get("import_parts") or [{"sha256": record["prepared"]["interchange_sha256"], "mesh_count": record["prepared"]["mesh_count"]}]
            if any(part_counts.get(part["sha256"], 0) != part["mesh_count"] for part in expected_parts):
                issues.append("background_or_mesh_part_source_count_mismatch")
        if issues:
            errors.append({"input_id": identity, "issues": issues})
        details.append({"input_id": identity, "kind": record["kind"], "actor_count": len(found), "mesh_paths": meshes,
                        "actual_world_bounds_m": actual_bounds, "maximum_bounds_error_m": bound_error, "import_part_source_counts": part_counts, "issues": issues})
    if foreign:
        errors.append({"foreign_scene_actors": foreign})
    expected_entities = {r["building_entity_id"] for r in manifest["meshes"] if r.get("building_entity_id")}
    for identity in sorted(expected_entities | set(entity_roots)):
        if identity not in expected_entities or len(entity_roots.get(identity, [])) != 1:
            errors.append({"building_entity_root_count_mismatch": identity})
    return {"schema": "citycode.ue_independent_readback.v1", "status": "passed" if not errors else "failed",
            "ue_executed": True, "independent_saved_map_readback": True, "new_level_path": level,
            "engine_version": unreal.SystemLibrary.get_engine_version(), "expected_input_count": len(expected),
            "assembly_manifest_sha256": digest_file(manifest_path), "assembly_result_sha256": digest_file(job["assembly_result"]),
            "source_binding_sha256": manifest["source_binding_sha256"], "scene_id": manifest["scene_id"],
            "actual_tagged_actor_count": sum(len(v) for v in by_id.values()), "errors": errors, "records": details,
            "inspection_lighting_checked": lighting_checked, "inspection_environment": environment,
            "generated_normals_checked": normals_checked, "generated_normal_check_count": len(normal_checks),
            "generated_normal_checks": normal_checks,
            "generated_normal_check_scope": "generated background only; provided building/massing normals and native Epic meshes are not changed",
            "building_entity_root_count": len(entity_roots),
            "bridge_actor_world_transforms_used": False,
            "scope": "persisted mesh/material/Nanite/source/bounds/proposal transforms; engine traffic simulation and visual review separate"}


def main():
    job_path = os.environ.get("UE_CITYCODE_JOB")
    if not job_path:
        raise RuntimeError("Set UE_CITYCODE_JOB to an absolute JSON job path")
    job = json.loads(Path(job_path).read_text(encoding="utf-8-sig"))
    result = None
    try:
        result = readback(job); write_json(job["output_json"], result)
        if result["errors"]:
            raise RuntimeError("Independent saved-map readback failed; see " + job["output_json"])
    except Exception as error:
        if result is None:
            write_json(job["output_json"], {"status": "failed", "error": str(error), "traceback": traceback.format_exc(), "ue_executed": True,
                "inspection_lighting_checked": False, "inspection_environment": None,
                "generated_normals_checked": False, "generated_normal_check_count": 0, "generated_normal_checks": []})
        raise


if __name__ == "__main__":
    main()
