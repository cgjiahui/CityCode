"""Run once from the existing UE editor Python console; never starts an editor.

The adjacent .job.json supplies verified baseline files, a CPU apply plan, the
joined template manifest, geometry_snapshot and separate output paths. Bridge
resources are verified before SaveMap creates a new map. Only placement actors
are changed; generated building/background components stay in place.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def load_sibling(name):
    spec = importlib.util.spec_from_file_location("citycode_gui_repair_"+name, str(Path(__file__).with_name(name+".py")))
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def validate_output_paths(job):
    protected = {Path(job[key]).resolve() for key in ("baseline_manifest", "baseline_assembly_result", "baseline_readback_result",
        "geometry_snapshot", "apply_plan", "template_manifest")}
    manifest = Path(job["output_manifest"])
    outputs = [manifest, Path(job["output_json"]), manifest.with_suffix(".catalog.json"),
        Path(job.get("output_native_catalog", manifest.with_suffix(".native_catalog.json"))),
        Path(job.get("output_native_proposal", manifest.with_suffix(".native_proposal.json")))]
    resolved = [path.resolve() for path in outputs]
    if protected.intersection(resolved) or len(set(resolved)) != len(resolved):
        raise ValueError("Repair output paths must be distinct and must not overwrite any verified input")


def actor_name(path):
    return path.rsplit(":PersistentLevel.", 1)[-1]


def normalized_snapshot(snapshot, ignore_scene=False):
    result = {}
    for path, value in snapshot.items():
        row = copy.deepcopy(value)
        row["parent_actor_path"] = actor_name(row["parent_actor_path"]) if row.get("parent_actor_path") else None
        if ignore_scene:
            row["tags"] = [tag for tag in row["tags"] if not tag.startswith("CityCodeScene=")]
        key = actor_name(path)
        if key in result:
            raise ValueError("Actor names are ambiguous across levels")
        result[key] = row
    return result


def same_state(left, right, ignore_scene=False):
    a, b = normalized_snapshot(left, ignore_scene), normalized_snapshot(right, ignore_scene)
    if set(a) != set(b):
        return False
    for key in a:
        x, y = a[key], b[key]
        for field in ("mesh_path", "parent_actor_path", "tags"):
            if x[field] != y[field]:
                return False
        for field, tolerance in (("position_cm", .001), ("scale", 1e-7)):
            if max(abs(p-q) for p, q in zip(x[field], y[field])) > tolerance:
                return False
        if abs(abs(sum(p*q for p, q in zip(x["rotation_quat_xyzw"], y["rotation_quat_xyzw"])))-1) > 1e-7:
            return False
    return True


def equivalent_placement(a, b):
    for field in ("id", "mesh_path", "kind", "placement_source", "group_id"):
        if a.get(field) != b.get(field):
            return False
    return (max(abs(p-q) for p, q in zip(a["position_cm"], b["position_cm"])) <= 1e-6
        and max(abs(p-q) for p, q in zip(a["scale"], b["scale"])) <= 1e-9
        and abs(abs(sum(p*q for p, q in zip(a["rotation_quat_xyzw"], b["rotation_quat_xyzw"])))-1) <= 1e-9)


def bound_bridge(row):
    if (row.get("category") != "predicted_bridge_span" or row.get("prepared_world_geometry") is not True
            or row.get("placement_source") != "predicted_elevated_structure"
            or row.get("rotation_deg", [0, 0, 0]) != [0, 0, 0] or row.get("scale", [1, 1, 1]) != [1, 1, 1]):
        raise ValueError("Actual bridge import did not preserve its anchor-only contract")
    result = {"id": row["placement_id"], "kind": row["category"], "mesh_path": row["mesh_path"],
        "position_cm": [float(v)*100 for v in row["position_m"]], "rotation_quat_xyzw": [0., 0., 0., 1.],
        "scale": [1., 1., 1.], "placement_source": row["placement_source"], "source_segment_id": None,
        "pivot_rule": "proposal_position_is_mesh_pivot", "proposal_position_m": row["position_m"], "catalog_mesh_verified": True}
    for field in ("prepared_world_geometry", "logical_module_count", "expected_native_bounds_m", "native_material_paths", "template_sha256", "inference"):
        result[field] = row.get(field)
    return result


def finalize_manifest(baseline_path, output, plan, plan_sha, imported):
    baseline_path, output = Path(baseline_path), Path(output)
    result = copy.deepcopy(read(baseline_path))
    placements = copy.deepcopy(plan["bound_native_placements"]) + [bound_bridge(row) for row in imported["placement_proposal"]["placements"]]
    if len({row["id"] for row in placements}) != len(placements):
        raise ValueError("Candidate placements have duplicate identities")
    scene = hashlib.sha256(json.dumps({"baseline": digest(baseline_path), "plan": plan_sha, "placements": placements}, sort_keys=True).encode()).hexdigest()[:12]
    for row in result["meshes"]:
        row["import_file"] = os.path.relpath(baseline_path.parent/row["import_file"], output.parent)
        for part in row.get("import_parts", []):
            part["file"] = os.path.relpath(baseline_path.parent/part["file"], output.parent)
    result["calibration"]["path"] = os.path.relpath(baseline_path.parent/result["calibration"]["path"], output.parent)
    result.update(scene_id=scene, new_level_path="/Game/Research/CityCodeOpenSource/Maps/CityCode_"+scene+"_ElevatedRepair",
        placements=placements, proposal_sources=plan["proposal_sources"], proposal_decisions=plan["proposal_decisions"],
        candidate_baseline_manifest_sha256=digest(baseline_path), candidate_apply_plan_sha256=plan_sha,
        candidate_template_manifest_sha256=plan["template_manifest_sha256"], ue_executed=False)
    result["proposal_sources"].append({"schema": imported["placement_proposal"]["schema"],
        "template_manifest_sha256": plan["template_manifest_sha256"], "actual_UE_import_verified": True})
    result["counts"].update(proposal_bound=len(placements), proposal_withheld=len(plan["proposal_decisions"]), proposal_unresolved=0)
    result["proposal_category_counts"] = dict(Counter(row["kind"] for row in placements))
    result["logical_bridge_module_count"] = sum(row.get("logical_module_count", 0) for row in placements)
    return result


def current_level(unreal):
    return unreal.EditorLevelLibrary.get_editor_world().get_path_name().split(".", 1)[0]


def save_candidate_copy(unreal, entry, old_level, level, old_file, old_file_sha, before):
    """UE 5.0 can save a duplicate while retaining the original editor world."""
    if not unreal.EditorLoadingAndSavingUtils.save_map(unreal.EditorLevelLibrary.get_editor_world(), level):
        raise RuntimeError("Could not save the city as a separate candidate map")
    actual_old_sha = digest(old_file)
    if actual_old_sha != old_file_sha:
        raise RuntimeError("SaveMap changed baseline bytes: before=%s after=%s current=%s" % (old_file_sha, actual_old_sha, current_level(unreal)))
    retained_old_world = current_level(unreal) == old_level
    if current_level(unreal) != level:
        if not retained_old_world:
            raise RuntimeError("SaveMap selected an unexpected world: " + current_level(unreal))
        new_file = Path(unreal.Paths.project_content_dir())/(level[len("/Game/"):]+".umap")
        if not new_file.is_file() or not new_file.stat().st_size:
            raise RuntimeError("SaveMap retained the original world and did not write the candidate map")
        # Load only our newly saved copy. The original map is still untouched.
        if not unreal.EditorLevelLibrary.load_level(level):
            raise RuntimeError("Could not open the newly saved candidate copy")
    if current_level(unreal) != level or digest(old_file) != old_file_sha:
        raise RuntimeError("Candidate load changed the baseline file or selected the wrong map")
    if not same_state(before, entry.tagged_geometry_snapshot()):
        raise ValueError("Saved candidate copy changed tagged geometry before any actor edits")
    return {"save_map_retained_original_world": retained_old_world, "explicit_candidate_load": retained_old_world,
        "copied_tagged_geometry_verified": True, "baseline_umap_sha256_unchanged": old_file_sha}


def validate_current(entry, manifest, audited, reference):
    import unreal
    actual = entry.tagged_geometry_snapshot()
    if not same_state(reference, actual):
        raise ValueError("Current GUI has changed tagged geometry since the verified saved snapshot")
    records = {row["id"]: row for row in manifest["meshes"]+manifest["placements"]}
    counts = Counter()
    expected_mesh_paths = {row["input_id"]: set(row["mesh_paths"]) for row in audited["records"]}
    source_counts = Counter()
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        tags = dict(str(tag).split("=", 1) for tag in actor.get_editor_property("tags") if "=" in str(tag))
        identity = tags.get("CityCodeInput")
        if identity is None:
            continue
        if identity not in records or tags.get("CityCodeScene") != manifest["scene_id"]:
            raise ValueError("Foreign or unexpected tagged geometry in current GUI")
        row = records[identity]; counts[identity] += 1
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        mesh = component.get_editor_property("static_mesh") if component else None
        if not mesh or mesh.get_path_name() not in expected_mesh_paths[identity]:
            raise ValueError("Current mesh identity differs: "+identity)
        if "prepared" in row:
            sha = unreal.EditorAssetLibrary.get_metadata_tag(mesh, "CityCodeInterchangeSHA256")
            parts = row.get("import_parts") or [{"sha256": row["prepared"]["interchange_sha256"], "mesh_count": row["prepared"]["mesh_count"]}]
            if sha not in {part["sha256"] for part in parts} or not mesh.get_path_name().startswith(entry.shared_destination(identity, sha)+"/"):
                raise ValueError("Current generated asset source/digest namespace differs: "+identity)
            source_counts[(identity, sha)] += 1
    for identity, row in records.items():
        expected = row["prepared"]["mesh_count"] if "prepared" in row else 1
        if counts[identity] != expected:
            raise ValueError("Current actor coverage differs: "+identity)
        if "prepared" in row:
            parts = row.get("import_parts") or [{"sha256": row["prepared"]["interchange_sha256"], "mesh_count": row["prepared"]["mesh_count"]}]
            if any(source_counts[(identity, part["sha256"])] != part["mesh_count"] for part in parts):
                raise ValueError("Current generated multi-part coverage differs: "+identity)
    return actual


def apply(job):
    import unreal
    entry = load_sibling("ue_entry")
    bound_job = {"assembly_manifest": job["baseline_manifest"], "assembly_result": job["baseline_assembly_result"],
        "readback_result": job["baseline_readback_result"]}
    # Interactive mode never loads a map. A separately guarded headless owner
    # can opt in to load the same verified baseline in its own editor process.
    _, _, _, baseline_sha, _, old_level, audited, baseline = entry.load_verified_saved_map(bound_job, load_level=job.get("headless") is True)
    if not audited.get("inspection_lighting_checked") or not audited.get("generated_normals_checked"):
        raise ValueError("Baseline must have explicit independent lighting and generated-normal checks")
    if current_level(unreal) != old_level:
        raise ValueError("Open the exact verified baseline map; this script never loads or closes a user's map")
    plan_path = Path(job["apply_plan"]); plan_sha = digest(plan_path); plan = read(plan_path)
    if (plan.get("schema") != "citycode.elevated_repair_gui_plan.v1" or plan.get("baseline_manifest_sha256") != baseline_sha
            or plan.get("baseline_scene_id") != baseline["scene_id"] or digest(job["template_manifest"]) != plan["template_manifest_sha256"]):
        raise ValueError("CPU candidate plan does not bind this exact baseline and joined deck")
    snapshot_file = read(job["geometry_snapshot"])
    if (snapshot_file.get("assembly_manifest_sha256") != baseline_sha or snapshot_file.get("new_level_path") != old_level
            or not snapshot_file.get("geometry_unchanged")):
        raise ValueError("Geometry snapshot is not bound to the verified baseline")
    reference = snapshot_file["tagged_geometry_snapshot"]
    registry = unreal.AssetRegistryHelpers.get_asset_registry(); registry.search_all_assets(True)
    if registry.is_loading_assets():
        raise RuntimeError("Asset registry must finish before repair import")
    before = validate_current(entry, baseline, audited, reference)
    immutable = {path: state for path, state in before.items()
        if any(tag.startswith("CityCodeEntityRoot=") for tag in state["tags"])
        or any(tag == "CityCodeInput="+row["id"] for row in baseline["meshes"] for tag in state["tags"])}
    output = Path(job["output_manifest"])
    if output.resolve() == Path(job["baseline_manifest"]).resolve() or output.exists():
        raise ValueError("Candidate manifest output must be new and separate")
    for record in plan["bound_native_placements"]:
        if not isinstance(unreal.load_asset(record["mesh_path"]), unreal.StaticMesh):
            raise ValueError("Candidate native mesh is unavailable: "+record["mesh_path"])
    imported = entry.import_bridge_template({"template_manifest": job["template_manifest"]})
    if not imported.get("bounds_verified") or not imported.get("native_materials_verified") or imported["count"] != plan["template_count"]:
        raise ValueError("Actual joined bridge import is incomplete")
    if not same_state(before, entry.tagged_geometry_snapshot()):
        raise ValueError("Resource import unexpectedly changed city actors")
    candidate = finalize_manifest(job["baseline_manifest"], output, plan, plan_sha, imported)
    if job.get("new_level_path"):
        requested = job["new_level_path"]
        if not requested.startswith(entry.CONTENT_ROOT+"/Maps/") or requested == old_level:
            raise ValueError("Candidate override must be a new map in the dedicated research Maps folder")
        candidate["new_level_path"] = requested
    merged_assets = {row["mesh_path"]: row for row in plan["native_catalog_assets"]}
    merged_assets.update({row["mesh_path"]: row for row in imported["assets"]})
    catalog_file = output.with_suffix(".catalog.json")
    if catalog_file.exists():
        raise ValueError("Candidate catalog output must be new and separate")
    write(catalog_file, {"schema": "citycode.ue_local_asset_catalog.v1", "assets": list(merged_assets.values()),
        "baseline_catalog_sha256": plan["native_catalog_sha256"], "template_manifest_sha256": plan["template_manifest_sha256"],
        "new_bridge_assets_verified_in_UE": True})
    candidate["catalog_sha256"] = digest(catalog_file)
    old, new = {row["id"]: row for row in baseline["placements"]}, {row["id"]: row for row in candidate["placements"]}
    changed = {key for key in old.keys() & new.keys() if not equivalent_placement(old[key], new[key])}
    removed, added = set(old)-set(new), set(new)-set(old)
    level = candidate["new_level_path"]
    if unreal.EditorAssetLibrary.does_asset_exist(level):
        raise ValueError("Refusing to overwrite an existing candidate map: "+level)
    old_file = Path(unreal.Paths.project_content_dir())/(old_level[len("/Game/"):]+".umap")
    old_file_sha = digest(old_file)
    copy_state = save_candidate_copy(unreal, entry, old_level, level, old_file, old_file_sha, before)
    target_actors = list(unreal.EditorLevelLibrary.get_all_level_actors())
    for actor in target_actors:
        tags = [str(tag) for tag in actor.get_editor_property("tags")]
        identity = next((tag.split("=", 1)[1] for tag in tags if tag.startswith("CityCodeInput=")), None)
        if identity in removed | changed:
            if not unreal.EditorLevelLibrary.destroy_actor(actor):
                raise RuntimeError("Cannot remove old candidate placement: "+identity)
            continue
        if "CityCodeScene="+baseline["scene_id"] in tags:
            actor.set_editor_property("tags", [unreal.Name("CityCodeScene="+candidate["scene_id"] if tag == "CityCodeScene="+baseline["scene_id"] else tag) for tag in tags])
    for identity in sorted(added | changed):
        record = new[identity]; mesh = unreal.load_asset(record["mesh_path"])
        actor = entry.spawn(mesh, identity, record["kind"], record["placement_source"],
            entry.make_transform(record["position_cm"], record["rotation_quat_xyzw"], record["scale"]), candidate["scene_id"])
        if record.get("group_id"):
            actor.set_editor_property("tags", list(actor.get_editor_property("tags"))+[unreal.Name("CityCodePropGroup="+record["group_id"])])
    after = entry.tagged_geometry_snapshot()
    stable_names = {actor_name(path) for path in immutable}
    retained = {path: state for path, state in after.items() if actor_name(path) in stable_names}
    if not same_state(immutable, retained, ignore_scene=True):
        raise ValueError("Generated building/background geometry changed; candidate map was not saved after edits")
    counts = Counter(); actors = []
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        tags = dict(str(tag).split("=", 1) for tag in actor.get_editor_property("tags") if "=" in str(tag))
        identity = tags.get("CityCodeInput")
        if not identity:
            continue
        if tags.get("CityCodeScene") != candidate["scene_id"]:
            raise ValueError("Candidate contains foreign scene tags")
        counts[identity] += 1
        mesh = actor.get_component_by_class(unreal.StaticMeshComponent).get_editor_property("static_mesh")
        if identity in new:
            expected = new[identity]; transform = actor.get_actor_transform()
            if (mesh.get_path_name() != expected["mesh_path"] or tags.get("CityCodePlacementSource") != expected["placement_source"]
                    or tags.get("CityCodeKind") != expected["kind"]
                    or max(abs(a-b) for a, b in zip(entry.vector(transform.translation), expected["position_cm"])) > .001
                    or max(abs(a-b) for a, b in zip(entry.vector(transform.scale3d), expected["scale"])) > 1e-6
                    or abs(abs(sum(a*b for a, b in zip([transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w],
                        expected["rotation_quat_xyzw"])))-1) > 1e-6
                    or (expected.get("group_id") and tags.get("CityCodePropGroup") != expected["group_id"])):
                raise ValueError("Candidate placement did not receive its exact mesh/transform/provenance: "+identity)
        actors.append({"actor_path": actor.get_path_name(), "input_id": identity, "mesh_path": mesh.get_path_name(), "kind": tags["CityCodeKind"]})
    for row in candidate["meshes"]+candidate["placements"]:
        if counts.pop(row["id"], 0) != (row["prepared"]["mesh_count"] if "prepared" in row else 1):
            raise ValueError("Candidate actor count differs: "+row["id"])
    if counts:
        raise ValueError("Unexpected candidate actor IDs")
    if (digest(plan_path) != plan_sha or digest(job["baseline_manifest"]) != baseline_sha
            or digest(job["template_manifest"]) != plan["template_manifest_sha256"]):
        raise ValueError("Repair inputs changed before commit")
    if not unreal.EditorLevelLibrary.save_current_level() or digest(old_file) != old_file_sha:
        raise RuntimeError("Candidate save failed or baseline map changed")
    write(output, candidate)
    write(job.get("output_native_catalog", output.with_suffix(".native_catalog.json")), imported)
    write(job.get("output_native_proposal", output.with_suffix(".native_proposal.json")), imported["placement_proposal"])
    return {"schema": "citycode.ue_assembly_result.v1", "status": "assembled_pending_independent_readback", "ue_executed": True,
        "engine_version": unreal.SystemLibrary.get_engine_version(), "new_level_path": level, "scene_id": candidate["scene_id"],
        "assembly_manifest_sha256": digest(output), "mesh_input_count": len(candidate["meshes"]), "actor_count": len(actors), "actors": actors,
        "building_entity_root_count": audited["building_entity_root_count"], "calibration": read(job["baseline_assembly_result"])["calibration"],
        "inspection_environment": audited["inspection_environment"], "bridge_actor_world_transforms_used": False,
        "incremental_repair": {"baseline_level": old_level, "baseline_manifest_sha256": baseline_sha, "baseline_umap_sha256_unchanged": old_file_sha,
            "save_copy_validation": copy_state,
            "apply_plan_sha256": plan_sha, "generated_geometry_unchanged": True, "added_placements": sorted(added),
            "removed_placements": sorted(removed), "replaced_placements": sorted(changed), "fresh_independent_readback_required": True}}


def main():
    import unreal
    path = Path(os.environ.get("UE_CITYCODE_REPAIR_JOB", str(Path(__file__).with_suffix(".job.json"))))
    job = read(path)
    validate_output_paths(job)
    try:
        result = apply(job)
    except Exception as error:
        result = {"status": "failed", "ue_executed": True, "error": str(error), "traceback": traceback.format_exc(),
            "scope": "only dedicated candidate resources/map; baseline map is not intentionally saved"}
        write(job["output_json"], result)
        unreal.log_error(result["traceback"])
        raise
    write(job["output_json"], result)
    unreal.log("CityCode candidate saved; fresh independent readback remains required: "+result["new_level_path"])
    # -ExecutePythonScript owns headless shutdown after the task returns.
    # Explicitly requesting another exit can cause a second editor teardown.
    # The GUI console path simply returns and leaves the editor open.


if __name__ == "__main__":
    main()
