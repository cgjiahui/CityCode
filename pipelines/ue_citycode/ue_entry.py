"""Run inside UE 5.0.3 Editor, using UE_CITYCODE_JOB=<absolute JSON path>.

Actions: preflight, export_catalog, export_mesh_geometry, assemble. Readback uses ue_readback.py in a
separate editor execution. This module never reads original map actor transforms.
"""
from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import re
import sys
import time
import traceback

import unreal

CONTENT_ROOT = "/Game/Research/CityCodeOpenSource"


def normal_policy():
    spec = importlib.util.spec_from_file_location("citycode_source_normal_policy", str(Path(__file__).with_name("normal_policy.py")))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def vector(value):
    return [float(value.x), float(value.y), float(value.z)]


def bounds(mesh):
    box = mesh.get_bounding_box()
    return [vector(box.min), vector(box.max)]


def digest_file(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def preflight(job):
    classes = ["AssetImportTask", "GLTFImportFactory", "GLTFImportOptions", "EditorAssetLibrary",
               "EditorLevelLibrary", "StaticMeshEditorSubsystem"]
    available = {name: hasattr(unreal, name) for name in classes}
    distance_fields = None
    if hasattr(unreal.SystemLibrary, "get_console_variable_int_value"):
        distance_fields = unreal.SystemLibrary.get_console_variable_int_value("r.GenerateMeshDistanceFields")
    spline_class = getattr(unreal, "SplineMeshComponent", None)
    return {"schema": "citycode.ue_preflight.v1", "engine_version": unreal.SystemLibrary.get_engine_version(),
            "project_file": unreal.Paths.get_project_file_path(), "python_version": sys.version,
            "required_plugin_capabilities": available, "ready": all(available.values()),
            "required_plugins": ["PythonScriptPlugin", "EditorScriptingUtilities", "GLTFImporter"],
            "runtime_generate_mesh_distance_fields": distance_fields,
            "spline_mesh_capabilities": {"SplineMeshActor": hasattr(unreal, "SplineMeshActor"),
                "SplineMeshComponent": spline_class is not None,
                **{name: bool(spline_class and hasattr(spline_class, name)) for name in
                   ("set_start_and_end", "set_start_scale", "set_end_scale", "set_forward_axis", "get_start_position", "get_end_position", "get_start_tangent", "get_end_tangent")}},
            "original_actor_transforms_read": False}


def export_catalog(job):
    registry = unreal.AssetRegistryHelpers.get_asset_registry()
    registry.scan_paths_synchronous(["/Game"], force_rescan=False)
    patterns = job.get("category_patterns", {
        "bridge_deck": r"bridge|overpass|highway|freeway|flyover|viaduct",
        "vehicle": r"vehicle|sedan|hatchback|suv|pickup|car_|parked|truck|van_",
        "streetlight": r"street.?light|lamp.?post|light.?pole",
        "traffic_light": r"traffic.?light|traffic.?signal",
        "tree": r"tree|maple|oak|palm",
        "bench": r"bench",
        "bin": r"trash|rubbish|waste|garbage|bin_",
    })
    explicit = set(job.get("mesh_paths", []))
    entries, errors = [], []
    assets = registry.get_assets_by_class("StaticMesh", search_sub_classes=True)
    selected = []
    for asset in assets:
        path = str(asset.object_path)
        if not path.startswith("/Game/") or path.startswith(CONTENT_ROOT + "/"):
            continue
        categories = [category for category, pattern in patterns.items() if re.search(pattern, path, re.I)]
        if categories or path in explicit:
            selected.append((path, categories))
    found = {path for path, _ in selected}
    for path in sorted(explicit - found):
        selected.append((path, ["explicit_request"]))
    limit = int(job.get("max_assets", 3000))
    selected.sort(key=lambda item: (item[0] not in explicit, "bridge_deck" not in item[1], item[0]))
    for asset_index, (path, categories) in enumerate(selected[:limit]):
        try:
            mesh = unreal.load_asset(path)
            if not isinstance(mesh, unreal.StaticMesh):
                raise ValueError("Requested path is not a StaticMesh")
            box = bounds(mesh)
            dimensions = [box[1][i] - box[0][i] for i in range(3)]
            materials = []
            for slot in mesh.get_editor_property("static_materials"):
                material = slot.get_editor_property("material_interface")
                materials.append({"slot_name": str(slot.get_editor_property("material_slot_name")),
                                  "material_path": material.get_path_name() if material else None})
            length_axis = "X" if dimensions[0] >= dimensions[1] else "Y"
            entries.append({"asset_id": path, "mesh_path": path, "categories": categories,
                "bounds_cm": box, "dimensions_cm": dimensions, "materials": materials,
                "local_axes": {"length_axis": length_axis, "up_axis": "Z", "inference": "largest_local_XY_bounding_extent"},
                "orientation_confirmed": False, "local_pivot_cm": [0, 0, 0],
                "local_top_center_cm": [(box[0][0]+box[1][0])/2, (box[0][1]+box[1][1])/2, box[1][2]],
                "placement_transform": None, "actor_world_transform_read": False})
        except Exception as error:
            errors.append({"mesh_path": path, "error": str(error)})
        if (asset_index+1) % 64 == 0:
            unreal.SystemLibrary.collect_garbage()
    return {"schema": "citycode.ue_local_asset_catalog.v1", "engine_version": unreal.SystemLibrary.get_engine_version(),
        "units": "cm", "coordinate_system": "mesh local UE XYZ; no placed-actor world transforms",
        "count": len(entries), "candidate_count": len(selected), "truncated": len(selected) > limit,
        "assets": entries, "errors": errors, "original_bridge_actor_transforms_used": False,
        "bridge_binding_contract": "Confirm local length axis/deck surface from mesh; place only from predicted proposals"}


def export_mesh_geometry(job):
    """Export selected mesh-local resources for deck-surface inspection.

    No actor/level is loaded. OBJ's exporter coordinate convention must be
    reconciled to native_mesh_bounds_cm before interpreting driving-surface Z.
    """
    folder = Path(job.get("geometry_output_directory", str(Path(job["output_json"]).parent / "local_geometry")))
    folder.mkdir(parents=True, exist_ok=True)
    results = []
    for path in job.get("mesh_paths", []):
        if not path.startswith("/Game/"):
            raise ValueError("Local geometry export requires a /Game/ static mesh path")
        mesh = unreal.load_asset(path)
        if not isinstance(mesh, unreal.StaticMesh):
            raise ValueError("Not a StaticMesh: " + path)
        filename = folder / (mesh.get_name() + "_" + hashlib.sha256(path.encode()).hexdigest()[:10] + ".obj")
        task = unreal.AssetExportTask()
        task.set_editor_property("object", mesh)
        task.set_editor_property("filename", str(filename))
        task.set_editor_property("automated", True)
        task.set_editor_property("prompt", False)
        task.set_editor_property("replace_identical", True)
        task.set_editor_property("exporter", unreal.StaticMeshExporterOBJ())
        if not unreal.Exporter.run_asset_export_task(task) or not filename.is_file():
            raise RuntimeError("Local mesh export failed: " + path)
        results.append({"mesh_path": path, "obj_path": str(filename), "obj_sha256": digest_file(filename),
            "native_mesh_bounds_cm": bounds(mesh), "actor_world_transform_read": False,
            "obj_axis_contract": "Infer exporter basis against native_mesh_bounds_cm; never assume OBJ axis convention"})
    return {"schema": "citycode.ue_local_mesh_geometry.v1", "count": len(results), "assets": results,
            "original_bridge_actor_transforms_used": False}


def material_paths(mesh):
    return [slot.get_editor_property("material_interface").get_path_name()
            if slot.get_editor_property("material_interface") else None
            for slot in mesh.get_editor_property("static_materials")]


def require_mesh_materials(meshes, destination):
    for mesh in meshes:
        paths = material_paths(mesh)
        if not paths or any(path is None for path in paths):
            raise ValueError("Mesh has missing material slots; repair dependencies before reuse: " + mesh.get_path_name())


def import_meshes(filename, destination, expected_sha=None, hash_verified=False, require_materials=False,
                  configure_missing_normals=False):
    if not destination.startswith(CONTENT_ROOT + "/"):
        raise ValueError("Import destination is outside dedicated content root")
    if expected_sha and not hash_verified and digest_file(filename) != expected_sha:
        raise ValueError("Prepared interchange digest mismatch: " + str(filename))
    # Destination includes a content digest. Reuse assets only when their source
    # hash metadata matches; never overwrite existing arbitrary project assets.
    existing = unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False)
    existing_meshes = [unreal.load_asset(path) for path in existing]
    existing_meshes = [asset for asset in existing_meshes if isinstance(asset, unreal.StaticMesh)]
    if existing_meshes:
        if expected_sha and all(unreal.EditorAssetLibrary.get_metadata_tag(mesh, "CityCodeInterchangeSHA256") == expected_sha for mesh in existing_meshes):
            if require_materials:
                require_mesh_materials(existing_meshes, destination)
            if configure_missing_normals:
                configure_source_normal_policy(filename, existing_meshes)
            return existing_meshes
        raise ValueError("Import destination exists with unverified content: " + destination)
    unreal.EditorAssetLibrary.make_directory(destination)
    task = unreal.AssetImportTask()
    task.set_editor_property("filename", str(filename))
    task.set_editor_property("destination_path", destination)
    task.set_editor_property("automated", True)
    task.set_editor_property("replace_existing", False)
    task.set_editor_property("save", False)
    task.set_editor_property("factory", unreal.GLTFImportFactory())
    options = unreal.GLTFImportOptions()
    options.set_editor_property("import_scale", 100.0)
    options.set_editor_property("generate_lightmap_u_vs", False)
    task.set_editor_property("options", options)
    unreal.AssetToolsHelpers.get_asset_tools().import_asset_tasks([task])
    # UE 5.0 GLTFImportFactory can return only its primary mesh in the task even
    # when it creates several valid meshes. Enumerate the isolated destination
    # on every import, then deduplicate by UObject path. Keep strict source-node
    # counts downstream: the additional meshes must never be silently omitted.
    paths = list(task.get_editor_property("imported_object_paths"))
    paths.extend(unreal.EditorAssetLibrary.list_assets(destination, recursive=True, include_folder=False))
    by_path = {}
    for path in paths:
        asset = unreal.load_asset(path)
        if isinstance(asset, unreal.StaticMesh):
            by_path[asset.get_path_name()] = asset
    meshes = [by_path[path] for path in sorted(by_path)]
    if not meshes:
        raise RuntimeError("GLTFImporter produced no StaticMesh: " + str(filename))
    for mesh in meshes:
        unreal.EditorAssetLibrary.set_metadata_tag(mesh, "CityCodeInterchangeSHA256", expected_sha or digest_file(filename))
    if require_materials:
        require_mesh_materials(meshes, destination)
    if configure_missing_normals:
        configure_source_normal_policy(filename, meshes)
    return meshes


def configure_source_normal_policy(filename, meshes, subsystem=None):
    """Only missing source NORMALs require a LOD0 rebuild in UE 5.0.

    Its glTF fallback crosses already-reflected positions, which reverses the
    direction compared with explicit imported NORMALs. Positions/UVs/materials
    and all other build options are retained. Saving belongs to the caller.
    """
    policy = normal_policy()
    requirements = policy.match_requirements(policy.read_requirements(filename), [mesh.get_path_name() for mesh in meshes])
    if any(not path.startswith(CONTENT_ROOT + "/Imported/") for path in requirements):
        raise ValueError("Normal rebuilding is limited to own generated Imported meshes")
    rows = []
    for mesh in meshes:
        path = mesh.get_path_name(); requirement = requirements[path]
        row = {"mesh_path": path, **requirement, "build_setting_changed": False}
        if requirement["recompute_normals_required"]:
            if subsystem is None:
                subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
            before = mesh_repair_snapshot(mesh, subsystem); materials = material_paths(mesh)
            options = subsystem.get_lod_build_settings(mesh, 0)
            row["recompute_normals_before"] = bool(options.get_editor_property("recompute_normals"))
            if not row["recompute_normals_before"]:
                options.set_editor_property("recompute_normals", True)
                subsystem.set_lod_build_settings(mesh, 0, options)
                row["build_setting_changed"] = True
            row["recompute_normals_after"] = bool(subsystem.get_lod_build_settings(mesh, 0).get_editor_property("recompute_normals"))
            if not row["recompute_normals_after"]:
                raise RuntimeError("Source-normal rebuild setting did not persist in memory: " + path)
            if before != mesh_repair_snapshot(mesh, subsystem) or materials != material_paths(mesh):
                raise ValueError("Normal rebuilding changed mesh bounds/Nanite/source metadata/material references")
        rows.append(row)
    return rows


def record_parts(record):
    if record.get("import_parts"):
        return record["import_parts"]
    return [{"file": record["import_file"], "sha256": record["prepared"]["interchange_sha256"],
             "bytes": record["prepared"]["interchange_bytes"], "mesh_count": record["prepared"]["mesh_count"]}]


def shared_destination(identity, digest):
    # Scene/proposal changes must reuse all immutable imported geometry assets.
    return CONTENT_ROOT + "/Imported/" + re.sub(r"[^A-Za-z0-9_]", "_", identity) + "_" + digest[:12]


def file_ready(path, part):
    try:
        first = path.stat()
        if first.st_size != part["bytes"]:
            return False, "missing_or_partial_length"
        digest = digest_file(path)
        second = path.stat()
        if first.st_size != second.st_size or first.st_mtime_ns != second.st_mtime_ns:
            return False, "upload_still_changing"
        if digest != part["sha256"]:
            return False, "hash_not_complete_or_mismatched"
        return True, None
    except (OSError, IOError):
        return False, "missing_or_locked"


def persist_mesh_nanite_material_usage(meshes, force_save_existing=False):
    """Save usage flags only on generated materials referenced by these meshes.

    UE's automatic editor usage fix regenerates the shader ID without saving
    the material package. Explicitly persist the supported usage API change so
    later editor sessions can reuse the same compiled material identity.
    """
    materials, excluded = {}, []
    for mesh in meshes:
        for slot in mesh.get_editor_property("static_materials"):
            material = slot.get_editor_property("material_interface")
            path = material.get_path_name() if material else None
            if material is None:
                excluded.append({"material_path": None, "reason": "missing_material"})
            elif not path.startswith((CONTENT_ROOT + "/Imported/", CONTENT_ROOT + "/MaterialRecovery/")):
                excluded.append({"material_path": path, "reason": "outside_generated_material_scope"})
            elif not isinstance(material, unreal.Material):
                excluded.append({"material_path": path, "reason": "not_UMaterial"})
            else:
                materials[path] = material
    result = {"scoped_material_count": len(materials), "modified_count": 0, "already_present_count": 0, "saved_material_count": 0,
        "excluded_count": len(excluded), "excluded": excluded, "records": []}
    if not materials:
        return result
    library = getattr(unreal, "MaterialEditingLibrary", None)
    usage = getattr(getattr(unreal, "MaterialUsage", None), "MATUSAGE_NANITE", None)
    if (library is None or usage is None or not hasattr(library, "has_material_usage")
            or not hasattr(library, "set_material_usage")):
        raise RuntimeError("Installed UE lacks supported MaterialEditingLibrary Nanite usage APIs")
    for path, material in sorted(materials.items()):
        present = bool(library.has_material_usage(material, usage))
        if not present:
            library.set_material_usage(material, usage)
            if not library.has_material_usage(material, usage):
                raise RuntimeError("Could not enable Nanite material usage: " + path)
            result["modified_count"] += 1
        else:
            result["already_present_count"] += 1
        saved = not present or force_save_existing
        if saved:
            if not unreal.EditorAssetLibrary.save_loaded_asset(material, only_if_is_dirty=False):
                raise RuntimeError("Could not persist Nanite usage material package: " + path)
            result["saved_material_count"] += 1
        record = {"material_path": path, "usage_before": present, "usage_after": True,
            "material_package_saved": saved, "save_reason": "usage_changed" if not present else ("force_persist_existing_usage" if saved else None)}
        result["records"].append(record)
    return result


def configure_nanite(meshes, enabled, subsystem):
    if not enabled:
        return
    for mesh in meshes:
        settings = subsystem.get_nanite_settings(mesh)
        if not settings.get_editor_property("enabled"):
            settings.set_editor_property("enabled", True)
            subsystem.set_nanite_settings(mesh, settings, True)
    # This also covers verified cache reuse: no fresh geometry import is needed
    # to repair a previously unpersisted material usage flag.
    return persist_mesh_nanite_material_usage(meshes)


def resume_completed(job, records, manifest_digest):
    """Reuse a separate checkpoint without loading its saved mesh UObjects.

    This is an explicit recovery optimization. The checkpoint must describe the
    exact immutable geometry manifest; the final fresh-process map readback
    still verifies persisted asset source metadata, materials and geometry.
    """
    if not job.get("resume_checkpoint_file"):
        return {}, None
    source = Path(job["resume_checkpoint_file"]).resolve()
    if os.path.normcase(str(source)) == os.path.normcase(str(Path(job["output_json"]).resolve())):
        raise ValueError("resume_checkpoint_file must be a separate read-only backup, not output_json")
    raw = source.read_bytes()
    prior = json.loads(raw.decode("utf-8-sig"))
    if prior.get("schema") != "citycode.ue_available_import_checkpoint.v1":
        raise ValueError("Unsupported resume checkpoint schema")
    if prior.get("assembly_manifest_sha256") != manifest_digest:
        raise ValueError("Resume checkpoint belongs to a different immutable geometry manifest")
    if prior.get("status") not in ("assets_imported_partial", "assets_imported_complete", "assets_import_stopped") or not isinstance(prior.get("imports"), list):
        raise ValueError("Resume source is not an import checkpoint containing completed rows")
    assigned = {record["id"] + ":" + part["sha256"]: (record, part)
                for record in records for part in record_parts(record)}
    completed, rejected = {}, []
    for row in prior["imports"]:
        reason = None
        key = str(row.get("input_id")) + ":" + str(row.get("sha256"))
        match = assigned.get(key)
        if not match:
            reason = "input_or_sha_not_assigned"
        elif key in completed:
            reason = "duplicate_checkpoint_row"
        else:
            record, part = match
            paths = row.get("mesh_paths")
            if row.get("file") != part["file"]:
                reason = "interchange_file_mismatch"
            elif not isinstance(paths, list) or not paths or any(not isinstance(path, str) for path in paths):
                reason = "invalid_mesh_paths"
            elif (len(paths) != part["mesh_count"] or len(set(paths)) != len(paths)
                  or row.get("mesh_count", len(paths)) != part["mesh_count"]):
                # Legacy checkpoints recorded paths without a mesh_count field.
                # Their unique path count must still equal the current proof.
                reason = "mesh_count_mismatch"
            elif bool(row.get("nanite_requested")) != bool(record.get("nanite", False)):
                reason = "nanite_request_mismatch"
            else:
                prefix = (shared_destination(record["id"], part["sha256"]) + "/").lower()
                if any(not path.lower().startswith(prefix) for path in paths):
                    reason = "mesh_outside_shared_digest_destination"
                elif any(not unreal.EditorAssetLibrary.does_asset_exist(path) for path in paths):
                    reason = "saved_asset_missing_from_registry"
        if reason:
            rejected.append({"input_id": row.get("input_id"), "file": row.get("file"), "reason": reason})
        else:
            completed[key] = dict(row, mesh_count=part["mesh_count"], resumed_from_checkpoint=True)
    return completed, {"file": str(source), "sha256": hashlib.sha256(raw).hexdigest(),
        "accepted_parts": len(completed), "rejected_parts": len(rejected), "rejected": rejected,
        "validation": "exact manifest SHA and assigned input/file/SHA/count; shared digest paths exist in registry; no UObjects loaded"}


def import_available(job):
    """Import checksum-complete files while SFTP is still uploading the rest.

    This action creates no map. It writes a checkpoint after every successful
    part. wait_for_complete is explicit, has no arbitrary timeout, and defaults
    to false for a single scan. Normal assemble retains strict fail-fast checks.
    """
    manifest_path = Path(job["assembly_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")); base = manifest_path.parent
    if manifest.get("schema") != "citycode.ue_citycode_assembly.v1":
        raise ValueError("Unsupported assembly manifest")
    records = [r for r in manifest["meshes"] if not job.get("include_kinds") or r["kind"] in job["include_kinds"]]
    worker_count, worker_index = int(job.get("worker_count", 1)), int(job.get("worker_index", 0))
    if worker_count < 1 or not 0 <= worker_index < worker_count:
        raise ValueError("worker_index must be in [0, worker_count)")
    records = [r for i, r in enumerate(records) if i % worker_count == worker_index]
    if any("prepared" not in record for record in records):
        raise ValueError("CPU interchange preparation must complete before uploading/importing")
    capability = preflight(job)
    if not capability["ready"]:
        raise RuntimeError("Required editor/plugin capabilities are missing")
    if job.get("require_distance_fields_disabled") and capability["runtime_generate_mesh_distance_fields"] != 0:
        raise RuntimeError("r.GenerateMeshDistanceFields is not 0; verify the process-only startup INI override")
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    attempts = 0
    total = sum(len(record_parts(r)) for r in records)
    manifest_digest = digest_file(manifest_path)
    completed, resumed = resume_completed(job, records, manifest_digest)
    def checkpoint(pending):
        result = {"schema": "citycode.ue_available_import_checkpoint.v1", "ue_executed": True,
            "status": "assets_imported_complete" if len(completed) == total else "assets_imported_partial",
            "map_created": False, "assembly_manifest_sha256": manifest_digest,
            "total_parts": total, "imported_parts": len(completed), "scan_count": attempts,
            "wait_for_complete": bool(job.get("wait_for_complete", False)),
            "include_kinds": job.get("include_kinds"),
            "worker_index": worker_index, "worker_count": worker_count,
            "runtime_generate_mesh_distance_fields": capability["runtime_generate_mesh_distance_fields"],
            "stop_file": job.get("stop_file", str(Path(job["output_json"]).with_name("import.stop"))),
            "imports": list(completed.values()), "pending": pending,
            "resume_checkpoint": resumed,
            "asset_cache_contract": "mesh ID plus immutable interchange SHA; independent of scene/proposal ID"}
        write_json(job["output_json"], result)
        return result
    if resumed is not None:
        checkpoint([])  # Observe accepted recovery rows before any new import.
    while True:
        attempts += 1; pending = []
        for record in records:
            for part in record_parts(record):
                if Path(job.get("stop_file", str(Path(job["output_json"]).with_name("import.stop")))).exists():
                    result = checkpoint(pending); result["status"] = "assets_import_stopped"; result["stopped_after_completed_part"] = True
                    write_json(job["output_json"], result)
                    return result
                key = record["id"] + ":" + part["sha256"]
                if key in completed:
                    continue
                ready, reason = file_ready(base / part["file"], part)
                if not ready:
                    pending.append({"input_id": record["id"], "file": part["file"], "reason": reason})
                    continue
                start_import = time.perf_counter()
                destination = shared_destination(record["id"], part["sha256"])
                meshes = import_meshes(base / part["file"], destination, part["sha256"], True,
                                       require_materials=True, configure_missing_normals=True)
                after_import = time.perf_counter()
                if len(meshes) != part["mesh_count"]:
                    raise ValueError("Imported mesh count mismatch: " + record["id"])
                configure_nanite(meshes, record.get("nanite", False), subsystem)
                after_nanite = time.perf_counter()
                unreal.EditorAssetLibrary.save_directory(destination, only_if_is_dirty=True, recursive=True)
                after_save = time.perf_counter()
                completed[key] = {"input_id": record["id"], "file": part["file"], "sha256": part["sha256"],
                    "mesh_paths": [mesh.get_path_name() for mesh in meshes], "mesh_count": len(meshes), "nanite_requested": record.get("nanite", False),
                    "timing_seconds": {"factory_import_or_cache": after_import-start_import,
                        "nanite_configuration": after_nanite-after_import, "one_dirty_directory_save": after_save-after_nanite}}
                checkpoint(pending)
                if len(completed) % 25 == 0:
                    unreal.log("CityCode cached available parts %d/%d" % (len(completed), total))
                    unreal.SystemLibrary.collect_garbage()
        result = checkpoint(pending)
        if len(completed) == total or not job.get("wait_for_complete", False):
            return result
        if digest_file(manifest_path) != manifest_digest:
            raise ValueError("Assembly manifest changed during import; restart explicitly with the new frozen manifest")
        time.sleep(max(.25, min(60., float(job.get("poll_seconds", 5.)))))


def import_bridge_template(job):
    """Import a locally regularized native deck module, then bind native materials.

    Its producer stores standard Y-up glTF (UE X, UE Z, UE Y). The measured
    GLTFImporter swap therefore yields native local UE geometry directly. This
    resource is consumed by native-placement proposals, without building-basis
    cancellation and without reading any original bridge actor placement.
    """
    path = Path(job["template_manifest"]).resolve()
    document = json.loads(path.read_text(encoding="utf-8-sig"))
    templates = document.get("templates", [document])
    materials = document["material_paths"]
    if len(materials) != 4:
        raise ValueError("Bridge module must declare its four native material slots in order")
    entries, placements = [], []
    for template in templates:
        is_span = "span_id" in template
        identity = template.get("span_id", "BRIDGE_DECK_MODULE")
        filename = path.parent / template["template_file"]
        expected = [[float(v)*100 for v in row] for row in template["expected_native_bounds_m"]]
        destination = shared_destination(identity, template["template_sha256"])
        meshes = import_meshes(filename, destination, template["template_sha256"])
        if len(meshes) != 1:
            raise ValueError("Each bridge template/span must import as exactly one native StaticMesh")
        mesh = meshes[0]; actual = bounds(mesh)
        if max(abs(actual[a][i]-expected[a][i]) for a in range(2) for i in range(3)) > .2:
            raise ValueError("Bridge native bounds/axis mismatch for %s: %s" % (identity, actual))
        if len(mesh.get_editor_property("static_materials")) != 4:
            raise ValueError("GLTFImporter changed the expected four-material bridge slot layout")
        for index, material_path in enumerate(materials):
            material = unreal.load_asset(material_path)
            if not isinstance(material, unreal.MaterialInterface):
                raise ValueError("Missing native bridge material: " + material_path)
            mesh.set_material(index, material)
        unreal.EditorAssetLibrary.set_metadata_tag(mesh, "CityCodeBridgePlacementContract", "native_template_predicted_structure_only")
        unreal.EditorAssetLibrary.save_directory(destination, only_if_is_dirty=True, recursive=True)
        material_result = [slot.get_editor_property("material_interface").get_path_name() for slot in mesh.get_editor_property("static_materials")]
        if material_result != materials:
            raise ValueError("Bridge native material override did not persist in its mesh slots")
        entry = {"asset_id": identity if is_span else "bridge_deck_default", "mesh_path": mesh.get_path_name(),
            "categories": ["predicted_bridge_span" if is_span else "bridge_deck"],
            "bounds_cm": actual, "dimensions_cm": [actual[1][i]-actual[0][i] for i in range(3)],
            "materials": [{"slot_index": i, "material_path": value} for i, value in enumerate(material_result)],
            "local_axes": {"length_axis": "X", "up_axis": "Z", "inference": "explicit_local_module_regularization"},
            "orientation_confirmed": True, "local_pivot_cm": [0., 0., 0.],
            "actor_world_transform_read": False, "template_manifest_sha256": digest_file(path),
            "template_sha256": template["template_sha256"], "placement_contract": "native_UE_local_resource_use_predicted_proposal_transforms"}
        if not is_span:
            entry["deck_top_local_z_cm"] = 0.
        entries.append(entry)
        if is_span:
            placements.append({"placement_id": identity, "asset_id": identity, "mesh_path": mesh.get_path_name(),
                "category": "predicted_bridge_span", "position_m": template["position_m"],
                "rotation_deg": template.get("rotation_deg", [0., 0., 0.]), "scale": template.get("scale", [1., 1., 1.]),
                "placement_source": "predicted_elevated_structure", "prepared_world_geometry": True,
                "logical_module_count": template["logical_module_count"], "inference": template.get("inference", "explicit_rule_regularization"),
                "native_material_paths": materials, "template_sha256": template["template_sha256"],
                "expected_native_bounds_m": template["expected_native_bounds_m"]})
    proposal = {"schema": "citycode.predicted_native_bridge_spans.v1", "coordinate_system": {"world": "UE", "units": "m", "up": "Z"},
        "placements": placements, "logical_module_count": sum(p["logical_module_count"] for p in placements),
        "placement_contract": "native deformed spans replace abstract rectangular modules; never place both", "ue_executed": True,
        "template_manifest_sha256": digest_file(path), "bridge_actor_world_transforms_used": False}
    if job.get("output_proposal"):
        write_json(job["output_proposal"], proposal)
    return {"schema": "citycode.ue_local_asset_catalog.v1", "status": "native_bridge_template_imported", "ue_executed": True,
        "count": len(entries), "assets": entries, "placement_proposal": proposal,
        "native_materials_verified": True, "bounds_verified": True,
        "bridge_actor_world_transforms_used": False}


def mesh_repair_snapshot(mesh, subsystem):
    settings = subsystem.get_nanite_settings(mesh)
    nanite = {"enabled": bool(settings.get_editor_property("enabled"))}
    for name in ("keep_percent_triangles", "trim_relative_error", "fallback_percent_triangles",
                 "fallback_relative_error", "position_precision", "minimum_residency", "preserve_area"):
        try:
            nanite[name] = settings.get_editor_property(name)
        except Exception:
            pass # These additional settings vary between installed UE versions.
    return {"bounds_cm": bounds(mesh), "nanite": nanite,
            "interchange_sha256": unreal.EditorAssetLibrary.get_metadata_tag(mesh, "CityCodeInterchangeSHA256")}


def repair_missing_materials(job):
    """Restore dependencies from the exact GLB, retaining saved map mesh paths."""
    manifest_path = Path(job["assembly_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    manifest_digest = digest_file(manifest_path)
    audit_path = Path(job["readback_result"]).resolve()
    if os.path.normcase(str(audit_path)) == os.path.normcase(str(Path(job["output_json"]).resolve())):
        raise ValueError("Material repair output must not overwrite its independent readback input")
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    if (manifest.get("schema") != "citycode.ue_citycode_assembly.v1"
            or audit.get("schema") != "citycode.ue_independent_readback.v1"
            or not audit.get("independent_saved_map_readback") or not audit.get("ue_executed")
            or audit.get("status") not in ("passed", "failed")
            or audit.get("assembly_manifest_sha256") != manifest_digest
            or audit.get("scene_id") != manifest["scene_id"]):
        raise ValueError("Material repair requires an independent readback of this exact frozen manifest")
    if not audit.get("new_level_path", "").startswith(CONTENT_ROOT + "/Maps/"):
        raise ValueError("Material repair readback is outside the dedicated map namespace")
    targets = []
    for error in audit.get("errors", []):
        if (not isinstance(error, dict) or not error.get("input_id")
                or set(error.get("issues", [])) != {"missing_material_slot"}):
            raise ValueError("Material-only repair refuses readback errors other than missing_material_slot")
        if error["input_id"] in targets:
            raise ValueError("Duplicate material repair target")
        targets.append(error["input_id"])
    records = {r["id"]: r for r in manifest["meshes"]}
    details = {r["input_id"]: r for r in audit["records"]}
    prepared_targets = []
    # Validate identities and immutable files for ALL targets before any writes.
    for identity in targets:
        record = records.get(identity); detail = details.get(identity)
        if not record or not detail or record.get("import_parts") or "prepared" not in record:
            raise ValueError("Unsupported material repair identity: " + identity)
        part = record_parts(record)[0]
        paths = detail.get("mesh_paths", [])
        prefix = (shared_destination(identity, part["sha256"]) + "/").lower()
        if (len(paths) != part["mesh_count"] or len(set(paths)) != len(paths)
                or any(not path.lower().startswith(prefix) for path in paths)):
            raise ValueError("Readback mesh identity/count does not match the immutable input: " + identity)
        ready, reason = file_ready(manifest_path.parent/part["file"], part)
        if not ready:
            raise ValueError("Material source is missing or changed (%s): %s" % (reason, identity))
        prepared_targets.append((record, part, paths))
    result = {"schema": "citycode.ue_material_dependency_repair.v1", "ue_executed": True,
        "status": "material_repair_partial", "assembly_manifest_sha256": manifest_digest,
        "readback_result_sha256": digest_file(audit_path), "new_level_path": audit["new_level_path"],
        "target_count": len(targets), "repaired_count": 0, "records": [],
        "map_modified": False, "geometry_replaced": False, "fresh_independent_readback_required": True}
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    for record, part, paths in prepared_targets:
        identity = record["id"]
        originals = [unreal.load_asset(path) for path in paths]
        if any(not isinstance(mesh, unreal.StaticMesh) for mesh in originals):
            raise ValueError("Original repair mesh is unavailable: " + identity)
        before = {mesh.get_path_name(): mesh_repair_snapshot(mesh, subsystem) for mesh in originals}
        if any(snapshot["interchange_sha256"] != part["sha256"] for snapshot in before.values()):
            raise ValueError("Original repair mesh source metadata mismatch: " + identity)
        if len({mesh.get_name() for mesh in originals}) != len(originals):
            raise ValueError("Ambiguous original repair mesh names: " + identity)
        recovery = CONTENT_ROOT + "/MaterialRecovery/" + re.sub(r"[^A-Za-z0-9_]", "_", identity) + "_" + part["sha256"][:12]
        repaired = []
        if any(not material_paths(mesh) or any(path is None for path in material_paths(mesh)) for mesh in originals):
            fresh = import_meshes(manifest_path.parent/part["file"], recovery, part["sha256"], True, require_materials=True)
            fresh_by_name = {mesh.get_name(): mesh for mesh in fresh}
            if len(fresh) != part["mesh_count"] or len(fresh_by_name) != len(fresh) or set(fresh_by_name) != {mesh.get_name() for mesh in originals}:
                raise ValueError("Recovery mesh names/count differ from original input: " + identity)
            for original in originals:
                replacement = fresh_by_name[original.get_name()]
                left, right = bounds(original), bounds(replacement)
                if max(abs(left[a][i]-right[a][i]) for a in range(2) for i in range(3)) > .2:
                    raise ValueError("Recovery geometry bounds differ from original mesh: " + identity)
                if len(material_paths(original)) != len(material_paths(replacement)):
                    raise ValueError("Recovery material slot layout differs from original mesh: " + identity)
            # Save every imported material/texture package BEFORE assigning the
            # original mesh references; saving a mesh alone caused the defect.
            if not unreal.EditorAssetLibrary.save_directory(recovery, only_if_is_dirty=False, recursive=True):
                raise RuntimeError("Could not persist all recovery dependencies: " + identity)
            for original in originals:
                replacement = fresh_by_name[original.get_name()]
                for index, slot in enumerate(replacement.get_editor_property("static_materials")):
                    original.set_material(index, slot.get_editor_property("material_interface"))
                if before[original.get_path_name()]["nanite"]["enabled"]:
                    persist_mesh_nanite_material_usage([original])
                if not unreal.EditorAssetLibrary.save_loaded_asset(original, only_if_is_dirty=False):
                    raise RuntimeError("Could not save repaired original mesh: " + original.get_path_name())
                repaired.append(original.get_path_name())
        require_mesh_materials(originals, shared_destination(identity, part["sha256"]))
        after = {mesh.get_path_name(): mesh_repair_snapshot(mesh, subsystem) for mesh in originals}
        if before != after:
            raise ValueError("Material repair changed mesh bounds, Nanite settings or source metadata: " + identity)
        result["records"].append({"input_id": identity, "mesh_paths": paths,
            "status": "repaired" if repaired else "already_has_materials", "material_paths": {mesh.get_path_name(): material_paths(mesh) for mesh in originals},
            "recovery_directory": recovery if repaired else None, "before": before, "after": after})
        result["repaired_count"] += 1
        write_json(job["output_json"], result)
        unreal.SystemLibrary.collect_garbage()
    if digest_file(manifest_path) != manifest_digest:
        raise ValueError("Frozen manifest changed during material repair")
    result["status"] = "materials_repaired_pending_independent_readback" if targets else "material_repair_noop"
    return result


def transpose(matrix):
    return [list(row) for row in zip(*matrix)]


def determinant(m):
    return (m[0][0]*(m[1][1]*m[2][2]-m[1][2]*m[2][1])
            -m[0][1]*(m[1][0]*m[2][2]-m[1][2]*m[2][0])
            +m[0][2]*(m[1][0]*m[2][1]-m[1][1]*m[2][0]))


def matvec(matrix, value):
    return [sum(a*b for a, b in zip(row, value)) for row in matrix]


def quat_from_matrix(m):
    trace = sum(m[i][i] for i in range(3))
    if trace > 0:
        s = math.sqrt(trace + 1)*2
        q = [(m[2][1]-m[1][2])/s, (m[0][2]-m[2][0])/s, (m[1][0]-m[0][1])/s, s/4]
    else:
        i = max(range(3), key=lambda index: m[index][index]); j, k = (i+1) % 3, (i+2) % 3
        s = math.sqrt(max(0, 1+m[i][i]-m[j][j]-m[k][k]))*2
        q = [0., 0., 0., 0.]; q[i] = s/4
        q[j] = (m[j][i]+m[i][j])/s; q[k] = (m[k][i]+m[i][k])/s
        q[3] = (m[k][j]-m[j][k])/s
    return q


def make_transform(position, quaternion, scale):
    result = unreal.Transform()
    result.set_editor_property("translation", unreal.Vector(*position))
    result.set_editor_property("rotation", unreal.Quat(*quaternion))
    result.set_editor_property("scale3d", unreal.Vector(*scale))
    return result


def calibrate(manifest, base):
    fixture = manifest["calibration"]
    meshes = import_meshes(base / fixture["path"], CONTENT_ROOT + "/Calibration/Fixture_" + fixture["sha256"][:12], fixture["sha256"])
    unreal.EditorAssetLibrary.save_directory(CONTENT_ROOT + "/Calibration/Fixture_" + fixture["sha256"][:12], only_if_is_dirty=True, recursive=True)
    if len(meshes) != 1:
        raise ValueError("Axis fixture must produce one StaticMesh")
    actual = bounds(meshes[0]); expected = fixture["bounds_m"]
    corners = list(itertools.product(*[(expected[0][i], expected[1][i]) for i in range(3)]))
    matches = []
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1., 1.), repeat=3):
            basis = [[signs[i] if permutation[i] == j else 0. for j in range(3)] for i in range(3)]
            converted = [[v*100 for v in matvec(basis, corner)] for corner in corners]
            target = [[min(p[i] for p in converted) for i in range(3)], [max(p[i] for p in converted) for i in range(3)]]
            if max(abs(actual[a][i]-target[a][i]) for a in range(2) for i in range(3)) < .1:
                matches.append(basis)
    if len(matches) != 1:
        raise ValueError("Importer axis/scale is not an unambiguous signed permutation at 100cm/m: " + str(actual))
    inverse = transpose(matches[0]); rotation = [row[:] for row in inverse]; scale = [1., 1., 1.]
    if determinant(rotation) < 0:
        scale[2] = -1.
        for row in rotation:
            row[2] *= -1
    q = quat_from_matrix(rotation)
    transform = make_transform([0, 0, 0], q, scale)
    for axis in range(3):
        value = [0., 0., 0.]; value[axis] = 1.
        got = vector(transform.transform_location(unreal.Vector(*value)))
        target = matvec(inverse, value)
        if max(abs(a-b) for a, b in zip(got, target)) > 1e-5:
            raise ValueError("Unreal quaternion convention failed the inverse basis check")
    return {"fixture_mesh": meshes[0].get_path_name(), "measured_bounds_cm": actual,
            "raw_gltf_to_ue_basis": matches[0], "actor_inverse_basis_quat_xyzw": q,
            "actor_inverse_basis_scale": scale, "import_scale_cm_per_m": 100., "verified": True}


def spawn(mesh, identity, kind, source, transform, scene_id, entity_id=None):
    actor = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor, unreal.Vector(0, 0, 0))
    if actor is None:
        raise RuntimeError("Could not spawn StaticMeshActor")
    actor.set_actor_label("CC_" + identity + "_" + mesh.get_name())
    component = actor.get_component_by_class(unreal.StaticMeshComponent)
    component.set_static_mesh(mesh)
    component.set_mobility(unreal.ComponentMobility.STATIC)
    if not actor.set_actor_transform(transform, False, True):
        # In some builds the exposed return value reflects a hit result rather
        # than success. Readback checks the actual persisted transform instead.
        unreal.log("CityCode transform assigned: " + identity)
    tags = ["CityCodeScene=" + scene_id, "CityCodeInput=" + identity, "CityCodeKind=" + kind,
            "CityCodePlacementSource=" + source]
    if entity_id:
        tags.append("CityCodeBuildingEntity=" + entity_id)
    actor.set_editor_property("tags", [unreal.Name(tag) for tag in tags])
    actor.set_folder_path("CityCode/" + kind)
    return actor


def inspection_daylight():
    # Import by a non-main module name: render jobs, frame callbacks and quit
    # behavior are not executed. Only the shared environment helper is called.
    path = Path(__file__).with_name("ue_render.py")
    spec = importlib.util.spec_from_file_location("citycode_shared_inspection_environment", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load the shared inspection lighting helper")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.inspection_daylight()


def tagged_geometry_snapshot():
    result = {}
    for actor in unreal.EditorLevelLibrary.get_all_level_actors():
        tags = sorted(str(tag) for tag in actor.get_editor_property("tags"))
        if not any(tag.startswith(("CityCodeInput=", "CityCodeEntityRoot=")) for tag in tags):
            continue
        component = actor.get_component_by_class(unreal.StaticMeshComponent)
        mesh = component.get_editor_property("static_mesh") if component else None
        transform = actor.get_actor_transform(); quaternion = transform.rotation
        parent = actor.get_attach_parent_actor()
        result[actor.get_path_name()] = {"mesh_path": mesh.get_path_name() if mesh else None,
            "position_cm": vector(transform.translation), "scale": vector(transform.scale3d),
            "rotation_quat_xyzw": [float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w)],
            "parent_actor_path": parent.get_path_name() if parent else None, "tags": tags}
    return result


def load_verified_saved_map(job, load_level=True):
    manifest_path = Path(job["assembly_manifest"]).resolve()
    assembly_path = Path(job["assembly_result"]).resolve()
    readback_path = Path(job["readback_result"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    assembled = json.loads(assembly_path.read_text(encoding="utf-8-sig"))
    audited = json.loads(readback_path.read_text(encoding="utf-8-sig"))
    manifest_sha = digest_file(manifest_path); assembly_sha = digest_file(assembly_path)
    level = assembled.get("new_level_path", "")
    if (manifest.get("schema") != "citycode.ue_citycode_assembly.v1"
            or assembled.get("status") != "assembled_pending_independent_readback" or not assembled.get("ue_executed")
            or assembled.get("assembly_manifest_sha256") != manifest_sha
            or assembled.get("scene_id") != manifest.get("scene_id")
            or audited.get("status") != "passed" or audited.get("errors")
            or not audited.get("independent_saved_map_readback") or not audited.get("ue_executed")
            or audited.get("assembly_manifest_sha256") != manifest_sha
            or audited.get("assembly_result_sha256") != assembly_sha
            or audited.get("scene_id") != manifest.get("scene_id") or audited.get("new_level_path") != level
            or not level.startswith(CONTENT_ROOT + "/Maps/")):
        raise ValueError("Saved-map maintenance requires the exact verified manifest, assembly result and dedicated saved map")
    if load_level and not unreal.EditorLevelLibrary.load_level(level):
        raise RuntimeError("Cannot load the independently verified CityCode map")
    return manifest_path, assembly_path, readback_path, manifest_sha, assembly_sha, level, audited, manifest


def configure_inspection_lighting(job):
    manifest_path, assembly_path, readback_path, manifest_sha, assembly_sha, level, audited, manifest = load_verified_saved_map(job)
    before = tagged_geometry_snapshot()
    if len(before) != audited["actual_tagged_actor_count"] + audited["building_entity_root_count"]:
        raise ValueError("Tagged city geometry count differs from the independently verified saved map")
    environment = inspection_daylight()
    after = tagged_geometry_snapshot()
    if before != after:
        raise ValueError("Inspection lighting changed tagged city geometry; the map was not saved")
    if digest_file(manifest_path) != manifest_sha or digest_file(assembly_path) != assembly_sha:
        raise ValueError("Verified lighting inputs changed while configuring the environment")
    if not unreal.EditorLevelLibrary.save_current_level():
        raise RuntimeError("Could not persist the dedicated map's inspection lighting")
    environment.update(session_only=False, maps_saved=True)
    snapshot_sha = hashlib.sha256(json.dumps(before, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    return {"schema": "citycode.ue_inspection_lighting_configuration.v1",
        "status": "configured_pending_independent_readback", "ue_executed": True, "new_level_path": level,
        "assembly_manifest_sha256": manifest_sha, "assembly_result_sha256": assembly_sha,
        "readback_result_sha256": digest_file(readback_path), "geometry_unchanged": True,
        "tagged_geometry_count": len(before), "geometry_snapshot_sha256_before": snapshot_sha,
        "geometry_snapshot_sha256_after": snapshot_sha, "tagged_geometry_snapshot": before,
        "inspection_environment": environment, "fresh_independent_readback_required": True}


def persist_nanite_material_usage(job):
    manifest_path, assembly_path, readback_path, manifest_sha, assembly_sha, level, audited, manifest = load_verified_saved_map(job)
    before = tagged_geometry_snapshot()
    if len(before) != audited["actual_tagged_actor_count"] + audited["building_entity_root_count"]:
        raise ValueError("Tagged city geometry count differs from the independently verified saved map")
    records = {record["id"]: record for record in manifest["meshes"]}
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    meshes, excluded_meshes, non_building_actors = {}, [], 0
    for state in before.values():
        tags = dict(tag.split("=", 1) for tag in state["tags"] if "=" in tag)
        identity = tags.get("CityCodeInput")
        record = records.get(identity)
        if not record or record["kind"] not in ("building", "massing"):
            non_building_actors += 1
            continue
        if tags.get("CityCodeScene") != manifest["scene_id"]:
            raise ValueError("Tagged building has a different scene identity")
        path = state["mesh_path"]
        if path in meshes:
            continue
        mesh = unreal.load_asset(path)
        if not isinstance(mesh, unreal.StaticMesh):
            raise ValueError("Verified building no longer references a StaticMesh: " + str(path))
        if not subsystem.get_nanite_settings(mesh).get_editor_property("enabled"):
            excluded_meshes.append({"mesh_path": path, "reason": "nanite_disabled"})
            continue
        meshes[path] = mesh
    mesh_before = {path: mesh_repair_snapshot(mesh, subsystem) for path, mesh in meshes.items()}
    usage = persist_mesh_nanite_material_usage(list(meshes.values()), force_save_existing=True)
    mesh_after = {path: mesh_repair_snapshot(mesh, subsystem) for path, mesh in meshes.items()}
    after = tagged_geometry_snapshot()
    if before != after or mesh_before != mesh_after:
        raise ValueError("Material usage maintenance changed city geometry or mesh settings")
    if digest_file(manifest_path) != manifest_sha or digest_file(assembly_path) != assembly_sha:
        raise ValueError("Verified scene inputs changed during material usage maintenance")
    snapshot_sha = hashlib.sha256(json.dumps(before, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    return {"schema": "citycode.ue_nanite_material_usage_persistence.v1", "ue_executed": True,
        "status": "nanite_material_usage_persisted_pending_independent_readback", "new_level_path": level,
        "process_id": os.getpid(),
        "assembly_manifest_sha256": manifest_sha, "assembly_result_sha256": assembly_sha,
        "readback_result_sha256": digest_file(readback_path), "geometry_unchanged": True,
        "nanite_settings_unchanged": True, "material_graph_modified": False, "map_modified": False,
        "tagged_geometry_count": len(before), "geometry_snapshot_sha256_before": snapshot_sha,
        "geometry_snapshot_sha256_after": snapshot_sha, "eligible_nanite_mesh_count": len(meshes),
        "non_building_or_entity_root_actor_count": non_building_actors, "excluded_meshes": excluded_meshes,
        **usage, "fresh_independent_readback_required": True}


def repair_background_normals(job):
    """Rebuild missing source normals on saved generated background assets only.

    The frozen GLBs and map references remain identical. The intentional saved
    mesh change is LOD0 BuildSettings.recompute_normals=True, not a reimport.
    """
    manifest_path, assembly_path, readback_path, manifest_sha, assembly_sha, level, audited, manifest = load_verified_saved_map(job)
    before = tagged_geometry_snapshot()
    if len(before) != audited["actual_tagged_actor_count"] + audited["building_entity_root_count"]:
        raise ValueError("Tagged city geometry count differs from the independently verified saved map")
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    background = {record["id"]: record for record in manifest["meshes"] if record["kind"] == "background"}
    selected = {identity: {} for identity in background}
    for state in before.values():
        tags = dict(tag.split("=", 1) for tag in state["tags"] if "=" in tag)
        identity = tags.get("CityCodeInput")
        if identity not in background:
            continue
        if tags.get("CityCodeScene") != manifest["scene_id"]:
            raise ValueError("Tagged background has a different scene identity")
        path = state["mesh_path"]
        if not path or not path.startswith(CONTENT_ROOT + "/Imported/"):
            raise ValueError("Background normal repair is limited to own generated Imported meshes")
        mesh = unreal.load_asset(path)
        if not isinstance(mesh, unreal.StaticMesh):
            raise ValueError("Verified background no longer references a StaticMesh: " + path)
        selected[identity][path] = mesh
    policy = normal_policy(); plans = []
    for identity, record in background.items():
        if len(selected[identity]) != record["prepared"]["mesh_count"]:
            raise ValueError("Background normal repair mesh count differs from the frozen manifest")
        covered = set()
        for part in record_parts(record):
            filename = manifest_path.parent / part["file"]
            ready, reason = file_ready(filename, part)
            if not ready:
                raise ValueError("Background normal repair source is missing or changed: " + str(reason))
            meshes = [mesh for mesh in selected[identity].values()
                      if unreal.EditorAssetLibrary.get_metadata_tag(mesh, "CityCodeInterchangeSHA256") == part["sha256"]]
            prefix = shared_destination(identity, part["sha256"]) + "/"
            if len(meshes) != part["mesh_count"] or any(not mesh.get_path_name().startswith(prefix) for mesh in meshes):
                raise ValueError("Background mesh identity/source digest destination differs from the frozen part")
            requirements = policy.match_requirements(policy.read_requirements(filename), [mesh.get_path_name() for mesh in meshes])
            if not all(row["recompute_normals_required"] for row in requirements.values()):
                raise ValueError("This focused repair requires missing source NORMALs on every selected background mesh")
            covered.update(requirements)
            plans.append((identity, part, filename, meshes))
        if covered != set(selected[identity]):
            raise ValueError("Background source-part coverage is incomplete")
    if not plans:
        raise ValueError("Verified scene has no generated background to repair")
    mesh_before = {path: {"mesh": mesh_repair_snapshot(mesh, subsystem), "materials": material_paths(mesh)}
                   for meshes in selected.values() for path, mesh in meshes.items()}
    rows = []
    result = {"schema": "citycode.ue_background_normal_repair.v1", "status": "background_normals_repair_in_progress",
        "ue_executed": True, "process_id": os.getpid(), "new_level_path": level,
        "assembly_manifest_sha256": manifest_sha, "assembly_result_sha256": assembly_sha,
        "readback_result_sha256": digest_file(readback_path), "records": rows, "saved_mesh_count": 0,
        "eligible_mesh_count": len(mesh_before), "map_modified": False,
        "intentional_build_setting_change": "LOD0.recompute_normals=True where source GLB lacks NORMAL",
        "fresh_independent_readback_required": True}
    for identity, part, filename, meshes in plans:
        applied = configure_source_normal_policy(filename, meshes, subsystem)
        for mesh, row in zip(meshes, applied):
            if not unreal.EditorAssetLibrary.save_loaded_asset(mesh, only_if_is_dirty=False):
                raise RuntimeError("Could not save background normal build setting: " + mesh.get_path_name())
            rows.append({**row, "input_id": identity, "source_interchange_sha256": part["sha256"], "mesh_package_saved": True})
        result["saved_mesh_count"] = len(rows)
        write_json(job["output_json"], result)
        unreal.log("CityCode background normals saved %d/%d" % (len(rows), len(mesh_before)))
    mesh_after = {path: {"mesh": mesh_repair_snapshot(mesh, subsystem), "materials": material_paths(mesh)}
                  for meshes in selected.values() for path, mesh in meshes.items()}
    after = tagged_geometry_snapshot()
    if before != after or mesh_before != mesh_after:
        raise ValueError("Background normal repair changed actor geometry/bounds/Nanite/source metadata/material references")
    if digest_file(manifest_path) != manifest_sha or digest_file(assembly_path) != assembly_sha:
        raise ValueError("Verified scene inputs changed during background normal repair")
    snapshot_sha = hashlib.sha256(json.dumps(before, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()
    result.update(status="background_normals_repaired_pending_independent_readback", geometry_unchanged=True,
        actor_transforms_and_mesh_references_unchanged=True, mesh_bounds_materials_nanite_source_unchanged=True,
        geometry_snapshot_sha256_before=snapshot_sha, geometry_snapshot_sha256_after=snapshot_sha,
        modified_count=sum(row["build_setting_changed"] for row in rows),
        already_configured_count=sum(not row["build_setting_changed"] for row in rows))
    return result


def assemble(job):
    manifest_path = Path(job["assembly_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig")); base = manifest_path.parent
    if manifest.get("schema") != "citycode.ue_citycode_assembly.v1":
        raise ValueError("Unsupported assembly manifest schema")
    if manifest["status"] != "cpu_ready_ue_execution_pending":
        raise ValueError("Assembly inputs are incomplete; bind proposals and bake every source first")
    level = job.get("new_level_path", manifest["new_level_path"])
    if not level.startswith(CONTENT_ROOT + "/Maps/"):
        raise ValueError("Map must be inside the dedicated /Game/Research/CityCodeOpenSource/Maps path")
    if unreal.EditorAssetLibrary.does_asset_exist(level):
        raise ValueError("Refusing to overwrite an existing map: " + level)
    capability = preflight(job)
    if not capability["ready"]:
        raise RuntimeError("Required editor/plugin capabilities are unavailable: " + str(capability))
    calibration = calibrate(manifest, base)
    # Prevalidate every imported file/catalog reference before creating a map.
    for record in manifest["meshes"]:
        for part in record_parts(record):
            path = base / part["file"]
            ready, reason = file_ready(path, part)
            if not ready:
                raise ValueError("Missing/changed interchange file (%s): %s" % (reason, path))
    for record in manifest["placements"]:
        if record["kind"] in ("bridge_deck", "elevated_deck", "predicted_bridge_span", "bridge_support", "bridge_railing") and record["placement_source"] != "predicted_elevated_structure":
            raise ValueError("Bridge provenance is not prediction based")
        if not isinstance(unreal.load_asset(record["mesh_path"]), unreal.StaticMesh):
            raise ValueError("Unresolvable proposal mesh: " + record["mesh_path"])
    if not unreal.EditorLevelLibrary.new_level(level):
        raise RuntimeError("Could not create new independent map")
    subsystem = unreal.get_editor_subsystem(unreal.StaticMeshEditorSubsystem)
    actors, completed, entity_roots = [], [], {}
    for index, record in enumerate(manifest["meshes"]):
        meshes = []
        for part in record_parts(record):
            imported = import_meshes(base / part["file"], shared_destination(record["id"], part["sha256"]), part["sha256"], True,
                                     require_materials=True, configure_missing_normals=True)
            if len(imported) != part["mesh_count"]:
                raise ValueError("Importer changed part mesh count for " + record["id"])
            configure_nanite(imported, record.get("nanite", False), subsystem)
            unreal.EditorAssetLibrary.save_directory(shared_destination(record["id"], part["sha256"]), only_if_is_dirty=True, recursive=True)
            meshes.extend(imported)
        if len(meshes) != record["prepared"]["mesh_count"]:
            raise ValueError("Importer changed mesh-instance count for " + record["id"])
        for mesh in meshes:
            transform = make_transform([v*100 for v in record["anchor_m"]],
                calibration["actor_inverse_basis_quat_xyzw"], calibration["actor_inverse_basis_scale"])
            actor = spawn(mesh, record["id"], record["kind"], record["placement_source"], transform, manifest["scene_id"], record.get("building_entity_id"))
            entity_id = record.get("building_entity_id")
            if entity_id:
                if entity_id not in entity_roots:
                    # An empty StaticMeshActor supplies a real root component;
                    # a bare AActor can have no RootComponent in UE 5.0.
                    parent = unreal.EditorLevelLibrary.spawn_actor_from_class(unreal.StaticMeshActor,
                        unreal.Vector(*[v*100 for v in record["anchor_m"]]))
                    parent.set_actor_label("CC_Entity_" + entity_id)
                    parent.get_component_by_class(unreal.StaticMeshComponent).set_mobility(unreal.ComponentMobility.STATIC)
                    parent.set_editor_property("tags", [unreal.Name("CityCodeEntityRoot=" + entity_id),
                        unreal.Name("CityCodeScene=" + manifest["scene_id"])])
                    parent.set_folder_path("CityCode/BuildingEntities")
                    entity_roots[entity_id] = parent
                actor.attach_to_actor(entity_roots[entity_id], unreal.Name("None"),
                    unreal.AttachmentRule.KEEP_WORLD, unreal.AttachmentRule.KEEP_WORLD,
                    unreal.AttachmentRule.KEEP_WORLD, False)
            actors.append({"actor_path": actor.get_path_name(), "input_id": record["id"], "mesh_path": mesh.get_path_name(), "kind": record["kind"]})
        completed.append(record["id"])
        if (index+1) % 25 == 0:
            unreal.log("CityCode imported %d/%d" % (index+1, len(manifest["meshes"])))
            unreal.EditorLevelLibrary.save_current_level()
    for record in manifest["placements"]:
        mesh = unreal.load_asset(record["mesh_path"])
        transform = make_transform(record["position_cm"], record["rotation_quat_xyzw"], record["scale"])
        actor = spawn(mesh, record["id"], record["kind"], record["placement_source"], transform, manifest["scene_id"])
        if record.get("group_id"):
            tags = list(actor.get_editor_property("tags"))
            tags.append(unreal.Name("CityCodePropGroup=" + record["group_id"]))
            actor.set_editor_property("tags", tags)
        actors.append({"actor_path": actor.get_path_name(), "input_id": record["id"], "mesh_path": mesh.get_path_name(), "kind": record["kind"]})
    environment = None
    if job.get("add_inspection_lighting", True):
        environment = inspection_daylight()
    if not unreal.EditorLevelLibrary.save_current_level():
        raise RuntimeError("New map failed to save")
    if environment is not None:
        environment.update(session_only=False, maps_saved=True)
    return {"schema": "citycode.ue_assembly_result.v1", "status": "assembled_pending_independent_readback",
        "ue_executed": True, "engine_version": unreal.SystemLibrary.get_engine_version(), "new_level_path": level,
        "assembly_manifest_sha256": digest_file(manifest_path), "scene_id": manifest["scene_id"],
        "calibration": calibration, "mesh_input_count": len(completed), "actor_count": len(actors), "actors": actors,
        "building_entity_root_count": len(entity_roots),
        "inspection_environment": environment,
        "bridge_actor_world_transforms_used": False}


def main():
    job_path = os.environ.get("UE_CITYCODE_JOB")
    if not job_path:
        raise RuntimeError("Set UE_CITYCODE_JOB to an absolute JSON job file")
    job = json.loads(Path(job_path).read_text(encoding="utf-8-sig"))
    output = job["output_json"]
    # A malformed maintenance job must not overwrite the evidence it validates,
    # even when the normal execution-error handler writes its result JSON.
    if job.get("action") in ("configure_inspection_lighting", "persist_nanite_material_usage", "repair_background_normals"):
        output_path = os.path.normcase(str(Path(output).resolve()))
        if any(output_path == os.path.normcase(str(Path(job[key]).resolve()))
               for key in ("assembly_manifest", "assembly_result", "readback_result")):
            raise ValueError("Maintenance output_json must be separate from every verified input")
    try:
        action = job.get("action", "preflight")
        result = {"preflight": preflight, "export_catalog": export_catalog, "export_mesh_geometry": export_mesh_geometry,
                  "import_available": import_available, "import_bridge_template": import_bridge_template,
                  "repair_missing_materials": repair_missing_materials,
                  "persist_nanite_material_usage": persist_nanite_material_usage,
                  "repair_background_normals": repair_background_normals,
                  "configure_inspection_lighting": configure_inspection_lighting, "assemble": assemble}[action](job)
        write_json(output, result)
        unreal.log("CityCode output: " + str(output))
    except Exception as error:
        write_json(output, {"schema": "citycode.ue_execution_failure.v1", "status": "failed", "action": job.get("action"),
            "error": str(error), "traceback": traceback.format_exc(), "ue_executed": True})
        raise


if __name__ == "__main__":
    main()
