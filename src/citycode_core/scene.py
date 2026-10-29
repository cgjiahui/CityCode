"""Compile a frozen scene into structure, asset, and assembly layers.

Resources remain relative to the release root; native Epic assets remain UE
references. A release is immutable. This is a scene compiler, not a transaction
server and not a replacement for the geometry/UE verifiers.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile

from . import SCHEMA_VERSION


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, value):
    """Reject absolute/escaping runtime URIs, including symlink escapes."""
    value = str(value).replace("\\", "/")
    uri = PurePosixPath(value)
    require(not uri.is_absolute() and ":" not in value, "Non-relative resource: "+value)
    path = (Path(root)/str(uri)).resolve()
    require(path.is_relative_to(Path(root).resolve()), "Resource escapes root: "+value)
    return path


def unique(rows, key):
    result = {r[key]: r for r in rows}
    require(len(result) == len(rows), "Duplicate "+key)
    return result


def matrix(value):
    require(len(value) == 4 and all(len(row) == 4 for row in value), "Expected 4x4 matrix")
    require(all(math.isfinite(x) for row in value for x in row), "Non-finite matrix")
    require(value[3] == [0, 0, 0, 1], "Non-affine matrix")
    a, b, c = (row[:3] for row in value[:3])
    det = a[0]*(b[1]*c[2]-b[2]*c[1])-a[1]*(b[0]*c[2]-b[2]*c[0])+a[2]*(b[0]*c[1]-b[1]*c[0])
    require(abs(det) > 1e-12, "Singular matrix")


def native_matrix(row):
    q = row["rotation_quat_xyzw"]
    require(len(q) == 4 and abs(sum(x*x for x in q)-1) < 1e-5, "Invalid native rotation")
    x, y, z, w = q
    r = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    require(len(row["position_cm"]) == len(row["scale"]) == 3, "Invalid native TRS")
    m = [[r[i][j]*row["scale"][j] for j in range(3)]+[row["position_cm"][i]/100] for i in range(3)]+[[0, 0, 0, 1]]
    matrix(m)
    return m


class Resources:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.rows = {}

    def add(self, path, expected=None, size=None, role="input"):
        path = Path(path).resolve()
        require(path.is_relative_to(self.root), "Input outside release: "+str(path))
        uri = path.relative_to(self.root).as_posix()
        ident = "file_"+hashlib.sha256(uri.encode()).hexdigest()[:24]
        if ident not in self.rows:
            require(path.is_file(), "Missing resource: "+uri)
            before = path.stat()
            digest = sha(path)
            require((before.st_size, before.st_mtime_ns) == (path.stat().st_size, path.stat().st_mtime_ns), "Resource changed while reading: "+uri)
            self.rows[ident] = dict(id=ident, uri=uri, bytes=before.st_size, sha256=digest, role=role)
        row = self.rows[ident]
        require(expected is None or expected == row["sha256"], "SHA256 mismatch: "+uri)
        require(size is None or size == row["bytes"], "Size mismatch: "+uri)
        return ident


def rebase_provenance(value, root):
    """Normalize known release paths in opaque producer evidence, not UE URIs."""
    if isinstance(value, dict):
        return {k: rebase_provenance(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [rebase_provenance(v, root) for v in value]
    if isinstance(value, str) and value.startswith(str(root)+os.sep):
        return Path(value).relative_to(root).as_posix()
    return value


def interchange_path(value, assembly_path, root, relocated=None):
    value = str(value).replace("\\", "/")
    if relocated is not None:
        parts = PurePosixPath(value).parts
        require("interchange" in parts, "Explicit relocation requires an interchange/ suffix")
        suffix = PurePosixPath(*parts[parts.index("interchange")+1:])
        return inside(relocated, suffix)
    path = (assembly_path.parent/value).resolve()
    require(path.is_relative_to(root), "Interchange outside release: "+value)
    return path


def compile_scene(root, config, output, interchange_root=None):
    root, output = Path(root).resolve(), Path(output).resolve()
    require(output.is_relative_to(root), "CityCode output must be inside release root")
    require(not output.exists(), "Refusing to overwrite immutable CityCode release: "+str(output))
    resources = Resources(root)
    paths = {k: inside(root, v) for k, v in config.items() if k != "release_id"}
    inputs = {k: resources.add(p) for k, p in paths.items()}
    source = {k: read(p) for k, p in paths.items()}
    binding, ue = source["binding"], source["assembly"]
    require(ue.get("schema") == "citycode.ue_citycode_assembly.v1", "Unsupported UE assembly schema")
    require(not ue.get("pending_interchange_ids"), "Assembly has unfinished geometry")
    require(ue["source_binding_sha256"] == resources.rows[inputs["binding"]]["sha256"], "Assembly/binding mismatch")
    require(binding["source_structure"]["footprints_sha256"] == resources.rows[inputs["structures"]]["sha256"], "Binding/structures mismatch")
    require(len(source["structures"]["features"]) == binding["source_structure"]["instance_count"], "Missing structures")
    lod = ue["building_lod"]
    placements = binding["asset_instances"]+binding.get("massing_instances", [])
    bindings = unique(placements, "id")
    original = unique(source["original_binding"]["asset_instances"]+source["original_binding"].get("massing_instances", []), "id") if "original_binding" in source else bindings
    structure_ids = {f["properties"]["predicted_global_building_id"] for f in source["structures"]["features"]}
    models, instances = [], []
    for row in placements:
        require(set(row["source_building_ids"]) <= structure_ids, "Unknown structure source: "+row["id"])
        variants = {}
        for label, record in row["lods"].items():
            file = (paths["binding"].parent/record["path"]).resolve()
            variants[label] = {"resource_id": resources.add(file, record["sha256"], record["bytes"], "building_model"),
                               "faces": record.get("faces")}
        require(lod in variants, "Missing selected LOD for "+row["id"])
        models.append(dict(id=row["id"], lods=variants))
        current = row["world_from_local_4x4"]
        requested = original.get(row["id"], row)["world_from_local_4x4"]
        instances.append(dict(id=row["id"], asset_id=row["id"], source_ids=row["source_building_ids"],
            kind="massing" if row["id"].startswith("MASSING_") else "building", selected_lod=lod,
            requested_transform=requested, validated_transform=current, transform_changed=requested != current,
            status=row.get("status"), building_entity_id=row.get("building_entity_id"),
            donor_asset_id=row.get("donor_asset_id"), parent_building_id=row.get("parent_building_id")))
    native_catalog = unique(source["catalog"]["assets"], "mesh_path")
    require(sha(paths["catalog"]) == ue["catalog_sha256"], "UE catalog mismatch")
    native_assets = {}
    native_instances = []
    for row in ue["placements"]:
        require(row["mesh_path"] in native_catalog, "Native mesh missing from catalog")
        asset_id = "ue_"+hashlib.sha256(row["mesh_path"].encode()).hexdigest()[:24]
        native_assets[asset_id] = dict(id=asset_id, uri=row["mesh_path"], kind="ue_project_asset", requires_local_city_sample=True)
        native_instances.append(dict(id=row["id"], asset_id=asset_id, kind=row["kind"], group_id=row.get("group_id"),
            source_segment_id=row.get("source_segment_id"), world_from_asset_m=native_matrix(row), ue=deepcopy(row)))
    relocated = Path(interchange_root).resolve() if interchange_root else None
    def import_ref(value, digest, size=None):
        return resources.add(interchange_path(value, paths["assembly"], root, relocated), digest, size, "ue_interchange")
    ue_meshes = []
    for row in ue["meshes"]:
        record = deepcopy(row)
        record["source_resource_id"] = resources.add(inside(root, record.pop("source_file")), row["source_sha256"], row["source_bytes"], "source_geometry")
        if row["kind"] in ("building", "massing"):
            require(row["id"] in bindings, "Unknown mesh binding")
            require(row["world_from_scene_4x4"] == bindings[row["id"]]["world_from_local_4x4"], "Assembly transform differs from binding")
            require(row["source_sha256"] == bindings[row["id"]]["lods"][lod]["sha256"], "Assembly uses different LOD bytes")
        # The Windows delivery used the original basename for the filtered
        # background. Its explicit derivative is the same bytes, not a guessed
        # replacement or a hash-check exemption.
        derivative = row.get("background_derivative", {})
        if row.get("import_parts") and derivative.get("sha256") == row["prepared"]["interchange_sha256"]:
            record.pop("import_file")
            record["import_resource_id"] = resources.add(inside(root, derivative["file"]), derivative["sha256"], role="filtered_background")
        else:
            record["import_resource_id"] = import_ref(record.pop("import_file"), row["prepared"]["interchange_sha256"], row["prepared"]["interchange_bytes"])
        for part in record.get("import_parts", []):
            part["resource_id"] = import_ref(part.pop("file"), part["sha256"], part["bytes"])
        if record.get("background_derivative"):
            derivative = record["background_derivative"]
            derivative["resource_id"] = resources.add(inside(root, derivative.pop("file")), derivative["sha256"], role="filtered_background")
        ue_meshes.append(record)
    require({r["id"] for r in ue_meshes if r["kind"] in ("building", "massing")} == set(bindings), "Not all placements have meshes")
    settings = {k: deepcopy(v) for k, v in ue.items() if k not in ("meshes", "placements")}
    calibration = settings["calibration"]
    calibration["resource_id"] = import_ref(calibration.pop("path"), calibration["sha256"])
    source_to_instances = {i: [] for i in structure_ids}
    for row in instances:
        for ident in row["source_ids"]:
            source_to_instances[ident].append(row["id"])
    structure_rows = []
    for feature in source["structures"]["features"]:
        props = feature["properties"]; ident = props["predicted_global_building_id"]
        structure_rows.append(dict(id=ident, footprint=feature["geometry"], base_z_m=props["base_z_m"],
            top_z_m=props["top_z_m"], height_m=props["height_m"], height_method=props["height_method"],
            observation_ids=props.get("member_observation_ids", []), instance_ids=source_to_instances[ident],
            binding_status="bound" if source_to_instances[ident] else "unbound"))
    layers = {
        "frames": {"world": {"name": "UE_WORLD", "units": "metre", "up_axis": "Z", "handedness": "left", "matrix_convention": "row-major; column vectors; world = matrix @ local"},
                   "ue_units_per_metre": 100, "background_source_to_world": "swap Y/Z once; recorded on background mesh only", "native_mesh_local_units": "centimetre; world_from_asset_m maps native coordinates converted to metres"},
        "buildings": {"structures": structure_rows, "entities": binding["building_entities"]},
        "environment": {"descriptor": source["background"], "background_mesh_ids": [r["id"] for r in ue_meshes if r["kind"] == "background"]},
        "roads": {"ground": source["ground_roads"], "elevated_centerlines": source["elevated_centerlines"],
                  "network_connections": source["network"], "elevated_placement": source["elevated_proposal"],
                  "topology_policy": "Keep explicit levels/connections; never merge elevated and ground roads by XY coincidence."},
        "assets": {"models": models, "native": list(native_assets.values())},
        "assembly": {"buildings": instances, "native": native_instances, "ue_meshes": ue_meshes, "ue_settings": settings, "ue_catalog": source["catalog"]},
        "relations": {"source_to_instances": source_to_instances, "prop_groups": source["props_groups"]["groups"]},
        "provenance": {"inputs": inputs, "compiler_schema_version": SCHEMA_VERSION,
                       "physical_review": binding.get("physical_postprocess", {}),
                       "data_sources": {"elevated_alignment": "predicted structure and explicit design completions", "ground_roads": "provided road reference; retains source attribution", "native_assets": "user-owned City Sample project"}},
        "resources": {"path_base": "CityCode_open_source root", "files": list(resources.rows.values())},
    }
    layers = rebase_provenance(layers, root)
    counts = scene_counts(layers)
    validate_layers(layers)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".citycode-", dir=output.parent))
    try:
        layer_table = {}
        for name, value in layers.items():
            file = staging/(name+".json"); write(file, value)
            layer_table[name] = {"path": file.name, "sha256": sha(file)}
        manifest = dict(schema="citycode.manifest.v2", schema_version=SCHEMA_VERSION,
            release_id=config.get("release_id", "scene_"+ue["scene_id"]), scene_id=ue["scene_id"], immutable=True,
            status="compiled", counts=counts, layers=layer_table,
            compatibility={"predecessor_schema": "1.0.0", "latest_historical_scene_release": "v1.4", "legacy_v1_runtime_compatible": False,
                "changes": ["Explicit many-to-many structure/entity/instance relationships", "Explicit reference-data and local UE resource contracts"]},
            validation={"compiler_reference_and_resource_checks": "passed", "ue_executed_by_compiler": False,
                        "scope": "File hashes, coordinates, references and assembly consistency. Geometry, materials, collision and saved-map checks remain separate."})
        write(staging/"manifest.json", manifest)
        sums = "".join(sha(p)+"  "+p.name+"\n" for p in sorted(staging.iterdir()))
        (staging/"checksums.sha256").write_text(sums)
        require(not output.exists(), "Output appeared during compilation")
        staging.rename(output)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return manifest


def scene_counts(layers):
    structures = layers["buildings"]["structures"]
    instances = layers["assembly"]["buildings"]
    kinds = Counter(r["kind"] for r in instances)
    return dict(structure_sources=len(structures), building_entities=len(layers["buildings"]["entities"]),
        building_placements=kinds["building"], massing_placements=kinds["massing"], total_building_placements=len(instances),
        unbound_structure_sources=sum(not r["instance_ids"] for r in structures),
        ground_road_nodes=len(layers["roads"]["ground"]["nodes"]), ground_road_edges=len(layers["roads"]["ground"]["edges"]),
        elevated_spans=len(layers["roads"]["elevated_centerlines"]["features"]),
        prop_groups=len(layers["relations"]["prop_groups"]), native_placements=len(layers["assembly"]["native"]))


def validate_layers(layers):
    structures = unique(layers["buildings"]["structures"], "id")
    instances = unique(layers["assembly"]["buildings"], "id")
    models = unique(layers["assets"]["models"], "id")
    resources = unique(layers["resources"]["files"], "id")
    native_assets = unique(layers["assets"]["native"], "id")
    native = unique(layers["assembly"]["native"], "id")
    require(not set(instances).intersection(native), "Instance IDs overlap")
    refs = {k: [] for k in structures}
    for row in instances.values():
        matrix(row["requested_transform"]); matrix(row["validated_transform"])
        require(row["transform_changed"] == (row["requested_transform"] != row["validated_transform"]), "Incorrect transform flag")
        require(row["asset_id"] in models, "Missing model")
        require(row["selected_lod"] in models[row["asset_id"]]["lods"], "Missing LOD")
        require(bool(row["source_ids"]), "Placement without structural source")
        for ident in row["source_ids"]:
            require(ident in refs, "Missing structure")
            refs[ident].append(row["id"])
    for row in structures.values():
        require(row["instance_ids"] == refs[row["id"]], "Structure/instance references disagree")
        require(row["binding_status"] == ("bound" if row["instance_ids"] else "unbound"), "Wrong binding status")
        require(row["top_z_m"] >= row["base_z_m"] and abs(row["top_z_m"]-row["base_z_m"]-row["height_m"]) < 1e-4, "Invalid structure height")
    require(refs == layers["relations"]["source_to_instances"], "Relation index mismatch")
    entities = unique(layers["buildings"]["entities"], "id")
    entity_members = []
    for row in entities.values():
        require(set(row["asset_instance_ids"]) <= set(instances), "Missing entity component")
        require(set(row["source_building_ids"]) <= set(structures), "Unknown entity source")
        entity_members.extend(row["asset_instance_ids"])
    require(len(entity_members) == len(set(entity_members)), "Component belongs to multiple entities")
    require(set(entity_members) == {r["id"] for r in instances.values() if r["kind"] == "building"}, "Entity component coverage mismatch")
    for row in models.values():
        require(all(r["resource_id"] in resources for r in row["lods"].values()), "Missing model resource")
    mesh_ids = set()
    for row in layers["assembly"]["ue_meshes"]:
        require(row["id"] not in mesh_ids, "Duplicate UE mesh")
        mesh_ids.add(row["id"])
        require(row["source_resource_id"] in resources and row["import_resource_id"] in resources, "Missing UE resource")
        matrix(row["world_from_scene_4x4"])
        if row["kind"] in ("building", "massing"):
            require(row["id"] in instances, "UE mesh without binding")
            instance = instances[row["id"]]
            require(row["world_from_scene_4x4"] == instance["validated_transform"], "UE/binding transform mismatch")
            require(row["source_resource_id"] == models[instance["asset_id"]]["lods"][instance["selected_lod"]]["resource_id"], "UE/model resource mismatch")
        for part in row.get("import_parts", []):
            require(part["resource_id"] in resources, "Missing background part")
        if row.get("background_derivative"):
            require(row["background_derivative"]["resource_id"] in resources, "Missing background derivative")
    require(set(instances) <= mesh_ids, "Not all instances have UE geometry")
    groups = unique(layers["relations"]["prop_groups"], "group_id")
    for row in native.values():
        require(row["asset_id"] in native_assets, "Missing native asset")
        require(row["ue"]["mesh_path"] == native_assets[row["asset_id"]]["uri"], "Native asset mismatch")
        require(row["world_from_asset_m"] == native_matrix(row["ue"]), "Native transform mismatch")
        require(row["id"] == row["ue"]["id"], "Native ID mismatch")
        require(row["group_id"] is None or row["group_id"] in groups, "Missing prop group")
    for ident, group in groups.items():
        require(sum(r["group_id"] == ident for r in native.values()) == group["part_count"], "Prop group part count mismatch: "+ident)
    ground = layers["roads"]["ground"]
    nodes = unique(ground["nodes"], "node_id")
    unique(ground["edges"], "edge_id")
    for edge in ground["edges"]:
        require(edge["source"] in nodes and edge["target"] in nodes, "Broken road graph")
    features = layers["roads"]["elevated_centerlines"]["features"]
    require(len({f["properties"]["segment_id"] for f in features}) == len(features), "Duplicate elevated span")


def load_scene(directory):
    directory = Path(directory)
    manifest = read(directory/"manifest.json")
    require(manifest.get("schema") == "citycode.manifest.v2" and manifest.get("schema_version") == SCHEMA_VERSION, "Unsupported CityCode version")
    for line in (directory/"checksums.sha256").read_text().splitlines():
        expected, filename = line.split("  ", 1)
        require(sha(inside(directory, filename)) == expected, "CityCode checksum mismatch: "+filename)
    layers = {}
    for key, desc in manifest["layers"].items():
        path = inside(directory, desc["path"])
        require(sha(path) == desc["sha256"], "Changed CityCode layer: "+key)
        layers[key] = read(path)
    validate_layers(layers)
    require(scene_counts(layers) == manifest["counts"], "CityCode count mismatch")
    return manifest, layers


def validate(directory, root, verify_files=False):
    manifest, layers = load_scene(directory)
    for row in layers["resources"]["files"]:
        path = inside(root, row["uri"])
        require(path.is_file() and path.stat().st_size == row["bytes"], "Missing/changed file: "+row["uri"])
        if verify_files:
            require(sha(path) == row["sha256"], "Resource hash mismatch: "+row["uri"])
    return dict(status="passed", schema_version=manifest["schema_version"], counts=manifest["counts"],
                checked_resources=len(layers["resources"]["files"]), file_hashes_verified=verify_files, ue_executed=False)


def export_ue(directory, root, output):
    """Emit the existing UE adapter protocol from CityCode layers, without UE."""
    output, root = Path(output).resolve(), Path(root).resolve()
    require(output.is_relative_to(root), "UE output must be under release root")
    catalog_file = output.with_suffix(".catalog.json")
    require(not output.exists() and not catalog_file.exists(), "Refusing to overwrite UE output")
    validate(directory, root)
    _, layers = load_scene(directory)
    resources = {r["id"]: r for r in layers["resources"]["files"]}
    def uri(ident):
        return resources[ident]["uri"]
    def imported(ident):
        return Path(os.path.relpath(inside(root, uri(ident)), output.parent)).as_posix()
    result = deepcopy(layers["assembly"]["ue_settings"])
    result["meshes"] = deepcopy(layers["assembly"]["ue_meshes"])
    for row in result["meshes"]:
        row["source_file"] = uri(row.pop("source_resource_id"))
        row["import_file"] = imported(row.pop("import_resource_id"))
        for part in row.get("import_parts", []):
            part["file"] = imported(part.pop("resource_id"))
        if row.get("background_derivative"):
            d = row["background_derivative"]; d["file"] = uri(d.pop("resource_id"))
    calibration = result["calibration"]
    calibration["path"] = imported(calibration.pop("resource_id"))
    result["placements"] = [deepcopy(r["ue"]) for r in layers["assembly"]["native"]]
    result["source_binding"] = uri(layers["provenance"]["inputs"]["binding"])
    result["citycode_source"] = {"manifest": Path(directory).resolve().relative_to(root).as_posix()+"/manifest.json", "sha256": sha(Path(directory)/"manifest.json")}
    output.parent.mkdir(parents=True, exist_ok=True)
    write(catalog_file, layers["assembly"]["ue_catalog"])
    result["catalog_sha256"] = sha(catalog_file)
    write(output, result)
    return {"status": "exported", "meshes": len(result["meshes"]), "native_placements": len(result["placements"]), "ue_executed": False}
