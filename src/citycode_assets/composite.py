"""Combine registered asset parts with their embedded materials and scene nodes.

Part geometry remains in its original accessor coordinates. Container nodes put
every part in one metric glTF frame; a single bounded transform then fits the
whole union. This preserves relative positions and roof-height differences.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
from shapely.geometry import shape

from .lod import _read_glb, inspect_glb
from .registration import (FitPolicy, fit_predicted_asset, overlap_metrics,
    raster_triangle_footprint, read_glb_scene_geometry, sha256_file)


def _affine(matrix):
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("Finite affine 4x4 matrix required")
    if np.linalg.det(matrix[:3, :3]) <= 1e-12:
        raise ValueError("Composite part transform must preserve orientation and be invertible")
    return matrix


def _texture_indices(value, offset, parent=""):
    if isinstance(value, dict):
        if parent.endswith("Texture") and "index" in value:
            value["index"] += offset
        for key, child in value.items():
            _texture_indices(child, offset, key)
    elif isinstance(value, list):
        for child in value:
            _texture_indices(child, offset, parent)


def pack_glbs(parts: list[dict[str, Any]], destination: str | Path) -> dict[str, Any]:
    """Embed independently textured GLBs under supplied glTF container matrices.

    Each part declares asset_id, source_glb, source_sha256 and
    composite_from_asset_4x4. Accessor and encoded image bytes are copied without
    rewriting attributes, baking transforms or flattening materials.
    """
    if len(parts) < 1 or len({p["asset_id"] for p in parts}) != len(parts):
        raise ValueError("Distinct part asset IDs are required")
    if len({str(Path(p["source_glb"]).resolve()) for p in parts}) != len(parts):
        raise ValueError("The same complete source asset cannot be cloned within a composite")
    document = {"asset": {"version": "2.0", "generator": "CityCode registered-part assembly"},
                "buffers": [{"byteLength": 0}], "bufferViews": [], "accessors": [], "images": [],
                "samplers": [], "textures": [], "materials": [], "meshes": [], "nodes": [],
                "scenes": [{"nodes": []}], "scene": 0, "extensionsUsed": [], "extensionsRequired": []}
    binary = bytearray()
    records = []
    for part in parts:
        matrix = _affine(part["composite_from_asset_4x4"])
        scene = read_glb_scene_geometry(part["source_glb"])
        if scene["source_sha256"] != part["source_sha256"]:
            raise ValueError("Part source hash changed")
        # Validate encoded NORMAL, UV and embedded textured PBR before copying.
        inspect_glb(part["source_glb"])
        data, old, binary_offset = _read_glb(Path(part["source_glb"]))
        offsets = {key: len(document[key]) for key in
                   ("bufferViews", "accessors", "images", "samplers", "textures", "materials", "meshes", "nodes")}
        base = len(binary)
        binary.extend(data[binary_offset:binary_offset + old["buffers"][0]["byteLength"]])
        binary.extend(b"\0" * (-len(binary) % 4))
        for view in old.get("bufferViews", []):
            value = copy.deepcopy(view)
            if value.get("buffer", 0) != 0:
                raise ValueError("Only an embedded single GLB buffer is supported")
            value["buffer"] = 0
            value["byteOffset"] = base + value.get("byteOffset", 0)
            document["bufferViews"].append(value)
        for accessor in old.get("accessors", []):
            value = copy.deepcopy(accessor)
            if "sparse" in value:
                raise ValueError("Sparse accessors require an explicit conversion")
            value["bufferView"] += offsets["bufferViews"]
            document["accessors"].append(value)
        for image in old.get("images", []):
            value = copy.deepcopy(image)
            if "bufferView" in value:
                value["bufferView"] += offsets["bufferViews"]
            if value.get("uri") and not value["uri"].startswith("data:"):
                raise ValueError("External image resources are not portable")
            document["images"].append(value)
        document["samplers"].extend(copy.deepcopy(old.get("samplers", [])))
        for texture in old.get("textures", []):
            value = copy.deepcopy(texture)
            if "source" in value:
                value["source"] += offsets["images"]
            if "sampler" in value:
                value["sampler"] += offsets["samplers"]
            for extension in value.get("extensions", {}).values():
                if "source" in extension:
                    extension["source"] += offsets["images"]
            document["textures"].append(value)
        for material in old.get("materials", []):
            value = copy.deepcopy(material)
            _texture_indices(value, offsets["textures"])
            document["materials"].append(value)
        for mesh in old.get("meshes", []):
            value = copy.deepcopy(mesh)
            for primitive in value["primitives"]:
                primitive["attributes"] = {key: index + offsets["accessors"] for key, index in primitive["attributes"].items()}
                if "indices" in primitive:
                    primitive["indices"] += offsets["accessors"]
                if "material" in primitive:
                    primitive["material"] += offsets["materials"]
            document["meshes"].append(value)
        for node in old.get("nodes", []):
            value = copy.deepcopy(node)
            if "mesh" in value:
                value["mesh"] += offsets["meshes"]
            if "children" in value:
                value["children"] = [index + offsets["nodes"] for index in value["children"]]
            document["nodes"].append(value)
        roots = old["scenes"][old.get("scene", 0)]["nodes"]
        container = len(document["nodes"])
        document["nodes"].append({"name": part["asset_id"], "matrix": matrix.T.reshape(-1).tolist(),
                                  "children": [index + offsets["nodes"] for index in roots]})
        document["scenes"][0]["nodes"].append(container)
        for key in ("extensionsUsed", "extensionsRequired"):
            document[key] = sorted(set(document[key]) | set(old.get(key, [])))
        records.append({"asset_id": part["asset_id"], "source_glb": str(Path(part["source_glb"]).resolve()),
                        "source_sha256": scene["source_sha256"], "composite_from_asset_4x4": matrix.tolist(),
                        "source_triangles": len(scene["faces"]), "container_node": container})
    document["buffers"][0]["byteLength"] = len(binary)
    document["extras"] = {"citycode": {"component_count": len(parts), "source_assets":
                          [{"id": p["asset_id"], "sha256": p["source_sha256"]} for p in parts],
                          "geometry_uv_normal_texture_bytes_preserved": True}}
    encoded = json.dumps(document, separators=(",", ":"), allow_nan=False).encode()
    encoded += b" " * (-len(encoded) % 4)
    body = struct.pack("<I4s", len(encoded), b"JSON") + encoded + struct.pack("<I4s", len(binary), b"BIN\x00") + binary
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    temporary.write_bytes(struct.pack("<4sII", b"glTF", 2, len(body)+12) + body)
    temporary.replace(destination)
    packed_scene = read_glb_scene_geometry(destination)
    expected_vertices = []
    for part in parts:
        vertices = read_glb_scene_geometry(part["source_glb"])["vertices"]
        transform = _affine(part["composite_from_asset_4x4"])
        expected_vertices.append(vertices @ transform[:3, :3].T + transform[:3, 3])
    expected = np.concatenate(expected_vertices)
    if not np.allclose(packed_scene["vertices"], expected, atol=1e-8, rtol=1e-12):
        raise ValueError("Packed scene geometry does not reproduce the registered part placement")
    info = inspect_glb(destination)
    return {"schema": "citycode.packed_component_asset.v1", "output": str(destination.resolve()),
            "output_sha256": info["sha256"], "component_count": len(parts), "parts": records,
            "geometry_node_closure_max_error": float(np.max(np.abs(packed_scene["vertices"]-expected))),
            "mesh": info, "geometry_uv_normals_materials_remain_embedded": True}


def make_registered_composite(parts: list[dict[str, Any]], destination: str | Path) -> dict[str, Any]:
    """Reproduce old metric placements in one shared local glTF frame."""
    world_vertices = []
    for part in parts:
        scene = read_glb_scene_geometry(part["source_glb"])
        if scene["source_sha256"] != part["source_sha256"]:
            raise ValueError("Registered source changed")
        matrix = _affine(part["prior_world_from_asset_4x4"])
        world_vertices.append(scene["vertices"] @ matrix[:3, :3].T + matrix[:3, 3])
    vertices = np.concatenate(world_vertices)
    origin = (vertices.min(0)+vertices.max(0))/2
    origin[2] = vertices[:, 2].min()
    world_from_composite = np.array([[1, 0, 0, origin[0]], [0, 0, -1, origin[1]],
                                    [0, 1, 0, origin[2]], [0, 0, 0, 1]], dtype=float)
    composite_from_world = np.linalg.inv(world_from_composite)
    embedded = [{**part, "composite_from_asset_4x4":
                 (composite_from_world @ _affine(part["prior_world_from_asset_4x4"])).tolist()} for part in parts]
    output = pack_glbs(embedded, destination)
    output.update({"prior_world_from_composite_4x4": world_from_composite.tolist(),
                   "source_origin_world_m": origin.tolist(),
                   "old_registered_world_bounds": [vertices.min(0).tolist(), vertices.max(0).tolist()],
                   "relative_part_placement_policy": "one common affine change only; no independent part relocation or roof-height equalization"})
    for record, part, part_world in zip(output["parts"], parts, world_vertices):
        record.update({"prior_world_from_asset_4x4": part["prior_world_from_asset_4x4"],
                       "old_registered_base_z_m": float(part_world[:, 2].min()),
                       "old_registered_top_z_m": float(part_world[:, 2].max())})
    return output


def fit_registered_composite(composite: dict[str, Any], building: dict[str, Any], *, policy=None) -> dict[str, Any]:
    policy = policy or FitPolicy()
    result = fit_predicted_asset(composite["output"], building, composite["prior_world_from_composite_4x4"],
                                  expected_source_sha256=composite["output_sha256"], policy=policy,
                                  component_prior_world_from_asset=[p["prior_world_from_asset_4x4"]
                                                                   for p in composite["parts"]])
    result["component_assembly"] = composite
    result["parts"] = []
    world = np.asarray(result["world_from_asset_4x4"])
    for part in composite["parts"]:
        matrix = world @ np.asarray(part["composite_from_asset_4x4"])
        singular = np.linalg.svd(matrix[:2][:, [0, 2]], compute_uv=False)
        anisotropy = float(singular[0]/singular[-1])
        if anisotropy > policy.maximum_total_anisotropy + 1e-8:
            result["reasons"].append("component_total_xy_anisotropy_exceeds_bound:" + part["asset_id"])
        zratio = result["fit"]["relative_z_scale"]
        prior_base = composite["source_origin_world_m"][2]
        result["parts"].append({**part, "world_from_part_asset_4x4": matrix.tolist(),
            "building_local_from_part_asset_4x4": (np.asarray(building["local_from_world_4x4"]) @ matrix).tolist(),
            "normal_part_to_world_3x3": np.linalg.inv(matrix[:3, :3]).T.tolist(),
            "final_total_xy_anisotropy": anisotropy,
            "registered_base_z_m": result["predicted_base_z_m"] + zratio*(part["old_registered_base_z_m"]-prior_base),
            "registered_top_z_m": result["predicted_base_z_m"] + zratio*(part["old_registered_top_z_m"]-prior_base)})
    if not result["reasons"]:
        metrics = result["fit"]["actual_mesh_overlap"]
        margin = min(metrics["iou"]-.70, metrics["target_coverage"]-.80, metrics["asset_coverage"]-.80)
        if margin < .025:
            scene = read_glb_scene_geometry(composite["output"])
            footprint, profile = raster_triangle_footprint(scene["vertices"], scene["faces"], world,
                                                          cell_m=.05, max_dimension=4096)
            fine = overlap_metrics(footprint, shape(building["footprint_world"]))
            passed = fine["iou"] >= policy.minimum_footprint_iou and fine["target_coverage"] >= policy.minimum_target_coverage and fine["asset_coverage"] >= policy.minimum_asset_coverage
            result["finer_validation"] = {"metrics": fine, "projection": profile, "passes": passed}
            if not passed:
                result["reasons"].append("near_threshold_composite_not_stable_at_finer_projection")
    result["status"] = "accepted" if not result["reasons"] else "review_required"
    result["relative_roof_height_differences_preserved_up_to_common_z_scale"] = True
    result["actual_total_triangles_before_budgeted_lod_export"] = result["mesh"]["triangle_count"]
    return result


def component_face_budgets(component_count: int, total_targets=(250000, 100000)):
    if component_count < 1:
        raise ValueError("At least one component is required")
    budgets = []
    for index in range(component_count):
        budgets.append({int(target): int(target)//component_count + (index < int(target)%component_count)
                        for target in total_targets})
    assert all(sum(row[int(target)] for row in budgets) == int(target) for target in total_targets)
    return budgets


def assemble_budgeted_lod(fit: dict[str, Any], component_glbs: list[str | Path], destination: str | Path,
                         *, total_target: int) -> dict[str, Any]:
    """Join exported part LODs, then validate one placement and whole-building cap."""
    if fit["status"] != "accepted" or len(component_glbs) != len(fit["parts"]):
        raise ValueError("An approved fit and one LOD per part are required")
    parts = [{"asset_id": record["asset_id"], "source_glb": str(path), "source_sha256": sha256_file(path),
              "composite_from_asset_4x4": record["composite_from_asset_4x4"]}
             for record, path in zip(fit["parts"], component_glbs)]
    result = pack_glbs(parts, destination)
    if result["mesh"]["faces"] > int(total_target)*1.02:
        raise ValueError("Whole-building LOD exceeds the total face-count cap")
    scene = read_glb_scene_geometry(destination)
    footprint, profile = raster_triangle_footprint(scene["vertices"], scene["faces"],
        fit["world_from_asset_4x4"], cell_m=.05, max_dimension=4096)
    metrics = overlap_metrics(footprint, shape(fit["target_geometry"]))
    passed = (metrics["iou"] >= fit["policy"]["minimum_footprint_iou"] and
              metrics["target_coverage"] >= fit["policy"]["minimum_target_coverage"] and
              metrics["asset_coverage"] >= fit["policy"]["minimum_asset_coverage"])
    result.update({"total_face_target": int(total_target), "world_from_asset_4x4": fit["world_from_asset_4x4"],
                   "building_local_from_asset_4x4": fit["building_local_from_asset_4x4"],
                   "world_from_building_local_4x4": fit["world_from_building_local_4x4"],
                   "actual_mesh_overlap": metrics, "validation_projection": profile,
                   "status": "accepted" if passed else "review_required",
                   "relative_part_positions_and_heights_preserved": True})
    return result
