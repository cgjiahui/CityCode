"""Filter only a UE interchange copy using predicted class-4 XY evidence.

Input coordinates are already local UE XYZ metres, with an explicit world
anchor. Published coastal source geometry is never edited. PNG payloads and
original vertex bufferViews are copied as bytes, without texture decoding.
Boundary triangles are clipped with interpolated attributes, so low ground and
high surfaces outside the prediction mask survive even for large triangles.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import mmap
from pathlib import Path
import resource
import time

import numpy as np
import shapely
from scipy import ndimage
from shapely.geometry import LineString, Point, Polygon, box
from shapely.ops import polygonize, unary_union

from interchange import Writer, accessor, digest_file, read_glb, scene_meshes

ROOT = Path(__file__).resolve().parents[2]


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def release_mapped_pages(binary):
    # The source is a multi-gigabyte mmap. Discard read-only resident pages
    # after streaming them; retained NumPy accessor values are independent copies.
    mapping = binary.obj
    if hasattr(mapping, "madvise"):
        mapping.madvise(mmap.MADV_DONTNEED)


class PredictionMask:
    def __init__(self, mask, origin, cell, metadata=None):
        self.mask = np.asarray(mask, bool)
        self.origin = np.asarray(origin, float)
        self.cell = float(cell)
        self.metadata = metadata or {}
        if (self.mask.ndim != 2 or self.origin.shape != (2,) or not np.isfinite(self.origin).all()
                or not np.isfinite(self.cell) or self.cell <= 0):
            raise ValueError("Prediction mask needs a 2D array, finite corner origin and positive metre cell size")
        self.height, self.width = self.mask.shape
        self.integral = np.pad(self.mask.astype(np.int32).cumsum(0, dtype=np.int32).cumsum(1, dtype=np.int32),
                               ((1, 0), (1, 0)))
        self._geometry = None

    @classmethod
    def load(cls, path):
        path = Path(path)
        descriptor = path.with_suffix(".json")
        metadata = json.loads(descriptor.read_text(encoding="utf-8-sig"))
        if metadata.get("uses_ground_truth_for_mask") is not False:
            raise ValueError("Mask descriptor must explicitly exclude GT road/bridge geometry")
        if metadata.get("mask_sha256") != digest_file(path):
            raise ValueError("Prediction mask checksum differs from its descriptor")
        if metadata.get("coordinate_system") != {"world": "UE", "units": "m", "up": "Z"}:
            raise ValueError("Expected world UE XYZ metres prediction mask")
        with np.load(path) as data:
            mask = data["remove_high_surface_mask"].copy()
            source = data["source_retained_class4_mask"].astype(bool)
            origin = data["origin_xy_m"].copy()
            cell = float(data["cell_size_m"])
        if mask.shape != source.shape or np.any(source & ~mask):
            raise ValueError("Filtering mask must contain the retained class-4 source mask")
        expected = ndimage.distance_transform_edt(~source, sampling=cell) <= float(metadata["margin_m"])+1e-9
        if not source.any():
            expected[:] = False
        if not np.array_equal(mask, expected):
            raise ValueError("Mask is not the declared exact EDT-bounded source expansion")
        return cls(mask, origin, cell, metadata)

    def bbox_classes(self, triangles):
        """Return boxes with any mask cell, and boxes entirely inside mask."""
        xy = triangles[:, :, :2]
        lower = np.floor((xy.min(1)-self.origin)/self.cell).astype(np.int64)
        upper = np.floor((xy.max(1)-self.origin)/self.cell).astype(np.int64)
        inside = ((lower >= 0).all(1) & (upper[:, 0] < self.width) & (upper[:, 1] < self.height))
        lx = np.clip(lower[:, 0], 0, self.width); ly = np.clip(lower[:, 1], 0, self.height)
        ux = np.clip(upper[:, 0]+1, 0, self.width); uy = np.clip(upper[:, 1]+1, 0, self.height)
        count = self.integral[uy, ux]-self.integral[ly, ux]-self.integral[uy, lx]+self.integral[ly, lx]
        size = (ux-lx)*(uy-ly)
        return count > 0, inside & (count == size) & (size > 0)

    @property
    def geometry(self):
        if self._geometry is None:
            # Polygonize only exposed grid edges; do not allocate a polygon per
            # filled cell (the full mask has several hundred thousand cells).
            pad = np.pad(self.mask, 1)
            segments = []
            for direction, neighbour, shift_a, shift_b in (
                    ("left", pad[1:-1, :-2], (0, 0), (0, 1)),
                    ("right", pad[1:-1, 2:], (1, 0), (1, 1)),
                    ("bottom", pad[:-2, 1:-1], (0, 0), (1, 0)),
                    ("top", pad[2:, 1:-1], (0, 1), (1, 1))):
                ys, xs = np.nonzero(self.mask & ~neighbour)
                for x, y in zip(xs, ys):
                    a = self.origin+(np.array([x, y])+shift_a)*self.cell
                    b = self.origin+(np.array([x, y])+shift_b)*self.cell
                    segments.append(LineString([a, b]))
            selected = []
            for poly in polygonize(segments):
                q = np.asarray(poly.representative_point().coords[0])
                ix, iy = np.floor((q-self.origin)/self.cell).astype(int)
                if 0 <= ix < self.width and 0 <= iy < self.height and self.mask[iy, ix]:
                    selected.append(poly)
            self._geometry = unary_union(selected)
            expected_area = int(self.mask.sum())*self.cell**2
            if abs(self._geometry.area-expected_area) > max(1e-6, expected_area*1e-10):
                raise ValueError("Polygonized prediction mask area differs from its raster cells")
        return self._geometry


def clip_weights(weights, values, threshold, below=True):
    """Sutherland-Hodgman clip in barycentric coordinates."""
    if not len(weights):
        return []
    out = []
    for a, b in zip(weights, np.roll(weights, -1, axis=0)):
        av, bv = float(a @ values-threshold), float(b @ values-threshold)
        ain, bin = (av <= 1e-10, bv <= 1e-10) if below else (av >= -1e-10, bv >= -1e-10)
        if ain:
            out.append(a)
        if ain != bin:
            t = av/(av-bv)
            out.append(a+(b-a)*t)
    return np.asarray(out) if out else []


def weight_area(weights):
    if len(weights) < 3:
        return 0.
    uv = np.asarray(weights)[:, 1:]
    return abs(float(np.sum(uv[:, 0]*np.roll(uv[:, 1], -1)-uv[:, 1]*np.roll(uv[:, 0], -1))))*.5


def fan(weights):
    result = []
    if len(weights) >= 3:
        for i in range(1, len(weights)-1):
            triangle = np.array([weights[0], weights[i], weights[i+1]])
            signed = float(np.linalg.det(np.array([triangle[1, 1:]-triangle[0, 1:], triangle[2, 1:]-triangle[0, 1:]])))
            if abs(signed) > 1e-12:
                result.append(triangle if signed > 0 else triangle[[0, 2, 1]])
    return result


def subtract_convex(weights, constraints):
    """Keep a convex polygon outside the intersection of linear halfplanes."""
    current = weights
    outside = []
    for values, threshold, below in constraints:
        piece = clip_weights(current, values, threshold, not below)
        if weight_area(piece) > 1e-12:
            outside.append(piece)
        current = clip_weights(current, values, threshold, below)
        if weight_area(current) <= 1e-12:
            break
    return outside


def vertical_remainder(vertices, mask_geometry, threshold):
    """Clip vertical triangles by mask intervals, preserving their lower parts."""
    weights = np.eye(3)
    xy = vertices[:, :2]
    axis = int(np.argmax(np.ptp(xy, axis=0)))
    values = xy[:, axis]
    if np.ptp(values) < 1e-9:
        if not mask_geometry.covers(Point(xy[0])):
            return None
        remaining = clip_weights(weights, vertices[:, 2], threshold)
        return fan(remaining)
    lo, hi = int(np.argmin(values)), int(np.argmax(values))
    overlap = LineString([xy[lo], xy[hi]]).intersection(mask_geometry)
    lines = list(overlap.geoms) if hasattr(overlap, "geoms") else [overlap]
    intervals = []
    for line in lines:
        if line.geom_type == "LineString" and line.length > 1e-9:
            ts = np.asarray(line.coords)[:, axis]
            intervals.append((float(ts.min()), float(ts.max())))
    pieces = [weights]
    for a, b in intervals:
        result = []
        for piece in pieces:
            result.extend(subtract_convex(piece, [(values, a, False), (values, b, True),
                                                  (vertices[:, 2], threshold, False)]))
        pieces = result
    if abs(sum(weight_area(p) for p in pieces)-.5) < 1e-10:
        return None
    return [tri for piece in pieces for tri in fan(piece)]


def clipped_triangle(vertices, mask_geometry, threshold):
    """None=unchanged; []=removed; otherwise barycentric triangles to retain."""
    vertices = np.asarray(vertices, float)
    if np.max(vertices[:, 2]) <= threshold:
        return None
    xy = vertices[:, :2]
    transform = np.column_stack((xy[1]-xy[0], xy[2]-xy[0]))
    # A vertical source wall can acquire a micron-scale XY sliver when its
    # interpolated coordinates are serialized as float32. Inverting that thin
    # projection amplifies quantization into a large barycentric error. Treat
    # projections below several source-coordinate ULPs as vertical intervals.
    edge = float(np.max(np.linalg.norm(xy-np.roll(xy, -1, axis=0), axis=1)))
    roundoff = abs(float(np.spacing(np.float32(max(1., np.max(np.abs(xy)))))))
    determinant_floor = max(1e-9, edge*roundoff*8.)
    if abs(float(np.linalg.det(transform))) < determinant_floor:
        return vertical_remainder(vertices, mask_geometry, threshold)
    triangle = Polygon(xy)
    if not triangle.intersects(mask_geometry):
        return None
    high = clip_weights(np.eye(3), vertices[:, 2], threshold, below=False)
    if weight_area(high) <= 1e-12:
        return None
    high_poly = Polygon(np.asarray(high) @ xy)
    removed = high_poly.intersection(mask_geometry)
    if removed.area <= max(1e-10, triangle.area*1e-12):
        return None
    remaining = triangle.difference(removed)
    if remaining.is_empty:
        return []
    pieces = list(remaining.geoms) if hasattr(remaining, "geoms") else [remaining]
    result = []
    inverse = np.linalg.inv(transform)
    for piece in pieces:
        if piece.geom_type != "Polygon" or piece.area < 1e-10:
            continue
        for tri in shapely.constrained_delaunay_triangles(piece).geoms:
            points = np.asarray(tri.exterior.coords)[:3]
            uv = (points-xy[0]) @ inverse.T
            barycentric = np.column_stack((1-uv.sum(1), uv))
            result.extend(fan(barycentric))
    retained_area = sum(weight_area(t) for t in result)*triangle.area*2
    if abs(retained_area-remaining.area) > max(1e-6, triangle.area*1e-8):
        raise ValueError("Boundary clipping failed area conservation")
    return result


def stable_serialized_pieces(pieces, original, mask, anchor, threshold):
    """Remove only tiny new slivers that float32 cannot represent consistently.

    This runs on newly clipped faces, never original low-ground triangles.
    Actual serialization may collapse a cut vertex onto an almost vertical
    edge. Recheck that serialized triangle; an unstable fragment below 0.01m²
    is dropped with its surface area recorded, rather than leaving a duplicate
    high wall at the mask boundary. Larger instability is an explicit error.
    """
    result, dropped, dropped_area = [], 0, 0.
    for weights in pieces:
        actual = (weights @ (original-anchor)).astype(np.float32).astype(float)+anchor
        high = actual[:, 2].max() > threshold
        if high:
            possible, _ = mask.bbox_classes(actual[None, ...])
            if possible[0]:
                residual = clipped_triangle(actual, mask.geometry, threshold)
                if residual is not None and .5-sum(weight_area(p) for p in residual) > 1e-6:
                    area = float(np.linalg.norm(np.cross(actual[1]-actual[0], actual[2]-actual[0]))*.5)
                    if area > .01:
                        raise ValueError(f"Float32 boundary instability exceeds 0.01m2: {area}")
                    dropped += 1; dropped_area += area
                    continue
        result.append(weights)
    return result, dropped, dropped_area


def copied_blob(writer, doc, binary, index):
    view = doc["bufferViews"][index]
    start, size = view.get("byteOffset", 0), view["byteLength"]
    data = binary[start:start+size]
    new = writer.blob(data, view.get("target"))
    data.release()
    release_mapped_pages(binary)
    for key, value in view.items():
        if key not in ("buffer", "byteOffset", "byteLength", "target"):
            writer.doc["bufferViews"][new][key] = copy.deepcopy(value)
    return new


def image_hashes(doc, binary):
    hashes = []
    for image in doc.get("images", []):
        if "bufferView" not in image:
            raise ValueError("Only embedded original image bytes may be copied")
        view = doc["bufferViews"][image["bufferView"]]
        start = view.get("byteOffset", 0)
        hashes.append({"mimeType": image["mimeType"], "bytes": view["byteLength"],
                       "sha256": hashlib.sha256(binary[start:start+view["byteLength"]]).hexdigest()})
        release_mapped_pages(binary)
    return hashes


def filter_glb(source, destination, mask, anchor=(0., 0., 0.), threshold=2., chunk_faces=65536):
    started = time.monotonic()
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if source == destination:
        raise ValueError("UE derivative must have a different output path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    anchor = np.asarray(anchor, float)
    if anchor.shape != (3,) or not np.isfinite(anchor).all() or not np.isfinite(threshold):
        raise ValueError("Finite world anchor and height threshold are required")
    doc, binary, source_hash = read_glb(source)
    active = list(scene_meshes(doc))
    if any(not np.allclose(matrix, np.eye(4), atol=1e-10) for _, _, matrix in active):
        raise ValueError("Expected an already-baked local UE XYZ interchange with identity mesh nodes")
    source_images = image_hashes(doc, binary)
    writer = Writer(directory=destination.parent)
    writer.doc = {k: copy.deepcopy(v) for k, v in doc.items() if k not in ("buffers", "bufferViews")}
    writer.doc["bufferViews"] = []
    writer.doc["asset"]["generator"] = "CityCode UE-only predicted elevated-surface filter v1"
    for i in range(len(doc["bufferViews"])):
        if copied_blob(writer, doc, binary, i) != i:
            raise AssertionError("Original bufferView indices changed")
    records = []
    total_faces = output_faces = clipped_faces = whole_removed = 0
    lows, highs = [], []
    mesh_map, kept_meshes = {}, []
    active_mesh_ids = {mesh_index for _, mesh_index, _ in active}
    for mi, mesh in enumerate(doc["meshes"]):
        if mi not in active_mesh_ids:
            continue
        out_mesh = {**copy.deepcopy(mesh), "primitives": []}
        for pi, primitive in enumerate(mesh["primitives"]):
            if primitive.get("mode", 4) != 4 or primitive.get("extensions") or primitive.get("targets"):
                raise ValueError("Only uncompressed static triangle primitives are supported")
            attrs = {}
            descriptors = {}
            for name, index in primitive["attributes"].items():
                attrs[name], descriptors[name] = accessor(doc, binary, index)
            vertices = attrs["POSITION"]
            if descriptors["POSITION"]["componentType"] != 5126 or vertices.shape[1] != 3:
                raise ValueError("Baked metre POSITION must use unnormalized float VEC3")
            if "indices" in primitive:
                indices, index_descriptor = accessor(doc, binary, primitive["indices"])
                indices = indices.reshape(-1, 3)
            else:
                if len(vertices) % 3:
                    raise ValueError("Non-indexed triangle vertex count is not divisible by three")
                indices = np.arange(len(vertices), dtype=np.uint32).reshape(-1, 3)
                index_descriptor = {"componentType": 5125, "type": "SCALAR"}
            if len(indices) and (indices.min() < 0 or indices.max() >= len(vertices)):
                raise ValueError("Source index exceeds vertex extent")
            keep = np.ones(len(indices), bool)
            extras = {name: [] for name in attrs}
            new_indices = []
            added = 0
            removed_here = clipped_here = ground_here = boundary_here = 0
            float32_artifacts = 0
            float32_artifact_area = 0.
            max_boundary_edge = 0.
            removed_projected_area = 0.
            for start in range(0, len(indices), chunk_faces):
                ix = indices[start:start+chunk_faces]
                world = vertices[ix].astype(float)+anchor
                high = world[:, :, 2].max(1) > threshold
                ground_here += int((~high).sum())
                any_mask, inside_mask = mask.bbox_classes(world)
                remove_fast = high & inside_mask & (world[:, :, 2].min(1) > threshold)
                fast_indices = np.flatnonzero(remove_fast)
                keep[start+fast_indices] = False
                removed_here += len(fast_indices)
                for local in np.flatnonzero(high & any_mask & ~remove_fast):
                    tri = world[local]
                    if inside_mask[local]:
                        pieces = fan(clip_weights(np.eye(3), tri[:, 2], threshold))
                    else:
                        pieces = clipped_triangle(tri, mask.geometry, threshold)
                    if pieces is None:
                        continue
                    pieces, artifacts, artifact_area = stable_serialized_pieces(pieces, tri, mask, anchor, threshold)
                    float32_artifacts += artifacts; float32_artifact_area += artifact_area
                    original_index = start+local
                    keep[original_index] = False
                    if not pieces:
                        removed_here += 1
                        continue
                    clipped_here += 1
                    if not inside_mask[local]:
                        boundary_here += 1
                        edge = float(np.max(np.linalg.norm(tri-np.roll(tri, -1, axis=0), axis=1)))
                        max_boundary_edge = max(max_boundary_edge, edge)
                    xy_area = abs(float(np.linalg.det(np.column_stack((tri[1, :2]-tri[0, :2], tri[2, :2]-tri[0, :2])))))*.5
                    removed_projected_area += xy_area*(1-2*sum(weight_area(p) for p in pieces))
                    for weights in pieces:
                        for name, array in attrs.items():
                            value = weights @ array[ix[local]].astype(float)
                            if name in ("NORMAL", "TANGENT"):
                                value[:, :3] /= np.maximum(np.linalg.norm(value[:, :3], axis=1, keepdims=True), 1e-30)
                            if array.dtype.kind in "iu":
                                value = np.rint(value)
                            extras[name].append(value.astype(array.dtype))
                        new_indices.append(np.arange(len(vertices)+added, len(vertices)+added+3, dtype=np.uint32))
                        added += 3
            retained = indices[keep].astype(np.uint32)
            if new_indices:
                retained = np.concatenate((retained, np.asarray(new_indices, np.uint32)))
            source_count = len(indices)
            output_count = len(retained)
            total_faces += source_count; output_faces += output_count
            clipped_faces += clipped_here; whole_removed += removed_here
            rec = {"mesh_index": mi, "mesh_name": mesh.get("name", ""), "primitive_index": pi,
                   "source_faces": source_count, "unchanged_faces_retained": int(keep.sum()),
                   "input_ground_faces_retained_unchanged": ground_here,
                   "whole_high_faces_removed": removed_here, "input_faces_clipped": clipped_here,
                   "mask_boundary_faces_clipped": boundary_here, "maximum_boundary_edge_m": max_boundary_edge,
                   "boundary_clipping_removed_projected_area_m2": max(0., removed_projected_area),
                   "output_faces": output_count, "added_interpolated_vertices": added}
            rec.update(float32_boundary_artifact_triangles_dropped=float32_artifacts,
                       float32_boundary_artifact_surface_area_m2=float32_artifact_area)
            records.append(rec)
            if not output_count:
                continue
            p = copy.deepcopy(primitive)
            if added:
                for name, array in attrs.items():
                    combined = np.concatenate((array, np.concatenate(extras[name])))
                    descriptor = {**descriptors[name], "count": len(combined)}
                    p["attributes"][name] = writer.array(combined, descriptor)
                    if name == "POSITION":
                        vertices = combined
                del combined
            if removed_here or clipped_here or "indices" not in primitive:
                desc = {**index_descriptor, "componentType": 5125, "type": "SCALAR", "count": retained.size}
                p["indices"] = writer.array(retained.reshape(-1, 1).astype("<u4"), desc, 34963)
            used = np.unique(retained)
            actual = vertices[used].astype(float)+anchor
            lows.append(actual.min(0)); highs.append(actual.max(0))
            out_mesh["primitives"].append(p)
            del attrs, descriptors, vertices, indices, retained, extras, new_indices
            release_mapped_pages(binary)
        if out_mesh["primitives"]:
            mesh_map[mi] = len(kept_meshes)
            kept_meshes.append(out_mesh)
        if (mi+1) % 25 == 0:
            print(f"Filtered {mi+1}/{len(doc['meshes'])} meshes; removed={whole_removed}, clipped={clipped_faces}", flush=True)
    writer.doc["meshes"] = kept_meshes
    nodes, node_map = [], {}
    for i, node in enumerate(doc["nodes"]):
        out_node = copy.deepcopy(node)
        if "mesh" in node:
            if node["mesh"] not in mesh_map:
                if not node.get("children"):
                    continue
                out_node.pop("mesh")
            else:
                out_node["mesh"] = mesh_map[node["mesh"]]
        node_map[i] = len(nodes); nodes.append(out_node)
    for node in nodes:
        if "children" in node:
            node["children"] = [node_map[c] for c in node["children"] if c in node_map]
    writer.doc["nodes"] = nodes
    for scene in writer.doc["scenes"]:
        scene["nodes"] = [node_map[i] for i in scene.get("nodes", []) if i in node_map]
    writer.doc.setdefault("extras", {})["citycode_background_filter"] = {
        "coordinate_system": "local UE XYZ metres; no second axis swap",
        "world_anchor_m": anchor.tolist(), "threshold_world_z_m": threshold,
        "mask_provenance": "predicted class-4 retained semantic XY with bounded margin; no GT bridge transforms",
        "source_background_provenance": "published mixed prediction/reference coastal background, not pure prediction"}
    writer.save(destination)
    output_doc, output_binary, output_hash = read_glb(destination)
    if image_hashes(output_doc, output_binary) != source_images:
        raise ValueError("Original image bytes changed during filtering")
    output_binary.release()
    binary.release()
    if digest_file(source) != source_hash:
        raise ValueError("Read-only source changed during derivative preparation")
    result = {"schema": "citycode.ue_background_prediction_filter.v1", "status": "passed",
              "source_interchange": str(source), "source_interchange_sha256": source_hash,
              "source_unchanged": True, "output_file": str(destination), "output_sha256": output_hash,
              "output_bytes": destination.stat().st_size, "output_face_count": output_faces,
              "output_mesh_count": len(kept_meshes), "output_node_count": len(nodes),
              "output_world_bounds_m": [np.min(lows, axis=0).tolist(), np.max(highs, axis=0).tolist()] if lows else None,
              "input_face_count": total_faces, "input_mesh_count": len(active_mesh_ids),
              "whole_high_faces_removed": whole_removed, "input_faces_clipped": clipped_faces,
              "empty_primitives_dropped": sum(r["output_faces"] == 0 for r in records),
              "empty_meshes_dropped": len(active_mesh_ids)-len(kept_meshes),
              "low_ground_faces_retained_unchanged": sum(r["input_ground_faces_retained_unchanged"] for r in records),
              "float32_boundary_artifact_triangles_dropped": sum(r["float32_boundary_artifact_triangles_dropped"] for r in records),
              "float32_boundary_artifact_surface_area_m2": sum(r["float32_boundary_artifact_surface_area_m2"] for r in records),
              "image_payloads_byte_identical": True, "image_count": len(source_images),
              "image_payloads": source_images, "pred_buffer_m": mask.metadata.get("margin_m"),
              "high_surface_threshold_world_z_m": threshold, "world_anchor_m": anchor.tolist(),
              "coordinate_system": {"world": "UE", "units": "m", "up": "Z"},
              "axis_policy": "Input already baked local UE XYZ metres; add anchor exactly once, no Y/Z swap",
              "triangle_policy": "Exact XY mask overlap; retain z<=threshold and mask-exterior fragments; interpolate UV/attributes barycentrically, preserve winding; vertical triangles use dominant-XY intervals. Drop only newly generated float32-unstable slivers <=0.01m2 each, with total actual surface area audited; original low-ground faces remain unchanged.",
              "source_background_provenance": "Published mixed prediction/reference coastal background; this filter does not make it pure prediction",
              "mask_uses_ground_truth": False, "uses_gt_bridge_transforms": False,
              "peak_process_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              "elapsed_seconds": time.monotonic()-started, "primitives": records}
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    d = ROOT/"build/ue_citycode/interchange"
    p.add_argument("--source", type=Path, default=d/"BACKGROUND_COASTAL.glb")
    p.add_argument("--output", type=Path, default=d/"BACKGROUND_COASTAL_GROUND.glb")
    p.add_argument("--mask", type=Path, default=ROOT/"data/smallcity_p384_scene_enrichment/prediction_elevated_filter_mask.npz")
    p.add_argument("--assembly", type=Path, default=ROOT/"build/ue_citycode/assembly_manifest.json")
    p.add_argument("--threshold", type=float, default=2.)
    args = p.parse_args()
    assembly = json.loads(args.assembly.read_text(encoding="utf-8-sig"))
    record = next(r for r in assembly["meshes"] if r["id"] == "BACKGROUND_COASTAL")
    published_source = ROOT/record["source_file"]
    if digest_file(published_source) != record["source_sha256"]:
        raise ValueError("Published coastal source differs from the assembly's read-only source binding")
    mask = PredictionMask.load(args.mask)
    proof = filter_glb(args.source, args.output, mask, record["anchor_m"], args.threshold)
    if digest_file(published_source) != record["source_sha256"]:
        raise ValueError("Published coastal source changed during derivative filtering")
    proof.update(pred_mask_file=str(args.mask), pred_mask_sha256=digest_file(args.mask),
                 pred_mask_descriptor_sha256=digest_file(args.mask.with_suffix(".json")),
                 published_source_file=record["source_file"], published_source_sha256=record["source_sha256"],
                 published_source_unchanged=True, filter_code_sha256=digest_file(__file__))
    json_write(args.output.with_suffix(".filter.json"), proof)
    print(json.dumps({k: v for k, v in proof.items() if k not in ("primitives", "image_payloads")}, indent=2))


if __name__ == "__main__":
    main()
