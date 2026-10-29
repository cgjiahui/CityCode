"""Read-only actual-triangle QA of candidate elevated geometry and buildings."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from interchange import accessor, digest_file, read_glb
from prepare_elevated_repair import ROOT, height_hits, write


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def load_triangles(path, anchor, native_gltf=False):
    doc, binary, sha = read_glb(path); triangles = []
    for mesh in doc["meshes"]:
        for primitive in mesh["primitives"]:
            points, _ = accessor(doc, binary, primitive["attributes"]["POSITION"])
            if "indices" in primitive:
                indices, _ = accessor(doc, binary, primitive["indices"])
                points = points[indices.reshape(-1)]
            if native_gltf:
                points = points[:, [0, 2, 1]]
            current = points.reshape(-1, 3, 3).astype(float) + np.asarray(anchor)
            if native_gltf:
                current = current[:, [0, 2, 1]]
            triangles.append(current)
    return np.concatenate(triangles), sha


def triangle_intersections(triangle, candidates, tolerance=1e-7):
    """Separating axes include in-plane axes for coplanar triangles."""
    ae = np.roll(triangle, -1, axis=0)-triangle
    be = np.roll(candidates, -1, axis=1)-candidates
    an = np.cross(ae[0], ae[1]); bn = np.cross(be[:, 0], be[:, 1])
    axes = [np.broadcast_to(an, (len(candidates), 3)), bn]
    axes.extend(np.cross(np.broadcast_to(edge, (len(candidates), 3)), be[:, j]) for edge in ae for j in range(3))
    axes.extend(np.broadcast_to(np.cross(an, edge), (len(candidates), 3)) for edge in ae)
    axes.extend(np.cross(bn, be[:, j]) for j in range(3))
    # Quantized union output and reconstructed building GLBs can contain
    # collapsed triangles. They have no surface and cannot prove a collision.
    intersects = (np.linalg.norm(bn, axis=1) > 1e-12) & (np.linalg.norm(an) > 1e-12)
    for axis in axes:
        length = np.linalg.norm(axis, axis=1); nonzero = length > 1e-12
        axis = axis/np.maximum(length[:, None], 1e-30)
        pa, pb = triangle@axis.T, np.einsum("nij,nj->ni", candidates, axis)
        separate = (pa.max(0) < pb.min(1)-tolerance) | (pb.max(1) < pa.min(0)-tolerance)
        intersects &= ~(separate & nonzero)
    return intersects


class TriangleBoxes:
    """Conservative XY grid; exact 3D boxes and SAT remain authoritative."""
    def __init__(self, triangles, cell_size=2.):
        self.low, self.high = triangles.min(1), triangles.max(1)
        self.cell_size = float(cell_size); self.cells = {}; self.large = []
        lo = np.floor(self.low[:, :2]/self.cell_size).astype(np.int64)
        hi = np.floor(self.high[:, :2]/self.cell_size).astype(np.int64)
        for index, (a, b) in enumerate(zip(lo, hi)):
            if np.prod(b-a+1) > 256:
                self.large.append(index)
                continue
            for x in range(a[0], b[0]+1):
                for y in range(a[1], b[1]+1):
                    self.cells.setdefault((x, y), []).append(index)

    def query(self, box):
        a, b = np.floor(np.asarray(box)[:, :2]/self.cell_size).astype(np.int64)
        chosen = set(self.large)
        if np.prod(b-a+1) > 4096:
            return np.flatnonzero(intersects_box(self.low, self.high, box))
        for x in range(a[0], b[0]+1):
            for y in range(a[1], b[1]+1):
                chosen.update(self.cells.get((x, y), ()))
        indices = np.asarray(sorted(chosen), dtype=np.int64)
        return indices[intersects_box(self.low[indices], self.high[indices], box)]


def contact_points(a, b):
    result = []
    for edges, other in ((a, b), (b, a)):
        ab, ac = other[1]-other[0], other[2]-other[0]
        n = np.cross(ab, ac); n /= max(np.linalg.norm(n), 1e-30)
        d00, d01, d11 = np.dot(ab, ab), np.dot(ab, ac), np.dot(ac, ac)
        determinant = d00*d11-d01*d01
        if abs(determinant) < 1e-20:
            continue
        for p, q in zip(edges, np.roll(edges, -1, axis=0)):
            denominator = np.dot(n, q-p)
            if abs(denominator) < 1e-12:
                continue
            t = np.dot(n, other[0]-p)/denominator
            if not -1e-7 <= t <= 1+1e-7:
                continue
            point = p+t*(q-p); delta = point-other[0]
            u = (d11*np.dot(delta, ab)-d01*np.dot(delta, ac))/determinant
            v = (d00*np.dot(delta, ac)-d01*np.dot(delta, ab))/determinant
            if u >= -1e-7 and v >= -1e-7 and u+v <= 1+1e-7:
                result.append(point)
    return result


def intersects_box(low, high, box):
    return np.all(high >= box[0]-1e-7, axis=-1) & np.all(low <= box[1]+1e-7, axis=-1)


def containment_probes(triangles, samples):
    occupied, open_columns = [], 0
    lo, hi = triangles[:, :, :2].min(1), triangles[:, :, :2].max(1)
    for point in samples:
        chosen = np.all(lo <= point[:2], axis=1) & np.all(hi >= point[:2], axis=1)
        heights = np.sort(height_hits(triangles[chosen], point[:2]))
        heights = heights[np.r_[True, np.diff(heights) > .001]] if len(heights) else heights
        if len(heights) % 2:
            open_columns += 1
        elif any(a+.001 < point[2] < b-.001 for a, b in zip(heights[::2], heights[1::2])):
            occupied.append(point.tolist())
    return occupied, open_columns


def audit(source_manifest, joined_manifest, building_manifest, output, *, only_buildings=None, proposal=None, ground=None):
    source_manifest, joined_manifest, building_manifest = map(Path, (source_manifest, joined_manifest, building_manifest))
    before = {str(path): digest_file(path) for path in (source_manifest, joined_manifest, building_manifest)}
    source, joined, buildings = map(read, (source_manifest, joined_manifest, building_manifest))
    source_boxes = []
    for row in source["templates"]:
        triangles, sha = load_triangles(source_manifest.parent/row["template_file"], row["position_m"], True)
        if sha != row["template_sha256"]:
            raise ValueError("Source span differs from its manifest: " + row["span_id"])
        actual = np.asarray([triangles.min((0, 1)), triangles.max((0, 1))])
        declared = np.asarray(row["expected_native_bounds_m"])+np.asarray(row["position_m"])
        if np.max(abs(actual-declared)) > 2e-5:
            raise ValueError("Source span actual bounds differ from its manifest: " + row["span_id"])
        source_boxes.append(actual)
    source_boxes = np.asarray(source_boxes)
    chunks = []
    for row in joined["templates"]:
        triangles, sha = load_triangles(joined_manifest.parent/row["template_file"], row["position_m"], True)
        if sha != row["template_sha256"]:
            raise ValueError("Joined mesh differs from its manifest")
        chunks.append(triangles)
    deck = np.concatenate(chunks); deck_low, deck_high = deck.min(1), deck.max(1)
    all_rows = [row for row in buildings["meshes"] if row["kind"] in ("building", "massing", "surface")]
    rows = [row for row in all_rows if not only_buildings or row["id"] in only_buildings]
    details, pairs, possible_inside = [], 0, 0
    for index, row in enumerate(rows):
        box = np.asarray(row["prepared"]["world_bounds_m"])
        broad = intersects_box(source_boxes[:, 0], source_boxes[:, 1], box)
        # The joined mesh is authoritative: junction patches can extend beyond
        # an individual source span's declared corridor.
        candidates = np.flatnonzero(intersects_box(deck_low, deck_high, box))
        if not len(candidates):
            details.append({"building_id": row["id"], "status": "clear_actual_geometry_aabbs", "surface_intersection_pairs": 0})
            continue
        building, sha = load_triangles(building_manifest.parent/row["import_file"], row["anchor_m"])
        if sha != row["prepared"]["interchange_sha256"]:
            raise ValueError("Prepared building differs from its frozen manifest: " + row["id"])
        body_index = TriangleBoxes(building)
        contacts, points = [], []
        for deck_index in candidates:
            triangle = deck[deck_index]
            selected = body_index.query(np.asarray([triangle.min(0), triangle.max(0)]))
            if not len(selected):
                continue
            selected = selected[triangle_intersections(triangle, building[selected])]
            for triangle_index in selected:
                contacts.append([int(deck_index), int(triangle_index)])
                points.extend(contact_points(triangle, building[triangle_index]))
        samples = deck[candidates].mean(1)
        samples = samples[np.linspace(0, len(samples)-1, min(64, len(samples))).astype(int)] if len(samples) else samples
        occupied, odd = containment_probes(building, samples)
        p = np.asarray(points).reshape(-1, 3)
        details.append({"building_id": row["id"], "prepared_sha256": sha,
            "status": "actual_surface_contact" if contacts else ("possible_closed_column_containment" if occupied else "no_actual_surface_contact"),
            "source_span_aabb_candidates": [source["templates"][i]["span_id"] for i in np.flatnonzero(broad)],
            "surface_intersection_pairs": len(contacts), "intersecting_triangle_indices": contacts,
            "contact_bounds_m": [p.min(0).tolist(), p.max(0).tolist()] if len(p) else None,
            "sampled_containment_points_m": occupied, "sampled_open_column_count": odd,
            "sampled_containment_probe_count": len(samples)})
        pairs += len(contacts); possible_inside += bool(occupied)
        if (index+1) % 20 == 0:
            print("Audited actual candidate/building geometry %d/%d" % (index+1, len(rows)), flush=True)
    result = {"schema": "citycode.actual_elevated_building_audit.v1", "ue_executed": False, "read_only": True,
        "status": "requires_geometry_review" if pairs or possible_inside else "no_actual_surface_crossings_or_sampled_containment_found",
        "source_manifest_sha256": before[str(source_manifest)], "joined_manifest_sha256": before[str(joined_manifest)],
        "building_manifest_sha256": before[str(building_manifest)], "building_input_count": len(rows),
        "all_building_input_count": len(all_rows), "subset": bool(only_buildings), "joined_triangle_count": len(deck),
        "actual_surface_intersection_pair_count": pairs, "possible_containment_building_count": possible_inside, "records": details,
        "method": "Exact prepared world vertices and 3D triangle SAT including coplanar axes; sampled vertical occupancy is separate and odd/open columns remain uncertain",
        "not_checked": "Traffic reachability, structural loading, and unsampled solid containment in open/nonmanifold generated buildings"}
    if proposal and ground:
        from bridge_supports import GroundMaps
        maps = GroundMaps(ground); ground_rows = []
        for span in read(proposal)["continuous_segments"]:
            profile = span["profile"]; points = np.asarray([p["position_m"] for p in profile])
            measured = maps.sample(points[:, :2]); delta = points[:, 2]-measured["scene_ground_z"]
            finite = np.isfinite(delta) & ~measured["solid_fallback"]
            ground_rows.append({"span_id": span["segment_id"], "ground_transition": bool(span.get("ground_transition")),
                "observed_profile_samples": int(finite.sum()), "unsupported_samples": int((~finite).sum()),
                "minimum_top_above_existing_scene_ground_m": float(delta[finite].min()) if finite.any() else None,
                "landing_ground_z_m": span.get("landing_ground_z_m"), "landing_contact_xy_m": span.get("landing_contact_xy_m")})
        result["ground_profile_checks"] = {"method": "25cm existing exact-triangle-derived semantic ground cache; intentional approach embedding is reported, not erased",
            "ground_metadata_sha256": digest_file(Path(ground)/"BACKGROUND_SAMPLING.json"), "records": ground_rows}
    if any(digest_file(path) != value for path, value in before.items()):
        raise ValueError("Candidate or baseline changed during audit; rerun after producers freeze")
    write(output, result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-spans", type=Path, required=True); p.add_argument("--joined", type=Path, required=True)
    p.add_argument("--buildings", type=Path, default=ROOT/"build/ue_citycode/assembly_manifest.json")
    p.add_argument("--only-building", action="append"); p.add_argument("--proposal", type=Path)
    p.add_argument("--ground", type=Path); p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    result = audit(a.source_spans, a.joined, a.buildings, a.output,
        only_buildings=a.only_building, proposal=a.proposal, ground=a.ground)
    print(json.dumps({k: result[k] for k in ("status", "building_input_count", "joined_triangle_count", "actual_surface_intersection_pair_count", "possible_containment_building_count")}, indent=2))


if __name__ == "__main__":
    main()
