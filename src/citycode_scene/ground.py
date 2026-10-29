"""Ground contact evidence for placed buildings and layered background meshes.

Build semantic background height maps, then compare actual low mesh surfaces at
globally aligned cell centres. This measures geometry, not structural stability.
It neither edits placement matrices nor infers ground from a bridge or plinth
bottom. Role selection and the background-to-world transform are explicit inputs.
"""
from __future__ import annotations

from collections import Counter
import ctypes as C
import json
import math
import mmap
from pathlib import Path
import struct
import subprocess
import time

import numpy as np
import shapely
from shapely.geometry import shape

from .geometry import json_write, sha256


DEFAULT_BACKGROUND_TO_WORLD = np.array(
    [[1., 0., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]])
DEFAULT_ROLES = {
    "mesh_roles": {},
    "continuous_name_substrings": ["continuous_base"],
    "solid_names": ["land_solid"],
    "water_names": ["water_plane"],
    "excluded_names": ["geometry_0"],
    "default_role": "generated_detail",
}


def _node_matrix(node):
    if "matrix" in node:
        if any(k in node for k in ("translation", "rotation", "scale")):
            raise ValueError("Background node has both matrix and TRS")
        m = np.asarray(node["matrix"], float).reshape(4, 4).T
    else:
        x, y, z, w = np.asarray(node.get("rotation", [0, 0, 0, 1]), float)
        norm = math.sqrt(x*x+y*y+z*z+w*w)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError("Invalid background quaternion")
        x, y, z, w = x/norm, y/norm, z/norm, w/norm
        r = np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                      [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                      [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
        m = np.eye(4)
        m[:3, :3] = r @ np.diag(node.get("scale", [1, 1, 1]))
        m[:3, 3] = node.get("translation", [0, 0, 0])
    if not np.isfinite(m).all() or not np.allclose(m[3], [0, 0, 0, 1]):
        raise ValueError("Background node transform must be finite and affine")
    return m


class _BackgroundGLB:
    """Memory-map accessors and iterate only the active glTF scene."""

    def __init__(self, path, world_transform, roles):
        self.path = Path(path)
        self.file = self.path.open("rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        if len(self.mm) < 20 or struct.unpack_from("<4sII", self.mm) != (b"glTF", 2, len(self.mm)):
            raise ValueError("Invalid background GLB v2")
        offset = 12
        self.doc = None
        self.binary = None
        while offset < len(self.mm):
            n, kind = struct.unpack_from("<I4s", self.mm, offset)
            offset += 8
            if offset+n > len(self.mm):
                raise ValueError("Background GLB chunk exceeds file length")
            if kind == b"JSON":
                self.doc = json.loads(self.mm[offset:offset+n])
            elif kind == b"BIN\0":
                self.binary, self.binary_length = offset, n
            offset += n
        if self.doc is None or self.binary is None:
            raise ValueError("Background must contain JSON and embedded binary chunks")
        if any(b.get("uri") for b in self.doc.get("buffers", [])):
            raise ValueError("External background buffers are unsupported")
        self.transform = np.asarray(world_transform, float)
        if self.transform.shape != (4, 4) or not np.isfinite(self.transform).all() or not np.allclose(self.transform[3], [0, 0, 0, 1]):
            raise ValueError("Background-to-world transform must be finite affine 4x4")
        if abs(np.linalg.det(self.transform[:3, :3])) < 1e-12:
            raise ValueError("Singular background-to-world transform")
        self.roles = roles
        nodes = self.doc.get("nodes", [])
        scenes = self.doc.get("scenes", [])
        if scenes:
            roots = scenes[self.doc.get("scene", 0)].get("nodes", [])
        else:
            children = {c for n in nodes for c in n.get("children", [])}
            roots = [i for i in range(len(nodes)) if i not in children]
        self.instances = []

        def walk(i, parent, ancestors):
            if i in ancestors or not 0 <= i < len(nodes):
                raise ValueError("Invalid or cyclic background scene graph")
            node = nodes[i]
            if "skin" in node or node.get("weights"):
                raise ValueError("Deformed background geometry requires explicit baking")
            matrix = parent @ _node_matrix(node)
            if "mesh" in node:
                mesh = self.doc["meshes"][node["mesh"]]
                role = self.role(mesh.get("name", ""))
                self.instances.append((i, node["mesh"], mesh, self.transform @ matrix, role))
            for c in node.get("children", []):
                walk(c, matrix, ancestors | {i})

        for root in roots:
            walk(root, np.eye(4), set())

    def close(self):
        self.mm.close()
        self.file.close()

    def role(self, name):
        c = self.roles
        if name in c["mesh_roles"]:
            role = c["mesh_roles"][name]
        elif any(s in name for s in c["continuous_name_substrings"]):
            role = "continuous_ground"
        elif name in c["solid_names"]:
            role = "solid_body"
        elif name in c["water_names"]:
            role = "water"
        elif name in c["excluded_names"]:
            role = "excluded"
        else:
            role = c["default_role"]
        if role not in {"continuous_ground", "generated_detail", "solid_body", "water", "excluded"}:
            raise ValueError(f"Unsupported background semantic role {role!r}")
        return role

    def accessor(self, i):
        a = self.doc["accessors"][i]
        if "sparse" in a or "bufferView" not in a:
            raise ValueError("Sparse/background decompression requires explicit preprocessing")
        bv = self.doc["bufferViews"][a["bufferView"]]
        if bv.get("buffer", 0) != 0:
            raise ValueError("Only the embedded GLB buffer is supported")
        types = {5120: "i1", 5121: "u1", 5122: "<i2", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
        dimensions = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}
        dtype = np.dtype(types[a["componentType"]])
        dim = dimensions[a["type"]]
        stride = bv.get("byteStride", dtype.itemsize * dim)
        start = bv.get("byteOffset", 0) + a.get("byteOffset", 0)
        size = (a["count"]-1)*stride + dtype.itemsize*dim if a["count"] else 0
        if start < 0 or start+size > self.binary_length:
            raise ValueError("Background accessor exceeds binary buffer")
        return np.ndarray((a["count"], dim), dtype=dtype, buffer=self.mm,
                          offset=self.binary+start, strides=(stride, dtype.itemsize))

    def bounds(self, instance):
        _, _, mesh, matrix, _ = instance
        parts = []
        for p in mesh["primitives"]:
            a = self.doc["accessors"][p["attributes"]["POSITION"]]
            if "min" in a and "max" in a:
                lo, hi = a["min"], a["max"]
                v = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            else:
                v = self.accessor(p["attributes"]["POSITION"]).copy()
            parts.append(v @ matrix[:3, :3].T + matrix[:3, 3])
        v = np.concatenate(parts)
        if not np.isfinite(v).all():
            raise ValueError("Nonfinite background bounds")
        return np.array([v.min(0), v.max(0)])

    def primitives(self, instance):
        _, _, mesh, matrix, _ = instance
        for p in mesh["primitives"]:
            if p.get("mode", 4) != 4 or p.get("targets") or "KHR_draco_mesh_compression" in p.get("extensions", {}):
                raise ValueError("Background must use uncompressed static triangle primitives")
            v = self.accessor(p["attributes"]["POSITION"]).copy().astype(np.float64)
            v = np.ascontiguousarray(v @ matrix[:3, :3].T + matrix[:3, 3])
            indices = self.accessor(p["indices"]).ravel().copy() if "indices" in p else np.arange(len(v))
            if len(indices) % 3 or len(indices) and (indices.min() < 0 or indices.max() >= len(v)):
                raise ValueError("Invalid background triangle indices")
            if len(v) > np.iinfo(np.int32).max or not np.isfinite(v).all():
                raise ValueError("Invalid background primitive extent")
            yield v, np.ascontiguousarray(indices.reshape(-1, 3), dtype=np.int32)


class _GroundRaster:
    def __init__(self, workdir):
        source = Path(__file__).with_name("_ground_raster.cpp")
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        library = workdir / ("ground_raster_" + sha256(source)[:16] + ".so")
        if not library.exists():
            temporary = library.with_suffix(".tmp.so")
            subprocess.run(["c++", "-O3", "-shared", "-fPIC", "-std=c++17", str(source), "-o", str(temporary)], check=True)
            temporary.replace(library)
        self.lib = C.CDLL(str(library))
        self.function = self.lib.ground_rasterize
        self.function.argtypes = [C.c_void_p, C.c_void_p, C.c_int64, C.c_int64, C.c_int64,
                                  C.c_double, C.c_double, C.c_double, C.c_int, C.c_double,
                                  C.c_double, C.c_double, C.c_void_p, C.c_void_p, C.c_void_p]
        self.function.restype = None

    def add(self, vertices, faces, origin, cell, mode, maps, normal_z=.25, near_limit=2., fallback_cap=float("nan")):
        v = np.ascontiguousarray(vertices, np.float64)
        f = np.ascontiguousarray(faces, np.int32)
        h, w = maps[0].shape
        self.function(v.ctypes.data, f.ctypes.data, len(f), w, h, *map(float, origin), float(cell),
                      mode, float(normal_z), float(near_limit), float(fallback_cap),
                      *[a.ctypes.data for a in maps])

    def validate(self):
        # A sloped analytic surface checks interpolation and XY/Z coordinate use.
        v = np.array([[0, 0, 1], [4, 0, 1.8], [0, 4, 2.2]], float)
        f = np.array([[0, 1, 2]], np.int32)
        maps = [np.full((4, 4), -np.inf, np.float32) for _ in range(3)]
        self.add(v, f, (0, 0), 1, 0, maps)
        y, x = np.mgrid[:4, :4]
        valid = x+y <= 3
        expected = 1+.2*(x+.5)+.3*(y+.5)
        np.testing.assert_allclose(maps[0][valid], expected[valid], atol=1e-6)
        assert not np.isfinite(maps[0][~valid]).any()
        v[:, 2] += 1
        self.add(v, f, (0, 0), 1, 1, maps)
        np.testing.assert_allclose(maps[1][valid], expected[valid]+1, atol=1e-6)
        v[:, 2] += 5
        self.add(v, f, (0, 0), 1, 1, maps)
        np.testing.assert_allclose(maps[1][valid], expected[valid]+1, atol=1e-6)
        np.testing.assert_allclose(maps[2][valid], expected[valid]+6, atol=1e-6)


def _resolve(root, path):
    path = Path(path)
    return path.resolve() if path.is_absolute() else (Path(root)/path).resolve()


def _geojson_geometry(path):
    d = json.loads(Path(path).read_text())
    if d.get("type") == "Feature":
        return d["geometry"]
    if d.get("type") == "FeatureCollection":
        return shapely.geometry.mapping(shapely.union_all([shape(f["geometry"]) for f in d["features"] if f.get("geometry")]))
    if "geometry" in d:
        return d["geometry"]
    return d


def build_background_maps(release_root, background_path, workdir, *, cell_m=.25,
                          role_config=None, land_path=None, background_transform=None,
                          detail_ground_limit_m=2., minimum_normal_z=.25,
                          maximum_grid_cells=100_000_000):
    """Create reusable layered ground maps; return their JSON metadata dict.

    ``role_config`` can override DEFAULT_ROLES, including explicit mesh-name
    roles. Background transforms are applied after glTF scene-node transforms.
    The default transform converts the published background's (X,Z,Y) storage.
    A land polygon, when available, bounds fallback to the solid body's top cap.
    No recognizable continuous ground is an error, never an implicit highest-
    surface fallback. Output paths are relative to ``metadata_path``.
    """
    started = time.monotonic()
    root, workdir = Path(release_root).resolve(), Path(workdir).resolve()
    source = _resolve(root, background_path)
    workdir.mkdir(parents=True, exist_ok=True)
    metadata_path = workdir / "BACKGROUND_SAMPLING.json"
    if not np.isfinite(cell_m) or cell_m <= 0 or not 0 < minimum_normal_z <= 1 or detail_ground_limit_m < 0:
        raise ValueError("Invalid ground raster policy")
    roles = {**DEFAULT_ROLES, **(role_config or {})}
    transform = DEFAULT_BACKGROUND_TO_WORLD if background_transform is None else np.asarray(background_transform, float)
    land_source = _resolve(root, land_path) if land_path else source.with_name("land.geojson")
    land_source = land_source if land_source.is_file() else None
    identity = {"source_sha256": sha256(source), "cell_m": float(cell_m), "role_config": roles,
                "background_to_world_4x4": transform.tolist(),
                "detail_near_ground_limit_m": float(detail_ground_limit_m),
                "minimum_support_normal_z": float(minimum_normal_z),
                "land_sha256": sha256(land_source) if land_source else None,
                "builder_source_sha256": sha256(__file__),
                "raster_source_sha256": sha256(Path(__file__).with_name("_ground_raster.cpp"))}
    if metadata_path.is_file():
        old = json.loads(metadata_path.read_text())
        if old.get("identity") == identity and all((workdir/f).is_file() for f in old.get("maps", {}).values()) and old.get("maps"):
            return old
    bg = _BackgroundGLB(source, transform, roles)
    try:
        continuous_instances = [i for i in bg.instances if i[4] == "continuous_ground"]
        if not continuous_instances:
            raise ValueError("Background has no explicitly recognized continuous ground; configure semantic mesh roles")
        extent_instances = [i for i in bg.instances if i[4] in {"continuous_ground", "solid_body"}]
        bounds = np.array([bg.bounds(i) for i in extent_instances])
        # Affine rotations can place an otherwise exact grid corner a few ULPs
        # below its integer coordinate. Do not grow an extra border cell then.
        origin = np.floor(bounds[:, 0, :2].min(0)/cell_m+1e-8)*cell_m
        upper = np.ceil(bounds[:, 1, :2].max(0)/cell_m-1e-8)*cell_m
        width, height = np.rint((upper-origin)/cell_m).astype(int)
        if min(width, height) < 1 or int(width)*int(height) > maximum_grid_cells:
            raise ValueError("Ground grid exceeds configured extent; choose a coarser cell or cropped background")
        solid_bounds = [bg.bounds(i) for i in bg.instances if i[4] == "solid_body"]
        solid_cap = float(max(b[1, 2] for b in solid_bounds)) if solid_bounds else None
        maps = []
        names = ["continuous_top", "detail_near_ground_top", "detail_max_top"]
        for name in names:
            a = np.lib.format.open_memmap(workdir/(name+".npy"), mode="w+", dtype="<f4", shape=(int(height), int(width)))
            a[:] = -np.inf
            maps.append(a)
        raster = _GroundRaster(workdir/"native")
        raster.validate()
        records = []
        # Build continuous first so detail can be separated from overhead layers.
        for role, mode in (("continuous_ground", 0), ("generated_detail", 1)):
            for instance in bg.instances:
                if instance[4] != role:
                    continue
                triangles = 0
                for v, f in bg.primitives(instance):
                    triangles += len(f)
                    raster.add(v, f, origin, cell_m, mode, maps, minimum_normal_z,
                               detail_ground_limit_m, solid_cap if solid_cap is not None else float("nan"))
                records.append({"node_index": instance[0], "mesh_index": instance[1],
                                "name": instance[2].get("name", ""), "kind": role, "triangles": triangles})
        for a in maps:
            a.flush()
        try:
            relative_source = str(source.relative_to(root))
        except ValueError:
            relative_source = str(source)
        metadata = {"schema": "citycode.scene_ground_sampling.v2", "status": "background_raster_ready",
                    "source": relative_source, "source_sha256": identity["source_sha256"], "identity": identity,
                    "metadata_path": str(metadata_path), "cell_m": float(cell_m), "origin_xy": origin.tolist(),
                    "shape_yx": [int(height), int(width)], "background_to_world_4x4": transform.tolist(),
                    "coordinate_transform": "active glTF scene-node transforms, then explicit background-to-world affine; world XYZ metres with Z up",
                    "maps": {name: name+".npy" for name in names}, "role_config": roles,
                    "land_geometry": _geojson_geometry(land_source) if land_source else None,
                    "land_source": str(land_source) if land_source else None, "solid_cap_m": solid_cap,
                    "solid_body_used_as_ground": False, "detail_near_ground_limit_m": float(detail_ground_limit_m),
                    "minimum_support_normal_z": float(minimum_normal_z), "numerical_raster_checks_passed": True,
                    "method": "Maximum exact barycentric triangle height per semantic role at metric cell centres. Near-vertical sides excluded. Continuous terrain and generated near-ground detail remain separate; high detail is overhead, never unconditional ground.",
                    "meshes": records, "excluded_mesh_roles": dict(Counter(i[4] for i in bg.instances if i[4] not in {"continuous_ground", "generated_detail"})),
                    "elapsed_seconds": time.monotonic()-started, "API_calls": 0, "GPU_jobs": 0}
        json_write(metadata_path, metadata)
        return metadata
    finally:
        bg.close()


def gap_statistics(values, tolerance_m=.3):
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return None
    q = np.quantile(x, [.05, .25, .5, .75, .95])
    result = {"count": len(x), "min_m": float(x.min()), "p05_m": float(q[0]), "p25_m": float(q[1]),
              "median_m": float(q[2]), "p75_m": float(q[3]), "p95_m": float(q[4]), "max_m": float(x.max()),
              "contact_within_tolerance_fraction": float((np.abs(x) <= tolerance_m).mean()),
              "above_0p50m_fraction": float((x > .5).mean()), "below_minus_0p50m_fraction": float((x < -.5).mean()),
              "above_1m_fraction": float((x > 1).mean()), "below_minus_1m_fraction": float((x < -1).mean())}
    # Stable compatibility key for published 0.30m reports; still literal 0.30m
    # when the caller selects another threshold.
    result["contact_within_0p30m_fraction"] = float((np.abs(x) <= .3).mean())
    return result


def contact_category(s, tolerance_m=.3):
    if s is None:
        return "no_ground_sample"
    if s["p05_m"] > 1:
        return "suspended_over_1m_at_95pct_of_foot"
    if s["p05_m"] > .5:
        return "suspended_over_0p5m_at_95pct_of_foot"
    if s["median_m"] < -1:
        return "embedded_over_1m_at_half_of_foot"
    if s["median_m"] < -.5:
        return "embedded_over_0p5m_at_half_of_foot"
    if s["p05_m"] < -.5 and s["p95_m"] > .5:
        return "mixed_gap_and_embedding"
    if s["median_m"] > .5:
        return "partial_gap_over_0p5m"
    if s["p05_m"] < -tolerance_m or s["p95_m"] > tolerance_m:
        return "local_or_small_height_difference"
    return "within_0p3m_at_90pct_of_foot" if tolerance_m == .3 else "within_contact_tolerance_at_90pct_of_foot"


class GroundMaps:
    """Read existing maps and sample stable/visible/overhead references."""

    def __init__(self, metadata):
        if isinstance(metadata, (str, Path)):
            path = Path(metadata).resolve()
            if path.is_dir():
                path /= "BACKGROUND_SAMPLING.json"
            self.meta = json.loads(path.read_text())
            folder = path.parent
        else:
            self.meta = metadata
            folder = Path(metadata["metadata_path"]).parent
        self.folder = folder
        self.maps = {name: np.load(folder/path, mmap_mode="r") for name, path in self.meta["maps"].items()}
        self.origin = np.array(self.meta["origin_xy"], float)
        self.cell = self.meta["cell_m"]
        self.height, self.width = self.maps["continuous_top"].shape
        self.land = shape(self.meta["land_geometry"]) if self.meta.get("land_geometry") else None

    def sample(self, xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        ij = np.floor((xy-self.origin)/self.cell).astype(np.int64)
        ix, iy = ij[:, 0], ij[:, 1]
        inside = (ix >= 0) & (iy >= 0) & (ix < self.width) & (iy < self.height)
        vals = {name: np.full(len(xy), -np.inf, float) for name in self.maps}
        for name, a in self.maps.items():
            vals[name][inside] = a[iy[inside], ix[inside]]
        stable = vals["continuous_top"]
        detail = vals["detail_near_ground_top"]
        visible = np.maximum(stable, detail)
        on_land = shapely.intersects_xy(self.land, xy[:, 0], xy[:, 1]) if self.land is not None else np.isfinite(stable)
        cap = self.meta.get("solid_cap_m")
        fallback = ~np.isfinite(visible) & on_land
        if cap is None or self.land is None:
            fallback[:] = False
        else:
            visible[fallback] = cap
        stable[~np.isfinite(stable)] = np.nan
        visible[~np.isfinite(visible)] = np.nan
        return {"continuous_z": stable, "scene_ground_z": visible,
                "detail_max_z": vals["detail_max_top"], "detail_near_ground_z": detail,
                "on_land": on_land, "solid_fallback": fallback}


def audit_contacts(cache_dir, ground_map_path, workdir, *, tolerance_m=.3, foot_band_m=.5,
                   minimum_samples=8, low_vertex_confirmation_below=32, save_samples=True):
    """Measure every enabled placement in a cache; write and return its report.

    Actual low surfaces are uniformly sampled within ``minZ + foot_band_m``.
    Global-minimum band coverage is reported as sampling coverage only. It never
    establishes bottom closure, watertightness, solidity or bearing capacity.
    A separate low-vertex sample supports review when the area grid is sparse.
    """
    started = time.monotonic()
    cache_dir, workdir = Path(cache_dir).resolve(), Path(workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    if not np.isfinite(tolerance_m) or not np.isfinite(foot_band_m) or tolerance_m <= 0 or foot_band_m <= 0 or minimum_samples < 1:
        raise ValueError("Contact thresholds and minimum samples must be positive")
    index = json.loads((cache_dir/"INDEX.json").read_text())
    ground = GroundMaps(ground_map_path)
    if not np.isclose(index["cell_m"], ground.cell):
        raise ValueError("Building and background cell sizes must match for aligned support samples")
    if save_samples:
        (workdir/"samples").mkdir(exist_ok=True)
    rows = []
    for entry in index["records"]:
        with np.load(cache_dir/entry["columns"]) as c:
            zmin = c["zmin"]
            base = float(entry["world_bounds"][0][2])
            foot = np.isfinite(zmin) & (zmin <= base+foot_band_m)
            iy, ix = np.nonzero(foot)
            z = zmin[iy, ix].astype(float)
            xy = c["origin"]+(np.column_stack([ix, iy])+.5)*float(c["cell"])
            # Exact point-to-map alignment is required; never silently treat a
            # half-cell displacement as exact contact.
            if len(xy):
                fractional = (xy-ground.origin)/ground.cell-.5
                if not np.allclose(fractional, np.rint(fractional), atol=1e-5):
                    raise ValueError("Building support grid is not globally aligned with background grid")
            sampled = ground.sample(xy)
            cg, actual = sampled["continuous_z"], sampled["scene_ground_z"]
            stable_stats = gap_statistics(z-cg, tolerance_m)
            visible_stats = gap_statistics(z-actual, tolerance_m)
            projected = np.isfinite(zmin)
            projection_count = int(projected.sum())
            lower = zmin[projected]
            lower_stats = gap_statistics(lower, tolerance_m)
            detail = sampled["detail_max_z"]
            near = sampled["detail_near_ground_z"]
            row = {"id": entry["id"], "kind": entry["kind"], "source_ids": entry["source_ids"],
                   "entity_id": entry["entity_id"], "parent_source_id": entry.get("parent_source_id"),
                   "base_min_world_z_m": base, "sampled_bottom_z": gap_statistics(z, tolerance_m),
                   "foot_band_above_min_z_m": foot_band_m, "foot_samples": len(z),
                   "foot_sample_area_m2": len(z)*ground.cell**2,
                   "whole_mesh_projection_area_m2": entry.get("sampled_projection_area_m2", projection_count*ground.cell**2),
                   "bottom_band_projection_fraction": len(z)/max(1, projection_count),
                   "land_support_fraction": float(sampled["on_land"].mean()) if len(z) else None,
                   "continuous_ground_coverage_fraction": float(np.isfinite(cg).mean()) if len(z) else None,
                   "scene_ground_coverage_fraction": float(np.isfinite(actual).mean()) if len(z) else None,
                   "solid_fallback_samples": int(sampled["solid_fallback"].sum()),
                   "continuous_ground_gap": stable_stats, "scene_ground_gap": visible_stats,
                   "continuous_category": contact_category(stable_stats, tolerance_m),
                   "scene_category": contact_category(visible_stats, tolerance_m),
                   "high_background_layer_above_foot_2m_fraction": float((np.isfinite(detail) & (detail > z+2)).mean()) if len(z) else None,
                   "ground_reference_ambiguity_fraction": float((np.isfinite(cg) & np.isfinite(near) & (near-cg > .5)).mean()) if len(z) else None,
                   "source_model_sha256": entry["source_model_sha256"], "geometry_lod": entry["lod"],
                   "crossing_even_fraction": entry.get("column_raster", {}).get("even_crossing_fraction"),
                   "lower_envelope_diagnostic": {"whole_projection": lower_stats,
                       "fraction_within_mesh_min_plus_m": {str(b): float((lower <= base+b).mean()) if len(lower) else 0.
                                                          for b in (.25, .5, 1., 2.)},
                       "interpretation": "Sampling coverage of actual lower-envelope intersections, not a test of bottom closure, solidity, watertightness or structural support."}}
        if len(z) < minimum_samples:
            row["scene_category"] = "insufficient_foot_samples"
        elif np.isfinite(actual).mean() < .95:
            row["scene_category"] = "background_missing_at_part_of_foot"
        if len(z) < low_vertex_confirmation_below:
            with np.load(cache_dir/entry["geometry"]) as data:
                v = data["vertices"]
                v = v[v[:, 2] <= base+foot_band_m]
                bins, inverse = np.unique(np.floor((v[:, :2]-ground.origin)/ground.cell).astype(np.int64), axis=0, return_inverse=True)
                low_z = np.full(len(bins), np.inf)
                np.minimum.at(low_z, inverse, v[:, 2])
                low_xy = ground.origin+(bins+.5)*ground.cell
                low_ground = ground.sample(low_xy)
                row["low_vertex_confirmation"] = {"unique_xy_bins": len(bins),
                    "method": "Minimum actual low vertex per unique metric XY cell; supplementary non-area-weighted evidence",
                    "continuous_ground_gap": gap_statistics(low_z-low_ground["continuous_z"], tolerance_m),
                    "scene_ground_gap": gap_statistics(low_z-low_ground["scene_ground_z"], tolerance_m)}
        rows.append(row)
        if save_samples:
            np.savez_compressed(workdir/"samples"/(entry["id"]+".npz"), xy=xy.astype("<f4"), bottom_z=z.astype("<f4"),
                                continuous_z=cg.astype("<f4"), scene_ground_z=actual.astype("<f4"),
                                detail_max_z=detail.astype("<f4"), on_land=sampled["on_land"])
    summary = {"instances": len(rows), "scene_categories": dict(Counter(r["scene_category"] for r in rows)),
               "continuous_categories": dict(Counter(r["continuous_category"] for r in rows)),
               "kind_counts": dict(Counter(r["kind"] for r in rows)),
               "instances_with_sampled_foot_but_no_ground": sum(r["foot_samples"] > 0 and r["scene_ground_coverage_fraction"] == 0 for r in rows),
               "foot_sampling_under_10pct_projection": sum(r["bottom_band_projection_fraction"] < .1 for r in rows)}
    report = {"schema": "citycode.ground_contact_audit.v2", "status": "measured_read_only",
              "scope": "All enabled actual model placements in the supplied cache; observation reference surfaces excluded",
              "binding_sha256": index["binding_sha256"], "background_sha256": ground.meta["source_sha256"],
              "background_sampling_metadata": str(ground.folder/"BACKGROUND_SAMPLING.json"), "cell_m": ground.cell,
              "method": "Globally aligned metric cell-centre lower triangle intersections within mesh minZ plus foot band; continuous terrain and nearby detail independently reported.",
              "tolerances": {"small_contact_m": tolerance_m, "review_gap_or_embedding_m": .5,
                             "strong_gap_or_embedding_m": 1., "foot_band_m": foot_band_m},
              "limitations": ["Surface geometry measurements are not a rigid-body physics/stability simulation.",
                              "A small fraction near global minimum is sampling coverage, not proof of an open or missing bottom.",
                              "Background detail can contain local objects; never lift whole buildings using detail alone.",
                              "Sparse/stepped/sloping bases and uncapped/nonmanifold geometry require evidence review.",
                              "Recompute support after XY or background changes; old vertical predictions do not transfer automatically."],
              "summary": summary, "records": rows, "elapsed_seconds": time.monotonic()-started,
              "API_calls": 0, "GPU_jobs": 0}
    json_write(workdir/"GROUND_CONTACTS.json", report)
    return report


def ground_proposals(contact_report, policy=None):
    """Return deterministic, review-required Z proposals for arbitrary inputs.

    This function changes no files. It conserves common logical building parts
    and attached parent/wing heights. All final poses must be re-audited after
    horizontal correction; a proposal is not an approval or a physics verdict.
    Quantile predictions are simple same-XY translations, never invented new
    contact fractions. ``rigid_groups`` may explicitly bind additional IDs.
    """
    if isinstance(contact_report, (str, Path)):
        contact_report = json.loads(Path(contact_report).read_text())
    cfg = {"contact_tolerance_m": .3, "clear_gap_m": .5,
           "clear_visible_embedding_m": .5, "continuous_embedding_m": .45,
           "maximum_ground_median_disagreement_m": .15,
           "maximum_ground_shift_m": 1.2, "minimum_area_samples": 8,
           "vertex_confirmation_below_area_samples": 32,
           "minimum_vertex_confirmation_bins": 32, "quantize_m": .005,
           "preserve_attached_grounding": True, "rigid_groups": [], **(policy or {})}
    tol = float(cfg["contact_tolerance_m"])
    step = float(cfg["quantize_m"])
    if not 0 < tol or not 0 < step or not 0 < cfg["maximum_ground_shift_m"]:
        raise ValueError("Positive grounding proposal tolerances are required")
    rows = {r["id"]: r for r in contact_report["records"]}
    if len(rows) != len(contact_report["records"]):
        raise ValueError("Ground contact placement IDs must be unique")
    parents = {aid: aid for aid in rows}

    def find(aid):
        while aid != parents[aid]:
            parents[aid] = parents[parents[aid]]
            aid = parents[aid]
        return aid

    def union(ids):
        ids = list(ids)
        if any(aid not in rows for aid in ids):
            raise ValueError("Unknown placement in grounding rigid_groups")
        for aid in ids[1:]:
            a, b = find(ids[0]), find(aid)
            parents[max(a, b)] = min(a, b)

    entities = {}
    source_to_base = {}
    for aid, row in rows.items():
        entities.setdefault(row.get("entity_id") or aid, []).append(aid)
        if row.get("kind") == "base":
            for source in row.get("source_ids", []):
                source_to_base[source] = aid
    for ids in entities.values():
        union(ids)
    if cfg["preserve_attached_grounding"]:
        for aid, row in rows.items():
            parent = source_to_base.get(row.get("parent_source_id"))
            if parent:
                union([aid, parent])
    for ids in cfg["rigid_groups"]:
        union(ids)
    groups = {}
    for aid in sorted(rows):
        groups.setdefault(find(aid), []).append(aid)

    triggers = {}
    reviews = []
    for aid, row in rows.items():
        sg, cg = row.get("scene_ground_gap"), row.get("continuous_ground_gap")
        if not sg or not cg or row["foot_samples"] < cfg["minimum_area_samples"]:
            if row["foot_samples"] < cfg["minimum_area_samples"]:
                reviews.append({"placement_ids": [aid], "reason": "Insufficient uniform foot-area samples; keep unchanged pending geometry evidence."})
            continue
        if (row.get("continuous_ground_coverage_fraction") or 0) < .95 or (row.get("scene_ground_coverage_fraction") or 0) < .95:
            reviews.append({"placement_ids": [aid], "reason": "Missing ground coverage; no global shift until the support reference is established."})
            continue
        reason = None
        if sg["median_m"] > cfg["clear_gap_m"] and cg["median_m"] > cfg["clear_gap_m"]:
            reason = "floating"
        elif sg["median_m"] < -cfg["clear_visible_embedding_m"] and cg["median_m"] < -cfg["continuous_embedding_m"] and abs(sg["median_m"]-cg["median_m"]) < cfg["maximum_ground_median_disagreement_m"]:
            reason = "embedded"
        elif (sg["median_m"] < -.5 and cg["median_m"] >= -.45) or (sg.get("below_minus_1m_fraction", 0) > .05 and cg["median_m"]-sg["median_m"] > .25):
            reviews.append({"placement_ids": [aid], "reason": "Embedding is driven by local generated detail rather than the continuous floor; review local background instead of lifting the building."})
        if reason is None:
            continue
        if row["foot_samples"] < cfg["vertex_confirmation_below_area_samples"]:
            v = row.get("low_vertex_confirmation") or {}
            vs = v.get("scene_ground_gap") if reason == "floating" else v.get("continuous_ground_gap")
            agrees = vs and (vs["median_m"] > cfg["clear_gap_m"] if reason == "floating" else vs["median_m"] < -cfg["continuous_embedding_m"])
            if v.get("unique_xy_bins", 0) < cfg["minimum_vertex_confirmation_bins"] or not agrees:
                reviews.append({"placement_ids": [aid], "reason": "Sparse foot-area trigger lacks independent low-vertex support confirmation; keep unchanged."})
                continue
        triggers[aid] = reason

    proposals = []
    for ids in groups.values():
        active = {aid: triggers[aid] for aid in ids if aid in triggers}
        if not active:
            continue
        signs = set(active.values())
        if len(signs) != 1:
            reviews.append({"placement_ids": ids, "reason": "Rigid group has opposing floating and embedding triggers; do not break relative assembly heights."})
            continue
        reason = next(iter(signs))
        field = "scene_ground_gap" if reason == "floating" else "continuous_ground_gap"
        metrics = [rows[aid].get(field) for aid in ids]
        if any(s is None or rows[aid]["foot_samples"] < cfg["minimum_area_samples"]
               or (rows[aid].get("continuous_ground_coverage_fraction") or 0) < .95
               or (rows[aid].get("scene_ground_coverage_fraction") or 0) < .95
               for aid, s in zip(ids, metrics)):
            reviews.append({"placement_ids": ids, "reason": "A connected part lacks area support evidence; no common rigid shift is approved."})
            continue
        low = max(-tol-s["p05_m"] for s in metrics)
        high = min(tol-s["p95_m"] for s in metrics)
        if len(ids) > 1:
            feasible_low = max(low, -cfg["maximum_ground_shift_m"])
            feasible_high = min(high, cfg["maximum_ground_shift_m"])
            grid_low = math.ceil(feasible_low/step-1e-8)*step
            grid_high = math.floor(feasible_high/step+1e-8)*step
            if grid_low > grid_high+1e-8:
                reviews.append({"placement_ids": ids, "reason": "Connected parts have no common 5th–95th percentile interval within tolerance and shift budget; preserve their relative transforms and review locally.",
                                "candidate_interval_m": [float(low), float(high)]})
                continue
            dz = min(grid_high, max(grid_low, round((feasible_low+feasible_high)/(2*step))*step))
        elif reason == "floating":
            ideal = max(-metrics[0]["median_m"], low)
            dz = math.ceil(ideal/step-1e-8)*step
        else:
            ideal = min(-metrics[0]["median_m"], high)
            dz = math.floor(ideal/step+1e-8)*step
        dz = round(float(dz), 9)
        if not np.isfinite(dz) or abs(dz) > cfg["maximum_ground_shift_m"] or reason == "floating" and dz >= 0 or reason == "embedded" and dz <= 0:
            reviews.append({"placement_ids": ids, "reason": "No signed vertical correction satisfies the reviewed shift budget."})
            continue
        predicted = []
        for aid in ids:
            member = {"id": aid, "source_model_sha256": rows[aid]["source_model_sha256"], "before": {}, "predicted_same_xy": {}}
            for name in ("continuous_ground_gap", "scene_ground_gap"):
                s = rows[aid][name]
                member["before"][name] = {q: s[q] for q in ("p05_m", "median_m", "p95_m")}
                member["predicted_same_xy"][name] = {q: float(s[q]+dz) for q in ("p05_m", "median_m", "p95_m")}
            predicted.append(member)
        proposals.append({"id": "GROUND_Z_"+ids[0], "action": "translate_world_z", "placement_ids": ids,
                          "delta_xyz_m": [0., 0., dz], "triggered_by": active, "ground_reference": field,
                          "preserve_relative_transforms": len(ids) > 1,
                          "common_90pct_support_interval_m": [float(low), float(high)],
                          "selection_reason": "Both ground references confirm floating; lower with the -tolerance tail protected." if reason == "floating" else "Continuous ground independently confirms embedding; raise with the +tolerance tail protected.",
                          "members": predicted})
    return {"schema": "citycode.ground_z_draft.v1", "status": "draft_requires_agent_review",
            "input_binding_sha256": contact_report["binding_sha256"],
            "input_background_sha256": contact_report["background_sha256"], "policy": cfg,
            "operations": proposals, "review_cases": reviews,
            "summary": {"placements": len(rows), "proposed_rigid_operations": len(proposals),
                        "proposed_adjusted_placements": sum(len(o["placement_ids"]) for o in proposals),
                        "manual_review_cases": len(reviews)},
            "preconditions": ["Agent review is required before freezing or replay.",
                              "Refresh this report after any XY/scale/background change.",
                              "After applying accepted Z, recheck all road clearances and building collisions.",
                              "Quantile predictions hold at the same XY samples; final contact fractions require fresh sampling."]}
