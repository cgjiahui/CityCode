"""Prediction-only height correction and the standard three-stage generator.

All paths are caller-supplied. This module never imports research evaluation
entry points, reads reference building geometry, or downloads model resources.
"""
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from shapely import affinity, set_precision
from shapely.geometry import Polygon, box, mapping, shape
from shapely.ops import unary_union
from shapely.validation import make_valid

from .appearance import sha256_file, write_json

def polygons(g):
 if g.geom_type=='Polygon':return [g]
 return [p for x in getattr(g,'geoms',[]) for p in polygons(x)]

def clean(g):
 g=g if g.is_valid else g.buffer(0)
 return unary_union(polygons(g))

def norm(g):
 c=g.centroid;s=math.hypot(g.bounds[2]-g.bounds[0],g.bounds[3]-g.bounds[1]);return affinity.scale(affinity.translate(g,-c.x,-c.y),1/s,1/s,origin=(0,0)),[c.x,c.y,s]

def transform(p,x):
 a,s,dx,dy=x;return affinity.translate(affinity.scale(affinity.rotate(p,a,origin=(0,0)),s,s,origin=(0,0)),dx,dy)

def iou(p,t):
 a=p.intersection(t).area;return a/(p.area+t.area-a)

def boundary(g,n=512):
 lines=[]
 for p in polygons(g):lines.extend([p.exterior,*p.interiors])
 length=sum(l.length for l in lines);pts=[]
 for l in lines:
  k=max(4,round(n*l.length/length));pts.extend([l.interpolate(i/k,normalized=True).coords[0] for i in range(k)])
 return np.asarray(pts)

def fit(pred,target):
 p,pn=norm(pred);t,tn=norm(target)
 # Simplification only in optimization; final metrics use unsimplified contours.
 ps=p.simplify(.001,preserve_topology=True);ts=t.simplify(.001,preserve_topology=True)
 starts=sorted([(iou(transform(ps,[a,1,0,0]),ts),a) for a in range(0,360,5)],reverse=True)[:3]
 candidates=[]
 for score,a in starts:
  res=minimize(lambda x:1-iou(transform(ps,x),ts),[a,1,0,0],method='Powell',bounds=[(a-10,a+10),(.5,2),(-.5,.5),(-.5,.5)],options={'maxiter':35,'xtol':.0005,'ftol':.00005})
  candidates.extend([[a,1,0,0],res.x.tolist()])
 x=max(candidates,key=lambda v:iou(transform(p,v),t));aligned=transform(p,x);inter=aligned.intersection(t).area
 pa,ta=boundary(aligned),boundary(t);dist=(cKDTree(pa).query(ta)[0].mean()+cKDTree(ta).query(pa)[0].mean())/2;diag=math.hypot(t.bounds[2]-t.bounds[0],t.bounds[3]-t.bounds[1])
 return dict(iou=iou(aligned,t),spill=(aligned.area-inter)/aligned.area,coverage=inter/t.area,boundary_distance_normalized=dist/diag,alignment=dict(rotation_deg=float(x[0]%360),scale_after_area_normalization=float(x[1]),translation=[float(x[2]),float(x[3])],prediction_normalization=pn,target_normalization=tn),target=mapping(t),prediction=mapping(aligned))

def normalize_geometry(g, margin=0.08):
    """Center and uniformly fit a metric footprint into [-.5,.5]^2."""
    minx, miny, maxx, maxy = g.bounds
    extent = max(maxx - minx, maxy - miny)
    if not extent > 0:
        raise ValueError("Degenerate control footprint")
    center = ((minx + maxx) / 2, (miny + maxy) / 2)
    scale = (1 - 2 * margin) / extent
    out = affinity.scale(affinity.translate(g, -center[0], -center[1]),
                         scale, scale, origin=(0, 0))
    return out, center, scale

def align_control_to_baseline(bundle, baseline_coords):
    """Resolve arbitrary model-frame yaw against the frozen image-only sample."""
    if not len(baseline_coords):
        # A small number of direct-crop observations produce no stock Stage-1
        # occupancy at seed 42.  There is no image-derived model frame to fit in
        # that case, so retain the already centered, uniformly normalized
        # predicted control.  This is explicit and preferable to borrowing the
        # orientation cache from a different input image.
        bundle = dict(bundle)
        bundle["normalized_model_frame"] = bundle["normalized"]
        bundle["model_frame_alignment"] = {
            "fallback": True,
            "method": "centered predicted control because same-image baseline occupancy is empty",
            "rotation_deg": 0.0,
            "scale_after_area_normalization": 1.0,
            "translation": [0.0, 0.0],
        }
        return bundle
    xy = np.unique(baseline_coords[:, :2], axis=0)
    pred = unary_union([box(x / 32 - .5, y / 32 - .5,
                            (x + 1) / 32 - .5, (y + 1) / 32 - .5) for x, y in xy])
    # fit(source, target) transforms source to target.  We need control -> model frame.
    control_clean = clean(make_valid(set_precision(bundle["normalized"], 1e-7)))
    pred_clean = clean(make_valid(set_precision(pred, 1e-7)))
    try:
        result = fit(control_clean, pred_clean)
    except Exception as exc:
        # Dense recovered contours occasionally contain numerically fragile
        # courtyard rings: rotating the simplified polygon inside fit() can
        # trigger GEOS "free hole"/"side location conflict" errors even after
        # make_valid.  Alignment only needs the exterior support, so retry with
        # solid exterior shells.  The fallback is recorded in the audit data.
        def exterior_support(g):
            if g.geom_type == "Polygon":
                return Polygon(g.exterior.coords)
            return unary_union([exterior_support(part) for part in g.geoms
                                if part.geom_type in ("Polygon", "MultiPolygon")])
        try:
            result = fit(clean(make_valid(exterior_support(control_clean))),
                         clean(make_valid(exterior_support(pred_clean))))
            fallback_method = "exterior-shell support for model-frame alignment"
        except Exception:
            # A convex proxy is the final deterministic fallback for extremely
            # fragile multi-component rings. Use it only to estimate the
            # similarity transform, then apply that transform to the original
            # repaired footprint so its concavities are retained in the mask.
            result = fit(control_clean.convex_hull, pred_clean.convex_hull)
            a = result["alignment"]
            pcx, pcy, pscale = a["prediction_normalization"]
            original_normalized = affinity.scale(
                affinity.translate(control_clean, -pcx, -pcy),
                1 / pscale, 1 / pscale, origin=(0, 0))
            aligned_original = affinity.translate(
                affinity.scale(
                    affinity.rotate(original_normalized, a["rotation_deg"], origin=(0, 0)),
                    a["scale_after_area_normalization"],
                    a["scale_after_area_normalization"], origin=(0, 0)),
                *a["translation"])
            result["prediction"] = mapping(clean(make_valid(aligned_original)))
            fallback_method = "convex-proxy similarity fit applied to original repaired support"
        result["alignment"]["topology_fallback"] = {
            "used": True,
            "method": fallback_method,
            "original_error": repr(exc),
        }
    a = result["alignment"]
    # fit() returns both geometries in the target's normalized comparison
    # frame. Convert the aligned control back to the actual TRELLIS grid frame
    # before rasterizing it; otherwise the mask is spuriously centered at zero.
    aligned_normalized = shape(result["prediction"])
    tcx, tcy, tscale = a["target_normalization"]
    aligned = affinity.translate(
        affinity.scale(aligned_normalized, tscale, tscale, origin=(0, 0)),
        tcx, tcy,
    )
    aligned = clean(make_valid(set_precision(aligned, 1e-7)))
    bundle = dict(bundle)
    bundle["normalized_model_frame"] = aligned
    # Preserve the supplied metric height/sqrt(area), but express its height
    # in the final model frame after the control footprint has been registered.
    height_fraction = bundle["ratio"] * math.sqrt(float(aligned.area))
    bundle["height_voxels"] = max(1, min(32, int(round(height_fraction * 32))))
    bundle["model_frame_alignment"] = a
    return bundle

def image_height_span(config: dict) -> int | None:
    """Validate the explicit image-only Z-span mode without loading a model."""
    if config.get("structure_conditioning") != "image_height_span":
        return None
    span = config.get("structure_height_voxels")
    if type(span) is not int or not 1 <= span <= 32:
        raise ValueError("image_height_span requires structure_height_voxels as an integer from1 to32")
    return span


def stretch_height(coords, requested_height):
    source = np.unique(np.asarray(coords, np.int32), axis=0)
    old_min, old_max = int(source[:, 2].min()), int(source[:, 2].max())
    center = (old_min + old_max) / 2
    new_min = max(0, int(round(center - (requested_height - 1) / 2)))
    new_max = min(31, new_min + requested_height - 1)
    new_min = max(0, new_max - requested_height + 1)
    old_span = max(1, old_max - old_min)

    def mapped(z):
        values = new_min + (np.asarray(z, np.float64) - old_min) / old_span * (new_max - new_min)
        return np.clip(np.rint(values), new_min, new_max).astype(np.int32)

    result = {(int(x), int(y), int(z)) for (x, y), z in
              zip(source[:, :2], mapped(source[:, 2]))}
    by_xy = defaultdict(list)
    for x, y, z in source:
        by_xy[(int(x), int(y))].append(int(z))
    before_fill = len(result)
    for (x, y), zs in by_xy.items():
        ordered = sorted(set(zs))
        for z0, z1 in zip(ordered, ordered[1:]):
            # Never close a gap that already existed in the image-only sample.
            if z1 != z0 + 1:
                continue
            a, b = int(mapped(z0)), int(mapped(z1))
            for z in range(min(a, b), max(a, b) + 1):
                result.add((x, y, z))
    output = np.asarray(sorted(result), np.int32)
    assert np.array_equal(np.unique(output[:, :2], axis=0),
                          np.unique(source[:, :2], axis=0))
    return output, {
        "before_tokens": int(len(source)), "after_tokens": int(len(output)),
        "old_z_min": old_min, "old_z_max": old_max,
        "z_min": new_min, "z_max": new_max,
        "requested_height_voxels": int(requested_height),
        "stretch_created_gap_tokens_added": int(len(result) - before_fill),
        "xy_added": 0, "xy_removed": 0, "footprint_mask_used": False,
        "preexisting_z_gaps_closed": False,
    }


def predicted_height_coordinates(coords, footprint, height_m: float):
    """Preserve the historical scalar-height conversion and unchanged XY support.

    Similarity alignment estimates the scalar's model-frame units. Its footprint
    is never applied as an occupancy mask; only the requested Z span is changed.
    """
    source = np.asarray(coords, dtype=np.int32)
    if source.ndim != 2 or source.shape[1] != 3 or not len(source):
        raise ValueError("Expected nonempty N x 3 structure coordinates")
    if source.min() < 0 or source.max() >= 32:
        raise ValueError("Structure coordinates must be on the 32-cubed grid")
    geom = make_valid(shape(footprint) if isinstance(footprint, dict) else footprint)
    geom = clean(geom)
    if geom.is_empty or geom.area <= 0 or not math.isfinite(height_m) or height_m <= 0:
        raise ValueError("Prediction footprint and height must be valid and positive")
    normalized, center, metric_to_normalized = normalize_geometry(geom)
    ratio = float(height_m) / math.sqrt(float(geom.area))
    bundle = align_control_to_baseline({"geometry": geom, "normalized": normalized,
        "ratio": ratio, "height_m": float(height_m), "center": center,
        "metric_to_normalized": metric_to_normalized}, source)
    corrected, audit = stretch_height(source, bundle["height_voxels"])
    audit.update(height_m=float(height_m), footprint_area_m2=float(geom.area),
                 height_to_sqrt_area=ratio,
                 aligned_control_area=float(bundle["normalized_model_frame"].area),
                 model_frame_alignment=bundle["model_frame_alignment"],
                 mechanism="image-only structure; prediction-derived Z remap; stretch-created gaps only")
    return corrected, audit


def configure_offline_runtime(trellis_root: Path) -> None:
    for key, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "HF_DATASETS_OFFLINE": "1", "WANDB_MODE": "offline",
                       "ATTN_BACKEND": "sdpa", "SPARSE_ATTN_BACKEND": "flash_attn",
                       "OPENCV_IO_ENABLE_OPENEXR": "1"}.items():
        os.environ[key] = value
    root = Path(trellis_root).resolve()
    if not (root / "trellis2").is_dir():
        raise FileNotFoundError(root / "trellis2")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def pipeline_specification(config: dict) -> dict:
    """Resolve distributed relative resource paths without rewriting the recipe.

    The unmodified portable JSON supplies the inference identity, so relocating
    the example does not change a request merely because a disk path changed.
    Existing absolute-path runtime files remain supported.
    """
    import copy
    pipeline = copy.deepcopy(json.loads(Path(config['pipeline_config']).read_text())['args'])
    root = Path(config['model_package_root']).resolve()
    for key, value in pipeline['models'].items():
        pipeline['models'][key] = str((root/value).resolve()) if not Path(value).is_absolute() else value
    for key in ['image_cond_model', 'rembg_model']:
        value = pipeline[key]['args']['model_name']
        if not Path(value).is_absolute():pipeline[key]['args']['model_name'] = str((root/value).resolve())
    return pipeline


def preflight_resources(config: dict) -> dict:
    """Check local identities and files without allocating a GPU or loading weights."""
    if config.get("seed", 42) != 42:
        raise ValueError("The standard inference recipe fixes seed 42")
    pipeline_path = Path(config["pipeline_config"])
    pipeline = pipeline_specification(config)
    package_root = Path(config["model_package_root"])
    selected = json.loads(Path(config["model_manifest"]).read_text())
    checked = []
    for record in selected["models"]:
        for item in [record, record["architecture"]]:
            path = package_root / item["path"]
            if not path.is_file() or path.stat().st_size != item["bytes"]:
                raise ValueError(f"Selected model resource missing or size changed: {path}")
            if path.suffix == ".json" and sha256_file(path) != item["sha256"]:
                raise ValueError(f"Architecture hash mismatch: {path}")
            checked.append({"path": str(path), "bytes": path.stat().st_size})
    if [(m["role"], m["step"]) for m in selected["models"]] != [("structure", 98440), ("shape", 41730), ("material", 44940)]:
        raise ValueError("The standard requires the selected three checkpoints")
    for key in ("sparse_structure_decoder", "shape_slat_decoder", "tex_slat_decoder", "shape_slat_flow_model_512"):
        for suffix in (".json", ".safetensors"):
            path = Path(pipeline["models"][key] + suffix)
            if not path.is_file():
                raise FileNotFoundError(path)
            checked.append({"path": str(path), "bytes": path.stat().st_size})
    for key in ("image_cond_model", "rembg_model"):
        directory = Path(pipeline[key]["args"]["model_name"])
        for name in ("config.json", "model.safetensors"):
            path = directory / name
            if not path.is_file():
                raise FileNotFoundError(path)
            checked.append({"path": str(path), "bytes": path.stat().st_size})
    return {"status": "local_resources_present", "files": checked,
            "gpu_load_verified": False, "weights_rehashed": False,
            "pipeline_config_sha256": sha256_file(pipeline_path),
            "model_manifest_sha256": sha256_file(Path(config["model_manifest"]))}


def _seed(torch, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _condition(torch, pipeline, image, resolution):
    encoded = pipeline.get_cond([image], resolution, include_neg_cond=False)["cond"]
    if encoded.ndim != 3 or encoded.shape[0] != 1:
        raise ValueError("Expected one completed image's DINO tokens")
    fused = encoded.reshape(1, -1, encoded.shape[-1]).contiguous()
    return {"cond": fused, "neg_cond": torch.zeros_like(fused[:, :1])}


def verified_generation_cache(output: Path, identity: str) -> dict | None:
    """Accept a decoded cache only when its identity and bytes still match.

    A damaged completed cache needs a fresh output directory. It is never
    silently replaced, so provenance from an interrupted or changed run stays
    available for inspection.
    """
    output = Path(output)
    manifest_path, raw_path = output / "raw_manifest.json", output / "raw_mesh.pt"
    if not manifest_path.is_file():
        return None
    previous = json.loads(manifest_path.read_text())
    if previous.get("identity_sha256") != identity:
        raise ValueError("Raw generation cache has a different identity; use a new output directory")
    if (previous.get("status") != "complete" or not raw_path.is_file()
            or raw_path.stat().st_size != previous.get("bytes")
            or sha256_file(raw_path) != previous.get("sha256")):
        raise ValueError("Completed raw generation cache is missing or changed; use a new output directory")
    return previous


class GenerationRuntime:
    """Load final selected denoisers directly, retaining the standard computation."""

    def __init__(self, config: dict):
        self.config = config
        self.preflight = preflight_resources(config)
        configure_offline_runtime(Path(config["trellis_root"]))
        import torch
        from trellis2 import models
        from trellis2.pipelines import Trellis2ImageTo3DPipeline
        from trellis2.pipelines import samplers, rembg
        from trellis2.modules import image_feature_extractor

        if not torch.cuda.is_available():
            raise RuntimeError("GenerationRuntime requires an allocated CUDA worker")
        self.torch = torch
        original = pipeline_specification(config)
        loaded = {}
        for key in ("sparse_structure_decoder", "shape_slat_decoder", "tex_slat_decoder", "shape_slat_flow_model_512"):
            loaded[key] = models.from_pretrained(original["models"][key]).eval()
        selected = json.loads(Path(config["model_manifest"]).read_text())
        names = {"structure": "sparse_structure_flow_model", "shape": "shape_slat_flow_model_1024",
                 "material": "tex_slat_flow_model_1024"}
        for item in selected["models"]:
            root = Path(config["model_package_root"])
            architecture = json.loads((root / item["architecture"]["path"]).read_text())
            model = getattr(models, architecture["name"])(**architecture["args"])
            state = torch.load(root / item["path"], map_location="cpu", weights_only=True)
            model.load_state_dict(state, strict=True)
            loaded[names[item["role"]]] = model.eval()
            del state
            gc.collect()
        kwargs = {"models": loaded, "low_vram": True, "default_pipeline_type": "1024_cascade"}
        for key in ("sparse_structure_sampler", "shape_slat_sampler", "tex_slat_sampler"):
            settings = original[key]
            kwargs[key] = getattr(samplers, settings["name"])(**settings["args"])
            kwargs[key + "_params"] = settings["params"]
            if settings["params"]["steps"] != 12:
                raise ValueError("Standard sampling uses 12 steps")
        for key in ("shape_slat_normalization", "tex_slat_normalization"):
            kwargs[key] = original[key]
        for key, module in (("image_cond_model", image_feature_extractor), ("rembg_model", rembg)):
            specification = original[key]
            kwargs[key] = getattr(module, specification["name"])(**specification["args"])
        self.pipeline = Trellis2ImageTo3DPipeline(**kwargs)
        self.pipeline.cuda()

    def generate(self, building_path: Path, appearance_directory: Path, output: Path) -> dict:
        """Write one decoded raw mesh reusable for both requested LOD exports."""
        from PIL import Image
        from .appearance import completion_contract

        torch, pipeline = self.torch, self.pipeline
        explicit_height = image_height_span(self.config)
        building_path, appearance_directory, output = map(Path, (building_path, appearance_directory, output))
        building = json.loads(building_path.read_text())
        source = appearance_directory / "generated_main_oblique.png"
        prepared = json.loads((appearance_directory / "appearance_input.json").read_text())
        receipt = json.loads((appearance_directory / "completion.json").read_text())
        if (prepared["building_id"] != building["building_id"] or
                prepared["provenance"]["building_sha256"] != sha256_file(building_path) or
                receipt.get("status") != "complete" or
                receipt.get("request_identity_sha256") != completion_contract(prepared) or
                receipt.get("output_sha256") != sha256_file(source)):
            raise ValueError("Completed image is not bound to the current building inputs")
        contract = {"image_sha256": receipt["output_sha256"],
                    "building_sha256": sha256_file(building_path),
                    "appearance_input_identity": prepared["input_identity_sha256"],
                    "model_manifest_sha256": self.preflight["model_manifest_sha256"],
                    "pipeline_config_sha256": self.preflight["pipeline_config_sha256"],
                    "generation_source_sha256": sha256_file(Path(__file__)), "seed": 42}
        if self.config.get("structure_conditioning") == "source_footprint":
            from . import structure_conditioning
            contract["structure_conditioning"] = {"mode": "source_footprint",
                "source_sha256": sha256_file(Path(structure_conditioning.__file__)),
                "ground_truth_used": False}
        elif explicit_height is not None:
            contract["structure_conditioning"] = {"mode": "image_height_span",
                "structure_height_voxels": explicit_height,
                "source": "explicit historical scalar", "reference_geometry_used": False}
        identity = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "raw_manifest.json"
        raw_path = output / "raw_mesh.pt"
        previous = verified_generation_cache(output, identity)
        if previous is not None:
            return previous
        start = time.monotonic()
        with Image.open(source) as opened:
            image = opened.convert("RGB")
        if image.size != (1024, 1024):
            raise ValueError("Completed image must be 1024 square")
        image = pipeline.preprocess_image(image)
        image.save(output / "preprocessed.png")
        torch.cuda.empty_cache()
        with torch.no_grad():
            # Stage1 and Stage2/3 were independent historical jobs. Reset RNG
            # between them, preserving the continuation branch's random stream.
            _seed(torch, 42)
            cond512 = _condition(torch, pipeline, image, 512)
            original_coords = pipeline.sample_sparse_structure(cond512, 32, num_samples=1,
                                                                 sampler_params={}).cpu().numpy()[:, 1:]
            np.savez_compressed(output / "stage1_image_only.npz", coords=original_coords)
            if self.config.get("structure_conditioning") == "source_footprint":
                from .structure_conditioning import conditioned_structure_coordinates
                corrected, height_audit = conditioned_structure_coordinates(original_coords, building)
            elif explicit_height is not None:
                corrected, height_audit = stretch_height(original_coords, explicit_height)
                height_audit.update(structure_conditioning="image_height_span",
                    structure_height_voxels=explicit_height,
                    height_source="explicit historical scalar", reference_geometry_used=False,
                    mechanism="image-only structure; explicit historical scalar Z remap; stretch-created gaps only")
            else:
                corrected, height_audit = predicted_height_coordinates(original_coords,
                    building["footprint_world"], float(building["height_m"]))
            np.savez_compressed(output / "stage1_height_corrected.npz", coords=corrected)
            write_json(output / "height_correction.json", height_audit)
            coords = torch.from_numpy(np.column_stack([np.zeros(len(corrected), dtype=np.int32), corrected]))
            coords = coords.to(device=pipeline.device, dtype=torch.int32).contiguous()
            _seed(torch, 42)
            cond512 = _condition(torch, pipeline, image, 512)
            cond1024 = _condition(torch, pipeline, image, 1024)
            shape_latent, resolution = pipeline.sample_shape_slat_cascade(cond512, cond1024,
                pipeline.models["shape_slat_flow_model_512"], pipeline.models["shape_slat_flow_model_1024"],
                512, 1024, coords, sampler_params={}, max_num_tokens=49152)
            texture_latent = pipeline.sample_tex_slat(cond1024, pipeline.models["tex_slat_flow_model_1024"],
                                                      shape_latent, sampler_params={})
            torch.cuda.empty_cache()
            mesh = pipeline.decode_latent(shape_latent, texture_latent, int(resolution))[0]
            if len(mesh.faces) <= 8:
                raise ValueError("Degenerate decoded mesh")
            payload = {key: getattr(mesh, key).detach().cpu().contiguous()
                       for key in ("vertices", "faces", "attrs", "coords")}
            if not torch.isfinite(payload["vertices"]).all() or not torch.isfinite(payload["attrs"]).all():
                raise ValueError("Nonfinite decoded mesh")
            payload["resolution"] = int(resolution)
            payload["attr_layout"] = {key: (value.start, value.stop, value.step)
                                      for key, value in pipeline.pbr_attr_layout.items()}
            temporary = output / "raw_mesh.partial.pt"
            torch.save(payload, temporary)
            temporary.replace(raw_path)
        record = {"schema": "citycode.generated_raw_mesh.v1", "status": "complete",
                  "building_id": building["building_id"], "identity_sha256": identity,
                  "contract": contract, "input_sha256": receipt["output_sha256"],
                  "sha256": sha256_file(raw_path), "bytes": raw_path.stat().st_size,
                  "vertices": len(payload["vertices"]), "faces": len(payload["faces"]),
                  "resolution": int(resolution), "cache_kind": "decoded_mesh_before_uv",
                  "supported_lods": ["0.25M", "0.1M"], "inference_seconds": time.monotonic() - start}
        record["placement_inputs"] = {name: {"path": filename, "sha256": sha256_file(output / filename)}
            for name, filename in (("height_correction", "height_correction.json"),
                                   ("image_only_structure", "stage1_image_only.npz"),
                                   ("height_corrected_structure", "stage1_height_corrected.npz"))}
        record["native_coordinate_system"] = "TRELLIS XYZ normalized model frame Z-up"
        record["glb_export_axis_transform"] = "glTF(x,y,z)=(native_x,native_z,-native_y)"
        write_json(manifest_path, record)
        del payload, mesh, shape_latent, texture_latent, coords, cond512, cond1024
        gc.collect()
        torch.cuda.empty_cache()
        return record
