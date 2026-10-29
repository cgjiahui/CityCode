#!/usr/bin/env python3
"""Deterministically select recovered structures using prediction-only evidence."""

from __future__ import annotations

import argparse, csv, hashlib, json, math
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import trimesh
from PIL import Image
from shapely import make_valid
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, mapping, shape
from shapely.ops import unary_union
from shapely.strtree import STRtree

from city_pipeline.buildings.observations import (
    UE_Z_UP_TO_GLTF_Y_UP, deterministic_color, extrude_geometry,
    footprint_shape_metrics,
)
from ._config import PROJECT_ROOT, RUN_ROOT, SOURCE_RUN_ROOT

PIPELINE_NAME = "building_selection"
FORBIDDEN_FIELD_FRAGMENTS = ("gt_", "ground_truth", "building_id_gt", "iou_gt", "purity_gt", "coverage_gt")

@dataclass(frozen=True)
class Parameters:
    component_relative_threshold: float = 0.04
    component_minimum_area_m2: float = 5.0
    simplify_m: float = 0.20
    min_area_m2: float = 24.0
    max_elongation: float = 7.0
    min_compactness: float = 0.10
    min_quality_score: float = 0.30
    severe_intersection_m2: float = 20.0
    severe_containment: float = 0.50
    duplicate_iou: float = 0.55
    supermask_min_children: int = 2
    supermask_union_coverage: float = 0.62
    supermask_child_area_ratio: float = 0.82
    supermask_quality_margin: float = -0.10
    supermask_penalty: float = 2.4
    unstable_shape_penalty: float = 0.55

@dataclass
class Candidate:
    source_index: int
    properties: dict[str, Any]
    observation_properties: dict[str, Any]
    original_geometry: Polygon | MultiPolygon
    geometry: Polygon | MultiPolygon
    score: float
    metrics: dict[str, float]
    decision: str = "eligible"
    explained_by: tuple[int, ...] = ()

    @property
    def source_id(self) -> str:
        return str(self.properties["predicted_global_building_id"])

    @property
    def observation_id(self) -> str:
        return str(self.properties["canonical_observation_id"])

def now(): return datetime.now().astimezone().isoformat(timespec="seconds")
def relative(p: Path) -> str:
    try: return p.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError: return p.resolve().as_posix()
def read_json(p: Path): return json.loads(p.read_text(encoding="utf-8"))
def write_json(p: Path, value: Any):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")
def write_jsonl(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for row in rows: f.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
def write_csv(p: Path, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not rows: p.write_text("", encoding="utf-8"); return
    with p.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def parts(g):
    if g.is_empty: return []
    if isinstance(g, Polygon): return [g]
    if isinstance(g, MultiPolygon): return list(g.geoms)
    if isinstance(g, GeometryCollection): return [p for c in g.geoms for p in parts(c)]
    return []
def polygonal(g):
    ps=parts(make_valid(g))
    if not ps: raise ValueError("empty polygon")
    return unary_union(ps)
def clean_geometry(g, p: Parameters):
    ps=parts(g); total=sum(x.area for x in ps); largest=max(ps,key=lambda x:x.area)
    kept=[x for x in ps if x is largest or (x.area >= p.component_minimum_area_m2 and x.area/max(total,1e-9)>p.component_relative_threshold)]
    out=polygonal(unary_union(kept))
    return polygonal(out.simplify(p.simplify_m,preserve_topology=True)) if p.simplify_m else out

def shape_metrics(g):
    m=footprint_shape_metrics(g)
    return {k:float(v) for k,v in m.items() if isinstance(v,(int,float))}

def candidate_score(prop, obs, m, p):
    canonical=min(max(float(prop.get("canonical_score",0))/120,0),1.5)
    preliminary=min(max(float(obs.get("preliminary_quality",0))/120,0),1.2)
    support=min(math.log1p(int(prop.get("member_count",1)))/math.log(6),1.0)
    core=1.0 if obs.get("reliable_core_contains_centroid") else 0.0
    boundary={"interior_keep":1.0,"boundary_risk_review":0.35,"boundary_clipped_exclude":0.0}.get(str(obs.get("boundary_status")),0.2)
    stable=min(int(obs.get("stable_layer_count",0))/3,1.0)
    ambiguity=min(int(obs.get("ambiguity_count",0))/4,1.0)
    score=1.20*canonical+0.55*preliminary+0.40*support+0.25*core+0.20*boundary+0.18*stable-0.22*ambiguity
    if m.get("elongation",1)>4.5: score-=p.unstable_shape_penalty*min((m["elongation"]-4.5)/3,1.5)
    if m.get("compactness",1)<0.2: score-=p.unstable_shape_penalty*min((0.2-m["compactness"])/0.15,1.5)
    if m.get("component_count_ge_5m2",1)>1: score-=0.12*(m["component_count_ge_5m2"]-1)
    return float(score)

def load_candidates(source_root: Path, p: Parameters):
    fused=source_root/"global_fused"
    globals_doc=read_json(fused/"global_building_footprints.geojson")
    obs_doc=read_json(fused/"building_observation_footprints.geojson")
    obs_by_id={str(f["properties"]["observation_id"]):f["properties"] for f in obs_doc["features"]}
    result=[]; removed_count=0; removed_area=0.0
    for i,f in enumerate(globals_doc["features"]):
        prop=dict(f["properties"]); oid=str(prop["canonical_observation_id"])
        if oid not in obs_by_id: raise KeyError(oid)
        original=polygonal(shape(f["geometry"])); cleaned=clean_geometry(original,p)
        removed_count+=max(len(parts(original))-len(parts(cleaned)),0); removed_area+=max(original.area-cleaned.area,0)
        m=shape_metrics(cleaned); score=candidate_score(prop,obs_by_id[oid],m,p)
        result.append(Candidate(i,prop,dict(obs_by_id[oid]),original,cleaned,score,m))
    return result,{"source_global_candidates":len(result),"source_observations":len(obs_doc["features"]),"removed_component_count":removed_count,"removed_component_area_m2":removed_area}

def relation_graph(candidates, p):
    gs=[c.geometry for c in candidates]; tree=STRtree(gs); edges=set(); raw=[]
    for i,g in enumerate(gs):
        for rj in tree.query(g,predicate="intersects"):
            j=int(rj)
            if j<=i: continue
            inter=float(g.intersection(gs[j]).area)
            if inter<1: continue
            contain=inter/max(min(g.area,gs[j].area),1e-9); iou=inter/max(g.area+gs[j].area-inter,1e-9)
            raw.append((i,j,inter,contain,iou))
            if (inter>=p.severe_intersection_m2 and contain>=p.severe_containment) or iou>=p.duplicate_iou: edges.add((i,j))
    return edges,raw

def detect_supermasks(candidates, raw, p):
    neighbors=defaultdict(list)
    severe={(min(a,b),max(a,b)) for a,b,_,contain,iou in raw if contain>=p.severe_containment or iou>=p.duplicate_iou}
    for a,b,inter,contain,iou in raw:
        ga,gb=candidates[a].geometry,candidates[b].geometry
        parent,child=(a,b) if ga.area>=gb.area else (b,a)
        if inter/max(candidates[child].geometry.area,1e-9)>=p.severe_containment and candidates[child].geometry.area/candidates[parent].geometry.area<=p.supermask_child_area_ratio:
            neighbors[parent].append(child)
    relations=[]
    for parent,children in sorted(neighbors.items()):
        ordered=sorted(set(children),key=lambda i:(-candidates[i].score,candidates[i].source_id))
        chosen=[]
        for child in ordered:
            if all((min(child,x),max(child,x)) not in severe for x in chosen): chosen.append(child)
        if len(chosen)<p.supermask_min_children: continue
        union=unary_union([candidates[i].geometry for i in chosen])
        coverage=float(union.intersection(candidates[parent].geometry).area/max(candidates[parent].geometry.area,1e-9))
        quality=float(np.mean([candidates[i].score for i in chosen]))
        if coverage>=p.supermask_union_coverage and quality>=candidates[parent].score+p.supermask_quality_margin:
            candidates[parent].score-=p.supermask_penalty
            candidates[parent].decision="supermask_dominated"
            candidates[parent].explained_by=tuple(chosen)
            relations.append({"parent":parent,"children":chosen,"union_coverage":coverage,"parent_score_after_penalty":candidates[parent].score,"mean_child_score":quality})
    return relations

def eligible(c, p):
    m=c.metrics; obs=c.observation_properties
    if c.geometry.area<p.min_area_m2: return False,"area_gate"
    if m.get("elongation",1)>p.max_elongation: return False,"elongation_gate"
    if m.get("compactness",1)<p.min_compactness: return False,"compactness_gate"
    if c.score<p.min_quality_score: return False,"quality_gate"
    if int(obs.get("stable_layer_count",0))<=0 and not obs.get("reliable_core_contains_centroid") and str(obs.get("boundary_status"))!="interior_keep": return False,"unsupported_boundary_gate"
    return True,"eligible"

def solve_component(vertices, adjacency, candidates):
    # Exact deterministic branch-and-bound for ordinary components; greedy fallback for large ones.
    vertices=tuple(sorted(vertices,key=lambda i:(-candidates[i].score,candidates[i].source_id)))
    if len(vertices)>32:
        selected=[]
        for v in vertices:
            if all(v not in adjacency[x] for x in selected): selected.append(v)
        return set(selected)
    positive={v:max(candidates[v].score,0.0) for v in vertices}; best_score=-1e18; best=()
    def rec(remaining, chosen, score):
        nonlocal best_score,best
        if score+sum(positive[v] for v in remaining)<best_score-1e-12: return
        if not remaining:
            key=tuple(sorted(chosen,key=lambda i:candidates[i].source_id))
            if score>best_score+1e-12 or (abs(score-best_score)<=1e-12 and key<best): best_score=score; best=key
            return
        v=max(remaining,key=lambda x:(len(adjacency[x]&remaining),candidates[x].score,candidates[x].source_id))
        rec(remaining-{v}-adjacency[v],chosen+(v,),score+candidates[v].score)
        rec(remaining-{v},chosen,score)
    rec(set(vertices),(),0.0); return set(best)

def select(candidates,p):
    edges,raw=relation_graph(candidates,p); supermasks=detect_supermasks(candidates,raw,p)
    pool=set(); gates=[]
    for i,c in enumerate(candidates):
        ok,reason=eligible(c,p); c.decision=reason if c.decision=="eligible" else c.decision
        if ok: pool.add(i)
        else: gates.append({"source_id":c.source_id,"reason":reason,"score":c.score})
    adj={i:set() for i in pool}
    for a,b in edges:
        if a in pool and b in pool: adj[a].add(b);adj[b].add(a)
    unseen=set(pool); selected=set()
    while unseen:
        root=min(unseen); comp={root}; q=[root]; unseen.remove(root)
        while q:
            v=q.pop()
            for n in adj[v]&unseen: unseen.remove(n);comp.add(n);q.append(n)
        selected|=solve_component(comp,adj,candidates)
    for i in pool-selected:
        if candidates[i].decision=="eligible": candidates[i].decision="conflict_suppressed"
    residual=[x for x in raw if x[0] in selected and x[1] in selected and x[2]>=p.severe_intersection_m2 and x[3]>=p.severe_containment]
    return sorted(selected),edges,raw,supermasks,gates,residual

def signature(oid): return hashlib.sha256(oid.encode()).hexdigest()[:16]
def build_records(indices,candidates):
    rows=[]
    for i in indices:
        c=candidates[i]; cen=c.geometry.centroid
        rows.append({"candidate":c,"signature":signature(c.observation_id),"sort":(round(cen.x,3),round(cen.y,3),signature(c.observation_id))})
    rows.sort(key=lambda x:x["sort"])
    for n,r in enumerate(rows,1): r["index"]=n;r["id"]=f"PGB_SC_{n:06d}"
    return rows

def prediction_rows(records):
    gr=[];orr=[];gf=[];of=[]
    for r in records:
        c=r["candidate"];p=c.properties;o=c.observation_properties;g=c.geometry;cen=g.centroid
        group="joint_supermask" if len(c.explained_by)>=2 and c.decision!="supermask_dominated" else "fine"
        grow={"predicted_global_building_id":r["id"],"global_building_index":r["index"],"membership_signature":r["signature"],"canonical_observation_id":c.observation_id,"member_observation_ids_json":json.dumps([c.observation_id]),"member_count":1,"patch_count":1,"sample_count":1,"canonical_patch_id":o["patch_id"],"canonical_view_id":o["view_id"],"canonical_instance_id":int(o["rgb_instance_id"]),"canonical_boundary_status":o["boundary_status"],"canonical_centroid_in_reliable_core":bool(o["reliable_core_contains_centroid"]),"canonical_reliable_core_edge_distance_m":float(o["reliable_core_edge_distance_m"]),"canonical_score":float(c.score*100),"footprint_area_m2":float(g.area),"centroid_x_m":float(cen.x),"centroid_y_m":float(cen.y),"base_z_m":float(o["base_z_m"]),"top_z_m":float(o["top_z_m"]),"height_m":float(o["height_m"]),"height_method":o["height_method"],"has_matching_conflict":False,"selection_group":group,"source_candidate_id":c.source_id,"canonical_rgb_path":o["rgb_path"],"canonical_source_instance_ids_path":o["source_instance_ids_path"],"canonical_bev_instance_ids_path":o["bev_instance_ids_path"],"canonical_building_point_cloud_npz_path":o["building_point_cloud_npz_path"],"canonical_glb":"","canonical_glb_ue_z_up":""}
        orow={"observation_id":c.observation_id,"predicted_global_building_id":r["id"],"global_building_index":r["index"],"sample_id":o["sample_id"],"patch_id":o["patch_id"],"view_id":o["view_id"],"rgb_instance_id":int(o["rgb_instance_id"]),"bev_instance_id":int(o["bev_instance_id"]),"boundary_status":o["boundary_status"],"raster_area_m2":float(o["raster_area_m2"]),"repaired_footprint_area_m2":float(g.area),"centroid_x_m":float(cen.x),"centroid_y_m":float(cen.y),"base_z_m":float(o["base_z_m"]),"top_z_m":float(o["top_z_m"]),"height_m":float(o["height_m"]),"height_method":o["height_method"],"stable_layer_count":int(o["stable_layer_count"]),"preliminary_quality":float(o["preliminary_quality"]),"canonical_score":float(c.score*100),"ambiguity_count":int(o["ambiguity_count"]),"source_pixel_count":int(o["source_pixel_count"]),"source_edge_distance_px":int(o["source_edge_distance_px"]),"reliable_core_contains_centroid":bool(o["reliable_core_contains_centroid"]),"reliable_core_edge_distance_m":float(o["reliable_core_edge_distance_m"]),"source_mean_rgb_json":json.dumps(o.get("source_mean_rgb",[])),"source_std_rgb_json":json.dumps(o.get("source_std_rgb",[])),"rgb_path":o["rgb_path"],"source_instance_ids_path":o["source_instance_ids_path"],"bev_instance_ids_path":o["bev_instance_ids_path"],"building_point_cloud_npz_path":o["building_point_cloud_npz_path"],"selection_group":group,"source_candidate_id":c.source_id}
        gp=dict(grow);gp["member_observation_ids"]=json.loads(gp.pop("member_observation_ids_json"));op=dict(orow);op["source_mean_rgb"]=json.loads(op.pop("source_mean_rgb_json"));op["source_std_rgb"]=json.loads(op.pop("source_std_rgb_json"));op["boundary_evidence"]=o.get("boundary_evidence",[])
        gr.append(grow);orr.append(orow);gf.append({"type":"Feature","properties":gp,"geometry":mapping(g)});of.append({"type":"Feature","properties":op,"geometry":mapping(g)})
    return gr,orr,{"type":"FeatureCollection","features":gf},{"type":"FeatureCollection","features":of}

def export_glb(root,records,global_rows):
    vr=root/"canonical_building_glb";ur=root/"canonical_building_glb_ue_z_up";vr.mkdir(parents=True,exist_ok=True);ur.mkdir(parents=True,exist_ok=True)
    sv=trimesh.Scene();su=trimesh.Scene();verts=faces=0
    for r,row in zip(records,global_rows,strict=True):
        gid=r["id"];c=r["candidate"];meta={"predicted_global_building_id":gid,"canonical_observation_id":c.observation_id,"global_building_index":r["index"]}
        mu=extrude_geometry(c.geometry,float(row["base_z_m"]),float(row["top_z_m"]),deterministic_color(gid),meta);mv=mu.copy();mv.apply_transform(UE_Z_UP_TO_GLTF_Y_UP)
        su.add_geometry(mu,node_name=gid,geom_name=gid);sv.add_geometry(mv,node_name=gid,geom_name=gid)
        a=trimesh.Scene();a.add_geometry(mu,node_name=gid,geom_name=gid);up=ur/f"{gid}.glb";up.write_bytes(a.export(file_type="glb"))
        a=trimesh.Scene();a.add_geometry(mv,node_name=gid,geom_name=gid);vp=vr/f"{gid}.glb";vp.write_bytes(a.export(file_type="glb"))
        row["canonical_glb"]=relative(vp);row["canonical_glb_ue_z_up"]=relative(up);verts+=len(mu.vertices);faces+=len(mu.faces)
    vp=root/"full_city_coarse.glb";up=root/"full_city_coarse_ue_z_up.glb";vp.write_bytes(sv.export(file_type="glb"));up.write_bytes(su.export(file_type="glb"))
    return {"full_city_coarse_glb":relative(vp),"full_city_coarse_glb_ue_z_up":relative(up),"canonical_building_glb_root":relative(vr),"canonical_building_glb_ue_z_up_root":relative(ur),"mesh_count":len(records),"vertex_count":verts,"face_count":faces}

def identity_rasters(root,records):
    by=defaultdict(list)
    for r in records: by[r["candidate"].observation_properties["sample_id"]].append(r)
    # Preserve all 79 input sample views. A view with no selected canonical
    # observation gets an explicit all-zero identity raster and empty mapping.
    source_manifest=read_json(SOURCE_RUN_ROOT/"global_fused/manifest.json")
    source_records={x["sample_id"]:x for x in source_manifest["identity_chain"]["sample_records"]}
    out=[]
    for sid in sorted(source_records):
        items=by.get(sid,[])
        if items:o=items[0]["candidate"].observation_properties
        else:
            source_map=read_json(PROJECT_ROOT/source_records[sid]["mapping"])
            o={"patch_id":source_map["patch_id"],"view_id":source_map["view_id"],"rgb_path":source_map["rgb"],"source_instance_ids_path":source_map["source_instance_ids"],"bev_instance_ids_path":source_map["bev_instance_ids"],"building_point_cloud_npz_path":source_map["building_point_cloud_npz"]}
        src=PROJECT_ROOT/o["source_instance_ids_path"];bev=PROJECT_ROOT/o["bev_instance_ids_path"]
        s=np.asarray(Image.open(src),dtype=np.uint16);b=np.asarray(Image.open(bev),dtype=np.uint16);lut=np.zeros(max(int(s.max()),int(b.max()))+1,dtype=np.uint16);maps=[]
        for r in items:
            oo=r["candidate"].observation_properties;i=int(oo["rgb_instance_id"]);lut[i]=r["index"];maps.append({"observation_id":r["candidate"].observation_id,"instance_id":i,"global_building_index":r["index"],"predicted_global_building_id":r["id"]})
        sp=root/"source_rgb_global_building_index"/f"{sid}__source_global_building_index.png";bp=root/"bev_global_building_index"/f"{sid}__bev_global_building_index.png";sp.parent.mkdir(parents=True,exist_ok=True);bp.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(lut[s]).save(sp,compress_level=1);Image.fromarray(lut[b]).save(bp,compress_level=1)
        mp=root/"per_sample"/f"{sid}__observation_to_global.json";write_json(mp,{"schema":"citycode.observation_identity_map.v1","sample_id":sid,"patch_id":o["patch_id"],"view_id":o["view_id"],"rgb":o["rgb_path"],"source_instance_ids":o["source_instance_ids_path"],"source_global_building_index":relative(sp),"bev_instance_ids":o["bev_instance_ids_path"],"bev_global_building_index":relative(bp),"building_point_cloud_npz":o["building_point_cloud_npz_path"],"instance_to_global":sorted(maps,key=lambda x:x["instance_id"])})
        out.append({"sample_id":sid,"patch_id":o["patch_id"],"view_id":o["view_id"],"observation_count":len(items),"mapping":relative(mp),"source_global_building_index":relative(sp),"bev_global_building_index":relative(bp)})
    return out

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--source-run-root",type=Path,default=SOURCE_RUN_ROOT);ap.add_argument("--output-root",type=Path,default=RUN_ROOT/"global_fused");ap.add_argument("--config",type=Path);ap.add_argument("--skip-glb",action="store_true");ap.add_argument("--skip-rasters",action="store_true");a=ap.parse_args()
    params=Parameters(**(read_json(a.config) if a.config else {}));root=a.output_root.resolve();root.mkdir(parents=True,exist_ok=True)
    candidates,context=load_candidates(a.source_run_root.resolve(),params);selected,edges,raw,supermasks,gates,residual=select(candidates,params);records=build_records(selected,candidates);gr,orr,gf,of=prediction_rows(records)
    glb={};samples=[]
    if not a.skip_glb: glb=export_glb(root,records,gr)
    if not a.skip_rasters: samples=identity_rasters(root,records)
    for row,feat in zip(gr,gf["features"],strict=True): feat["properties"]["canonical_glb"]=row["canonical_glb"];feat["properties"]["canonical_glb_ue_z_up"]=row["canonical_glb_ue_z_up"]
    write_csv(root/"global_buildings.csv",gr);write_jsonl(root/"global_buildings.jsonl",gr);pd.DataFrame(gr).to_parquet(root/"global_buildings.parquet",index=False);write_json(root/"global_building_footprints.geojson",gf)
    write_csv(root/"observation_to_global.csv",orr);write_jsonl(root/"building_observations.jsonl",orr);pd.DataFrame(orr).to_parquet(root/"observation_to_global.parquet",index=False);write_json(root/"building_observation_footprints.geojson",of)
    decisions=[{"source_candidate_id":c.source_id,"canonical_observation_id":c.observation_id,"decision":("selected" if i in selected else c.decision),"prediction_score":c.score,"explained_by_source_ids_json":json.dumps([candidates[x].source_id for x in c.explained_by]),**c.metrics} for i,c in enumerate(candidates)]
    write_csv(root/"selection_decisions.csv",decisions);write_jsonl(root/"selection_decisions.jsonl",decisions);write_csv(root/"candidate_matches.csv",[]);write_json(root/"matching_conflicts.json",[]);write_json(root/"duplicate_groups.json",[]);write_json(root/"global_building_index.json",{str(r["index"]):r["id"] for r in records})
    pubs=[{"predicted_global_building_id":r["id"],"global_building_index":r["index"],"membership_signature":r["signature"],"cluster_centroid_x_m":r["candidate"].geometry.centroid.x,"cluster_centroid_y_m":r["candidate"].geometry.centroid.y,"canonical_observation_id":r["candidate"].observation_id,"member_observation_ids_json":json.dumps([r["candidate"].observation_id]),"member_count":1,"all_members_outside_reliable_core":not bool(r["candidate"].observation_properties["reliable_core_contains_centroid"]),"all_members_boundary_risk_or_exclude":r["candidate"].observation_properties["boundary_status"]!="interior_keep","all_members_without_stable_height_layer":int(r["candidate"].observation_properties["stable_layer_count"])<=0,"low_confidence_evidence_gate":False,**r["candidate"].metrics,"aggressive_shape_gate":False,"auto_removed_from_published_outputs":False,"auto_remove_reason":"","broad_quarantine_candidate":False} for r in records]
    write_csv(root/"global_building_publication_decisions.csv",pubs);write_jsonl(root/"global_building_publication_decisions.jsonl",pubs);write_csv(root/"broad_fragment_quarantine_candidates.csv",[]);write_jsonl(root/"broad_fragment_quarantine_candidates.jsonl",[]);write_json(root/"broad_fragment_quarantine_candidates.geojson",{"type":"FeatureCollection","features":[]})
    selected_raw=[x for x in raw if x[0] in selected and x[1] in selected]
    summary={"selected_entity_count":len(records),"fine_entity_count":sum(x["selection_group"]=="fine" for x in gr),"joint_entity_count":sum(x["selection_group"]!="fine" for x in gr),"eligible_count":len(candidates)-len(gates),"severe_graph_edge_count":len(edges),"candidate_raw_intersection_pair_count":len(raw),"raw_intersection_pair_count":len(selected_raw),"raw_intersection_area_m2":float(sum(x[2] for x in selected_raw)),"residual_severe_overlap_count":len(residual),"supermask_dominated_count":len(supermasks),"component_cleanup":context}
    manifest={"schema":"citycode.building_selection.v1","created_at":now(),"status":"complete","pipeline":PIPELINE_NAME,"prediction_contract":{"mode":"deterministic recovered-observation selection","input_whitelist":[relative(a.source_run_root/"global_fused/global_building_footprints.geojson"),relative(a.source_run_root/"global_fused/building_observation_footprints.geojson"),"referenced source/BEV instance rasters for release identity maps only"],"geometry_source":"cleaned immutable recovered canonical observation","height_source":"immutable recovered canonical observation","selection_uses_ground_truth":False,"oracle_decisions_used":False,"id_assignment":"deterministic selected-observation centroid ordering"},"inputs":{"source_run_root":relative(a.source_run_root)},"parameters":asdict(params),"selection_summary":summary,"publication_filter":{"enabled":False,"semantics":"prediction-only gates occur before stable ID assignment"},"totals":{"sample_views":len(samples) if samples else 79,"observations_before_publication_filter":len(orr),"observations":len(orr),"candidate_edges":len(raw),"accepted_candidate_edges":len(edges),"selected_one_to_one_edges":0,"merged_edges":0,"near_consensus_edges":0,"predicted_global_buildings_before_publication_filter":len(gr),"predicted_global_buildings":len(gr),"singleton_global_buildings":len(gr),"duplicate_groups":0,"global_buildings_removed":len(candidates)-len(gr),"observations_removed":len(candidates)-len(gr),"broad_quarantine_candidates":0},"identity_chain":{"sample_records":samples},"outputs":{"observation_to_global_csv":relative(root/"observation_to_global.csv"),"observation_to_global_parquet":relative(root/"observation_to_global.parquet"),"building_observations_jsonl":relative(root/"building_observations.jsonl"),"global_buildings_csv":relative(root/"global_buildings.csv"),"global_buildings_jsonl":relative(root/"global_buildings.jsonl"),"global_buildings_parquet":relative(root/"global_buildings.parquet"),"global_building_footprints_geojson":relative(root/"global_building_footprints.geojson"),"building_observation_footprints_geojson":relative(root/"building_observation_footprints.geojson"),"candidate_matches_csv":relative(root/"candidate_matches.csv"),"duplicate_groups":relative(root/"duplicate_groups.json"),"matching_conflicts":relative(root/"matching_conflicts.json"),"global_building_publication_decisions_csv":relative(root/"global_building_publication_decisions.csv"),"global_building_publication_decisions_jsonl":relative(root/"global_building_publication_decisions.jsonl"),"broad_fragment_quarantine_candidates_csv":relative(root/"broad_fragment_quarantine_candidates.csv"),"broad_fragment_quarantine_candidates_jsonl":relative(root/"broad_fragment_quarantine_candidates.jsonl"),"broad_fragment_quarantine_candidates_geojson":relative(root/"broad_fragment_quarantine_candidates.geojson"),**glb}}
    write_json(root/"manifest.json",manifest);write_json(root/"PREDICTION_ONLY_METRICS.json",summary);print(json.dumps(summary,indent=2));return 0

if __name__=="__main__": raise SystemExit(main())
