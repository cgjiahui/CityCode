"""Observation replacement, local consensus boundaries and identity joins."""
import argparse,copy,hashlib,json
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
from shapely.geometry import shape,mapping
from shapely.ops import unary_union,polygonize
from shapely.strtree import STRtree
from ._config import SOURCE_RUN_ROOT
from .height_refinement import install_guard
ROOT = Path("UNCONFIGURED_ROOT")
SOURCE = Path("UNCONFIGURED_SOURCE")
POOL = Path("UNCONFIGURED_POOL")
OBS=SOURCE_RUN_ROOT/'global_fused/building_observation_footprints.geojson'
CONFIGS={}

def safe(g,old,other):
 if g.is_empty or g.geom_type not in ['Polygon','MultiPolygon'] or not g.is_valid or not .7<=g.area/old.area<=1.3:return False
 for h in other:
  if not g.intersects(h):continue
  a=g.intersection(h).area;before=old.intersection(h).area
  if a>=20 and a/min(g.area,h.area)>=.5:return False
  if a-before>max(10,.03*g.area):return False
 return True

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--configs',nargs='*');ap.add_argument('--extra-configs');ap.add_argument('--input-root',type=Path,default=SOURCE);args=ap.parse_args();configs=dict(CONFIGS)
 if args.extra_configs:configs.update(json.loads(args.extra_configs))
 src=args.input_root/'global_building_footprints.geojson';audit=install_guard([src,POOL,OBS]);doc=json.loads(src.read_text());pool=json.loads(POOL.read_text())['features'];obs=json.loads(OBS.read_text())['features'];pgeoms=[shape(f['geometry']) for f in pool];tree=STRtree(pgeoms);oids={f['properties']['observation_id']:f for f in obs};groups=defaultdict(list)
 for o in obs:
  p=o['properties'];g=shape(o['geometry'])
  if not g.is_valid:g=g.buffer(0)
  if p['boundary_status']=='boundary_clipped_exclude' or p['stable_layer_count']<1:continue
  ranked=[]
  for j in tree.query(g,predicate='intersects'):
   a=g.intersection(pgeoms[j]).area;iou=a/(g.area+pgeoms[j].area-a)
   if iou>=.25:ranked.append((iou,int(j)))
  if not ranked:continue
  ranked.sort(reverse=True);iou,j=ranked[0]
  if len(ranked)>1 and ranked[1][0]>.9*iou:continue
  owner=pool[j]['properties'];height=owner.get('body_top_z_m',owner['top_z_m'])
  if abs(p['top_z_m']-height)>max(12,.3*owner['height_m']):continue
  w=iou**2*max(.1,p['preliminary_quality']/100)**2*(1 if p['boundary_status']=='interior_keep' else .4)*(1 if p['reliable_core_contains_centroid'] else .5)
  groups[owner['predicted_global_building_id']].append((o,g,iou,w))
 def peers(gid,threshold):
  bypatch={}
  for x in groups[gid]:
   if x[2]<threshold:continue
   patch=x[0]['properties']['patch_id']
   if patch not in bypatch or x[3]>bypatch[patch][3]:bypatch[patch]=x
  return list(bypatch.values())
 def vote_geometry(pp,t):
  total=sum(x[3] for x in pp);cells=polygonize(unary_union([x[1].boundary for x in pp]));keep=[]
  for cell in cells:
   c=cell.representative_point()
   if sum(x[3] for x in pp if x[1].covers(c))>=t*total:keep.append(cell)
  return unary_union(keep)
 ROOT.mkdir(exist_ok=True)
 for name,cfg in configs.items():
  if args.configs and name not in args.configs:continue
  out=ROOT/'trials'/name
  if out.exists():raise FileExistsError(out)
  fs=copy.deepcopy(doc['features']);changes=[];method=cfg.get('method');geoms=[shape(f['geometry']) for f in fs];existing={f['properties']['predicted_global_building_id'] for f in fs}
  if method in ['replace','local']:
   for i,f in enumerate(fs):
    p=f['properties'];old=geoms[i];gid=p['predicted_global_building_id'];pp=peers(gid,cfg['iou']);weak=len(p['consensus_observation_ids'])==1 or p['canonical_boundary_status']!='interior_keep'
    if method=='replace':
     if cfg['scope']=='weak' and not weak:continue
     eligible=[x for x in pp if x[0]['properties']['boundary_status']=='interior_keep' and x[0]['properties']['preliminary_quality']>=70]
     if not eligible:continue
     best=max(eligible,key=lambda x:x[3]);canonical=oids[p['canonical_observation_id']]['properties'];bp=best[0]['properties']
     if bp['observation_id']==p['canonical_observation_id'] or (canonical['boundary_status']=='interior_keep' and bp['preliminary_quality']<canonical['preliminary_quality']+cfg.get('quality_gain',5)):continue
     g=best[1].intersection(old.buffer(cfg.get('max_shift',3)));evidence=[bp['observation_id']]
    else:
     if len(pp)<2:continue
     voted=vote_geometry(pp,cfg['vote']);radius=cfg.get('max_shift',2);g=old.intersection(voted.buffer(cfg.get('tolerance',.25))).union(voted.intersection(old.buffer(radius)));evidence=[x[0]['properties']['observation_id'] for x in pp]
    if old.symmetric_difference(g).area<1 or not safe(g,old,[h for j,h in enumerate(geoms) if j!=i]):continue
    # Keep every modeled upper roof: changing body boundaries must not erase its height evidence.
    roofs=[shape(x['geometry']) for x in p.get('roof_layers',[])];g=unary_union([g]+roofs)
    if not safe(g,old,[h for j,h in enumerate(geoms) if j!=i]):continue
    f['geometry']=mapping(g);p['footprint_area_m2']=g.area;p['centroid_x_m']=g.centroid.x;p['centroid_y_m']=g.centroid.y;p['v7_geometry_method']=method;p['v7_observation_ids']=evidence;geoms[i]=g;changes.append({'id':gid,'action':method,'area_before':old.area,'area_after':g.area,'evidence':evidence})
  elif method=='recover':
   for candidate in pool:
    p=candidate['properties'];gid=p['predicted_global_building_id']
    if gid in existing:continue
    pp=peers(gid,cfg['iou'])
    if len(pp)<cfg['min_patches'] or not any(x[0]['properties']['boundary_status']=='interior_keep' and x[0]['properties']['preliminary_quality']>=70 for x in pp):continue
    old=shape(candidate['geometry']);g=vote_geometry(pp,.5).intersection(old.buffer(3))
    if not safe(g,old,geoms):continue
    f=copy.deepcopy(candidate);p=f['properties'];p['footprint_area_m2']=g.area;p['centroid_x_m']=g.centroid.x;p['centroid_y_m']=g.centroid.y;p['v7_geometry_method']='recovered_with_independent_observations';p['v7_observation_ids']=[x[0]['properties']['observation_id'] for x in pp];f['geometry']=mapping(g)
    if any(shape(l['geometry']).difference(g).area>.1 for l in p.get('roof_layers',[])):continue
    fs.append(f);geoms.append(g);existing.add(gid);changes.append({'id':gid,'action':'recover','evidence':p['v7_observation_ids']})
  elif method=='join':
   # Require two independent observations that each span both adjacent pieces.
   removed=set();st=STRtree(geoms);observation_geoms=[shape(o['geometry']) for o in obs];observation_tree=STRtree(observation_geoms)
   for i,g in enumerate(geoms):
    if i in removed:continue
    for jj in st.query(g.buffer(cfg['gap']),predicate='intersects'):
     j=int(jj)
     if j<=i or j in removed:continue
     p,q=fs[i]['properties'],fs[j]['properties']
     if abs(p['top_z_m']-q['top_z_m'])>3:continue
     union=g.union(geoms[j]);support=set()
     for oi in observation_tree.query(union,predicate='intersects'):
      o=obs[int(oi)];op=o['properties']
      if op['boundary_status']!='interior_keep':continue
      h=observation_geoms[int(oi)]
      if h.intersects(union) and h.intersection(g).area/g.area>=.8 and h.intersection(geoms[j]).area/geoms[j].area>=.8 and h.intersection(union).area/h.area>=.8:support.add(op['patch_id'])
     if len(support)<2:continue
     # Only equal-height bodies without roof layers in this identity-only ablation.
     if p.get('roof_layers') or q.get('roof_layers'):continue
     p.setdefault('v7_joined_ids',[]).append(q['predicted_global_building_id']);p['footprint_area_m2']=union.area;p['centroid_x_m']=union.centroid.x;p['centroid_y_m']=union.centroid.y;fs[i]['geometry']=mapping(union);geoms[i]=union;g=union;removed.add(j);changes.append({'id':p['predicted_global_building_id'],'action':'join','absorbed':q['predicted_global_building_id'],'support_patches':sorted(support)})
   fs=[f for i,f in enumerate(fs) if i not in removed]
  out.mkdir(parents=True);payload=json.dumps({'type':'FeatureCollection','features':fs});(out/'global_building_footprints.geojson').write_text(payload);(out/'changes.json').write_text(json.dumps(changes,indent=2));(out/'manifest.json').write_text(json.dumps({'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),'prediction_contract':'Observation-based structure repair. Height retained for boundary/replacement methods.','parameters':cfg,'prediction_sha256':hashlib.sha256(payload.encode()).hexdigest(),'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [src,POOL,OBS]},'runtime_read_audit':sorted(audit),'building_count':len(fs),'change_count':len(changes)},indent=2));print(name,len(fs),len(changes),flush=True)
if __name__=='__main__':main()
