from pathlib import Path
from functools import lru_cache
import json,time,math
from collections import Counter
import numpy as np
from scipy.ndimage import distance_transform_edt
from shapely.geometry import shape
from shapely.strtree import STRtree
CACHE = None
CELL = .25
records = []
by = {}

def configure(cache_dir):
 global CACHE, CELL, records, by, index
 CACHE = Path(cache_dir).resolve()
 index = json.loads((CACHE/'INDEX.json').read_text())
 CELL = index['cell_m']
 records = index['records']
 by = {r['id']: r for r in records}
 source_entity = {sid:r['entity_id'] for r in records if r['kind']=='base' for sid in r['source_ids']}
 for r in records:
  r['physical_family_id'] = source_entity.get(r.get('parent_source_id'),r['entity_id'])
 load_columns.cache_clear()

@lru_cache(maxsize=64)
def load_columns(path):
 with np.load(CACHE/path) as n:d={k:n[k] for k in n.files}
 return prepare_columns(d)

def prepare_columns(d):
 d['gx'],d['gy']=np.rint(d['origin']/CELL).astype(int);d['h'],d['w']=map(int,d['shape']);mask=np.isfinite(d['zmin']);d['interior_radius']=np.maximum(0,distance_transform_edt(mask)*CELL-CELL/2)
 return d

def columns_for(record):
 return record['_columns'] if '_columns' in record else load_columns(record['columns'])

def spans(d,gx,gy):
 x,y=gx-d['gx'],gy-d['gy']
 if not (0<=x<d['w'] and 0<=y<d['h']):return [],False
 pos=y*d['w']+x;a,b=map(int,d['offsets'][pos:pos+2]);z=d['hits'][a:b]
 if len(z)%2:return [],True
 return [(float(z[i]),float(z[i+1])) for i in range(0,len(z),2)],False

def merge(intervals):
 out=[]
 for lo,hi in sorted(intervals):
  if out and lo<=out[-1][1]+.002:out[-1]=(out[-1][0],max(hi,out[-1][1]))
  else:out.append((lo,hi))
 return out

def actual_spans(r,gx,gy):
 intervals=[];uncertain=False
 for component in r.get('components',[r]):
  values,unknown=spans(columns_for(component),gx,gy);intervals.extend(values);uncertain|=unknown
 return merge(intervals),uncertain

def intersection_length(a,b):
 total=0.;i=j=0
 while i<len(a) and j<len(b):
  total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
  if a[i][1]<b[j][1]:i+=1
  else:j+=1
 return total

def measure(a,b):
 ca=columns_for(a);cb=columns_for(b);gx0=max(ca['gx'],cb['gx']);gy0=max(ca['gy'],cb['gy']);gx1=min(ca['gx']+ca['w'],cb['gx']+cb['w']);gy1=min(ca['gy']+ca['h'],cb['gy']+cb['h'])
 if gx1<=gx0 or gy1<=gy0:return None
 sa=(slice(gy0-ca['gy'],gy1-ca['gy']),slice(gx0-ca['gx'],gx1-ca['gx']));sb=(slice(gy0-cb['gy'],gy1-cb['gy']),slice(gx0-cb['gx'],gx1-cb['gx']))
 lowa,hia=ca['zmin'][sa],ca['zmax'][sa];lowb,hib=cb['zmin'][sb],cb['zmax'][sb]
 both=np.isfinite(lowa)&np.isfinite(lowb);xy=int(both.sum())
 if not xy:return None
 envelope=np.minimum(hia,hib)-np.maximum(lowa,lowb);eligible=both&(envelope>.05)
 known=[];unknown=[];core=0.;vol=0.;maxheight=0.;maxwhere=None
 for y,x in zip(*np.where(eligible)):
  gx,gy=int(gx0+x),int(gy0+y);aa,ua=actual_spans(a,gx,gy);bb,ub=actual_spans(b,gx,gy);dz=intersection_length(aa,bb)
  radius=min(float(ca['interior_radius'][gy-ca['gy'],gx-ca['gx']]),float(cb['interior_radius'][gy-cb['gy'],gx-cb['gx']]))
  if dz>.05:
   known.append((gx,gy,dz,radius));vol+=dz*CELL**2
   if radius>core:core=radius;maxwhere=[(gx+.5)*CELL,(gy+.5)*CELL]
   maxheight=max(maxheight,dz)
  if (ua or ub) and envelope[y,x]>.05:unknown.append((gx,gy,float(envelope[y,x])))
 area=len(known)*CELL**2;unknownarea=len(unknown)*CELL**2
 if area and core>.35 and maxheight>.5:
  classification='clear_overlap' if area>=10 and core>=.5 and maxheight>=1 else 'review_overlap'
 elif area:classification='minor_contact'
 elif unknownarea:classification='uncertain_open_surface'
 else:classification='XY_only_no_common_solid'
 return {'asset_a':a['id'],'asset_b':b['id'],'same_building_family':a['physical_family_id']==b['physical_family_id'],'family_a':a['physical_family_id'],'family_b':b['physical_family_id'],'kind_a':a['kind'],'kind_b':b['kind'],
 'classification':classification,'xy_sample_overlap_m2':xy*CELL**2,'known_solid_overlap_area_m2':area,'uncertain_column_area_m2':unknownarea,'known_overlap_volume_m3':vol,'max_common_vertical_occupied_length_m':maxheight,'max_horizontal_interior_radius_m':core,
 'overlap_fraction_smaller_projection':area/max(.0625,min(a['sampled_projection_area_m2'],b['sampled_projection_area_m2'])),'core_location_xy':maxwhere,
 'proof_scope':'0.25m vertical rays through actual 0.1M world meshes; even crossings paired per component, components unioned; odd columns remain uncertain',
 'sample_overlap_xy':[[float((x+.5)*CELL),float((y+.5)*CELL)] for x,y,dz,rr in sorted(known,key=lambda q:(q[3],q[2]),reverse=True)[:12]]}
def audit_pairs(cache_dir, output_dir):
 configure(cache_dir)
 R=Path(output_dir);R.mkdir(parents=True,exist_ok=True)
 start=time.time();polys=[shape(r['footprint_world']).buffer(CELL) for r in records];tree=STRtree(polys);results=[];candidates=0
 for i,a in enumerate(records):
  for j in tree.query(polys[i],predicate='intersects'):
   j=int(j)
   if j<=i:continue
   b=records[j]
   if min(a['world_bounds'][1][2],b['world_bounds'][1][2])-max(a['world_bounds'][0][2],b['world_bounds'][0][2])<=.05:continue
   candidates+=1;value=measure(a,b)
   if value:results.append(value)
  if i%50==0:print(json.dumps({'assets_checked':i,'candidate_pairs':candidates,'overlap_pairs':len(results),'seconds':time.time()-start}),flush=True)
 internal=[]
 for parent in records:
  components=parent.get('components',[])
  for i,a in enumerate(components):
   a={**a,'kind':'composite_component','physical_family_id':parent['physical_family_id']}
   for b in components[i+1:]:
    b={**b,'kind':'composite_component','physical_family_id':parent['physical_family_id']};value=measure(a,b)
    if value:internal.append(value)
 results.sort(key=lambda r:(r['classification']=='clear_overlap',r['known_overlap_volume_m3']),reverse=True)
 summary={'schema':'citycode.building_collision_audit.v1','status':'measured','binding_sha256':index['binding_sha256'],'placements':len(records),'broadphase_candidate_pairs':candidates,'sampled_XY_overlap_pairs':len(results),
 'independent_building_pair_counts':dict(Counter(r['classification'] for r in results if not r['same_building_family'])),'same_building_family_pair_counts':dict(Counter(r['classification'] for r in results if r['same_building_family'])),'internal_composite_pair_counts':dict(Counter(r['classification'] for r in internal)), 'internal_composite_assets_checked':sum(bool(r.get('components')) for r in records), 'internal_component_pairs_tested':sum(len(r.get('components',[]))*(len(r.get('components',[]))-1)//2 for r in records),
 'thresholds':{'raster_cell_m':CELL,'candidate_only_boundary_padding_m':.25,'separated_vertical_tolerance_m':.05,'review_horizontal_interior_radius_m':.35,'review_vertical_overlap_m':.5,'clear_overlap_area_m2':10,'clear_overlap_radius_m':.5,'clear_overlap_vertical_m':1,'small_overlap_is_not_automatically_a_placement_error':True},
 'no_automatic_changes':True,'physics_engine_simulation_performed':False,'mesh_closure_not_assumed_for_odd_columns':True,'elapsed_seconds':time.time()-start}
 (R/'SUMMARY.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n');(R/'PAIRS.json').write_text(json.dumps({'pairs':results,'internal_components':internal},ensure_ascii=False,indent=2)+'\n');print(json.dumps(summary,ensure_ascii=False),flush=True)

 return summary
