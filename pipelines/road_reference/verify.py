"""Verify the portable GT road bundle: hashes, identities, links and geometry."""
import argparse,collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
def read(p):return json.loads(p.read_text())
def unique(rows,key):
 ids=[r[key] for r in rows];assert len(ids)==len(set(ids)),f'Duplicate {key}';return set(ids)
def verify(root):
 m=read(root/'manifest.json');assert m['schema']=='citycode.gt_road_reference.v1'
 for row in m['files']:
  p=(root/row['path']).resolve();assert p.is_relative_to(root.resolve()) and p.is_file() and not (root/row['path']).is_symlink()
  assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],str(p)
 lanes=read(root/m['roles']['vehicle_placement'])['features'];sections=read(root/m['roles']['road_surfaces'])['features']
 lane_ids=unique([f['properties'] for f in lanes],'lane_id');section_ids=unique([f['properties'] for f in sections],'road_section_id')
 for f in lanes:
  p=f['properties'];assert p['road_section_id'] in section_ids
  assert f['geometry']['type']=='LineString' and len(f['geometry']['coordinates'])>=2
  assert all(len(x)==3 and all(math.isfinite(v) for v in x) for x in f['geometry']['coordinates'])
  for key in ['incoming_lane_ids','outgoing_lane_ids','adjacent_lane_ids']:
   assert set(p.get(key,[]))<=lane_ids,(p['lane_id'],key)
 for f in sections:assert set(f['properties'].get('lane_ids',[]))<=lane_ids
 adjacency=read(root/m['roles']['road_section_adjacency'])['nodes']
 assert unique(adjacency,'road_section_id')==section_ids
 for n in adjacency:
  for key in ['successor_road_section_ids','predecessor_road_section_ids','adjacent_road_section_ids']:
   assert set(n.get(key,[]))<=section_ids,(n['road_section_id'],key)
 counts={};graphs={}
 for name in ['ground_only','height_aware']:
  g=read(root/name/'road_graph.json');ids=unique(g['nodes'],'node_id');unique(g['edges'],'edge_id')
  for e in g['edges']:
   assert e['source'] in ids and e['target'] in ids
   xy=e['world_polyline_xy_m'];assert len(xy)>=2 and all(len(x)==2 and all(math.isfinite(v) for v in x) for x in xy)
   if name=='height_aware':
    xyz=e['world_polyline_xyz_m'];assert len(xyz)==len(xy)
    assert all(len(x)==3 and all(math.isfinite(v) for v in x) for x in xyz)
  for kind in ['nodes','edges']:
   assert [json.loads(line) for line in (root/name/(kind+'.jsonl')).read_text().splitlines() if line.strip()]==g[kind]
  counts[name]={'nodes':len(g['nodes']),'edges':len(g['edges']),'elevation_classes':dict(collections.Counter(e['elevation_class'] for e in g['edges']))};graphs[name]=g
 elevated=read(root/m['roles']['elevated_and_ramp_reference'])['features'];support=read(root/m['roles']['elevated_support'])['features']
 expected={e['edge_id'] for e in graphs['height_aware']['edges'] if e['elevation_class'] in ['ramp','elevated']}
 assert {f['properties']['edge_id'] for f in elevated}==expected
 assert {f['properties']['road_section_id'] for f in support}<=section_ids
 return {'status':'passed','files_verified':len(m['files']),'bytes':sum(r['bytes'] for r in m['files']),'road_sections':len(sections),'lanes':len(lanes),'vehicle_tagged_lanes':sum('Vehicle' in f['properties']['tags'] for f in lanes),'graphs':counts,'elevated_ramp_edges':len(elevated),'elevated_support_sections':len(support),'checks':['SHA256 and byte sizes','unique IDs','lane XYZ finite and directed link targets exist','section membership and adjacency targets exist','corridor endpoints exist','JSONL matches graph arrays','elevated subset matches layered graph'],'note':'Structural integrity only; UE asset assembly, collision checks and traffic behaviour remain pending.'}
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--bundle',type=Path,default=ROOT/'data/smallcity_p384_gt_roads');p.add_argument('--output',type=Path);a=p.parse_args();result=verify(a.bundle)
 if a.output:a.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(result,ensure_ascii=False))
