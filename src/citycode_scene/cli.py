"""Audit, propose, review and replay scene placement corrections on CPU."""
from pathlib import Path
import argparse
import json

from .cache import build_cache
from .geometry import json_write, sha256
from .pairs import audit_pairs
from .replay import apply_decisions
from .roads import road_audit
from .ground import build_background_maps, audit_contacts, ground_proposals


def resolve(root, value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root/path).resolve()


def audit(config, root, output, *, binding=None, reuse_cache=None):
    output = Path(output).resolve()
    binding = Path(binding).resolve() if binding else resolve(root, config['building_binding'])
    cache = output/'cache'
    index = build_cache(binding, cache, workers=config.get('cpu_workers', 4), reuse_cache_dir=reuse_cache)
    buildings = audit_pairs(cache, output/'buildings')
    roads = None
    if config.get('lane_xyz'):
        roads = road_audit(cache, resolve(root, config['lane_xyz']), output/'roads', save_masks=config.get('save_masks', False))
    ground = None
    if config.get('background'):
        maps = build_background_maps(root, resolve(root, config['background']), output/'ground_maps', **config.get('ground_map_options', {}))
        ground = audit_contacts(cache, maps, output/'ground', tolerance_m=config.get('ground_tolerance_m', .3))
    summary = {'schema':'citycode.scene_audit.v1', 'binding_sha256':sha256(binding),
        'placements':len(index['records']), 'building_pairs':buildings,
        'road_reports':{k:v['summary'] for k,v in (roads or {}).items()},
        'ground_summary':ground['summary'] if ground else None,
        'static_geometry_only':True, 'formal_models_modified':False}
    json_write(output/'SUMMARY.json', summary)
    return summary


def propose(config, root, audit_dir):
    """Produce measured candidates and a review queue, never silently delete."""
    from .optimizer import PlacementSearch
    audit_dir = Path(audit_dir).resolve()
    cache = audit_dir/'cache'
    source = json.loads((cache/'INDEX.json').read_text())
    source_binding = resolve(root, source['binding'])
    if not source_binding.is_file() or sha256(source_binding) != source['binding_sha256']:
        raise ValueError('Binding changed after audit; rerun the audit before proposing corrections')
    lane_summary = audit_dir/'roads/vehicle_4_5m/SUMMARY.json'
    if config.get('lane_xyz') and lane_summary.is_file():
        expected_lane = json.loads(lane_summary.read_text())['lane_geojson_sha256']
        if sha256(resolve(root, config['lane_xyz'])) != expected_lane:
            raise ValueError('Road geometry changed after audit; rerun the audit')
    ground_summary = audit_dir/'ground/GROUND_CONTACTS.json'
    if config.get('background') and ground_summary.is_file():
        expected_background = json.loads(ground_summary.read_text())['background_sha256']
        if sha256(resolve(root, config['background'])) != expected_background:
            raise ValueError('Background changed after audit; rerun the audit')
    search = PlacementSearch(cache, audit_dir/'proposals', resolve(root, config['lane_xyz']) if config.get('lane_xyz') else None)
    measured = json.loads((audit_dir/'buildings/PAIRS.json').read_text())['pairs']
    queue = []
    for row in measured:
        if row['same_building_family'] or row['classification'] != 'clear_overlap':
            continue
        candidates = search.propose_pair(row['asset_a'], row['asset_b'])
        queue.append({'kind':'building_pair', 'asset_ids':[row['asset_a'], row['asset_b']],
            'before':row, 'candidates':candidates[:5], 'agent_decision_required':True})
    if config.get('lane_xyz'):
        from .navigation import propose_navigation
        seen_vehicle_groups = set()
        for name in ['vehicle_4_5m','pedestrian_2_2m']:
            path = audit_dir/'roads'/name/'DETAILS.json'
            if path.is_file():
                for row in json.loads(path.read_text())['instances']:
                    if 'centerline_obstruction' in row['status']:
                        candidates = []
                        if name.startswith('vehicle'):
                            ids = search.group(row['id'])
                            key = tuple(sorted(ids))
                            if key in seen_vehicle_groups:
                                continue
                            seen_vehicle_groups.add(key)
                            candidates = propose_navigation(search, ids, search.road_model,
                                max_shift_m=config['policy']['maximum_horizontal_shift_m'],
                                min_scale=config['policy'].get('preferred_minimum_horizontal_scale', .95))[:5]
                        queue.append({'kind':'vehicle_clearance' if name.startswith('vehicle') else 'walking_route_context_review',
                            'asset_ids':[row['id']], 'before':row,
                            'candidates':candidates,
                            'agent_decision_required':True,
                            'automatic_exclusion':False,
                            'instruction':'Try bounded translation and uniform horizontal shrink; check all neighbours and ground afterwards.' if name.startswith('vehicle') else
                                'Allow steps, entrance and arcade edge contacts. Correct only a clearly blocked passage; a route flag alone does not justify deletion.'})
    ground_path = audit_dir/'ground/GROUND_CONTACTS.json'
    ground_draft = None
    if ground_path.is_file():
        ground_report = json.loads(ground_path.read_text())
        ground_draft = ground_proposals(ground_report, config['policy'])
        for row in ground_report['records']:
            g = row.get('scene_ground_gap') or {}
            c = row.get('continuous_ground_gap') or {}
            if g.get('median_m',0)>.5 or g.get('median_m',0)<-.5:
                queue.append({'kind':'ground_contact', 'asset_ids':[row['id']],
                    'scene_gap':g, 'continuous_gap':c, 'agent_decision_required':True,
                    'instruction':'Resample after XY changes; preserve multi-part joints. A generated background bump alone is not evidence to raise the whole building.'})
    binding_folder = source_binding.parent
    scene_inputs = []
    import os
    for role, key in [('background','background'), ('lane_xyz','lane_xyz')]:
        if config.get(key):
            path = resolve(root, config[key])
            scene_inputs.append({'role':role, 'path':os.path.relpath(path,binding_folder), 'sha256':sha256(path)})
    result = {'schema':'citycode.scene_agent_review_queue.v1', 'status':'draft_agent_review_required',
        'input_binding_sha256':source['binding_sha256'], 'queue':queue,
        'scene_inputs':scene_inputs,
        'ground_proposals':ground_draft,
        'policy':config['policy'], 'decisions':[], 'automatic_exclusions':0,
        'review_contract':'Agent inspects measured candidates and scene evidence, chooses adjust/retain or exceptional exclude, then freezes a citycode.scene_placement_decisions.v1 recipe. Rerun audit after application.'}
    json_write(audit_dir/'AGENT_REVIEW_QUEUE.json',result)
    return {'review_cases':len(queue),'queue':str(audit_dir/'AGENT_REVIEW_QUEUE.json')}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=Path('workflow/04_citycode_assembly/physical.standard.json'))
    parser.add_argument('--root',type=Path)
    sub=parser.add_subparsers(dest='command',required=True)
    check=sub.add_parser('audit',help='Read-only full actual-mesh scene check')
    check.add_argument('--binding',type=Path)
    check.add_argument('--output',type=Path)
    check.add_argument('--reuse-cache',type=Path)
    draft=sub.add_parser('propose',help='Measured proposals and context-review queue for an agent')
    draft.add_argument('--audit-dir',type=Path,required=True)
    replay=sub.add_parser('replay',help='Apply the packaged frozen decisions; no image API or generation')
    replay.add_argument('--decisions',type=Path)
    replay.add_argument('--output-binding',type=Path)
    replay.add_argument('--verify',action='store_true',help='Rebuild and audit all physical placements after replay')
    replay.add_argument('--audit-output',type=Path)
    replay.add_argument('--reuse-cache',type=Path)
    args=parser.parse_args(argv)
    checkout = Path(__file__).resolve().parents[2]
    default_root = checkout if (checkout/'workflow').is_dir() else Path.cwd()
    root=args.root.resolve() if args.root else default_root
    config_path=resolve(root,args.config)
    config=json.loads(config_path.read_text())
    if args.command=='audit':
        result=audit(config,root,resolve(root,args.output or config['audit_output']),
            binding=resolve(root,args.binding) if args.binding else None,
            reuse_cache=resolve(root,args.reuse_cache) if args.reuse_cache else None)
    elif args.command=='propose':
        result=propose(config,root,resolve(root,args.audit_dir))
    else:
        output=resolve(root,args.output_binding or config['physical_binding'])
        result=apply_decisions(resolve(root,config['building_binding']),resolve(root,args.decisions or config['frozen_decisions']),output)
        if args.verify:
            result={'replay':result,'audit':audit(config,root,resolve(root,args.audit_output or config['audit_output']),
                binding=output,reuse_cache=resolve(root,args.reuse_cache) if args.reuse_cache else None)}
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
