"""Require explicit cross-patch visible background before proposed edge removal.

Input geometry proposals come from prediction-only boundary hypotheses. This
gate never creates its own expansion or removes extra area outside a proposal.
Camera height, projection, strict ROI and predicted depth occlusion are retained.
"""
from __future__ import annotations
import copy
from collections import defaultdict
import json
import math
from pathlib import Path
import numpy as np
import shapely
from shapely.geometry import Polygon, mapping, shape
from shapely.ops import unary_union
from .projection import ViewEvidence


def parts(g):
    if g.is_empty:
        return []
    if g.geom_type == 'Polygon':
        return [g]
    return [p for child in getattr(g, 'geoms', []) for p in parts(child)]


def _validate_config(config):
    for key in ('cell_size_m', 'depth_tolerance_m', 'background_margin_px',
                'image_margin_px', 'negative_fraction_per_cell',
                'minimum_negative_patches'):
        value = config[key]
        if not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Nonfinite or nonscalar '+key)
    if not 0 < config['cell_size_m'] <= 2:
        raise ValueError('Boundary cell size must be in (0,2] metres')
    if config['depth_tolerance_m'] < 0 or config['image_margin_px'] < 0:
        raise ValueError('Negative tolerance')
    if not .6 <= config['negative_fraction_per_cell'] <= 1:
        raise ValueError('At least three of five cell samples must agree')
    if (config['minimum_negative_patches'] < 2 or
            int(config['minimum_negative_patches']) != config['minimum_negative_patches']):
        raise ValueError('At least two independent patches are required')
    uncertainty = config.get('height_uncertainty_m', 0.)
    if not isinstance(uncertainty, (int, float)) or not math.isfinite(uncertainty) or uncertainty < 0:
        raise ValueError('Height uncertainty must be finite and nonnegative')


def prepare_cells(parent, proposed, cell_size):
    parents = {f['properties']['predicted_global_building_id']: f for f in parent}
    if set(parents) != {f['properties']['predicted_global_building_id'] for f in proposed}:
        raise ValueError('Boundary gate cannot change building identities')
    records = []
    for proposal in proposed:
        p = proposal['properties']; gid = p['predicted_global_building_id']
        original = parents[gid]; op = original['properties']
        old, new = shape(original['geometry']), shape(proposal['geometry'])
        if any(op.get(key) != p.get(key) for key in
               ('base_z_m', 'top_z_m', 'height_m', 'roof_layers', 'canonical_observation_id')):
            raise ValueError('Boundary gate requires unchanged height, roof and identity')
        if new.difference(old).area > 1e-7:
            raise ValueError('This gate accepts shrink-only proposals')
        removed = old.difference(new)
        if removed.area < 1e-6:
            continue
        # Body exterior projects at its modeled body roof, not the highest tower.
        top = float(op.get('body_top_z_m', op['top_z_m']))
        used = set()
        for part in parts(removed):
            x0, y0, x1, y1 = np.asarray(part.bounds) / cell_size
            ix = np.arange(math.floor(x0), math.ceil(x1))
            iy = np.arange(math.floor(y0), math.ceil(y1))
            if len(ix)*len(iy) > 2000000:
                raise ValueError('Unexpected huge local boundary proposal')
            xx, yy = np.meshgrid(ix, iy)
            cells = shapely.box(xx.ravel()*cell_size, yy.ravel()*cell_size,
                                (xx.ravel()+1)*cell_size, (yy.ravel()+1)*cell_size)
            hit = np.flatnonzero(shapely.intersects(cells, part))
            for k in hit:
                key = (int(xx.ravel()[k]), int(yy.ravel()[k]))
                if key in used:
                    continue
                used.add(key)
                clipped = cells[k].intersection(removed)
                if clipped.area < 1e-7:
                    continue
                # Evidence must sample the actual proposed removal, including
                # sub-cell slivers. A cell-centre/background hit outside that
                # polygon cannot justify deleting any part of this building.
                for component in parts(clipped):
                    if component.area < 1e-7:
                        continue
                    xy = [component.representative_point().coords[0]]
                    xy.extend(component.exterior.interpolate(f, normalized=True).coords[0]
                              for f in (0., .25, .5, .75))
                    coordinates = np.c_[np.asarray(xy), np.full(5, top)]
                    records.append({'building_id': gid, 'cell': cells[k],
                                    'proposed_removal': component, 'xyz': coordinates})
    return records


def accumulate_evidence(records, samples, config):
    """Every physical patch votes once; a visible roof silhouette vetoes trim."""
    _validate_config(config)
    if not records:
        return np.zeros(0, int), np.zeros(0, bool), []
    points = np.concatenate([r['xyz'] for r in records])
    patch_negative = {}; patch_roof = {}; summaries = []
    for sample_id, paths in sorted(samples.items()):
        camera = json.loads(Path(paths['camera']).read_text())
        bounds = camera.get('data_bounds_xy_world_m')
        if bounds is None:
            raise ValueError('Patch prism needed for boundary visibility')
        x0, y0, x1, y1 = bounds
        relevant_points = ((points[:, 0] >= x0) & (points[:, 0] <= x1)
                           & (points[:, 1] >= y0) & (points[:, 1] <= y1))
        relevant_cells = np.flatnonzero(relevant_points.reshape(-1, 5).any(axis=1))
        if not len(relevant_cells):
            continue
        evidence = ViewEvidence.load(paths, int(config['background_margin_px']))
        indices = (relevant_cells[:, None]*5+np.arange(5)).ravel()
        # All nonzero instance labels are a conservative roof veto. No canonical
        # instance ID is reused across views and no other building is background.
        owners = np.unique(evidence.instance_ids[evidence.instance_ids > 0])
        uncertainty = config.get('height_uncertainty_m', 0.)
        offsets = (-uncertainty, 0., uncertainty) if uncertainty else (0.,)
        negative_points = np.ones(len(indices), bool)
        roof_points = np.zeros(len(indices), bool)
        for dz in offsets:
            query = points[indices].copy(); query[:, 2] += dz
            states = evidence.classify(query, owners, config['depth_tolerance_m'],
                                       config['image_margin_px'])
            # A small existing-height uncertainty interval is a confidence
            # check only: no world height is changed or retuned by this module.
            negative_points &= states['explicit_background']
            roof_points |= states['same_instance_silhouette']
        neg = negative_points.reshape(-1, 5).mean(axis=1)
        roof = roof_points.reshape(-1, 5).any(axis=1)
        full_negative = np.zeros(len(records), bool)
        full_roof = np.zeros(len(records), bool)
        full_negative[relevant_cells] = neg >= config['negative_fraction_per_cell']
        full_roof[relevant_cells] = roof
        patch_id = sample_id.split('__')[0]
        # Should future data include several views, agreement is conservative:
        # any clear roof view vetoes; at most one negative vote per patch.
        patch_negative[patch_id] = patch_negative.get(patch_id, np.zeros(len(records), bool)) | full_negative
        patch_roof[patch_id] = patch_roof.get(patch_id, np.zeros(len(records), bool)) | full_roof
        summaries.append({'sample_id': sample_id, 'tested_cells': len(relevant_cells),
                          'negative_cells': int(full_negative.sum()),
                          'roof_veto_cells': int(full_roof.sum())})
    negative_count = np.zeros(len(records), int)
    roof_veto = np.zeros(len(records), bool)
    for patch in patch_negative:
        negative_count += patch_negative[patch] & ~patch_roof[patch]
        roof_veto |= patch_roof[patch]
    return negative_count, roof_veto, summaries


def gated_refine(parent, proposed, samples, config):
    _validate_config(config)
    cells = prepare_cells(parent, proposed, config['cell_size_m'])
    counts, veto, evidence = accumulate_evidence(cells, samples, config)
    accepted = (counts >= config['minimum_negative_patches']) & ~veto
    by_building = defaultdict(list)
    for record, keep in zip(cells, accepted):
        if keep:
            by_building[record['building_id']].append(record['proposed_removal'])
    result = copy.deepcopy(parent); changes = []
    for feature in result:
        p = feature['properties']; gid = p['predicted_global_building_id']
        if gid not in by_building:
            continue
        old = shape(feature['geometry']); old_parts = parts(old)
        removal = unary_union(by_building[gid])
        # Grid clipping must not create courtyards or detach a building fragment.
        exterior = unary_union([Polygon(part.exterior) for part in old_parts]).boundary
        removal = unary_union([part for part in parts(removal) if part.distance(exterior) < 1e-7])
        if removal.is_empty:
            continue
        new = old.difference(removal)
        if (not new.is_valid or new.is_empty or len(parts(new)) != len(old_parts)
                or sum(len(x.interiors) for x in parts(new)) != sum(len(x.interiors) for x in old_parts)):
            continue
        if (old.area-new.area > .03*old.area or
                any(new.intersection(component).area < .95*component.area
                    for component in old_parts)):
            continue
        old_holes = unary_union([Polygon(ring) for part in old_parts for ring in part.interiors])
        new_holes = unary_union([Polygon(ring) for part in parts(new) for ring in part.interiors])
        if old_holes.symmetric_difference(new_holes).area > 1e-7:
            continue
        if any(shape(layer['geometry']).difference(new).area > 1e-7 for layer in p.get('roof_layers', [])):
            continue
        if new.difference(old).area > 1e-7:
            raise AssertionError('Visibility gate introduced expansion')
        delta = old.area-new.area
        if delta < .05:
            continue
        feature['geometry'] = mapping(new)
        p.update(footprint_area_m2=new.area, centroid_x_m=new.centroid.x,
                 centroid_y_m=new.centroid.y)
        p['round4_visible_boundary'] = {'method': 'explicit_background_multiregion_gate',
                                       'removed_area_m2': delta}
        changes.append({'prediction_id': gid, 'removed_area_m2': delta})
    return result, {'proposed_cells': len(cells), 'accepted_cells': int(accepted.sum()),
                    'roof_veto_cells': int(veto.sum()), 'changes': changes,
                    'removed_area_m2': sum(r['removed_area_m2'] for r in changes),
                    'view_evidence': evidence, 'GT_used': False}
