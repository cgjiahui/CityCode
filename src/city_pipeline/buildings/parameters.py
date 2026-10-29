"""Explicit global postprocessing thresholds; absent overrides preserve V9."""
from __future__ import annotations
import copy
import math

SELECTION_DEFAULTS = {
    'consensus_weight': 1.0, 'area_weight': .5,
    'min_area_m2': 35.0, 'conflict_containment': .15,
}


def normalize_selection(supplied=None):
    """Only scene-wide numeric selection settings; never IDs or reference files."""
    if supplied is not None and not isinstance(supplied, dict):
        raise ValueError('selection must contain global numeric parameters')
    supplied = supplied or {}
    if set(supplied) - set(SELECTION_DEFAULTS):
        raise ValueError('Unsupported selection parameter')
    result = {**SELECTION_DEFAULTS, **supplied}
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite numeric selection parameter required: ' + key)
    if result['consensus_weight'] < 0 or not 0 <= result['area_weight'] <= 1:
        raise ValueError('Invalid selection weight')
    if result['min_area_m2'] <= 0 or not 0 < result['conflict_containment'] <= 1:
        raise ValueError('Invalid selection area or containment')
    return result

DEFAULTS = {
    'boundary': {'offset_m': .75, 'minimum_component_area_retention': .85, 'scope': 'weak'},
    'prune': {'single_all_boundary_max_area_m2': 1200.},
    'filter': {'height_le_m': 2., 'width_lt_m': 8., 'support_equals': 1, 'quality_lt': 80.},
    'local_repair': {'method': 'local', 'iou': .5, 'vote': .53, 'max_shift': 2., 'tolerance': .25},
    'replace_repair': {'method': 'replace', 'iou': .3, 'scope': 'weak', 'max_shift': 2., 'quality_gain': 0.},
}

def stage_parameters(stage, supplied=None):
    """Validate an override without accepting IDs, files or executable expressions."""
    if stage not in DEFAULTS:
        if supplied:
            raise ValueError('This stage has no configurable thresholds: ' + stage)
        return {}
    if supplied is not None and not isinstance(supplied, dict):
        raise ValueError('Stage thresholds must be an object')
    supplied = copy.deepcopy(supplied or {})
    if set(supplied) - set(DEFAULTS[stage]):
        raise ValueError('Unknown stage threshold: ' + ', '.join(sorted(set(supplied) - set(DEFAULTS[stage]))))
    result = {**DEFAULTS[stage], **supplied}
    for key, value in result.items():
        if key == 'method':
            if value != DEFAULTS[stage]['method']:
                raise ValueError('Repair method cannot be changed by a threshold override')
        elif key == 'scope':
            choices = {'weak', 'single', 'all'} if stage == 'boundary' else {'weak', 'all'}
            if value not in choices:
                raise ValueError('Unknown prediction-side scope')
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError('Finite nonnegative numeric threshold required: ' + key)
    if stage == 'boundary' and not 0 < result['minimum_component_area_retention'] <= 1:
        raise ValueError('Area retention must be within (0,1]')
    if stage in {'local_repair', 'replace_repair'} and not .25 <= result['iou'] <= 1:
        raise ValueError('Peer IoU must be within [0.25,1]; observation ownership requires at least 0.25')
    if stage == 'local_repair' and not 0 < result['vote'] <= 1:
        raise ValueError('Voting fraction must be within (0,1]')
    if stage == 'filter':
        if int(result['support_equals']) != result['support_equals'] or result['support_equals'] < 1:
            raise ValueError('Support count must be a positive integer')
        if result['quality_lt'] > 100:
            raise ValueError('Quality is on a 0 to 100 scale')
    return result

def normalize_postprocess(supplied=None):
    if supplied is not None and not isinstance(supplied, dict):
        raise ValueError('postprocess must contain named global stage thresholds')
    supplied = supplied or {}
    if set(supplied) - set(DEFAULTS):
        raise ValueError('Unsupported postprocessing stage')
    return {stage: stage_parameters(stage, params) for stage, params in supplied.items()}
