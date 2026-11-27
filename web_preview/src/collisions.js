import * as THREE from 'three';

export const COLLISION_KINDS = { building: 'Building contact', vehicle: 'Lane edge', ground: 'Ground contact' };
export const COLLISION_COLORS = { building: '#b97a36', vehicle: '#377d88', ground: '#8271a4' };
export const RELATED_COLORS = ['#398ba0', '#d58a3e', '#8672b2'];
const validVector = value => Array.isArray(value) && value.length === 3 && value.every(Number.isFinite);
const validBounds = value => Array.isArray(value) && value.length === 2 && value.every(validVector) && value[0].every((v, i) => v <= value[1][i]);

export function initialCollisionSettings(search) {
  return { collisionEnabled: new URLSearchParams(search).get('inspect') === 'collisions', collisionKind: 'all', collisionPriorityOnly: true };
}

export function collisionCases(audit) {
  const defaults = new Set(audit?.default_case_ids || []), seen = new Set();
  return (audit?.cases || []).filter(item => {
    if (!item?.id || !COLLISION_KINDS[item.kind] || seen.has(item.id)) return false;
    seen.add(item.id); return true;
  }).map(item => ({ ...item, asset_ids: [...new Set(item.asset_ids || [])],
    default_visible: item.default_visible ?? defaults.has(item.id),
    focus_position: validVector(item.focus_position) ? item.focus_position : validBounds(item.bounds) ? item.bounds[0].map((v, i) => (v + item.bounds[1][i]) / 2) : null,
    patches: (item.patches || []).filter(p => validVector(p.center) && validVector(p.size) && p.size.every(v => v >= 0)) }));
}

export function filterCollisionCases(cases, { kind = 'all', priorityOnly = true } = {}) {
  return cases.filter(item => (kind === 'all' || item.kind === kind) && (!priorityOnly || item.default_visible));
}

export function initialCollisionCase(cases, hash = '') {
  let decoded = hash.replace(/^#/, '');
  try { decoded = decodeURIComponent(decoded); } catch { /* An incomplete URL hash is not a scene error. */ }
  return cases.find(c => c.id === decoded) || cases.find(c => c.default_visible && c.metrics?.classification === 'clear_overlap') || cases.find(c => c.default_visible) || cases[0] || null;
}

export function collisionStatus(item) {
  if (item.kind === 'ground') return { label: 'Local terrain rise', tone: 'ground', explanation: 'The building base aligns with the continuous ground. Local detail accounts for the difference in elevation.' };
  if (item.kind === 'vehicle') return { label: 'Lane-edge contact', tone: 'vehicle', explanation: 'Lane-edge contact. Centerline obstruction is measured separately. Use the neighborhood view to inspect the boundary.' };
  if (item.review_status === 'reviewed_retained' || item.severity === 'resolved') return { label: 'Reviewed and retained', tone: 'building', explanation: 'This contact was reviewed and retained during physical postprocessing. Inspect the surrounding boundaries in the neighborhood view.' };
  if (item.review_status === 'boundary_review' || ['review_overlap', 'clear_overlap'].includes(item.metrics?.classification)) return { label: 'Boundary contact', tone: 'building', explanation: 'The geometry check detected a boundary overlap. Inspect the adjacent geometry to assess its extent.' };
  return { label: 'Minor contact', tone: 'minor', explanation: 'Small overlaps at steps, walls and adjoining boundaries are allowed.' };
}

const metric = (label, value, unit = '', digits = 2) => Number.isFinite(value) ? { label, value: `${value.toFixed(digits)}${unit}` } : null;
export function collisionMetrics(item) {
  const m = item.metrics || {};
  if (item.kind === 'building') return [metric('Projected contact area', m.overlap_area_m2, ' m²'),
    metric('Horizontal penetration depth', m.horizontal_depth_m, ' m'), metric('Maximum shared height', m.max_vertical_overlap_m, ' m'),
    metric('Area before adjustment', m.before_overlap_area_m2, ' m²')].filter(Boolean);
  if (item.kind === 'vehicle') return [metric('Edge contact area', m.occupied_area_m2, ' m²'),
    metric('Maximum edge intrusion', m.max_intrusion_m, ' m'), metric('Obstructed centerline length', m.max_single_lane_obstructed_length_m, ' m')].filter(Boolean);
  return [metric('Offset from continuous ground', m.continuous_gap_median_m, ' m'), metric('Offset from terrain detail', m.scene_gap_median_m, ' m'),
    metric('Sampled footprint area', m.foot_area_m2, ' m²')].filter(Boolean);
}

// The full list remains available. Only nearby positions are clustered in the
// whole-city marker layer; no contact patches are drawn until a case is selected.
export function collisionHotspots(cases, { overview = true, selectedId = null, center = null, radius = 120 } = {}) {
  const located = cases.filter(c => c.focus_position && (overview || !center || c.id === selectedId || Math.hypot(c.focus_position[0] - center[0], c.focus_position[1] - center[1]) <= radius * 1.5));
  if (!overview) return located.map(c => ({ id: c.id, caseId: c.id, caseIds: [c.id], kind: c.kind, position: c.focus_position, count: 1 }));
  const groups = new Map();
  for (const item of located) {
    const [x, y] = item.focus_position, key = `${Math.floor(x / 90)}:${Math.floor(y / 90)}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(item);
  }
  return [...groups.entries()].map(([key, items]) => {
    const anchor = items.find(c => c.id === selectedId) || items.find(c => c.default_visible) || items[0];
    return { id: `hotspot:${key}`, caseId: anchor.id, caseIds: items.map(c => c.id), kind: anchor.kind, position: anchor.focus_position, count: items.length };
  });
}

export function collisionFrameBounds(item, assets) {
  const ids = new Set(item?.asset_ids || []), found = assets.filter(a => ids.has(a.id) && validBounds(a.bounds));
  const boxes = found.map(a => a.bounds);
  if (!boxes.length && validBounds(item?.asset_bounds)) boxes.push(item.asset_bounds);
  if (!boxes.length && validBounds(item?.bounds)) boxes.push(item.bounds);
  if (!boxes.length && item?.focus_position) boxes.push([item.focus_position.map(v => v - 10), item.focus_position.map(v => v + 10)]);
  if (!boxes.length) return null;
  return [0, 1].map(side => [0, 1, 2].map(axis => Math[side ? 'max' : 'min'](...boxes.map(b => b[side][axis]))));
}

function disposeGroup(group) {
  const geometries = new Set(), materials = new Set();
  group.traverse(node => { if (node.geometry) geometries.add(node.geometry); for (const material of [].concat(node.material || [])) materials.add(material); });
  geometries.forEach(g => g.dispose()); materials.forEach(m => m.dispose()); group.clear();
}

export class CollisionLayers {
  constructor(scene, audit, assets) {
    this.scene = scene; this.cases = collisionCases(audit); this.byId = new Map(this.cases.map(c => [c.id, c])); this.assets = assets;
    this.group = new THREE.Group(); this.group.name = 'collision-review'; this.group.visible = false; scene.add(this.group);
    this.markers = new THREE.Group(); this.detail = new THREE.Group(); this.group.add(this.markers, this.detail);
    this.textures = new Map(); this.enabled = false; this.caseId = null; this.patchCount = 0; this.highlighted = [];
  }

  markerTexture(kind, count) {
    const label = count > 1 ? count > 9 ? '9+' : String(count) : '', key = `${kind}:${label}`;
    if (this.textures.has(key)) return this.textures.get(key);
    const canvas = document.createElement('canvas'); canvas.width = canvas.height = 64;
    const ctx = canvas.getContext('2d');
    ctx.beginPath(); ctx.arc(32, 32, 27, 0, Math.PI * 2); ctx.fillStyle = '#ffffff'; ctx.fill();
    ctx.beginPath(); ctx.arc(32, 32, 22, 0, Math.PI * 2); ctx.fillStyle = COLLISION_COLORS[kind]; ctx.fill();
    if (label) { ctx.fillStyle = '#fff'; ctx.font = '600 28px sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle'; ctx.fillText(label, 32, 33); }
    else { ctx.beginPath(); ctx.arc(32, 32, 6, 0, Math.PI * 2); ctx.fillStyle = '#fff'; ctx.fill(); }
    const texture = new THREE.CanvasTexture(canvas); texture.colorSpace = THREE.SRGBColorSpace; this.textures.set(key, texture); return texture;
  }

  configure({ enabled = false, kind = 'all', priorityOnly = true, selectedId = null, overview = true, center = null, radius = 120 } = {}) {
    this.enabled = enabled; this.group.visible = enabled;
    const nextId = enabled ? selectedId : null;
    const visible = enabled ? filterCollisionCases(this.cases, { kind, priorityOnly }) : [];
    const hotspots = collisionHotspots(visible, { overview, selectedId: nextId, center, radius });
    const markerKey = JSON.stringify(hotspots.map(h => [h.id, h.caseId, h.count]));
    if (markerKey !== this.markerKey) {
      disposeGroup(this.markers); this.markerKey = markerKey;
      for (const h of hotspots) {
        const sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.markerTexture(h.kind, h.count), depthTest: false, depthWrite: false, transparent: true }));
        sprite.position.fromArray(h.position); sprite.position.z += 1.5; sprite.renderOrder = 35;
        sprite.userData = { collisionId: h.caseId, collisionIds: h.caseIds, selected: h.caseIds.includes(nextId) }; this.markers.add(sprite);
      }
    } else for (const sprite of this.markers.children) sprite.userData.selected = sprite.userData.collisionIds.includes(nextId);
    if (nextId !== this.caseId) { this.caseId = nextId; this.updateDetail(); }
    this.visibleCount = visible.length;
  }

  updateDetail() {
    disposeGroup(this.detail); this.patchCount = 0; this.highlighted = [];
    const item = this.byId.get(this.caseId); if (!item) return;
    for (const [index, id] of item.asset_ids.entries()) {
      const asset = this.assets.get(id); if (!asset || !validBounds(asset.bounds)) continue;
      const box = new THREE.Box3(new THREE.Vector3(...asset.bounds[0]), new THREE.Vector3(...asset.bounds[1]));
      const helper = new THREE.Box3Helper(box.expandByScalar(.25), RELATED_COLORS[index % RELATED_COLORS.length]);
      helper.material.depthTest = false; helper.material.transparent = true; helper.material.opacity = .9; helper.renderOrder = 24; this.detail.add(helper); this.highlighted.push(id);
    }
    const patches = item.patches.slice(0, 1000);
    if (patches.length) {
      const mesh = new THREE.InstancedMesh(new THREE.BoxGeometry(1, 1, 1),
        new THREE.MeshBasicMaterial({ color: COLLISION_COLORS[item.kind], transparent: true, opacity: .42, depthTest: false, depthWrite: false }), patches.length);
      const matrix = new THREE.Matrix4(), position = new THREE.Vector3(), scale = new THREE.Vector3(), rotation = new THREE.Quaternion();
      patches.forEach((p, i) => {
        position.fromArray(p.center); scale.fromArray(p.size.map(v => Math.max(.035, v)));
        matrix.compose(position, rotation, scale); mesh.setMatrixAt(i, matrix);
      });
      mesh.instanceMatrix.needsUpdate = true; mesh.frustumCulled = false; mesh.renderOrder = 27; this.detail.add(mesh); this.patchCount = patches.length;
    } else if (item.focus_position) {
      const ring = new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(Array.from({ length: 48 }, (_, i) => {
        const angle = i / 48 * Math.PI * 2; return new THREE.Vector3(item.focus_position[0] + 3 * Math.cos(angle), item.focus_position[1] + 3 * Math.sin(angle), item.focus_position[2]);
      })), new THREE.LineBasicMaterial({ color: COLLISION_COLORS[item.kind], depthTest: false }));
      ring.renderOrder = 27; this.detail.add(ring);
    }
  }

  update(camera, height) {
    if (!this.enabled) return;
    const scaleFactor = 2 * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)) / Math.max(1, height);
    for (const marker of this.markers.children) {
      const size = camera.position.distanceTo(marker.position) * scaleFactor * (marker.userData.selected ? 27 : 19);
      marker.scale.setScalar(size);
    }
  }

  pick(ray) { return this.enabled ? ray.intersectObjects(this.markers.children, false)[0]?.object.userData.collisionId : null; }
  stats() { return { enabled: this.enabled, selectedId: this.caseId, markerCount: this.enabled ? this.markers.children.length : 0, caseCount: this.enabled ? this.visibleCount : 0, patchCount: this.enabled ? this.patchCount : 0, highlighted: this.enabled ? this.highlighted : [] }; }
  dispose() { disposeGroup(this.group); for (const texture of this.textures.values()) texture.dispose(); this.textures.clear(); this.scene.remove(this.group); }
}
