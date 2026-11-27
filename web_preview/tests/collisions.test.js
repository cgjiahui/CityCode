import test from 'node:test';
import assert from 'node:assert/strict';
import { Matrix4, Scene, Texture, Vector3 } from 'three';
import { CollisionLayers, collisionCases, collisionFrameBounds, collisionHotspots, collisionStatus,
  filterCollisionCases, initialCollisionCase, initialCollisionSettings } from '../src/collisions.js';

const contact = (id, extra = {}) => ({ id, kind: 'building', asset_ids: ['A', 'B'], label: `Contact ${id}`,
  severity: 'minor', review_status: 'minor_contact', default_visible: false,
  focus_position: [10, 20, 5], bounds: [[9, 19, 4], [11, 21, 6]],
  patches: [{ center: [10, 20, 5], size: [.25, .5, 4] }], ...extra });
const audit = { default_case_ids: ['important'], cases: [contact('minor'), contact('important', { default_visible: true, review_status: 'reviewed_retained' }),
  contact('lane', { kind: 'vehicle' }), contact('ground', { kind: 'ground', default_visible: true })] };

test('inspection layer is off by default and absent audit data is harmless', () => {
  assert.equal(initialCollisionSettings('').collisionEnabled, false);
  assert.equal(initialCollisionSettings('?inspect=collisions').collisionEnabled, true);
  assert.deepEqual(collisionCases(undefined), []);
  assert.equal(initialCollisionCase([], 'missing'), null);
  const layer = new CollisionLayers(new Scene(), null, new Map());
  assert.equal(layer.group.visible, false); assert.equal(layer.stats().patchCount, 0);
  layer.dispose();
});

test('priority and category filters retain all records in the underlying audit', () => {
  const cases = collisionCases(audit);
  assert.deepEqual(filterCollisionCases(cases).map(c => c.id), ['important', 'ground']);
  assert.deepEqual(filterCollisionCases(cases, { kind: 'vehicle', priorityOnly: true }), []);
  assert.deepEqual(filterCollisionCases(cases, { kind: 'vehicle', priorityOnly: false }).map(c => c.id), ['lane']);
  assert.equal(filterCollisionCases(cases, { priorityOnly: false }).length, 4);
  assert.equal(initialCollisionCase(cases).id, 'important');
  assert.equal(initialCollisionCase(cases, '#minor').id, 'minor');
  assert.equal(audit.cases.length, 4);
});

test('retained contacts and background detail are described without severe-collision or floating claims', () => {
  assert.equal(collisionStatus(audit.cases[1]).label, 'Reviewed and retained');
  assert.equal(collisionStatus(audit.cases[0]).label, 'Minor contact');
  assert.equal(collisionStatus(audit.cases[2]).label, 'Lane-edge contact');
  assert.equal(collisionStatus(audit.cases[3]).label, 'Local terrain rise');
  assert.equal(collisionStatus(contact('boundary', { metrics: { classification: 'review_overlap' } })).label, 'Boundary contact');
  assert.equal(collisionStatus(contact('boundary-status', { review_status: 'boundary_review' })).label, 'Boundary contact');
});

test('whole-city hotspots aggregate neighbours while anchoring the chosen case at its real XYZ', () => {
  const cases = [contact('a', { focus_position: [1, 2, 3] }), contact('b', { focus_position: [40, 50, 60] }), contact('far', { focus_position: [500, 500, 7] })];
  const markers = collisionHotspots(cases, { selectedId: 'b' });
  assert.equal(markers.length, 2); assert.equal(markers[0].count, 2);
  assert.equal(markers[0].caseId, 'b'); assert.deepEqual(markers[0].position, [40, 50, 60]);
  assert.deepEqual(markers.flatMap(m => m.caseIds).sort(), ['a', 'b', 'far']);
  assert.equal(collisionHotspots(cases, { overview: false, center: [0, 0, 0], radius: 100 }).length, 2);
});

test('camera context includes both related buildings rather than fitting only a tiny wall patch', () => {
  const item = contact('pair');
  assert.deepEqual(collisionFrameBounds(item, [{ id: 'A', bounds: [[0, 0, 0], [30, 30, 100]] },
    { id: 'B', bounds: [[28, 0, 0], [60, 30, 80]] }]), [[0, 0, 0], [60, 30, 100]]);
  assert.deepEqual(collisionFrameBounds(item, []), item.bounds);
});

test('invalid patch vectors are skipped without swapping valid world axes', () => {
  const values = collisionCases({ cases: [contact('valid', { patches: [
    { center: [1, 2, 3], size: [.25, .5, 8] }, { center: [1, 2], size: [1, 1, 1] },
    { center: [1, 2, 3], size: [-1, 1, 1] }, { center: [1, NaN, 3], size: [1, 1, 1] }] })] });
  assert.deepEqual(values[0].patches, [{ center: [1, 2, 3], size: [.25, .5, 8] }]);
});

test('global markers do not draw patches until selection; closing clears detail and associated highlights', () => {
  const scene = new Scene(), assets = new Map([
    ['A', { bounds: [[0, 0, 0], [30, 30, 100]] }], ['B', { bounds: [[28, 0, 0], [60, 30, 80]] }],
  ]);
  const layer = new CollisionLayers(scene, audit, assets), texture = new Texture();
  // Canvas label rendering is browser QA; the 3D layer lifecycle needs no DOM.
  layer.markerTexture = () => texture; layer.textures.set('fixture', texture);
  layer.configure({ enabled: true, priorityOnly: false });
  assert.equal(layer.stats().patchCount, 0); assert.equal(layer.detail.children.length, 0);
  layer.configure({ enabled: true, priorityOnly: false, selectedId: 'important' });
  assert.equal(layer.stats().patchCount, 1); assert.deepEqual(layer.stats().highlighted, ['A', 'B']);
  const patches = layer.detail.children.find(c => c.isInstancedMesh), m = new Matrix4();
  patches.getMatrixAt(0, m);
  assert.deepEqual(new Vector3().setFromMatrixPosition(m).toArray(), [10, 20, 5]);
  assert.deepEqual(new Vector3().setFromMatrixScale(m).toArray(), [.25, .5, 4]);
  layer.configure({ enabled: false, selectedId: 'important' });
  assert.equal(layer.group.visible, false); assert.equal(layer.stats().markerCount, 0);
  assert.equal(layer.stats().patchCount, 0); assert.deepEqual(layer.stats().highlighted, []);
  layer.dispose(); assert.equal(scene.children.length, 0);
});
