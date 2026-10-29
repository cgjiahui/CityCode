import test from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import { mapGeometry, neuralWorldMatrix, BackgroundLayers } from '../src/background.js';

test('historical neural XYZ maps height to UE Z without mirroring world Y', () => {
  assert.deepEqual(new THREE.Vector3(-552, 12, -832).applyMatrix4(neuralWorldMatrix()).toArray(), [-552, -832, 12]);
});

test('neural background replaces map and base while elevated overlay stays independent', () => {
  const layer = new BackgroundLayers(new THREE.Scene(), null, { layers: { elevated: { faces: 1 } } }, () => {}, () => {});
  layer.map = new THREE.Object3D(); layer.base = new THREE.Object3D();
  layer.elevated = new THREE.Object3D(); layer.neural = new THREE.Object3D();
  layer.settings = { view:'city', backgroundMode:'pixal3d', backgroundMap:true, backgroundGround:true, backgroundElevated:true };
  layer.update();
  assert.equal(layer.neural.visible, true); assert.equal(layer.map.visible, false);
  assert.equal(layer.base.visible, false); assert.equal(layer.elevated.visible, true);
  layer.settings.backgroundMode = 'surface'; layer.update();
  assert.equal(layer.neural.visible, false); assert.equal(layer.map.visible, true); assert.equal(layer.base.visible, true);
  layer.settings.view = 'single'; layer.update();
  assert.ok([layer.map,layer.base,layer.elevated,layer.neural].every(o=>!o.visible));
  layer.dispose();
});

test('image top-left maps to UE xmin/ymin without flipping the north/south extent', () => {
  const geometry = mapGeometry([-1032,-832,952.05,832.1], -1.2);
  const positions = geometry.attributes.position, uv = geometry.attributes.uv;
  assert.equal(positions.getX(0), -1032); assert.equal(positions.getY(0), -832);
  assert.deepEqual([uv.getX(0),uv.getY(0)], [0,1]);
  assert.deepEqual([uv.getX(2),uv.getY(2)], [1,0]);
  assert.ok(Math.abs(positions.getY(2)-832.1)<.001);
  assert.ok(geometry.attributes.normal.getZ(0)>.999);
  geometry.dispose();
});

 test('coastal solid replaces old backgrounds; water can be hidden without hiding land', () => {
 const layer = new BackgroundLayers(new THREE.Scene(), null, { layers: { elevated: { faces: 1 } } }, () => {}, () => {});
 for (const role of ['coastal','water','neural','map','base','elevated']) layer[role] = new THREE.Object3D();
 layer.settings = {view:'city',backgroundMode:'coastal',backgroundWater:true,backgroundMap:true,backgroundGround:true};layer.update();
 assert.equal(layer.coastal.visible,true);assert.equal(layer.water.visible,true);
 assert.ok([layer.neural,layer.map,layer.base].every(o=>!o.visible));
 layer.settings.backgroundWater=false;layer.update();assert.equal(layer.water.visible,false);assert.equal(layer.coastal.visible,true);
 layer.settings.view='single';layer.update();assert.equal(layer.coastal.visible,false);
 layer.dispose();
 });

test('RGB land solid replaces the flat map and measured ground, with separate water visibility', () => {
  const layer = new BackgroundLayers(new THREE.Scene(), null,
    {files:{map_solid:{}},layers:{elevated:{faces:1}}}, () => {}, () => {});
  for (const role of ['map','base','elevated','neural','coastal','water','mapSolid','mapWater']) layer[role] = new THREE.Object3D();
  layer.settings = {view:'city',backgroundMode:'surface',backgroundMap:true,mapSolid:true,
    backgroundGround:true,backgroundElevated:true,backgroundWater:true};
  layer.update();
  assert.equal(layer.mapSolid.visible,true); assert.equal(layer.map.visible,false); assert.equal(layer.base.visible,false);
  assert.equal(layer.stats().mapWaterVisible,true); assert.equal(layer.elevated.visible,true);
  layer.settings.backgroundWater=false; layer.update();
  assert.equal(layer.mapSolid.visible,true); assert.equal(layer.stats().mapWaterVisible,false);
  layer.settings.backgroundMap=false; layer.update();
  assert.equal(layer.mapSolid.visible,false); assert.equal(layer.base.visible,true);
  assert.equal(layer.stats().mapWaterVisible,false);
  layer.settings.backgroundMap=true; layer.settings.mapSolid=false; layer.update();
  assert.equal(layer.map.visible,true); assert.equal(layer.base.visible,true); assert.equal(layer.mapSolid.visible,false);
  layer.settings.mapSolid=true; layer.settings.backgroundMode='coastal'; layer.update();
  assert.equal(layer.coastal.visible,true); assert.equal(layer.mapSolid.visible,false);
  layer.settings.backgroundMode='surface'; layer.settings.view='single'; layer.update();
  assert.ok([layer.map,layer.mapSolid,layer.base,layer.elevated,layer.neural,layer.coastal].every(object=>!object.visible));
  layer.dispose();
});

test('RGB land solid loads without old flat or ground meshes and gracefully supports older bundles', () => {
  const layer = new BackgroundLayers(new THREE.Scene(), null,
    {files:{map_solid:{}},layers:{elevated:{faces:1}}}, () => {}, () => {});
  const requests=[]; layer.load=role=>requests.push(role);
  const settings={view:'city',backgroundMode:'surface',backgroundMap:true,mapSolid:true,backgroundGround:true};
  layer.configure(settings);
  assert.deepEqual(requests,['map_solid']);
  requests.length=0; delete layer.data.files.map_solid;
  layer.configure(settings);
  assert.deepEqual(requests,['map','surfaces']);
  layer.dispose();
});
