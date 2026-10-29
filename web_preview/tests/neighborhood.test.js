import test from 'node:test';
import assert from 'node:assert/strict';
import { Matrix4, Vector3 } from 'three';
import { neighborhood, wholeCityPlan, MODEL_LIMIT } from '../src/scene.js';
import { comparisonSettings, initialSceneSettings, completeCitySettings, massingSourceFor } from '../src/massing.js';

const asset = (number, x, extra = {}) => ({ id: `${number}`, group: [`${number}`], center: [x, 0, 0], lods: { '0.1M': { bytes: 4*1024*1024 } }, ...extra });
test('connected components stay together even outside the requested neighborhood', () => {
  const a = asset(553, 0, { group: ['553', '570'] }), b = asset(570, 200);
  const result = neighborhood([a, b, asset(9, 70)], a, 40, true);
  assert.deepEqual(result.assets.map(x => x.id), ['553', '570']);
});
test('nearest real models are bounded; selected building has priority', () => {
  const values = Array.from({length: 50}, (_, n) => asset(n, n*2));
  const result = neighborhood(values, values[49], 200);
  assert.equal(result.assets.length, MODEL_LIMIT);
  assert.equal(result.assets[0].id, '49'); assert.equal(result.limited, true);
});
test('optional source surfaces do not appear until explicitly enabled', () => {
  const values = [asset(574, 0), asset(578, 1, { surface: true, enabled: false }), asset(22, 2, { surface: true, enabled: true })];
  assert.deepEqual(neighborhood(values, values[0], 100).assets.map(x => x.id), ['574', '22']);
  assert.deepEqual(neighborhood(values, values[0], 100, false, true).assets.map(x => x.id), ['574', '22', '578']);
});
test('single-building mode excludes both accepted and reference observation surfaces', () => {
  const values = [asset(571, 0), asset(573, 1, { surface: true, enabled: false }), asset(323, 2, { surface: true, enabled: true })];
  assert.deepEqual(neighborhood(values, values[0], 100, true, true).assets.map(x => x.id), ['571']);
});
test('accepted attachments and raw references have independent switches', () => {
  const values = [asset(1, 0), asset(2, 1, {surface:true,enabled:false}), asset(3, 2, {surface:true,enabled:true})];
  assert.deepEqual(neighborhood(values, values[0], 100, false, false, MODEL_LIMIT, false).assets.map(x=>x.id), ['1']);
  assert.deepEqual(neighborhood(values, values[0], 100, false, true, MODEL_LIMIT, false).assets.map(x=>x.id), ['1','2']);
  assert.deepEqual(neighborhood(values, values[0], 100, false, false, MODEL_LIMIT, true).assets.map(x=>x.id), ['1','3']);
});
test('oversized neighbor cannot exhaust the remaining download budget', () => {
  const values = [asset(1, 0), asset(2, 1, {lods: {'0.1M': {bytes: 150*1024*1024}}}), asset(3, 2)];
  assert.deepEqual(neighborhood(values, values[0], 100).assets.map(x=>x.id), ['1','3']);
});
test('binding row matrices map local glTF Y to world Z exactly', () => {
  const matrix = [[2,0,0,10],[0,0,-3,20],[0,4,0,.48],[0,0,0,1]];
  const result = new Vector3(1,2,3).applyMatrix4(new Matrix4().set(...matrix.flat()));
  assert.deepEqual(result.toArray(), [12,11,8.48]);
});
test('whole-city buildings are never restricted to the local neighborhood cap', () => {
  const values=Array.from({length:542},(_,i)=>asset(i,i*100,{lods:{preview:{bytes:100},'0.1M':{bytes:200}}}));
  const plan=wholeCityPlan(values,{accepted:false});
  assert.equal(plan.size,542);assert.equal(plan.get('541'),'preview');
});
test('whole-city original geometry uses every original low LOD without substituting preview geometry', () => {
  const values=Array.from({length:542},(_,i)=>asset(i,i*100,{lods:{preview:{bytes:100},'0.1M':{bytes:200}}}));
  const plan=wholeCityPlan(values,{quality:'0.1M'});
  assert.equal(plan.size,542);assert.ok([...plan.values()].every(lod=>lod==='0.1M'));
});
test('whole-city fine geometry selects every available0.25M building while observations keep their own LOD', () => {
  const values=[asset('building',0,{lods:{preview:{bytes:100},'0.1M':{bytes:200},'0.25M':{bytes:400}}}),
    asset('older',1),asset('observed',2,{surface:true,enabled:false})];
  const plan=wholeCityPlan(values,{quality:'0.25M',references:true});
  assert.equal(plan.get('building'),'0.25M');
  assert.equal(plan.get('older'),'0.1M');
  assert.equal(plan.get('observed'),'0.1M');
  assert.equal(initialSceneSettings('').cityQuality,'preview');
});
test('all observation references can load across the city without loading any building', () => {
  const values=[asset(1,0),...Array.from({length:12},(_,i)=>asset(`r${i}`,i*1000,{surface:true,enabled:false})),asset('accepted',20000,{surface:true,enabled:true})];
  const references=wholeCityPlan(values,{buildings:false,accepted:false,references:true});
  assert.equal(references.size,12);assert.ok(!references.has('1'));assert.ok(!references.has('accepted'));
  assert.equal(wholeCityPlan(values,{buildings:false,accepted:true,references:false}).size,1);
  assert.equal(wholeCityPlan(values,{buildings:false,accepted:false,references:false}).size,0);
});
test('inferred massing switches independently from buildings and raw observation references', () => {
  const values=[asset(1,0),asset('wing',1,{massing:true}),asset('raw',2,{surface:true,enabled:false})];
  assert.deepEqual([...wholeCityPlan(values,{massing:false}).keys()],['1']);
  assert.deepEqual([...wholeCityPlan(values,{massing:true}).keys()],['1','wing']);
  assert.deepEqual([...wholeCityPlan(values,{massing:false,references:true}).keys()],['1','raw']);
  assert.deepEqual([...wholeCityPlan(values,{buildings:false,massing:true,references:true}).keys()],['raw']);
});

test('a known parent remains in context ahead of unrelated nearby buildings', () => {
  const selected=asset('part',0,{massing:true,parent_source_id:'parent_source'});
  const parent=asset('parent',30,{sources:['parent_source']});
  const values=[selected,asset('unrelated',1),parent];
  assert.deepEqual(neighborhood(values,selected,100,false,false,2).assets.map(a=>a.id),['part','parent']);
});

test('source inspection groups all parts and comparison never enables raw reference fragments', () => {
  const settings=initialSceneSettings('?inspect=massing');
  assert.equal(settings.view,'neighborhood');assert.equal(settings.massingColors,true);
  const source={id:'source',part_ids:['a','b','c']};
  assert.equal(massingSourceFor({massing_sources:[source]},{id:'b',massing:true}),source);
  assert.equal(comparisonSettings(settings,'before').massing,false);
  const onlyParts=comparisonSettings({...settings,surfaces:true},'parts');
  assert.equal(onlyParts.single,true);assert.equal(onlyParts.surfaces,false);
  assert.equal(comparisonSettings(onlyParts,'after').view,'neighborhood');
});

test('complete city restores buildings, terrain and elevated surfaces after local inspection', () => {
  const previous = { ...initialSceneSettings('?inspect=massing&buildings=0'),
    cityQuality:'0.1M', cityPaused:true, single:true, massing:false, surfaces:true, collisionEnabled:true };
  const next = completeCitySettings(previous, {files:{coastal:{}}});
  assert.equal(next.view,'city'); assert.equal(next.single,false); assert.equal(next.cityPaused,false);
  assert.equal(next.cityBuildings,true); assert.equal(next.massing,true); assert.equal(next.cityQuality,'0.1M');
  assert.equal(next.backgroundMode,'coastal'); assert.equal(next.backgroundWater,true); assert.equal(next.backgroundElevated,true);
  assert.equal(next.elevatedStyle,'highlight'); assert.equal(next.surfaces,false); assert.equal(next.acceptedSurfaces,false);
  assert.equal(next.massingColors,false); assert.equal(next.sourceOutline,false); assert.equal(next.collisionEnabled,false);
  assert.equal(previous.cityBuildings,false);
  const surface = completeCitySettings(previous, {files:{}});
  assert.equal(surface.backgroundMode,'surface'); assert.equal(surface.backgroundGround,true);
  assert.equal(initialSceneSettings('?inspect=massing&elevated=1').backgroundElevated,true);
  assert.equal(initialSceneSettings('?elevated=0').backgroundElevated,false);
  assert.equal(initialSceneSettings('?background=surface&ground=1').backgroundGround,true);
  assert.equal(initialSceneSettings('?background=surface').mapSolid,true);
  assert.equal(initialSceneSettings('?map_solid=0&water=0').mapSolid,false);
  assert.equal(initialSceneSettings('?water=0').backgroundWater,false);
});
