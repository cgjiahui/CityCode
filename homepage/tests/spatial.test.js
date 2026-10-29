import test from 'node:test';
import assert from 'node:assert/strict';
import {inGeometry,collisionIndex,slideMove,nearestWalkPoint,chooseDetails,projectedSpan} from '../src/spatial.js';
const polygon=points=>({type:'Polygon',coordinates:[points]});
const land=polygon([[-20,-20],[20,-20],[20,20],[-20,20],[-20,-20]]);
test('walking respects land holes and water boundaries',()=>{
  const lake={...land,coordinates:[...land.coordinates,[[-2,-2],[2,-2],[2,2],[-2,2],[-2,-2]]]};
  assert.equal(inGeometry(0,0,lake),false);assert.equal(inGeometry(10,10,lake),true);assert.equal(inGeometry(21,0,lake),false);
});
test('large movement cannot tunnel through a narrow building, but can slide along it',()=>{
  const buildings=[{bounds:[[0,-10,0],[.5,10,20]],footprint:polygon([[0,-10],[.5,-10],[.5,10],[0,10],[0,-10]])}];
  const blocked=collisionIndex(buildings,land),result=slideMove([-5,0,1.7],[12,4],blocked);
  assert.ok(result[0]<0);assert.ok(result[1]>3.8);assert.equal(blocked(0.2,0,1.7),true);
  assert.equal(blocked(.2,0,25),false);
});
test('street spawn chooses a free road node rather than a roof or water',()=>{
  const blocked=collisionIndex([{bounds:[[-1,-1,0],[1,1,10]],footprint:polygon([[-1,-1],[1,-1],[1,1],[-1,1]])}],land);
  assert.deepEqual(nearestWalkPoint([[0,0],[30,0],[4,0],[8,0]],[0,0],blocked),[4,0]);
});
test('detail demand excludes offscreen buildings and has bounded near/medium budgets',()=>{
  const instances=Array.from({length:80},(_,i)=>({id:String(i),bounds:[[i*4,0,0],[i*4+2,2,30]]}));
  const result=chooseDetails(instances,[0,0,2],a=>a.id!=='0',3,7);
  assert.equal(result.size,10);assert.equal(result.has('0'),false);assert.equal([...result.values()].filter(v=>v==='near').length,3);assert.equal(result.get('1'),'near');
});
test('high detail respects the total triangle budget and falls back to medium',()=>{
  const rows=Array.from({length:20},(_,i)=>({id:String(i),bounds:[[i*4,0,0],[i*4+2,2,30]]}));
  const chosen=chooseDetails(rows,[0,0,2],()=>true,4,16,{maxFaces:600000,faceCount:(_,level)=>level==='near'?240000:30000});
  assert.equal([...chosen.values()].filter(v=>v==='near').length,2);
  assert.equal([...chosen.values()].reduce((sum,v)=>sum+(v==='near'?240000:30000),0),600000);
});
test('a large visible tower upgrades even beyond the old 500m distance cutoff',()=>{
  const tower={id:'tower',bounds:[[950,-25,0],[1050,25,220]]};
  const small={id:'small',bounds:[[950,40,0],[970,60,15]]};
  const camera=[0,0,300],pixels=a=>projectedSpan(a.bounds,camera,Math.PI/4,900);
  assert.ok(pixels(tower)>200);assert.ok(pixels(small)<70);
  const selected=chooseDetails([tower,small],camera,()=>true,4,16,{projectedPixels:pixels,nearPixels:220});
  assert.equal(selected.has('tower'),true);assert.equal(selected.has('small'),false);
});
test('screen coverage orders upgrades and shrinking the viewport reduces detail demand',()=>{
  const rows=[{id:'low',bounds:[[100,-5,0],[110,5,10]]},{id:'high',bounds:[[500,-10,0],[530,10,200]]}];
  const camera=[0,0,40],select=height=>chooseDetails(rows,camera,()=>true,0,1,{projectedPixels:a=>projectedSpan(a.bounds,camera,Math.PI/4,height),minMediumPixels:70});
  assert.deepEqual([...select(900).keys()],['high']);assert.equal(select(100).size,0);
});
