import test from 'node:test';
import assert from 'node:assert/strict';
import {resolveSceneQuality} from '../src/quality.js';

test('phones and tablets are identified while all devices default to high quality',()=>{
  for(const device of [
    {userAgent:'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X)'},
    {userAgent:'Mozilla/5.0 (Linux; Android 16; Tablet)'},
    {userAgent:'Mozilla/5.0 (Macintosh; Intel Mac OS X)',platform:'MacIntel',maxTouchPoints:5},
    {userAgentData:{mobile:true}},
  ]){assert.equal(resolveSceneQuality('',device).mobileDevice,true);assert.equal(resolveSceneQuality('',device).saver,false);};
  assert.equal(resolveSceneQuality('',{userAgent:'Windows NT 10.0',maxTouchPoints:10}).saver,false);
  assert.equal(resolveSceneQuality('',{userAgent:'Macintosh',maxTouchPoints:0}).saver,false);
});

test('a landscape phone keeps its device identity and saver mode requires an explicit choice',()=>{
  const phone={userAgent:'iPhone',width:390,height:844};
  assert.equal(resolveSceneQuality('',phone).saver,false);
  assert.equal(resolveSceneQuality('?quality=mobile',phone).saver,true);
  assert.equal(resolveSceneQuality('',{...phone,width:844,height:390}).mobileDevice,true);
  assert.equal(resolveSceneQuality('?quality=mobile',{...phone,width:844,height:390}).saver,true);
  assert.equal(resolveSceneQuality('',{userAgent:'Windows',width:320,height:740}).saver,false);
});

test('explicit quality overrides work on either device and historical cache-busting values stay automatic',()=>{
  assert.equal(resolveSceneQuality('?quality=full',{userAgent:'iPhone'}).saver,false);
  assert.equal(resolveSceneQuality('?quality=mobile',{userAgent:'Windows'}).saver,true);
  assert.equal(resolveSceneQuality('?quality=full-20261029',{userAgent:'iPhone'}).saver,false);
  assert.equal(resolveSceneQuality('?quality=full',{userAgent:'iPhone'}).preference,'full');
});
