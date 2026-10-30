import test from 'node:test';
import assert from 'node:assert/strict';
import {AssetStore} from '../src/assets.js';
const tick=()=>new Promise(resolve=>setImmediate(resolve));
function install(t,fetch,cache=null){
  const beforeFetch=globalThis.fetch,beforeCaches=globalThis.caches;
  globalThis.fetch=fetch;globalThis.caches=cache?{open:async()=>cache}:undefined;
  t.after(()=>{globalThis.fetch=beforeFetch;globalThis.caches=beforeCaches;});
}

test('saver transfers share requests, report bytes, and avoid cloning or writing large responses',async t=>{
  let requests=0,clones=0,writes=0;
  install(t,async()=>{requests++;const r=new Response(Uint8Array.from([1,2,3,4]));const clone=r.clone.bind(r);r.clone=()=>{clones++;return clone();};return r;},{match:async()=>null,put:async()=>{writes++;}});
  const store=new AssetStore(null,{concurrency:2,persist:false});
  const [a,b]=await Promise.all([store.get('asset',{bytes:4,core:true}),store.get('asset',{bytes:4,core:true})]);
  assert.deepEqual([...new Uint8Array(a)],[1,2,3,4]);assert.equal(a,b);assert.equal(requests,1);assert.equal(clones,0);assert.equal(writes,0);assert.equal(store.stats.coreBytes,4);
});

test('truncated and oversized downloads fail before being accepted',async t=>{
  install(t,async()=>new Response(Uint8Array.from([1,2,3])));
  const store=new AssetStore(null,{persist:false});
  await assert.rejects(store.get('short',{bytes:4}),/incomplete/);
  await assert.rejects(store.get('long',{bytes:2}),/unexpected size/);
  assert.equal(store.inflight.size,0);
});

test('saver retains the download concurrency limit across queued resources',async t=>{
  let active=0,peak=0;const releases=[];
  install(t,async()=>{active++;peak=Math.max(peak,active);return new Response(new ReadableStream({start(controller){releases.push(()=>{active--;controller.enqueue(Uint8Array.from([1,2]));controller.close();});}}));});
  const store=new AssetStore(null,{concurrency:2,persist:false});
  const pending=Promise.all(['a','b','c','d'].map(u=>store.get(u,{bytes:2})));
  await tick();assert.equal(active,2);releases.splice(0).forEach(f=>f());await tick();assert.equal(active,2);releases.splice(0).forEach(f=>f());await pending;
  assert.equal(peak,2);assert.equal(store.stats.networkBytes,8);
});

test('the default mode still persists responses and can read immutable cached bytes',async t=>{
  let requests=0;const cached=new Map();
  install(t,async()=>{requests++;return new Response(Uint8Array.from([7,8]));},{match:async u=>cached.get(u)?.clone(),put:async(u,r)=>cached.set(u,r),delete:async u=>cached.delete(u)});
  const first=new AssetStore();await first.get('immutable',{bytes:2});await Promise.all(first.writes);
  const second=new AssetStore();const data=await second.get('immutable',{bytes:2});assert.deepEqual([...new Uint8Array(data)],[7,8]);assert.equal(requests,1);assert.equal(second.stats.cacheHits,1);
});
