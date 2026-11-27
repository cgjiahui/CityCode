import * as T from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {KTX2Loader} from 'three/addons/loaders/KTX2Loader.js';
import {resolveSceneQuality} from './quality.js';
import {MeshoptDecoder} from 'three/addons/libs/meshopt_decoder.module.js';
import {addProps,addBridges} from './props.js';
import {chooseDetails,collisionIndex,slideMove,nearestWalkPoint,projectedSpan} from './spatial.js';
import {AssetStore,TaskPool} from './assets.js';
import {NativeProps} from './nativeProps.js';
import {addAtmosphere,createMaterialEnvironment} from './atmosphere.js';

const configuredBase=import.meta.env.VITE_CITYCODE_ASSET_BASE_URL||'city/';
const base=new URL(configuredBase.endsWith('/')?configuredBase:configuredBase+'/',document.baseURI).href;
let resourceVersion='';
const resourceHashes=new Map();
const url=path=>{const version=resourceHashes.get(path)||resourceVersion;return base+path+(version?'?v='+version:'');};
const vec=a=>new T.Vector3(...a);
const box=b=>new T.Box3(vec(b[0]),vec(b[1]));
function dispose(root) {
  const geometries=new Set(),materials=new Set(),textures=new Set();
  root.traverse(o=>{if(o.geometry)geometries.add(o.geometry);for(const m of (Array.isArray(o.material)?o.material:[o.material]))if(m)materials.add(m);});
  for(const m of materials){for(const v of Object.values(m))if(v?.isTexture)textures.add(v);m.dispose();}
  geometries.forEach(g=>g.dispose());textures.forEach(t=>t.dispose());
}
function shapes(geometry) {
  if(geometry.type==='GeometryCollection')return geometry.geometries.flatMap(shapes);
  const polys=geometry.type==='Polygon'?[geometry.coordinates]:geometry.type==='MultiPolygon'?geometry.coordinates:[];
  return polys.map(rings=>{const s=new T.Shape(rings[0].map(p=>new T.Vector2(...p)));for(const ring of rings.slice(1))s.holes.push(new T.Path(ring.map(p=>new T.Vector2(...p))));return s;});
}
function mappedGround(bounds,texture,z,terrain,maps=null,textureBounds=bounds) {
  const w=bounds[2]-bounds[0],h=bounds[3]-bounds[1];
  const axes=terrain.groundAxes?.map((axis,i)=>[...new Set([bounds[i],...axis.filter(v=>v>bounds[i]&&v<bounds[i+2]),bounds[i+2]])]);
  const g=new T.PlaneGeometry(w,h,axes?axes[0].length-1:Math.ceil(w/8),axes?axes[1].length-1:Math.ceil(h/8));g.translate((bounds[0]+bounds[2])/2,(bounds[1]+bounds[3])/2,0);
  const p=g.attributes.position,uv=g.attributes.uv;
  if(axes){const [xs,ys]=axes;ys.reverse();for(let i=0;i<p.count;i++)p.setXY(i,xs[i%xs.length],ys[Math.floor(i/xs.length)]);}
  for(let i=0;i<p.count;i++){p.setZ(i,terrain.heightAt(p.getX(i),p.getY(i)));uv.setXY(i,(p.getX(i)-textureBounds[0])/(textureBounds[2]-textureBounds[0]),(p.getY(i)-textureBounds[1])/(textureBounds[3]-textureBounds[1]));}
  g.computeVertexNormals();
  texture.colorSpace=T.SRGBColorSpace;texture.flipY=false;
  // A slope-dependent offset can pull the ground through shallow GT ramps.
  // Overview/detail use identical grids and a 12mm lift, so one depth unit is enough.
  const common={map:texture,polygonOffset:z>0,polygonOffsetFactor:0,polygonOffsetUnits:-1};
  const mat=maps?new T.MeshStandardMaterial({...common,normalMap:maps.normal,normalMapType:T.ObjectSpaceNormalMap,roughnessMap:maps.orm,metalnessMap:maps.orm,roughness:1,metalness:1,envMap:terrain.materialEnvironment.texture,envMapIntensity:.35}):new T.MeshBasicMaterial(common);
  mat.onBeforeCompile=s=>{s.uniforms.cityLandMask={value:terrain.landMask};s.uniforms.cityMapBounds={value:new T.Vector4(...terrain.data.ground.bounds)};
    s.vertexShader='varying vec2 cityMapXY;\n'+s.vertexShader;s.vertexShader=s.vertexShader.replace('#include <begin_vertex>','#include <begin_vertex>\ncityMapXY=position.xy;');
    s.fragmentShader='varying vec2 cityMapXY; uniform sampler2D cityLandMask; uniform vec4 cityMapBounds;\n'+s.fragmentShader;
    s.fragmentShader=s.fragmentShader.replace('#include <alphatest_fragment>','#include <alphatest_fragment>\nif(texture2D(cityLandMask,(cityMapXY-cityMapBounds.xy)/(cityMapBounds.zw-cityMapBounds.xy)).r<0.5) discard;');
  };mat.customProgramCacheKey=()=> maps?'city-ground-pbr-world-normal-v1':'city-ground-coast-v1';
  const mesh=new T.Mesh(g,mat);mesh.position.z=z;return mesh;
}

export class CityViewer {
  constructor(host,onState,{quality=resolveSceneQuality(location.search,navigator)}={}) {
    this.quality=quality;
    this.host=host;this.onState=onState;this.disposed=false;this.dirty=true;this.walking=false;this.keys=new Set();this.listeners=[];
    this.pending=0;this.failed=new Set();this.far=new Map();this.fine=new Map();this.cache=new Map();this.groundTiles=new Map();this.wanted=new Map();this.requests=new Set();this.tileRequests=new Set();
    this.nativeProps=new NativeProps(this);this.gtDecks=new Map();this.gtNearRequests=new Set();this.gtWanted=new Set();
    this.textureDetails=new Map();this.textureActive=0;this.textureWanted=[];this.baseReadyAt=null;this.startedAt=performance.now();
    this.assets=new AssetStore(stats=>{if(this.disposed)return;const total=this.baseBytes;this.update({downloadPercent:total?Math.min(100,Math.floor(stats.coreBytes/total*100)):null});},{concurrency:this.quality.saver?2:6,persist:!this.quality.saver});
    this.parsePool=new TaskPool(this.quality.saver?1:2);this.districtActive=0;
    this.status={loaded:0,total:0,phase:'Opening the city',error:'',walking:false,mobileSaver:quality.saver,mobileDevice:quality.mobileDevice,qualityPreference:quality.preference};
    this.scene=new T.Scene();addAtmosphere(this.scene);
    this.camera=new T.PerspectiveCamera(48,1,.3,7000);this.camera.up.set(0,0,1);
    this.renderer=new T.WebGLRenderer({antialias:!this.quality.saver,powerPreference:this.quality.saver?'low-power':'high-performance'});
    this.renderer.outputColorSpace=T.SRGBColorSpace;
    // Lift facade midtones while rolling bright roofs and road markings into white.
    this.renderer.toneMapping=T.ReinhardToneMapping;
    this.renderer.toneMappingExposure=2.25;
    this.renderer.setPixelRatio(this.quality.saver?1:Math.min(devicePixelRatio,innerWidth<700?1.25:1.5));
    this.renderer.setClearColor('#dbe4e7');host.appendChild(this.renderer.domElement);
    const gl=this.renderer.getContext(),ext=gl.getExtension('WEBGL_debug_renderer_info');
    this.device=ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):gl.getParameter(gl.RENDERER);this.software=/swiftshader|llvmpipe|software/i.test(this.device);
    if(this.software)this.renderer.setPixelRatio(1);
    this.materialEnvironment=createMaterialEnvironment(this.renderer);
    this.controls=new OrbitControls(this.camera,this.renderer.domElement);this.controls.enableDamping=true;this.controls.dampingFactor=.1;
    this.controls.maxPolarAngle=Math.PI/2-.025;this.controls.minDistance=12;this.controls.maxDistance=3200;this.controls.screenSpacePanning=false;
    this.controls.addEventListener('change',()=>{this.dirty=true;});
    const workers=this.quality.saver?1:Math.min(4,Math.max(2,Math.floor((navigator.hardwareConcurrency||4)/2)));
    MeshoptDecoder.useWorkers?.(workers);
    this.ktx=new KTX2Loader().setTranscoderPath(new URL('basis/',document.baseURI).href).setWorkerLimit(workers).detectSupport(this.renderer);
    this.loader=new GLTFLoader().setMeshoptDecoder(MeshoptDecoder).setKTX2Loader(this.ktx);
    this.resize=new ResizeObserver(()=>{const w=host.clientWidth,h=host.clientHeight;this.renderer.setSize(w,h);this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.dirty=true;});this.resize.observe(host);
    this.listen(window,'keydown',e=>{if(['KeyW','KeyA','KeyS','KeyD','ArrowUp','ArrowDown','ArrowLeft','ArrowRight','ShiftLeft','ShiftRight'].includes(e.code)&&this.walking){e.preventDefault();this.keys.add(e.code);}if(e.code==='Escape'&&this.walking)this.setWalking(false);});
    this.listen(window,'keyup',e=>this.keys.delete(e.code));this.listen(window,'blur',()=>this.keys.clear());
    this.listen(document,'visibilitychange',()=>{this.keys.clear();this.lastTime=performance.now();this.lastDetails=0;this.dirty=true;if(!document.hidden)this.pumpDistricts();});
    const canvas=this.renderer.domElement;canvas.tabIndex=0;canvas.setAttribute('aria-label','3D city. Drag to orbit the view.');
    this.listen(canvas,'pointerdown',e=>{if(this.walking){this.drag={id:e.pointerId,x:e.clientX,y:e.clientY};canvas.setPointerCapture(e.pointerId);}canvas.focus({preventScroll:true});});
    this.listen(canvas,'pointermove',e=>{if(this.walking&&this.drag?.id===e.pointerId){this.yaw-=(e.clientX-this.drag.x)*.004;this.pitch=Math.max(-1.35,Math.min(1.35,this.pitch-(e.clientY-this.drag.y)*.004));this.drag={id:e.pointerId,x:e.clientX,y:e.clientY};this.look();}});
    this.listen(canvas,'pointerup',()=>this.drag=null);this.listen(canvas,'pointercancel',()=>this.drag=null);
    this.listen(canvas,'webglcontextlost',e=>{e.preventDefault();this.update({error:'Graphics resources were released. Reload the page to continue.'});});
    this.lastTime=performance.now();this.lastDetails=0;this.renderTimes=[];this.frames=0;this.loop=this.loop.bind(this);this.raf=requestAnimationFrame(this.loop);
  }
  listen(el,event,fn){el.addEventListener(event,fn);this.listeners.push(()=>el.removeEventListener(event,fn));}
  update(part){this.status={...this.status,...part};this.onState?.(this.status);}
  async init() {
    try {
      const res=await fetch(url('scene.json'));if(!res.ok)throw Error(`Unable to read the scene manifest（${res.status}）`);this.data=await res.json();resourceVersion=this.data.revision||'';if(this.disposed)return;
      resourceHashes.clear();
      for(const d of this.data.districts)resourceHashes.set('districts/'+d.path,d.sha256);
      for(const m of Object.values(this.data.models))for(const r of Object.values(m.levels))resourceHashes.set('models/'+r.path,r.sha256);
      const g=this.data.ground;for(const path of [g.overview,g.land_mask,...g.tiles.map(t=>t.path)])resourceHashes.set(path,g.source_image_sha256);
      for(const r of [g.materials,...g.tiles.map(t=>t.materials)].filter(Boolean))for(const t of Object.values(r))resourceHashes.set(t.path,t.sha256);
      for(const r of Object.values(this.data.native_props?.assets||{}))resourceHashes.set(r.path,r.sha256);
      this.externalTextures=new Map((this.data.external_textures||[]).map(r=>[r.path,r]));
      for(const r of this.externalTextures.values())resourceHashes.set(r.path,r.sha256);
      // Externalized textures share the same bounded transfer queue and durable
      // content cache as geometry. Embedded GLB images retain the normal loader.
      const originalTextureLoad=this.ktx.load.bind(this.ktx);
      this.ktx.load=(value,onLoad,onProgress,onError)=>{
        const path=value.startsWith(base)?value.slice(base.length).split(/[?#]/)[0]:null;
        const record=this.externalTextures.get(path);
        if(!record)return originalTextureLoad(value,onLoad,onProgress,onError);
        const target=url(path),manager=this.ktx.manager;manager.itemStart(target);
        this.assets.get(target,{bytes:record.bytes}).then(buffer=>new Promise((resolve,reject)=>this.ktx.parse(buffer,resolve,reject)))
          .then(texture=>{onLoad?.(texture);manager.itemEnd(target);})
          .catch(error=>{onError?.(error);manager.itemError(target);manager.itemEnd(target);});
      };
      resourceHashes.set(g.height.path,g.height.source_sha256);
      if(this.data.elevated)resourceHashes.set(this.data.elevated.path,this.data.elevated.sha256);
      for(const d of this.data.elevated?.decks||[])for(const r of Object.values(d.levels))resourceHashes.set(r.path,r.sha256);
      for(const r of [...this.data.districts,this.data.elevated].filter(Boolean))for(const t of r.texture_details||[])resourceHashes.set(t.path,t.sha256);
      this.baseBytes=this.data.streaming?.base_bytes??this.data.districts.reduce((n,r)=>n+r.bytes,0)+(this.data.elevated?.bytes||0)+(this.data.elevated?.decks||[]).reduce((n,d)=>n+d.levels.far.bytes,0);
      this.update({instances:this.data.stats.instances,elevatedReady:!this.data.elevated,webPreview:['web-preview','portable-high-quality-preview'].includes(this.data.presentation?.quality)});
      this.instances=new Map(this.data.instances.map(a=>[a.id,{...a,box:box(a.bounds)}]));
      this.blocked=collisionIndex(this.data.instances,this.data.ground.land);this.home();
      await this.ground();if(this.disposed)return;
      this.walkable=[];this.elevatedReady=!this.data.elevated;
      if(this.data.elevated)this.elevatedTask=this.loadElevated();else this.walkable=addBridges(this.scene,this.data);
      this.propLayer=addProps(this.scene,this.data,this.heightAt);this.propDraws=this.propLayer.draws;
      this.districtQueue=[...this.data.districts].sort((a,b)=>vec(a.bounds.min).distanceTo(this.camera.position)-vec(b.bounds.min).distanceTo(this.camera.position));
      this.update({total:this.data.districts.length,phase:'Loading the city'});
      this.pumpDistricts();this.dirty=true;
    } catch(e){if(!this.disposed)this.update({coreError:true,error:e.message,phase:'Loading incomplete'});}
  }
  async loadGLTF(path,record={},core=false){
    const buffer=await this.assets.get(url(path),{bytes:record.bytes,core,priority:core?10:0});
    return this.parsePool.run(()=>this.loader.parseAsync(buffer,base),core?10:0);
  }
  async loadKTX(path,bytes,priority=0){
    const buffer=await this.assets.get(url(path),{bytes,priority});
    return new Promise((resolve,reject)=>this.ktx.parse(buffer,resolve,reject));
  }
  baseReady(){return this.far.size===this.data?.districts.length&&this.elevatedReady;}
  registerTextureDetails(gltf,record,kind){
    if(this.quality.saver||!record.texture_details?.length)return;
    const references=new Map(record.texture_details.map(r=>[r.texture_index,r]));
    const bounds=new T.Box3(vec(record.bounds.min),vec(record.bounds.max));
    const seen=new Set();
    gltf.scene.traverse(o=>{if(!o.isMesh)return;for(const material of Array.isArray(o.material)?o.material:[o.material]){
      if(seen.has(material))continue;seen.add(material);
      for(const slot of ['map','normalMap','roughnessMap','metalnessMap','aoMap','emissiveMap']){
        const old=material[slot],index=old&&gltf.parser.associations.get(old)?.textures,r=references.get(index);if(!r)continue;
        const key=record.path+'/'+index;
        if(!this.textureDetails.has(key))this.textureDetails.set(key,{record:r,bounds,kind,bindings:[],done:false,loading:false});
        this.textureDetails.get(key).bindings.push({material,slot,old});
      }
    }});
  }
  selectTextureDetails(frustum){
    if(!this.baseReady()||this.quality.saver)return;
    if(this.baseReadyAt===null)this.baseReadyAt=performance.now()-this.startedAt;
    const entries=[...this.textureDetails.values()];
    for(const entry of entries){
      const visible=frustum.intersectsBox(entry.bounds),near=entry.bounds.distanceToPoint(this.camera.position)<400;
      entry.priority=visible&&(entry.kind==='district'||near)?2:entry.kind==='district'?-1:-2;
    }
    this.textureWanted=entries.filter(e=>!e.done&&!e.failed)
      .sort((a,b)=>b.priority-a.priority||a.bounds.distanceToPoint(this.camera.position)-b.bounds.distanceToPoint(this.camera.position));
    this.update({refining:this.textureWanted.length>0,textureDone:entries.filter(e=>e.done).length,textureTotal:entries.length,textureFailed:entries.filter(e=>e.failed).length});
    this.pumpTextureDetails();
  }
  pumpTextureDetails(){
    if(this.disposed||document.hidden||this.quality.saver)return;
    for(const entry of this.textureWanted){
      if(this.textureActive>=2)break;
      if(entry.done||entry.loading||entry.failed)continue;
      // Visible textures first; offscreen textures use one idle slot, leaving
      // headroom for buildings and road detail while the user moves around.
      if(entry.priority<0&&(this.textureActive||this.pending||this.gtNearRequests.size||this.nativeProps.pending.size))continue;
      this.loadTextureDetail(entry);
    }
  }
  async loadTextureDetail(entry){
    entry.loading=true;this.textureActive++;
    try{
      const high=await this.loadKTX(entry.record.path,entry.record.bytes,entry.priority);
      if(this.disposed){high.dispose();return;}
      const old=entry.bindings[0].old;
      for(const key of ['colorSpace','flipY','wrapS','wrapT','minFilter','magFilter','anisotropy','channel','matrixAutoUpdate'])high[key]=old[key];
      high.offset.copy(old.offset);high.repeat.copy(old.repeat);high.center.copy(old.center);high.rotation=old.rotation;high.matrix.copy(old.matrix);
      for(const {material,slot} of entry.bindings){material[slot]=high;material.needsUpdate=true;}
      new Set(entry.bindings.map(b=>b.old)).forEach(t=>t.dispose());entry.bindings.length=0;entry.done=true;this.dirty=true;
    }catch(e){entry.failed=true;console.warn('HD texture unavailable; keeping base quality',e.message);}
    finally{entry.loading=false;this.textureActive--;this.pumpTextureDetails();}
  }
  elevatedMaterials(root,walkable=false) {
      root.traverse(o=>{if(!o.isMesh)return;
        for(const m of Array.isArray(o.material)?o.material:[o.material]){
          for(const t of [m.map,m.normalMap,m.roughnessMap,m.metalnessMap,m.aoMap,m.emissiveMap])if(t)t.anisotropy=Math.min(16,this.renderer.capabilities.getMaxAnisotropy());
          // Native mesh decals sit very close to the road. Keep their original
          // position while resolving coplanar depth precision at city distances.
          if(m.transparent||m.alphaTest>0){m.polygonOffset=true;m.polygonOffsetFactor=-1;m.polygonOffsetUnits=-1;}
        }
        if(walkable&&(o.userData.walkable||o.name.toLowerCase().includes('deck_visible')))this.walkable.push(o);
      });
  }
  async loadElevated() {
    try {
      const gltf=await this.loadGLTF(this.data.elevated.path,this.data.elevated,true);if(this.disposed){dispose(gltf.scene);return;}
      gltf.scene.name='SmallCity GT elevated roads';this.elevatedMaterials(gltf.scene,true);
      this.elevatedRoot=gltf.scene;this.scene.add(gltf.scene);gltf.scene.updateMatrixWorld(true);
      this.registerTextureDetails(gltf,this.data.elevated,'elevated');
      const queue=[...(this.data.elevated.decks||[])];
      await Promise.all(Array.from({length:this.quality.saver?1:3},async()=>{while(queue.length&&!this.disposed){
        const record=queue.shift(),tile=await this.loadGLTF(record.levels.far.path,record.levels.far,true);
        if(this.disposed){dispose(tile.scene);return;}
        this.elevatedMaterials(tile.scene,true);this.scene.add(tile.scene);tile.scene.updateMatrixWorld(true);
        this.gtDecks.set(record.id,{record,box:new T.Box3(vec(record.bounds.min),vec(record.bounds.max)),far:tile.scene,near:null});this.dirty=true;
      }}));
      if(this.disposed)return;
      if(!this.walkable.length)throw Error('Elevated road surfaces did not load completely');
      this.elevatedReady=true;this.dirty=true;this.update({phase:this.far.size===this.data.districts.length?'City ready':'Loading the city',elevatedReady:true});
    }catch(e){this.failed.add('elevated');this.update({coreError:true,error:'Elevated roads failed to load. Check your connection and reload.'});console.error(e);}
  }
  selectElevatedDetails(frustum) {
    if(this.quality.saver)return;
    const limit=this.software?2:this.host.clientWidth<700?3:6;
    const candidates=[...this.gtDecks.entries()].map(([id,e])=>({id,e,d:e.box.distanceToPoint(this.camera.position)}))
      .filter(({e,d})=>e.record.levels.near&&d<(e.near?320:250)&&frustum.intersectsBox(e.box)).sort((a,b)=>a.d-b.d).slice(0,limit);
    this.gtWanted=new Set(candidates.map(c=>c.id));
    for(const [id,e] of this.gtDecks)if(e.near&&!this.gtWanted.has(id)){e.far.visible=true;this.scene.remove(e.near);dispose(e.near);e.near=null;this.dirty=true;}
    for(const {id,e} of candidates){if(this.gtNearRequests.size>=2)break;if(!e.near&&!this.gtNearRequests.has(id)&&!this.failed.has('gt/'+id))this.loadElevatedDetail(id,e);}
  }
  async loadElevatedDetail(id,entry) {
    this.gtNearRequests.add(id);
    try{
      const tile=await this.loadGLTF(entry.record.levels.near.path,entry.record.levels.near);
      if(this.disposed||!this.gtWanted.has(id)){dispose(tile.scene);return;}
      this.elevatedMaterials(tile.scene);this.scene.add(tile.scene);entry.near=tile.scene;entry.far.visible=false;this.dirty=true;
    }catch(e){this.failed.add('gt/'+id);console.warn('HD road textures unavailable; keeping base quality',e.message);}
    finally{this.gtNearRequests.delete(id);}
  }
  async ground() {
    const g=this.data.ground;
    const [surface,mask,response]=await Promise.all([g.materials?this.loadGroundMaterials(g.materials):new T.TextureLoader().loadAsync(url(g.overview)).then(base_color=>({base_color})),new T.TextureLoader().loadAsync(url(g.land_mask)),fetch(url(g.height.path))]);
    const tex=surface.base_color;
    if(!response.ok)throw Error('Unable to read terrain heights');const heights=new Int16Array(await response.arrayBuffer()),meta=g.height;
    if(heights.length!==meta.width*meta.height)throw Error('Terrain height data is incomplete');
    this.heightAt=(x,y)=>{const u=Math.max(0,Math.min(meta.width-1.00001,(x-meta.origin[0])/meta.cell_m)),v=Math.max(0,Math.min(meta.height-1.00001,(y-meta.origin[1])/meta.cell_m)),i=Math.floor(u),j=Math.floor(v),fx=u-i,fy=v-j;return ((1-fy)*((1-fx)*heights[j*meta.width+i]+fx*heights[j*meta.width+i+1])+fy*((1-fx)*heights[(j+1)*meta.width+i]+fx*heights[(j+1)*meta.width+i+1]))/100;};
    // Use the same world-aligned grid in overview and detail tiles so their
    // surfaces coincide. Refine only strips around GT ramp contacts to 2m.
    if(meta.contacts?.length)this.groundAxes=[0,1].map(axis=>{
      const lo=g.bounds[axis],hi=g.bounds[axis+2],values=new Set([lo,hi]);
      for(let p=lo;p<hi;p+=8)values.add(p);
      for(const point of meta.contacts)for(let p=lo+Math.floor((point[axis]-lo-10)/2)*2;p<=point[axis]+10;p+=2)if(p>lo&&p<hi)values.add(p);
      return [...values].sort((a,b)=>a-b);
    });
    if(this.disposed){tex.dispose();mask.dispose();return;}this.landMask=mask;mask.flipY=false;mask.minFilter=T.LinearFilter;mask.generateMipmaps=false;
    tex.anisotropy=Math.min(8,this.renderer.capabilities.getMaxAnisotropy());this.scene.add(mappedGround(g.bounds,tex,0,this,g.materials?surface:null));
    const coast=new T.ExtrudeGeometry(shapes(g.land),{depth:8,bevelEnabled:false,steps:1});coast.translate(0,0,-8.03);
    const cp=coast.attributes.position;for(let i=0;i<cp.count;i++)if(cp.getZ(i)>-1)cp.setZ(i,this.heightAt(cp.getX(i),cp.getY(i))-.02);coast.computeVertexNormals();
    // Draw only the extruded coastline sides. A second coplanar cap produces
    // triangle-shaped z-fighting at the city overview distance.
    this.scene.add(new T.Mesh(coast,[new T.MeshBasicMaterial({visible:false}),new T.MeshStandardMaterial({color:'#aaa799',roughness:1})]));
    const water=new T.Mesh(new T.PlaneGeometry(14000,14000),new T.MeshStandardMaterial({color:'#7c98aa',roughness:.95}));water.position.z=g.water_z_m;this.scene.add(water);
  }
  home(){if(!this.data)return;if(this.walking)this.setWalking(false);const b=box(this.data.bounds),c=b.getCenter(new T.Vector3()),size=b.getSize(new T.Vector3());c.z=35;this.controls.target.copy(c);const d=Math.max(size.x,size.y);this.camera.position.copy(c).add(new T.Vector3(d*.62,-d*.8,d*.72));this.controls.update();this.homeDistance=this.camera.position.distanceTo(c);this.dirty=true;}
  async pumpDistricts() {
    if(this.disposed||document.hidden||this.status.coreError||!this.districtQueue)return;
    while(this.districtActive<(this.quality.saver?1:this.host.clientWidth<700?4:6)&&this.districtQueue.length)this.loadDistrict(this.districtQueue.shift());
  }
  async loadDistrict(record) {
    this.districtActive++;
    try {
      const gltf=await this.loadGLTF('districts/'+record.path,record,true);if(this.disposed){dispose(gltf.scene);return;}
      const count=record.instances.length,mask=new T.DataTexture(new Uint8Array(count).fill(255),count,1,T.RedFormat);mask.needsUpdate=true;
      const entry={root:gltf.scene,mask,record};
      gltf.scene.traverse(o=>{if(!o.isMesh)return;
        if(o.material.map)o.material.map.anisotropy=Math.min(8,this.renderer.capabilities.getMaxAnisotropy());
        if(!o.geometry.getAttribute('_building_slot'))throw Error('Building detail index is missing');
        o.material.onBeforeCompile=shader=>{
          shader.uniforms.cityMask={value:mask};shader.uniforms.cityMaskWidth={value:count};
          shader.vertexShader='attribute float _building_slot; uniform sampler2D cityMask; uniform float cityMaskWidth;\n'+shader.vertexShader;
          shader.vertexShader=shader.vertexShader.replace('#include <project_vertex>','#include <project_vertex>\nif (texture2D(cityMask, vec2((_building_slot+0.5)/cityMaskWidth,0.5)).r < 0.5) gl_Position=vec4(2.0,2.0,2.0,1.0);');
        };o.material.customProgramCacheKey=()=> 'city-district-mask-v1';
      });
      this.far.set(record.id,entry);record.instances.forEach((id,i)=>{this.instances.get(id).farSlot=[entry,i];if(this.fine.has(id))mask.image.data[i]=0;});mask.needsUpdate=true;
      this.scene.add(gltf.scene);this.dirty=true;
      this.registerTextureDetails(gltf,record,'district');
      this.update({loaded:this.far.size,phase:this.far.size===this.data.districts.length?(this.elevatedReady?'City ready':'Loading elevated roads'):'Loading the city'});
    } catch(e){this.failed.add(record.id);this.update({coreError:true,error:'Some districts failed to load. Check your connection and reload.'});console.error(e);}
    finally{this.districtActive--;this.pumpDistricts();}
  }
  mask(id,show){const slot=this.instances.get(id)?.farSlot;if(slot){slot[0].mask.image.data[slot[1]]=show?255:0;slot[0].mask.needsUpdate=true;}}
  removeFine(id){const entry=this.fine.get(id);if(!entry)return;this.mask(id,true);this.scene.remove(entry.root);entry.resource.refs--;entry.resource.used=performance.now();this.fine.delete(id);this.dirty=true;}
  async loadDetail(id,level) {
    const a=this.instances.get(id),key=a.mesh+'/'+level;this.pending++;this.requests.add(key);
    try {
      let resource=this.cache.get(key);
      if(!resource){const r=this.data.models[a.mesh].levels[level],gltf=await this.loadGLTF('models/'+r.path,r);if(this.disposed){dispose(gltf.scene);return;}
        gltf.scene.traverse(o=>{if(o.isMesh)for(const m of Array.isArray(o.material)?o.material:[o.material])if(m.map)m.map.anisotropy=Math.min(8,this.renderer.capabilities.getMaxAnisotropy());});
        resource={root:gltf.scene,refs:0,used:performance.now()};this.cache.set(key,resource);
      }
      // Re-evaluate current demand after an asynchronous load; never resurrect an old camera request.
      for(const [wantedId,wantedLevel] of this.wanted) {
        if(wantedLevel!==level||this.instances.get(wantedId).mesh!==a.mesh||this.fine.get(wantedId)?.key===key)continue;
        this.removeFine(wantedId);const root=new T.Group();root.matrixAutoUpdate=false;root.matrix.set(...this.instances.get(wantedId).matrix.flat());root.add(resource.root.clone(true));this.scene.add(root);resource.refs++;
        this.fine.set(wantedId,{root,key,resource,level});this.mask(wantedId,false);
      }
      this.dirty=true;this.trimCache();
    } catch(e){this.failed.add(key);console.warn('Near detail unavailable; keeping the city model',key,e.message);}
    finally{this.pending--;this.requests.delete(key);if(!this.disposed)this.pumpDetails();}
  }
  trimCache(){const unused=[...this.cache.entries()].filter(([,r])=>!r.refs).sort((a,b)=>a[1].used-b[1].used);while(unused.length>(this.quality.saver?1:6)){const [key,r]=unused.shift();dispose(r.root);this.cache.delete(key);}}
  pumpDetails(){if(this.disposed||document.hidden||!this.data)return;for(const [id,level] of this.wanted){if(this.pending>=(this.quality.saver?1:3))break;const key=this.instances.get(id).mesh+'/'+level;if(this.fine.get(id)?.key===key||this.requests.has(key)||this.failed.has(key))continue;this.loadDetail(id,level);}}
  detailSelection(){
    if(!this.data||!this.far.size)return;
    this.camera.updateMatrixWorld();const frustum=new T.Frustum().setFromProjectionMatrix(new T.Matrix4().multiplyMatrices(this.camera.projectionMatrix,this.camera.matrixWorldInverse));
    // Complete city coverage before optional single-building/highway details
    // compete for the same six HTTP connections and decoder workers.
    if(!this.baseReady())return;
    this.selectTextureDetails(frustum);
    this.selectElevatedDetails(frustum);if(!this.quality.saver)this.nativeProps.select(frustum);
    const compact=this.host.clientWidth<700,limited=compact||this.software;
    const camera=this.camera.position.toArray();
    const detailInstances=[...this.instances.values()].filter(a=>{const levels=this.data.models[a.mesh]?.levels;return levels?.medium&&levels?.near&&(!this.quality.saver||levels.medium.bytes<=4000000);});
    this.wanted=chooseDetails(detailInstances,camera,a=>frustum.intersectsBox(a.box),this.quality.saver?0:limited?2:8,this.quality.saver?4:limited?12:40,{maxFaces:this.quality.saver?600000:this.software?1300000:compact?2000000:6000000,faceCount:(a,level)=>this.data.models[a.mesh].levels[level].faces,projectedPixels:a=>projectedSpan(a.bounds,camera,this.camera.fov*Math.PI/180,this.host.clientHeight),minMediumPixels:70,nearPixels:220});
    // Portable medium/near tiers reference one file. Crossing the distance
    // threshold must not download or parse that same model a second time.
    for(const [id,level] of this.wanted){const levels=this.data.models[this.instances.get(id).mesh].levels;if(levels.medium.path===levels[level].path&&levels.medium.sha256===levels[level].sha256)this.wanted.set(id,'medium');}
    const overview=!this.walking&&this.camera.position.distanceTo(this.controls.target)>(this.homeDistance||2000)*.8;
    if(overview!==this.status.overview)this.update({overview});
    for(const id of this.fine.keys())if(!this.wanted.has(id))this.removeFine(id);
    this.trimCache();this.pumpDetails();this.selectGround(frustum);
  }
  selectGround(frustum){
    if(this.quality.saver)return;
    const candidates=this.data.ground.tiles.map(t=>({t,d:Math.hypot(this.camera.position.x-(t.bounds[0]+t.bounds[2])/2,this.camera.position.y-(t.bounds[1]+t.bounds[3])/2)}))
      .filter(({t,d})=>this.camera.position.z<550&&d<550&&frustum.intersectsBox(new T.Box3(new T.Vector3(t.bounds[0],t.bounds[1],-1),new T.Vector3(t.bounds[2],t.bounds[3],3)))).sort((a,b)=>a.d-b.d).slice(0,this.data.ground.materials?(this.software||this.host.clientWidth<700?2:4):(this.host.clientWidth<700?4:9));
    this.groundWanted=new Set(candidates.map(v=>v.t.id));
    for(const [id,r] of this.groundTiles)if(!this.groundWanted.has(id)){this.scene.remove(r);dispose(r);this.groundTiles.delete(id);this.dirty=true;}
    for(const {t} of candidates){if(this.tileRequests.size>=2)break;if(this.groundTiles.has(t.id)||this.tileRequests.has(t.id)||this.failed.has(t.id))continue;this.loadGround(t);}
  }
  async loadGroundMaterials(records){
    const settled=await Promise.allSettled(Object.entries(records).map(async([key,r])=>[key,await this.loadKTX(r.path,r.bytes,1)]));
    if(settled.some(r=>r.status==='rejected')){for(const r of settled)if(r.status==='fulfilled')r.value[1].dispose();throw settled.find(r=>r.status==='rejected').reason;}
    const maps=Object.fromEntries(settled.map(r=>r.value));
    for(const [key,t] of Object.entries(maps)){t.flipY=false;t.colorSpace=key==='base_color'?T.SRGBColorSpace:T.NoColorSpace;t.anisotropy=Math.min(16,this.renderer.capabilities.getMaxAnisotropy());}
    return maps;
  }
  async loadGround(tile){this.tileRequests.add(tile.id);try{
    const maps=tile.materials?await this.loadGroundMaterials(tile.materials):{base_color:await this.ktx.loadAsync(url(tile.path))};
    if(this.disposed||!this.groundWanted.has(tile.id)){Object.values(maps).forEach(t=>t.dispose());return;}
    const tex=maps.base_color;tex.anisotropy=Math.min(16,this.renderer.capabilities.getMaxAnisotropy());
    const mesh=mappedGround(tile.bounds,tex,.012,this,tile.materials?maps:null,tile.texture_bounds||tile.bounds);this.groundTiles.set(tile.id,mesh);this.scene.add(mesh);this.dirty=true;
  }catch(e){this.failed.add(tile.id);console.warn('Ground detail temporarily unavailable',e.message);}finally{this.tileRequests.delete(tile.id);}}
  setWalking(enabled) {
    if(!this.data||enabled===this.walking)return;
    if(enabled){const safe=(x,y)=>[[-1,0],[1,0],[0,-1],[0,1],[0,0]].some(([dx,dy])=>this.blocked(x+dx,y+dy,this.heightAt(x,y)+1.7));const p=nearestWalkPoint(this.data.walk_points,this.controls.target.toArray(),safe);if(!p){this.update({error:'No accessible street starting point was found here.'});return;}
      this.orbitSaved={position:this.camera.position.clone(),target:this.controls.target.clone()};this.camera.position.set(p[0],p[1],this.heightAt(p[0],p[1])+1.7);
      const directions=Array.from({length:32},(_,i)=>{const angle=i*Math.PI/16;let distance=0;for(let d=1;d<100;d+=2){if(safe(p[0]+Math.cos(angle)*d,p[1]+Math.sin(angle)*d))break;distance=d;}return {angle,distance};}).sort((a,b)=>b.distance-a.distance);
      this.yaw=directions[0].angle;this.pitch=.04;this.controls.enabled=false;this.walking=true;this.camera.near=.08;this.camera.updateProjectionMatrix();this.look();
    }else{this.walking=false;this.controls.enabled=true;this.camera.near=.3;this.camera.updateProjectionMatrix();this.camera.position.copy(this.orbitSaved.position);this.controls.target.copy(this.orbitSaved.target);this.controls.update();}
    this.keys.clear();this.update({walking:this.walking});this.dirty=true;this.lastDetails=0;
  }
  look(){const d=new T.Vector3(Math.cos(this.yaw)*Math.cos(this.pitch),Math.sin(this.yaw)*Math.cos(this.pitch),Math.sin(this.pitch));this.camera.lookAt(this.camera.position.clone().add(d));this.dirty=true;}
  move(dt){
    const forward=(this.keys.has('KeyW')||this.keys.has('ArrowUp')?1:0)-(this.keys.has('KeyS')||this.keys.has('ArrowDown')?1:0);
    const right=(this.keys.has('KeyD')||this.keys.has('ArrowRight')?1:0)-(this.keys.has('KeyA')||this.keys.has('ArrowLeft')?1:0);
    if(!forward&&!right)return;
    const speed=(this.keys.has('ShiftLeft')||this.keys.has('ShiftRight')?9:3.8)*dt/(Math.hypot(forward,right)||1),dx=(Math.cos(this.yaw)*forward+Math.sin(this.yaw)*right)*speed,dy=(Math.sin(this.yaw)*forward-Math.cos(this.yaw)*right)*speed;
    const next=slideMove(this.camera.position.toArray(),[dx,dy],this.blocked);
    // Only step onto a deck close to current foot height: walking under a bridge must not teleport upward.
    const ray=new T.Raycaster(new T.Vector3(next[0],next[1],next[2]-.95),new T.Vector3(0,0,-1),0,1.5);
    const hit=ray.intersectObjects(this.walkable,false)[0];const foot=hit?.point.z??this.heightAt(next[0],next[1]);
    if(foot<next[2]-1.7+.75&&foot>next[2]-1.7-1.2)next[2]=foot+1.7;
    this.camera.position.fromArray(next);this.look();
  }
  loop(time){
    if(this.disposed)return;this.raf=requestAnimationFrame(this.loop);if(document.hidden)return;
    const dt=Math.min(.05,(time-this.lastTime)/1000);this.lastTime=time;
    if(this.walking)this.move(dt);else if(this.controls.update())this.dirty=true;
    if(time-this.lastDetails>650){this.detailSelection();this.lastDetails=time;}
    if(this.dirty){const arrow=document.querySelector('.north span');if(arrow)arrow.style.transform=`rotate(${(this.walking?this.yaw-Math.PI/2:this.controls.getAzimuthalAngle())*180/Math.PI}deg)`;const start=performance.now();this.renderer.render(this.scene,this.camera);this.renderTimes.push(performance.now()-start);if(this.renderTimes.length>120)this.renderTimes.shift();this.frames++;this.dirty=false;}
  }
  getStats(){const info=this.renderer.info;return {...this.status,device:this.device,software:this.software,camera:this.camera.position.toArray(),drawCalls:info.render.calls,triangles:info.render.triangles,geometryCount:info.memory.geometries,textureCount:info.memory.textures,nativePropTemplates:this.nativeProps.loaded.size,nativePropInstances:this.nativeProps.active.size,nativePropPending:this.nativeProps.pending.size,nativePropFailures:[...this.nativeProps.failed],groundMaterial:this.data?.ground.materials?'PBR':'Beauty RGB',gtRoadTiles:this.gtDecks.size,gtNearTiles:[...this.gtDecks.values()].filter(e=>e.near).length,gtPending:this.gtNearRequests.size,detailInstances:this.fine.size,cachedResources:this.cache.size,pending:this.pending,mapTiles:this.groundTiles.size,frames:this.frames,meanSubmitMs:this.renderTimes.reduce((a,b)=>a+b,0)/Math.max(1,this.renderTimes.length),failures:[...this.failed],baseReadyMs:this.baseReadyAt,assetTransfers:{...this.assets.stats},textureUpgrades:[...this.textureDetails.values()].filter(e=>e.done).length,textureUpgradeTotal:this.textureDetails.size,texturePending:this.textureActive,textureWanted:this.textureWanted.filter(e=>!e.done).length};}
  destroy(){this.disposed=true;cancelAnimationFrame(this.raf);this.resize.disconnect();this.listeners.forEach(f=>f());this.controls.dispose();for(const r of this.far.values())r.mask.dispose();this.landMask?.dispose();dispose(this.scene);this.materialEnvironment.dispose();for(const r of this.cache.values())dispose(r.root);this.ktx.dispose();this.renderer.dispose();this.renderer.domElement.remove();}
}
