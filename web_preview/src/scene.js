import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/addons/libs/meshopt_decoder.module.js';
import { RoomEnvironment } from 'three/addons/environments/RoomEnvironment.js';
import { applySurfaceDisplay, applyPartColor, materialTextures } from './materials.js';
import { PART_COLORS } from './massing.js';
import { BackgroundLayers } from './background.js';
import { CollisionLayers, collisionFrameBounds } from './collisions.js';

export const MODEL_LIMIT = 16;
const BYTE_LIMIT = 110 * 1024 * 1024;
const MAX_IN_FLIGHT = 2;
const vector = a => new THREE.Vector3(...a);
const distance = (a, b) => Math.hypot(a.center[0]-b.center[0], a.center[1]-b.center[1]);

export function wholeCityPlan(assets, { buildings = true, quality = 'preview', accepted = true, references = false, massing = true } = {}) {
  if (!['preview', '0.1M', '0.25M'].includes(quality)) throw new Error('Unknown whole-city quality');
  return new Map(assets.filter(a => a.surface ? (a.enabled ? accepted : references) : buildings && (massing || !a.massing))
    .map(a => [a.id, a.surface ? '0.1M' : a.lods[quality] ? quality : '0.1M']));
}

async function limitTextureSize(object, limit) {
  const textures = new Set();
  object.traverse(node => {
    for (const material of [].concat(node.material || []))
      for (const value of Object.values(material)) if (value?.isTexture) textures.add(value);
  });
  const replacements = new Map(), oldImages = new Set();
  for (const texture of textures) {
    const source = texture.image;
    if (!source?.width || !source?.height || Math.max(source.width, source.height) <= limit) continue;
    let resized = replacements.get(source);
    if (!resized) {
      const scale = limit/Math.max(source.width, source.height);
      resized = await createImageBitmap(source, { resizeWidth: Math.max(1, Math.round(source.width*scale)),
        resizeHeight: Math.max(1, Math.round(source.height*scale)), resizeQuality: 'high',
        premultiplyAlpha: 'none', colorSpaceConversion: 'none' });
      replacements.set(source, resized); oldImages.add(source);
    }
    texture.image = resized; texture.needsUpdate = true;
  }
  oldImages.forEach(image => image.close?.());
  return Math.max(0, ...[...textures].map(texture => Math.max(texture.image?.width || 0, texture.image?.height || 0)));
}

export function neighborhood(assets, selected, radius, single = false, surfaces = false, limit = MODEL_LIMIT, acceptedSurfaces = true) {
  const primary = assets.filter(a => !a.surface);
  const group = new Set(selected.group);
  const parents = new Set(selected.parent_source_id ? primary.filter(a=>a.sources?.includes(selected.parent_source_id)).map(a=>a.id) : []);
  const near = primary.filter(a => group.has(a.id) || (!single && (parents.has(a.id) || distance(a, selected) <= radius)))
    .sort((a, b) => (group.has(b.id)-group.has(a.id)) || (parents.has(b.id)-parents.has(a.id)) || distance(a, selected)-distance(b, selected));
  const wanted = near.slice(0, limit);
  let bytes = 0;
  const budgeted = wanted.filter(a => {
    const accepted = group.has(a.id) || bytes+a.lods['0.1M'].bytes <= BYTE_LIMIT;
    if (accepted) bytes += a.lods['0.1M'].bytes;
    return accepted;
  });
  // These are partial observed meshes, never part of an isolated generated model.
  // Accepted attachments follow the release binding; raw reference fragments
  // require their own explicit switch and must not silently re-enable attachments.
  const extra = single ? [] : assets.filter(a => a.surface &&
      (a.enabled ? acceptedSurfaces : surfaces) && distance(a, selected) <= radius)
    .sort((a, b) => (Boolean(b.enabled)-Boolean(a.enabled)) || distance(a, selected)-distance(b, selected));
  return { assets: [...budgeted, ...extra.slice(0, 4)], total: near.length, limited: near.length > budgeted.length };
}

function disposeTree(object) {
  const geometries = new Set(), materials = new Set(), textures = new Set();
  object.traverse(node => {
    if (node.geometry) geometries.add(node.geometry);
    if (node.material) for (const material of [].concat(node.material)) {
      materials.add(material);
      for (const texture of materialTextures(material)) textures.add(texture);
    }
  });
  geometries.forEach(g => g.dispose());
  textures.forEach(t => { t.dispose(); t.source?.data?.close?.(); });
  materials.forEach(m => m.dispose());
}

export class CityScene {
  constructor(host, data, callbacks) {
    this.host = host; this.data = data; this.callbacks = callbacks;
    this.assets = new Map(data.assets.map(a => [a.id, a]));
    this.massingSources = new Map((data.massing_sources || []).flatMap(source=>source.part_ids.map(id=>[id,source])));
    this.models = new Map(); this.jobs = new Map(); this.wanted = new Map(); this.failures = new Map();
    this.cityModels = new Map(); this.cityJobs = new Map(); this.cityFailures = new Map(); this.cityWanted = new Map();
    this.cityActive = false; this.cityBuildings = true; this.cityQuality = 'preview'; this.cityPaused = false; this.cityFitted = false; this.cityAutoFit = true;
    this.proxies = new Map(); this.outlines = new Map(); this.disposed = false; this.dirty = true;
    this.radius = 120; this.single = false; this.showSurfaces = false; this.showAcceptedSurfaces = true; this.showOutlines = false;
    this.showStructures = false; this.overview = true; this.high = false; this.showMassing = true;
    this.surfaceDisplay = 'neutral';
    this.massingColors = false; this.showSourceOutline = false;
    this.collisionSettings = { enabled: false, kind: 'all', priorityOnly: true }; this.collisionCase = null;
    this.scene = new THREE.Scene(); this.scene.background = new THREE.Color('#edf0f2');
    this.camera = new THREE.PerspectiveCamera(40, 1, .5, 16000); this.camera.up.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, logarithmicDepthBuffer: true, powerPreference: 'high-performance' });
    const gl = this.renderer.getContext(), debug = gl.getExtension('WEBGL_debug_renderer_info');
    const rendererName = debug ? gl.getParameter(debug.UNMASKED_RENDERER_WEBGL) : '';
    this.softwareRenderer = /swiftshader|llvmpipe|software|softpipe/i.test(rendererName);
    this.modelLimit = this.softwareRenderer ? 6 : MODEL_LIMIT;
    this.renderer.setPixelRatio(this.softwareRenderer ? .8 : Math.min(window.devicePixelRatio, 1.5));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.0;
    host.appendChild(this.renderer.domElement);
    this.renderer.domElement.setAttribute('aria-label', '可旋转和缩放的建筑场景');
    this.renderer.domElement.addEventListener('webglcontextlost', this.contextLost = event => {
      event.preventDefault(); this.callbacks.onError('浏览器图形资源不足，请刷新并缩小周边范围。');
    });
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true; this.controls.dampingFactor = .09;
    this.controls.maxPolarAngle = Math.PI * .485; this.controls.minDistance = 8; this.controls.maxDistance = 6500;
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xaab3bf, 2));
    const sun = new THREE.DirectionalLight(0xfff5e5, 3.2); sun.position.set(-500, -800, 1800); this.scene.add(sun);
    const fill = new THREE.DirectionalLight(0xddeaff, 1.2); fill.position.set(700, 800, 500); this.scene.add(fill);
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    const environment = new RoomEnvironment();
    this.environment = pmrem.fromScene(environment).texture; this.scene.environment = this.environment;
    this.scene.environmentIntensity = .45;
    environment.dispose(); pmrem.dispose();
    const min = data.bounds[0], max = data.bounds[1];
    this.cityBox = new THREE.Box3(vector(min), vector(max));
    const size = Math.max(max[0]-min[0], max[1]-min[1])+350;
    const center = this.cityBox.getCenter(new THREE.Vector3());
    this.ground = new THREE.Mesh(new THREE.PlaneGeometry(size, size), new THREE.MeshStandardMaterial({ color: '#e2e7ea', roughness: 1 }));
    this.ground.position.set(center.x, center.y, min[2]-1.5); this.scene.add(this.ground);
    const grid = new THREE.GridHelper(size, Math.round(size/100), 0xcbd3d9, 0xd8dfe3);
    grid.rotation.x = Math.PI/2; grid.position.copy(this.ground.position); grid.position.z += .08;
    this.scene.add(grid); this.grid = grid;
    this.proxyMaterial = new THREE.MeshStandardMaterial({ color: '#cbd2d8', roughness: .95, flatShading: true });
    this.outlineMaterial = new THREE.LineBasicMaterial({ color: '#c77c35', depthTest: false, transparent: true, opacity: .8 });
    for (const asset of data.assets.filter(a => !a.surface)) this.addProxy(asset);
    this.collisions = new CollisionLayers(this.scene, data.collision_audit, this.assets);
    this.loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    if (data.background) this.background = new BackgroundLayers(this.scene, this.loader, data.background,
      () => { this.dirty = true; this.report(); }, disposeTree);
    this.resizeObserver = new ResizeObserver(() => this.resize()); this.resizeObserver.observe(host);
    this.resize(); this.fit(this.cityBox, 1.15);
    this.controls.addEventListener('change', () => {
      this.dirty = true;
      const direction = this.camera.position.clone().sub(this.controls.target);
      this.callbacks.onBearing?.(Math.atan2(direction.x, direction.y)*180/Math.PI);
    });
    let down;
    this.pointerDown = e => { down = [e.clientX, e.clientY]; this.cityAutoFit = false; };
    this.pointerUp = e => {
      if (e.button !== 0 || !down || Math.hypot(e.clientX-down[0], e.clientY-down[1]) > 5) return;
      const rect = this.renderer.domElement.getBoundingClientRect();
      const ray = new THREE.Raycaster();
      ray.setFromCamera(new THREE.Vector2((e.clientX-rect.left)/rect.width*2-1, -(e.clientY-rect.top)/rect.height*2+1), this.camera);
      const collisionId = this.collisions.pick(ray);
      if (collisionId) { this.callbacks.onCollisionSelect?.(collisionId); return; }
      // Lightweight source solids remain the pick targets even after a real model is loaded.
      const hits = ray.intersectObjects([...this.proxies.values()], true);
      if (hits.length) this.callbacks.onSelect(hits[0].object.userData.assetId);
    };
    this.renderer.domElement.addEventListener('pointerdown', this.pointerDown);
    this.renderer.domElement.addEventListener('pointerup', this.pointerUp);
    const animate = () => {
      if (this.disposed) return;
      this.frame = requestAnimationFrame(animate); this.controls.update();
      if (this.dirty) { this.collisions.update(this.camera, this.host.clientHeight); this.renderer.render(this.scene, this.camera); this.dirty = false; }
    };
    animate();
  }

  addProxy(asset) {
    const group = new THREE.Group();
    for (const polygon of asset.polygons) {
      if (polygon[0].length < 4) continue;
      const shape = new THREE.Shape(polygon[0].map(p => new THREE.Vector2(...p)));
      for (const hole of polygon.slice(1)) shape.holes.push(new THREE.Path(hole.map(p => new THREE.Vector2(...p))));
      const geometry = new THREE.ExtrudeGeometry(shape, { depth: Math.max(1, asset.top-asset.base), bevelEnabled: false, curveSegments: 1, steps: 1 });
      const mesh = new THREE.Mesh(geometry, this.proxyMaterial); mesh.position.z = asset.base;
      mesh.userData.assetId = asset.id; group.add(mesh);
    }
    this.proxies.set(asset.id, group); this.scene.add(group);
    const lines = new THREE.Group();
    for (const poly of asset.polygons) for (const ring of poly) {
      const geometry = new THREE.BufferGeometry().setFromPoints(ring.map(p => new THREE.Vector3(...p, asset.base+.3)));
      const line = new THREE.Line(geometry, this.outlineMaterial); line.renderOrder = 8; lines.add(line);
    }
    lines.visible = false; this.scene.add(lines); this.outlines.set(asset.id, lines);
  }

  resize() {
    const { width, height } = this.host.getBoundingClientRect();
    this.camera.aspect = width/Math.max(1, height); this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
    this.dirty = true;
  }

  fit(box, padding = 1.5, worldPoints = null) {
    const center = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
    const direction = (this.selected?.massing && this.showSourceOutline && !this.overview ?
      new THREE.Vector3(.45,-.65,1.6) : new THREE.Vector3(.72, -1, .85)).normalize();
    const right = new THREE.Vector3().crossVectors(this.camera.up, direction).normalize();
    const up = new THREE.Vector3().crossVectors(direction, right).normalize();
    const tangent = Math.tan(THREE.MathUtils.degToRad(this.camera.fov/2));
    const points = worldPoints || [];
    if (!worldPoints) for (const x of [-.5,.5]) for (const y of [-.5,.5]) for (const z of [-.5,.5])
      points.push(center.clone().add(new THREE.Vector3(size.x*x,size.y*y,size.z*z)));
    let minX=Infinity,maxX=-Infinity,minY=Infinity,maxY=-Infinity;
    for (const point of points) {
      const v=point.clone().sub(center), x=v.dot(right), y=v.dot(up);
      minX=Math.min(minX,x);maxX=Math.max(maxX,x);minY=Math.min(minY,y);maxY=Math.max(maxY,y);
    }
    center.addScaledVector(right,(minX+maxX)/2).addScaledVector(up,(minY+maxY)/2);
    let distance = 50;
    for (const point of points) {
      const v = point.clone().sub(center);
      distance = Math.max(distance, v.dot(direction)+Math.abs(v.dot(right))*padding/(tangent*this.camera.aspect),
        v.dot(direction)+Math.abs(v.dot(up))*padding/tangent);
    }
    this.controls.target.copy(center);
    this.camera.position.copy(center).add(direction.multiplyScalar(distance));
    this.controls.update();
  }

  select(id, { refit = true } = {}) {
    this.selected = this.assets.get(id); if (!this.selected || this.selected.surface) return;
    this.high = false;
    this.refreshDesired(); this.updateVisibility();
    if (refit) { this.cityAutoFit = false; this.frameSelection(); }
    this.updateSelectionBox();
    this.updateSourceOutline(); this.syncCollisions(); this.report();
  }

  syncCollisions() {
    this.collisions.configure({ ...this.collisionSettings, selectedId: this.collisionCase?.id,
      overview: this.overview, center: this.selected?.center, radius: this.radius });
    this.dirty = true;
  }

  selectCollision(id, { refit = true } = {}) {
    this.collisionCase = this.collisions.byId.get(id) || null;
    this.refreshDesired(); this.updateVisibility(); this.syncCollisions();
    if (refit && this.collisionCase) this.focusCollision();
    this.report();
  }

  focusCollision() {
    if (!this.collisionCase) return;
    const bounds = collisionFrameBounds(this.collisionCase, this.data.assets);
    if (!bounds) return;
    const box = new THREE.Box3(vector(bounds[0]), vector(bounds[1]));
    // Keep the associated buildings and enough nearby context in view, rather
    // than fitting a tiny patch inside a wall and putting the camera in it.
    box.expandByVector(new THREE.Vector3(20, 20, 3));
    this.cityAutoFit = false; this.fit(box, 1.35); this.dirty = true;
  }

  frameSelection() {
    if (!this.selected) return;
    const box = new THREE.Box3();
    for (const id of this.selected.group) {
      const a = this.assets.get(id); if (a) box.union(new THREE.Box3(vector(a.bounds[0]), vector(a.bounds[1])));
    }
    const center = box.getCenter(new THREE.Vector3());
    if (!this.single) {
      const half = this.radius*.55;
      box.expandByPoint(new THREE.Vector3(center.x-half, center.y-half, box.min.z));
      box.expandByPoint(new THREE.Vector3(center.x+half, center.y+half, box.min.z));
    }
    this.fit(box, this.single ? 1.45 : 1.3);
  }

  allCity() {
    this.overview = true; this.single = false; this.high = false; this.cityAutoFit = true;
    this.refreshDesired(); this.refreshCity();
    this.updateVisibility(); this.fit(this.cityBox, 1.1, this.cityFitPoints); this.report();
  }

  configure({ radius, single, view = single ? 'single' : 'neighborhood', surfaces, acceptedSurfaces = true,
    outlines, structures, cityBuildings = true, cityQuality = 'preview', cityPaused = false, surfaceDisplay = 'neutral', massing = true,
    massingColors = false, sourceOutline = false,
    collisionEnabled = false, collisionKind = 'all', collisionPriorityOnly = true,
    backgroundMap = true, mapSolid = true, backgroundGround = false, backgroundElevated = true, backgroundWater = true, elevatedStyle = 'highlight', backgroundMode = 'surface' }) {
    const wasOverview = this.overview;
    const needsLoad = radius !== this.radius || single !== this.single || surfaces !== this.showSurfaces || acceptedSurfaces !== this.showAcceptedSurfaces || massing !== this.showMassing;
    const nextSingle = view === 'single';
    const needsFit = radius !== this.radius || nextSingle !== this.single || (view === 'city') !== wasOverview;
    const cityChanged = cityBuildings !== this.cityBuildings || cityQuality !== this.cityQuality ||
      surfaces !== this.showSurfaces || acceptedSurfaces !== this.showAcceptedSurfaces || massing !== this.showMassing;
    this.radius = radius; this.single = nextSingle; this.overview = view === 'city';
    this.showSurfaces = surfaces; this.showAcceptedSurfaces = acceptedSurfaces;
    this.cityBuildings = cityBuildings; this.cityQuality = cityQuality; this.cityPaused = cityPaused;
    this.showMassing = massing;
    this.massingColors = massingColors; this.showSourceOutline = sourceOutline;
    this.collisionSettings = { enabled: collisionEnabled, kind: collisionKind, priorityOnly: collisionPriorityOnly };
    this.background?.configure({ view, backgroundMap, mapSolid, backgroundGround, backgroundElevated, backgroundWater, elevatedStyle, backgroundMode });
    this.ground.visible = !this.background || (view !== 'single' && backgroundMode === 'surface' && !backgroundMap && !backgroundGround);
    this.grid.visible = this.ground.visible;
    this.showOutlines = outlines; this.showStructures = structures;
    if (surfaceDisplay !== this.surfaceDisplay) {
      this.surfaceDisplay = surfaceDisplay;
      this.scene.environmentIntensity = surfaceDisplay === 'original' ? 1 : .45;
      for (const object of [...this.models.values(), ...this.cityModels.values()]) applySurfaceDisplay(object, surfaceDisplay);
    }
    if (wasOverview !== this.overview) this.high = false;
    if (cityChanged || wasOverview !== this.overview || !this.cityActive) this.refreshCity();
    if (!this.overview) for (const job of this.cityJobs.values()) job.abort();
    else if (this.cityPaused) for (const job of this.cityJobs.values()) job.abort();
    else this.pumpCity();
    if (!this.cityBuildings && this.overview) this.high = false;
    if (needsLoad || cityChanged || wasOverview !== this.overview || nextSingle !== Boolean(single)) this.refreshDesired();
    this.updateVisibility();
    this.updateSourceOutline();
    this.syncCollisions();
    if (needsFit) { if (this.overview) this.fit(this.cityBox, 1.1, this.cityFitPoints); else this.frameSelection(); }
    this.report();
  }

  loadHigh() { if (this.selected && (!this.overview || this.cityBuildings)) { this.high = true; this.refreshDesired(); } }

  refreshDesired() {
    if (!this.selected) return;
    const plan = this.overview ? { assets: this.high && this.cityBuildings && (this.showMassing || !this.selected.massing) ? this.selected.group.map(id => this.assets.get(id)) : [], total: 0, limited: false } :
      neighborhood(this.data.assets, this.selected, this.radius, this.single, this.showSurfaces, this.modelLimit, this.showAcceptedSurfaces);
    // Keep the same neighbour allocation when comparing massing on/off.
    // A hidden part must not pull a different building into the local model cap.
    if (!this.showMassing) plan.assets=plan.assets.filter(a=>!a.massing ||
      (this.selected.massing && this.selected.group.includes(a.id)));
    if (this.collisionSettings.enabled && this.collisionCase && !this.overview && !this.single) {
      const required = this.collisionCase.asset_ids.map(id => this.assets.get(id)).filter(a => a && !a.surface);
      const ids = new Set(required.map(a => a.id));
      plan.assets = [...required, ...plan.assets.filter(a => !ids.has(a.id))].slice(0, Math.max(this.modelLimit, required.length));
    }
    this.plan = plan; this.wanted = new Map(plan.assets.map(a => [a.id, this.high && this.selected.group.includes(a.id) ? '0.25M' : '0.1M']));
    for (const [key, object] of this.models) {
      if (!this.wanted.has(key) || this.wanted.get(key) !== object.userData.lod) this.removeModel(key);
    }
    for (const [key, job] of this.jobs) if (!this.wanted.has(key) || job.lod !== this.wanted.get(key)) job.abort();
    this.failures.clear(); this.pump(); this.report();
  }

  refreshCity() {
    this.cityActive = true;
    this.cityWanted = wholeCityPlan(this.data.assets, { buildings: this.cityBuildings, quality: this.cityQuality,
      accepted: this.showAcceptedSurfaces, references: this.showSurfaces, massing:this.showMassing });
    for (const [id, object] of this.cityModels) if (!this.cityWanted.has(id)) {
      this.scene.remove(object); disposeTree(object); this.cityModels.delete(id);
    }
    for (const [id, job] of this.cityJobs) if (!this.cityWanted.has(id) || this.cityWanted.get(id) !== job.lod) job.abort();
    this.cityFailures.clear(); this.pumpCity();
  }

  async fetchModel(asset, lod, controller, scope = 'local') {
    const response = await fetch(asset.lods[lod].url, { signal: controller.signal });
    if (!response.ok) throw new Error(`模型文件读取失败 (${response.status})`);
    const bytes = await response.arrayBuffer();
    if (controller.signal.aborted) return null;
    const gltf = await this.loader.parseAsync(bytes, '');
    const valid = () => !this.disposed && !controller.signal.aborted &&
      (scope === 'city' ? this.overview && this.cityWanted.get(asset.id) === lod : this.wanted.get(asset.id) === lod);
    if (!valid()) { disposeTree(gltf.scene); return null; }
    let textureMaxPx = 0;
    try { if (scope === 'city') textureMaxPx = await limitTextureSize(gltf.scene, this.softwareRenderer ? 128 : 512); }
    catch (error) { disposeTree(gltf.scene); throw error; }
    if (!valid()) { disposeTree(gltf.scene); return null; }
    const group = new THREE.Group(); group.matrixAutoUpdate = false;
    // Binding matrices are serialized as rows. Matrix4.set accepts row-major arguments.
    group.matrix.set(...asset.matrix.flat()); group.add(gltf.scene);
    let faces = 0;
    gltf.scene.traverse(node => { if (node.isMesh) faces += (node.geometry.index?.count || node.geometry.attributes.position.count)/3; });
    group.userData = { assetId: asset.id, lod, bytes: asset.lods[lod].bytes, textureMaxPx, faces };
    applySurfaceDisplay(group, this.surfaceDisplay);
    return group;
  }

  pump() {
    if (this.disposed) return;
    for (const [id, lod] of this.wanted) {
      if (this.jobs.size >= MAX_IN_FLIGHT) break;
      if (this.models.has(id) || this.jobs.has(id) || this.failures.has(id)) continue;
      const controller = new AbortController(); controller.lod = lod; this.jobs.set(id, controller);
      this.fetchModel(this.assets.get(id), lod, controller).then(object => {
        if (object) { this.models.set(id, object); this.scene.add(object); }
      }).catch(error => {
        if (error.name !== 'AbortError') this.failures.set(id, error.message);
      }).finally(() => {
        this.jobs.delete(id);
        if (!this.disposed) { this.updateVisibility(); this.updateSelectionBox(); this.report(); this.pump(); }
      });
    }
    this.report();
  }

  pumpCity() {
    if (this.disposed || !this.overview || !this.cityActive || this.cityPaused) return;
    for (const [id, lod] of this.cityWanted) {
      if (this.cityJobs.size >= (this.cityQuality === 'preview' ? 3 : 2)) break;
      if (this.cityModels.get(id)?.userData.lod === lod || this.cityJobs.has(id) || this.cityFailures.has(id)) continue;
      const asset = this.assets.get(id), controller = new AbortController(); controller.lod = lod; this.cityJobs.set(id, controller);
      this.fetchModel(asset, lod, controller, 'city').then(object => {
        if (object) {
          const previous = this.cityModels.get(id);
          if (previous) { this.scene.remove(previous); disposeTree(previous); }
          this.cityModels.set(id, object); this.scene.add(object);
        }
      }).catch(error => {
        if (error.name !== 'AbortError') this.cityFailures.set(id, error.message);
      }).finally(() => {
        this.cityJobs.delete(id);
        if (!this.disposed) {
          const buildings = [...this.cityModels].filter(([key]) => !this.assets.get(key).surface);
          const wantedBuildings=[...this.cityWanted.keys()].filter(id=>!this.assets.get(id).surface).length;
          if (this.overview && !this.cityFitted && wantedBuildings>0 && buildings.length === wantedBuildings) {
            const box = new THREE.Box3();
            this.cityFitPoints=[];
            for (const [, object] of buildings) {
              object.updateWorldMatrix(true, true); box.expandByObject(object);
              object.traverse(node => {
                if(!node.isMesh) return;
                node.geometry.computeBoundingBox();const b=node.geometry.boundingBox;
                for(const x of [b.min.x,b.max.x])for(const y of [b.min.y,b.max.y])for(const z of [b.min.z,b.max.z])
                  this.cityFitPoints.push(new THREE.Vector3(x,y,z).applyMatrix4(node.matrixWorld));
              });
            }
            this.cityBox.copy(box); if(this.cityAutoFit)this.fit(this.cityBox,1.12,this.cityFitPoints); this.cityFitted = true;
          }
          this.updateVisibility(); this.pumpCity(); this.report();
        }
      });
    }
    this.report();
  }

  removeModel(id) {
    const object = this.models.get(id); if (!object) return;
    this.scene.remove(object); disposeTree(object); this.models.delete(id);
  }

  updateVisibility() {
    this.dirty = true;
    const selectedGroup = new Set(this.selected?.group || []);
    for (const [id, object] of this.models) {
      object.visible = (this.showMassing || !this.assets.get(id).massing) && this.wanted.has(id);
    }
    for (const [id, object] of this.cityModels) {
      const asset = this.assets.get(id);
      object.visible = (this.showMassing || !asset.massing) && !this.models.has(id) && (this.overview ? this.cityWanted.has(id) :
        !asset.surface && !this.single && this.showStructures && this.selected && distance(asset, this.selected) <= this.radius+100);
    }
    for (const [id, proxy] of this.proxies) proxy.visible = (this.showMassing || !this.assets.get(id).massing) && this.showStructures && !this.models.has(id) && !this.cityModels.get(id)?.visible && (!this.single || this.overview || selectedGroup.has(id));
    for (const [id, lines] of this.outlines) lines.visible = (this.showMassing || !this.assets.get(id).massing) && this.showOutlines && (this.overview || (this.single ? selectedGroup.has(id) : this.wanted.has(id)));
    for (const [id, object] of [...this.models, ...this.cityModels]) {
      if (!this.assets.get(id).massing) continue;
      const index = this.selected?.group.indexOf(id) ?? -1;
      applyPartColor(object,this.massingColors && index>=0 ? PART_COLORS[index%PART_COLORS.length] : null);
    }
    if (this.selectionBox) this.selectionBox.visible = (!this.overview || this.cityBuildings) && !(this.selected?.massing && this.showSourceOutline) && !(this.collisionSettings.enabled && this.collisionCase);
  }

  updateSourceOutline() {
    if (this.sourceOutline) {
      this.scene.remove(this.sourceOutline);disposeTree(this.sourceOutline);this.sourceOutline=null;
    }
    const source = this.massingSources.get(this.selected?.id);
    if (!source || !this.showSourceOutline || this.overview) return;
    const group = new THREE.Group(), material = new THREE.LineBasicMaterial({color:'#117c99',depthTest:false,transparent:true,opacity:.95});
    for (const polygon of source.polygons) for (const ring of polygon) {
      const geometry = new THREE.BufferGeometry().setFromPoints(ring.map(([x,y])=>new THREE.Vector3(x,y,source.outline_z)));
      const line = new THREE.Line(geometry,material);line.renderOrder=12;group.add(line);
    }
    this.sourceOutline=group;this.scene.add(group);this.dirty=true;
  }

  updateSelectionBox() {
    this.dirty = true;
    if (this.selectionBox) { this.scene.remove(this.selectionBox); this.selectionBox.geometry.dispose(); this.selectionBox.material.dispose(); this.selectionBox = null; }
    if (!this.selected) return;
    const box = new THREE.Box3();
    for (const id of this.selected.group) {
      const object = this.models.get(id) || this.cityModels.get(id), asset = this.assets.get(id);
      if (object) { object.updateWorldMatrix(true, true); box.expandByObject(object, true); }
      else if (asset) box.union(new THREE.Box3(vector(asset.bounds[0]), vector(asset.bounds[1])));
    }
    this.selectionBox = new THREE.Box3Helper(box.expandByScalar(.3), 0xc77c35);
    this.selectionBox.material.depthTest = false; this.selectionBox.material.transparent = true; this.selectionBox.material.opacity = .85;
    this.selectionBox.renderOrder = 10; this.scene.add(this.selectionBox);
    this.selectionBox.visible = (!this.overview || this.cityBuildings) && !(this.selected?.massing && this.showSourceOutline) && !(this.collisionSettings.enabled && this.collisionCase);
  }

  report() {
    const selectedIds = this.selected?.group || [];
    const buildings = [...this.models].filter(([id]) => !this.assets.get(id).surface);
    const cityMatches = [...this.cityModels].filter(([id, object]) => this.cityWanted.get(id) === object.userData.lod);
    const cityBuildings = cityMatches.filter(([id]) => !this.assets.get(id).surface);
    const active = this.overview ? cityMatches : [...this.models];
    const surfaces = active.filter(([id]) => this.assets.get(id).surface).map(([id]) => this.assets.get(id));
    const rawReferences = surfaces.filter(a => !a.enabled).map(a => a.number);
    const acceptedSurfaces = surfaces.filter(a => a.enabled).map(a => a.number);
    const pendingModels = [...this.jobs.keys()].filter(id => !this.assets.get(id).surface).length;
    const totals = [...this.cityWanted.keys()].map(id => this.assets.get(id));
    this.callbacks.onStats({ loaded: buildings.length, pending: this.jobs.size+(this.overview ? this.cityJobs.size : 0), pendingModels,
      desired: [...this.wanted.keys()].filter(id => !this.assets.get(id).surface).length,
      rawReferences, acceptedSurfaces,
      selectedPartsVisible: selectedIds.filter(id=>this.assets.get(id)?.massing && (this.models.get(id)?.visible || this.cityModels.get(id)?.visible)).length,
      coloredParts: this.massingColors ? selectedIds.filter(id=>this.assets.get(id)?.massing && (this.models.get(id)?.visible || this.cityModels.get(id)?.visible)).length : 0,
      sourceOutlineVisible: Boolean(this.sourceOutline?.children.length),
      errors: [...this.failures].map(([id, message]) => `${this.assets.get(id).surface ? '观测面 ' : '建筑 '}${this.assets.get(id).number}：${message}`),
      limited: this.plan?.limited || false, total: this.plan?.total || 0,
      bytes: buildings.reduce((sum, [, object]) => sum+object.userData.bytes, 0),
      selectedLoaded: selectedIds.length > 0 && selectedIds.every(id => this.models.has(id) || (this.overview && this.cityModels.has(id))),
      high: this.high, overview: this.overview, cityLoaded: cityBuildings.length,
      collisions: this.collisions.stats(),
      background: this.background?.stats(),
      cityDesired: totals.filter(a => !a.surface).length, cityTotalLoaded: cityMatches.length,
      cityTotalDesired: this.cityWanted.size, cityPending: this.cityJobs.size,
      cityReferencesDesired: totals.filter(a => a.surface && !a.enabled).length,
      cityAcceptedDesired: totals.filter(a => a.surface && a.enabled).length,
      cityBytes: cityMatches.reduce((sum, [, object]) => sum+object.userData.bytes, 0),
      cityFaces: cityBuildings.reduce((sum, [, object]) => sum+object.userData.faces, 0),
      cityTextureMaxPx: Math.max(0, ...cityMatches.map(([, object]) => object.userData.textureMaxPx)),
      cityErrors: this.cityFailures.size, cityPaused: this.cityPaused, softwareRenderer: this.softwareRenderer });
  }

  zoom(factor) { this.camera.position.sub(this.controls.target).multiplyScalar(factor).add(this.controls.target); this.controls.update(); }
  focusElevated() {
    if (!this.data.background) return;
    this.cityAutoFit = false;
    const bounds = this.data.background.layers.elevated.bounds;
    this.fit(new THREE.Box3(vector(bounds[0]), vector(bounds[1])), 1.15);
  }
  focusCoast() {
    if (!this.data.background?.files.coastal && !this.data.background?.files.map_solid) return;
    this.cityAutoFit = false;
    this.controls.target.set(-200, -630, 0);
    this.camera.position.set(-270, -850, 65);
    this.controls.update(); this.dirty = true;
  }
  top() {
    const offset = this.camera.position.distanceTo(this.controls.target);
    this.camera.position.copy(this.controls.target).add(new THREE.Vector3(0, -.001, offset)); this.controls.update();
  }
  dispose() {
    this.disposed = true; cancelAnimationFrame(this.frame);
    for (const job of this.jobs.values()) job.abort();
    for (const job of this.cityJobs.values()) job.abort();
    this.background?.dispose();
    this.collisions.dispose();
    this.resizeObserver.disconnect(); this.controls.dispose();
    this.renderer.domElement.removeEventListener('pointerdown', this.pointerDown);
    this.renderer.domElement.removeEventListener('pointerup', this.pointerUp);
    this.renderer.domElement.removeEventListener('webglcontextlost', this.contextLost);
    disposeTree(this.scene); this.environment.dispose(); this.renderer.dispose(); this.renderer.domElement.remove();
  }
}
