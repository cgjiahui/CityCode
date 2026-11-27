import * as THREE from 'three';

// Historical Pixal3D stores world coordinates as (UE X, UE Z-height, UE Y).
export function neuralWorldMatrix() {
  return new THREE.Matrix4().set(1,0,0,0, 0,0,1,0, 0,1,0,0, 0,0,0,1);
}

// Image rows increase toward UE +Y. The top image edge belongs at world ymin.
export function mapGeometry(bounds, z) {
  const [x0, y0, x1, y1] = bounds;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute([x0,y0,z, x1,y0,z, x1,y1,z, x0,y1,z], 3));
  geometry.setAttribute('uv', new THREE.Float32BufferAttribute([0,1, 1,1, 1,0, 0,0], 2));
  geometry.setIndex([0,1,2, 0,2,3]); geometry.computeVertexNormals();
  return geometry;
}

export class BackgroundLayers {
  constructor(scene, loader, data, changed, disposeTree) {
    Object.assign(this, { scene, loader, data, changed, disposeTree });
    this.settings = {}; this.originalMaterials = new Map(); this.jobs = new Map(); this.errors = {};
    this.highlight = new THREE.MeshBasicMaterial({ color: '#ef9c36', side: THREE.DoubleSide, toneMapped: false });
  }

  configure(settings) {
    this.settings = settings;
    if (settings.view !== 'single') {
      const neural = settings.backgroundMode !== 'surface';
      const solid = !neural && settings.backgroundMap && settings.mapSolid && this.data.files.map_solid;
      if (settings.backgroundMode === 'coastal' && this.data.files.coastal && !this.coastal && !this.errors.coastal) this.load('coastal');
      if (settings.backgroundMode === 'pixal3d' && this.data.files.neural && !this.neural && !this.errors.neural) this.load('neural');
      if (solid && !this.mapSolid && !this.errors.map_solid) this.load('map_solid');
      if (!neural && settings.backgroundMap && !solid && !this.map && !this.errors.map) this.load('map');
      if (((!neural && settings.backgroundGround && !solid) || settings.backgroundElevated) && !this.surfaces && !this.errors.surfaces) this.load('surfaces');
    }
    this.update();
  }

  async load(role) {
    if (this.jobs.has(role) || this.disposed) return;
    const controller = new AbortController(); this.jobs.set(role, controller); this.changed();
    try {
      const response = await fetch(this.data.files[role].url, { signal: controller.signal });
      if (!response.ok) throw new Error(`Could not load background file (${response.status})`);
      if (role === 'map') {
        const blob = await response.blob(); const url = URL.createObjectURL(blob);
        let texture;
        try { texture = await new THREE.TextureLoader().loadAsync(url); }
        finally { URL.revokeObjectURL(url); }
        if (this.disposed) { texture.dispose(); return; }
        texture.colorSpace = THREE.SRGBColorSpace;
        this.map = new THREE.Mesh(mapGeometry(this.data.bounds_xy, this.data.map_plane_z_m),
          new THREE.MeshBasicMaterial({ map: texture, side: THREE.DoubleSide, toneMapped: false }));
        this.map.name = 'historical_rgb_map'; this.scene.add(this.map);
      } else if (role === 'map_solid') {
        const gltf = await this.loader.parseAsync(await response.arrayBuffer(), '');
        if (this.disposed) { this.disposeTree(gltf.scene); return; }
        const land = gltf.scene.getObjectByName('land'), water = gltf.scene.getObjectByName('water');
        if (!land || !water) { this.disposeTree(gltf.scene); throw new Error('The RGB map is missing a separate land or water layer'); }
        // This asset already uses UE XYZ/Z-up, just like the height surfaces.
        gltf.scene.traverse(node => {
          if (!node.isMesh) return;
          const source = node.material;
          node.material = new THREE.MeshBasicMaterial({ map: source.map, color: source.color,
            side: THREE.DoubleSide, toneMapped: false });
          source.dispose();
        });
        this.mapSolid = gltf.scene; this.mapLand = land; this.mapWater = water;
        this.scene.add(this.mapSolid);
      } else if (role === 'neural' || role === 'coastal') {
        const gltf = await this.loader.parseAsync(await response.arrayBuffer(), '');
        if (this.disposed) { this.disposeTree(gltf.scene); return; }
        this[role] = gltf.scene;
        this[role].applyMatrix4(neuralWorldMatrix());
        this[role].traverse(node => {
          if (!node.isMesh) return;
          const source = node.material;
          node.material = new THREE.MeshBasicMaterial({ map: source.map, color: source.color,
            side: THREE.DoubleSide, toneMapped: false });
          source.dispose();
        });
        this.scene.add(this[role]);
        if (role === 'coastal') this.water = this.coastal.getObjectByName('water_plane');
      } else {
        const gltf = await this.loader.parseAsync(await response.arrayBuffer(), '');
        if (this.disposed) { this.disposeTree(gltf.scene); return; }
        const base = gltf.scene.getObjectByName('base'), elevated = gltf.scene.getObjectByName('elevated');
        if (!base || !elevated) { this.disposeTree(gltf.scene); throw new Error('The background is missing a separate ground or elevated layer'); }
        this.surfaces = gltf.scene; this.base = base; this.elevated = elevated;
        this.surfaces.traverse(node => {
          if (!node.isMesh) return;
          // Keep native world Z and both layers; no building placement transform.
          const source = node.material;
          node.material = new THREE.MeshBasicMaterial({ map: source.map, color: source.color,
            side: THREE.DoubleSide, toneMapped: false });
          source.dispose();
        });
        elevated.traverse(node => { if (node.isMesh) this.originalMaterials.set(node, node.material); });
        this.scene.add(this.surfaces);
      }
    } catch (error) {
      if (error.name !== 'AbortError') this.errors[role] = error.message;
    } finally {
      this.jobs.delete(role);
      if (!this.disposed) { this.update(); this.changed(); }
    }
  }

  update() {
    const s = this.settings, visible = s.view !== 'single';
    const neural = s.backgroundMode !== 'surface';
    const solid = !neural && s.backgroundMap && s.mapSolid && Boolean(this.data.files?.map_solid);
    if (this.coastal) this.coastal.visible = visible && s.backgroundMode === 'coastal';
    if (this.water) this.water.visible = s.backgroundWater !== false;
    if (this.neural) this.neural.visible = visible && s.backgroundMode === 'pixal3d';
    if (this.mapSolid) this.mapSolid.visible = visible && Boolean(solid);
    if (this.mapWater) this.mapWater.visible = s.backgroundWater !== false;
    if (this.map) this.map.visible = visible && !neural && Boolean(s.backgroundMap) && !solid;
    if (this.base) this.base.visible = visible && !neural && Boolean(s.backgroundGround) && !solid;
    if (this.elevated) this.elevated.visible = visible && Boolean(s.backgroundElevated);
    for (const [node, original] of this.originalMaterials) node.material = s.elevatedStyle === 'texture' ? original : this.highlight;
    this.changed();
  }

  stats() {
    return { available: true, mapLoaded: Boolean(this.map), surfacesLoaded: Boolean(this.surfaces),
      mapSolidLoaded: Boolean(this.mapSolid), mapSolidVisible: Boolean(this.mapSolid?.visible),
      mapWaterVisible: Boolean(this.mapSolid?.visible && this.mapWater?.visible), mapSolidFaces: this.data.map_solid?.faces,
      coastalLoaded: Boolean(this.coastal), coastalVisible: Boolean(this.coastal?.visible),
      waterVisible: Boolean(this.coastal?.visible && this.water?.visible), coastalFaces: this.data.coastal?.faces,
      neuralLoaded: Boolean(this.neural), neuralVisible: Boolean(this.neural?.visible),
      mode: this.settings.backgroundMode || 'surface', neuralFaces: this.data.neural?.faces,
      mapVisible: Boolean(this.map?.visible), groundVisible: Boolean(this.base?.visible),
      elevatedVisible: Boolean(this.elevated?.visible), elevatedStyle: this.settings.elevatedStyle,
      pending: this.jobs.size, errors: Object.values(this.errors), elevatedFaces: this.data.layers.elevated.faces };
  }

  dispose() {
    this.disposed = true;
    for (const job of this.jobs.values()) job.abort();
    for (const [node, original] of this.originalMaterials) node.material = original;
    this.highlight.dispose();
    for (const object of [this.map, this.mapSolid, this.surfaces, this.neural, this.coastal]) if (object) { this.scene.remove(object); this.disposeTree(object); }
  }
}
