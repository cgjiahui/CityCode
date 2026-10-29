// Display overrides stay in memory. Keep the imported glTF maps available both
// for restoring the source material and for disposing textures while matte.
const sourceMaterials = new WeakMap();
const partMaterials = new WeakMap();

export function applyPartColor(object, color) {
  object.traverse(node => {
    for (const material of [].concat(node.material || [])) {
      if (!material.color) continue;
      let source = partMaterials.get(material);
      if (!source && !color) continue;
      if (!source) {
        source = { color: material.color.clone(), map: material.map,
          emissive: material.emissive?.clone(), vertexColors: material.vertexColors };
        partMaterials.set(material, source);
      }
      if (color) {
        material.color.set(color); material.map = null; material.vertexColors = false;
        material.emissive?.set(0x000000);
      } else {
        material.color.copy(source.color); material.map = source.map; material.vertexColors = source.vertexColors;
        if (source.emissive) material.emissive.copy(source.emissive);
      }
      material.needsUpdate = true;
    }
  });
}

export function applySurfaceDisplay(object, mode) {
  if (!['neutral', 'original'].includes(mode)) throw new Error('Unknown surface display');
  object.traverse(node => {
    for (const material of [].concat(node.material || [])) {
      if (!material.isMeshStandardMaterial) continue;
      let source = sourceMaterials.get(material);
      if (!source) {
        source = { roughness: material.roughness, metalness: material.metalness,
          roughnessMap: material.roughnessMap, metalnessMap: material.metalnessMap };
        sourceMaterials.set(material, source);
      }
      Object.assign(material, mode === 'original' ? source :
        { roughness: .9, metalness: 0, roughnessMap: null, metalnessMap: null });
      material.needsUpdate = true;
    }
  });
}

export function materialTextures(material) {
  return new Set([...Object.values(material), ...Object.values(sourceMaterials.get(material) || {}),
    ...Object.values(partMaterials.get(material) || {})]
    .filter(value => value?.isTexture));
}
