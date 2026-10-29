import assert from 'node:assert/strict';
import test from 'node:test';
import * as THREE from 'three';
import { applySurfaceDisplay, applyPartColor, materialTextures } from '../src/materials.js';

test('matte viewing preserves geometry/color and restores the imported PBR maps exactly', () => {
  const map = new THREE.Texture(), packed = new THREE.Texture();
  const material = new THREE.MeshStandardMaterial({ map, color: '#81766a', roughness: .67,
    metalness: .23, roughnessMap: packed, metalnessMap: packed });
  const geometry = new THREE.BoxGeometry(), mesh = new THREE.Mesh(geometry, material);
  const sourceColor = material.color.clone();
  for (let i = 0; i < 3; i++) {
    applySurfaceDisplay(mesh, 'neutral');
    assert.equal(mesh.geometry, geometry); assert.equal(material.map, map);
    assert.ok(material.color.equals(sourceColor)); assert.equal(material.metalness, 0);
    assert.equal(material.roughnessMap, null); assert.equal(material.metalnessMap, null);
    // Hidden imported maps must remain reachable for cleanup without being duplicated.
    assert.deepEqual(materialTextures(material), new Set([map, packed]));
    applySurfaceDisplay(mesh, 'original');
    assert.equal(material.roughness, .67); assert.equal(material.metalness, .23);
    assert.equal(material.roughnessMap, packed); assert.equal(material.metalnessMap, packed);
  }
  geometry.dispose(); material.dispose(); map.dispose(); packed.dispose();
});

test('shared and multiple materials restore independently; unlit observed colors stay unchanged', () => {
  const a = new THREE.MeshStandardMaterial({ roughness: .2, metalness: .4 });
  const b = new THREE.MeshStandardMaterial({ roughness: .7, metalness: .1 });
  const unlit = new THREE.MeshBasicMaterial({ color: '#aabbcc' });
  const group = new THREE.Group(), geometry = new THREE.BoxGeometry();
  group.add(new THREE.Mesh(geometry, [a,b]), new THREE.Mesh(geometry,a), new THREE.Mesh(geometry,unlit));
  applySurfaceDisplay(group,'neutral'); applySurfaceDisplay(group,'neutral');
  applySurfaceDisplay(group,'original');
  assert.equal(a.roughness,.2); assert.equal(a.metalness,.4);
  assert.equal(b.roughness,.7); assert.equal(b.metalness,.1);
  assert.equal(unlit.color.getHexString(),'aabbcc');
  geometry.dispose(); a.dispose(); b.dispose(); unlit.dispose();
});

test('massing colors preserve source textures and restore the actual model without changing geometry', () => {
  const texture=new THREE.Texture(), geometry=new THREE.BoxGeometry();
  const material=new THREE.MeshStandardMaterial({map:texture,color:'#556677',emissive:'#112233',vertexColors:true});
  const mesh=new THREE.Mesh(geometry,material), original=material.color.clone(), emissive=material.emissive.clone();
  const positions=geometry.attributes.position.array.slice();
  applyPartColor(mesh,'#e58b36');
  assert.equal(material.map,null);assert.equal(material.vertexColors,false);
  assert.ok(materialTextures(material).has(texture));
  applySurfaceDisplay(mesh,'neutral');applyPartColor(mesh,null);
  assert.equal(material.map,texture);assert.equal(material.vertexColors,true);
  assert.ok(material.color.equals(original));assert.ok(material.emissive.equals(emissive));
  assert.deepEqual(geometry.attributes.position.array,positions);
  geometry.dispose();material.dispose();texture.dispose();
});
