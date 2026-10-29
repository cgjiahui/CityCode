import * as T from 'three';

// Shared native templates, instanced at the accepted placements. Only nearby,
// visible instances replace the inexpensive silhouettes, after loading succeeds.
export class NativeProps {
  constructor(viewer){this.viewer=viewer;this.loaded=new Map();this.pending=new Set();this.failed=new Set();this.wanted=new Set();this.active=new Set();}
  select(frustum){
    const v=this.viewer,data=v.data.native_props;if(!data)return;
    const candidates=data.instances.map(p=>({p,d:Math.hypot(p.position[0]-v.camera.position.x,p.position[1]-v.camera.position.y)}))
      .filter(({p,d})=>d<240&&Math.abs(v.camera.position.z-v.heightAt(...p.position))<180&&frustum.intersectsSphere(new T.Sphere(new T.Vector3(p.position[0],p.position[1],v.heightAt(...p.position)+4),12)))
      .sort((a,b)=>a.d-b.d).slice(0,v.host.clientWidth<700?24:64);
    this.wanted=new Set(candidates.map(({p})=>p.id));
    for(const entry of this.loaded.values())this.apply(entry);
    const idle=[...this.loaded.entries()].filter(([,e])=>!e.root.visible).sort((a,b)=>a[1].used-b[1].used);
    while(idle.length>(v.host.clientWidth<700?2:4)){const [id,entry]=idle.shift();this.release(entry);this.loaded.delete(id);}
    for(const {p} of candidates){
      if(this.pending.size>=2)break;
      if(!this.loaded.has(p.asset_id)&&!this.pending.has(p.asset_id)&&!this.failed.has(p.asset_id))this.load(p.asset_id);
    }
  }
  apply(entry){
    const v=this.viewer,zero=new T.Matrix4().makeScale(0,0,0);let changed=false,visible=0;
    for(let i=0;i<entry.placements.length;i++){
      const p=entry.placements[i],on=this.wanted.has(p.id);if(on)visible++;
      if(this.active.has(p.id)===on)continue;
      if(on)this.active.add(p.id);else this.active.delete(p.id);
      for(const batch of entry.batches){batch.mesh.setMatrixAt(i,on?batch.matrices[i]:zero);batch.mesh.instanceMatrix.needsUpdate=true;}
      v.propLayer.hide(p.id,on);changed=true;
    }
    entry.root.visible=visible>0;if(visible)entry.used=performance.now();if(changed)v.dirty=true;
  }
  release(entry){
    this.viewer.scene.remove(entry.root);
    const geometry=new Set(),materials=new Set(),textures=new Set();
    entry.root.traverse(node=>{if(!node.isMesh)return;geometry.add(node.geometry);for(const m of Array.isArray(node.material)?node.material:[node.material])materials.add(m);node.dispose?.();});
    for(const m of materials){for(const [key,t] of Object.entries(m))if(t?.isTexture&&key!=='envMap')textures.add(t);m.dispose();}
    geometry.forEach(g=>g.dispose());textures.forEach(t=>t.dispose());
  }
  async load(id){
    const v=this.viewer,record=v.data.native_props.assets[id];this.pending.add(id);
    try{
      const gltf=await v.loadGLTF(record.path,record);if(v.disposed)return;
      const placements=v.data.native_props.instances.filter(p=>p.asset_id===id),root=new T.Group(),batches=[];
      gltf.scene.updateMatrixWorld(true);
      const transforms=placements.map(p=>{
        const scale=Array.isArray(p.scale)?p.scale:[p.scale,p.scale,p.scale];
        const position=new T.Vector3(p.position[0],p.position[1],v.heightAt(...p.position)+.025-record.bounds.min[2]*scale[2]);
        return new T.Matrix4().compose(position,new T.Quaternion().setFromAxisAngle(new T.Vector3(0,0,1),p.yaw*Math.PI/180),new T.Vector3(...scale));
      });
      gltf.scene.traverse(node=>{if(!node.isMesh)return;
        for(const material of Array.isArray(node.material)?node.material:[node.material]){
          material.envMap=v.materialEnvironment.texture;material.envMapIntensity=.8;
          // An opaque fallback for windows is not used: native opacity/glass
          // stays intact, and the complete multi-part car is loaded as a unit.
          for(const key of ['map','normalMap','roughnessMap','metalnessMap','emissiveMap'])if(material[key])material[key].anisotropy=Math.min(8,v.renderer.capabilities.getMaxAnisotropy());
        }
        const mesh=new T.InstancedMesh(node.geometry,node.material,placements.length),matrices=transforms.map(t=>t.clone().multiply(node.matrixWorld));
        matrices.forEach((m,i)=>mesh.setMatrixAt(i,m));mesh.computeBoundingSphere();
        for(let i=0;i<matrices.length;i++)mesh.setMatrixAt(i,new T.Matrix4().makeScale(0,0,0));
        mesh.instanceMatrix.needsUpdate=true;root.add(mesh);batches.push({mesh,matrices});
      });
      if(!batches.length)throw Error('Native street template has no visible geometry');
      const entry={root,batches,placements,used:performance.now()};this.loaded.set(id,entry);v.scene.add(root);this.apply(entry);v.dirty=true;
    }catch(error){this.failed.add(id);console.warn('Street detail unavailable; keeping base geometry',id,error.message);}
    finally{this.pending.delete(id);}
  }
}
