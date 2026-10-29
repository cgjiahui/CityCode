import * as T from 'three';
import {mergeGeometries} from 'three/addons/utils/BufferGeometryUtils.js';

// Original browser geometry. No City Sample meshes, textures or materials.
export function addProps(scene,data,heightAt) {
  const groups=new Map(), matrix=new T.Matrix4(), temp=new T.Object3D(),slots=new Map(),hidden=new Set();
  const material=c=>new T.MeshStandardMaterial({color:c,roughness:.86,metalness:0});
  const mats={paint:material('#9eaaa9'),dark:material('#26363c'),metal:material('#697576'),wood:material('#7c6956'),leaf:material('#657865'),light:material('#d9d3b6'),road:material('#858981'),concrete:material('#aeb0a5')};
  const box=(x,y,z)=>new T.BoxGeometry(x,y,z);
  const cyl=(r,h)=>new T.CylinderGeometry(r,r,h,7).rotateX(Math.PI/2);
  const sphere=r=>new T.IcosahedronGeometry(r,1);
  function part(key,geo,mat,group,offset=[0,0,0],scale=[1,1,1]) {
    if(!groups.has(key))groups.set(key,{geo,mat,transforms:[],ids:[]});else geo.dispose();
    const s=group.scale instanceof Array?group.scale:[group.scale||1,group.scale||1,group.scale||1];
    temp.position.set(...group.position);temp.rotation.set(0,0,group.yaw*Math.PI/180);temp.scale.set(...s);temp.updateMatrix();
    matrix.makeTranslation(...offset).scale(new T.Vector3(...scale));
    groups.get(key).transforms.push(temp.matrix.clone().multiply(matrix));
    groups.get(key).ids.push(group.id);
  }
  for(const original of data.props) {
    const g={...original,position:[original.position[0],original.position[1],heightAt(original.position[0],original.position[1])+.02]};
    switch(g.category) {
      case 'parked_car':case 'car':
        part('car-body',box(4.4,1.85,.62),mats.paint,g,[0,0,.65]);
        part('car-cabin',box(2.5,1.66,.65),mats.dark,g,[-.15,0,1.22]);
        part('car-roof',box(2.2,1.6,.08),mats.paint,g,[-.15,0,1.58]);
        for(const x of [-1.3,1.3])for(const y of [-.9,.9])part('car-wheel',new T.CylinderGeometry(.36,.36,.18,10),mats.dark,g,[x,y,.36]);
        break;
      case 'street_lamp':case 'street_light':case 'streetlight':case 'lamp':
        part('lamp-pole',cyl(.1,7.5),mats.metal,g,[0,0,3.75]);part('lamp-arm',box(2,.1,.1),mats.metal,g,[1,0,7.4]);part('lamp-head',box(.85,.3,.15),mats.light,g,[2,0,7.35]);break;
      case 'traffic_signal':case 'traffic_light':case 'signal':
        part('signal-pole',cyl(.09,4.2),mats.metal,g,[0,0,2.1]);part('signal-box',box(.3,.3,.9),mats.dark,g,[0,.15,3.8]);part('signal-lens',sphere(.1),mats.light,g,[0,.32,4]);break;
      case 'street_tree':case 'tree':
        part('tree-trunk',cyl(.18,3.8),mats.wood,g,[0,0,1.9]);part('tree-leaf',sphere(2),mats.leaf,g,[0,0,4.7],[1,1,1.35]);break;
      case 'bench':
        part('bench-seat',box(1.8,.55,.12),mats.wood,g,[0,0,.48]);part('bench-back',box(1.8,.12,.52),mats.wood,g,[0,.25,.8]);for(const x of [-.65,.65])part('bench-leg',box(.1,.45,.45),mats.dark,g,[x,0,.225]);break;
      case 'bin':case 'trash_bin':
        part('bin',cyl(.3,.85),mats.metal,g,[0,0,.425]);break;
    }
  }
  for(const {geo,mat,transforms,ids} of groups.values()) {
    const mesh=new T.InstancedMesh(geo,mat,transforms.length);transforms.forEach((m,i)=>{
      mesh.setMatrixAt(i,m);if(!slots.has(ids[i]))slots.set(ids[i],[]);slots.get(ids[i]).push({mesh,index:i,matrix:m});
    });mesh.computeBoundingSphere();scene.add(mesh);
  }
  const empty=new T.Matrix4().makeScale(0,0,0);
  return {draws:groups.size,hidden,hide(id,value){
    if(hidden.has(id)===value)return;
    if(value)hidden.add(id);else hidden.delete(id);
    for(const slot of slots.get(id)||[]){slot.mesh.setMatrixAt(slot.index,value?empty:slot.matrix);slot.mesh.instanceMatrix.needsUpdate=true;}
  }};
}

export function addBridges(scene,data) {
  const road=[],rails=[],piers=[],bodies=[],walkable=[];
  const canvas=document.createElement('canvas');canvas.width=canvas.height=256;const ctx=canvas.getContext('2d');
  ctx.fillStyle='#737775';ctx.fillRect(0,0,256,256);
  let seed=17;for(let i=0;i<12000;i++){seed=(Math.imul(seed,1664525)+1013904223)>>>0;const x=seed%256;seed=(Math.imul(seed,1664525)+1013904223)>>>0;const y=seed%256;ctx.fillStyle=i%2?'#ffffff08':'#00000008';ctx.fillRect(x,y,1,1);}
  ctx.fillStyle='#d9d9ce';ctx.fillRect(6,0,2,256);ctx.fillRect(248,0,2,256);for(const x of [64,192])ctx.fillRect(x,0,2,100);
  ctx.fillStyle='#d6bb73';ctx.fillRect(126,0,1,256);ctx.fillRect(130,0,1,256);
  const asphalt=new T.CanvasTexture(canvas);asphalt.colorSpace=T.SRGBColorSpace;asphalt.wrapS=asphalt.wrapT=T.RepeatWrapping;asphalt.anisotropy=4;
  for(const segment of data.bridges) {
    const p=segment.profile;if(p.length<2)continue;
    const positions=[],uv=[],indices=[];let station=0;
    for(let i=0;i<p.length;i++) {
      const v=p[i],prev=p[Math.max(0,i-1)].p,next=p[Math.min(p.length-1,i+1)].p;
      const dx=next[0]-prev[0],dy=next[1]-prev[1],len=Math.hypot(dx,dy)||1;
      if(i)station+=Math.hypot(...v.p.map((x,j)=>x-p[i-1].p[j]));
      for(const side of [-1,1]) {positions.push(v.p[0]-dy/len*v.w/2*side,v.p[1]+dx/len*v.w/2*side,v.p[2]+.015);uv.push(side===1?1:0,station/12);}
      if(i) {const n=2*i;indices.push(n-2,n,n-1,n-1,n,n+1);}
      if(i) {
        for(const side of [-1,1]) {
          const n=2*i+(side===1?1:0),a=new T.Vector3().fromArray(positions,(n-2)*3),b=new T.Vector3().fromArray(positions,n*3);
          const direction=b.clone().sub(a),g=new T.BoxGeometry(.2,.45,direction.length());
          g.applyQuaternion(new T.Quaternion().setFromUnitVectors(new T.Vector3(0,0,1),direction.normalize()));
          g.translate(...a.add(b).multiplyScalar(.5).add(new T.Vector3(0,0,.4)).toArray());rails.push(g);
        }
      }
    }
    const g=new T.BufferGeometry();g.setAttribute('position',new T.Float32BufferAttribute(positions,3));g.setAttribute('uv',new T.Float32BufferAttribute(uv,2));g.setIndex(indices);g.computeVertexNormals();road.push(g);
    const lower=positions.map((v,i)=>i%3===2?v-1.6:v),verts=[...positions,...lower],offset=positions.length/3,faces=[];
    for(let i=0;i<indices.length;i+=3)faces.push(indices[i]+offset,indices[i+2]+offset,indices[i+1]+offset);
    for(let i=2;i<offset;i+=2)for(const side of [0,1]){const a=i-2+side,b=i+side;faces.push(a,b,a+offset,b,b+offset,a+offset);}
    for(const i of [0,offset-2])faces.push(i,i+1,i+offset,i+1,i+offset+1,i+offset);
    const body=new T.BufferGeometry();body.setAttribute('position',new T.Float32BufferAttribute(verts,3));body.setIndex(faces);body.computeVertexNormals();bodies.push(body);
  }
  if(road.length) {
    const mesh=new T.Mesh(mergeGeometries(road),new T.MeshStandardMaterial({map:asphalt,roughness:1,side:T.DoubleSide}));scene.add(mesh);walkable.push(mesh);road.forEach(g=>g.dispose());
  }
  for(const s of data.supports) {
    const g=new T.BoxGeometry(1.3,1.7,Math.max(.1,s.height));g.rotateZ(s.yaw*Math.PI/180);g.translate(s.position[0],s.position[1],s.position[2]+s.height/2);piers.push(g);
  }
  for(const list of [rails,piers,bodies])if(list.length){scene.add(new T.Mesh(mergeGeometries(list),new T.MeshStandardMaterial({color:'#a7aaa2',roughness:1,side:T.DoubleSide})));list.forEach(g=>g.dispose());}
  return walkable;
}
