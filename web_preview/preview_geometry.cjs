// Preview-only derived geometry. Original nodes, vertex attributes and placements remain intact.
const fs = require('fs');
const path = require('path');
const simplifier = require(path.join(__dirname, '../src/citycode_assets/vendor/meshoptimizer_v025/meshopt_simplifier.js'));
const widths = { SCALAR: 1, VEC2: 2, VEC3: 3, VEC4: 4, MAT4: 16 };
const sizes = { 5121: 1, 5123: 2, 5125: 4, 5126: 4 };
function parse(input) {
  const buffer = fs.readFileSync(input);
  const length = buffer.readUInt32LE(12);
  return { buffer, doc: JSON.parse(buffer.subarray(20, 20+length)), offset: 28+length };
}
function read(scene, index) {
  const a = scene.doc.accessors[index], v = scene.doc.bufferViews[a.bufferView];
  if (a.sparse || !sizes[a.componentType]) throw new Error('Unsupported preview accessor');
  const width = widths[a.type], step = v.byteStride || width*sizes[a.componentType];
  const start = scene.offset+(v.byteOffset||0)+(a.byteOffset||0);
  const result = new Float32Array(a.count*width);
  for (let i=0;i<a.count;i++) for(let j=0;j<width;j++) {
    const offset=start+i*step+j*sizes[a.componentType];
    result[i*width+j] = a.componentType===5126 ? scene.buffer.readFloatLE(offset) :
      a.componentType===5125 ? scene.buffer.readUInt32LE(offset) :
      a.componentType===5123 ? scene.buffer.readUInt16LE(offset) : scene.buffer.readUInt8(offset);
  }
  return { values: result, width };
}
(async()=>{
  await simplifier.ready;
  const scene=parse(process.argv[2]), records=[], chunks=[]; let byteOffset=0;
  const blob=array=>{const data=Buffer.from(array.buffer,array.byteOffset,array.byteLength); const offset=byteOffset; chunks.push(data); byteOffset+=data.length; return {offset,bytes:data.length};};
  let totalFaces=0;
  for(const mesh of scene.doc.meshes) for(const primitive of mesh.primitives) totalFaces+=scene.doc.accessors[primitive.indices].count/3;
  for(let m=0;m<scene.doc.meshes.length;m++) for(let p=0;p<scene.doc.meshes[m].primitives.length;p++) {
    const primitive=scene.doc.meshes[m].primitives[p];
    if(primitive.mode && primitive.mode!==4 || primitive.targets || scene.doc.skins?.length || scene.doc.animations?.length) throw new Error('Only static triangles supported');
    const position=read(scene,primitive.attributes.POSITION).values;
    const index=Uint32Array.from(read(scene,primitive.indices).values), count=position.length/3;
    const arrays=Object.fromEntries(Object.entries(primitive.attributes).map(([name,a])=>[name,read(scene,a)]));
    const attributes=new Float32Array(count*5), locks=new Uint8Array(count);
    for(let i=0;i<count;i++) {
      for(let j=0;j<3;j++) attributes[i*5+j]=arrays.NORMAL?.values[i*3+j]||0;
      for(let j=0;j<2;j++) attributes[i*5+3+j]=arrays.TEXCOORD_0?.values[i*2+j]||0;
    }
    for(let axis=0;axis<3;axis++) {
      let lo=0,hi=0;
      for(let i=1;i<count;i++){if(position[i*3+axis]<position[lo*3+axis])lo=i;if(position[i*3+axis]>position[hi*3+axis])hi=i;}
      locks[lo]=locks[hi]=1;
    }
    const target=Math.min(index.length,Math.max(144,Math.floor(2500*(index.length/3)/totalFaces)*3));
    let [indices,error]=simplifier.simplifyWithAttributes(index,position,3,attributes,5,[.002,.002,.002,.04,.04],locks,target,.06,['Permissive']);
    const used=[...new Set(indices)].sort((a,b)=>a-b), remap=new Map(used.map((old,n)=>[old,n]));
    const outAttributes={};
    for(const [name,attr] of Object.entries(arrays)) {
      const values=new Float32Array(used.length*attr.width);
      used.forEach((old,n)=>{for(let j=0;j<attr.width;j++)values[n*attr.width+j]=attr.values[old*attr.width+j];});
      const min=Array(attr.width).fill(Infinity),max=Array(attr.width).fill(-Infinity);
      for(let i=0;i<values.length;i++){const j=i%attr.width;min[j]=Math.min(min[j],values[i]);max[j]=Math.max(max[j],values[i]);}
      outAttributes[name]={...blob(values),count:used.length,width:attr.width,min,max};
    }
    records.push({mesh:m,primitive:p,attributes:outAttributes,index:{...blob(Uint32Array.from(indices,i=>remap.get(i))),count:indices.length},source_faces:index.length/3,faces:indices.length/3,error});
  }
  fs.writeFileSync(process.argv[3]+'.bin',Buffer.concat(chunks));
  fs.writeFileSync(process.argv[3]+'.json',JSON.stringify(records));
})().catch(error=>{console.error(error.message);process.exitCode=1;});
