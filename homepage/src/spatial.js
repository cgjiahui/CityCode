// Shared, renderer-independent navigation rules. Coordinates are metres, Z up.
export function inRing(x,y,ring) {
  let inside=false;
  for(let i=0,j=ring.length-1;i<ring.length;j=i++) {
    const [xi,yi]=ring[i],[xj,yj]=ring[j];
    if((yi>y)!==(yj>y)&&x<(xj-xi)*(y-yi)/(yj-yi)+xi)inside=!inside;
  }
  return inside;
}
export function inGeometry(x,y,g) {
  const polys=g.type==='Polygon'?[g.coordinates]:g.type==='MultiPolygon'?g.coordinates:[];
  return polys.some(p=>inRing(x,y,p[0])&&!p.slice(1).some(r=>inRing(x,y,r)));
}
export function collisionIndex(instances,land) {
  const cells=new Map(),size=64;
  for(const a of instances) {
    const [lo,hi]=a.bounds;
    for(let x=Math.floor(lo[0]/size);x<=Math.floor(hi[0]/size);x++)for(let y=Math.floor(lo[1]/size);y<=Math.floor(hi[1]/size);y++) {
      const key=`${x},${y}`;if(!cells.has(key))cells.set(key,[]);cells.get(key).push(a);
    }
  }
  return (x,y,z=1.7)=>{
    if(!inGeometry(x,y,land))return true;
    const nearby=cells.get(`${Math.floor(x/size)},${Math.floor(y/size)}`)||[];
    return nearby.some(a=>z>a.bounds[0][2]-.3&&z<a.bounds[1][2]+.3&&inGeometry(x,y,a.footprint));
  };
}
export function slideMove(position,delta,blocked) {
  const p=[...position],distance=Math.hypot(delta[0],delta[1]);
  const n=Math.max(1,Math.ceil(distance/.25));
  for(let i=0;i<n;i++)for(let axis=0;axis<2;axis++) {
    const q=[...p];q[axis]+=delta[axis]/n;
    const r=.35;
    if(![[r,0],[-r,0],[0,r],[0,-r],[0,0]].some(([x,y])=>blocked(q[0]+x,q[1]+y,q[2])))p[axis]=q[axis];
  }
  return p;
}
export function nearestWalkPoint(points,target,blocked) {
  return points.filter(p=>!blocked(p[0],p[1],1.7)).sort((a,b)=>(a[0]-target[0])**2+(a[1]-target[1])**2-((b[0]-target[0])**2+(b[1]-target[1])**2))[0]||null;
}
export function chooseDetails(instances,camera,visible,maxNear=4,maxMedium=16,budget={}) {
  const candidates=instances.filter(a=>visible(a)).map(a=>{
    const [lo,hi]=a.bounds;
    const distance=Math.hypot(...camera.map((v,i)=>Math.max(lo[i]-v,0,v-hi[i])));
    const pixels=budget.projectedPixels?.(a);
    return {a,distance,pixels};
  }).filter(v=>v.pixels===undefined?v.distance<500:v.pixels>=(budget.minMediumPixels??70))
    .sort((a,b)=>a.pixels===undefined?a.distance-b.distance:b.pixels-a.pixels);
  const result=new Map();let near=0,medium=0,faces=0;
  for(const {a,distance,pixels} of candidates) {
    const cost=level=>budget.faceCount?.(a,level)||0;
    const close=pixels===undefined?distance<135:pixels>=(budget.nearPixels??220);
    if(close&&near<maxNear&&faces+cost('near')<=(budget.maxFaces??Infinity)){result.set(a.id,'near');near++;faces+=cost('near');}
    else if(medium<maxMedium&&faces+cost('medium')<=(budget.maxFaces??Infinity)){result.set(a.id,'medium');medium++;faces+=cost('medium');}
  }
  return result;
}

export function projectedSpan(bounds,camera,verticalFovRadians,viewportHeight) {
  const [lo,hi]=bounds,extent=hi.map((v,i)=>v-lo[i]),center=hi.map((v,i)=>(v+lo[i])/2);
  const distance=Math.max(1,Math.hypot(...center.map((v,i)=>v-camera[i]))-Math.hypot(...extent)*.25);
  return Math.max(...extent)*viewportHeight/(2*Math.tan(verticalFovRadians/2)*distance);
}
