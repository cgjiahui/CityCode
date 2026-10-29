import React,{memo,useEffect,useState} from 'react';

// Independent state keeps 79 tiles out of the frequent scene-progress updates.
export const ObservationMap=memo(function ObservationMap(){
  const [data,setData]=useState(null),[expanded,setExpanded]=useState(false),[selected,setSelected]=useState(null);
  useEffect(()=>{
    const controller=new AbortController();
    fetch(new URL('observations/manifest.json',document.baseURI),{signal:controller.signal})
      .then(r=>{if(!r.ok)throw Error('Input images are temporarily unavailable');return r.json();}).then(setData)
      .catch(e=>{if(e.name!=='AbortError')console.warn(e.message);});
    return()=>controller.abort();
  },[]);
  useEffect(()=>{if(!expanded)return;const close=e=>{if(e.key==='Escape')setExpanded(false);};window.addEventListener('keydown',close);return()=>window.removeEventListener('keydown',close);},[expanded]);
  if(!data)return null;
  const atlas=new URL('observations/'+data.atlas,document.baseURI).href;
  const sprite=patch=>({backgroundImage:`url("${atlas}")`,backgroundSize:`${data.columns*100}% ${data.rows*100}%`,backgroundPosition:`${patch.column/(data.columns-1)*100}% ${patch.row/(data.rows-1)*100}%`});
  const active=selected??data.patches[Math.floor(data.count/2)];
  const extent=data.patch_size_m.map(value=>Math.round(value/100)*100).join(' × ');
  const imageSize=data.input_image_px.join(' × ');
  return <aside className={`observation-map${expanded?' expanded':''}`} aria-label="Low-altitude visual input patches">
    <div className="observation-heading"><div><span className="observation-eyebrow">LOW-ALTITUDE VISUAL INPUT</span><strong>One image per region <span>{data.count} RGB patches</span></strong></div>
      <button onClick={()=>setExpanded(!expanded)} aria-label={expanded?'Collapse input collage':'Expand input collage'} aria-expanded={expanded}>{expanded?'−':'↗'}</button></div>
    <div className="observation-stage">
      <div className="observation-board" style={{aspectRatio:`${data.columns} / ${data.rows}`}}>
        {data.patches.map((patch,i)=><button key={patch.id} className={`observation-tile${selected?.id===patch.id?' selected':''}`}
          aria-label={`View input ${patch.id}`} title={patch.id} onClick={()=>{setSelected(patch);setExpanded(true);}}
          style={{...sprite(patch),width:`${88/data.columns}%`,height:`${88/data.rows}%`,left:`${(patch.column+.06)/data.columns*100}%`,top:`${(patch.row+.06)/data.rows*100}%`,
            '--turn':`${((i*7)%9-4)*.55}deg`,'--lift':`${(i*13)%7}px`}}/>) }
      </div>
    </div>
    {expanded&&<div className="observation-detail"><div className="observation-preview" role="img" aria-label={`Input thumbnail ${active.id}`} style={sprite(active)}/><div><strong>{active.id.replace('SC_P320_','')}</strong><p>One low-altitude RGB image<br/>{imageSize} px<br/>Altitude {Math.round(active.height_m)} m</p><span>Select a patch to inspect</span></div></div>}
    <p className="observation-caption"><span className="observation-footprint">≈{extent} m per patch</span>{expanded?'Overlapping RGB patches · One image per region.':'From visual input to local 3D.'}</p>
  </aside>;
});
