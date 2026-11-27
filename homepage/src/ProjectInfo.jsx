import {useEffect,useRef} from 'react';

export const repository=(import.meta.env.VITE_CITYCODE_REPOSITORY_URL??'https://github.com/cgjiahui/CityCode').trim();
export const paper=(import.meta.env.VITE_CITYCODE_PAPER_URL??'https://arxiv.org/abs/2610.18342').trim();
export const repositoryPrivate=import.meta.env.VITE_CITYCODE_REPOSITORY_PRIVATE!=='false';

export function ProjectInfo({liteScene=false,onClose,lightweight=false}) {
  const dialog=useRef(null);
  useEffect(()=>{dialog.current.showModal();},[]);
  return <dialog ref={dialog} className="project-dialog" aria-labelledby="project-title" onClose={onClose}
    onClick={event=>{if(event.target===dialog.current)dialog.current.close();}}>
    <article className="project-content">
      <div className="project-heading"><span>THE PROJECT</span><button aria-label="Close project information" onClick={()=>dialog.current.close()}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true"><path d="m6 6 12 12M18 6 6 18"/></svg></button></div>
      <h2 id="project-title">CityCode<span>.</span></h2>
      <p className="project-intro">From one image per region<br/>to a city you can explore.</p>
      <p>Starting from one low-altitude RGB image per region, CityCode recovers building structure, prepares textured building assets and a city background, and assembles them into a scene in Unreal Engine.</p>
      <ol className="project-workflow">
        <li><span>01</span><div><strong>Recover structure</strong><p>Locate buildings and recover their shape and height.</p></div></li>
        <li><span>02</span><div><strong>Prepare assets</strong><p>Associate or generate textured building models.</p></div></li>
        <li><span>03</span><div><strong>Build the background</strong><p>Prepare terrain, water and the road network.</p></div></li>
        <li><span>04</span><div><strong>Assemble the city</strong><p>Place buildings and street objects in a shared scene.</p></div></li>
      </ol>
      <section className="project-demo-note"><h3>Inside this demo</h3><p>Explore the generated city from above, or enter Street walk to experience it at ground level. The visual input consists of 79 overlapping patches, each covering approximately 400 × 400 m and represented by one RGB image.</p>{lightweight&&<p>This web preview uses reduced geometry and compressed textures while retaining the full city layout, ground materials and street objects. View the full visual quality in Unreal Engine.</p>}{liteScene?<p>The lightweight scene preserves the generated buildings and uses a baked ground overview and simplified street objects.</p>:<p>This showcase combines generated buildings and recovered terrain with City Sample ground materials and native assets. It is a presentation of the assembled scene; the downloadable reconstruction has its own resource inventory.</p>}<small>City Sample content © Epic Games. Respective asset licenses apply.</small></section>
      <footer className="project-links">{paper&&<a href={paper} target="_blank" rel="noopener noreferrer">Read the paper ↗</a>}{repository&&<a href={repository} target="_blank" rel="noopener noreferrer">GitHub ↗</a>}<button onClick={()=>dialog.current.close()}>Back to the city</button></footer>
      {repository&&repositoryPrivate&&<p className="repository-access">Repository access is required while the release is being prepared.</p>}
    </article>
  </dialog>;
}
