import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { CityScene } from './scene.js';
import { PART_COLORS, massingSourceFor, initialSceneSettings, completeCitySettings, comparisonSettings } from './massing.js';
import { COLLISION_KINDS, COLLISION_COLORS, RELATED_COLORS, collisionCases, filterCollisionCases,
  initialCollisionSettings, initialCollisionCase, collisionStatus, collisionMetrics } from './collisions.js';
import './styles.css';

function Icon({ name, size = 19, ...props }) {
  const paths = {
    cube: <><path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="M3 8v9l9 5 9-5V8M12 13v9"/></>,
    layers: <><path d="m12 3 10 6-10 6L2 9 10 3ZM2 13l10 6 10-6M2 17l10 6 10-6"/></>,
    home: <><path d="m3 10 9-8 9 8M5 8v13h14V8M9 21v-8h6v8"/></>,
    search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="m16 16 5 5"/></>,
    expand: <><path d="M14 3h7v7M21 3l-7 7M10 21H3v-7M3 21l7-7"/></>,
    mouse: <><rect x="5" y="2" width="14" height="20" rx="7"/><path d="M12 2v8M5 10h14"/></>,
    plus: <path d="M12 4v16M4 12h16"/>, minus: <path d="M4 12h16"/>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 10v7M12 6v1"/></>,
    surface: <><path d="m4 20 8-17 8 17H4ZM12 8v7M12 17v1"/></>,
    close: <path d="m5 5 14 14M19 5 5 19"/>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" {...props}>{paths[name] || paths.cube}</svg>;
}

const formatSize = bytes => bytes >= 1e9 ? `${(bytes/1e9).toFixed(2)} GB` : `${Math.round(bytes/1e6)} MB`;

function SceneControls({ data, settings, setSettings, stats, onMode, onElevated, onCoast }) {
  const totals = useMemo(() => {
    const buildings = data.assets.filter(a => !a.surface), surfaces = data.assets.filter(a => a.surface);
    return { bytes: buildings.reduce((sum,a) => sum+a.lods['0.1M'].bytes, 0),
      fineBytes: buildings.reduce((sum,a) => sum+(a.lods['0.25M'] || a.lods['0.1M']).bytes, 0),
      previewBytes: buildings.reduce((sum,a) => sum+(a.lods.preview || a.lods['0.1M']).bytes, 0),
      accepted: surfaces.filter(a=>a.enabled).length, references: surfaces.filter(a=>!a.enabled).length };
  }, [data]);
  const update = (key, value) => setSettings(previous => ({...previous,[key]:value}));
  const city = settings.view === 'city', isolated = settings.view === 'single', neural = settings.backgroundMode !== 'surface', coastal = settings.backgroundMode === 'coastal';
  const hasMapSolid = Boolean(data.background?.files.map_solid);
  const solidMap = !neural && settings.backgroundMap && settings.mapSolid && hasMapSolid;
  return <section className="scene-controls" aria-label="City loading settings">
    <div className="scene-control-title"><h2>City layers</h2><span>{data.counts.assets} models</span></div>
    <div className="mode-switch" role="group" aria-label="View extent">{[['city','City'],['neighborhood','Nearby'],['single','Single']].map(([mode,label])=><button key={mode} className={settings.view===mode?'active':''} onClick={()=>onMode(mode)}>{label}</button>)}</div>
    {data.background ? <div className="background-controls" aria-label="Terrain and elevated roads">
      <div className="scene-control-title"><h2>Terrain and elevated roads</h2><span>Historical terrain</span></div>
      <label className="quality-select">Terrain mode<select aria-label="Terrain mode" value={settings.backgroundMode} disabled={isolated} onChange={e=>update('backgroundMode',e.target.value)}><option value="surface">Map and height surface</option>{data.background.files.coastal ? <option value="coastal">Coastal land and water</option> : null}{data.background.files.neural ? <option value="pixal3d">Original Pixal3D comparison</option> : null}</select></label>
      <Toggle checked={settings.backgroundMap && !isolated && !neural} onChange={v=>update('backgroundMap',v)} disabled={isolated || neural} icon="layers">Fused RGB map</Toggle>
      {hasMapSolid ? <Toggle checked={solidMap && !isolated} onChange={v=>update('mapSolid',v)} disabled={isolated || neural || !settings.backgroundMap} icon="cube">Extruded map land</Toggle> : null}
      <Toggle checked={settings.backgroundGround && !isolated && !neural && !solidMap} onChange={v=>update('backgroundGround',v)} disabled={isolated || neural || solidMap} icon="surface">Ground height surface</Toggle>
      {solidMap && !isolated ? <p className="global-help">Turn off extruded map land to inspect the original ground height surface separately.</p> : null}
      <Toggle checked={settings.backgroundElevated && !isolated} onChange={v=>update('backgroundElevated',v)} disabled={isolated} icon="layers">Elevated road surface</Toggle>
      <label className="quality-select">Elevated road style<select aria-label="Elevated road style" value={settings.elevatedStyle} onChange={e=>update('elevatedStyle',e.target.value)}><option value="highlight">Orange highlight</option><option value="texture">Map texture</option></select></label>
      <button className="focus-elevated" onClick={onElevated}>Focus elevated roads</button>
      {coastal || solidMap ? <Toggle checked={settings.backgroundWater && !isolated} onChange={v=>update('backgroundWater',v)} disabled={isolated} icon="layers">Water surface</Toggle> : null}
      {coastal ? <button className="focus-elevated" disabled={isolated} onClick={onCoast}>Inspect coastline thickness</button> : null}
      {solidMap ? <button className="focus-elevated" disabled={isolated} onClick={onCoast}>Inspect map coastline thickness</button> : null}
      <p className="global-help">{coastal ? 'Land and docks are clipped to visible ground-truth heights. The sides extend to a base at −8 m. Hide the water at −3.5 m to inspect the full thickness. Building positions are preserved.' : neural ? 'Lightweight preview of the historical Pixal3D result, including generated detail and continuous ground. Hide the elevated overlay to inspect it. The data package contains the full model.' : solidMap ? 'The RGB map has separate land and water layers. Land is at 0 m, water at −3.5 m and the base at −8 m. Extrusion affects only the display.' : 'The flat RGB map can be overlaid with ground height and elevated surfaces. This layer omits bridge piers and undersides.'}</p>
      <div className="background-progress" role="status" data-map-solid-loaded={String(stats.background?.mapSolidLoaded||false)} data-map-solid-visible={String(stats.background?.mapSolidVisible||false)} data-map-water-visible={String(stats.background?.mapWaterVisible||false)} data-coastal-loaded={String(stats.background?.coastalLoaded||false)} data-coastal-visible={String(stats.background?.coastalVisible||false)} data-water-visible={String(stats.background?.waterVisible||false)} data-mode={settings.backgroundMode} data-neural-loaded={String(stats.background?.neuralLoaded||false)} data-neural-visible={String(stats.background?.neuralVisible||false)} data-map-loaded={String(stats.background?.mapLoaded||false)} data-surfaces-loaded={String(stats.background?.surfacesLoaded||false)} data-map-visible={String(stats.background?.mapVisible||false)} data-ground-visible={String(stats.background?.groundVisible||false)} data-elevated-visible={String(stats.background?.elevatedVisible||false)} data-style={stats.background?.elevatedStyle||''}>
        {stats.background?.errors?.length ? stats.background.errors.join('; ') : stats.background?.pending ? 'Loading terrain…' : coastal && stats.background?.coastalLoaded ? `Coastal land ready · ${stats.background.coastalFaces?.toLocaleString()} faces` : neural && stats.background?.neuralLoaded ? `Pixal3D ready · ${stats.background.neuralFaces?.toLocaleString()} faces` : solidMap && stats.background?.mapSolidLoaded ? `Extruded map ready · ${stats.background.mapSolidFaces?.toLocaleString()} faces` : stats.background?.surfacesLoaded ? `Terrain ready · Elevated surface ${stats.background?.elevatedFaces?.toLocaleString() || '—'} faces` : 'Terrain loads by layer'}
      </div>
    </div> : <p className="global-help">This dataset has no terrain preview.</p>}
    <label className="quality-select">City model detail<select aria-label="City model detail" value={settings.cityQuality} onChange={e=>update('cityQuality',e.target.value)}><option value="preview">Lightweight · {formatSize(totals.previewBytes)}</option><option value="0.1M">0.1M · Original mesh · {formatSize(totals.bytes)}</option><option value="0.25M">0.25M · Detailed mesh · {formatSize(totals.fineBytes)}</option></select></label>
    <label className="quality-select">Surface material<select aria-label="Surface material" value={settings.surfaceDisplay} onChange={e=>update('surfaceDisplay',e.target.value)}><option value="neutral">Neutral matte (default)</option><option value="original">Original with reflections</option></select></label>
    <p className="global-help">{settings.surfaceDisplay==='original' ? 'Compare the original reflections here. Lightweight copies use simplified materials.' : 'Reduced reflections make shapes and colors easier to inspect. This affects only the viewer.'}</p>
    <Toggle checked={settings.cityBuildings} onChange={v=>update('cityBuildings',v)} disabled={!city} icon="cube">City buildings · {data.counts.assets}</Toggle>
    {data.counts.massing_parts ? <Toggle checked={settings.massing} onChange={v=>update('massing',v)} icon="cube">Massing additions · {data.counts.massing_parts}</Toggle> : null}
    <Toggle checked={settings.acceptedSurfaces && !isolated} onChange={v=>update('acceptedSurfaces',v)} disabled={isolated} icon="surface">Accepted auxiliary surfaces · {totals.accepted}</Toggle>
    <Toggle checked={settings.surfaces && !isolated} onChange={v=>update('surfaces',v)} disabled={isolated} icon="surface">Original observation references · {totals.references}</Toggle>
    <p className="global-help">{isolated ? 'Single-building mode hides observation surfaces.' : city ? 'The three city layers load independently.' : 'Neighborhood mode loads observation surfaces within the selected radius.'}{settings.surfaces&&!isolated ? ' Observation references can be fragmented, layered or floating. They are shown separately from generated buildings.' : ''}</p>
    {settings.cityQuality!=='preview' ? <p className="quality-help">Load all {settings.cityQuality} meshes at approximately {settings.cityQuality==='0.25M'?'250,000':'100,000'} triangles per model. The browser limits textures to 512 px. Pause loading or turn off buildings to free memory.</p> : null}
    <div className="global-actions"><button className="load-city-button" onClick={()=>onMode('city')}>Load entire city</button><button className="pause-city-button" disabled={!city || (!stats.cityPaused && stats.cityTotalLoaded===stats.cityTotalDesired)} onClick={()=>update('cityPaused',!settings.cityPaused)}>{settings.cityPaused?'Resume':'Pause'}</button></div>
    {city ? <div className="global-progress" role="status" aria-label="City loading progress"><progress max={Math.max(1,stats.cityTotalDesired||0)} value={stats.cityTotalLoaded||0}/><span>Building {stats.cityLoaded||0}/{stats.cityDesired||0} · Auxiliary {stats.acceptedSurfaces?.length||0}/{stats.cityAcceptedDesired||0} · References {stats.rawReferences?.length||0}/{stats.cityReferencesDesired||0}{stats.cityPaused?' · Paused':''}</span></div> : null}
  </section>;
}

function BuildingList({ data, selected, onSelect, open, onClose, inspection, children }) {
  const [query, setQuery] = useState('');
  const [scope, setScope] = useState(() => new URLSearchParams(window.location.search).get('inspect')==='massing' ? 'source_groups' : 'all');
  const assetsById = useMemo(() => new Map(data.assets.map(a=>[a.id,a])), [data]);
  const sourceGroups = data.massing_sources || [];
  const reviewedCount = data.assets.filter(a => a.review_candidate).length;
  const items = useMemo(() => scope==='source_groups' ? sourceGroups
    .filter(source=>!query || source.number.includes(query) || source.id.toLowerCase().includes(query.toLowerCase()))
    .map(source=>({...assetsById.get(source.part_ids[0]),sourceEntry:true,display_number:source.number,partCount:source.part_ids.length})) : data.assets.filter(a => !a.surface &&
    (query ? a.sources.some(id => id.toLowerCase().includes(query.toLowerCase())) : scope === 'all' || (scope === 'massing' ? a.massing : scope === 'review' ? a.review_candidate : a.recent))), [data, assetsById, sourceGroups, query, scope]);
  if (inspection.open && inspection.available) return <aside className={`sidebar collision-sidebar ${open ? 'sidebar-open' : ''}`} aria-label="Collision inspection list">
    <button className="icon-button mobile-only sidebar-close" aria-label="Close building list" onClick={onClose}><Icon name="close"/></button>
    {inspection.panel}
    <details className="collision-scene-settings"><summary>City and layer settings</summary>{children}</details>
  </aside>;
  return <aside className={`sidebar ${open ? 'sidebar-open' : ''}`} aria-label="Building list">
    <button className="icon-button mobile-only sidebar-close" aria-label="Close building list" onClick={onClose}><Icon name="close"/></button>
    <button className="collision-entry" onClick={inspection.onOpen} disabled={!inspection.available} aria-label="Open collision inspection"><Icon name="surface"/><span>Collision inspection<small>{inspection.available ? 'Current results · Select to locate' : 'No inspection results in this dataset'}</small></span></button>
    {sourceGroups.length ? <button className="massing-entry" onClick={()=>{
      setScope('source_groups');setQuery('');onSelect(selected?.massing ? selected.id : sourceGroups[0].part_ids[0]);onClose();
    }}><Icon name="layers"/>View {sourceGroups.length} local sources</button> : null}
    {scope!=='source_groups' ? children : null}
    <div className="rail-title"><h1>{scope==='source_groups' ? 'Local source' : 'Building list'}</h1></div>
    <p className="muted rail-subtitle">{query ? `Found ${items.length} items` : scope==='source_groups' ? `${sourceGroups.length} sources → ${data.counts.massing_parts} parts` : scope === 'all' ? `All ${data.counts.assets} assets` : scope === 'massing' ? `Massing additions ${data.counts.massing_parts} parts` : scope === 'review' ? `Later additions: ${reviewedCount} assets` : '11 recent assets'}</p>
    {scope==='source_groups' && data.massing_pruning?.removed_parts ? <p className="global-help">Removed {data.massing_pruning.removed_parts} individual additions. Gaps are allowed. Groups removed: {(data.massing_pruning.explicit_excluded_source_ids||[]).map(id=>String(Number(id.split('_').at(-1)))).join(', ')}.{data.massing_pruning.empty_after_part_review_source_ids?.length ? <> Sources with no retained parts: {data.massing_pruning.empty_after_part_review_source_ids.map(id=>String(Number(id.split('_').at(-1)))).join(', ')}.</> : null}</p> : null}
    <label className="search"><Icon name="search"/><input value={query} onChange={e => setQuery(e.target.value.trim())} placeholder="Search by building number" aria-label="Search by building number"/></label>
    <select className="list-scope" aria-label="Building list filter" value={scope} onChange={e => { setScope(e.target.value); setQuery(''); }}>{sourceGroups.length ? <option value="source_groups">Local sources · {sourceGroups.length} sources</option> : null}<option value="recent">11 recent assets</option><option value="review">Later additions · {reviewedCount} assets</option>{data.counts.massing_parts ? <option value="massing">Massing additions · {data.counts.massing_parts} parts</option> : null}<option value="all">All building assets · {data.counts.assets} assets</option></select>
    <div className="building-list" role="list">
      {items.map(asset => <div role="listitem" key={asset.id}><button className={`building-row ${(asset.sourceEntry ? selected?.sources.includes(asset.sources[0]) : asset.id === selected?.id) ? 'selected' : ''}`} onClick={() => { onSelect(asset.id); onClose(); }} aria-label={asset.sourceEntry ? `View local source ${asset.number}` : `Select building ${asset.number}`} aria-current={(asset.sourceEntry ? selected?.sources.includes(asset.sources[0]) : asset.id === selected?.id) ? 'true' : undefined}>
        <span>{asset.display_number || asset.number}</span>{asset.sourceEntry ? <span className="row-note">{asset.partCount} parts</span> : asset.massing ? <span className="row-note">Massing addition</span> : asset.group.length > 1 ? <span className="row-note">Connected</span> : null}
      </button></div>)}
      {!items.length ? <p className="empty-list">No matching buildings available for assembly.</p> : null}
    </div>
    <button className="text-button all-buildings" onClick={() => { setScope(v => v === 'all' ? 'recent' : 'all'); setQuery(''); }}>{scope === 'all' ? 'Back to recent assets' : `View all ${data.counts.assets} assets`}</button>
    <div className="legend"><div><i className="swatch proxy"/>Gray: recovered structure</div><div><i className="swatch model"/>Color: building models</div></div>
    {scope==='source_groups' ? <details className="massing-extra"><summary>City and terrain settings</summary>{children}</details> : null}
  </aside>;
}

function Evidence({ asset, onExpand }) {
  const [role, setRole] = useState('front');
  useEffect(() => setRole('front'), [asset.id]);
  const labels = asset.massing ? {front:'After completion',previous:'Before completion',context:'Added parts only',source:'Original observation',back:'Rear view'} : { front: 'Current model', previous: 'Previous model', appearance: 'Completion image', source: 'Original observation', context: 'Neighborhood view', back: 'Rear view' };
  const available = Object.keys(labels).filter(key => asset.evidence[key]);
  const actualRole = asset.evidence[role] ? role : available[0];
  return <div className="evidence">
    {actualRole ? <button className="evidence-image" onClick={() => onExpand({ url: asset.evidence[actualRole], title: `Building ${asset.number} · ${labels[actualRole]}` })} aria-label="Enlarge model reference">
      <img src={asset.evidence[actualRole]} alt={`Building ${asset.number} ${labels[actualRole]}`} loading="lazy"/><span className="image-expand"><Icon name="expand" size={17}/></span>
    </button> : <div className="evidence-empty"><Icon name="cube" size={32}/><span>View model in scene</span></div>}
    {available.length > 1 ? <div className="evidence-tabs" role="group" aria-label="View image evidence">{available.map(key => <button key={key} className={actualRole === key ? 'active' : ''} onClick={() => setRole(key)}>{labels[key]}</button>)}</div> : null}
  </div>;
}

function Toggle({ checked, onChange, icon, disabled = false, children }) {
  return <label className={`toggle-row ${checked ? 'checked' : ''} ${disabled ? 'disabled' : ''}`}><Icon name={icon}/><span>{children}</span><input type="checkbox" checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)}/><i className="toggle-indicator"/></label>;
}

function CollisionPanel({ audit, cases, settings, selectedId, onSelect, onChange, onClose }) {
  const visible = useMemo(() => filterCollisionCases(cases, { kind: settings.collisionKind, priorityOnly: settings.collisionPriorityOnly }), [cases, settings.collisionKind, settings.collisionPriorityOnly]);
  const counts = audit.counts || {}, centerline = counts.vehicle_centerline_obstructions;
  return <section className="collision-panel" aria-label="Collision inspection">
    <div className="collision-heading"><h1>Collision inspection</h1><button onClick={onClose} aria-label="Close collision inspection"><Icon name="close" size={16}/></button></div>
    <p className="collision-subtitle">Physical postprocessing results</p>
    <div className="collision-summary"><strong>{centerline === 0 ? 'No centerline obstruction' : Number.isFinite(centerline) ? `Centerline obstruction ${centerline}` : 'Centerline clearance checked separately'}</strong><span>Minor edge contacts retained after review</span></div>
    <Toggle checked={settings.collisionEnabled} onChange={v => onChange('collisionEnabled', v)} icon="layers">Inspection layer</Toggle>
    <label className="quality-select collision-kind">Inspection type<select aria-label="Collision type" value={settings.collisionKind} onChange={e => onChange('collisionKind', e.target.value)}>
      <option value="all">All types · {cases.length}</option>{Object.entries(COLLISION_KINDS).map(([kind, label]) => <option key={kind} value={kind}>{label} · {cases.filter(c => c.kind === kind).length}</option>)}
    </select></label>
    <div className="collision-scope" role="group" aria-label="Contact filter">{[[true, 'Priority only'], [false, 'All contacts']].map(([priority, label]) => <button key={label} aria-pressed={settings.collisionPriorityOnly === priority} className={settings.collisionPriorityOnly === priority ? 'active' : ''} onClick={() => onChange('collisionPriorityOnly', priority)}>{label}</button>)}</div>
    <p className="collision-help">Priority marks locations for further inspection. Select a city marker to see its sampled contact area.</p>
    <p className="collision-list-count" role="status">{visible.length} {settings.collisionPriorityOnly ? 'priority locations' : 'contact records'}</p>
    <div className="collision-list" role="list" aria-label="Contact locations">{visible.map(item => {
      const status = collisionStatus(item), primary = collisionMetrics(item)[0];
      return <div role="listitem" key={item.id}><button className={`collision-row ${item.id === selectedId ? 'selected' : ''}`} aria-label={`View ${item.label}`} aria-current={item.id === selectedId ? 'true' : undefined} onClick={() => onSelect(item.id)}>
        <i style={{ background: COLLISION_COLORS[item.kind] }}/><span><strong>{item.label}</strong><small>{status.label}{primary ? ` · ${primary.value}` : ''}</small></span><span className="collision-row-arrow">›</span>
      </button></div>;
    })}{!visible.length ? <p className="collision-empty">No {settings.collisionPriorityOnly ? 'priority locations in this category. Select All contacts to see other records.' : 'contacts in this category.'}</p> : null}</div>
    <div className="collision-legend">{Object.entries(COLLISION_KINDS).map(([kind, label]) => <span key={kind}><i style={{ background: COLLISION_COLORS[kind] }}/>{label}</span>)}</div>
    <button className="collision-back" onClick={onClose}>Close inspection and return to buildings</button>
  </section>;
}

function CollisionInspector({ item, data, settings, onSelect, onFocus, onClose, stats }) {
  const status = collisionStatus(item), metrics = collisionMetrics(item), assets = item.asset_ids.map(id => data.assets.find(a => a.id === id)).filter(Boolean);
  return <aside className="inspector collision-inspector" aria-label="Contact details">
    <div className="collision-detail-heading"><span>{COLLISION_KINDS[item.kind]}</span><button className="icon-button" aria-label="Clear contact selection" onClick={onClose}><Icon name="close" size={18}/></button></div>
    <h2>{item.label.split(' · ')[0]}</h2><span className={`collision-status ${status.tone}`}>{status.label}</span>
    <p className="collision-description">{item.description || status.explanation}</p>
    {item.reason ? <p className="collision-reason">{item.reason}</p> : null}
    <div className="collision-asset-list" aria-label="Related buildings">{assets.map((asset, index) => <button key={asset.id} onClick={() => onSelect(asset.id)} aria-label={`View related building ${asset.display_number || asset.number}`}><i style={{ background: RELATED_COLORS[index % RELATED_COLORS.length] }}/><span>Building {asset.display_number || asset.number}</span><small>Inspect alone ›</small></button>)}</div>
    <dl className="collision-metrics">{metrics.map(m => <div key={m.label}><dt>{m.label}</dt><dd>{m.value}</dd></div>)}</dl>
    <button className="detail-button" onClick={() => onFocus(item.id)}><Icon name="home"/>Focus neighborhood</button>
    <p className="collision-key">Blue and orange outlines mark related buildings<br/>Patches: sampled contact area</p>
    <p className="collision-patch-note">{!settings.collisionEnabled ? 'Enable the inspection layer in the sidebar to see markers.' : stats.collisions?.patchCount ? `Showing ${stats.collisions.patchCount} local patches${item.patches_truncated ? ' (sampled extent)' : ''}.` : 'Showing the contact marker.'}{item.kind === 'ground' ? ' This difference comes from terrain detail.' : ''}</p>
    <div className="inspector-note"><Icon name="info"/><span>{status.explanation}<small>Results reflect the current layout. Minor contacts may be retained.</small></span></div>
  </aside>;
}

function MassingInspector({ asset, source, data, settings, setSettings, onHigh, stats, onExpand, onSelect, onTop }) {
  const parts = source.part_ids.map(id=>data.assets.find(a=>a.id===id));
  const mode = !settings.massing ? 'before' : settings.single ? 'parts' : 'after';
  const update = (key,value)=>setSettings(previous=>({...previous,[key]:value}));
  const evidence = {...asset,evidence:{front:source.evidence.after,previous:source.evidence.before,
    context:source.evidence.parts,source:source.evidence.source}};
  return <aside className="inspector massing-inspector" aria-label="Local source details">
    <h2>Local source <span>{source.number}</span></h2>
    <p className="massing-count">1 structural source <span>→</span> {parts.length} model parts</p>
    <div className="view-buttons massing-comparison" role="group" aria-label="Compare massing">
      {[['before','Before completion'],['after','After completion'],['parts','Parts only']].map(([key,label])=><button key={key}
        className={mode===key?'active':''} aria-pressed={mode===key}
        onClick={()=>setSettings(previous=>comparisonSettings(previous,key))}>{label}</button>)}
    </div>
    <div className="massing-toggles">
      <Toggle checked={settings.massingColors} onChange={v=>update('massingColors',v)} icon="cube">Color-code added parts</Toggle>
      <Toggle checked={settings.sourceOutline} onChange={v=>update('sourceOutline',v)} icon="layers">Show extracted footprint</Toggle>
    </div>
    <button className="massing-top" onClick={onTop}>Inspect parts from above</button>
    <p className="massing-key"><i/>Cyan: extracted footprint in plan view<br/>{settings.massingColors ? 'Colors distinguish individual parts placed for this source.' : 'Showing original part textures.'}</p>
    <p className="massing-description">{source.description || 'Observations cover part of this building. Existing model parts approximate the missing volume.'}</p>
    {source.parent_source_id ? <p className="massing-parent">Parent building: {Number(source.parent_source_id.split('_').at(-1))}</p> : <p className="massing-parent">No unique parent building identified.</p>}
    <div className="massing-part-list" aria-label="Added parts for this source">{parts.map((part,index)=><button key={part.id}
      className={part.id===asset.id?'selected':''} aria-label={`View added part ${source.number}·${index+1}`}
      onClick={()=>onSelect(part.id)}><i style={{background:PART_COLORS[index%PART_COLORS.length]}}/>
      <span>Part {index+1}<small>Donor asset {part.donor_asset_id ? Number(part.donor_asset_id.split('_').at(-1)) : '—'}</small></span>
      <span className="part-height">Approx. height {Math.round(part.top-part.base)} m</span></button>)}</div>
    <p className="group-note">These {parts.length} parts represent podiums, wings or perimeter volumes. Unseen geometry and appearance are inferred.</p>
    <button className="detail-button" onClick={onHigh} disabled={stats.high}><Icon name="layers"/>{stats.high?'0.25M models selected':'Load detailed 0.25M group'}</button>
    <details className="massing-evidence"><summary>Original observations and comparison images</summary><Evidence asset={evidence} onExpand={onExpand}/></details>
  </aside>;
}

function Inspector({ asset, data, settings, setSettings, onHigh, stats, onExpand, onFocus, onSelect, onTop }) {
  if (!asset) return <aside className="inspector"/>;
  const source = massingSourceFor(data,asset);
  if (source) return <MassingInspector {...{asset,source,data,settings,setSettings,onHigh,stats,onExpand,onSelect,onTop}}/>;
  const tag = asset.massing ? 'Massing addition' : asset.status === 'accepted_scene_completion' ? 'Appearance completion' : asset.group.length > 1 ? 'Connected buildings' : 'Accepted';
  const update = (key, value) => setSettings(previous => ({ ...previous, [key]: value }));
  return <aside className="inspector" aria-label="Building details">
    <h2>Building <span>{asset.display_number || asset.number}</span></h2>
    <div className="selection-status"><span className="muted">Selection</span><span className="status-label">{tag}</span></div>
    <Evidence asset={asset} onExpand={onExpand}/>
    <div className="view-buttons"><button className={settings.view==='neighborhood' ? 'active' : ''} onClick={() => onFocus('neighborhood')}><Icon name="cube"/>Neighborhood</button><button className={settings.view==='single' ? 'active' : ''} onClick={() => onFocus('single')}><Icon name="cube"/>Single building</button></div>
    <div className="range-control"><label htmlFor="radius">Neighborhood radius <strong>{settings.radius} m</strong></label><input id="radius" type="range" min="40" max="400" step="20" disabled={settings.view!=='neighborhood'} style={{'--range-progress': `${(settings.radius-40)/3.6}%`}} value={settings.radius} onChange={e => update('radius', Number(e.target.value))}/><div className="range-labels"><span>40</span><span>400</span></div></div>
    <div className="layers">
      <button className="detail-button" onClick={onHigh} disabled={stats.high || (settings.view==='city'&&!settings.cityBuildings)}><Icon name="layers"/><span>{stats.high ? '0.25M models selected' : 'Load detailed 0.25M model'}</span></button>
      <Toggle checked={settings.structures} onChange={v => update('structures', v)} icon="cube">Structural overview</Toggle>
      <Toggle checked={settings.outlines} onChange={v => update('outlines', v)} icon="layers">Structure outlines</Toggle>
    </div>
    <p className="surface-explanation">{settings.single ? 'Single-building mode isolates this building or group and hides observation surfaces and fragments.' : 'These references are local meshes recovered from early point predictions. They can be fragmented, layered or floating.'}</p>
    {!settings.single && settings.surfaces ? <p className="surface-active-note">{stats.rawReferences?.length ? `Visible references:${stats.rawReferences.join(' / ')}. Turn off observation references in the sidebar to hide them.` : stats.pending ? 'Loading observation references…' : 'No observation references loaded here.'}</p> : null}
    {asset.massing ? <p className="group-note">This source has {asset.group.length} added parts representing inferred podiums, wings or perimeter volumes. They share one building identity.</p> : asset.group.length > 1 ? <p className="group-note">Related buildings: {asset.group.map(id => Number(id.split('_').at(-1))).join(' + ')} are displayed as a group.</p> : null}
    {asset.number === '574' ? <p className="group-note">Layered observation fragments for 578 are hidden by default. Enable observation references in the sidebar to compare.</p> : null}
    {asset.number === '428' ? <p className="group-note">A small boundary overlap with 426 was recorded. Inspect it in the neighborhood view.</p> : null}
    <div className="inspector-note"><Icon name="info"/><span>This preview uses the released assets and their placements.<small>Static contacts and grounding have been checked. The review retained minor edge contacts.</small></span></div>
  </aside>;
}

function App() {
  const [data, setData] = useState(null), [error, setError] = useState('');
  const [selectedId, setSelectedId] = useState(null), [stats, setStats] = useState({ loaded: 0, pending: 0, desired: 0, errors: [] });
  const [settings, setSettings] = useState(() => ({ ...initialSceneSettings(window.location.search), ...initialCollisionSettings(window.location.search) }));
  const [sidebarMode, setSidebarMode] = useState(() => new URLSearchParams(window.location.search).get('inspect') === 'collisions' ? 'collisions' : 'buildings');
  const [selectedCollisionId, setSelectedCollisionId] = useState(null), [collisionFocus, setCollisionFocus] = useState(0);
  const [listRevision, setListRevision] = useState(0);
  const [railOpen, setRailOpen] = useState(false), [expanded, setExpanded] = useState(null), [bearing, setBearing] = useState(0);
  const host = useRef(null), engine = useRef(null);
  const contacts = useMemo(() => collisionCases(data?.collision_audit), [data]);
  const contactsById = useMemo(() => new Map(contacts.map(c => [c.id, c])), [contacts]);
  const onSelect = useCallback(id => {
    setSelectedCollisionId(null); engine.current?.selectCollision(null, { refit: false });
    setSelectedId(id);engine.current?.select(id);
    setSettings(previous=>id.startsWith('MASSING_') ? {...comparisonSettings(previous,'after'),massingColors:true,structures:false,backgroundElevated:false} : {...previous,massingColors:false,sourceOutline:false});
    const url = new URL(window.location.href); if (url.searchParams.get('inspect') === 'collisions') url.searchParams.delete('inspect');
    url.hash = id.startsWith('MASSING_') ? id : Number(id.split('_').at(-1)); history.replaceState(null, '', url);
  }, []);
  const onCollisionSelect = useCallback(id => {
    const item = contactsById.get(id); if (!item) return;
    setSelectedCollisionId(id); setCollisionFocus(v => v + 1); setSidebarMode('collisions'); setRailOpen(false);
    const assetId = item.asset_ids.find(aid => data.assets.some(a => a.id === aid && !a.surface));
    if (assetId) setSelectedId(assetId);
    setSettings(previous => ({ ...previous, view: 'neighborhood', single: false, collisionEnabled: true,
      collisionPriorityOnly: item.default_visible ? previous.collisionPriorityOnly : false, massingColors: false, sourceOutline: false,
      structures: true, outlines: false, surfaces: false, acceptedSurfaces: false }));
    const url = new URL(window.location.href); url.searchParams.set('inspect', 'collisions'); url.hash = id; history.replaceState(null, '', url);
  }, [contactsById, data]);
  const closeCollisions = () => { setSidebarMode('buildings'); setSelectedCollisionId(null); setSettings(previous => ({ ...previous, collisionEnabled: false }));
    const url = new URL(window.location.href); if (url.searchParams.get('inspect') === 'collisions') { url.searchParams.delete('inspect'); url.hash = selectedId || ''; history.replaceState(null, '', url); } };
  const changeCollisionSetting = (key, value) => {
    setSettings(previous => ({ ...previous, [key]: value }));
    if (selectedCollisionId && key !== 'collisionEnabled') {
      const next = { ...settings, [key]: value };
      if (!filterCollisionCases(contacts, { kind: next.collisionKind, priorityOnly: next.collisionPriorityOnly }).some(c => c.id === selectedCollisionId)) setSelectedCollisionId(null);
    }
  };
  const onMode = mode => {
    const next = {...settings,view:mode,single:mode==='single',cityPaused:mode==='city'?false:settings.cityPaused};
    setSettings(next); engine.current?.configure(next);
    if(mode==='city') engine.current?.allCity();
  };
  const onCompleteCity = () => {
    if (!data) return;
    const next = completeCitySettings(settings, data.background);
    const url = new URL(window.location.href);
    url.searchParams.delete('inspect'); url.hash = '';
    url.searchParams.set('background', next.backgroundMode); url.searchParams.set('buildings', '1'); url.searchParams.set('elevated', '1'); url.searchParams.set('ground', '1');
    history.replaceState(null, '', url);
    setSettings(next); setSelectedCollisionId(null); setSidebarMode('buildings'); setListRevision(v => v + 1); setRailOpen(false);
    engine.current?.configure(next); engine.current?.allCity();
  };
  const onElevated = () => {
    const next = {...settings, view:'city', single:false, backgroundElevated:true};
    setSettings(next); engine.current?.configure(next); engine.current?.focusElevated();
  };
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/scene', { signal: controller.signal }).then(r => { if (!r.ok) throw new Error(`Could not load scene index (${r.status})`); return r.json(); }).then(scene => {
      setData(scene);
      setSettings(previous => ({...previous, backgroundMode: previous.backgroundMode === 'coastal' && !scene.background?.files.coastal ? 'surface' : previous.backgroundMode === 'pixal3d' && !scene.background?.files.neural ? 'surface' : previous.backgroundMode}));
      const hash = window.location.hash.slice(1), requested = `PGB_SC_${hash.padStart(6, '0')}`;
      if (new URLSearchParams(window.location.search).get('inspect') === 'collisions') {
        const item = initialCollisionCase(collisionCases(scene.collision_audit), hash);
        if (item) {
          setSelectedCollisionId(item.id); setSelectedId(item.asset_ids.find(id => scene.assets.some(a => a.id === id)) || scene.assets[0]?.id);
          setSettings(previous => ({ ...previous, view: 'neighborhood', single: false, collisionEnabled: true,
            collisionPriorityOnly: Boolean(item.default_visible), structures: true, acceptedSurfaces: false, surfaces: false }));
          return;
        }
        setSettings(previous => ({ ...previous, collisionEnabled: false })); setSidebarMode('buildings');
      }
      const inspect = new URLSearchParams(window.location.search).get('inspect')==='massing';
      const requestedSource = inspect ? scene.massing_sources?.find(s=>s.id===hash || s.id===requested) : null;
      const selected = scene.assets.find(a => !a.surface && (a.id===(requestedSource?.part_ids[0] || hash) || a.sources.includes(requested)));
      setSelectedId(selected?.id || (inspect ? scene.massing_sources?.[0]?.part_ids[0] : null) || 'PGB_SC_000496');
    }).catch(e => { if (e.name !== 'AbortError') setError(e.message); });
    return () => controller.abort();
  }, []);
  useEffect(() => {
    if (!data || !host.current) return;
    try { engine.current = new CityScene(host.current, data, { onSelect, onCollisionSelect, onStats: setStats, onError: setError, onBearing: angle => setBearing(Math.round(angle)) }); }
    catch (e) { setError(`Could not start 3D view:${e.message}. Use a browser with WebGL 2 support.`); }
    return () => { engine.current?.dispose(); engine.current = null; };
  }, [data, onSelect, onCollisionSelect]);
  useEffect(() => { if (selectedId) engine.current?.select(selectedId,{refit:settings.view!=='city'}); }, [data, selectedId]);
  useEffect(() => engine.current?.configure(settings), [data,settings]);
  useEffect(() => { engine.current?.selectCollision(settings.collisionEnabled ? selectedCollisionId : null, { refit: true }); }, [data, selectedCollisionId, settings.collisionEnabled, collisionFocus]);
  useEffect(() => {
    const listener = e => { if (e.key === 'Escape') { setExpanded(null); setRailOpen(false); } };
    window.addEventListener('keydown', listener); return () => window.removeEventListener('keydown', listener);
  }, []);
  const selected = data?.assets.find(a => a.id === selectedId);
  const selectedContact = contactsById.get(selectedCollisionId);
  return <div className="app-shell">
    <header className="header"><div className="brand">CityCode</div><div className="header-divider"/><span className="header-title">Scene viewer</span><button className="mobile-list" onClick={() => setRailOpen(true)}>Layers and buildings</button><button className="overview-button" disabled={!data} onClick={onCompleteCity}><Icon name="cube"/>Full city scene</button></header>
    <main className="workspace">
      {data ? <BuildingList key={listRevision} data={data} selected={selected} onSelect={onSelect} open={railOpen} onClose={() => setRailOpen(false)} inspection={{ available: contacts.length > 0, open: sidebarMode === 'collisions', onOpen: () => { setSidebarMode('collisions'); setSettings(previous => ({ ...previous, collisionEnabled: true })); },
        panel: <CollisionPanel audit={data.collision_audit} cases={contacts} settings={settings} selectedId={selectedCollisionId} onSelect={onCollisionSelect} onChange={changeCollisionSetting} onClose={closeCollisions}/> }}><SceneControls data={data} settings={settings} setSettings={setSettings} stats={stats} onMode={onMode} onElevated={onElevated} onCoast={() => { engine.current?.focusCoast(); setRailOpen(false); }}/></BuildingList> : null}
      {railOpen ? <button className="mobile-shade" aria-label="Close list overlay" onClick={() => setRailOpen(false)}/> : null}
      <section className="canvas-panel" aria-label="3D scene viewer">
        <div className="canvas-host" ref={host}/>
        <div className="canvas-title"><Icon name="layers"/><span>{selected?.massing && !stats.overview ? `Source ${selected.number} · ${!settings.massing?'Before completion':settings.single?'Added parts only':'after completion and nearby buildings'}` : stats.overview ? `City view · ${settings.cityQuality==='preview'?'Lightweight models':settings.cityQuality+' geometry'}` : settings.single ? 'Single building' : 'Neighborhood · On demand'}</span></div>
        <div className="scene-meta">{data ? `${data.counts.entities} building groups · ${data.counts.base_assets ?? data.counts.assets} original assets${data.counts.massing_parts ? ` + ${data.counts.massing_parts} added parts` : ''}` : 'Loading scene…'}</div>
        {settings.collisionEnabled && contacts.length ? <div className="collision-map-note"><strong>{selectedContact ? selectedContact.label : 'Contact locations'}</strong><span>{selectedContact ? `${collisionStatus(selectedContact).label} · Select to see the sampled area` : settings.view === 'city' ? 'Nearby contacts share markers. Select one to inspect the neighborhood.' : 'Select a marker or a list entry to inspect the neighborhood.'}</span></div> : null}
        {data?.background && !settings.single ? <div className="background-legend"><i/>{settings.backgroundMode==='coastal'?'Coastal land and water':settings.backgroundMode==='pixal3d'?'Original Pixal3D · Lightweight preview':settings.mapSolid&&settings.backgroundMap&&data.background.files.map_solid?'RGB map · Extruded land':'RGB map · Flat reference'}{settings.backgroundElevated ? ` · ${settings.elevatedStyle==='highlight'?'Orange:':'Textured:'} elevated surface` : ''}<span>Buildings and reference terrain</span></div> : null}
        <div className="navigation"><button className="compass" title="Top" aria-label="Toggle top view" onClick={() => engine.current?.top()}><span>N</span><i style={{ transform: `rotate(${bearing}deg)` }}/></button><div className="zoom-buttons"><button title="Return to selected building" aria-label="Return to selected building" onClick={() => engine.current?.select(selectedId)}><Icon name="home"/></button><button aria-label="Zoom in" onClick={() => engine.current?.zoom(.75)}><Icon name="plus"/></button><button aria-label="Zoom out" onClick={() => engine.current?.zoom(1.3)}><Icon name="minus"/></button></div></div>
        {error ? <div className="scene-error" role="alert">{error}</div> : null}
        {!data && !error ? <div className="loading-screen">Preparing building scene…</div> : null}
        <div className="canvas-bottom"><div className="interaction-hint"><Icon name="mouse"/><span>Drag to orbit · Scroll to zoom · Right-drag to pan</span></div><div className="load-status" role="status" data-view={settings.view} data-collision-enabled={String(stats.collisions?.enabled || false)} data-collision-selected={stats.collisions?.selectedId || ''} data-collision-markers={stats.collisions?.markerCount || 0} data-collision-patches={stats.collisions?.patchCount || 0} data-collision-assets={stats.collisions?.highlighted?.join(',') || ''} data-selected-loaded={String(stats.selectedLoaded || false)} data-selected-parts-visible={stats.selectedPartsVisible||0} data-colored-parts={stats.coloredParts||0} data-source-outline={String(stats.sourceOutlineVisible||false)} data-loaded={stats.loaded} data-city-loaded={stats.cityLoaded || 0} data-city-desired={stats.cityDesired||0} data-city-paused={String(stats.cityPaused||false)} data-city-texture-max={stats.cityTextureMaxPx||0} data-city-faces={stats.cityFaces||0} data-pending={stats.pending || 0} data-reference-count={stats.rawReferences?.length || 0} data-reference-ids={stats.rawReferences?.join(',') || ''} data-accepted-surface-count={stats.acceptedSurfaces?.length || 0}>
          <i className={stats.pending || (!stats.cityPaused && stats.cityLoaded < stats.cityDesired) ? 'busy' : ''}/>{stats.overview ? `City models ${stats.cityLoaded||0} / ${stats.cityDesired||0}${stats.cityPaused?' · Paused':''}` : stats.pendingModels ? `Loading ${stats.loaded} / ${stats.desired} buildings` : `Loaded ${stats.loaded} building models`}
          {stats.overview&&stats.cityBytes ? <small>{formatSize(stats.cityBytes)}</small> : null}
          {!stats.overview && stats.bytes ? <small>{(stats.bytes/1024/1024).toFixed(0)} MB</small> : null}
        </div></div>
        {!settings.single && stats.rawReferences?.length ? <div className="reference-banner" role="note"><strong>{stats.overview?'City observation references':'Observation references'}: {stats.rawReferences.join(' / ')}</strong><span>Local predicted surfaces may be fragmented, layered or floating.</span></div> : null}
        {stats.errors?.length ? <div className="load-warning" role="alert">{stats.errors.join('; ')}. Select the building again to retry.</div> : null}
        {stats.cityErrors ? <div className="load-warning" role="alert">{stats.cityErrors} city resources failed to load. Select Load entire city to retry.</div> : null}
        {stats.limited && !stats.overview ? <div className="budget-note">There are {stats.total} models nearby. The closest are shown first. Select another structural outline to inspect it.</div> : null}
      </section>
      {selectedContact ? <CollisionInspector item={selectedContact} data={data} settings={settings} onSelect={onSelect} onFocus={onCollisionSelect} onClose={() => setSelectedCollisionId(null)} stats={stats}/> : selected ? <Inspector asset={selected} data={data} settings={settings} setSettings={setSettings} onHigh={() => engine.current?.loadHigh()} stats={stats} onExpand={setExpanded} onFocus={onMode} onSelect={onSelect} onTop={()=>engine.current?.top()}/> : null}
    </main>
    {expanded ? <div className="image-modal" role="dialog" aria-modal="true" aria-label={expanded.title} onClick={() => setExpanded(null)}><button className="modal-close" aria-label="Close image" autoFocus onClick={() => setExpanded(null)}><Icon name="close"/></button><figure onClick={e => e.stopPropagation()}><img src={expanded.url} alt={expanded.title}/><figcaption>{expanded.title}</figcaption></figure></div> : null}
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
