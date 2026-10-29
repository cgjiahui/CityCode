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
  return <section className="scene-controls" aria-label="全城加载设置">
    <div className="scene-control-title"><h2>全城加载</h2><span>{data.counts.assets} 份模型</span></div>
    <div className="mode-switch" role="group" aria-label="查看范围">{[['city','全城'],['neighborhood','周边'],['single','单体']].map(([mode,label])=><button key={mode} className={settings.view===mode?'active':''} onClick={()=>onMode(mode)}>{label}</button>)}</div>
    {data.background ? <div className="background-controls" aria-label="背景与升高结构">
      <div className="scene-control-title"><h2>背景与升高结构</h2><span>历史背景</span></div>
      <label className="quality-select">背景模式<select aria-label="背景模式" value={settings.backgroundMode} disabled={isolated} onChange={e=>update('backgroundMode',e.target.value)}><option value="surface">地图 / 高度表面</option>{data.background.files.coastal ? <option value="coastal">陆地加厚 · 平面水域</option> : null}{data.background.files.neural ? <option value="pixal3d">原始 Pixal3D · 对照</option> : null}</select></label>
      <Toggle checked={settings.backgroundMap && !isolated && !neural} onChange={v=>update('backgroundMap',v)} disabled={isolated || neural} icon="layers">融合RGB地图</Toggle>
      {hasMapSolid ? <Toggle checked={solidMap && !isolated} onChange={v=>update('mapSolid',v)} disabled={isolated || neural || !settings.backgroundMap} icon="cube">RGB地图陆地加厚</Toggle> : null}
      <Toggle checked={settings.backgroundGround && !isolated && !neural && !solidMap} onChange={v=>update('backgroundGround',v)} disabled={isolated || neural || solidMap} icon="surface">地面高度表面</Toggle>
      {solidMap && !isolated ? <p className="global-help">关闭地图陆地加厚后，可单独查看原地面高度表面。</p> : null}
      <Toggle checked={settings.backgroundElevated && !isolated} onChange={v=>update('backgroundElevated',v)} disabled={isolated} icon="layers">升高结构 · 高架表面</Toggle>
      <label className="quality-select">升高结构显示<select aria-label="升高结构显示" value={settings.elevatedStyle} onChange={e=>update('elevatedStyle',e.target.value)}><option value="highlight">橙色高亮</option><option value="texture">融合图纹理</option></select></label>
      <button className="focus-elevated" onClick={onElevated}>定位升高结构</button>
      {coastal || solidMap ? <Toggle checked={settings.backgroundWater && !isolated} onChange={v=>update('backgroundWater',v)} disabled={isolated} icon="layers">平面水域</Toggle> : null}
      {coastal ? <button className="focus-elevated" disabled={isolated} onClick={onCoast}>查看岸壁厚度</button> : null}
      {solidMap ? <button className="focus-elevated" disabled={isolated} onClick={onCoast}>查看地图岸壁厚度</button> : null}
      <p className="global-help">{coastal ? '按 GT 可见高度裁剪陆地和码头，补岸壁与底面；水位 −3.5m，底面 −8m。关闭水域可查看完整厚度。建筑位置不变。' : neural ? '历史 Pixal3D 成品的轻量预览，包含生成细节和连续地面。可关闭橙色升高层查看原貌；原始完整模型保留在数据包。' : solidMap ? '原RGB地图分为独立陆地和水面。陆地顶面 0m、水面 −3.5m、底面 −8m；岸壁和底面为显示加厚，不改变建筑或原高度数据。' : 'RGB地图是平面参考，可独立叠加地面高度和升高表面；高架尚无桥墩、桥底。'}</p>
      <div className="background-progress" role="status" data-map-solid-loaded={String(stats.background?.mapSolidLoaded||false)} data-map-solid-visible={String(stats.background?.mapSolidVisible||false)} data-map-water-visible={String(stats.background?.mapWaterVisible||false)} data-coastal-loaded={String(stats.background?.coastalLoaded||false)} data-coastal-visible={String(stats.background?.coastalVisible||false)} data-water-visible={String(stats.background?.waterVisible||false)} data-mode={settings.backgroundMode} data-neural-loaded={String(stats.background?.neuralLoaded||false)} data-neural-visible={String(stats.background?.neuralVisible||false)} data-map-loaded={String(stats.background?.mapLoaded||false)} data-surfaces-loaded={String(stats.background?.surfacesLoaded||false)} data-map-visible={String(stats.background?.mapVisible||false)} data-ground-visible={String(stats.background?.groundVisible||false)} data-elevated-visible={String(stats.background?.elevatedVisible||false)} data-style={stats.background?.elevatedStyle||''}>
        {stats.background?.errors?.length ? stats.background.errors.join('；') : stats.background?.pending ? '正在读取背景…' : coastal && stats.background?.coastalLoaded ? `陆地加厚已就绪 · ${stats.background.coastalFaces?.toLocaleString()} 面` : neural && stats.background?.neuralLoaded ? `Pixal3D 已就绪 · ${stats.background.neuralFaces?.toLocaleString()} 面` : solidMap && stats.background?.mapSolidLoaded ? `RGB陆地加厚已就绪 · ${stats.background.mapSolidFaces?.toLocaleString()} 面` : stats.background?.surfacesLoaded ? `背景已就绪 · 升高表面 ${stats.background?.elevatedFaces?.toLocaleString() || '—'} 面` : '背景按图层加载'}
      </div>
    </div> : <p className="global-help">此数据目录未提供背景预览。</p>}
    <label className="quality-select">全城模型档位<select aria-label="全城模型档位" value={settings.cityQuality} onChange={e=>update('cityQuality',e.target.value)}><option value="preview">轻量模型 · {formatSize(totals.previewBytes)}</option><option value="0.1M">0.1M · 原始几何 · {formatSize(totals.bytes)}</option><option value="0.25M">0.25M · 精细几何 · {formatSize(totals.fineBytes)}</option></select></label>
    <label className="quality-select">表面显示<select aria-label="表面显示" value={settings.surfaceDisplay} onChange={e=>update('surfaceDisplay',e.target.value)}><option value="neutral">中性哑光（默认）</option><option value="original">原始材质（含反光）</option></select></label>
    <p className="global-help">{settings.surfaceDisplay==='original' ? '保留模型原有反光，便于对照；轻量副本已简化材质。' : '降低镜面反光，便于检查形体与颜色；仅改变网页显示。'}</p>
    <Toggle checked={settings.cityBuildings} onChange={v=>update('cityBuildings',v)} disabled={!city} icon="cube">全城建筑模型 · {data.counts.assets}</Toggle>
    {data.counts.massing_parts ? <Toggle checked={settings.massing} onChange={v=>update('massing',v)} icon="cube">显示体量补全 · {data.counts.massing_parts}</Toggle> : null}
    <Toggle checked={settings.acceptedSurfaces && !isolated} onChange={v=>update('acceptedSurfaces',v)} disabled={isolated} icon="surface">显示已接纳附属面 · {totals.accepted}</Toggle>
    <Toggle checked={settings.surfaces && !isolated} onChange={v=>update('surfaces',v)} disabled={isolated} icon="surface">叠加原始观测参考 · {totals.references}</Toggle>
    <p className="global-help">{isolated ? '单体模式隐藏全部观测面。' : city ? '三类图层分别加载全城，互不影响。' : '周边模式按附近范围加载观测面。'}{settings.surfaces&&!isolated ? ' 原始参考可能碎裂、分层或悬浮，不属于生成建筑。' : ''}</p>
    {settings.cityQuality!=='preview' ? <p className="quality-help">加载全部{settings.cityQuality}网格，每模型目标约{settings.cityQuality==='0.25M'?'25':'10'}万三角面；贴图在浏览器缩至512像素。需要较多内存和显存，可暂停或关闭建筑图层释放资源。</p> : null}
    <div className="global-actions"><button className="load-city-button" onClick={()=>onMode('city')}>一次加载全城</button><button className="pause-city-button" disabled={!city || (!stats.cityPaused && stats.cityTotalLoaded===stats.cityTotalDesired)} onClick={()=>update('cityPaused',!settings.cityPaused)}>{settings.cityPaused?'继续':'暂停'}</button></div>
    {city ? <div className="global-progress" role="status" aria-label="全城加载进度"><progress max={Math.max(1,stats.cityTotalDesired||0)} value={stats.cityTotalLoaded||0}/><span>建筑 {stats.cityLoaded||0}/{stats.cityDesired||0} · 附属 {stats.acceptedSurfaces?.length||0}/{stats.cityAcceptedDesired||0} · 参考 {stats.rawReferences?.length||0}/{stats.cityReferencesDesired||0}{stats.cityPaused?' · 已暂停':''}</span></div> : null}
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
  if (inspection.open && inspection.available) return <aside className={`sidebar collision-sidebar ${open ? 'sidebar-open' : ''}`} aria-label="碰撞检查列表">
    <button className="icon-button mobile-only sidebar-close" aria-label="关闭建筑列表" onClick={onClose}><Icon name="close"/></button>
    {inspection.panel}
    <details className="collision-scene-settings"><summary>全城与图层设置</summary>{children}</details>
  </aside>;
  return <aside className={`sidebar ${open ? 'sidebar-open' : ''}`} aria-label="建筑列表">
    <button className="icon-button mobile-only sidebar-close" aria-label="关闭建筑列表" onClick={onClose}><Icon name="close"/></button>
    <button className="collision-entry" onClick={inspection.onOpen} disabled={!inspection.available} aria-label="打开碰撞检查"><Icon name="surface"/><span>碰撞检查<small>{inspection.available ? '当前处理结果 · 可定位查看' : '当前数据未提供检查结果'}</small></span></button>
    {sourceGroups.length ? <button className="massing-entry" onClick={()=>{
      setScope('source_groups');setQuery('');onSelect(selected?.massing ? selected.id : sourceGroups[0].part_ids[0]);onClose();
    }}><Icon name="layers"/>查看 {sourceGroups.length} 个局部来源</button> : null}
    {scope!=='source_groups' ? children : null}
    <div className="rail-title"><h1>{scope==='source_groups' ? '局部来源' : '建筑列表'}</h1></div>
    <p className="muted rail-subtitle">{query ? `找到 ${items.length} 项` : scope==='source_groups' ? `${sourceGroups.length} 个来源 → ${data.counts.massing_parts} 个部件` : scope === 'all' ? `全部 ${data.counts.assets} 份资产` : scope === 'massing' ? `本次体量补全 ${data.counts.massing_parts} 个部件` : scope === 'review' ? `后续补全的 ${reviewedCount} 份资产` : '最近处理的 11 项'}</p>
    {scope==='source_groups' && data.massing_pruning?.removed_parts ? <p className="global-help">逐件去掉 {data.massing_pruning.removed_parts} 个补入部件，允许轮廓留空。整组取消：{(data.massing_pruning.explicit_excluded_source_ids||[]).map(id=>String(Number(id.split('_').at(-1)))).join('、')}。{data.massing_pruning.empty_after_part_review_source_ids?.length ? <>逐件检查后无保留部件：{data.massing_pruning.empty_after_part_review_source_ids.map(id=>String(Number(id.split('_').at(-1)))).join('、')}。</> : null}</p> : null}
    <label className="search"><Icon name="search"/><input value={query} onChange={e => setQuery(e.target.value.trim())} placeholder="输入建筑编号搜索" aria-label="输入建筑编号搜索"/></label>
    <select className="list-scope" aria-label="建筑列表范围" value={scope} onChange={e => { setScope(e.target.value); setQuery(''); }}>{sourceGroups.length ? <option value="source_groups">局部来源 · {sourceGroups.length} 个</option> : null}<option value="recent">最近处理的 11 项</option><option value="review">后续补全资产 · {reviewedCount} 份</option>{data.counts.massing_parts ? <option value="massing">本次体量补全 · {data.counts.massing_parts} 个部件</option> : null}<option value="all">全部建筑资产 · {data.counts.assets} 份</option></select>
    <div className="building-list" role="list">
      {items.map(asset => <div role="listitem" key={asset.id}><button className={`building-row ${(asset.sourceEntry ? selected?.sources.includes(asset.sources[0]) : asset.id === selected?.id) ? 'selected' : ''}`} onClick={() => { onSelect(asset.id); onClose(); }} aria-label={asset.sourceEntry ? `查看局部来源 ${asset.number}` : `选择建筑 ${asset.number}`} aria-current={(asset.sourceEntry ? selected?.sources.includes(asset.sources[0]) : asset.id === selected?.id) ? 'true' : undefined}>
        <span>{asset.display_number || asset.number}</span>{asset.sourceEntry ? <span className="row-note">{asset.partCount} 个部件</span> : asset.massing ? <span className="row-note">体量补全</span> : asset.group.length > 1 ? <span className="row-note">连体</span> : null}
      </button></div>)}
      {!items.length ? <p className="empty-list">没有匹配的可装配建筑。</p> : null}
    </div>
    <button className="text-button all-buildings" onClick={() => { setScope(v => v === 'all' ? 'recent' : 'all'); setQuery(''); }}>{scope === 'all' ? '返回最近处理' : `查看全部 ${data.counts.assets} 份资产`}</button>
    <div className="legend"><div><i className="swatch proxy"/>浅灰：结构概览</div><div><i className="swatch model"/>彩色：真实模型</div></div>
    {scope==='source_groups' ? <details className="massing-extra"><summary>全城与背景设置</summary>{children}</details> : null}
  </aside>;
}

function Evidence({ asset, onExpand }) {
  const [role, setRole] = useState('front');
  useEffect(() => setRole('front'), [asset.id]);
  const labels = asset.massing ? {front:'补全后',previous:'补全前',context:'仅补入部件',source:'原观测',back:'背面'} : { front: '当前模型', previous: '之前模型', appearance: '补全图', source: '原观测', context: '周边截图', back: '背面' };
  const available = Object.keys(labels).filter(key => asset.evidence[key]);
  const actualRole = asset.evidence[role] ? role : available[0];
  return <div className="evidence">
    {actualRole ? <button className="evidence-image" onClick={() => onExpand({ url: asset.evidence[actualRole], title: `建筑 ${asset.number} · ${labels[actualRole]}` })} aria-label="放大模型参考图">
      <img src={asset.evidence[actualRole]} alt={`建筑 ${asset.number} ${labels[actualRole]}`} loading="lazy"/><span className="image-expand"><Icon name="expand" size={17}/></span>
    </button> : <div className="evidence-empty"><Icon name="cube" size={32}/><span>在场景中查看真实模型</span></div>}
    {available.length > 1 ? <div className="evidence-tabs" role="group" aria-label="查看图片证据">{available.map(key => <button key={key} className={actualRole === key ? 'active' : ''} onClick={() => setRole(key)}>{labels[key]}</button>)}</div> : null}
  </div>;
}

function Toggle({ checked, onChange, icon, disabled = false, children }) {
  return <label className={`toggle-row ${checked ? 'checked' : ''} ${disabled ? 'disabled' : ''}`}><Icon name={icon}/><span>{children}</span><input type="checkbox" checked={checked} disabled={disabled} onChange={e => onChange(e.target.checked)}/><i className="toggle-indicator"/></label>;
}

function CollisionPanel({ audit, cases, settings, selectedId, onSelect, onChange, onClose }) {
  const visible = useMemo(() => filterCollisionCases(cases, { kind: settings.collisionKind, priorityOnly: settings.collisionPriorityOnly }), [cases, settings.collisionKind, settings.collisionPriorityOnly]);
  const counts = audit.counts || {}, centerline = counts.vehicle_centerline_obstructions;
  return <section className="collision-panel" aria-label="碰撞检查">
    <div className="collision-heading"><h1>碰撞检查</h1><button onClick={onClose} aria-label="关闭碰撞检查"><Icon name="close" size={16}/></button></div>
    <p className="collision-subtitle">当前物理后处理结果</p>
    <div className="collision-summary"><strong>{centerline === 0 ? '车辆中心线占用 0' : Number.isFinite(centerline) ? `车辆中心线占用 ${centerline}` : '车辆中心线已独立复查'}</strong><span>少量边缘接触按审核保留</span></div>
    <Toggle checked={settings.collisionEnabled} onChange={v => onChange('collisionEnabled', v)} icon="layers">显示检查图层</Toggle>
    <label className="quality-select collision-kind">检查类型<select aria-label="碰撞检查类型" value={settings.collisionKind} onChange={e => onChange('collisionKind', e.target.value)}>
      <option value="all">全部类别 · {cases.length}</option>{Object.entries(COLLISION_KINDS).map(([kind, label]) => <option key={kind} value={kind}>{label} · {cases.filter(c => c.kind === kind).length}</option>)}
    </select></label>
    <div className="collision-scope" role="group" aria-label="碰撞检查范围">{[[true, '仅重点'], [false, '全部接触']].map(([priority, label]) => <button key={label} aria-pressed={settings.collisionPriorityOnly === priority} className={settings.collisionPriorityOnly === priority ? 'active' : ''} onClick={() => onChange('collisionPriorityOnly', priority)}>{label}</button>)}</div>
    <p className="collision-help">重点是便于继续查看的位置，不代表严重碰撞。全城仅显示位置热点，选中后显示局部接触范围。</p>
    <p className="collision-list-count" role="status">{visible.length} 处{settings.collisionPriorityOnly ? '重点位置' : '接触记录'}</p>
    <div className="collision-list" role="list" aria-label="接触位置列表">{visible.map(item => {
      const status = collisionStatus(item), primary = collisionMetrics(item)[0];
      return <div role="listitem" key={item.id}><button className={`collision-row ${item.id === selectedId ? 'selected' : ''}`} aria-label={`查看${item.label}`} aria-current={item.id === selectedId ? 'true' : undefined} onClick={() => onSelect(item.id)}>
        <i style={{ background: COLLISION_COLORS[item.kind] }}/><span><strong>{item.label}</strong><small>{status.label}{primary ? ` · ${primary.value}` : ''}</small></span><span className="collision-row-arrow">›</span>
      </button></div>;
    })}{!visible.length ? <p className="collision-empty">此类别没有{settings.collisionPriorityOnly ? '重点位置，可切换“全部接触”查看。' : '接触记录。'}</p> : null}</div>
    <div className="collision-legend">{Object.entries(COLLISION_KINDS).map(([kind, label]) => <span key={kind}><i style={{ background: COLLISION_COLORS[kind] }}/>{label}</span>)}</div>
    <button className="collision-back" onClick={onClose}>关闭检查，返回建筑列表</button>
  </section>;
}

function CollisionInspector({ item, data, settings, onSelect, onFocus, onClose, stats }) {
  const status = collisionStatus(item), metrics = collisionMetrics(item), assets = item.asset_ids.map(id => data.assets.find(a => a.id === id)).filter(Boolean);
  return <aside className="inspector collision-inspector" aria-label="接触详情">
    <div className="collision-detail-heading"><span>{COLLISION_KINDS[item.kind]}</span><button className="icon-button" aria-label="取消接触选择" onClick={onClose}><Icon name="close" size={18}/></button></div>
    <h2>{item.label.split(' · ')[0]}</h2><span className={`collision-status ${status.tone}`}>{status.label}</span>
    <p className="collision-description">{item.description || status.explanation}</p>
    {item.reason ? <p className="collision-reason">{item.reason}</p> : null}
    <div className="collision-asset-list" aria-label="关联建筑">{assets.map((asset, index) => <button key={asset.id} onClick={() => onSelect(asset.id)} aria-label={`查看关联建筑 ${asset.display_number || asset.number}`}><i style={{ background: RELATED_COLORS[index % RELATED_COLORS.length] }}/><span>建筑 {asset.display_number || asset.number}</span><small>单独查看 ›</small></button>)}</div>
    <dl className="collision-metrics">{metrics.map(m => <div key={m.label}><dt>{m.label}</dt><dd>{m.value}</dd></div>)}</dl>
    <button className="detail-button" onClick={() => onFocus(item.id)}><Icon name="home"/>定位附近场景</button>
    <p className="collision-key">蓝 / 橙框：关联建筑<br/>局部色块：当前接触采样范围</p>
    <p className="collision-patch-note">{!settings.collisionEnabled ? '检查图层已关闭，可在左侧重新开启。' : stats.collisions?.patchCount ? `当前显示 ${stats.collisions.patchCount} 个局部采样块${item.patches_truncated ? '（范围已抽样）' : ''}。` : '当前显示接触位置标记。'}{item.kind === 'ground' ? ' 此项来自背景细节，不代表建筑模型生成了新噪声。' : ''}</p>
    <div className="inspector-note"><Icon name="info"/><span>{status.explanation}<small>只展示当前处理后的结果；轻微接触不会自动移除建筑。</small></span></div>
  </aside>;
}

function MassingInspector({ asset, source, data, settings, setSettings, onHigh, stats, onExpand, onSelect, onTop }) {
  const parts = source.part_ids.map(id=>data.assets.find(a=>a.id===id));
  const mode = !settings.massing ? 'before' : settings.single ? 'parts' : 'after';
  const update = (key,value)=>setSettings(previous=>({...previous,[key]:value}));
  const evidence = {...asset,evidence:{front:source.evidence.after,previous:source.evidence.before,
    context:source.evidence.parts,source:source.evidence.source}};
  return <aside className="inspector massing-inspector" aria-label="局部来源详情">
    <h2>局部来源 <span>{source.number}</span></h2>
    <p className="massing-count">1 个结构来源 <span>→</span> {parts.length} 个模型部件</p>
    <div className="view-buttons massing-comparison" role="group" aria-label="体量补全对照">
      {[['before','补全前'],['after','补全后'],['parts','只看部件']].map(([key,label])=><button key={key}
        className={mode===key?'active':''} aria-pressed={mode===key}
        onClick={()=>setSettings(previous=>comparisonSettings(previous,key))}>{label}</button>)}
    </div>
    <div className="massing-toggles">
      <Toggle checked={settings.massingColors} onChange={v=>update('massingColors',v)} icon="cube">分色显示补入部件</Toggle>
      <Toggle checked={settings.sourceOutline} onChange={v=>update('sourceOutline',v)} icon="layers">显示原提取范围</Toggle>
    </div>
    <button className="massing-top" onClick={onTop}>俯视查看分块范围</button>
    <p className="massing-key"><i/>青线：原提取范围的平面投影<br/>{settings.massingColors ? '各颜色：同一来源内分别放置的模型部件。' : '当前显示部件原有纹理。'}</p>
    <p className="massing-description">{source.description || '此处只有局部建筑观测，使用多个已有模型部件补充缺失体量。'}</p>
    {source.parent_source_id ? <p className="massing-parent">关联主体：建筑 {Number(source.parent_source_id.split('_').at(-1))}</p> : <p className="massing-parent">未确定唯一所属主楼。</p>}
    <div className="massing-part-list" aria-label="此来源的补入部件">{parts.map((part,index)=><button key={part.id}
      className={part.id===asset.id?'selected':''} aria-label={`查看补全部件 ${source.number}·${index+1}`}
      onClick={()=>onSelect(part.id)}><i style={{background:PART_COLORS[index%PART_COLORS.length]}}/>
      <span>部件 {index+1}<small>复用资产 {part.donor_asset_id ? Number(part.donor_asset_id.split('_').at(-1)) : '—'}</small></span>
      <span className="part-height">高约 {Math.round(part.top-part.base)} m</span></button>)}</div>
    <p className="group-note">这些部件共同表达裙楼、翼部或边带，不是新增 {parts.length} 栋独立建筑。外观与不可见体量为近似推断。</p>
    <button className="detail-button" onClick={onHigh} disabled={stats.high}><Icon name="layers"/>{stats.high?'已选择0.25M模型':'加载这组0.25M精细模型'}</button>
    <details className="massing-evidence"><summary>原观测与补全前后截图</summary><Evidence asset={evidence} onExpand={onExpand}/></details>
  </aside>;
}

function Inspector({ asset, data, settings, setSettings, onHigh, stats, onExpand, onFocus, onSelect, onTop }) {
  if (!asset) return <aside className="inspector"/>;
  const source = massingSourceFor(data,asset);
  if (source) return <MassingInspector {...{asset,source,data,settings,setSettings,onHigh,stats,onExpand,onSelect,onTop}}/>;
  const tag = asset.massing ? '体量补全' : asset.status === 'accepted_scene_completion' ? '外观补全' : asset.group.length > 1 ? '连体建筑' : '已接纳';
  const update = (key, value) => setSettings(previous => ({ ...previous, [key]: value }));
  return <aside className="inspector" aria-label="建筑详情">
    <h2>建筑 <span>{asset.display_number || asset.number}</span></h2>
    <div className="selection-status"><span className="muted">选中状态</span><span className="status-label">{tag}</span></div>
    <Evidence asset={asset} onExpand={onExpand}/>
    <div className="view-buttons"><button className={settings.view==='neighborhood' ? 'active' : ''} onClick={() => onFocus('neighborhood')}><Icon name="cube"/>周边场景</button><button className={settings.view==='single' ? 'active' : ''} onClick={() => onFocus('single')}><Icon name="cube"/>单体查看</button></div>
    <div className="range-control"><label htmlFor="radius">周边范围 <strong>{settings.radius} m</strong></label><input id="radius" type="range" min="40" max="400" step="20" disabled={settings.view!=='neighborhood'} style={{'--range-progress': `${(settings.radius-40)/3.6}%`}} value={settings.radius} onChange={e => update('radius', Number(e.target.value))}/><div className="range-labels"><span>40</span><span>400</span></div></div>
    <div className="layers">
      <button className="detail-button" onClick={onHigh} disabled={stats.high || (settings.view==='city'&&!settings.cityBuildings)}><Icon name="layers"/><span>{stats.high ? '已选择0.25M模型' : '加载0.25M精细模型'}</span></button>
      <Toggle checked={settings.structures} onChange={v => update('structures', v)} icon="cube">显示结构概览</Toggle>
      <Toggle checked={settings.outlines} onChange={v => update('outlines', v)} icon="layers">显示结构轮廓</Toggle>
    </div>
    <p className="surface-explanation">{settings.single ? '单体模式仅显示当前建筑或建筑组，所有观测附属面与参考残片均已隐藏。' : '原始观测参考是早期预测点恢复的局部网格，可能碎裂、分层或悬浮，不属于生成建筑模型。'}</p>
    {!settings.single && settings.surfaces ? <p className="surface-active-note">{stats.rawReferences?.length ? `当前参考：${stats.rawReferences.join(' / ')}。关闭左侧参考开关即可移除。` : stats.pending ? '正在读取原始观测参考…' : '当前没有已加载的原始观测参考。'}</p> : null}
    {asset.massing ? <p className="group-note">同一来源的 {asset.group.length} 个补全部件；属于推断的裙楼、翼部或边带，不增加独立建筑计数。</p> : asset.group.length > 1 ? <p className="group-note">关联建筑：{asset.group.map(id => Number(id.split('_').at(-1))).join(' + ')}，按同一建筑组查看。</p> : null}
    {asset.number === '574' ? <p className="group-note">578 的分层观测残片默认关闭；可用左侧原始参考开关叠加对照。</p> : null}
    {asset.number === '428' ? <p className="group-note">与 426 的少量边界交叠已记录，可在周边场景检查。</p> : null}
    <div className="inspector-note"><Icon name="info"/><span>以当前发布资产与场景摆放结果预览。<small>已做静态接触与接地复查；少量边缘接触按审核保留。</small></span></div>
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
    fetch('/api/scene', { signal: controller.signal }).then(r => { if (!r.ok) throw new Error(`场景索引读取失败 (${r.status})`); return r.json(); }).then(scene => {
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
    catch (e) { setError(`无法启动 3D 视图：${e.message}。请使用支持 WebGL 2 的浏览器。`); }
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
    <header className="header"><div className="brand">CityCode</div><div className="header-divider"/><span className="header-title">场景预览</span><button className="mobile-list" onClick={() => setRailOpen(true)}>图层与建筑</button><button className="overview-button" disabled={!data} onClick={onCompleteCity}><Icon name="cube"/>完整城市场景</button></header>
    <main className="workspace">
      {data ? <BuildingList key={listRevision} data={data} selected={selected} onSelect={onSelect} open={railOpen} onClose={() => setRailOpen(false)} inspection={{ available: contacts.length > 0, open: sidebarMode === 'collisions', onOpen: () => { setSidebarMode('collisions'); setSettings(previous => ({ ...previous, collisionEnabled: true })); },
        panel: <CollisionPanel audit={data.collision_audit} cases={contacts} settings={settings} selectedId={selectedCollisionId} onSelect={onCollisionSelect} onChange={changeCollisionSetting} onClose={closeCollisions}/> }}><SceneControls data={data} settings={settings} setSettings={setSettings} stats={stats} onMode={onMode} onElevated={onElevated} onCoast={() => { engine.current?.focusCoast(); setRailOpen(false); }}/></BuildingList> : null}
      {railOpen ? <button className="mobile-shade" aria-label="关闭列表遮罩" onClick={() => setRailOpen(false)}/> : null}
      <section className="canvas-panel" aria-label="三维场景预览">
        <div className="canvas-host" ref={host}/>
        <div className="canvas-title"><Icon name="layers"/><span>{selected?.massing && !stats.overview ? `来源 ${selected.number} · ${!settings.massing?'补全前':settings.single?'仅补入部件':'补全后及周边建筑'}` : stats.overview ? `全城场景 · ${settings.cityQuality==='preview'?'轻量模型':settings.cityQuality+'几何'}` : settings.single ? '单体建筑模型' : '周边场景 · 按需加载'}</span></div>
        <div className="scene-meta">{data ? `${data.counts.entities} 个建筑组 · ${data.counts.base_assets ?? data.counts.assets} 份原资产${data.counts.massing_parts ? ` + ${data.counts.massing_parts} 个补全部件` : ''}` : '正在读取场景…'}</div>
        {settings.collisionEnabled && contacts.length ? <div className="collision-map-note"><strong>{selectedContact ? selectedContact.label : '当前接触位置'}</strong><span>{selectedContact ? `${collisionStatus(selectedContact).label} · 选中后显示局部范围` : settings.view === 'city' ? '邻近记录合并为位置热点 · 点击查看周边' : '点击热点或左侧记录查看周边'}</span></div> : null}
        {data?.background && !settings.single ? <div className="background-legend"><i/>{settings.backgroundMode==='coastal'?'立体陆地 · 平面水域':settings.backgroundMode==='pixal3d'?'原始 Pixal3D · 轻量预览':settings.mapSolid&&settings.backgroundMap&&data.background.files.map_solid?'RGB地图 · 陆地加厚':'RGB地图 · 平面参考'}{settings.backgroundElevated ? ` · ${settings.elevatedStyle==='highlight'?'橙色：':'纹理：'}升高表面` : ''}<span>当前建筑 · 参考背景</span></div> : null}
        <div className="navigation"><button className="compass" title="俯视" aria-label="切换俯视" onClick={() => engine.current?.top()}><span>N</span><i style={{ transform: `rotate(${bearing}deg)` }}/></button><div className="zoom-buttons"><button title="回到选中建筑" aria-label="回到选中建筑" onClick={() => engine.current?.select(selectedId)}><Icon name="home"/></button><button aria-label="放大" onClick={() => engine.current?.zoom(.75)}><Icon name="plus"/></button><button aria-label="缩小" onClick={() => engine.current?.zoom(1.3)}><Icon name="minus"/></button></div></div>
        {error ? <div className="scene-error" role="alert">{error}</div> : null}
        {!data && !error ? <div className="loading-screen">正在准备建筑场景…</div> : null}
        <div className="canvas-bottom"><div className="interaction-hint"><Icon name="mouse"/><span>拖动旋转 · 滚轮缩放 · 右键平移</span></div><div className="load-status" role="status" data-view={settings.view} data-collision-enabled={String(stats.collisions?.enabled || false)} data-collision-selected={stats.collisions?.selectedId || ''} data-collision-markers={stats.collisions?.markerCount || 0} data-collision-patches={stats.collisions?.patchCount || 0} data-collision-assets={stats.collisions?.highlighted?.join(',') || ''} data-selected-loaded={String(stats.selectedLoaded || false)} data-selected-parts-visible={stats.selectedPartsVisible||0} data-colored-parts={stats.coloredParts||0} data-source-outline={String(stats.sourceOutlineVisible||false)} data-loaded={stats.loaded} data-city-loaded={stats.cityLoaded || 0} data-city-desired={stats.cityDesired||0} data-city-paused={String(stats.cityPaused||false)} data-city-texture-max={stats.cityTextureMaxPx||0} data-city-faces={stats.cityFaces||0} data-pending={stats.pending || 0} data-reference-count={stats.rawReferences?.length || 0} data-reference-ids={stats.rawReferences?.join(',') || ''} data-accepted-surface-count={stats.acceptedSurfaces?.length || 0}>
          <i className={stats.pending || (!stats.cityPaused && stats.cityLoaded < stats.cityDesired) ? 'busy' : ''}/>{stats.overview ? `全城模型 ${stats.cityLoaded||0} / ${stats.cityDesired||0}${stats.cityPaused?' · 已暂停':''}` : stats.pendingModels ? `正在加载 ${stats.loaded} / ${stats.desired} 份建筑` : `已加载 ${stats.loaded} 份建筑模型`}
          {stats.overview&&stats.cityBytes ? <small>{formatSize(stats.cityBytes)}</small> : null}
          {!stats.overview && stats.bytes ? <small>{(stats.bytes/1024/1024).toFixed(0)} MB</small> : null}
        </div></div>
        {!settings.single && stats.rawReferences?.length ? <div className="reference-banner" role="note"><strong>{stats.overview?'全城':''}原始观测参考：{stats.rawReferences.join(' / ')}</strong><span>局部预测表面，可碎裂、分层或悬浮；不属于生成建筑模型。</span></div> : null}
        {stats.errors?.length ? <div className="load-warning" role="alert">{stats.errors.join('；')}。可重新选择建筑重试。</div> : null}
        {stats.cityErrors ? <div className="load-warning" role="alert">{stats.cityErrors} 份全城资源未能载入，可点击“一次加载全城”重试。</div> : null}
        {stats.limited && !stats.overview ? <div className="budget-note">附近共有 {stats.total} 份模型，先显示最近一批；其余以结构概览显示，可点击继续查看。</div> : null}
      </section>
      {selectedContact ? <CollisionInspector item={selectedContact} data={data} settings={settings} onSelect={onSelect} onFocus={onCollisionSelect} onClose={() => setSelectedCollisionId(null)} stats={stats}/> : selected ? <Inspector asset={selected} data={data} settings={settings} setSettings={setSettings} onHigh={() => engine.current?.loadHigh()} stats={stats} onExpand={setExpanded} onFocus={onMode} onSelect={onSelect} onTop={()=>engine.current?.top()}/> : null}
    </main>
    {expanded ? <div className="image-modal" role="dialog" aria-modal="true" aria-label={expanded.title} onClick={() => setExpanded(null)}><button className="modal-close" aria-label="关闭图片" autoFocus onClick={() => setExpanded(null)}><Icon name="close"/></button><figure onClick={e => e.stopPropagation()}><img src={expanded.url} alt={expanded.title}/><figcaption>{expanded.title}</figcaption></figure></div> : null}
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
