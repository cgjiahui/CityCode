// One entry per observed source; model parts remain separate physical placements.
export const PART_COLORS = ['#e58b36', '#3c9dab', '#9372c3', '#cc6f8f'];

export function massingSourceFor(data, asset) {
  return asset?.massing ? data.massing_sources?.find(source => source.part_ids.includes(asset.id)) : null;
}

export function initialSceneSettings(search) {
  const params = new URLSearchParams(search), inspect = params.get('inspect') === 'massing';
  return { view: inspect ? 'neighborhood' : 'city', cityBuildings: params.get('buildings') !== '0',
    cityQuality: 'preview', cityPaused: false, surfaceDisplay: 'neutral', massing: true,
    massingColors: inspect, sourceOutline: inspect,
    backgroundMode: ['surface', 'pixal3d', 'coastal'].includes(params.get('background')) ? params.get('background') : 'coastal',
    backgroundWater: params.get('water') !== '0', backgroundMap: true, mapSolid: params.get('map_solid') !== '0', backgroundGround: params.get('ground') === '1',
    backgroundElevated: params.has('elevated') ? params.get('elevated') !== '0' : !inspect,
    elevatedStyle: 'highlight', radius: inspect ? 100 : 120, single: false, surfaces: false, acceptedSurfaces: !inspect,
    outlines: false, structures: false };
}

export function completeCitySettings(previous, background) {
  return { ...previous, view: 'city', single: false, cityBuildings: true, cityPaused: false,
    massing: true, massingColors: false, sourceOutline: false, structures: false, outlines: false,
    surfaces: false, acceptedSurfaces: false, collisionEnabled: false,
    backgroundMode: background?.files?.coastal ? 'coastal' : background?.files?.neural ? 'pixal3d' : 'surface',
    backgroundWater: true, backgroundMap: true, backgroundGround: true,
    backgroundElevated: true, elevatedStyle: 'highlight' };
}

export function comparisonSettings(previous, mode) {
  if (!['before', 'after', 'parts'].includes(mode)) throw new Error('Unknown massing comparison');
  return { ...previous, view: mode === 'parts' ? 'single' : 'neighborhood', single: mode === 'parts',
    massing: mode !== 'before', surfaces: false, acceptedSurfaces: false, sourceOutline: true };
}
