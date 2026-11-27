export const liteSite = document.querySelector('meta[name="citycode-scene-profile"]')?.content === 'lite';
export const liteUrl = new URL('/CityCode-Assets/', document.baseURI).href;
export const standardUrl = new URL('/CityCode/', document.baseURI).href;
const preferenceKey = 'citycode:auto-lightweight';

export function automaticFallback() {
  try { return localStorage.getItem(preferenceKey) !== 'off'; }
  catch { return true; }
}

export function saveFallbackPreference(enabled) {
  try { localStorage.setItem(preferenceKey, enabled ? 'on' : 'off'); }
  catch { /* The current tab can still keep the choice in React state. */ }
}

// Only abandon an incomplete base scene. Near-detail failures leave a usable city.
export function watchScene(viewer, onFailure) {
  let previous, lastProgress = performance.now();
  function resetClock() { lastProgress = performance.now(); }
  const timer = setInterval(() => {
    if (document.hidden || viewer.disposed) { resetClock(); return; }
    const status = viewer.status;
    if (status.coreError) { stop(); onFailure(); return; }
    if (status.error || viewer.baseReady()) { stop(); return; }
    const signature = [status.phase, status.loaded, status.elevatedReady,
      viewer.assets.stats.networkBytes, viewer.assets.stats.cacheBytes].join('/');
    if (signature !== previous) { previous = signature; resetClock(); }
    else if (performance.now() - lastProgress >= 60000) { stop(); onFailure(); }
  }, 1000);
  document.addEventListener('visibilitychange', resetClock);
  function stop() {
    clearInterval(timer);
    document.removeEventListener('visibilitychange', resetClock);
  }
  return stop;
}
