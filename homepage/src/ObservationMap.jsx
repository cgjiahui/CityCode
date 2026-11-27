import React, {memo, useEffect, useRef, useState} from 'react';

const observationUrl = path => new URL('observations/' + path, document.baseURI).href;

// The scene reports loading progress frequently. The gallery only updates on selection.
export const ObservationMap = memo(function ObservationMap() {
  const [data, setData] = useState(null);
  const [selected, setSelected] = useState(null);
  const dialog = useRef(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(observationUrl('manifest.json'), {signal: controller.signal})
      .then(response => {
        if (!response.ok) throw Error('Input images are temporarily unavailable');
        return response.json();
      })
      .then(setData)
      .catch(error => { if (error.name !== 'AbortError') console.warn(error.message); });
    return () => controller.abort();
  }, []);

  if (!data) return null;
  const {patches, columns, rows} = data;
  const region = data.patch_size_m.map(m => Math.round(m / 100) * 100 + ' m').join(' × ');
  const atlas = observationUrl(data.atlas);
  const active = patches[selected];

  function open(index) {
    setSelected(index);
    dialog.current.showModal();
  }

  function move(step) {
    setSelected(index => (index + step + patches.length) % patches.length);
  }

  function handleKeys(event) {
    event.stopPropagation();
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      move(event.key === 'ArrowLeft' ? -1 : 1);
    }
  }

  return <>
    <aside className="input-card" aria-labelledby="input-heading">
      <h2 id="input-heading">Low-altitude visual input</h2>
      <p>We generate each <b>≈{region}</b> city region from <b>one aerial image.</b></p>
      <div className="input-grid" style={{aspectRatio: columns + ' / ' + rows, gridTemplateColumns: 'repeat(' + columns + ', 1fr)', gridTemplateRows: 'repeat(' + rows + ', 1fr)'}}>
        {patches.map((patch, index) => <button key={patch.id} aria-label={'View aerial image ' + (index + 1)} aria-haspopup="dialog"
          onClick={() => open(index)} style={{
            gridRow: patch.row + 1, gridColumn: patch.column + 1,
            backgroundImage: 'url("' + atlas + '")', backgroundSize: columns * 100 + '% ' + rows * 100 + '%',
            backgroundPosition: patch.column / (columns - 1) * 100 + '% ' + patch.row / (rows - 1) * 100 + '%'
          }}/>)}
      </div>
      <button className="input-browse" aria-haspopup="dialog" onClick={() => open(Math.floor(patches.length / 2))}>
        Explore {patches.length} aerial images <span aria-hidden="true">↗</span>
      </button>
    </aside>
    <dialog ref={dialog} className="input-dialog" aria-labelledby="input-dialog-title"
      onKeyDown={handleKeys} onClose={() => setSelected(null)}
      onClick={event => { if (event.target === dialog.current) dialog.current.close(); }}>
      {active && <div className="input-gallery">
        <header><div><h2 id="input-dialog-title">Low-altitude visual input</h2>
          <p>One aerial observation per ≈{region} region</p></div>
          <button aria-label="Close aerial image" onClick={() => dialog.current.close()}>×</button></header>
        <img key={active.id} className="input-full-image" src={observationUrl('detail/' + active.id + '.webp')}
          width="960" height="960" alt={'Low-altitude aerial view of region ' + active.id.replace('SC_P320_', '')}/>
        <footer><button aria-label="Previous aerial image" onClick={() => move(-1)}>←</button>
          <div aria-live="polite"><strong>Image {selected + 1} of {patches.length}</strong>
            <span>Region {active.id.replace('SC_P320_', '')} · Altitude {Math.round(active.height_m)} m</span></div>
          <button aria-label="Next aerial image" onClick={() => move(1)}>→</button></footer>
      </div>}
    </dialog>
  </>;
});
