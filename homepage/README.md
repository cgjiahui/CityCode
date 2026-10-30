# Interactive project homepage

The public [CityCode demo](https://cgjiahui.github.io/CityCode/) reads scene models and textures from separate HTTPS storage. This directory contains its editable React/Three.js source. The compiled GitHub Pages site is in `../docs/`.

```sh
npm ci
VITE_CITYCODE_ASSET_BASE_URL=https://citycode-scene-preview.atzuwx8kh3y1.chatgpt.site/city/ npm run build
```

The build output defaults to `.frontend-build/`. Do not add the multi-gigabyte scene resources to Git. The original runtime viewer and its own prebuilt frontend are in [web_preview](../web_preview/README.md).

High quality is the default on all devices. Phones and tablets, including iPads with desktop-style browser identifiers, also show a direct button to enter data-saver mode. Saver mode retains the base city, stops global HD-texture refinement and optional native street-object/high-resolution terrain downloads, and limits nearby building detail and decoder concurrency. It still downloads the existing base-city resources; it is not a separately reduced geometry pack. Desktop visitors keep the existing full-quality defaults.

Open **Controls → Scene quality** to choose High quality or Data saver. The setting reopens the scene; data saver uses `?quality=mobile`, and returning to high quality removes the override. Mobile visitors are prompted to use a computer for the best experience.

To migrate scene hosting, copy the same resource paths to a separate HTTPS object store and rebuild with `VITE_CITYCODE_ASSET_BASE_URL=https://assets.example.org/city/`. The storage origin must allow cross-origin GET/HEAD requests from `https://cgjiahui.github.io` and serve the model, texture and JSON content types correctly. This changes the resource host; the homepage can remain on GitHub Pages.

The header and About dialog link to [arXiv](https://arxiv.org/abs/2610.18342) and [GitHub](https://github.com/cgjiahui/CityCode). The repository is currently private. Set `VITE_CITYCODE_REPOSITORY_PRIVATE=false` when publishing the repository; an empty `VITE_CITYCODE_REPOSITORY_URL` hides that link. No author list, affiliations, or conference/review status are displayed by the homepage.
