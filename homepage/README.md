# Interactive project homepage

The public [CityCode demo](https://cgjiahui.github.io/CityCode/) reads scene models and textures from separate HTTPS storage. This directory contains its editable React/Three.js source. The compiled GitHub Pages site is in `../docs/`.

```sh
npm ci
VITE_CITYCODE_ASSET_BASE_URL=https://citycode-scene-preview.atzuwx8kh3y1.chatgpt.site/city/ npm run build
```

The build output defaults to `.frontend-build/`. Do not add the multi-gigabyte scene resources to Git. The original runtime viewer and its own prebuilt frontend are in [web_preview](../web_preview/README.md).

The header and About dialog link to [arXiv](https://arxiv.org/abs/2610.18342) and [GitHub](https://github.com/cgjiahui/CityCode). The repository is currently private. Set `VITE_CITYCODE_REPOSITORY_PRIVATE=false` when publishing the repository; an empty `VITE_CITYCODE_REPOSITORY_URL` hides that link. No author list, affiliations, or conference/review status are displayed by the homepage.
