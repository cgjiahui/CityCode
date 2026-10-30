# Interactive project homepage

The public [CityCode demo](https://cgjiahui.github.io/CityCode/) is a reduced-quality web preview. Full visual quality is available in Unreal Engine. The web package retains the full city layout, all 542 medium building models, ground materials and street objects, with compressed shared textures and fewer close-up tiers.

The complete published website is about 941 MB. Resources are served alongside the webpage on GitHub Pages. The deployment workflow retrieves a checksum-pinned archive from the separate public [web-preview resource repository](https://huggingface.co/datasets/cgjiahui/CityCode-WebPreview), verifies every scene file, and checks the 1 GB site limit. Visitors do not load resources from Hugging Face or the earlier preview domain. The main research dataset retains its separate access settings.

```sh
npm ci
npm run build
```

The build output defaults to `.frontend-build/`. To assemble a local static site from the repository root after copying the built frontend into `docs/`:

```sh
python3 homepage/tools/fetch_preview.py
python3 -m http.server --directory _site 8000
```

Keep the large scene archive outside Git history. `preview-assets.json` pins its download revision, size and SHA256. The Pages workflow caches that archive between deployments. Nearby details load on demand, identical medium/near tiers reuse one model, and immutable geometry and shared textures use a browser cache where available.

Web preview quality is the default on phones, tablets and computers. Mobile visitors can select **Use data saver**, or choose **Controls → Scene quality**. Saver mode limits optional close-up downloads and decoder concurrency; it still loads the base city. The setting reopens the scene and uses `?quality=mobile` for saver mode. A computer is recommended for the best browser experience.

The header links to [arXiv](https://arxiv.org/abs/2610.18342) and [GitHub](https://github.com/cgjiahui/CityCode), and identifies the reduced-quality web representation. No author list, affiliations, or review status are displayed.

An alternative resource origin can still be configured when building with `VITE_CITYCODE_ASSET_BASE_URL=https://assets.example.org/city/`. Such an origin must serve the same resource paths and allow cross-origin reads from the homepage. The default Pages deployment does not require this.
