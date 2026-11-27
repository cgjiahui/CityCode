# City and background viewer

This optional, local, read-only web viewer inspects completed building assets, placements, ground, and observation evidence. It does not invoke a model, image API, or GPU worker. The hosted project demo streams a web-optimized scene. This viewer opens the downloaded research assets on your own machine and is optional for city generation. Download the `scene` profile described in the [release guide](../release/README.md).

## Start

Run from the release root:

```sh
python web_preview/serve.py --port 8794
```

Open `http://127.0.0.1:8794/?background=coastal&buildings=1`. Forward port 8794 when running on a remote machine. The server listens on loopback by default; another port can be selected explicitly. The prepared lightweight source archive includes `web_preview/dist/`. For a source checkout without that directory, build the frontend first using the commands below.

The default scene contains **560 placements**: 542 base building placements and 18 local massing components. They correspond to 540 logical complete-building entities and 562 structure sources. The 16 observed-surface references are separate and disabled by default.

| URL option | Purpose |
|---|---|
| `?background=coastal&buildings=1` | Completed buildings with thickened land and separate water |
| `?buildings=0` | Inspect the background without loading city buildings |
| `?background=pixal3d` | Compare the historical neural background |
| `?background=surface&map_solid=1` | Fused RGB map extruded along the published land outline |
| `?background=coastal&inspect=collisions` | Inspect recorded physical-contact locations |
| `?inspect=massing#113` | Inspect one of the ten local massing-completion sources |

Drag to orbit, right-drag to pan, and scroll to zoom. Single-building mode hides the background; returning to city mode restores the selected layers. The full-scene action returns from local or collision inspection, enables buildings, massing, land, water, and the elevated layer, and disables observed fragments. On small screens, use the layer/building drawer.

## Models, detail, and material

The default city uses lightweight textured derivatives totaling approximately 133 MB. Full-city modes can load actual 0.1M models, approximately 2.73 GB, or 0.25M models, approximately 5.64 GB. Here M denotes the target triangle budget in millions; actual meshes vary. Complete packaged assets provide those two LODs, while a full 1M collection requires separate generation and review.

Neighborhood mode loads the selected building or component and nearby 0.1M models. The fine-model action upgrades the selected group to its original 0.25M geometry and textures. Browser texture limits reduce memory consumption without modifying server GLBs. Formal UE assembly defaults to 0.25M in `workflow/04_citycode_assembly/standard.json`.

The viewer preserves glTF node transforms, then applies the binding matrix in UE XYZ metres, Z-up. The shared model for sources 265/271 is instantiated once; related parts such as 553/570 and 406/408 remain grouped. Inference-derived components do not increase the independent-building count.

The neutral-matte display temporarily disables metallic/roughness reflections while retaining base color; original-material mode restores imported settings. Baked bright areas, missing geometry, or unobserved appearance can remain. Lightweight previews omit reflective maps, so they cannot restore material channels that were removed during simplification. Software-rendering detection reduces resolution, texture size, and nearby detail; a loaded scene is not a hardware frame-rate benchmark or a full manual quality audit.

## Background modes and coordinates

The default coastal model uses the published land outline, a thickened base, and independent planar water. Water is at −3.5 m and the base bottom at −8 m. The shoreline/thickness inspection action moves to a near-shore view; hiding water exposes the base. Building coordinates and placements do not change when the background mode changes.

Surface mode displays the fused RGB map and optional numerical height surfaces. Extruded-map mode has a flat land top at 0 m, water at −3.5 m, and bottom at −8 m; this display thickness is not newly measured terrain. With extrusion disabled, the complete RGB plane lies at −1.2 m and the numerical surface can be shown separately. Extrusion temporarily hides the overlapping height surface. Historical height holes remain: visible RGB below a hole does not imply recovered ground geometry.

The elevated layer is a separate numerical height surface, orange by default, with an option to use fused RGB. It does not include a complete bridge underside or piers. `elevated=1` explicitly enables it. Its XY intersections with ground features are not a road-connectivity graph.

The original neural background is `data/smallcity_p384_background_history/reference_mesh/pixal3d_background.glb`, approximately 936 MiB. The viewer loads its derivative `data/smallcity_p384_background_web_preview/pixal3d.glb`, approximately 26.7 MiB, with 444,411 triangles and 182 parts. Simplification preserves generated, elevated, and continuous-ground parts and world position; textures are reduced and reflective maps are removed. This derivative does not replace the original for fine geometry evaluation.

Historical Pixal3D GLBs use `(UE X, UE Z-height, UE Y)`. The viewer swaps Y/Z once to match UE Z-up and does not apply a building-placement matrix to the background. CPU height-surface exports already use UE Z-up. Background manifests retain their historical reference-assisted attribution; they are overlaid with the current building results for inspection. Road-reference data is separate in `data/smallcity_p384_gt_roads/`.

## Massing and observation inspection

The massing toggle compares 542 base placements with the completed 560-placement scene. Ten local sources use 18 reused podium, wing, or strip components. Missing volume and appearance are approximate inferences, not new measurements or confirmed additional buildings.

The before/after/components-only controls change actual 3D visibility under the same camera. Components-only mode hides surrounding buildings and observation fragments. Footprint outlines show the extracted source projection; optional component colors help distinguish parts. The evidence panel displays saved observations and review comparisons. Source IDs are retained, with component links such as `/#MASSING_PGB_SC_000534_01` and ordinary building links such as `/#574`.

Observed surfaces are separately triangulated predicted XYZ with projected observation colors. Occlusion, depth error, and multiple visible levels can produce colored fragments or strips. These references are not extra generated buildings or an unstructured point-cloud layer. They remain optional and disabled by default.

See [massing completion](../workflow/02_building_assets/MASSING_COMPLETION.md) for evidence, decisions, and geometric checks.

## Physical-contact inspection

The collision view reads `data/smallcity_p384_building_assets_completed/physical_postprocess/collision_preview.json`, bound to the current physical-manifest SHA256. Stale markers are not loaded after a binding change.

The default 11 priority locations include six reviewed building contacts and five local background protrusions. The full set includes 269 building contacts and 21 lane-edge records; recorded vehicle-centerline occupancy is zero. A priority marker is not automatically a severe collision, and a local ground bump does not imply that a whole building floats. Select a list item or hotspot to inspect nearby models and actual-height samples. Display controls do not change model placement.

For a newly audited result, regenerate matching markers from its complete audit:

```sh
python web_preview/prepare_collisions.py \
  --release-root data/smallcity_p384_building_assets_completed \
  --audit-root runs/scene_physics/current
```

Models, placements, road inputs, and ground samples must belong to the same result. Static mesh contact inspection does not replace UE collision simulation or vehicle-traversal validation.

## Authoritative inputs

The server prefers `data/smallcity_p384_building_assets_completed/binding_manifest_physical.json`, falling back to `binding_manifest.json` when no physical result exists. Base assets, massing components, and optional surfaces remain separate in the scene. The server exposes only build files, its scene index, and manifest-referenced resources, with HTTP range support.

To inspect the original 542-placement base cache instead:

```sh
python web_preview/serve.py --assets-root data/smallcity_p384_building_assets --port 8795
```

The base cache retains its historical four enabled observed surfaces and 12 optional references. The completed result makes all 16 optional to avoid overlapping the added massing components.

## Build and regenerate derivatives

A supplied `dist/` runs locally without a CDN. To build or test the frontend:

```sh
cd web_preview
npm ci
npm test
npm run build
python -m unittest discover -s tests -p 'test_*.py'
```

Versions are pinned by the lockfile. React, Three.js, Vite, and meshoptimizer retain their original licenses and notices.

From the release root, rebuild city derivatives on CPU:

```sh
python web_preview/prepare_city.py --workers 4
```

The default output is `data/smallcity_p384_building_assets_completed_web_preview/`, with approximately 2,500 triangles and 256-pixel base-color textures per instance. Matching verified files are reused. The manifest records actual counts, sizes, triangles, and source hashes. Restart the server after rebuilding.

Prepare a new base-background derivative outside the frozen data, then serve it:

```sh
environment/.venv-background/bin/python web_preview/prepare_background.py --output runs/background_preview
python web_preview/serve.py --background-root runs/background_preview --port 8796
```

Add the extruded RGB map using the published land outline:

```sh
environment/.venv-background/bin/python web_preview/prepare_map_solid.py \
  --preview-root runs/background_preview --register
```

The map has independent `land` and `water` nodes in UE Z-up. The input JPEG bytes, closure, bounds, and UV checks are recorded in `map_solid.json`. Use `map_solid=0` for the original plane or `water=0` to hide water.

To rebuild the historical neural preview in a prepared preview directory:

```sh
environment/.venv-background/bin/python web_preview/prepare_neural_background.py \
  --output-root runs/background_preview
```

The tool refuses to overwrite an existing neural GLB and uses bundled meshoptimizer on CPU. Source hashes and per-part checks are written to `pixal3d.json`. To make a lightweight coastal derivative, run `pipelines/background_postprocess/coastal.py --source ... --output ...` on that neural preview and register its path, hash, size, and statistics under `files.coastal` and `coastal` in the browser manifest with `view_only: true`. These operations do not invoke image APIs or GPU inference. Background provenance and full-resolution reconstruction are documented in [phase 03](../workflow/03_background_roads/README.md).
