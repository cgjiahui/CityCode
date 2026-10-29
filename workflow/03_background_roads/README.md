# 03: Background reconstruction, surfaces and road references

This stage replays the supplied background on a CPU from saved observations, image-completion responses and numerical heights. It does not submit image requests or run neural inference during standard replay. Sources remain attributed as `historical_gt_assisted`; the historical background is not presented as newly generated from the current 562 building sources.

Use [the release guide](../../release/README.md) to select data downloads, and [the viewer guide](../../web_preview/README.md) to inspect the result. The default `scene` download provides the assembled background and preview resources; full historical image replay requires the `background-replay` package and its dependencies.

## Data and code map

All paths below are relative to the release root.

| Resource | Location | Role |
|---|---|---|
| CLI and numerical implementation | `src/citycode_background/` | Validation, observation fusion, image replay, reprojection and two-layer meshing |
| Frozen input index | `data/smallcity_p384_background_history/manifest.json` | Observation/tile relationships, relative paths and frozen policies |
| Observation patches | `background_history/patches/<id>/` | Raw BEV, edit input, building/unobserved masks, repaired image and local coordinates |
| Editing tiles | `background_history/tiles/<id>/` | 42 image-model inputs and masks |
| Saved generation results | `background_history/prompts/`, `generated/v9/`, `generated/v10/` | Prompts, 42+18 response images and sanitized receipts |
| Global images | `background_history/global/` | Input, V9, water-restored V9 and final RGB |
| Numerical heights | `background_history/height/completed/` | Ground/elevated Z, confidence, tiers, provenance and `pixel_to_world.json` |
| Historical neural background | `background_history/reference_mesh/pixal3d_background.glb` | Complete saved Pixal3D result |
| CPU surfaces and result index | `data/smallcity_p384_background_completed/` | Two-layer mesh, validation and references to the unique saved RGB/patch/neural resources |
| Coastal result | `data/smallcity_p384_background_coastal/` | Thickened land, planar water and the supplied land outline |
| Compatibility report | `data/smallcity_p384_background_compatibility/` | Comparison of current footprints with the historical exclusion mask |
| Optional Pixal3D inputs | `data/smallcity_p384_background_pixal3d_inputs/` | Compressed height-input differences and patch coordinates |
| Optional neural pipeline | `pipelines/background_pixal3d/run.py` | CPU preparation, resource checks, GPU generation, projection, fusion and validation |
| Road-reference package | `data/smallcity_p384_gt_roads/` | Directed lanes, surfaces, ground/layered graphs and elevated references |

Within the abbreviated `background_history/` rows, the full prefix is `data/smallcity_p384_background_history/`. Configuration and historical acceptance records are [standard.json](standard.json), [VALIDATION.json](VALIDATION.json), [PIXAL3D_VALIDATION.json](PIXAL3D_VALIDATION.json) and [SOURCE_PROVENANCE.json](SOURCE_PROVENANCE.json). Historical source paths are attribution only; execution does not require the original research directories.

### Distinct patch and tile counts

| Layer | Count and dimensions | Meaning |
|---|---|---|
| Observations | 79 x 2560 x 2560 | Camera-derived local BEVs, including their small rotation/axis differences |
| Image-editing tiles | 42 x 2048 x 2048 | A 7-column, 6-row grid with blending overlaps |
| Saved responses | 42 V9 images and 18 V10 images | Frozen model outputs, not new requests |
| Global RGB | 13227 x 11094 at 0.15 m/pixel | Final fused and water-restored city image |
| Repaired observations | 79 x 2560 x 2560 | Final RGB reprojected into each original local BEV |

## Standard offline replay

From the release root, install a separate background environment to avoid changing the building-generation environment:

```sh
python -m pip install -c environment/background.requirements.txt -e ".[background]"
city-background verify --bundle data/smallcity_p384_background_history
city-background replay --bundle data/smallcity_p384_background_history \
  --output runs/background_replay
```

The supplied offline environment can also be prepared without network access:

```sh
environment/.venv-generation/bin/python environment/bootstrap_background.py
environment/.venv-background/bin/city-background replay \
  --bundle data/smallcity_p384_background_history --output runs/background_replay
```

The module form is `python -m citycode_background`. Standard replay requires OpenCV package version 5.0.0.93. Historical checks with 4.10 differed at 773 boundary pixels, so the entry point checks the version rather than ignoring differences. The complete tested numerical constraints are in [environment/background.requirements.txt](../../environment/background.requirements.txt).

Replay performs six operations:

1. Project 79 raw BEVs into a common world-XY grid, choosing the closest patch center where observations overlap. Reconstruct edit inputs using the frozen building and unobserved masks, and compare their pixels with the saved inputs.
2. Verify the 42 editing crops, then blend the 42 V9 responses with normalized two-dimensional overlap weights and round to RGB.
3. Restore saved outer-water pixels only inside the supplied water mask, preserving the recomputed V9 elsewhere. The original water synthesis used a fixed seed and feathering; replay consumes that frozen result rather than synthesizing it again.
4. Blend 18 V10 boundary responses and four water tiles while protecting existing water. Twenty interior tiles reuse V9 results.
5. Reproject the final RGB into 79 local BEVs using the original pixel/world transforms and OpenCV Lanczos4 sampling. Compare each result with its frozen counterpart.
6. Mesh the frozen ground and elevated heights independently, then reload the GLB to check triangle counts and finite coordinates.

Missing caches, changed hashes, dimensions or pixel mismatches fail explicitly. There is no fallback that submits a new paid image request. Outputs must be new directories outside the frozen input package; failures retain `RUN_STATUS.json`. Start another new directory to retry. `--rgb-only` skips meshing.

Full-resolution replay uses memory-mapped arrays and streaming PNG I/O. Allow at least 2 GiB free RAM and 30 GiB additional disk workspace for decoded arrays and outputs. Temporary data is created beside the output and cleaned after success; do not use a memory-backed `/tmp` for large runs. Linux requires a C compiler and libpng development headers/libraries, for example `build-essential libpng-dev` on Debian/Ubuntu. The bundled PNG helper is compiled locally without downloading dependencies.

To rebuild surfaces only:

```sh
city-background mesh --bundle data/smallcity_p384_background_history \
  --output runs/background_mesh --stride 16 --texture-size 4096
```

Default stride 16 corresponds to roughly 2.4 m mesh spacing; texture long edge is 4096. The original 0.15 m heights and full-resolution RGB remain unchanged. Every fine pixel in a coarse mesh cell must be valid, preventing triangles across excluded holes. Smaller strides increase geometric density, memory and file size.

CPU surface GLBs use UE XYZ metres with Z up, image columns along +X and rows along +Y. Ground and elevated layers may occupy the same XY coordinates. Z is world height, not camera distance. Do not apply a building placement matrix again to a world-space surface.

## Height and image provenance

Completed heights come from historical prediction bias correction, observation fusion, regularization and filling; they are not a fresh depth estimate from the repaired RGB. Historical support masks include reference-derived building, land and road-layer information. The height lineage and fill/provenance codes remain in `height/LINEAGE.json`. Standard replay starts from these completed arrays and does not rerun their upstream fusion.

The saved Pixal3D result is a different product from the CPU height mesh. Its historical numerical input included reference-height filling, and continuous base fill contributes to coverage. Preserve that attribution instead of describing it as a purely predicted height mesh. The original 936 MiB model remains unchanged; direct viewing does not require neural weights.

The historical exclusion footprint differs from the current 562-source structure. Their approximately 81.3% mask IoU is a compatibility comparison, not an image-quality score or a guarantee of geometric alignment:

```sh
city-background compare-structure --bundle data/smallcity_p384_background_history \
  --footprints data/smallcity_p384_structure/release/global_building_footprints.geojson \
  --output runs/background_compatibility
```

Textured repaired pixels may still lack a valid height because the historical height arrays retain building holes. New observations or buildings require new BEVs, exclusion masks, image completion and height projection/fusion/filling; replacing a few images in the strict replay package is not a validated new-scene workflow.

## Preparing new image requests

The 60 formal responses comprise 42 V9 requests and 18 V10 boundary revisions. Exploratory calls are outside this count. This command exports request inputs and a manifest only:

```sh
city-background prepare-requests --bundle data/smallcity_p384_background_history \
  --output runs/background_requests
```

A new completion run requires a separately configured image service or local model and may incur charges. The background module does not provide a service executor for these requests, and new outputs need not match historical pixels.

## Optional Pixal3D generation

`pipelines/background_pixal3d/run.py` is separate from CPU replay. It starts from supplied numerical heights, confidence/provenance arrays and transforms; the historical raw-reference preparation tool is outside the public entry points. New scenes must supply the same numerical input format.

```sh
# CPU input preparation; omit --patch-id to prepare all 79 patches.
environment/.venv-background/bin/python pipelines/background_pixal3d/run.py prepare \
  --output runs/background_pixal3d
# Inspect future commands without running GPU work or downloading models.
environment/.venv-background/bin/python pipelines/background_pixal3d/run.py plan \
  --work runs/background_pixal3d
# Check resources, then explicitly run in a prepared Pixal3D GPU environment.
python pipelines/background_pixal3d/run.py check \
  --models-root /absolute/local/models --torch-cache /absolute/local/torch_cache
python pipelines/background_pixal3d/run.py generate \
  --work runs/background_pixal3d \
  --models-root /absolute/local/models --torch-cache /absolute/local/torch_cache
```

The pipeline prepares 2560-pixel patches, injects height constraints into 32-cubed/96-cubed support, generates at resolution 1536 with 4096-pixel PBR textures, projects ground/elevated surfaces independently, assigns overlap triangles to the nearest patch center, and adds a 1 m continuous base shifted down 0.03 m. Merge, texture conversion and coverage/seam checks follow.

Source is in `pipelines/background_pixal3d/experiments/`; upstream inference code/licenses are in `third_party/Pixal3D/` and `third_party/NAF/`. Resource identities and source modifications are recorded in `MODEL_RESOURCES.json` and `SOURCE_MANIFEST.json` in that pipeline directory.

The model root must contain `Pixal3D/`, `dinov3-vitl16-pretrain-lvd1689m/`, `moge-2-vitl/` and `BiRefNet/`; the torch cache needs `hub/checkpoints/naf_release.pth`. This MoGe checkpoint estimates cameras and is different from stage01's P384 fine-tuned depth checkpoint. The roughly 25.2 GiB historical model chain is not bundled with this optional background pipeline.

Dependencies include CUDA PyTorch, nvdiffrast, o_voxel, cumesh, flex_gemm, natten and an appropriate attention backend; the historical configuration used Flash Attention 3. See `third_party/Pixal3D/requirements.txt`. Stage02's TRELLIS environment is not a validated substitute. `check` verifies local resources, not end-to-end GPU operation; GPU commands enforce offline execution. There is no newly validated one-command GPU environment or guarantee of byte-identical neural resampling.

The compressed numerical-input differences are about 573 MiB and reuse RGB and matching elevated arrays from the history package. Two historical patches passed exact CPU preparation checks recorded in [PIXAL3D_VALIDATION.json](PIXAL3D_VALIDATION.json); these checks did not execute neural inference. Temporary expanded arrays can be discarded after a run.

## Thickened land and planar water

The result is `data/smallcity_p384_background_coastal/background.glb`. Like the original Pixal3D GLB, it stores `(UE X, UE Z-height, UE Y)`; the viewer converts it to UE Z-up. This differs from the CPU surface GLB above.

The supplied `land.geojson`, 1 m mask and source hashes preserve a historical reference-height-derived visible land extent. It is not a native shoreline annotation; sub-metre details may be lost. Replay uses the supplied outline directly; a new scene supplies its own outline in the same GeoJSON format.

Coastal processing clips water triangles with interpolated UV/height, retains base-color texture, creates shore walls and a closed base, and adds an independent water plane. Default floors are 0 m for generated ground detail, -0.03 m for continuous underlay, -0.06 m for the plinth top and -8 m for its bottom; water is -3.5 m. Existing higher terrain remains. The base itself is closed, but decorative generated surfaces remain separate meshes, so the entire combination is not a single watertight mesh. The output does not retain original normal/metallic/roughness maps.

```sh
environment/.venv-background/bin/python pipelines/background_postprocess/coastal.py \
  --source data/smallcity_p384_background_history/reference_mesh/pixal3d_background.glb \
  --output runs/coastal_check/background.glb
```

Use a new output file. Options `--land`, `--water-z`, `--bottom` and `--land-floor` select the outline, water level, base depth and ground floor. Input/output hashes and parameters are written beside the GLB; temporary chunks are cleaned on success or failure. Dependencies are NumPy, trimesh, Shapely and mapbox-earcut.

The viewer uses the separately processed lightweight `background_web_preview/coastal.glb`, not the full original model. Planar water can be hidden to inspect walls/base, and the original Pixal3D preview remains available for comparison. The viewer uses logarithmic depth buffering for the city-scale view.

## Road references and assembly handoff

`data/smallcity_p384_gt_roads/manifest.json` describes the supplied road-reference package. It retains its original attribution and contains 15 formal source files, excluding six older exploratory variants and intermediate rasters.

| Layer | Contents and intended use |
|---|---|
| `source/` | UE ZoneGraph export: 2,457 road surfaces, 12,279 directed XYZ lanes and section adjacency; select the 3,337 lanes tagged `Vehicle` for cars and preserve `outgoing_lane_ids` direction |
| `ground_only/` | 164 nodes / 278 edges for ground-road planning; its 0.68 m height is only a display-reference plane |
| `height_aware/` | 231 nodes / 343 edges: 307 ground, 30 ramp and 6 elevated; retain layer distinctions |
| `elevated_reference/` | 36 two-dimensional reference edges and 60 support surfaces; resolve edge XYZ from `height_aware` using `edge_id` |

```sh
python pipelines/road_reference/verify.py
```

Validation checks hashes, unique IDs, lane/section references, graph endpoints, finite XYZ and the elevated subset. It does not certify traffic simulation or collision behavior. Sixty support surfaces are not sixty bridges, and an elevated height surface is not a complete bridge asset or drivable topology.

[Stage04](../04_citycode_assembly/README.md) combines current building bindings, coastal geometry, predicted elevated structure, the supplied lanes/graphs and local City Sample assets. Its input contract is `workflow/04_citycode_assembly/standard.json`. UE positions convert metres to centimetres by multiplying by 100; the background's Y/Z swap is applied once, while buildings use their own binding matrices. Native Epic geometry, materials and textures are resolved or generated in the user's project rather than redistributed in this package.
