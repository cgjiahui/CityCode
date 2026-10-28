# CityCode: From Observations to an Assembled City

CityCode recovers metric building structure from image observations, associates or generates building assets, prepares a city background, and assembles the result in Unreal Engine. The supplied SmallCity example includes completed assets and offline replay inputs.

Source code, scene data, neural-network weights, and offline dependencies are distributed separately. Start with the [download guide](release/README.md) and [data inventory](data/README.md). This source candidate is prepared for publication; public resource downloads and project-specific distribution licenses remain pending. See the [project homepage guide](homepage/README.md) for the interactive scene demo.

## Start here

| Task | Guide |
|---|---|
| Download and view the completed city | [Data packages](release/README.md), [web preview](web_preview/README.md) |
| Recover structure from the supplied prediction cache | [01: perception and structure](workflow/01_perception_structure/README.md) |
| Restore existing buildings or generate new assets | [02: building assets](workflow/02_building_assets/README.md) |
| Replay the background or prepare a new background model | [03: background and roads](workflow/03_background_roads/README.md) |
| Import into City Sample and assemble a UE map | [04: CityCode and UE assembly](workflow/04_citycode_assembly/README.md) |
| Locate observations, models, maps, or generation inputs | [Data directory](data/README.md) |
| Check current delivery and validation status | [STATUS.json](release/STATUS.json) |

The four stages share world coordinates and relative resource references. Building assets and the background form two branches after structure recovery; they meet at scene assembly.

## Repository layout

```text
CityCode/
  README.md                         Project overview and entry points
  THIRD_PARTY_NOTICES.md             Attribution and resource-license boundaries
  pyproject.toml                    Python dependencies and command entry points
  src/
    citycode_perception/            Input contracts, segmentation, depth, CLI
    city_pipeline/                  Shared geometry and structure recovery
    citycode_assets/                Building generation, registration, repair, export
    citycode_background/            Saved-image replay and height-field meshes
    citycode_scene/                 Static scene checks and physical placement review
    citycode_core/                  CityCode compilation, validation, UE export
  workflow/
    01_perception_structure/        Stage guide and the standard structure recipe
    02_building_assets/             Building guide, format, training, completion recipe
    03_background_roads/            Background guide and replay policy
    04_citycode_assembly/           UE guide, assembly and physical-review settings
  pipelines/
    background_pixal3d/             Optional background inference orchestration
    background_postprocess/         Land clipping, solid base, independent water
    road_reference/                Road-reference integrity checks
    scene_enrichment/              Predicted elevated roads and street-object layout
    scene_postprocess/              Static scene audit and frozen-decision replay
    ue_citycode/                   CPU preparation and Windows UE adapters
  web_preview/                     Read-only scene viewer, source and prebuilt frontend
  data/                            Separately installed inputs, results and replay caches
  models/                          Separately installed depth and building-network weights
  environment/                     Environment bootstrap scripts and dependency manifests
  third_party/                     Attributed inference dependencies and licenses
  tests/                           Geometry, interface, data and workflow regression tests
  release/                         Download catalogs, packaging tools and validation records
```

`workflow/` holds instructions and recipes, `src/` holds reusable implementation, and `pipelines/` connects these modules to batch processing and UE. Files under `data/` are inputs or results, not Python packages. Neural weights under `models/` are distinct from the generated building GLBs.

The local `build/`, root `dist/`, `runs/`, virtual environments, wheelhouses and `node_modules/` are excluded from the lightweight source archive. `web_preview/dist/` is deliberately included so the prebuilt viewer can start without a frontend build.

## View the supplied scene

Use Python 3.10 or newer in an independent environment:

```sh
python -m pip install -e .
python release/bundles.py plan --profile scene
```

Once a data hosting base URL is available, set `CITYCODE_DATA_BASE_URL` and install the scene packages:

```sh
python release/bundles.py install --profile scene \
  --downloads release/downloads.json --base-url "$CITYCODE_DATA_BASE_URL"
python web_preview/serve.py --port 8794
```

Open `http://127.0.0.1:8794/`; forward that port when working remotely. The source archive includes the built frontend. For an editable source checkout, frontend build instructions are in the [preview guide](web_preview/README.md). Local archive installation is also supported by the [download guide](release/README.md).

The default preview displays the completed buildings, thickened land and independent water. The viewer provides lightweight meshes for browsing and can load the actual 0.1M or 0.25M assets on request. It does not regenerate models.

## Replay or regenerate

Download the `replay` profile before using the supplied observations and prediction caches:

```sh
city-perception check-inputs --dataset data/smallcity_p384 --checksums
city-perception recover-cached --cache data/smallcity_p384_recovery_cache \
  --output data/smallcity_p384_structure_reproduced
city-perception verify-structure --run data/smallcity_p384_structure_reproduced
```

Use a new output directory. `recover-cached` and fresh `recover` use the same `standard` structure recipe, but start from different inputs. The cached example starts from saved predicted observations, point clouds and calibrated depth. Historical raw neural-depth files are unavailable; cached replay is not a fresh RGB-to-depth inference run. New scenes can use `predict-depth`, segmentation generation/import and `recover`; see [stage 01](workflow/01_perception_structure/README.md).

The default building workflow restores saved models, replays reviewed CPU operations and applies the frozen local-completion recipe. Optional regeneration uses saved images and generation conditions, then requires new image/model review. It has separate outputs and does not inherit the example's grouping or completion approvals. Start with [stage 02](workflow/02_building_assets/README.md).

Background replay uses the 60 saved image-edit responses and supplied numerical fields. It does not submit image API requests or run neural inference. Optional Pixal3D generation has separate model and GPU requirements; see [stage 03](workflow/03_background_roads/README.md).

## Scene contents

| Quantity | Meaning |
|---|---|
| 79 observations | One `tilt_light` RGB view per patch |
| 562 structure sources | Recovered source regions; not all represent independent buildings |
| 540 logical building entities | Grouped complete-building identities |
| 542 base placements + 18 massing parts | 560 enabled model placements in the completed scene |
| 16 observation surfaces | Optional visual references, disabled by default |
| 550 optional generation units | 411 reused-image units, 123 saved generated units and 16 component units |

Completed assets provide 0.1M and 0.25M variants. A complete 1M scene requires separately generated/exported, reviewed and bound assets; a missing high-resolution asset never silently falls back to another LOD.

The authoritative physical-placement binding is `data/smallcity_p384_building_assets_completed/binding_manifest_physical.json`. The pre-physical-review binding, source associations and frozen decisions remain available. [Stage 02](workflow/02_building_assets/README.md) explains the model and source distinctions.

## UE assembly

The supplied adapter was validated with UE 5.0.3 and a complete City Sample project compatible with UE 5.0. It defaults to 0.25M buildings. CPU preparation produces coordinate-baked import copies, background parts and predicted elevated-road geometry; Windows scripts import resources, save a new map and read it back independently.

Download the published building GLBs and maps, or use a newly reviewed building binding. The local import copies are generated under `build/`; the old machine-specific CityCode snapshot is excluded from data downloads. `finalize` compiles a CityCode scene for the actual local assembly. Follow the [stage 04 guide](workflow/04_citycode_assembly/README.md).

Vehicles, lamps, signals, trees, benches, bridge components and materials are resolved from the user's City Sample project. Native Epic assets and derived bridge meshes are not included as redistributed model files. Elevated alignment comes from predicted structure and explicitly recorded connection/design decisions, not copied original bridge-actor transforms.

## Scope and validation

The example provides completed results, offline replay inputs, inference adapters, reviewed placement/completion decisions and UE assembly code. Capture tooling, a general autonomous-agent service, an independent benchmark package and migrated training entry points are outside this release.

Runtime inputs are supplied resources; private original annotation files are not an additional prerequisite. Historical background fields, coastline and road references include GT-assisted sources, and a few historical building-review decisions used reference assistance. Their provenance remains recorded. The old background was not regenerated from the current building structures.

New models and new scenes need their own review. Static mesh/contact checks do not establish traffic simulation, physical bridge certification or complete UE collision behavior. Current test results, package-install checks and execution limits are recorded in [release/STATUS.json](release/STATUS.json). Historical run identities are retained separately; the current source identity is [release/SOURCE_MANIFEST.json](release/SOURCE_MANIFEST.json).

The Python wheel installs the reusable library and CLI commands. The full source archive also carries `workflow/`, `pipelines/`, Windows scripts, download tools and the viewer; the wheel alone is not the full scene reproduction bundle.

Project-code, observation-data, generated-content and weight distribution licenses remain to be finalized. Existing third-party copyright and license files are preserved; see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
