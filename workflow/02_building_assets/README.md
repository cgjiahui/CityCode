# Building assets: replay, generation and review

Stage 02 turns Stage 01 structures and observations into textured assets, explicit source associations, world placements and building groups. Its sequence is:

**Structure and observations → appearance completion → generation or asset reuse → registration and repairs → scene review and local massing → assembly bindings.**

This page is the execution guide. [Binding format](BINDING_FORMAT.md) defines the data contract, [local massing](MASSING_COMPLETION.md) explains inferred additions, and [training](TRAINING.md) documents the selected networks. [Resource manifests](MODEL_RESOURCES.json) identify weights and dependencies. Download groups are described in the [release guide](../../release/README.md).

## Cached result replay

Run commands from the release root with Python 3.10+, the declared CPU dependencies and Node.js. Node 24 was used for the special low-resolution export checks. Cached replay does not need CUDA or appearance-service credentials.

```sh
python -m pip install -e .
city-buildings --config workflow/02_building_assets/unified.standard.json
```

The standard configuration selects `execution_mode=verified_asset_cache`. It checks input identities, restores the supplied models, replays the recorded CPU operations and applies the frozen local-massing recipe. Missing caches cause an error; they do not trigger image requests or GPU generation.

| Input | Role |
|---|---|
| `data/smallcity_p384_structure/building_generation_inputs/` | Structures, footprints, heights, roof layers, proxies, observations and cameras |
| `data/smallcity_p384_building_assets/` | Accepted base-model cache, both LODs, appearance images, registration and review evidence |
| `data/smallcity_p384_massing_completion_inputs/` | Local-massing observations, measured geometry and review evidence |
| `workflow/02_building_assets/` | Cache inventory, dispositions, runtime configuration and massing recipe |
| `data/smallcity_p384_reused_generation_inputs/` | Optional regeneration inputs; the standard cache replay does not consume this bundle |

Keep the sibling-directory layout. Model paths are relative to their binding manifest; inputs and evidence are pinned by full SHA256, not merely by matching filenames or sizes.

Replay reads `example.cache.json`, restores the base models, and reruns source-supported orientation initialization, extrema-preserving low-LOD export, cross-LOD hole repair and frozen scene inclusion decisions. It then applies `massing.standard.json`, checks the donor files, transforms and evidence, and validates the final bindings.

The rebuildable intermediate directory is `data/smallcity_p384_building_assets_reproduced/`. The authoritative result is `data/smallcity_p384_building_assets_completed/binding_manifest.json`; the original cache is preserved.

| Object | Count |
|---|---:|
| Structural sources retained | 562 |
| Sources associated with base models | 543 |
| Base-model placements | 542 |
| Logical complete-building groups | 540 |
| Added local massing parts | 18 |
| Total model placements | 560 |
| Optional observation surfaces, disabled by default | 16 |

Sources 265/271 share one physical model; 406/408 and 553/570 retain their building-group relationships. Sources 499/514 remain recorded without additional placements. The 18 massing parts represent inferred podiums, wings or edge volumes, not 18 newly confirmed buildings.

Repeating the command resumes the same input identity. To create another output, copy the configuration and change both `output` and `asset_cache_restore_output`; outputs must remain siblings of the structure package. `prepare` checks inputs and `status` reads progress. In cache mode, `run`, `validate` and `finalize` restore and check the complete result; image and GPU phases report that they are unnecessary.

Historical acceptance included a fresh output-directory replay with a byte-identical final binding. Special checks reproduced source 11's low LOD and source 512's two repaired LODs; source 293's orientation initialization matched its frozen transform. These records describe cached-result acceptance, not a fresh neural batch.

## Asset and scene validation

```sh
city-assets validate-assets \
  --manifest data/smallcity_p384_building_assets_completed/binding_manifest.json \
  --require-complete
```

This checks source dispositions, relative paths, file identities, materials, normals, coordinates and consistency between the supplied `0.25M` and `0.1M` models. Add `--geometry` to recompute actual triangle projections, coverage and height; this is more expensive than checking the frozen geometry reports and their current model hashes. Geometry scores compare against the source structure and do not replace visual scene review.

Stage 04 instantiates each `asset_instances` and `massing_instances` entry once, then preserves `building_entities` grouping. LOD variants are alternative representations of one placement; optional `surface_instances` are an evidence layer.

After the buildings and Stage 03 background are ready, replay the separately frozen placement cleanup:

```sh
python pipelines/scene_postprocess/run.py replay --verify
```

This writes `binding_manifest_physical.json`, the preferred input for the viewer and Stage 04. It changes placements rather than the source structures or model bytes. The checks cover building intersections, road clearance and ground contact. Small movements or bounded scaling are preferred; exclusion is a last resort for clear conflicts. Pedestrian-path contact is not an automatic reason to remove a building. New scenes require their own audit and reviewed decisions; see the [assembly guide](../04_citycode_assembly/README.md).

## Packaged-image regeneration

The optional input manifest is `data/smallcity_p384_reused_generation_inputs/manifest.json`.

| Scope | Configuration in this directory | Generation units |
|---|---|---:|
| `reused` | `reused.regenerate.json` | 411 physical assets associated with 412 sources; 265/271 share one asset |
| `generated` | `generated.regenerate.json` | 123 previously generated assets |
| `components` | `components.regenerate.json` | 16 components belonging to 8 logical composite groups |

The supplied addition comprises 411 reused-asset inputs and 16 component inputs, or 427 units. Together with the 123 existing sets, the manifest indexes 550 generation units. This is not a count of independent buildings. Component coarse structures are explicit priors derived from published assets, rather than newly extracted observations.

For `reused`, `image_height_span` first generates Stage 1 structure from the saved completed image, then applies the per-asset scalar `structure_height_voxels` stored in `reused.height_spans.json`. It does not read old Stage 1 coordinates or silently substitute `source_footprint` conditioning. Current structures still support association, registration and acceptance. The other two scopes retain their own conditions. A conditioning change requires new preparation, review and freezing.

First run the read-only check:

```sh
python workflow/02_building_assets/rerun_packaged.py
```

The default is `--scope all --phase check-inputs`. It checks the packaged and shared file proofs and all three generate-mode configurations without creating results, requesting images or starting a GPU. It does not approve image semantics, model quality or a new full GPU batch.

For example, regenerate the reused scope in explicit stages:

```sh
python workflow/02_building_assets/rerun_packaged.py --scope reused --phase prepare
python workflow/02_building_assets/rerun_packaged.py --scope reused --phase review-images
```

Preparation imports the saved images and skips the appearance API. Inspect `reviews/appearance/index.html` under the selected output. Fill `appearance.template.json` with `generate` or `hold` and a reason, then save it as `reviews/appearance.json`.

```sh
python workflow/02_building_assets/rerun_packaged.py --scope reused --phase freeze
python workflow/02_building_assets/rerun_packaged.py --scope reused --phase gpu --gpu-count 2
python workflow/02_building_assets/rerun_packaged.py --scope reused --phase validate
python workflow/02_building_assets/rerun_packaged.py --scope reused --phase review-models
```

The GPU worker must already be allocated and configured. Release temporary GPU resources when that phase finishes; validation and review are CPU steps. Inspect the actual LODs and scene evidence, fill `models.template.json`, and save accepted decisions as `reviews/models.json` before finalization:

```sh
python workflow/02_building_assets/rerun_packaged.py --scope reused --phase finalize
```

Use `--scope generated` or `--scope components` for the other inputs. All phases dispatch to the same `citycode_assets.batch` implementation. Options include `--workers`, `--gpu-count` and an optional `--deadline-seconds`. There is no fixed runtime cutoff by default. `--scope all` processes the three configurations sequentially and applies an explicit deadline separately to each. Set GPU count to match the allocated worker.

The scopes have separate outputs and cannot overwrite the supplied base assets or completed scene. Resume unchanged inputs, or copy a generate-mode configuration and choose a new output for a new batch. Identical images, seeds, weights and height spans do not guarantee byte-identical meshes or textures.

New component meshes need review of source ownership, registration and seams across all eight groups. Old composite approvals and the donor, position and approval of the 18 example massing parts do not transfer to new meshes. Regeneration does not promise the same final placement count.

## New-scene generation

Stage 01 must first export a complete `building_generation_inputs/manifest.json`, including per-building footprints, heights, proxies, observations and cameras. A mask or single crop alone is not this contract. New input counts need not equal 562, and example IDs, exclusions and groups must not be copied into a new scene.

Generation uses `execution_mode=generate`. It rejects configurations carrying the old asset cache or the example massing recipe; restoring old assets is not substituted for inference.

Initialize a new batch from the release root. The following paths are examples that the user supplies, not shipped datasets:

```sh
city-buildings --phase init --config workflow/02_building_assets/my_city.json \
  --input-manifest data/my_city_structure/building_generation_inputs/manifest.json \
  --completed-images data/my_city_completed_images \
  --output data/my_city_building_assets \
  --lod-targets 1000000 250000 100000
```

With `--completed-images`, each file is a 1024×1024 PNG named `<building_id>.png`, depicting the corresponding complete building or confirmed connected group. Images may come from the user's own completion service. Initialization indexes their identities; preparation builds references and imports the images. Neither step sends an image request or approves image semantics. External imports are not represented as requests made by this program.

Without `--completed-images`, prepare the four references and prompt, then explicitly use `--phase images` to call the configured service. Partial imports leave the remaining images pending; `run` does not automatically send them. For filenames that do not match source IDs, use `--completed-images-manifest`:

```json
{
  "schema": "citycode.external_completed_images.v1",
  "records": [{
    "building_id": "MY_BUILDING_001",
    "building_sha256": "full SHA256 of the source building.json",
    "image": "relative/path/to/completed.png",
    "sha256": "full SHA256 of that PNG"
  }]
}
```

Initialization creates `my_city.json`, `my_city.sources.json`, and, for directory image imports, `my_city.images.json`. Complete sources become generation candidates, not automatically independent whole buildings. Before preparation, review the source dispositions and record supported grouping or record-only cases. The new output must be empty and beside its structure package; original inputs remain unchanged.

```sh
city-buildings --config workflow/02_building_assets/my_city.json --phase prepare
city-buildings --config workflow/02_building_assets/my_city.json --phase review-images
```

Review `reviews/appearance/index.html`, fill `reviews/appearance.template.json` with `generate` or `hold` and reasons, and save `reviews/appearance.json`. Templates do not grant approval. Changed structures, images or frozen dispositions require a new batch and review.

```sh
city-buildings --config workflow/02_building_assets/my_city.json --phase freeze
city-buildings --config workflow/02_building_assets/my_city.json --phase gpu --gpu-count 1
```

`freeze` pins inputs, review decisions and model resources. `gpu` runs on an existing CUDA worker, exits when its finite work finishes and does not allocate or retain a server. The default exports are `model_1m.glb`, `model_025m.glb` and `model_01m.glb`, with budgets of 1,000,000, 250,000 and 100,000 triangles; actual counts may be lower. Each is independently exported from the generated mesh. Subdividing a low LOD does not create 1M detail. For two LODs, initialize with `--lod-targets 250000 100000`.

Release temporary GPU resources after inference/export and continue on CPU:

```sh
city-buildings --config workflow/02_building_assets/my_city.json --phase validate
city-buildings --config workflow/02_building_assets/my_city.json --phase review-models
```

Inspect `reviews/models/index.html`, per-LOD front/back and roof views, and scene-intersection evidence. Fill `reviews/models.template.json` with `include` and reasons for accepted models, then save `reviews/models.json`. Decisions must match the actual model files and placements. Candidates needing visual scene-completion qualification also need the corresponding scene evidence.

```sh
city-buildings --config workflow/02_building_assets/my_city.json --phase finalize
```

Only consistent, accepted mesh, registration and review identities are finalized. Held or failed units remain pending. `--phase run` resumes between stages and reports the next missing image, review or CUDA step.

### Recovering an export that lacks only 0.1M

If this batch has valid `0.25M` and, when requested, `1M` exports, but CUDA export failed only at `0.1M`, release the GPU and run:

```sh
python workflow/02_building_assets/rerun_packaged.py --scope generated --phase recover-exports
```

For a custom configuration:

```sh
python workflow/02_building_assets/recover_exports.py --config workflow/02_building_assets/my_city.json
```

The first run creates `reviews/export_recovery.template.json` without modifying meshes. Review the current high-LOD models, choose `recover_0.1M_from_0.25M` with a reason, save `export_recovery.json` beside the template, and repeat the command. It checks source identity, generation records and high-LOD hashes, then uses CPU extrema-preserving simplification to derive `0.1M` from this batch's `0.25M`.

Missing `0.25M`, missing requested `1M`, a corrupted existing low LOD or unacceptable high-LOD registration are rejected. The default relative-error budget is 1%. If only the face target is exceeded, a reviewer may inspect the failed candidate and explicitly set `target_relative_error` to `0.02`, the 2% maximum. Roof/base extrema, connected components, materials and final geometry rules still apply. Failure is not marked complete, and this operation neither substitutes historical models nor approves scene inclusion.

## Single-asset appearance and inference

The example already supplies completed images. For a new building, the low-level commands are:

```sh
city-assets prepare-appearance --building /path/to/building.json \
  --output /path/to/appearance \
  --provider-config workflow/02_building_assets/appearance.standard.json
city-assets complete-appearance --appearance /path/to/appearance \
  --provider-config workflow/02_building_assets/appearance.standard.json
```

The first command prepares the coarse-model render, source observation, roof evidence, context and height hints on CPU. Only the second calls the configured service. Credentials come from an environment variable. Requests and response images retain identity proofs; no automatic retry or service substitution is performed, and new responses need not reproduce historical pixels.

After image review, run on a prepared CUDA worker:

```sh
city-assets generate --config /path/to/runtime.json \
  --building /path/to/building.json --appearance /path/to/appearance \
  --output /path/to/generated
```

The runtime supplies local model paths. The three selected networks generate coarse structure, detailed geometry and material. Explicit `source_footprint` conditioning can use the current footprint and recorded roof layers. CPU resource checks do not prove a successful GPU run.

## Reuse, registration and postprocessing

Source association asks whether an old asset and a current source represent the same building. Compare their original world footprints using IoU, both coverage directions and center distance, then check observations, parent relationships and height. Do not translate or scale footprints to establish identity. IDs may change between extraction runs.

Registration instead asks whether the actual mesh fits the current source. Use real GLB node transforms, projected triangles and height under bounded fitting. Record asset-to-building and building-to-world transforms separately and preserve materials and normals. Arbitrary stretching to maximize footprint overlap is not permitted.

For complete buildings, both LODs must independently reach footprint IoU ≥0.70, target coverage ≥0.80 and asset coverage ≥0.80; refine projections near thresholds. Good source-footprint association does not by itself establish good mesh coverage.

One old building split across current IDs may become one explicitly shared physical asset. Several old assets forming one current building may be composed, but ownership, contact and coverage must be checked and whole-building LODs exported. Unsupported complete-building matches or failed bounded fits require generation or a pending state. Incomplete facade strips or noise may be excluded from asset generation only with observations, reasons and hashes; Stage 01 structures remain. An unusual roof, changed ID, zero association score or temporary generation failure is not sufficient grounds for exclusion.

## Review and postprocessing

For new results, use the current observations, structures, completed images, actual meshes, transforms and neighboring scene. Review outputs before reusing any repair or inclusion decision. The recommended sequence is:

**Generation → registration and geometry checks → actual model/scene review → repair, retention, association or proposed additions → recheck and freeze → assembly bindings.**

`review-models` produces per-LOD views and `reviews/models/scene_intersections.json`. Actual mesh XY intersections are useful evidence, but they do not replace height, neighboring images or reviewer judgment. The [review policy](agent_review_policy.json) distinguishes supported buildings/groups, missing podiums/wings, already-covered regions, orientation or mesh defects, unreasonable patches and unexplained observations.

Store current repairs in `reviews/repairs.json`, bound to asset ID, source-building SHA and input-LOD SHAs. Supported operations include `extrema_lod_if_needed`, `observed_roof_projection`, a new orientation approval and cross-LOD repair recipes. Re-run `validate` and `review-models`, inspect the repaired files, update model decisions, and then `finalize`. To replay the earlier CPU postprocessing contract, use a new sibling output and the retained `postprocess_input_manifest.json`; `city-assets postprocess-assets` rejects already-postprocessed inputs or existing outputs.

Add massing only where credible building observations support a missing physical volume and the candidate has a reasonable scene relationship. A low coverage score or empty mask region alone is insufficient. Review each added part separately; an unacceptable narrow patch does not invalidate reasonable parts in the same group. Uncertain regions may remain empty.

The massing APIs are `construct_manifest`, `measure_manifest`, `render_review`, `prune_recipe`, `apply_recipe` and `validate_massing`. They require explicit candidates, identities, evidence and reviewed decisions; `apply_recipe` also requires the base binding, cache inventory and support-file contract. They are not a general autonomous scene-completion service, and a new-generation `finalize` does not automatically discover missing buildings or apply the example's 18 parts. See [local massing review](MASSING_COMPLETION.md#new-scene-massing-review).

## Module map

| Modules under `src/citycode_assets/` | Responsibility |
|---|---|
| `batch`, `cache_batch` | Source scheduling, identity checks, cached replay and resume |
| `appearance`, `provider` | Reference images, prompts, explicit requests and receipts |
| `generation`, `structure_conditioning` | Network loading, three-stage inference and coarse conditioning |
| `lod`, `extrema_lod` | Material-preserving exports and extrema protection |
| `placement`, `registration`, `orientation` | Initial pose, source-supported heading and bounded fitting |
| `roof_reconciliation`, `mesh_repair` | Observed-roof reconciliation and local triangle repairs |
| `composite`, `postprocess`, `observed_surface` | Composites, CPU repairs and evidence surfaces |
| `batch_delivery`, `scene_selection`, `scene_completion` | Validation, grouping and reviewed scene decisions |
| `massing_completion` | Explicit local-massing recipes and replay |
| `cli` | Individual commands and delivery checks |
