# 01: Perception and structural recovery

Segmentation, depth estimation and structural recovery form one stage. Predicted building instances and background semantics constrain depth-scale calibration, surface reconstruction and fusion across patches. The outputs are metric building structures and observation evidence for [building asset generation](../02_building_assets/README.md).

## Implementation and execution order

Users invoke one recovery command. Internal checkpoints support diagnosis and resumption; they do not require a sequence of manual commands.

| Responsibility | Implementation | Work performed |
|---|---|---|
| Observation preparation | `src/citycode_perception/` | Input validation, instance/background masks, depth prediction or import |
| Per-patch recovery | `src/city_pipeline/geometry/` | Depth-scale calibration, surfaces, metric footprints and heights |
| Cross-patch association | `buildings/observations.py`, `selection.py`, `consensus.py` | Observation quality, candidate conflicts and stable building identities |
| Structural refinement | `buildings/stage.py` and height/support/visibility modules | Preserve supported roofs, bound trimming, repair footprints and remove supported fragments |
| Export and validation | `buildings/export.py`, `evidence.py`, `verify_buildings.py` | Structural meshes, observations, cameras and geometry consistency |

`buildings/run.py` manages ordering, cache signatures, logs and resumption. `stage.py` dispatches named operations; parameters live in [standard.json](standard.json). Geometry kernels do not schedule the workflow.

The historical 0.2.1 refactor removed an empty review step, two unused intermediate mesh exports and legacy reference-evaluation entry points. It retained intermediate roof partitions used by the algorithm. That acceptance run passed 134 tests and replayed 15 structural checkpoints: all properties for 562 sources and 2,252 mesh files matched the supplied example byte for byte, and 583 closed mesh components passed geometry checks. This is historical validation, not a new inference run. Current release checks are recorded in [release/STATUS.json](../../release/STATUS.json).

Candidate selection, edge-fragment filtering and false-facade filtering already perform instance cleanup. Subsequent review may exclude confirmed false or duplicate sources, mark incomplete real sources for repair, or retain their structure with a simpler asset when appearance evidence is weak. The standard workflow does not apply a per-building deletion list or unapproved review suggestions.

## Observation inputs

The supplied `data/smallcity_p384/` contains 79 physical patches, each with one `tilt_light` view. Required inputs are RGB, camera, `strict_valid_mask`, patch/index metadata and explicit plane priors. The spatial contract is a 384 m data extent, 320 m core, 32 m halo and 160 m stride, in metric world coordinates; `P320` in patch identifiers is a historical name.

Selection commands accept repeated `--patch SC_P320_R+000_C+003` arguments. By default they process every listed observation. The standard profile selects one view per physical patch; use `--view` to choose one when the dataset contains multiple views. Start with one patch when preparing a new scene.

## Segmentation

Two prompts have distinct roles. Their canonical files are in `src/citycode_perception/prompts/`; the local `prompts/` directory contains relative links.

| Prompt | Output contract | Downstream role |
|---|---|---|
| V9.2 joint building instances | One color per building, shared by its roof and facade; roads blue, other pixels black | Building identity and visible pixels for backprojection, footprint/height estimation and fusion |
| V7B planar background semantics | Fixed colors for ground roads, water, other hardscape and elevated roads; buildings/uncertainty black | Background classes, scale anchors, false-instance filtering and background inputs |

Both operate on the same RGB. V9.2 alone lacks the background classes and ground/water anchors; V7B alone lacks individual building identities.

### Use the supplied predictions

`data/smallcity_p384_predictions/masks/` contains the exact 79 x 2 PNGs used by the pipeline: 1024 x 1024 RGB responses, prompt snapshots, a relative-path manifest, available model/time provenance, input/output hashes and checksums. The historical package contains 164 files totaling 148,694,986 bytes; it excludes duplicate working RGB, previews, other views and private API metadata.

The predictions were compared with the published RGB and original E00 inputs. Both geometry readers loaded all 79 results, and 163 checksum entries passed. Pass this directory directly to `recover --masks`; preparation, API execution and reimport are unnecessary for this example. Keep observations and predictions in their sibling `data/` directories after installation.

### Generate or import masks for another scene

```sh
city-perception prepare-masks --dataset data/smallcity_p384 --output runs/mask_requests
```

Request preparation resizes RGB to 2048 x 2048 and requests 1024 x 1024 PNG results. Import preserves original responses and uses nearest-neighbor resizing for geometry masks. RGB, prompt, request and result identities are recorded; unknown model provenance remains unknown.

An external image-editing service may supply results at `structure_v9_2/results/<patch>__<view>.png` and `structure_v7b/results/<patch>__<view>.png`. Exported request filenames and historical directories with sample manifests are also supported:

```sh
city-perception import-masks --dataset data/smallcity_p384 \
  --results supplied_mask_results --output runs/masks
```

The optional Vertex executor requires explicit `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `CITYCODE_VERTEX_MODEL` and `GOOGLE_OAUTH_ACCESS_TOKEN` configuration:

```sh
city-perception run-masks --requests runs/mask_requests
```

`run-masks` contacts the service; preparation and import are offline. Tokens are not written to results. After execution, normalize responses with `import-masks`; `--results` may point to the request directory.

## Depth prediction and import

The adapter loads a MoGe v2 **full checkpoint**, containing `model_config` and model state. The selected P384 checkpoint is `models/depth/step_00004800.pt`, with inference source in `third_party/MoGe/`. Its file identity and 521 state-tensor keys/shapes were validated; see [the depth model guide](../../models/depth/README.md). Preparing these resources does not constitute a new GPU inference run.

The adapter neither searches the research workspace nor downloads weights. Install optional dependencies and obtain model files before moving to an offline GPU environment:

```sh
python -m pip install -e '.[depth]'
python -m pip install -e third_party/MoGe
city-perception model-info
city-perception predict-depth --dataset data/smallcity_p384 \
  --checkpoint models/depth/step_00004800.pt --model-source third_party/MoGe \
  --checkpoint-sha256 63cf3463631911ea3ce586d576926ec41230f93df547755e594994ded3bbc873 \
  --output runs/depth
```

The third-party package pins its `utils3d` dependency. Output is a source-resolution float32 array of camera-forward Zc depth in metres; invalid values and pixels outside the strict ROI are NaN. A target long edge of 1024 is rounded down to a multiple of 14: the supplied square view uses a 1022 x 1022 network input, then returns to 2560 x 2560. Camera intrinsics supply the known field of view.

This step does not perform background-class scale calibration. The compatibility key `depth_ground_scaled_zc_m` is an identity alias of the raw file, not a second calibrated array. Numerical recovery later applies predicted V7B semantics and explicit priors.

Existing predictions can be imported instead:

```sh
city-perception import-depth --dataset data/smallcity_p384 \
  --input supplied_depth --output runs/depth
```

Without a manifest, supply native-resolution arrays as `depth_raw_zc_m/<patch>__<view>__depth_raw_zc_m.npy`. With a manifest, the reader uses `samples[].outputs.depth_raw_zc_m` and preserves available calibration provenance. Normal recovery requires raw model depth and rejects explicitly calibrated inputs. Historical calibrated caches are reserved for explicit compatibility validation.

## Recover from predictions

```sh
city-perception recover --dataset data/smallcity_p384 \
  --depth runs/depth --masks data/smallcity_p384_predictions/masks \
  --output runs/structure --structure-only
city-perception verify-structure --run runs/structure
```

`--priors` selects another explicit plane configuration; otherwise dataset priors apply. The numerical sequence is validity filtering, scale/depth processing, surface/BEV reconstruction, footprint and height estimation, cross-patch fusion and structural cleanup. Its execution inputs are observations, cameras, predictions and plane priors.

Only the `standard` profile is public. It combines facade/fragment cleanup, supported-roof preservation, bounded footprint repair and robust height consensus, with candidate-selection area weight 0.5, conservative visible-background trimming and local predicted-surface protection. Trimming occurs after height consensus and does not change identity or height. Unknown, occluded or ambiguous foreground is not treated as removable background.

Mesh export retains positive-area components and holes. An alternative triangulation is used only when the original fails the same fidelity checks. GLB coordinates are float32, so validation uses coordinate/perimeter error bounds; six holes in the supplied example are below that representation's resolution. Details are in [VALIDATION.json](VALIDATION.json).

Count independent support patches from `consensus_observation_ids`. Selection fields `member_count` and `patch_count` describe canonical selection, not all supporting evidence. Missing `roof_layers` means no separately modeled elevated roof layer, not an absent building or roof.

Height consensus gives one vote per independent patch and uses Huber aggregation: quality at least 40, a 4 m residual transition, at least two patches, at most 3 m change per building and at most 1 m point-q95 compensation. Existing roof layers remain. Association IoU compares predicted observations with predicted buildings.

`--structure-config` overrides scene-wide numerical parameters only; its profile must be `standard`. Object lists and per-object selection files are not accepted.

## Replay the supplied example

```sh
city-perception recover-cached --cache data/smallcity_p384_recovery_cache \
  --output data/smallcity_p384_structure_reproduced
city-perception verify-structure --run data/smallcity_p384_structure_reproduced
```

The portable cache starts from predicted E10 observations and direct points, and supplies the predicted inputs, E09 surfaces and image evidence needed for refinement. It uses the same structural recipe as `recover`. Historical raw depth is unavailable, so this replay does not rerun neural inference, segmentation APIs or upstream E00-E10. Cached calibrated depth cannot be supplied to `recover` as raw depth.

By default recovery exports full per-building evidence, merging canonical/member/consensus, boundary-repair and roof-support observations while retaining their individual roles. Observed contours may differ from the final repaired structure; both representations remain separate. Each full camera includes its native-resolution validity mask, while cropped intrinsics are stored in `observation.json`.

`--structure-only` explicitly defers image crops and does not produce `building_generation_inputs/`. The supplied accepted example contains 79 patches, 562 structural sources and 1,345 evidence observations, with geometry, evidence, resume and relocation checks recorded in [VALIDATION.json](VALIDATION.json).

| Output | Purpose |
|---|---|
| `release/global_building_footprints.geojson` | Footprints, base/top/height, roof layers and fused identities |
| `release/full_city_layered_ue_z_up.glb` | Numerical structural mesh for geometric constraints |
| `building_generation_inputs/manifest.json` | Source RGB/RGBA/context, BEV roof crops, local LoD1, cameras and transforms |
| Raw run: `upstream/e09_full_pipeline/`; cache: `source/e09_full_pipeline/` | Predicted background semantics, surfaces, RGB and world transforms |
| Raw run: `upstream/e10_height_guided/`; cache: `source/global_fused/` and input manifest | Per-observation structures and direct predicted-point provenance |
| `RUN_STATUS.json`, `logs/` | Status, input/output identities and resume checks |

Use a new output directory. A failed run can resume with `--resume` only when code, configuration and inputs are unchanged. `verify-structure` checks this stage's geometry consistency; a research evaluation module is outside the public workflow.
