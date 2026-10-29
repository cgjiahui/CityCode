# Data resources

Source code, scene assets, model weights, and offline dependencies are distributed separately. Use the [release guide](../release/README.md) for downloads and the [catalog](../release/catalog.json) for exact package sizes, dependencies, and checksums. The research workspace is not a download dependency.

## Download and install

Data and weights are stored in the [CityCode HF resource repository](https://huggingface.co/datasets/cgjiahui/CityCode). From the source checkout, sign in with `hf auth login` while access is private, then run `python release/bundles.py install --profile scene`. The installer uses `release/HOSTING.json`, checks the completed upload and SHA256 values, and restores the directories below automatically. Full setup and other profiles are in the [download guide](../release/README.md).

## Inputs

| Purpose | Installed directory | Contents |
|---|---|---|
| Original observations | `smallcity_p384/` | 79 tilt-light RGB images, cameras, valid-ROI masks, patch index, and explicit plane priors |
| Segmentation predictions | `smallcity_p384_predictions/` | Building-instance and background-semantic predictions |
| Structure replay | `smallcity_p384_recovery_cache/` | Predicted point clouds, calibrated depth, numerical surfaces, and observation evidence |
| Building generation | `smallcity_p384_structure/building_generation_inputs/` | Footprints, heights, roof layers, per-building images, and cameras for 562 structure sources |
| Optional building regeneration | `smallcity_p384_reused_generation_inputs/` | Images and an index of 550 generation units: 411 reused assets, 123 previously generated assets, and 16 components |
| Background replay | `smallcity_p384_background_history/` | Observed BEVs, 60 saved image-completion responses, final RGB, numerical heights, and a historical 3D background |
| Optional background generation | `smallcity_p384_background_pixal3d_inputs/` | Compressed numerical-input differences and patch-coordinate index |

Raw observations, prediction caches, and completed assets have different roles. Calibrated cached depth is not raw neural-model output. Replaying existing assets does not validate newly generated models.

## Completed results

| Result | Directory or entry point |
|---|---|
| Building structures | `smallcity_p384_structure/RESULT.json` and its `release/` directory |
| Building models and physical placement | `smallcity_p384_building_assets_completed/binding_manifest_physical.json` and its referenced GLBs |
| Placement before physical postprocessing | `binding_manifest.json` in the same building-results directory |
| Ground and water | `smallcity_p384_background_coastal/`; numerical heights are in `smallcity_p384_background_history/height/completed/` |
| Road references and scene-layout inputs | `smallcity_p384_gt_roads/` and `smallcity_p384_scene_enrichment/`; historical directory names and source attribution are retained |
| Browser derivatives | `smallcity_p384_building_assets_completed_web_preview/` and `smallcity_p384_background_web_preview/` |
| Replay evidence and caches | `smallcity_p384_building_assets/`, process files outside the core result selection, and `smallcity_p384_massing_completion_inputs/` |

There are **562 structure sources, 540 logical complete-building entities, and 560 model placements**: 542 base placements plus 18 local massing components. The 16 observed-surface references are optional and disabled by default. Packaged assets provide 0.1M and 0.25M LODs. A complete 1M result requires separate generation, export, and review.

## Download profiles and relative paths

The installer verifies SHA256 hashes and restores the original relative layout. Shared large objects are deduplicated for transfer; a small set of component-prior meshes remains independent so input-only downloads do not require the full building collection. Do not remove apparently duplicated cache paths or relocate a directory without updating all of its references.

| Profile | Use |
|---|---|
| `scene` | View completed buildings and ground, or prepare UE assembly, without full historical caches or network weights |
| `replay` | Add observations, predictions, and frozen process inputs to replay existing results |
| `regeneration-inputs` | Check all 550 optional building-generation inputs; GPU execution additionally needs weights and a compatible environment |
| `generate` | Building-generation inputs, network weights, and offline dependencies; new outputs need review |
| `depth` | Original observations and depth weights for new depth prediction |

The local `smallcity_p384_citycode/` snapshot references local UE build derivatives and is excluded from default downloads. Phase 04 converts downloaded or newly generated models locally, then compiles a CityCode scene for those actual outputs. A reproducible conversion copy is not a missing building asset.

The legacy `smallcity_p384_web_preview/` has been superseded by the completed-results preview and is not downloaded again. Original process records remain available through the appropriate packages. The 60 saved background responses, coordinate transforms, decisions, and source attribution remain part of the replay data.

Runtime inputs come from the selected packages and the user's City Sample project resources. Original annotation files do not need to be obtained separately. Historical provenance remains recorded in each resource manifest.
