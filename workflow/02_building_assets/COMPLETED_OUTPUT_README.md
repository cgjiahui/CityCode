# Completed phase-02 building result

The standard command restores accepted base assets and replays frozen Agent-reviewed massing decisions:

```sh
city-buildings --config workflow/02_building_assets/unified.standard.json
```

Cached-result replay makes no new images, GPU jobs, or neural meshes. It preserves 562 structure sources and 540 logical complete-building entities. The result contains **542 base placements plus 18 reused massing components for 10 local sources: 560 model placements in total**. The components approximate podiums, wings, or strips; they are not 18 additional confirmed buildings. All 16 observed-surface references are optional and disabled by default.

`binding_manifest.json` records the completed generation/association result. After scene physical postprocessing, the viewer and phase 04 prefer `binding_manifest_physical.json`. Consumers instantiate each `asset_instances` and `massing_instances` entry once; LOD variants and optional observed surfaces are not extra buildings. Asset paths are relative to this directory. Keep the sibling structure package and its evidence in the published relative layout.

| Record | Purpose |
|---|---|
| `massing_completion/RESULT.json` | Final completion counts |
| `massing_completion/selection_pruning.json` | Frozen component-selection and pruning decisions |
| `massing_completion/final_validation.json` | Validation across all structure sources |
| `massing_completion/geometry_validation.json` | Actual geometry checks for both LODs of the 18 components |
| `massing_completion/EXECUTION.json` | Frozen execution and input identities, including preserved base assets |
| `massing_completion/review/` | Comparisons using the same camera |
| `massing_completion/coverage_summary.csv` | XY coverage of the 10 selected local observation footprints, not full-city measurement accuracy |

The viewer's massing toggle compares the 542 base placements with the 560-placement completed result. Appearance and unobserved volume remain approximate inferences. Follow the [building workflow](../../workflow/02_building_assets/README.md), [massing details](../../workflow/02_building_assets/MASSING_COMPLETION.md), and [phase-04 assembly guide](../../workflow/04_citycode_assembly/README.md).
