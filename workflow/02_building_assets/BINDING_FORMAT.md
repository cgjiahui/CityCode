# Building-asset binding format

The binding schema is `citycode.building_asset_bindings.v1`, with `path_base=manifest_directory`. Owned asset paths must resolve inside that directory and carry full SHA256 proofs. Source evidence may reference the separately delivered structure package; validation does not follow arbitrary absolute provenance paths.

`source_structure` records `instance_count`, the source input manifest's `manifest_sha256`, and the full `building_ids` set. `buildings` retains one disposition per source ID.

| Source status | Meaning |
|---|---|
| `accepted_reuse` | An existing model passed adaptation and both-LOD acceptance |
| `accepted_generated` | A newly generated asset passed both-LOD acceptance |
| `accepted_composite` | A whole-building composite passed both-LOD acceptance |
| `accepted_repaired` | A detailed asset passed traceable local repair and actual both-LOD checks |
| `accepted_structure` | A source-structure representation passed its geometry/material contract; detailed generation is not claimed |
| `resolved_fragment` | An incomplete observation is recorded as a separate local surface, not a complete building |
| `excluded_incomplete`, `excluded_noise` | Asset qualification was excluded with reasons and observation evidence; source ID remains |
| `pending_review`, `pending_appearance`, `pending_generation`, `pending_registration`, `pending_lod_export`, `failed` | Incomplete; cannot count as successful delivery |

## LODs and transforms

An accepted complete-building record carries current `target_geometry`, `base_z_m`, `top_z_m`, and `lods` entries for `0.25M` and `0.1M`. Each LOD includes `path`, `sha256`, `bytes` and actual scene triangle count `faces`. A mesh instantiated through multiple nodes counts once per node toward the face budget. Both files share asset-local coordinates and placement.

A structure or local-surface representation may use identical bytes for both LODs if `postprocess.lods_identical=true` and each meets its face limit. This exception does not replace two-LOD delivery for detailed assets.

New generation may additionally provide `1M` as `model_1m.glb`, with a 1,000,000-triangle budget. It is independently exported from the same generated mesh and shares the placement matrix. Actual triangles, materials, normals, coverage, height and coordinate consistency are checked when present. LODs are alternatives for one asset, not additional placements. The supplied example remains complete at `0.25M` and `0.1M`; new-batch initialization requests three LODs by default.

Matrices use column vectors:

```text
world_from_local_4x4                   = M: glTF Y-up asset -> UE Z-up metric world
building_local_from_asset_4x4          = L: asset -> building-local frame
world_from_building_local_4x4          = T: building-local frame -> world
T * L = M
normal_transform_world_3x3 = inverse(M[0:3, 0:3]).transpose()
local_coordinate_system = "glTF Y-up"
```

Original GLB node transforms remain part of the model; do not discard or apply them twice. Normalize normals after the complete transform. Ground/base and top-height tolerance is `min(0.20m, max(0.05m, 0.01 * height))`. Relative drift between LOD-local bounds must not exceed 2.5%; that bounds tolerance does not relax ground-contact checks.

## Physical instances and building groups

When multiple source IDs describe one confirmed physical asset, their source records carry `asset_instance_group_id` and individual `source_target_geometry`. The shared `target_geometry` is the original-position union of member source footprints. Top-level `asset_instance_groups` records group `id`, complete `building_ids`, LODs and the shared `world_from_local_4x4`. Member models and transforms must match the group. Stage 04 instantiates the physical group once and retains all source associations.

A composite needs whole-building GLBs for both supplied LODs. Component inventories may be retained for provenance, but cannot replace those whole-building exports. Records sharing identical LOD content and the same placement must have an explicit shared group, preventing duplicate assembly through copied files or path aliases. Reusing a model at genuinely different placements can create distinct entities.

`asset_instances` is the physical base-model list consumed by Stage 04. It includes accepted detailed assets and explicitly marked structure representations. Each entry has `id`, complete `source_building_ids`, status, `lods` and `world_from_local_4x4`; source records refer to it with `asset_instance_id`.

The list must exactly cover accepted building sources. A multi-source physical instance must belong to an explicit shared group. Every member's `asset_instance_id` must equal the group ID, and each group maps to one physical entity with matching files and transforms. Historical bindings without this list can still be checked, but current delivery uses it to prevent duplicate assembly.

`building_entities` records the logical complete-building grouping over physical base-model parts. It is distinct from both source count and placement count.

## Optional observation surfaces

`surface_instances` separately records `resolved_fragment` surfaces, excluded from the building-asset count. Each has a unique `id`, its fragment's `source_building_ids`, status, LODs, `world_from_local_4x4`, nullable `parent_building_id`, and Boolean `assembly_enabled`. A source points to it through `surface_instance_id`, rather than `asset_instance_id`.

Each fragment appears once. Surface and source records must agree on model, placement, parent and visibility. Stage 04 only enables surfaces marked `assembly_enabled=true`; other surfaces are optional evidence. A parent relationship does not add the fragment to the parent's `source_building_ids`, increase its instance count or prove that the parent mesh covers the fragment. Unsupported parent relationships remain `null`.

In the current completed example, all 16 observation surfaces remain present but have `assembly_enabled=false`, avoiding default overlap with the inferred massing.

## Generated placement and orientation provenance

A generated model's initial pose inverts the recorded relationship between the current predicted footprint and model plane, explicitly converting glTF Y-up. Initial height scale comes from executed voxel-height conditioning. Actual meshes then undergo the same bounded registration checks as reused assets. Do not copy an old model matrix because its numeric ID matches, or derive an initial scale from the final mesh so that acceptance is guaranteed by construction.

Height correction and the two structure-coordinate files are hash-bound through the generation manifest's `placement_inputs`.

When reviewed source-region correspondences change the initial heading, `provenance.placement_initialization` records the old/new initial poses, rigid yaw adjustment, evidence hashes and mechanism-source hash. The local yaw bound is relative to the new initialization; it must not be claimed as the same bound around the old pose. Final placement still comes from bounded fitting and both-LOD acceptance.

## Postprocessing records

`postprocess` uses `citycode.asset_postprocess_record.v1`.

| Representation | Required provenance and qualification |
|---|---|
| `method=source_structure`, `representation=structure` | Current source and layered-mesh hashes; world-to-local relocation; unchanged source vertices/faces and roof layers; observed roof imagery and neutral unknown facades; `detailed_asset_accepted=false`, `facade_detail_recovered=false` |
| `method=local_mesh_repair`, `representation=repaired_detailed` | Original LODs, repair recipe, adopted source triangles and repaired geometry/material checks; old acceptance cannot be relabeled as approval for a new mesh |
| `method=observed_fragment_surface`, `representation=observed_surface` | Supporting-point NPZ, source and mesh-neighborhood metadata, original indices, actual coverage, parent and visibility; `complete_building=false`, `detailed_asset_accepted=false`; 0.12m visualization-only thickness |
| `method=retain_fragment_evidence`, `representation=evidence_only` | Source, actual observations, reviewer and per-item exclusion reasons; no building or surface assembly entity |

Preserve original structure identity in `source_target_geometry`, `source_base_z_m` and `source_top_z_m`. Structure representations retain source bounds and height. A local surface's current `target_geometry` is the projected union of supported surfaces; its `base_z_m`/`top_z_m` include actual surface extent and visualization thickness. Original geometry remains in `postprocess.source_target_geometry` and Stage 01. `visible_source_coverage` describes direct support inside that source footprint, not complete-building coverage or detailed-reconstruction success.

`postprocessing` summarizes the input-binding SHA, decision file, action counts, protected previous assets/sources, accepted representation counts, repairs and default/optional surfaces. `fine_generation_complete` is reported separately. It was `false` when historical version 0.2.8 contained 32 structure representations; 0.2.9 refined them and set it to `true`, while retaining historical actions under `records_scope`. Those transitions preserved their existing asset/source identities and all 562 Stage 01 records.

`city-assets postprocess-assets` accepts an unprocessed binding and decisions bound to its full SHA256, then writes a new sibling result. Existing `postprocessed_assets/` or output files are rejected. For replay, use a data copy and retained `postprocess_input_manifest.json`; see [review and postprocessing](README.md#review-and-postprocessing).

`city-assets validate-assets` can inspect drafts. Formal delivery uses `--require-complete --geometry`: every source has a disposition and each actual LOD passes the geometry, material, normal and identity rules for its representation. Structure checks verify relocation of original layered meshes; surface checks verify direct support, complete mesh neighborhoods and retained triangles; repairs check the current recipe and output. Complete source accounting does not mean every source became a detailed building or that UE assembly has already run.

## Fine-refinement history

`fine_refinement` records frozen inputs, the three-stage recipe, actual output hashes, CPU roof constraints and both-LOD acceptance. New assets retain `accepted_generated`. Geometry altered by roof-layer or boundary reconciliation must say so in per-item provenance, rather than claiming an unchanged neural mesh or a purely rigid fit.

Historical refinement preserved its 512 detailed assets and 17 fragment dispositions, with 544 physical assets at that checkpoint. These are historical checkpoint counts, not current completed-scene totals. Original structure models remain historical inputs and are not active base instances in the completed result.

## Inferred massing parts

`massing_instances` separately records podiums, wings or edge volumes instantiated from existing detailed models. Each has a unique `id`, `source_building_ids`, `donor_asset_id`, donor-LOD proofs, `world_from_local_4x4`, inverse-transpose normal transform, `target_geometry`, `target_base_z_m`, `target_top_z_m`, and the qualifications `independent_building=false`, `appearance_and_body_inferred=true`, `geometry_bytes_reused=true`.

Donor bytes match the base asset, but the new placement has its own transform; do not reuse the donor's original placement matrix. Top-level `massing_completion` pins the reviewed recipe, original binding, measured base projections, LOD reports and statistics. Source-level `massing_completion.part_ids` names the associated parts without changing its original status or increasing `building_entities` complete-building count. Previously accepted visual-completion assets keep evidence for the updated scene.

Stage 04 instantiates `asset_instances` and `massing_instances` exactly once each. Their sum is model placements, not independent buildings. The current result has **542 + 18 = 560 placements, 540 logical complete-building groups and 562 sources**. The preferred placement-cleaned binding is in `data/smallcity_p384_building_assets_completed/`; the original base cache remains available for replay and comparison.

Massing uses its own explicitly qualified acceptance thresholds; they cannot approve a complete-building asset or turn a reused part into a new neural-generation result. [Local massing review](MASSING_COMPLETION.md) explains the thresholds and per-part decisions. `pruning` records candidate dispositions, and `source_decisions[].removed_part_ids` preserves removed associations. Removing an addition does not automatically re-enable the original observation surface. Retained donor meshes and transforms remain unchanged; removed candidates do not enter `massing_instances`.
