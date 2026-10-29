# Local massing completion and per-part review

This optional scene step follows acceptance of the 542 base models. A structural source may represent a podium roof, wing, terrace or edge strip rather than an independent building. Review each candidate against observations, actual meshes and its surrounding scene. A small bad patch does not invalidate a reasonable neighboring part, and empty mask regions do not have to be filled.

## Published example

The accepted result has **542 base placements + 18 local massing parts = 560 model placements**. The base models form 540 logical complete-building groups; all 562 Stage 01 source records remain. The 16 original observation surfaces are optional and disabled by default.

| Source | Original candidates | Retained | Recorded decision |
|---|---:|---:|---|
| 113 | 3 | 2 | Keep the main body and continuous low wing; remove the small connector |
| 190 | 3 | 2 | Remove the 4×7m patch |
| 254 | 1 | 1 | Retain |
| 313 | 2 | 2 | Retain complete square volumes, including one below 250m² |
| 323 | 3 | 2 | Remove the slender 8m-wide, approximately 97m-high patch |
| 363 | 4 | 3 | Keep three coherent bodies; remove the slender 7m-wide patch |
| 416 | 3 | 1 | Keep the approximately 6×18.5m, 19m-high low main volume; remove two small patches |
| 531 | 4 | 2 | Keep the main body and 14.5×14.5m block; remove 3m- and 4m-wide strips |
| 565 | 2 | 2 | Retain |
| 578 | 2 | 1 | Remove the 6×12m, approximately 81m-high patch |

Sources 232, 330, 404 and 573 have explicit whole-group cancellation decisions. Source 239's three candidates, approximately 3×40.5m, 5×6.5m and 2×10.5m, were individually removed. Source 394's approximately 3×19.5m strip and source 534's approximately 2.5×16.5m strip were also individually removed. Those groups have no remaining parts, rather than an implicit rule that one bad member cancels every member. Sources 499/514 remain record-only.

Of 45 original candidates, 18 were retained and 27 removed. Per-part review restored six reasonable bodies that an earlier uniform size filter had rejected. Base-model files and placements were unchanged; retained donor meshes and transforms were not newly stretched or repositioned during that pruning review.

## Representation and decisions

The example reuses existing textured assets without new image requests or neural mesh generation. Both donor LOD byte sequences remain unchanged. Independent world transforms can include anisotropic scaling, which may alter the apparent windows or floor proportions. This is approximate inferred scene massing, not an undeformed copy or a measured recovery of unseen architecture. Viewer highlight colors can be disabled to inspect the original texture.

Old width/area screening values, such as 8m, 120m² and 250m² for tall candidates, select cases for review; they are not universal deletion rules. Final `retain` decisions, reasons, reviewer identity and candidate proofs are stored per part in `pruning.policy.part_reviews`. Identity covers donor files, source, target, height and matrix. Changing a model or placement invalidates approval. Per-part retention cannot override an explicit whole-group cancellation.

In `src/citycode_assets/massing_completion.py`, `prune_recipe` creates unapproved candidates and clears old scene approvals. `render_review` renders actual meshes before and after the addition with the same camera. A reviewer must inspect those models and scenes before approving the current recipe. Passing a size screen does not prove that a building should exist.

## New-scene massing review

After generating models for a new scene, review the current observations, structures, completed images, actual meshes and placed neighbors. The following evidence should support an addition together:

1. A continuous roof, facade or other credible building surface supports approximate location and height. Roads, shadows, scattered edges and empty masks alone are insufficient.
2. Determine whether it belongs to an independent building, connected group or part of an existing main building, and check the corresponding height layer. A small roof fragment does not establish a new tower extending from ground level.
3. Check actual triangles, placement and height to establish that the volume is missing. If already represented, record the association rather than adding another model. XY projection or a bounding box is insufficient on its own.
4. Inspect the selected mesh and local before/after scene for ground contact, main-body conflicts, road occupation, proportion and stretching. Leave uncertain regions empty.

Prefer a supported completed image and a generated whole building or suitable component. Reusing a donor is another explicit, reviewed choice. Select suitable shape and style; do not compress a model into a thin wall, stretch it into a needle or fill gaps with many small buildings solely to match a mask.

Review **each added part** separately. A coherent main body may be retained below an example size threshold; an implausible patch may be removed despite good coverage. Save reasons, and cancel a whole group only with explicit approval. Uncertain sources retain their records and pending reasons; original observation surfaces remain disabled by default.

| API in `massing_completion.py` | Role |
|---|---|
| `construct_manifest` | Build explicit candidates |
| `measure_manifest` | Measure actual LOD projections, heights and intersections |
| `render_review` | Render actual models and surrounding before/after scenes |
| `prune_recipe` | Save per-part disposition and invalidate old approvals |
| `apply_recipe` | Replay the reviewed recipe |
| `validate_massing` | Check identities and current results |

`apply_recipe` needs an explicit base binding, cache inventory and support files. A new-generation `finalize` does not automatically run a generic massing agent. Prepare a new recipe, evidence and approval for the actual scene.

Freeze at least the source-file and base-binding identities, all requested donor-LOD SHAs, target region, base/top heights, transforms, per-part decisions/reasons/reviewers, geometry reports and before/after views. Changed meshes, placements, groups or relevant context require new review. Example approvals for 10 sources and 18 parts cannot be transferred. If new outputs request `1M`, that LOD needs separate acceptance; the supplied two-LOD checks do not approve it automatically. Frozen decisions can be replayed, but new reviewer judgments are not claimed to be deterministic.

## Replay and measured limits

`massing.standard.json` stores 19 source dispositions, 18 retained parts, donor hashes, targets and transforms. `pruning.decisions` records the original 45 candidates and screening reasons; `removed_part_ids` preserves actual removals per source.

The [standard building command](README.md#cached-result-replay) restores the base assets and replays `massing_instances`. Mismatched source, donor or coordinate identities are rejected. It consumes supplied inputs on CPU without image requests or GPU inference.

Acceptance checks actual triangle XY projections in both supplied LODs, source support, height/contact and core intersections with base models and other additions. Target-rectangle coverage is at least 65%; at least 94% of the model projection lies within the target rectangle and source. Boundary overlap is at most 3%; after insetting both sides by 0.5m, intersection area must not exceed 0.25m². Complete-building thresholds remain unchanged. These screens are not closed-solid physics simulation or independent building-identity verification.

Coverage is diagnostic, not the final pruning objective. `coverage_summary.csv` describes model XY coverage for these 10 local sources, not citywide accuracy. Five assets previously accepted through visual scene-completion review retain evidence for the updated scene. Existing historical source attribution is preserved in the records.

## Data and inspection

| Resource | Location |
|---|---|
| Recipe | `workflow/02_building_assets/massing.standard.json` |
| Support evidence | `data/smallcity_p384_massing_completion_inputs/` |
| Final building binding | `data/smallcity_p384_building_assets_completed/binding_manifest.json` |
| Per-part decisions | Output `massing_completion/selection_pruning.json` |
| Geometry and execution identities | Output `massing_completion/geometry_validation.json`, `massing_completion/final_validation.json`, `massing_completion/EXECUTION.json` |
| Visual comparisons | Output `massing_completion/review/` |

In the local viewer, `http://127.0.0.1:8794/?inspect=massing#416` opens a local source. The list covers the 10 sources and 18 retained parts; compare before/after, original textures and the 542/560 placement counts.
