# Resource Inventory — 2026-10-28

This inventory describes the separately distributed resources. Duplicate content is identified by full SHA256; file names alone do not determine whether content can be removed. Logical paths needed by replay remain available.

| Resource package | Logical size, GiB | Owned objects, GiB | Files |
|---|---:|---:|---:|
| `observations` | 0.766 | 0.766 | 320 |
| `structure` | 1.836 | 1.800 | 20696 |
| `buildings` | 7.805 | 7.761 | 1128 |
| `maps` | 3.605 | 3.605 | 20 |
| `scene-layout` | 0.538 | 0.538 | 68 |
| `preview` | 0.193 | 0.189 | 1132 |
| `prediction-cache` | 6.357 | 5.593 | 1197 |
| `building-regeneration` | 1.238 | 1.232 | 3708 |
| `building-replay` | 10.673 | 1.251 | 7714 |
| `background-replay` | 4.398 | 4.396 | 801 |
| `pixal3d-inputs` | 0.560 | 0.560 | 5 |
| `depth-weights` | 1.233 | 1.233 | 5 |
| `generation-weights` | 20.299 | 20.299 | 27 |
| `offline-environment` | 3.199 | 3.199 | 82 |

Owned-object size excludes dependency packages. Shared large objects normally have one owner. Eight component-prior meshes, about 43.25 MiB in total, are also available in the independent regeneration-input package to avoid requiring the complete building set.

| Profile | Dependency-complete object size, GiB |
|---|---:|
| `scene` | 13.893 |
| `replay` | 27.131 |
| `regeneration-inputs` | 3.032 |
| `generate` | 26.531 |
| `depth` | 1.999 |
| `all` | 52.422 |

The logical resource total is **62.700 GiB**. Distribution uses **52.422 GiB** of content objects and avoids **10.278 GiB** of repeated transfer.
The archive set contains **60 parts / 52.443 GiB**, including tar headers and alignment; each part is at most 1 GiB.

The download set excludes local UE conversion copies, the old machine-specific CityCode snapshot, the superseded lightweight preview, virtual environments, run caches and old Python wheels.

Building replay caches and completed results share models, images and review records. Keeping their logical paths preserves replay contracts; object-based distribution avoids sending identical large payloads repeatedly. Maps include distinct geometry, textures, numerical heights, water and outline resources with separate purposes.

Original RGB images and cameras are in `data/smallcity_p384/`. Building-generation inputs are in `data/smallcity_p384_structure/building_generation_inputs/`. Prediction-replay inputs are in `data/smallcity_p384_recovery_cache/`. See the [data directory guide](../data/README.md) for the remaining locations and roles.

The complete resource set is hosted in the private [CityCode HF dataset repository](https://huggingface.co/datasets/cgjiahui/CityCode). All 60 archive parts were uploaded and verified by size and SHA256. The source download configuration pins the verified dataset revision. Private access requires an authorized HF account.
