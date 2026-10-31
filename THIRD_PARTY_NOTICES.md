# Third-Party Code and Resources

The root [MIT License](LICENSE) covers project-owned CityCode source code and its accompanying software documentation. It does not replace the licenses of vendored third-party code, external dependencies, datasets, model weights, scene assets, or paper figures/videos. Original license texts, copyright headers and applicable notices remain with their respective components.

## Included inference and geometry code

| Component | Upstream | Included attribution and scope |
|---|---|---|
| MoGe v2 inference code | [microsoft/MoGe](https://github.com/microsoft/MoGe) | Extracted from the locally used AerialMetric-associated version; original MIT license and Microsoft copyright retained. The source manifest is `third_party/MoGe/SOURCE_MANIFEST.json`. Training data, weights and training/evaluation entry points are separate. |
| DINOv2 code within MoGe | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Meta copyright and Apache 2.0 headers retained; license text is `third_party/MoGe/moge/model/dinov2/LICENSE`. |
| MoGe-path utils3d dependency | [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d) | External dependency pinned to commit `3fab839f0be9931dac7c8488eb0e1600c236e183`; its source is not copied here. |
| TRELLIS.2 and o-voxel runtime | `third_party/TRELLIS2/` | Original Microsoft MIT license and nested dependency copyright files retained. File identities are recorded in `models/building_generation/base/MANIFEST.json`. The upstream training entry point is not a project training-reproduction commitment. |
| Optional Pixal3D inference code | `third_party/Pixal3D/` | Original MIT license and NOTICE retained. Current source changes are listed in `pipelines/background_pixal3d/SOURCE_MANIFEST.json`. |
| Optional NAF inference code | `third_party/NAF/` | Original Apache 2.0 license retained; source identities share the Pixal3D source manifest. |
| Eigen, within o-voxel | `third_party/TRELLIS2/o-voxel/third_party/eigen/` | Primarily MPL-2.0; see [LICENSE](third_party/TRELLIS2/o-voxel/third_party/eigen/LICENSE) and [COPYING.README](third_party/TRELLIS2/o-voxel/third_party/eigen/COPYING.README). Additional BSD, Apache, MINPACK and benchmark-specific GPL notices are retained in the tree. The benchmark [COPYING](third_party/TRELLIS2/o-voxel/third_party/eigen/bench/btl/COPYING) is GPL-2.0. These files are not relicensed under CityCode's MIT license. |
| meshoptimizer 0.25 | [zeux/meshoptimizer](https://github.com/zeux/meshoptimizer) | Geometry processing; Arseny Kapoulkine's MIT license is retained in [vendor/meshoptimizer_v025/LICENSE.md](src/citycode_assets/vendor/meshoptimizer_v025/LICENSE.md). |

Redistributing these components requires preserving their applicable license texts and notices. A README acknowledgement does not replace those requirements. When modifying Apache-2.0-covered files, retain upstream notices and identify the modifications as required by that license.

## External dependencies and non-code resources

The offline wheelhouse contains the distributions used by the packaged inference environment, with their bundled licenses. Versions and SHA256 values are listed in `environment/OFFLINE_WHEELS.json`. TRELLIS uses a separately pinned `utils3d==0.0.2` distribution; this is distinct from the MoGe-path dependency above.

DINOv3, BiRefNet, base networks and fine-tuned weights retain their own distribution terms; source-code licenses do not replace weight licenses. Optional background-model weights are not bundled with Pixal3D/NAF source; see `pipelines/background_pixal3d/MODEL_RESOURCES.json`.

City Sample content is copyright Epic Games and retains its applicable [Epic content terms](https://www.unrealengine.com/eula/content). UE assembly uses the user's compatible City Sample installation. Neither the CityCode MIT license nor this notice grants additional rights to Epic content.

## Browser software

The project homepage includes React, React DOM, Scheduler, Three.js, meshoptimizer's browser decoder, KTX-Parse, ZSTDDecoder/Zstandard, and the Basis Universal transcoder. Their licenses include MIT, Apache-2.0 and BSD 3-Clause. Full texts and component versions are included in [docs/THIRD_PARTY_NOTICES.txt](docs/THIRD_PARTY_NOTICES.txt) and [docs/licenses](docs/licenses).

The offline research viewer has its own [web_preview/THIRD_PARTY_NOTICES.txt](web_preview/THIRD_PARTY_NOTICES.txt). Build-time dependencies retain their upstream licenses and are pinned by each viewer's lockfile. Upstream English README, changelog and license files are retained as third-party documentation.
