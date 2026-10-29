# Third-Party Code and Resources

Third-party notices apply to their respective components. They do not assign a license to the project's own code, observations, model weights or service-generated content. Those distribution terms remain to be finalized for this local candidate.

| Component | Upstream | Included attribution and scope |
|---|---|---|
| MoGe v2 inference code | [microsoft/MoGe](https://github.com/microsoft/MoGe) | Extracted from the locally used AerialMetric-associated version; original MIT license and Microsoft copyright retained. The source manifest is `third_party/MoGe/SOURCE_MANIFEST.json`. Training data, weights and training/evaluation entry points are separate. |
| DINOv2 code within MoGe | [facebookresearch/dinov2](https://github.com/facebookresearch/dinov2) | Meta copyright and Apache 2.0 headers retained; license text is `third_party/MoGe/moge/model/dinov2/LICENSE`. |
| MoGe-path utils3d dependency | [EasternJournalist/utils3d](https://github.com/EasternJournalist/utils3d) | External dependency pinned to commit `3fab839f0be9931dac7c8488eb0e1600c236e183`; its source is not copied here. |
| TRELLIS.2 and o-voxel runtime | `third_party/TRELLIS2/` | Original Microsoft MIT license and nested dependency copyright files retained. File identities are recorded in `models/building_generation/base/MANIFEST.json`. The upstream training entry point is not a project training-reproduction commitment. |
| Optional Pixal3D inference code | `third_party/Pixal3D/` | Original MIT license and NOTICE retained. Current source changes are listed in `pipelines/background_pixal3d/SOURCE_MANIFEST.json`. |
| Optional NAF inference code | `third_party/NAF/` | Original Apache 2.0 license retained; source identities share the Pixal3D source manifest. |

The selected P384 full depth checkpoint is different from the upstream Aerial LoRA. Use `city-perception model-info` to inspect its required identity.

The offline wheelhouse contains the distributions used by the packaged inference environment, with their bundled licenses. Versions and SHA256 values are listed in `environment/OFFLINE_WHEELS.json`. TRELLIS uses a separately pinned `utils3d==0.0.2` distribution; this is distinct from the MoGe-path dependency above.

DINOv3, BiRefNet, base networks and fine-tuned weights retain their own distribution terms; source-code licenses do not replace weight licenses. Optional background-model weights are not bundled with Pixal3D/NAF source; see `pipelines/background_pixal3d/MODEL_RESOURCES.json`.

Browser dependencies and notices are also listed in [web_preview/THIRD_PARTY_NOTICES.txt](web_preview/THIRD_PARTY_NOTICES.txt). Upstream English README, changelog and license files are retained as third-party documentation.
