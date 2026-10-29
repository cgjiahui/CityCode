# Standard building-generation checkpoints

These three fine-tuned networks are the normal, non-EMA checkpoints used by the historical full-2.8 inference recipe. Full SHA256 values match the recorded inference identities.

| Stage | Step | Checkpoint | Bytes |
|---|---:|---|---:|
| Coarse structure | 98440 | `structure/denoiser_step0098440.pt` | 5,171,023,441 |
| Fine geometry | 41730 | `shape/denoiser_step0041730.pt` | 5,169,220,934 |
| PBR material | 44940 | `material/denoiser_step0044940.pt` | 5,169,417,542 |

The weights total 15,509,661,917 bytes, approximately 15.51 GB. Each subdirectory also contains its architecture JSON. [MANIFEST.json](MANIFEST.json) records roles, configs, sizes, provenance, and hashes using paths relative to the release root.

From this directory, verify the three weights, architecture files, and manifest:

```sh
sha256sum --check CHECKSUMS.sha256
```

[CHECKSUMS.sha256](CHECKSUMS.sha256) uses paths relative to this directory and remains valid after relocating the entire directory. Stage 1 was restored from verified local inference weights; stages 2 and 3 were restored from an object-storage archive. Internal transfer details are not runtime inputs.

Base decoders, the official 512 shape model, DINOv3, and background removal are in the sibling `base/`; TRELLIS.2 source and offline dependencies are packaged separately. See the [building workflow](../../../workflow/02_building_assets/README.md) and [training description](../../../workflow/02_building_assets/TRAINING.md). Full new-batch GPU inference and resource-distribution licensing remain separate from local packaging checks.
