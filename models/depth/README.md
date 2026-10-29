# P384 depth model

`step_00004800.pt` is the selected full fine-tuned MoGe v2 checkpoint: P384 warmstart full, step 4800, approximately 1.23 GiB. It is an independent copy whose SHA256 matches the phase-01 resource record. Local materialization does not imply external publication.

| Entry | Role |
|---|---|
| [MANIFEST.json](MANIFEST.json) | Checkpoint identity, provenance, size, and hash |
| [model_config.json](model_config.json) | Network configuration extracted from the checkpoint |
| [VALIDATION.json](VALIDATION.json) | 521 state-tensor keys and shapes match the packaged MoGe source; checked offline without GPU inference |
| [MoGe source](../../third_party/MoGe) | v2 inference code and license |
| [Inference adapter](../../src/citycode_perception/depth.py) | RGB, intrinsics, and valid ROI to native-resolution metric camera-forward Zc |

In a GPU environment with PyTorch, MoGe, and its dependencies, run from the release root:

```sh
city-perception predict-depth --dataset data/smallcity_p384 \
  --checkpoint models/depth/step_00004800.pt --model-source third_party/MoGe \
  --checkpoint-sha256 63cf3463631911ea3ce586d576926ec41230f93df547755e594994ded3bbc873 \
  --output runs/depth
```

The input is original camera imagery, not a completed BEV map. The output still needs scale handling, world-coordinate projection, semantic layering, cross-patch fusion, and hole treatment before it becomes a city ground-height field. Phase 03 normally uses the packaged historical two-layer heights and does not invoke this model.

Existing calibrated depth and E09/E10 cache evidence retain their actual provenance and must not be relabeled as new raw predictions. See the [phase-01 guide](../../workflow/01_perception_structure/README.md) for the input and execution contract.
