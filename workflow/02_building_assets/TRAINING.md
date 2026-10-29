# Building generation: selected networks and training provenance

The three selected non-EMA checkpoints are present in the [standard model bundle](../../models/building_generation/standard/README.md), with full SHA256 identities matched to the historical inference weights. Public inference adapters are implemented. The training entry point has not been ported into this release; this document describes the verified training setup, not a new training run. Execution and validation boundaries are in the [building workflow](README.md).

## Model roles

Coarse structure, detailed geometry and PBR material were fine-tuned independently from their corresponding official initializations. They were not trained jointly end to end. The external appearance-completion service is a preceding image-input step, not an appearance model trained here.

The historical `full_2.8_B0` combination uses A2 structure step 98440, the existing multiview shape checkpoint step 41730, and material step 44940. Version 2.8 added inference-time predicted-height correction while reusing those networks.

## Data and shared settings

The training collection contains 5,402 assets: 5,132 training and 270 validation, with no independent test split. It includes consistently oriented buildings with base plates removed, RGB renders, cameras and structure/shape/PBR latents. Earlier counts of 5,453 and earlier split ratios do not describe this selected run.

Training samples 2–4 views of the same asset; validation uses four fixed views. Concatenated DINOv3 features provide image conditioning. Inference uses one completed building image. These multiview training images are a different interface from the four appearance-completion reference images.

Shared optimization uses AdamW, learning rate `1e-5`, weight decay `0.01`, betas `(0.9, 0.95)`, bf16 mixed precision and EMA `0.9999`. The configured 300 epochs are an upper bound, not evidence that every selected run reached 300. Public inference uses ordinary denoiser checkpoints; do not silently substitute EMA files or a different branch at the same step.

| Stage | Training input and target | Objective | Selected checkpoint |
|---|---|---|---|
| Structure | 512px RGB features → sparse structure latent of a building without its base plate | Flow MSE plus silhouette and height/area-ratio terms | A2 epoch 115 / step 98440 |
| Shape | 1024px RGB features and sparse coordinates from training assets → 32-channel shape latent | Shape-latent flow MSE | Epoch 65 / step 41730 |
| Material | 1024px RGB features and training shape latent → 32-channel PBR latent | PBR-latent flow MSE | Epoch 140 / step 44940 |

The effective structure loss is:

```text
L = L_flow + alpha * (
      0.05 * L_soft_IoU
    + 0.02 * L_balanced_BCE
    + 0.02 * L_log_height_area_ratio
)
alpha = min(1, (step + 1) / 4280)
ratio = height / sqrt(footprint_area)
```

The ratio term compares predicted and target log ratios using SmoothL1. Predicted height uses the 1%–99% vertical range of soft occupancy mass; area comes from its soft vertical projection, not bounding-box area. Structure-decoder parameters are frozen, but loss gradients pass through the decoder to the structure network. Old Dice/outside configuration fields are not active loss terms for A2.

Shape and material training do not add mesh Chamfer, rendered-RGB or LPIPS losses. Shape training uses sparse support from the training assets; material training uses training shape latents. Predicted upstream support/shape is introduced during inference. Parent-checkpoint fields record lineage, not an upstream network running at each training step.

## Frozen components and inference

The image feature extractor and structure/shape/material encoders and decoders were not fine-tuned in this standard setup. Encoders precompute training targets; decoders reconstruct occupancy, meshes and materials. The structure decoder also carries the differentiable geometric supervision.

Historical 2.8 inference adjusts the generated structure's Z coordinates using predicted height relative to predicted footprint area. It fills new empty Z layers introduced by stretching while preserving XY support. Later source-refinement configurations explicitly use `source_footprint` and recorded roof layers; this is a conditioning choice, not a separately trained network.

Shape inference follows a 512→1024 cascade, then produces 1024 material. Only the 1024 shape branch was fine-tuned here; the 512 branch uses official weights. The public loader directly loads the three selected fine-tuned networks and the components actually used by this computation.

## Resource scope

| Resource | Standard inference | Training or exact resume |
|---|---|---|
| Selected structure, shape and material denoisers | Required; identities in [STANDARD_PLAN.json](STANDARD_PLAN.json) | Used when reproducing the selected output or resuming the corresponding setup |
| Three decoders, official shape512, DINOv3 and background-removal model | Required | Used according to the training/data-preparation stage |
| Official structure, shape1024 and texture1024 initial weights | Not needed by the public loader, which directly loads fine-tuned weights | Required when retraining from official initialization |
| Latent encoders and data preparation | Not part of final asset inference | Required when rebuilding latent training targets |
| EMA, optimizer/misc state and other epochs | Excluded from default inference | Retained separately for exact resume or research comparisons |

The historical research loader first loaded official models and then replaced them with fine-tuned weights, including some networks unused by the selected cascade. The public adapter avoids those redundant initialization loads. They may still be useful for training or historical reproduction and are recorded separately.

Selected weights, base components and offline dependencies are locally packaged; distribution permissions and external hosting remain separate. Historical checkpoints and optimizer state are not added wholesale to the source archive.

## Validation scope

This description was checked against historical configurations, training implementations, frozen weight identities and completed inference records. It does not certify a new training run or a new full-batch GPU inference. Current release status is recorded in [release/STATUS.json](../../release/STATUS.json); asset acceptance is recorded by the corresponding binding.

Historical package 0.2.9 reused the same weights. Its 32-building refinement explicitly conditioned inference on source footprints and recorded roof layers, followed by per-item CPU roof reconciliation. That geometric postprocessing is not model training.
