# Model resources

Model weights are separate from source code and scene GLBs. Viewing or replaying packaged scene assets does not require loading neural weights. Download profiles and resource locations are described in the [release guide](../release/README.md).

| Resource | Entry point |
|---|---|
| Three fine-tuned building-generation networks and architecture configs | [Building generation](building_generation/standard/README.md) |
| Base decoders, official 512 shape model, image features, background removal, and TRELLIS.2 source identities | [Base-resource manifest](building_generation/base/MANIFEST.json) |
| Full MoGe v2 P384 depth checkpoint | [Depth model](depth/README.md) |
| Fixed offline generation dependencies | [Environment guide](../environment/README.md) |

The selected files are materialized in the local release candidate with recorded SHA256 identities. They are not part of the Python wheel or ordinary source checkout. Source licenses do not automatically grant weight-distribution rights; resource licensing and external publication remain separate decisions.

For cached and new-generation workflows, use the [building guide](../workflow/02_building_assets/README.md). Packaged resources and CPU checks do not certify full GPU inference on a new device.
