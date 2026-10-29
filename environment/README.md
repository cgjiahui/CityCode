# Offline runtime environments

## Building generation

The fixed wheelhouse targets Python 3.10, Linux x86_64, PyTorch 2.6.0, and CUDA 12.4 runtime libraries. `OFFLINE_WHEELS.json` records 80 local wheels and SHA256 values. It includes common dependencies, PyTorch/CUDA libraries, CuMesh, o_voxel, nvdiffrast, flash-attn, flex_gemm, and utils3d; `GPU_WHEELS.json` lists the six GPU extensions separately. Wheels were repackaged from installed distribution contents without rebuilding extensions, retaining bundled notices. `zstandard==0.25.0` is explicitly pinned for o_voxel.

System requirements include Python 3.10, a compatible NVIDIA driver, g++, an appropriate CUDA compilation toolchain, and Node.js for extrema-preserving LOD processing; Node 24 was used for the local check. CUDA runtime wheels do not supply the driver or complete compiler toolchain. Extension compatibility on a new GPU needs device validation.

Prepare the environment on a CPU development machine:

```sh
python3.10 environment/bootstrap_generation.py --check-only
python3.10 environment/bootstrap_generation.py --venv environment/.venv-generation
```

The installer verifies wheels, creates an isolated virtual environment without system site-packages, installs only local dependencies and this project, and runs dependency/import checks. It rejects an unrelated nonempty destination and does not modify an existing conda environment. Prepare weights, inputs, and review records before executing the [GPU task entry point](../workflow/02_building_assets/run_gpu.sh); workers do not download missing resources.

CPU installation and imports do not prove GPU model loading or inference. Inspect `CITYCODE_ENVIRONMENT.json`, each run's `execution/GPU_RESULT.json`, and actual outputs. On an already allocated GPU, check model/export imports before inference:

```sh
environment/.venv-generation/bin/python environment/check_generation_imports.py --require-cuda
```

This check does not allocate a worker, download resources, or load weights. FlexGEMM's delayed import requires a real CUDA driver, so the CPU check leaves that portion unverified.

Weights are not part of the wheelhouse. Selected weights are indexed by `models/building_generation/standard/MANIFEST.json`; base networks, image features, background removal, and TRELLIS.2 source by `models/building_generation/base/MANIFEST.json`. Resource licensing and public hosting are separate from local environment readiness.

## Background replay

Phase 03 uses CPU libraries without generation weights, image APIs, or a GPU runtime. Versions are fixed in `background.requirements.txt`. The low-memory PNG reader/writer needs a C compiler and libpng development files on Linux, for example Debian/Ubuntu packages `build-essential libpng-dev`. Its bundled code is compiled locally into the output directory without downloading components.

Full-city replay uses disk-backed image and mosaic arrays, then removes temporary files. Allow at least 2 GiB of available RAM and 30 GiB of additional working disk space. Do not use an in-memory filesystem such as `/tmp` for these large outputs. See the [background guide](../workflow/03_background_roads/README.md).

Standard replay pins OpenCV 5.0.0.93. `bootstrap_background.py` uses local wheels and `background.requirements.txt` to create a separate `.venv-background`, preserving the generation environment. `background_wheels/ORIGIN.json` records how the local OpenCV distribution was repackaged after checking original RECORD hashes; it does not claim a newly downloaded upstream wheel.

For an ordinary network-enabled source installation:

```sh
pip install -c environment/background.requirements.txt -e ".[background]"
```
