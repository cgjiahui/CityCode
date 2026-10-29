#!/usr/bin/env bash
# Task-scoped worker entry. Allocate separately, after images are ready.
set -euo pipefail
CITYCODE_PROJECT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
CITYCODE_PYTHON=${CITYCODE_PYTHON:-"${CITYCODE_PROJECT}/environment/.venv-generation/bin/python"}
CITYCODE_CONFIG=${1:-"${CITYCODE_PROJECT}/workflow/02_building_assets/unified.standard.json"}
CITYCODE_GPUS=${2:-4}
CITYCODE_DEADLINE=${3:-}
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 WANDB_MODE=offline
export PYTHONNOUSERSITE=1 PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export PYTHONPATH="${CITYCODE_PROJECT}/src"
export PATH="$(dirname "${CITYCODE_PYTHON}"):${PATH}"
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=4
export OPENCV_IO_ENABLE_OPENEXR=1
CITYCODE_OUT=$("${CITYCODE_PYTHON}" -c 'from citycode_assets.batch import load_config; import sys; print(load_config(sys.argv[1])["output"])' "${CITYCODE_CONFIG}")
export TORCH_EXTENSIONS_DIR="${CITYCODE_OUT}/execution/torch_extensions"
mkdir -p "${TORCH_EXTENSIONS_DIR}"
cd "${CITYCODE_PROJECT}"
CITYCODE_ARGS=(--config "${CITYCODE_CONFIG}" --phase gpu --gpu-count "${CITYCODE_GPUS}")
if [[ -n "${CITYCODE_DEADLINE}" ]]; then
  CITYCODE_ARGS+=(--deadline-seconds "${CITYCODE_DEADLINE}")
fi
exec "${CITYCODE_PYTHON}" -u -m citycode_assets.batch "${CITYCODE_ARGS[@]}"
