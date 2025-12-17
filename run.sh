#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${DATA_PATH:-data/Tweets.csv}"
OUT_ONNX="${OUT_ONNX:-/artifacts/model.onnx}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-v1}"

if [[ ! -f "${DATA_PATH}" ]]; then
  echo "Dataset not found: ${DATA_PATH}"
  echo "Put Tweets.csv there or set DATA_PATH env var."
  exit 1
fi

GIT_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo unknown)"
export GIT_COMMIT

docker compose build

docker compose run --rm \
  -e GIT_COMMIT="${GIT_COMMIT}" \
  backend python train.py \
    --data "/data/$(basename "${DATA_PATH}")" \
    --out "${OUT_ONNX}" \
    --experiment-name "${EXPERIMENT_NAME}"

docker compose up -d --force-recreate
echo "OK: frontend=http://localhost:8080 backend=http://localhost:8000 prometheus=http://localhost:9090"
