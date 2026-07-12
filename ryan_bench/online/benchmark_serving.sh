#!/bin/bash
# Drive load against the proxy (port 8000) using vLLM's benchmark_serving.py.
# Override any knob inline, e.g.:  QPS=5 NUM_PROMPTS=100 ./benchmark_serving.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

MODEL="${MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
RESULTS_DIR="${RESULTS_DIR:-./results}"
DATASET_NAME="${DATASET_NAME:-random}"
NUM_PROMPTS="${NUM_PROMPTS:-10}"
QPS="${QPS:-1}"
INPUT_LEN="${INPUT_LEN:-28000}"
OUTPUT_LEN="${OUTPUT_LEN:-32}"
PREFIX_LEN="${PREFIX_LEN:-0}"
PROXY_PORT="${PROXY_PORT:-8000}"
TAG="${TAG:-test}"

python3 ../../benchmarks/benchmark_serving.py \
        --backend vllm \
        --model "$MODEL" \
        --dataset-name "$DATASET_NAME" \
        --num-prompts "$NUM_PROMPTS" \
        --random-input-len "$INPUT_LEN" \
        --random-output-len "$OUTPUT_LEN" \
        --random-prefix-len "$PREFIX_LEN" \
        --port "$PROXY_PORT" \
        --result-dir "$RESULTS_DIR" \
        --result-filename "${TAG}-qps-${QPS}.json" \
        --request-rate "$QPS"
