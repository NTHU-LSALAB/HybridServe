#!/bin/bash
# Launch the prefill node (kv_producer) as an OpenAI-compatible API server.
# Config comes from env.sh; override any value inline, e.g.:
#   KV_IP=10.121.187.102 NCCL_SOCKET_IFNAME=enp1s0f0 ./prefill.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env.sh

CUDA_VISIBLE_DEVICES="$PREFILL_CUDA_DEVICE" \
NCCL_P2P_DISABLE=1 \
NCCL_NET_GDR_LEVEL=0 \
NCCL_IB_DISABLE=1 \
NCCL_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME" \
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --port "$PREFILL_PORT" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --dtype half \
  --kv-transfer-config \
  "{\"kv_connector\":\"PyNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_rank\":0,\"kv_parallel_size\":2,\"kv_buffer_size\":$KV_BUFFER_SIZE,\"kv_ip\":\"$KV_IP\"}"
